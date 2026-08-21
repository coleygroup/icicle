"""Model to infer fragmentation from SMILES.

This model is used to infer the fragmentation of a molecule from its SMILES. It
is used to generate the fragmentation DAG.
"""

import json
from typing import Any, Dict, List, Tuple, Union

import dgl
import numpy as np
import torch
import torch.nn as nn

from icicle.data.fragmentation_engine import MAX_BONDS, FragmentEngine
from icicle.data.tree_processing import TreeProcessor
from icicle.models.base_model import BaseFragmentGenerator
from icicle.utils import (
    NORM_VEC,
    formula_from_smi,
    formula_to_dense,
    pad_packed_tensor,
)

from .decoder import auto_regressive_decode
from .embedder import get_embedder
from .encoder import GNNEncoder
from .layers import MLPBlocks


class FragmentGenerator(BaseFragmentGenerator):
    def __init__(
        self,
        hidden_size: int,
        tree_processor: TreeProcessor,
        set_transformer_layers: int,
        layers: int,
        learning_rate: float,
        weight_decay: float,
        dropout: float,
        pool_op: str,
        max_broken: int,
        inject_early: bool,
        encode_formulae: bool,
        add_hs: bool,
        favor_positive_samples_factor: float,
        min_mz: float,
        max_mz: float,
        bin_width: float,
        **kwargs,
    ):
        super().__init__(
            min_mz=min_mz, max_mz=max_mz, bin_width=bin_width, **kwargs
        )

        # Store params
        self.hidden_size = hidden_size
        self.layers = layers
        self.set_transformer_layers = set_transformer_layers
        self.encode_formulae = encode_formulae
        self.add_hs = add_hs
        self.favor_positive_samples_factor = favor_positive_samples_factor
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.inject_early = inject_early

        self.save_hyperparameters()

        # Setup components
        self.tree_processor = tree_processor

        self._setup_model_components(pool_op, max_broken)
        self._init_loss_functions()

    def _setup_model_components(self, pool_op, max_broken):
        """Setup neural network components."""
        self._setup_formula_encoding()
        self._setup_gnn()
        self._setup_pooling(pool_op)
        self._setup_output_layers(max_broken)

    def _setup_formula_encoding(self):
        """Setup formula encoding components."""
        self.formula_in_dim = 0
        if self.encode_formulae:
            self.embedder = get_embedder("abs-sines")
            self.formula_dim = NORM_VEC.shape[0]
            self.formula_in_dim = self.formula_dim * self.embedder.num_dim * 2

    def _setup_gnn(self):
        """Setup GNN components."""
        node_feats = self.tree_processor.get_node_feats()
        edge_feats = MAX_BONDS

        # Account for additional dimensions when inject_early=True
        if self.inject_early:
            # Add hidden_size for root embeddings that get concatenated
            gnn_input_feats = node_feats + self.hidden_size
        else:
            gnn_input_feats = node_feats

        self.gnn = GNNEncoder(
            hidden_size=self.hidden_size,
            num_step_message_passing=self.layers,
            set_transformer_layers=self.set_transformer_layers,
            gnn_node_feats=gnn_input_feats,  # Use adjusted feature size
            gnn_edge_feats=edge_feats,
            dropout=self.dropout,
        )

        if self.inject_early:
            # Create separate GNN for root processing (original node features)
            self.root_module = GNNEncoder(
                hidden_size=self.hidden_size,
                num_step_message_passing=self.layers,
                set_transformer_layers=self.set_transformer_layers,
                gnn_node_feats=node_feats,  # Original node features only
                gnn_edge_feats=edge_feats,
                dropout=self.dropout,
            )
        else:
            # When not injecting early, can reuse the same GNN
            self.root_module = self.gnn

    def _setup_pooling(self, pool_op):
        """Setup pooling layers."""
        if pool_op == "avg":
            self.pool = dgl.nn.AvgPooling()
        elif pool_op == "attn":
            self.pool = dgl.nn.GlobalAttentionPooling(
                nn.Linear(self.hidden_size, 1)
            )
        else:
            raise NotImplementedError(
                f"Unsupported pooling operation: {pool_op}"
            )

    def _setup_output_layers(self, max_broken):
        """Setup output layers."""
        self.max_broken = max_broken + 1
        self.register_buffer("broken_onehot", torch.eye(self.max_broken))
        self.broken_clamp = max_broken

        self.output_map = MLPBlocks(
            input_size=self.hidden_size * 3
            + self.max_broken
            + self.formula_in_dim,
            hidden_size=self.hidden_size,
            output_size=1,
            dropout=self.dropout,
            num_layers=1,
            use_residuals=True,
            use_batchnorm=False,
        )

    def _init_loss_functions(self):
        """Initialize loss functions."""
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass."""
        graphs = batch["frag_graphs"]
        root_repr = batch["root_reprs"]
        ind_maps = batch["inds"]
        broken = batch["broken_bonds"]
        root_forms = batch.get("root_form_vecs")
        frag_forms = batch.get("frag_form_vecs")

        # Process root representation
        root_embeddings = self._process_root(root_repr)

        # Process fragments
        ext_root = root_embeddings[ind_maps]
        ext_root_atoms = torch.repeat_interleave(
            ext_root, graphs.batch_num_nodes(), dim=0
        )

        # Get fragment embeddings
        concat_list = self._get_feature_concat_list(graphs, ext_root_atoms)

        with graphs.local_scope():
            graphs.ndata["h"] = torch.cat(concat_list, -1).float()
            frag_embeddings = self.gnn(graphs)
            avg_frags = self.pool(graphs, frag_embeddings)

        # Prepare prediction inputs
        prediction_inputs = self._prepare_prediction_inputs(
            graphs,
            ext_root_atoms,
            frag_embeddings,
            avg_frags,
            broken,
            root_forms,
            frag_forms,
            ind_maps,
        )

        # Generate predictions
        output = self.output_map(prediction_inputs)
        padded_out = pad_packed_tensor(output, graphs.batch_num_nodes(), 0)
        return torch.squeeze(padded_out, -1)

    def predict_mol(
        self,
        smi: Union[str, List[str]],
        device: str = "cpu",
        max_nodes: int = 100,
        threshold: float = 0.01,
        **kwargs,
    ) -> Dict[str, Any]:
        """Base class interface."""
        decode_final_step = kwargs.get("decode_final_step", True)

        # Prepare inputs
        batch_data = self._prepare_batch_input(smi)

        # Get model device (Lightning manages this)
        model_device = next(self.parameters()).device

        # Initialize fragmentation engines and root data
        root_data = self._initialize_root_data(batch_data, model_device)

        # Setup root representation
        root_repr, root_graph_dict = self._setup_root_representation(
            root_data["root_frag"],
            root_data["engine"],
            root_data["root_smi"],
            model_device,
        )

        # Initialize tracking structures
        tracking_data = self._initialize_tracking_structures(
            root_data, batch_data["batch_size"]
        )

        # Run autoregressive generation loop
        result = self._run_autoregressive_loop(
            root_data,
            root_repr,
            root_graph_dict,
            tracking_data,
            batch_data,
            max_nodes,
            threshold,
            model_device,
            decode_final_step,
            batch_data["batch_size"],
        )

        return self._prepare_final_output(
            result, max_nodes, batch_data["batched_input"]
        )

    def _process_root(self, root_repr):
        """Process root - no device handling needed."""
        with root_repr.local_scope():
            root_embeddings = self.root_module(root_repr)
            return self.pool(root_repr, root_embeddings)

    def _get_feature_concat_list(self, graphs, ext_root_atoms):
        """Get features - no device handling needed."""
        concat_list = [graphs.ndata["h"]]
        if self.inject_early:
            concat_list.append(ext_root_atoms)
        return concat_list

    def _prepare_prediction_inputs(
        self,
        graphs,
        ext_root_atoms,
        frag_embeddings,
        avg_frags,
        broken,
        root_forms,
        frag_forms,
        ind_maps,
    ):
        """Prepare inputs - Lightning handles devices."""

        ext_frag_atoms = torch.repeat_interleave(
            avg_frags, graphs.batch_num_nodes(), dim=0
        )

        exp_num = graphs.batch_num_nodes()
        fragment_positions = torch.zeros_like(ind_maps)
        if len(ind_maps) > 0:
            fragment_positions[0] = 0
            for i in range(1, len(ind_maps)):
                if ind_maps[i] == ind_maps[i - 1]:
                    # Same batch item, increment position
                    fragment_positions[i] = fragment_positions[i - 1] + 1
                else:
                    # New batch item, reset position to 0
                    fragment_positions[i] = 0

        # Extract broken bond values for actual fragments (not padded ones)
        broken_flat = broken[ind_maps, fragment_positions]

        broken_flat = torch.clamp(broken_flat, max=self.broken_clamp)
        ext_frag_broken = torch.repeat_interleave(broken_flat, exp_num, dim=0)
        broken_onehots = self.broken_onehot[ext_frag_broken.long()]

        mlp_cat_vec = [
            ext_root_atoms,
            ext_root_atoms - ext_frag_atoms,
            frag_embeddings,
            broken_onehots,
        ]

        if self.encode_formulae and root_forms is not None:
            form_cat = self._prepare_formula_features(
                root_forms, frag_forms, exp_num, ind_maps
            )
            mlp_cat_vec.extend(form_cat)

        hidden = torch.cat(mlp_cat_vec, dim=1)
        return (hidden - hidden.mean(dim=0)) / (hidden.std(dim=0) + 1e-6)

    def _prepare_formula_features(
        self, root_forms, frag_forms, exp_num, ind_maps
    ):
        """Prepare formula features - no device handling."""
        if ind_maps is not None:
            root_forms = root_forms[ind_maps]

        diffs = root_forms - frag_forms
        form_encodings = self.embedder(frag_forms)
        diff_encodings = self.embedder(diffs)

        form_atom_exp = torch.repeat_interleave(form_encodings, exp_num, dim=0)
        diff_atom_exp = torch.repeat_interleave(diff_encodings, exp_num, dim=0)

        return [form_atom_exp, diff_atom_exp]

    def training_step(self, batch, batch_idx):
        """Training step."""
        pred_leaving = self(batch)

        loss = self.loss_fn(
            pred_leaving,
            batch["targ_atoms"],
            batch["frag_atoms"],
        )

        batch_size = len(batch["names"])
        self.log("train_loss", loss, batch_size=batch_size, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        pred_leaving = self(batch)

        loss = self.loss_fn(
            pred_leaving,
            batch["targ_atoms"],
            batch["frag_atoms"],
        )

        batch_size = len(batch["names"])
        self.log("val_loss", loss, batch_size=batch_size, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        """Lightning handles device automatically."""
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        """Simple optimizer - let Lightning handle the rest."""
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

    def loss_fn(self, outputs, targets, natoms):
        """Loss function - no device handling needed."""
        targets = targets.float().to(outputs.dtype)
        loss = self.bce_loss(outputs, targets)

        # Class weighting
        loss = loss * (
            (1) / (1 + self.favor_positive_samples_factor)
            + (self.favor_positive_samples_factor)
            / (1 + self.favor_positive_samples_factor)
            * targets
        )

        is_valid = (
            torch.arange(loss.shape[1], device=loss.device)[None, :]
            < natoms[:, None]
        )
        pooled_loss = torch.sum(loss * is_valid) / torch.sum(natoms)

        return pooled_loss

    def _prepare_batch_input(self, root_smi):
        """Prepare input data for batch processing."""
        if isinstance(root_smi, str):
            return {
                "batched_input": False,
                "root_smi": [root_smi],
                "batch_size": 1,
            }
        return {
            "batched_input": True,
            "root_smi": root_smi,
            "batch_size": len(root_smi),
        }

    def _initialize_root_data(
        self, batch_data: Dict[str, Any], device: torch.device
    ) -> Dict[str, Any]:
        """Initialize fragmentation engines and root molecule data."""
        engine = [FragmentEngine(rsmi) for rsmi in batch_data["root_smi"]]
        root_frag = [e.get_root_frag() for e in engine]
        root_form = [formula_from_smi(rsmi) for rsmi in batch_data["root_smi"]]

        # Process tree-like structures first
        root_hash = [e.wl_hash(rf) for e, rf in zip(engine, root_frag)]
        root_score = [
            e.score_fragment(rf)[1] for e, rf in zip(engine, root_frag)
        ]

        # Handle root form vectors
        dense_vectors = np.array(
            [formula_to_dense(str(rf)) for rf in root_form]
        )
        root_form_vec = torch.tensor(
            dense_vectors, dtype=torch.float32, device=device
        )

        return {
            "engine": engine,
            "root_frag": root_frag,
            "root_hash": root_hash,
            "root_score": root_score,
            "root_form": root_form,
            "root_form_vec": root_form_vec,
            "root_smi": batch_data["root_smi"],
        }

    def _setup_root_representation(self, root_frag, engine, root_smi, device):
        """Setup root molecule representation based on encoding type."""
        root_graph_dict = [
            self.tree_processor.featurize_frag(
                frag=rf, engine=e, add_random_walk=False
            )
            for rf, e in zip(root_frag, engine)
        ]

        root_repr = dgl.batch([rg["graph"] for rg in root_graph_dict])
        # Modern DGL with CUDA should handle device transfer automatically
        # but we can ensure it's on the right device
        root_repr = root_repr.to(device)
        self.tree_processor.add_pe_embed(root_repr)
        return root_repr, root_graph_dict

    def _initialize_tracking_structures(self, root_data, batch_size):
        """Initialize data structures for tracking fragments and scores."""
        form_to_min_score = [{} for _ in range(batch_size)]
        frag_hash_to_entry = [{} for _ in range(batch_size)]
        frag_to_hash = [{} for _ in range(batch_size)]

        # Initialize root entries
        root_entry = [
            {
                "frag": int(rf),
                "frag_hash": rh,
                "parents": [],
                "atoms_pulled": [],
                "left_pred": [],
                "max_broken": 0,
                "tree_depth": 0,
                "id": 0,
                "prob_gen": 1,
                "score": rs,
            }
            for rf, rh, rs in zip(
                root_data["root_frag"],
                root_data["root_hash"],
                root_data["root_score"],
            )
        ]

        # Update tracking structures with root data
        for e, re, rf, rh, f2ms, fh2e in zip(
            root_data["engine"],
            root_entry,
            root_data["root_frag"],
            root_data["root_hash"],
            form_to_min_score,
            frag_hash_to_entry,
        ):
            re.update(e.atom_pass_stats(rf, depth=0))
            f2ms[re["form"]] = re["score"]
            fh2e[rh] = re

        return {
            "form_to_min_score": form_to_min_score,
            "frag_hash_to_entry": frag_hash_to_entry,
            "frag_to_hash": frag_to_hash,
            "stack": [[rf] for rf in root_data["root_frag"]],
            "depth": 0,
            "id_": list(range(1, batch_size + 1)),
            "engine": root_data["engine"],
        }

    def _run_autoregressive_loop(
        self,
        root_data,
        root_repr,
        root_graph_dict,
        tracking_data,
        batch_data,
        max_nodes,
        threshold,
        device,
        decode_final_step,
        batch_size,
    ):
        """Run autoregressive fragmentation prediction loop."""
        with torch.no_grad():
            max_depth = root_data["engine"][0].params.max_tree_depth

            while tracking_data["depth"] < max_depth:
                # Process fragments and prepare batch
                batch_info = self._prepare_fragment_batch(
                    tracking_data["stack"],
                    root_data["engine"],
                    root_graph_dict,
                    device,
                )

                if len(batch_info["new_info_dicts"]) == 0:
                    break

                # Update tracking data with new batch info
                tracking_data["stack"] = batch_info["new_stack"]

                # Process fragment forms
                frag_form_data = self._process_fragment_forms(
                    batch_info, root_data["engine"], device
                )

                # Add new_frag_hashes to batch_info
                batch_info["new_frag_hashes"] = frag_form_data[
                    "new_frag_hashes"
                ]

                # Update fragment hashes
                self._update_fragment_hashes(
                    tracking_data["frag_to_hash"],
                    tracking_data["stack"],
                    batch_info["new_frag_hashes"],
                    batch_info["reverse_idx"],
                )

                # Get model predictions
                model_outputs = self._get_fragment_predictions(
                    batch_info,
                    frag_form_data,
                    tracking_data,
                    root_data,
                    root_repr,
                    device,
                )

                tracking_data["depth"] += 1

                # Process probabilities
                prob_data = self._process_probabilities(
                    tracking_data["frag_hash_to_entry"],
                    max_nodes,
                    threshold,
                    batch_size,
                )

                # Process fragment predictions
                sorted_data = self._process_fragment_predictions(
                    batch_info,
                    model_outputs,
                    tracking_data["frag_hash_to_entry"],
                    batch_size,
                )

                if (
                    tracking_data["depth"] == max_depth
                    and not decode_final_step
                ):
                    return self._prepare_batch_for_processing(
                        tracking_data,
                        prob_data["min_prob"],
                        sorted_data,
                        batch_size,
                        max_nodes,
                        threshold,
                    )

                # Update tracking data
                self._update_tracking_data(
                    tracking_data,
                    prob_data,
                    sorted_data,
                    batch_size,
                    max_nodes,
                    threshold,
                )

            # Filter final results
            return self._filter_final_results(
                tracking_data["frag_hash_to_entry"],
                tracking_data["form_to_min_score"],
            )

    def _prepare_fragment_batch(self, stack, engine, root_graph_dict, device):
        """Prepare batch of fragments for processing."""
        batch_info = {
            "new_info_dicts": [],
            "new_graphs": [],
            "new_stack": [],
            "reverse_idx": [],
            "batched_select": [],
            "batched_num_nodes": [],
            "batched_old_edge_idx": [0],
        }

        idx_offset = 0
        for rev_i, (st, e, rg) in enumerate(
            zip(stack, engine, root_graph_dict)
        ):
            for i in st:
                info = self.tree_processor.get_frag_info(i, e)
                if len(info["new_to_old"]) > 1:
                    self._add_fragment_to_batch(
                        batch_info, info, rev_i, rg, i, idx_offset
                    )
                    idx_offset += rg["graph"].number_of_nodes()

        if len(batch_info["new_info_dicts"]) > 0:
            batch_info.update(self._prepare_batch_tensors(batch_info, device))

        return batch_info

    def _add_fragment_to_batch(
        self, batch_info, info, rev_i, rg, fragment, idx_offset
    ):
        """Add fragment information to batch."""
        batch_info["new_info_dicts"].append(info)
        batch_info["reverse_idx"].append(rev_i)
        batch_info["new_graphs"].append(rg["graph"])
        batch_info["new_stack"].append(fragment)
        batch_info["batched_select"].append(info["new_to_old"] + idx_offset)
        batch_info["batched_num_nodes"].append(len(info["new_to_old"]))
        batch_info["batched_old_edge_idx"].append(
            batch_info["batched_old_edge_idx"][-1]
            + rg["graph"].number_of_edges()
        )

    def _prepare_batch_tensors(self, batch_info, device):
        """Prepare batch tensors from collected information."""
        batch_info["batched_select"] = torch.from_numpy(
            np.concatenate(batch_info["batched_select"])
        ).to(device)

        batch_info["batched_num_nodes"] = torch.LongTensor(
            batch_info["batched_num_nodes"]
        ).to(device)

        batch_info["batched_old_edge_idx"] = torch.LongTensor(
            batch_info["batched_old_edge_idx"][1:]
        ).to(device)

        # Create fragment batch - DGL handles device transfer
        frag_batch = dgl.batch(batch_info["new_graphs"]).to(device)
        frag_batch = frag_batch.subgraph(batch_info["batched_select"])

        batch_info["batched_num_edges"] = torch.bincount(
            torch.bucketize(
                frag_batch.edata[dgl.EID],
                batch_info["batched_old_edge_idx"],
                right=True,
            )
        )

        frag_batch.set_batch_num_nodes(batch_info["batched_num_nodes"])
        frag_batch.set_batch_num_edges(batch_info["batched_num_edges"])

        self.tree_processor.add_pe_embed(frag_batch)
        batch_info["frag_batch"] = frag_batch

        return batch_info

    def _process_fragment_forms(self, batch_info, engine, device):
        """Process fragment forms for the batch."""
        frag_forms = [i["form"] for i in batch_info["new_info_dicts"]]
        frag_form_vecs = [formula_to_dense(i) for i in frag_forms]

        return {
            "frag_forms": frag_forms,
            "frag_form_vecs": torch.FloatTensor(np.array(frag_form_vecs)).to(
                device
            ),
            "new_frag_hashes": [
                engine[ri].wl_hash(i)
                for i, ri in zip(
                    batch_info["new_stack"], batch_info["reverse_idx"]
                )
            ],
        }

    def _get_fragment_predictions(
        self,
        batch_info,
        frag_form_data,
        tracking_data,
        root_data,
        root_repr,
        device,
    ):
        """Get model predictions for fragments."""
        inds = torch.tensor(batch_info["reverse_idx"]).long().to(device)

        broken_nums = np.array(
            [
                tracking_data["frag_hash_to_entry"][ri][h]["max_broken"]
                for h, ri in zip(
                    frag_form_data["new_frag_hashes"],
                    batch_info["reverse_idx"],
                )
            ]
        )

        # Create a batch dictionary for the forward method
        batch = {
            "frag_graphs": batch_info["frag_batch"],
            "root_reprs": root_repr,
            "inds": inds,
            "broken_bonds": torch.FloatTensor(broken_nums)
            .unsqueeze(0)
            .to(device),
            "root_form_vecs": root_data["root_form_vec"],
            "frag_form_vecs": frag_form_data["frag_form_vecs"],
        }

        pred_leaving = self.forward(batch)

        return pred_leaving.cpu()

    def _process_probabilities(
        self, frag_hash_to_entry, max_nodes, threshold, batch_size
    ):
        """Process and sort fragment probabilities."""
        cur_probs = [
            sorted([i["prob_gen"] for i in fh2e.values()])[::-1]
            for fh2e in frag_hash_to_entry
        ]

        if max_nodes is None:
            min_prob = torch.full((batch_size,), threshold)
        else:
            cur_prob_len = torch.LongTensor([len(cp) for cp in cur_probs])
            thresh_prob = torch.FloatTensor(
                [cp[:max_nodes][-1] for cp in cur_probs]
            )
            min_prob = torch.where(
                cur_prob_len < max_nodes,
                torch.full_like(thresh_prob, threshold),
                thresh_prob,
            )

        return {"min_prob": min_prob, "cur_probs": cur_probs}

    def _process_fragment_predictions(
        self, batch_info, pred_leaving, frag_hash_to_entry, batch_size
    ):
        """Process fragment predictions and create sorted order."""
        new_items = list(
            zip(
                batch_info["new_stack"],
                batch_info["new_frag_hashes"],
                pred_leaving,
                [d["new_to_old"] for d in batch_info["new_info_dicts"]],
                batch_info["reverse_idx"],
            )
        )

        sorted_order = [[] for _ in range(batch_size)]

        for item_ind, item in enumerate(new_items):
            frag_hash = item[1]
            valid_atoms = len(item[3])
            pred_vals_f = torch.sigmoid(item[2])
            pred_vals_f = pred_vals_f[:valid_atoms]
            rev_idx = item[-1]
            parent_prob = frag_hash_to_entry[rev_idx][frag_hash]["prob_gen"]

            for atom_ind, (atom_pred, prob_gen) in enumerate(
                zip(pred_vals_f, parent_prob * pred_vals_f)
            ):
                sorted_order[rev_idx].append(
                    {
                        "item_ind": item_ind,
                        "atom_ind": atom_ind,
                        "prob_gen": prob_gen.item(),
                        "atom_pred": atom_pred.item(),
                        "orig_entry": item,
                    }
                )

        return [
            sorted(so, key=lambda x: -x["prob_gen"]) for so in sorted_order
        ]

    def _update_fragment_hashes(
        self, frag_to_hash, stack, new_frag_hashes, reverse_idx
    ):
        """Update fragment hash mappings."""
        for st, nfh, ri in zip(stack, new_frag_hashes, reverse_idx):
            frag_to_hash[ri][st] = nfh

    def _update_tracking_data(
        self,
        tracking_data,
        prob_data,
        sorted_data,
        batch_size,
        max_nodes,
        threshold,
    ):
        """Update tracking structures after processing batch."""
        new_stack = [[] for _ in range(batch_size)]

        batch_to_process = self._prepare_batch_for_processing(
            tracking_data,
            prob_data["min_prob"],
            sorted_data,
            batch_size,
            max_nodes,
            threshold,
        )

        new_vals = [auto_regressive_decode(**b) for b in batch_to_process]

        for rev_idx in range(batch_size):
            tracking_data["frag_hash_to_entry"][rev_idx] = new_vals[rev_idx][
                "frag_hash_to_entry"
            ]
            tracking_data["frag_to_hash"][rev_idx] = new_vals[rev_idx][
                "frag_to_hash"
            ]
            tracking_data["form_to_min_score"][rev_idx] = new_vals[rev_idx][
                "form_to_min_score"
            ]
            tracking_data["id_"][rev_idx] = new_vals[rev_idx]["id_"]
            new_stack[rev_idx] = new_vals[rev_idx]["new_stack"]

        tracking_data["stack"] = new_stack

    def _prepare_batch_for_processing(
        self,
        tracking_data,
        min_prob,
        sorted_order,
        batch_size,
        max_nodes,
        threshold,
    ):
        """Prepare batch data for processing."""
        return [
            {
                "frag_hash_to_entry": tracking_data["frag_hash_to_entry"][
                    rev_idx
                ],
                "frag_to_hash": tracking_data["frag_to_hash"][rev_idx],
                "form_to_min_score": tracking_data["form_to_min_score"][
                    rev_idx
                ],
                "engine": tracking_data["engine"][rev_idx],
                "min_prob": min_prob[rev_idx],
                "id_": tracking_data["id_"][rev_idx],
                "sorted_order": sorted_order[rev_idx],
                "depth": tracking_data["depth"],
                "max_nodes": max_nodes,
                "threshold": threshold,
            }
            for rev_idx in range(batch_size)
        ]

    def _filter_final_results(self, frag_hash_to_entry, form_to_min_score):
        """Filter results based on minimum scores."""
        return [
            {k: v for k, v in fh2e.items() if f2ms[v["form"]] == v["score"]}
            for fh2e, f2ms in zip(frag_hash_to_entry, form_to_min_score)
        ]

    def _prepare_final_output(
        self, frag_hash_to_entry, max_nodes, batched_input
    ):
        """Prepare final output based on filtering criteria."""
        if max_nodes is not None:
            max_nodes = int(max_nodes)
            return_entries = []
            for fh2e in frag_hash_to_entry:
                sorted_keys = sorted(
                    list(fh2e.keys()),
                    key=lambda x: -fh2e[x]["prob_gen"],
                )
                fh2e = {k: fh2e[k] for k in sorted_keys[:max_nodes]}
                return_entries.append(fh2e)
            frag_hash_to_entry = return_entries

        return frag_hash_to_entry if batched_input else frag_hash_to_entry[0]

    @staticmethod
    def parallel_consumer_decoder(data: Dict[str, Any]) -> Tuple[str, str]:
        """Static method for parallel processing."""
        param_dic = data["param_dic"]
        max_nodes = param_dic["max_nodes"]

        new_val = auto_regressive_decode(**param_dic)
        frag_hash_to_entry = new_val["frag_hash_to_entry"]
        form_to_min_score = new_val["form_to_min_score"]

        frag_hash_to_entry = {
            k: v
            for k, v in frag_hash_to_entry.items()
            if form_to_min_score[v["form"]] == v["score"]
        }

        if max_nodes is not None:
            sorted_keys = sorted(
                list(frag_hash_to_entry.keys()),
                key=lambda x: -frag_hash_to_entry[x]["prob_gen"],
            )
            frag_hash_to_entry = {
                k: frag_hash_to_entry[k] for k in sorted_keys[:max_nodes]
            }

        output_dict = {
            "root_inchi": data["root_inchi"],
            "name": data["name"],
            "frags": frag_hash_to_entry,
        }
        output_str = json.dumps(output_dict, indent=2)

        return data["out_name"], output_str
