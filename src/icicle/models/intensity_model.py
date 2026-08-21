"""Model to infer intensities from fragmentation DAGs.

This model is used to infer the intensities of fragments from their
fragmentation DAG. It can be trained either on full enumeration on-the-fly or a
pre-computed fragmentation DAG.
"""

import logging
from typing import Any, Dict, List, Optional, Union

import dgl
import dgl.nn as dgl_nn
import torch
from torch import nn

from icicle.data.fragmentation_engine import MAX_BONDS
from icicle.data.isotope_distribution import (
    DifferentiableIsotopePatternModule,
)
from icicle.models.base_model import BaseIntensityModel
from icicle.models.embedder import get_embedder
from icicle.models.encoder import GNNEncoder
from icicle.models.layers import MLPBlocks, get_clones
from icicle.models.losses import (
    CompositeWeightedCosineSimilarityLoss,
    CosineSimilarityLoss,
    EntropyLoss,
    JensenShannonLoss,
    MSELoss,
    WeightedCosineSimilarityLoss,
)
from icicle.utils import (
    ELEMENT_DIM,
    ELEMENT_GROUP_DIM,
    ISOTOPE_PATTERNS_SIMPLIFIED,
    MAX_H,
    NORM_VEC,
    element_to_ind,
    pad_packed_tensor,
)


class IntensityPredictor(BaseIntensityModel):
    """Model that predicts the intensities of fragments."""

    def __init__(
        self,
        hidden_size: int,
        min_mz: float,
        max_mz: float,
        bin_width: float,
        gnn_message_passing_steps: int,
        fragment_feature_mlp_layers: int,
        intra_graph_attention_layers: int,
        learning_rate: float,
        lr_decay_rate: float,
        weight_decay: float,
        dropout: float,
        pool_op: str,
        pe_embed_k: int,
        max_broken: int,
        h_shift_range: int,
        inter_fragment_attention_layers: int,
        loss_fn: str,
        inject_early: bool,
        embed_elem_group: bool,
        encode_formulae: bool,
        add_hs: bool,
        lower_mz_cutoff: float,
        add_isotopes: bool,
        min_isotope_intensity: float = 0.01,
        max_isotopes: int = 0,
        warmup: int = 1000,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            min_mz=min_mz, max_mz=max_mz, bin_width=bin_width, **kwargs
        )

        self.save_hyperparameters()

        self.hidden_size = hidden_size
        self.min_mz = min_mz
        self.max_mz = max_mz
        self.bin_width = bin_width
        self.gnn_message_passing_steps = gnn_message_passing_steps
        self.fragment_feature_mlp_layers = fragment_feature_mlp_layers
        self.intra_graph_attention_layers = intra_graph_attention_layers
        self.learning_rate = learning_rate
        self.lr_decay_rate = lr_decay_rate
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.pool_op = pool_op
        self.pe_embed_k = pe_embed_k
        self.max_broken = max_broken
        self.h_shift_range = h_shift_range
        self.inter_fragment_attention_layers = inter_fragment_attention_layers
        self.loss_fn_name = loss_fn
        self.inject_early = inject_early
        self.embed_elem_group = embed_elem_group
        self.encode_formulae = encode_formulae
        self.add_hs = add_hs
        self.add_isotopes = add_isotopes
        self.max_isotopes = max_isotopes
        self.min_isotope_intensity = min_isotope_intensity
        self.lower_mz_cutoff = lower_mz_cutoff
        self.warmup = warmup

        self.node_feats = self._calculate_node_feats()
        self._init_dimensions()
        self._init_embedders()
        self._init_network_params()
        self._build_network()
        self._init_loss_and_outputs()

        self.add_isotopes = add_isotopes
        self.min_isotope_intensity = min_isotope_intensity
        self.max_total_isotope_shift_for_module = 15

        self.element_to_form_idx = element_to_ind
        num_formula_elements = len(element_to_ind)

        max_iso_variants = max(
            len(p) for p in ISOTOPE_PATTERNS_SIMPLIFIED.values()
        )

        iso_shifts_data = torch.zeros(
            num_formula_elements, max_iso_variants, dtype=torch.float32
        )
        iso_abundances_data = torch.zeros(
            num_formula_elements, max_iso_variants, dtype=torch.float32
        )

        for element_symbol, iso_list in ISOTOPE_PATTERNS_SIMPLIFIED.items():
            if element_symbol in self.element_to_form_idx:
                el_idx = self.element_to_form_idx[element_symbol]
                for i, (shift, abundance) in enumerate(iso_list):
                    iso_shifts_data[el_idx, i] = float(shift)
                    iso_abundances_data[el_idx, i] = float(abundance)

        if self.add_isotopes:
            self.differentiable_isotope_module = DifferentiableIsotopePatternModule(
                element_to_idx=self.element_to_form_idx,
                iso_shifts_data=iso_shifts_data,
                iso_abundances_data=iso_abundances_data,
                min_intensity_threshold=self.min_isotope_intensity,
                max_total_isotope_shift=self.max_total_isotope_shift_for_module,
            )

            self.register_buffer("iso_shifts_base_data", iso_shifts_data)
            self.register_buffer(
                "iso_abundances_base_data", iso_abundances_data
            )

    def _calculate_node_feats(self):
        dim = ELEMENT_DIM
        if self.embed_elem_group:
            dim += ELEMENT_GROUP_DIM
        if self.add_hs:
            dim += MAX_H
        if self.pe_embed_k > 0:
            dim += self.pe_embed_k
        if self.inject_early:
            dim += self.hidden_size
        return dim

    def _init_dimensions(self):
        self.formula_in_dim = 0
        if self.encode_formulae:
            self.formula_dim = NORM_VEC.shape[0]
            self.formula_in_dim = (
                self.formula_dim * get_embedder("abs-sines").num_dim * 2
            )

    def _init_embedders(self):
        if self.encode_formulae:
            self.embedder = get_embedder("abs-sines")

    def _init_network_params(self):
        self.max_broken_dim = self.max_broken + 1
        self.broken_clamp = self.max_broken
        self.broken_onehot = nn.Parameter(
            torch.eye(self.max_broken_dim),
            requires_grad=True,
        )

    def _build_network(self):
        self.gnn_encoder = GNNEncoder(
            hidden_size=self.hidden_size,
            num_step_message_passing=self.gnn_message_passing_steps,
            set_transform_layers=self.intra_graph_attention_layers,
            gnn_node_feats=self.node_feats,
            gnn_edge_feats=MAX_BONDS,
            dropout=self.dropout,
        )

        if self.inject_early:
            root_node_feats_base = (
                ELEMENT_DIM
                + (ELEMENT_GROUP_DIM if self.embed_elem_group else 0)
                + (MAX_H if self.add_hs else 0)
                + (self.pe_embed_k if self.pe_embed_k > 0 else 0)
            )
            self.root_module = GNNEncoder(
                hidden_size=self.hidden_size,
                num_step_message_passing=self.gnn_message_passing_steps,
                set_transform_layers=self.intra_graph_attention_layers,
                gnn_node_feats=root_node_feats_base,
                gnn_edge_feats=MAX_BONDS,
                dropout=self.dropout,
            )
        else:
            self.root_module = self.gnn_encoder

        mlp_input_size = (
            self.hidden_size * 3 + self.max_broken_dim + self.formula_in_dim
        )

        self.intermediate_out = MLPBlocks(
            input_size=mlp_input_size,
            hidden_size=self.hidden_size,
            output_size=self.hidden_size,
            dropout=self.dropout,
            num_layers=max(1, self.fragment_feature_mlp_layers),
            use_residuals=True,
        )

        if self.inter_fragment_attention_layers > 0:
            trans_layer = nn.TransformerEncoderLayer(
                self.hidden_size,
                nhead=4,
                batch_first=True,
                norm_first=True,
                dim_feedforward=self.hidden_size * 2,
                dropout=self.dropout,
            )
            self.trans_layers = get_clones(
                trans_layer, self.inter_fragment_attention_layers
            )
        else:
            self.trans_layers = nn.ModuleList()

        if self.pool_op == "avg":
            self.pool = dgl_nn.AvgPooling()
        elif self.pool_op == "attn":
            self.pool = dgl_nn.GlobalAttentionPooling(
                nn.Linear(self.hidden_size, 1)
            )
        elif self.pool_op == "max":
            self.pool = dgl_nn.MaxPooling()
        elif self.pool_op == "sum":
            self.pool = dgl_nn.SumPooling()
        else:
            raise NotImplementedError(f"Unsupported pooling: {self.pool_op}")

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Parameter):
                if module.requires_grad:
                    if module not in self.broken_onehot:
                        nn.init.uniform_(module, -0.1, 0.1)

    def _init_loss_and_outputs(self):
        self.mass_bins = int((self.max_mz - self.min_mz) / self.bin_width)
        buckets = torch.linspace(self.min_mz, self.max_mz, self.mass_bins)
        self.inten_buckets = nn.Parameter(
            buckets.double(), requires_grad=False
        )
        self.output_size = self.h_shift_range * 2 + 1
        self.output_map = nn.Linear(self.hidden_size, self.output_size)
        self.isomer_attn_out = nn.Linear(self.hidden_size, self.output_size)

        nn.init.xavier_uniform_(self.output_map.weight, gain=0.1)
        nn.init.zeros_(self.output_map.bias)
        nn.init.xavier_uniform_(self.isomer_attn_out.weight, gain=0.1)
        nn.init.zeros_(self.isomer_attn_out.bias)

        self._setup_loss_functions()

    def _setup_loss_functions(self):
        loss_config = {
            "cosine_similarity": CosineSimilarityLoss(),
            "weighted_cosine_nist_gc": WeightedCosineSimilarityLoss(
                weighting="nist_gc"
            ),
            "composite_weighted_cosine_nist_gc": CompositeWeightedCosineSimilarityLoss(
                weighting="nist_gc"
            ),
            "mse": MSELoss(),
            "jensen_shannon": JensenShannonLoss(),
            "entropy": EntropyLoss(),
        }

        self.loss_fn = loss_config.get(self.loss_fn_name)
        if self.loss_fn is None:
            raise ValueError(f"Unknown loss function {self.loss_fn_name}")

    def forward(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        device = self.inten_buckets.device

        graphs = batch["frag_graphs"]
        root_repr = batch["root_reprs"]
        ind_maps = batch["inds"]
        num_frags = batch["num_frags"]
        broken = batch["broken_bonds"]

        root_embeddings = self._process_root_embeddings(root_repr)

        frag_embeddings, avg_frags = self._process_fragment_embeddings(
            graphs, root_embeddings, ind_maps
        )

        hidden = self._build_hidden_states(
            ext_root=root_embeddings[ind_maps],
            avg_frags=avg_frags,
            broken=broken,
            num_frags=num_frags,
            device=device,
            root_forms=batch.get("root_form_vecs"),
            frag_forms=batch.get("frag_form_vecs"),
        )

        if self.inter_fragment_attention_layers > 0:
            hidden = self._process_with_transformers(
                hidden=hidden, num_frags=num_frags, device=device
            )

        batch_size, max_frags, hidden_dim = hidden.shape
        raw_output = self.output_map(hidden)
        raw_attn_weights = self.isomer_attn_out(hidden)
        raw_output = torch.relu(raw_output)

        arange_frags = torch.arange(max_frags, device=device)
        frag_mask = arange_frags[None, :] < num_frags[:, None]
        expanded_frag_mask = frag_mask.unsqueeze(-1).expand_as(raw_output)

        raw_output = raw_output * expanded_frag_mask.float()
        raw_attn_weights = raw_attn_weights.masked_fill(
            ~expanded_frag_mask, -torch.inf
        )

        if "masses" in batch and batch["masses"] is not None:
            fragment_base_masses = batch["masses"][:, :, 0, self.h_shift_range]
        else:
            logging.warning(
                "Batch does not contain 'masses'. Cannot accurately bin H-shifted fragments."
            )
            fragment_base_masses = torch.zeros(
                (batch_size, max_frags), device=device
            )

        h_shifts = torch.arange(
            -self.h_shift_range,
            self.h_shift_range + 1,
            device=device,
            dtype=torch.float32,
        )
        shifted_fragment_masses = fragment_base_masses.unsqueeze(
            -1
        ) + h_shifts.unsqueeze(0).unsqueeze(0)

        # Common calculations for both isotope and non-isotope paths
        # Use flatten() which ensures contiguous memory layout for faster GPU operations
        shifted_fragment_masses_flat = shifted_fragment_masses.flatten()
        raw_output_flat = raw_output.flatten()
        raw_attn_weights_flat = raw_attn_weights.flatten()
        expanded_frag_mask_flat = expanded_frag_mask.flatten()

        bin_indices = torch.bucketize(
            shifted_fragment_masses_flat, self.inten_buckets, right=False
        )
        bin_indices = torch.clamp(bin_indices, 0, self.mass_bins - 1)

        valid_indices = torch.nonzero(expanded_frag_mask_flat).squeeze(1)
        bin_indices_valid = bin_indices[valid_indices]
        raw_output_valid = raw_output_flat[valid_indices]

        batch_idx_expanded = (
            torch.arange(batch_size, device=device).unsqueeze(1).unsqueeze(2)
        )
        batch_idx_expanded = batch_idx_expanded.expand(
            batch_size, max_frags, self.output_size
        ).contiguous()
        batch_idx_expanded_flat = batch_idx_expanded.view(-1)[valid_indices]

        attention_per_spectrum = raw_attn_weights.view(batch_size, -1)
        attention_per_spectrum_masked = attention_per_spectrum.masked_fill(
            ~expanded_frag_mask.reshape(batch_size, -1), -torch.inf
        )
        attention_normalized_per_spectrum = torch.softmax(
            attention_per_spectrum_masked, dim=-1
        )
        attention_normalized_per_spectrum_valid = (
            attention_normalized_per_spectrum.reshape(-1)[valid_indices]
        )

        raw_output_valid = raw_output_flat[valid_indices]
        attention_normalized_per_spectrum_valid = (
            attention_normalized_per_spectrum.reshape(-1)[valid_indices]
        )
        shifted_fragment_masses_valid = shifted_fragment_masses_flat[
            valid_indices
        ]

        # Get the original batch indices for these valid peaks
        batch_idx_expanded_orig_peaks = (
            torch.arange(batch_size, device=device)
            .unsqueeze(1)
            .unsqueeze(2)
            .expand(batch_size, max_frags, self.output_size)
            .contiguous()
            .view(-1)[valid_indices]
        )
        if self.add_isotopes:
            # Expand to (B, max_frags, H_shifts, formula_dim) then flatten
            frag_form_vecs_full_expanded = (
                batch["frag_form_vecs"]
                .unsqueeze(2)
                .expand(
                    batch_size,
                    max_frags,
                    self.output_size,
                    self.formula_dim,  # use self.formula_dim from _init_dimensions
                )
                .reshape(-1, self.formula_dim)
            )  # (B*max_frags*H_shifts, formula_dim)

            # Select only the fragment formula vectors corresponding to `valid_indices`
            frag_form_vecs_valid_aligned = frag_form_vecs_full_expanded[
                valid_indices
            ]  # (N_orig_peaks, formula_dim)

            # Call the differentiable isotope module
            (
                all_isotope_masses_t,
                all_isotope_intensities_t,
                all_isotope_batch_indices_t,
            ) = self.differentiable_isotope_module(
                raw_intensities_per_h_shift=raw_output_valid,
                base_masses_per_h_shift=shifted_fragment_masses_valid,
                frag_form_vecs=frag_form_vecs_valid_aligned,
                attn_weights_norm=attention_normalized_per_spectrum_valid,
                original_batch_indices=batch_idx_expanded_orig_peaks,
            )

            # Perform the final binning of the isotope-expanded peaks
            bin_indices_iso = torch.bucketize(
                all_isotope_masses_t, self.inten_buckets, right=False
            )
            bin_indices_iso = torch.clamp(
                bin_indices_iso, 0, self.mass_bins - 1
            )

            global_flat_indices_iso = (
                all_isotope_batch_indices_t * self.mass_bins + bin_indices_iso
            )

            output_binned = torch.zeros(
                batch_size * self.mass_bins,
                device=device,
                dtype=torch.float32,
            )
            output_binned.scatter_add_(
                dim=0,
                index=global_flat_indices_iso.long(),
                src=all_isotope_intensities_t,
            )
            output_binned = output_binned.view(batch_size, self.mass_bins)
        else:
            global_flat_indices = (
                batch_idx_expanded_flat * self.mass_bins + bin_indices_valid
            )
            output_binned = self._process_non_isotope_binning(
                batch_size,
                self.mass_bins,
                global_flat_indices,
                raw_output_valid,
                attention_normalized_per_spectrum_valid,
                device,
            )

        output_binned = torch.relu(output_binned)

        if self.lower_mz_cutoff > 0:
            mz_mask = (self.inten_buckets >= self.lower_mz_cutoff).float()
            output_binned = output_binned * mz_mask.unsqueeze(0)

        return {"output_binned": output_binned}

    def _process_non_isotope_binning(
        self,
        batch_size,
        mass_bins,
        global_flat_indices,
        raw_output_valid,
        attention_normalized_per_spectrum_valid,
        device,
    ):
        """Helper to encapsulate the original non-isotope binning logic."""
        temp_output_flat = torch.zeros(
            batch_size * mass_bins, device=device, dtype=torch.float32
        )
        temp_output_flat.scatter_add_(
            dim=0,
            index=global_flat_indices.long(),
            src=(raw_output_valid * attention_normalized_per_spectrum_valid),
        )
        return temp_output_flat.view(batch_size, mass_bins)

    def _process_root_embeddings(
        self, root_repr: Union[dgl.DGLGraph, torch.Tensor]
    ):
        """Process root molecule representations."""
        with root_repr.local_scope():
            if self.inject_early:
                root_node_feats_base = (
                    ELEMENT_DIM
                    + (ELEMENT_GROUP_DIM if self.embed_elem_group else 0)
                    + (MAX_H if self.add_hs else 0)
                    + (self.pe_embed_k if self.pe_embed_k > 0 else 0)
                )
                root_embeddings_h = self.root_module(root_repr)
                return self.pool(root_repr, root_embeddings_h)
            else:
                root_embeddings_h = self.root_module(root_repr)
                return self.pool(root_repr, root_embeddings_h)

    def _process_fragment_embeddings(
        self,
        graphs: dgl.DGLGraph,
        root_embeddings: torch.Tensor,
        ind_maps: torch.Tensor,
    ):
        """Process fragment embeddings."""
        ext_root = root_embeddings[ind_maps]

        concat_list = [graphs.ndata["h"]]

        if self.inject_early:
            ext_root_atoms = torch.repeat_interleave(
                ext_root, graphs.batch_num_nodes(), dim=0
            )
            concat_list.append(ext_root_atoms)

        with graphs.local_scope():
            graphs.ndata["h"] = torch.cat(concat_list, -1).float()
            frag_embeddings = self.gnn_encoder(graphs)
            avg_frags = self.pool(graphs, frag_embeddings)
        return frag_embeddings, avg_frags

    def _build_hidden_states(
        self,
        ext_root: torch.Tensor,
        avg_frags: torch.Tensor,
        broken: torch.Tensor,
        num_frags: torch.Tensor,
        device: torch.device,
        root_forms: Optional[torch.Tensor] = None,
        frag_forms: Optional[torch.Tensor] = None,
    ):
        """Build hidden representations incorporating root context, fragment
        features, broken bonds, and formula encodings."""

        batch_size, max_frags_padded_for_broken = broken.shape

        arange_max_frags = torch.arange(
            max_frags_padded_for_broken, device=device
        )
        valid_frag_mask_1d_for_broken = (
            arange_max_frags[None, :] < num_frags[:, None]
        ).view(-1)

        broken_flat = broken.view(-1)[valid_frag_mask_1d_for_broken]

        broken_clamped = torch.clamp(broken_flat, max=self.broken_clamp)
        broken_onehots = self.broken_onehot[broken_clamped.long()]

        hidden_flat = torch.cat(
            [ext_root, ext_root - avg_frags, avg_frags, broken_onehots], dim=1
        )

        if (
            self.encode_formulae
            and root_forms is not None
            and frag_forms is not None
            and hasattr(self, "embedder")
        ):
            batch_size, max_frags_padded_for_forms, formula_dim = (
                frag_forms.shape
            )

            arange_max_frags_for_forms = torch.arange(
                max_frags_padded_for_forms, device=device
            )
            valid_frag_mask_1d_for_forms = (
                arange_max_frags_for_forms[None, :] < num_frags[:, None]
            ).view(-1)

            frag_forms_flat = frag_forms.view(-1, formula_dim)[
                valid_frag_mask_1d_for_forms
            ]

            root_forms_expanded = torch.repeat_interleave(
                root_forms, num_frags, dim=0
            )

            diffs = root_forms_expanded - frag_forms_flat

            form_encodings = self.embedder(frag_forms_flat.float())
            diff_encodings = self.embedder(diffs.float())

            hidden_flat = torch.cat(
                [hidden_flat, form_encodings, diff_encodings], dim=-1
            )

        padded_hidden = pad_packed_tensor(hidden_flat, num_frags, 0)

        padded_hidden = self.intermediate_out(padded_hidden)

        return padded_hidden

    def _process_with_transformers(
        self,
        hidden: torch.Tensor,
        num_frags: torch.Tensor,
        device: torch.device,
    ):
        """Apply transformer layers for inter-fragment attention."""
        if len(self.trans_layers) == 0:
            return hidden

        arange_frags = torch.arange(hidden.shape[1], device=device)
        attn_mask = ~(arange_frags[None, :] < num_frags[:, None])

        for idx, trans_layer in enumerate(self.trans_layers):
            scale_factor = 1.0 / (idx + 1)
            residual = hidden
            hidden = trans_layer(hidden, src_key_padding_mask=attn_mask)
            hidden = residual + hidden * scale_factor

        return hidden

    def predict_intensities(
        self,
        graphs: dgl.DGLGraph,
        root_reprs: Union[dgl.DGLGraph, torch.Tensor],
        ind_maps: torch.Tensor,
        num_frags: torch.Tensor,
        **kwargs: Any,
    ) -> Dict[str, List[torch.Tensor]]:
        """Predict intensities (BaseIntensityModel interface)."""
        batch = {
            "frag_graphs": graphs,
            "root_reprs": root_reprs,
            "inds": ind_maps,
            "num_frags": num_frags,
            "broken_bonds": kwargs.get(
                "broken_bonds", torch.zeros_like(ind_maps)
            ),
            "masses": kwargs.get("masses"),
            "root_form_vecs": kwargs.get("root_form_vecs"),
            "frag_form_vecs": kwargs.get("frag_form_vecs"),
            "formulae": kwargs.get("formulae"),
        }

        outputs = self.forward(batch)
        intensities_binned = outputs["output_binned"]

        # Only transfer to CPU if not in training mode to avoid performance hit
        if self.training:
            intensity_list = [
                intensities_binned[i] for i in range(len(num_frags))
            ]
        else:
            intensity_list = [
                intensities_binned[i].cpu() for i in range(len(num_frags))
            ]

        return {"spec": intensity_list}

    def _common_step(
        self, batch: Dict[str, torch.Tensor], name: str = "train"
    ) -> Dict[str, torch.Tensor]:
        """Common logic for training, validation, and test steps."""

        pred_obj = self.forward(batch)
        pred_intensities = pred_obj["output_binned"]

        targets = batch["inten_targs"]

        if pred_intensities.shape != targets.shape:
            logging.warning(
                f"Shape mismatch between predictions {pred_intensities.shape} and"
                f"targets {targets.shape}. Skipping loss calculation for this batch."
            )
            return {
                "loss": torch.tensor(
                    0.0, device=pred_intensities.device, requires_grad=True
                )
            }

        loss_result = self.loss_fn(
            pred_intensities, targets, self.inten_buckets
        )
        loss = loss_result["loss"].mean()

        batch_size = len(batch.get("names", batch["num_frags"]))

        self.log(
            f"{name}_loss",
            loss,
            batch_size=batch_size,
            prog_bar=True,
            on_step=(name == "train"),
            on_epoch=True,
        )

        return {"loss": loss}

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss_dict = self._common_step(batch, name="train")
        return loss_dict["loss"]

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss_dict = self._common_step(batch, name="val")
        return loss_dict["loss"]

    def test_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss_dict = self._common_step(batch, name="test")
        return loss_dict["loss"]

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            eps=1e-8,
            betas=(0.9, 0.999),
        )

        if self.warmup > 0:

            def lr_lambda(step):
                if step < self.warmup:
                    return float(step) / float(max(1, self.warmup))
                else:
                    return self.lr_decay_rate ** (
                        (step - self.warmup) / 1000.0
                    )

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "frequency": 1,
                    "interval": "step",
                },
            }
        else:
            return optimizer
