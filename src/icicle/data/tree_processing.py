"""Tree processing module.

Handles conversion of molecular structures to graph representations, including
node and edge featurization, positional encoding, and tree structure
processing.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import dgl
import numpy as np
import torch
from numpy.typing import NDArray

from icicle.utils import (
    ELEMENT_DIM,
    ELEMENT_GROUP_DIM,
    MAX_H,
    element_to_group,
    element_to_position,
    formula_from_smi,
    formula_to_dense,
    random_walk_pe,
)

from .fragmentation_engine import (
    MAX_BONDS,
    FragmentationParams,
    FragmentEngine,
    HeteroWeights,
)

FragInfo = Dict[str, Union[np.ndarray, str]]
MolDict = Dict[str, Any]


@dataclass
class TreeProcessingConfig:
    """Configuration for molecular tree processing."""

    pe_embed_k: int
    add_hs: bool
    embed_elem_group: bool
    min_mz: float
    max_mz: float
    bin_width: float
    max_broken_bonds: int
    max_tree_depth: int
    num_h_shifts: int
    hetero_weights_cc: int
    hetero_weights_other: int

    def get_fragmentation_params(
        self, mol_str_type: str = "smiles"
    ) -> FragmentationParams:
        """Get FragmentationParams from current config."""
        return FragmentationParams(
            max_tree_depth=self.max_tree_depth,
            max_broken_bonds=self.max_broken_bonds,
            num_h_shifts=self.num_h_shifts,
            mol_str_type=mol_str_type,
            hetero_weights=HeteroWeights(
                cc_bond=self.hetero_weights_cc,
                other_bond=self.hetero_weights_other,
            ),
        )


class TreeProcessor:
    """Process molecular trees for fragmentation prediction.

    Handles conversion of molecular structures to graph representations,
    including node and edge featurization, positional encoding, and tree
    structure processing.
    """

    def __init__(
        self, config: Optional[TreeProcessingConfig] = None, **kwargs: Any
    ) -> None:
        """Initialize tree processor.

        Args:
            config: Configuration for tree processing. If None, uses defaults.
            **kwargs: Override config parameters
        """
        if config is None:
            raise ValueError(
                "TreeProcessor requires an explicit TreeProcessingConfig; "
                "no defaults are available."
            )
        self.config = config

        # Override with kwargs for backward compatibility
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        self.logger = logging.getLogger(__name__)

        # Setup mass binning
        self.bins = np.linspace(
            self.config.min_mz,
            self.config.max_mz,
            int(
                (self.config.max_mz - self.config.min_mz)
                / self.config.bin_width
            ),
        )

    def get_frag_info(
        self,
        frag: int,
        engine: FragmentEngine,
    ) -> FragInfo:
        """Get fragment information and index mappings.

        Args:
            frag: Fragment ID
            engine: Fragmentation engine

        Returns
        -------
            Dictionary containing mappings and formula
        """
        kept_atom_inds, _ = engine.get_present_atoms(frag)
        kept_atom_inds_arr = np.array(kept_atom_inds)
        form = engine.formula_from_kept_inds(kept_atom_inds_arr)

        num_atoms = engine.natoms
        num_kept = len(kept_atom_inds)
        new_inds = np.arange(num_kept)

        old_to_new: NDArray[np.int_] = np.zeros(num_atoms, dtype=np.int_)
        old_to_new[kept_atom_inds] = new_inds

        new_to_old: NDArray[np.int_] = np.zeros(num_kept, dtype=np.int_)
        new_to_old[new_inds] = kept_atom_inds

        return {
            "new_to_old": new_to_old,
            "old_to_new": old_to_new,
            "form": form,
        }

    def add_pe_embed(self, graph):
        pe_embeds = random_walk_pe(
            graph, k=self.config.pe_embed_k, eweight_name="e_ind"
        )
        graph.ndata["h"] = torch.cat((graph.ndata["h"], pe_embeds), -1).float()
        return graph

    def process_tree(
        self,
        tree: dict,
        include_targets: bool = False,
        last_row: bool = False,
        convert_to_dgl: bool = True,
    ) -> Dict[str, Any]:
        """Process complete molecular tree.

        Args:
            tree: Tree data from MAGMA
            include_targets: Whether to include intensity targets
            last_row: Whether to process final tree depth
            convert_to_dgl: Whether to convert to DGL graphs

        Returns
        -------
            Processed tree data
        """
        if convert_to_dgl:
            out_dict = self._convert_to_dgl(tree, include_targets, last_row)
            if "collision_energy" in tree:
                out_dict["collision_energy"] = tree["collision_energy"]
        else:
            out_dict = tree

        # Process graphs
        dgl_inputs = out_dict["dgl_frags"]
        root_repr = out_dict["root_repr"]

        # Add positional encoding
        if self.config.pe_embed_k > 0:
            for graph in dgl_inputs:
                self.add_pe_embedding(graph)

            if isinstance(root_repr, dgl.DGLGraph):
                self.add_pe_embedding(root_repr)

        # Process intensity targets
        if include_targets:
            out_dict = self._process_intensity_targets(out_dict)

        return out_dict

    def _process_intensity_targets(
        self, out_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process intensity targets by binning them according to
        configuration.

        Args:
            out_dict: Dictionary containing tree processing outputs including intensity targets
        """
        if "inten_targs" not in out_dict or out_dict["inten_targs"] is None:
            raise ValueError(
                "Intensity targets not found in output dictionary"
            )

        intens = out_dict["inten_targs"]

        # Clip digitized mass values to valid bin range
        bin_posts = np.clip(
            np.digitize(intens[:, 0], self.bins), 0, len(self.bins) - 1
        )

        # Create new output array and fill with maximum intensity per bin
        new_out = np.zeros_like(self.bins)
        for bin_post, inten in zip(bin_posts, intens[:, 1]):
            new_out[bin_post] = max(new_out[bin_post], inten)

        # Update the intensity targets in the output dictionary
        out_dict["inten_targs"] = new_out

        return out_dict

    def process_tree_gen(
        self, tree: dict, convert_to_dgl: bool = True
    ) -> Dict[str, Any]:
        """Process tree for generation model."""
        proc_out = self.process_tree(
            tree,
            include_targets=False,
            last_row=False,
            convert_to_dgl=convert_to_dgl,
        )

        # Extract relevant keys
        keys = {
            "root_repr",
            "dgl_frags",
            "targs",
            "max_broken",
            "form_vecs",
            "root_form_vec",
            "collision_energy",
        }
        dgl_tree = {k: proc_out[k] for k in keys if k in proc_out}

        return {"dgl_tree": dgl_tree, "tree": tree}

    def _convert_to_dgl_inten(self, tree):
        """Convert intensity data to DGL format."""
        # Create a dummy single-node graph for root representation
        root_repr = dgl.graph(([0], [0]))
        root_repr.ndata["h"] = torch.zeros(1, self.get_node_feats())

        # Create a simple graph for the fragment
        frag_graph = dgl.graph(([0], [0]))
        frag_graph.ndata["h"] = torch.zeros(1, self.get_node_feats())

        return {
            "root_repr": root_repr,
            "dgl_frags": [frag_graph],  # Single fragment
        }

    def process_tree_inten(
        self, tree: dict, convert_to_dgl: bool = True
    ) -> Dict[str, Any]:
        """Process tree for intensity model."""
        proc_out = self.process_tree(
            tree,
            include_targets=False,
            last_row=True,
            convert_to_dgl=convert_to_dgl,
        )

        keys = {
            "root_repr",
            "dgl_frags",
            "masses",
            "inten_frag_ids",
            "max_remove_hs",
            "max_add_hs",
            "max_broken",
            "form_vecs",
            "root_form_vec",
            "collision_energy",
        }
        dgl_tree = {k: proc_out[k] for k in keys if k in proc_out}

        return {"dgl_tree": dgl_tree, "tree": tree}

    def process_tree_inten_pred(
        self, tree: dict, convert_to_dgl: bool = True
    ) -> Dict[str, Any]:
        """Process tree for intensity prediction."""
        proc_out = self.process_tree(
            tree,
            include_targets=False,
            last_row=True,
            convert_to_dgl=convert_to_dgl,
        )

        # Extract relevant keys
        keys = {
            "root_repr",
            "dgl_frags",
            "masses",
            "inten_targs",
            "inten_frag_ids",
            "max_remove_hs",
            "max_add_hs",
            "max_broken",
            "form_vecs",
            "root_form_vec",
            "collision_energy",
        }
        dgl_tree = {k: proc_out[k] for k in keys if k in proc_out}

        return {"dgl_tree": dgl_tree, "tree": tree}

    def add_pe_embedding(self, graph: dgl.DGLGraph) -> None:
        """Add positional encoding to graph."""
        pe_embeds = random_walk_pe(
            graph, k=self.config.pe_embed_k, eweight_name="e_ind"
        )
        graph.ndata["h"] = torch.cat(
            [graph.ndata["h"], pe_embeds], dim=-1
        ).float()

    def get_node_feats(self) -> int:
        """Get number of node features."""
        dim = ELEMENT_DIM  # Base element dimension

        if self.config.embed_elem_group:
            dim += ELEMENT_GROUP_DIM

        if self.config.add_hs:
            dim += MAX_H

        if self.config.pe_embed_k > 0:
            dim += self.config.pe_embed_k

        return dim

    def _convert_to_dgl(
        self,
        tree: Dict[str, Any],
        include_targets: bool = True,
        last_row: bool = False,
    ) -> Dict[str, Any]:
        """_convert_to_dgl.

        Args:
            tree (dict): tree dictionary
            include_targets (bool): Try to add inten targets for supervising
                the inten model
            last_row:
        """
        root_smiles = tree["root_canonical_smiles"]

        fragmentation_params = self.config.get_fragmentation_params()

        engine = FragmentEngine(
            mol_str=root_smiles,
            params=fragmentation_params,
        )
        bottom_depth = engine.params.max_tree_depth

        root_frag = engine.get_root_frag()
        root_graph_dict = self.featurize_frag(
            frag=root_frag,
            engine=engine,
        )
        root_repr = root_graph_dict["graph"]

        root_form = formula_from_smi(root_smiles)

        (
            masses,
            inten_frag_ids,
            dgl_inputs,
            frag_targets,
            max_broken_list,
        ) = (
            [],
            [],
            [],
            [],
            [],
        )
        forms = []
        max_remove_hs_list, max_add_hs_list = [], []
        for k, sub_frag in tree["frags"].items():
            max_broken_num = sub_frag["max_broken"]
            tree_depth = sub_frag["tree_depth"]

            # Skip because we never fragment last row
            if (not last_row) and (tree_depth == bottom_depth):
                continue

            binary_targs = sub_frag["atoms_pulled"]
            frag = sub_frag["frag"]

            # Get frag dict and target
            frag_dict = self.featurize_frag(
                frag,
                engine,
            )
            forms.append(frag_dict["form"])
            old_to_new = frag_dict["old_to_new"]
            graph = frag_dict["graph"]
            max_broken_list.append(max_broken_num)

            max_remove_hs_list.append(sub_frag["max_remove_hs"])
            max_add_hs_list.append(sub_frag["max_add_hs"])

            inten_frag_ids.append(k)

            # For gen model only!!
            targ_vec = np.zeros(graph.num_nodes())
            for j in old_to_new[binary_targs]:
                targ_vec[j] = 1

            graph = frag_dict["graph"]

            # Define targ vec
            dgl_inputs.append(graph)
            masses.append(sub_frag["base_mass"])
            frag_targets.append(torch.from_numpy(targ_vec))

        # Intensity targets come from experimental spectra, not from the tree.
        # MAGMa trees don't have intensity data - they only mark which
        # fragments are observed. Handled by the dataset, not the tree
        # processor, which only provides the graph structure.
        inten_targets_arr = None
        if include_targets:
            pass

        masses = (
            engine.shift_bucket_masses[None, None, :]
            + np.array(masses)[:, None, None]
        )

        max_remove_hs_arr = np.array(max_remove_hs_list, dtype=np.int_)
        max_add_hs_arr = np.array(max_add_hs_list, dtype=np.int_)
        max_broken_arr = np.array(max_broken_list, dtype=np.int_)

        # Feat each form
        all_form_vecs: NDArray[Any] = np.array(
            [formula_to_dense(i) for i in forms]
        )

        root_form_vec = formula_to_dense(root_form)

        out_dict = {
            "root_repr": root_repr,
            "dgl_frags": dgl_inputs,
            "masses": masses,
            "inten_frag_ids": inten_frag_ids,
            "max_remove_hs": max_remove_hs_arr,
            "max_add_hs": max_add_hs_arr,
            "max_broken": max_broken_arr,
            "targs": frag_targets,
            "form_vecs": all_form_vecs,
            "root_form_vec": root_form_vec,
        }

        return out_dict

    def _build_mol_node_features(self, engine: FragmentEngine) -> np.ndarray:
        """Precompute and cache full-molecule node feature matrix on the
        engine.

        All fragments are atom-subsets of the root molecule, so we build the
        N_atoms × feat_dim matrix once and reuse it by indexing with
        kept_atom_inds. Eliminates per-atom isinstance loops for every
        fragment.
        """
        cache_key = "_tp_node_feats"
        if hasattr(engine, cache_key):
            return getattr(engine, cache_key)

        feats = np.vstack(
            [element_to_position[s] for s in engine.atom_symbols]
        ).astype(np.float32)

        if self.config.embed_elem_group:
            groups = np.vstack(
                [element_to_group[s] for s in engine.atom_symbols]
            ).astype(np.float32)
            feats = np.concatenate([feats, groups], axis=1)

        if self.config.add_hs:
            h_feats = np.eye(MAX_H, dtype=np.float32)[
                engine.atom_hs.astype(int)
            ]
            feats = np.concatenate([feats, h_feats], axis=1)

        setattr(engine, cache_key, feats)
        return feats

    def featurize_frag(
        self,
        frag: int,
        engine: FragmentEngine,
        add_random_walk: bool = False,
    ) -> MolDict:
        """Featurize a molecular fragment."""
        kept_atom_inds, _ = engine.get_present_atoms(frag)
        kept_bond_orders, kept_bonds = engine.get_present_edges(frag)

        info = self.get_frag_info(frag, engine)
        old_to_new: NDArray[np.int_] = np.array(
            info["old_to_new"], dtype=np.int_
        )

        new_bond_inds: NDArray[np.int_] = np.empty((0, 2), dtype=np.int_)
        if kept_bonds:
            new_bond_inds = old_to_new[np.vstack(kept_bonds)]

        # Index into precomputed full-molecule feature matrix — no per-atom loop
        mol_node_feats = self._build_mol_node_features(engine)
        node_data = torch.from_numpy(mol_node_feats[np.array(kept_atom_inds)])

        bond_types_arr = np.array(kept_bond_orders)
        if len(new_bond_inds) > 0:
            src = torch.from_numpy(new_bond_inds[:, 0])
            dst = torch.from_numpy(new_bond_inds[:, 1])
            bond_types_tensor = torch.from_numpy(bond_types_arr)
            src_cat = torch.cat([src, dst])
            dst_cat = torch.cat([dst, src])
            bond_types_cat = torch.cat([bond_types_tensor, bond_types_tensor])
            bond_feats = torch.eye(MAX_BONDS)[bond_types_cat]
        else:
            src_cat = torch.empty(0, dtype=torch.long)
            dst_cat = torch.empty(0, dtype=torch.long)
            bond_types_cat = torch.empty(0, dtype=torch.long)
            bond_feats = torch.empty((0, MAX_BONDS))

        graph = dgl.graph((src_cat, dst_cat), num_nodes=len(kept_atom_inds))
        graph.ndata["h"] = node_data.float()
        graph.edata["e"] = bond_feats.float()
        graph.edata["e_ind"] = bond_types_cat

        if add_random_walk and hasattr(self.config, "add_pe_embed"):
            self.add_pe_embedding(graph)

        return {"graph": graph, **info}
