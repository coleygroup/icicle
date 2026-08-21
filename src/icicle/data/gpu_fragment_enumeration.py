"""GPU-accelerated BFS fragment enumeration.

Replaces the CPU FragmentEngine BFS for streaming/batch inference. All BFS
state lives as batched DGL graphs + tensors on device.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import dgl
import numpy as np
import torch

from icicle.data.fragmentation_engine import Fragment, FragmentEngine
from icicle.utils.graph_utils import (
    batch_remove_single_atoms,
    form_vec_to_mass,
    frag_to_form_vec,
    msg_passing_frag_graph_hash,
    slice_batched_graph,
)


def enumerate_fragments_gpu(
    root_graph: dgl.DGLGraph,
    device,
    tree_processor,
    add_hs: bool,
    embed_elem_group: bool,
    h_shift_range: int,
    encode_formulae: bool,
    max_tree_depth: int = 3,
    max_broken_bonds: int = 6,
    max_nodes: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """GPU BFS for a single molecule.

    Returns the same input dict format as _fragments_to_intensity_inputs, so
    the intensity predictor forward pass is unchanged. All BFS state lives as
    batched DGL graphs + tensors on device; no Python Mol objects.
    """
    root_graph = root_graph.to(device)

    root_batch = dgl.batch([root_graph])
    root_broken = torch.zeros(1, device=device)

    root_hash = msg_passing_frag_graph_hash(root_batch)
    seen_hashes: set = set(map(tuple, root_hash.tolist()))

    # Collect surviving fragments per BFS depth as batched DGL graphs (no Python objects).
    all_batches: List[dgl.DGLGraph] = [root_batch]
    all_brokens: List[torch.Tensor] = [root_broken]

    frontier_batch = root_batch
    frontier_broken = root_broken

    for _depth in range(max_tree_depth):
        N = frontier_batch.batch_size
        if N == 0:
            break

        batch_num_nodes = frontier_batch.batch_num_nodes()
        total_n = int(batch_num_nodes.sum().item())

        # Build per-node local index and graph membership — fully vectorized, no Python loop.
        global_idx = torch.arange(total_n, device=device)
        node_off = torch.repeat_interleave(
            torch.cat(
                [
                    torch.zeros(1, dtype=torch.long, device=device),
                    batch_num_nodes[:-1].cumsum(0),
                ]
            ),
            batch_num_nodes,
        )
        sel_idx = global_idx - node_off  # local atom index within its graph
        batch_idx = torch.repeat_interleave(
            torch.arange(N, device=device), batch_num_nodes
        )

        # Core GPU op: remove every atom from every frontier fragment simultaneously.
        # CPU equivalent: the atom-removal inner loop in FragmentEngine.generate_fragments.
        new_batch, new_bond_orders, new_map = batch_remove_single_atoms(
            frontier_batch,
            batch_idx,
            sel_idx,
            {"broken_bonds": frontier_broken[batch_idx]},
        )
        if new_batch is None:
            break

        new_broken = new_map["broken_bonds"] + new_bond_orders
        new_hashes = msg_passing_frag_graph_hash(new_batch)

        # One GPU->CPU sync per depth level for duplicate/bond filtering (unavoidable).
        new_broken_cpu = new_broken.tolist()
        new_hashes_cpu = new_hashes.tolist()

        keep_indices = []
        for j in range(new_batch.batch_size):
            if new_broken_cpu[j] > max_broken_bonds:
                continue
            h_key = tuple(new_hashes_cpu[j])
            if h_key in seen_hashes:
                continue
            seen_hashes.add(h_key)
            keep_indices.append(j)

        if not keep_indices:
            break

        keep_idx = torch.tensor(keep_indices, dtype=torch.long, device=device)
        # Pure tensor sub-batch selection — no DGL object allocation per graph.
        frontier_batch, _, _ = slice_batched_graph(new_batch, keep_idx)
        frontier_broken = new_broken[keep_idx]

        all_batches.append(frontier_batch)
        all_brokens.append(frontier_broken)

    if len(all_batches) == 0:
        return None

    # Flatten all depth levels into one batched fragment graph.
    batched_frags = dgl.batch(all_batches)
    broken_t = torch.cat(all_brokens)  # [F]
    n_frags = batched_frags.batch_size

    if max_nodes is not None and n_frags > max_nodes:
        cap_idx = broken_t.argsort()[:max_nodes]
        batched_frags, _, _ = slice_batched_graph(batched_frags, cap_idx)
        broken_t = broken_t[cap_idx]
        n_frags = max_nodes

    tree_processor.add_pe_embed(batched_frags)

    # Fresh copy for root so positional encoding isn't applied twice.
    root_repr = dgl.batch([root_graph])
    tree_processor.add_pe_embed(root_repr)

    # Formula vectors → monoisotopic masses; all ops stay on device.
    with batched_frags.local_scope():
        form_vecs = frag_to_form_vec(batched_frags, add_hs, embed_elem_group)
    base_masses = form_vec_to_mass(form_vecs)  # [F]

    h_shifts = torch.arange(
        -h_shift_range, h_shift_range + 1, device=device, dtype=torch.float
    )
    masses_t = (base_masses.unsqueeze(-1) + h_shifts).unsqueeze(0).unsqueeze(2)

    # ind_maps: all F fragments belong to molecule 0 (single-molecule call).
    ind_maps = torch.zeros(n_frags, dtype=torch.long, device=device)
    num_frags = torch.tensor([n_frags], dtype=torch.long, device=device)
    h_range_t = torch.full(
        (1, n_frags), h_shift_range, dtype=torch.float, device=device
    )

    result: Dict[str, Any] = {
        "graphs": batched_frags,
        "root_reprs": root_repr,
        "ind_maps": ind_maps,
        "num_frags": num_frags,
        "broken_bonds": broken_t.unsqueeze(0),  # [1, F]
        "masses": masses_t,
        "max_add_hs": h_range_t,
        "max_remove_hs": h_range_t,
    }

    if encode_formulae:
        result["frag_form_vecs"] = form_vecs.unsqueeze(0)  # [1, F, elem_dim]
        result["root_form_vecs"] = form_vecs[:1].clone()

    return result


def enumerate_fragments_gpu_batched(
    root_graphs: List[dgl.DGLGraph],
    device,
    tree_processor,
    add_hs: bool,
    embed_elem_group: bool,
    h_shift_range: int,
    encode_formulae: bool,
    max_tree_depth: int = 3,
    max_broken_bonds: int = 6,
    max_nodes: Optional[int] = None,
) -> Optional[Tuple[Dict, List[int]]]:
    """GPU BFS for a full batch of molecules in one pass.

    All M molecules are batched into one DGL graph and share every GPU kernel.
    Molecule identity tracked via mol_id through batch_remove_single_atoms.
    Returns (bi_dict, frags_per_mol_list) or None on failure.
    """
    M = len(root_graphs)
    root_graphs_gpu = [g.to(device) for g in root_graphs]
    root_batch = dgl.batch(root_graphs_gpu)
    del root_graphs

    # mol_id floats propagate through batch_remove_single_atoms via map_info,
    # letting us track which child fragment belongs to which molecule without unbatching.
    frontier_mol_ids = torch.arange(M, device=device, dtype=torch.float32)
    frontier_broken = torch.zeros(M, device=device)

    root_hashes = msg_passing_frag_graph_hash(root_batch)  # [M, H]
    H_DIM = root_hashes.shape[1]
    MOL_STRIDE: int = 10**12
    # Random projection collapses H-dim hash to scalar; MOL_STRIDE namespaces by molecule
    # so duplicate detection works across all M molecules without a Python set per mol.
    _hash_proj = torch.randn(H_DIM, device=device, dtype=torch.float32).mul_(
        1e8
    )

    def _to_keys(
        h_vecs: torch.Tensor, mol_ids_f: torch.Tensor
    ) -> torch.Tensor:
        return mol_ids_f.long() * MOL_STRIDE + (h_vecs @ _hash_proj).long()

    all_seen_keys: torch.Tensor = _to_keys(root_hashes, frontier_mol_ids)

    all_batches: List[dgl.DGLGraph] = [root_batch]
    all_brokens: List[torch.Tensor] = [frontier_broken]
    all_mol_ids: List[torch.Tensor] = [frontier_mol_ids]

    frontier_batch = root_batch

    for _depth in range(max_tree_depth):
        if max_nodes is not None:
            fm = frontier_mol_ids.round().long()
            per_m = torch.bincount(fm, minlength=M)
            if int(per_m.max().item()) > max_nodes:
                fb = frontier_broken
                capped: List[torch.Tensor] = []
                for m in range(M):
                    m_loc = (fm == m).nonzero(as_tuple=True)[0]
                    if m_loc.numel() > max_nodes:
                        m_loc = m_loc[fb[m_loc].argsort()[:max_nodes]]
                    if m_loc.numel() > 0:
                        capped.append(m_loc)
                if not capped:
                    break
                cap_idx = torch.cat(capped)
                frontier_batch, _, _ = slice_batched_graph(
                    frontier_batch, cap_idx
                )
                frontier_broken = frontier_broken[cap_idx]
                frontier_mol_ids = frontier_mol_ids[cap_idx]

        N = frontier_batch.batch_size
        if N == 0:
            break

        batch_num_nodes = frontier_batch.batch_num_nodes()
        total_n = int(batch_num_nodes.sum().item())

        global_idx = torch.arange(total_n, device=device)
        node_off = torch.repeat_interleave(
            torch.cat(
                [
                    torch.zeros(1, dtype=torch.long, device=device),
                    batch_num_nodes[:-1].cumsum(0),
                ]
            ),
            batch_num_nodes,
        )
        sel_idx = global_idx - node_off
        batch_idx = torch.repeat_interleave(
            torch.arange(N, device=device), batch_num_nodes
        )

        # Single GPU op expands every frontier fragment of every molecule simultaneously.
        # Equivalent to the atom-removal inner loop in FragmentEngine.generate_fragments,
        # but across all N frontier fragments × all atoms in one CUDA kernel call.
        new_batch, new_bond_orders, new_map = batch_remove_single_atoms(
            frontier_batch,
            batch_idx,
            sel_idx,
            {
                "broken_bonds": frontier_broken[batch_idx],
                "mol_id": frontier_mol_ids[batch_idx],
            },
        )
        if new_batch is None:
            break

        new_broken = new_map["broken_bonds"] + new_bond_orders
        new_mol_ids_f = new_map["mol_id"]
        new_hashes = msg_passing_frag_graph_hash(new_batch)

        # All filtering (duplicate + bond limit) stays on GPU — no CPU sync needed here.
        new_keys = _to_keys(new_hashes, new_mol_ids_f)
        bb_ok = new_broken <= max_broken_bonds
        not_seen = ~torch.isin(new_keys, all_seen_keys)
        keep_idx_all = (bb_ok & not_seen).nonzero(as_tuple=True)[0]

        if keep_idx_all.numel() == 0:
            break

        # Intra-batch dedup: same fragment reachable via multiple paths.
        new_keys_keep = new_keys[keep_idx_all]
        unique_keys, inv = torch.unique(
            new_keys_keep, return_inverse=True, sorted=False
        )
        n_keep = keep_idx_all.numel()
        # first_occ[c] = index of the first element with inv == c
        first_occ = torch.empty(
            unique_keys.shape[0], dtype=torch.long, device=device
        )
        first_occ.scatter_(
            0, inv.flip(0), torch.arange(n_keep, device=device).flip(0)
        )
        keep_idx = keep_idx_all[first_occ]

        all_seen_keys = torch.cat([all_seen_keys, unique_keys])

        frontier_batch, _, _ = slice_batched_graph(new_batch, keep_idx)
        del new_batch
        frontier_broken = new_broken[keep_idx]
        frontier_mol_ids = new_mol_ids_f[keep_idx]

        all_batches.append(frontier_batch)
        all_brokens.append(frontier_broken)
        all_mol_ids.append(frontier_mol_ids)

    batched_frags = dgl.batch(all_batches)
    broken_cat = torch.cat(all_brokens)
    mol_ids_cat = torch.cat(all_mol_ids).round().long()
    del all_batches, all_brokens, all_mol_ids
    n_total = batched_frags.batch_size

    if max_nodes is not None:
        frags_per_mol_tmp = torch.bincount(mol_ids_cat, minlength=M)
        if int(frags_per_mol_tmp.max().item()) > max_nodes:
            keep_mask = torch.ones(n_total, dtype=torch.bool, device=device)
            for m in range(M):
                if frags_per_mol_tmp[m].item() > max_nodes:
                    idx_m = (mol_ids_cat == m).nonzero(as_tuple=True)[0]
                    excess = idx_m[broken_cat[idx_m].argsort()[max_nodes:]]
                    keep_mask[excess] = False
            keep_idx = keep_mask.nonzero(as_tuple=True)[0]
            batched_frags, _, _ = slice_batched_graph(batched_frags, keep_idx)
            broken_cat = broken_cat[keep_idx]
            mol_ids_cat = mol_ids_cat[keep_idx]
            n_total = batched_frags.batch_size

    tree_processor.add_pe_embed(batched_frags)

    root_repr = dgl.batch(root_graphs_gpu)
    tree_processor.add_pe_embed(root_repr)

    frags_per_mol = torch.bincount(mol_ids_cat, minlength=M)
    max_frags = int(frags_per_mol.max().item())

    # Sort by molecule so fragment order matches row-major layout of padded [M, max_frags] tensors.
    order = mol_ids_cat.argsort(stable=True)
    sorted_mol_ids = mol_ids_cat[order]
    batched_frags, _, _ = slice_batched_graph(batched_frags, order)
    broken_cat = broken_cat[order]
    mol_ids_cat = sorted_mol_ids

    with batched_frags.local_scope():
        form_vecs = frag_to_form_vec(batched_frags, add_hs, embed_elem_group)
    base_masses = form_vec_to_mass(form_vecs)

    h_shifts = torch.arange(
        -h_shift_range, h_shift_range + 1, device=device, dtype=torch.float
    )
    H = h_shifts.numel()
    masses_all = base_masses.unsqueeze(-1) + h_shifts  # [total_F, H]

    mol_offsets = torch.cat(
        [
            torch.zeros(1, dtype=torch.long, device=device),
            frags_per_mol[:-1].cumsum(0),
        ]
    )
    pos_in_mol = (
        torch.arange(n_total, device=device) - mol_offsets[mol_ids_cat]
    )

    broken_padded = torch.zeros(M, max_frags, device=device)
    masses_padded = torch.zeros(M, max_frags, 1, H, device=device)
    h_range_t = torch.full(
        (M, max_frags), h_shift_range, dtype=torch.float, device=device
    )

    broken_padded[mol_ids_cat, pos_in_mol] = broken_cat
    masses_padded[mol_ids_cat, pos_in_mol, 0, :] = masses_all

    bi: Dict[str, Any] = {
        "graphs": batched_frags,
        "root_reprs": root_repr,
        "ind_maps": mol_ids_cat,
        "num_frags": frags_per_mol,
        "broken_bonds": broken_padded,
        "masses": masses_padded,
        "max_add_hs": h_range_t,
        "max_remove_hs": h_range_t,
    }

    if encode_formulae:
        formula_dim = form_vecs.shape[-1]
        frag_forms = torch.zeros(M, max_frags, formula_dim, device=device)
        frag_forms[mol_ids_cat, pos_in_mol, :] = form_vecs
        bi["frag_form_vecs"] = frag_forms
        with root_repr.local_scope():
            root_form_vecs_all = frag_to_form_vec(
                root_repr, add_hs, embed_elem_group
            )
        bi["root_form_vecs"] = root_form_vecs_all  # [M, elem_dim]

    return bi, frags_per_mol.tolist()


def populate_frag_to_entry_from_gpu(
    ind_maps_cpu: torch.Tensor,
    num_nodes_cpu: torch.Tensor,
    node_starts_cpu: torch.Tensor,
    n_id_all_cpu: torch.Tensor,
    mol_rank: int,
    base_masses_np: np.ndarray,
    engine: FragmentEngine,
) -> None:
    """Populate engine.frag_to_entry from GPU enumeration results.

    Atom membership per fragment is read from the n_id ndata preserved through
    batch_remove_single_atoms. No CPU BFS (generate_fragments) is called. All
    tensor-to-CPU transfers must be done once by the caller; this function only
    does cheap Python/numpy ops per fragment.
    """
    natoms = engine.natoms
    frag_idxs = (ind_maps_cpu == mol_rank).nonzero(as_tuple=True)[0].tolist()
    for i, fi in enumerate(frag_idxs):
        start = node_starts_cpu[fi].item()
        length = num_nodes_cpu[fi].item()
        atom_inds = n_id_all_cpu[start : start + length].numpy()
        if len(atom_inds) == 0 or int(atom_inds.max()) >= natoms:
            logging.debug(
                f"populate_frag_to_entry: skipping fragment fi={fi} "
                f"(atom_inds max {int(atom_inds.max())} >= natoms {natoms})"
            )
            continue
        bitmask = int(sum(1 << int(a) for a in atom_inds))
        form = engine.formula_from_kept_inds(atom_inds)
        base_mass = (
            float(base_masses_np[i]) if i < len(base_masses_np) else 0.0
        )
        engine.frag_to_entry[f"gpu_{fi}"] = Fragment(
            frag=bitmask,
            id=fi,
            sibling_hashes=[],
            parents=[],
            parent_hashes=[],
            parent_ind_removed=[],
            max_broken=0,
            tree_depth=0,
            score=0.0,
            base_mass=base_mass,
            form=form,
            frag_hs=0,
            max_remove_hs=0,
            max_add_hs=0,
        )
