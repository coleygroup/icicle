"""Graph utility functions for molecular fragment processing and GPU-
accelerated graph operations.

Covers batched DGL graph manipulation (slicing, atom removal, padding), random-
walk positional encodings, GPU-based connected-component labelling, fragment
hashing via message passing, and conversion between fragment graphs, chemical
formula vectors, and monoisotopic masses.
"""

from typing import Dict, Tuple

import dgl
import dgl.function as fn
import torch
import torch_scatter


def pad_packed_tensor(input, lengths, value):
    """pad_packed_tensor."""
    old_shape = input.shape
    device = input.device
    if not isinstance(lengths, torch.Tensor):
        lengths = lengths.clone().detach().long().to(device)
    else:
        lengths = lengths.to(device)
    max_len = (lengths.max()).item()

    batch_size = len(lengths)
    x = input.new(batch_size * max_len, *old_shape[1:])
    x.fill_(value)

    # Initialize a tensor with an index for every value in the array
    index = torch.ones(len(input), dtype=torch.int64, device=device)

    # Row shifts
    row_shifts = torch.cumsum(max_len - lengths, 0)

    # Calculate shifts for second row, third row... nth row (not the n+1th row)
    # Expand this out to match the shape of all entries after the first row
    row_shifts_expanded = row_shifts[:-1].repeat_interleave(lengths[1:])

    # Add this to the list of inds _after_ the first row
    cumsum_inds = torch.cumsum(index, 0) - 1
    cumsum_inds[lengths[0] :] += row_shifts_expanded
    x[cumsum_inds] = input
    return x.view(batch_size, max_len, *old_shape[1:])


def random_walk_pe(g, k, eweight_name=None):
    """Random Walk Positional Encoding, as introduced in
    `Graph Neural Networks with Learnable Structural and Positional Representations
    <https://arxiv.org/abs/2110.07875>`__

    This function computes the random walk positional encodings as landing probabilities
    from 1-step to k-step, starting from each node to itself.

    Parameters
    ----------
    g : DGLGraph
        The input graph. Must be homogeneous.
    k : int
        The number of random walk steps. The paper found the best value to be 16 and 20
        for two experiments.
    eweight_name : str, optional
        The name to retrieve the edge weights. Default: None, not using the edge weights.

    Returns
    -------
    Tensor
        The random walk positional encodings of shape :math:`(N, k)`, where :math:`N` is the
        number of nodes in the input graph.

    Example
    -------
    >>> import dgl
    >>> g = dgl.graph(([0,1,1], [1,1,0]))
    >>> dgl.random_walk_pe(g, 2)
    tensor([[0.0000, 0.5000],
            [0.5000, 0.7500]])
    """
    device = g.device
    N = g.num_nodes()  # number of nodes
    M = g.num_edges()  # number of edges

    row, col = g.edges()

    if eweight_name is None:
        value = torch.ones(M, device=device)
    else:
        value = g.edata[eweight_name].squeeze().to(device)
    # value_norm = torch_scatter.scatter(value, col, dim_size=N, reduce='sum').clamp(min=1)[col]
    value_norm = (
        torch_scatter.scatter(value, row, dim_size=N, reduce="sum")[row]
        + 1e-30
    )
    value = value / value_norm

    if N <= 2_000:  # Dense code path for faster computation:
        adj = torch.zeros((N, N), device=row.device)
        adj[row, col] = value
        loop_index = torch.arange(N, device=row.device)
    adj = torch.sparse_coo_tensor(
        indices=torch.stack((row, col)), values=value, size=(N, N)
    )

    def get_pe(out: torch.Tensor) -> torch.Tensor:
        if out.is_sparse:
            out = out.coalesce()
            row, col = out.indices()
            value = out.values()
            select = row == col
            ret_val = torch.zeros(N, dtype=out.dtype, device=out.device)
            ret_val[row[select]] = value[select]
            return ret_val
        return out[loop_index, loop_index]

    out = adj
    pe_list = [get_pe(out)]
    for _ in range(k - 1):
        out = out @ adj
        pe_list.append(get_pe(out))

    pe = torch.stack(pe_list, dim=-1)

    return pe


def connected_components(
    edge_index: torch.LongTensor, num_nodes: int
) -> torch.LongTensor:
    """GPU connected-components via label-propagation.

    Starting with each node's label = its 1-based index, repeatedly lets
    each node adopt the minimum label of its neighbours until convergence.
    Finally compresses labels to a 0-based range.

    Parameters
    ----------
    edge_index : LongTensor[2, E]
        Undirected GPU edge list.
    num_nodes : int
        Total number of nodes.

    Returns
    -------
    LongTensor[num_nodes]
        Component ID per node in [0..C]; 0 means isolated.
    """
    device = edge_index.device
    src, dst = edge_index
    labels = torch.arange(1, num_nodes + 1, device=device)

    # Fixed iterations, no CPU-GPU sync (no torch.equal check). Must exceed
    # the largest possible fragment-graph diameter: a linear alkyl chain near
    # the 750 Da MW cutoff has ~53 carbons -> diameter ~52. 64 gives margin
    # without a meaningful perf cost. Under-converging here silently produces
    # multiple labels for one true component, which corrupts the fragment
    # count downstream and previously caused a CUDA device-side assert
    # (out-of-bounds index) in slice_batched_graph.
    for _ in range(64):
        lbl_src = labels[src]
        lbl_dst = labels[dst]
        min_to_dst, _ = torch_scatter.scatter_min(
            lbl_src, dst, dim=0, dim_size=num_nodes
        )
        min_to_src, _ = torch_scatter.scatter_min(
            lbl_dst, src, dim=0, dim_size=num_nodes
        )
        best = torch.minimum(min_to_dst, min_to_src)
        labels = torch.minimum(labels, best)

    deg = torch.zeros(num_nodes, device=device, dtype=torch.long)
    deg.scatter_add_(0, src, torch.ones_like(src))
    deg.scatter_add_(0, dst, torch.ones_like(dst))
    iso = deg == 0

    non_iso_labels = labels[~iso]
    _, inv = torch.unique(non_iso_labels, return_inverse=True)
    out = torch.zeros(num_nodes, device=device, dtype=torch.long)
    out[~iso] = inv + 1
    return out


def slice_batched_graph(
    bg: dgl.DGLGraph, batch_idx: torch.LongTensor
) -> Tuple[dgl.DGLGraph, torch.Tensor, torch.Tensor]:
    """Replicate and reorder fragments in a batched DGL graph.

    Parameters
    ----------
    bg : DGLGraph
        Batched graph containing M fragments.
    batch_idx : LongTensor[K]
        Indices (0 <= idx < M) of fragments to replicate (duplicates allowed).

    Returns
    -------
    new_bg : DGLGraph
        New batched graph of the K requested fragments.
    sizes : LongTensor[K]
        Node counts per new fragment.
    offsets : LongTensor[K]
        Cumulative node offsets.
    """
    device = batch_idx.device
    nn_t = bg.batch_num_nodes()
    ne_t = bg.batch_num_edges()

    old_node_off = torch.cat(
        [torch.tensor([0], device=device), nn_t[:-1].cumsum(0)], dim=0
    )
    old_edge_off = torch.cat(
        [torch.tensor([0], device=device), ne_t[:-1].cumsum(0)], dim=0
    )

    sizes = nn_t[batch_idx]
    offsets = torch.cat(
        [torch.tensor([0], device=device), sizes[:-1].cumsum(0)], dim=0
    )
    sizes_e = ne_t[batch_idx]
    offsets_e = torch.cat(
        [torch.tensor([0], device=device), sizes_e[:-1].cumsum(0)], dim=0
    )

    old_node_idx = (
        torch.arange(sizes.sum().item(), device=device)
        - torch.repeat_interleave(offsets, sizes)
        + old_node_off[batch_idx].repeat_interleave(nn_t[batch_idx])
    )
    new_h = bg.ndata["h"][old_node_idx]
    new_n_id = bg.ndata["n_id"][old_node_idx]

    src, dst = bg.edges()
    edge_pos = (
        torch.arange(sizes_e.sum().item(), device=device)
        - torch.repeat_interleave(offsets_e, sizes_e)
        + old_edge_off[batch_idx].repeat_interleave(ne_t[batch_idx])
    )
    edge_j = torch.arange(len(batch_idx), device=device).repeat_interleave(
        ne_t[batch_idx]
    )

    old_src = src[edge_pos]
    old_dst = dst[edge_pos]
    old_frag = batch_idx[edge_j]

    new_src = (old_src - old_node_off[old_frag]) + offsets[edge_j]
    new_dst = (old_dst - old_node_off[old_frag]) + offsets[edge_j]

    new_e = bg.edata["e"][edge_pos]
    new_e_ind = bg.edata["e_ind"][edge_pos]

    new_bg = dgl.graph((new_src, new_dst), num_nodes=new_h.size(0))
    new_bg.ndata["h"] = new_h
    new_bg.ndata["n_id"] = new_n_id
    new_bg.edata["e"] = new_e
    new_bg.edata["e_ind"] = new_e_ind
    new_bg.set_batch_num_nodes(sizes.tolist())
    new_bg.set_batch_num_edges(ne_t[batch_idx].tolist())

    return new_bg, sizes, offsets


def batch_remove_single_atoms(
    frag_batch: dgl.DGLGraph,
    batch_idx: torch.LongTensor,
    sel_idx: torch.LongTensor,
    map_info: Dict[str, torch.Tensor],
) -> Tuple:
    """For K candidate atom removals across a batch of graphs, remove one atom
    per candidate, find connected components, and aggregate metadata.

    Parameters
    ----------
    frag_batch : DGLGraph
        Batched graph of B input fragments.
    batch_idx : LongTensor[K]
        Which graph (0..B-1) each removal applies to.
    sel_idx : LongTensor[K]
        Node index *within* that graph to delete.
    map_info : dict[str, Tensor[K]]
        Per-removal metadata to propagate to resulting fragments.

    Returns
    -------
    subg : DGLGraph or None
        Batched graph of resulting fragments, or None if none produced.
    broken_bonds : LongTensor[F] or None
        Cut-bond count per fragment.
    mapped_info : dict or None
        Propagated map_info values per fragment.
    """
    device = frag_batch.device
    K = batch_idx.size(0)

    job_batch, sizes, offsets = slice_batched_graph(frag_batch, batch_idx)

    global_removed = offsets + sel_idx
    N_tot = job_batch.num_nodes()
    rm_flat = torch.zeros(N_tot, dtype=torch.bool, device=device)
    rm_flat[global_removed] = True

    src, dst = job_batch.edges(order="eid")
    is_cut = rm_flat[src] ^ rm_flat[dst]
    cut_eids = torch.nonzero(is_cut, as_tuple=True)[0]
    src_c, dst_c = src[cut_eids], dst[cut_eids]
    rm_src = rm_flat[src_c]
    kept_nodes = torch.where(rm_src, dst_c, src_c)
    bond_types = job_batch.edata["e_ind"][cut_eids]

    subg = dgl.node_subgraph(job_batch, ~rm_flat)
    orig_ids = subg.ndata[dgl.NID]
    subg.ndata.pop("_ID", None)
    subg.edata.pop("_ID", None)
    inv_map = torch.full((N_tot,), -1, dtype=torch.long, device=device)
    inv_map[orig_ids] = torch.arange(orig_ids.numel(), device=device)
    kept_local = inv_map[kept_nodes]

    u_sub, v_sub = subg.edges(order="eid")
    comp_flat = connected_components(
        torch.stack([u_sub, v_sub], 0), num_nodes=subg.num_nodes()
    )

    comp_for_cut = comp_flat[kept_local]
    keep_comp_mask = (comp_flat > 0) & torch.isin(comp_flat, comp_for_cut)
    subg = dgl.node_subgraph(subg, keep_comp_mask)
    subg.ndata.pop("_ID", None)
    subg.edata.pop("_ID", None)
    rm_flat[~rm_flat.clone()] = ~keep_comp_mask
    comp_flat = comp_flat[keep_comp_mask] - 1

    if len(comp_flat) == 0:
        return None, None, None

    uniq_vals, comp_for_cut = torch.unique(comp_for_cut, return_inverse=True)
    if uniq_vals.min() > 0:
        comp_for_cut += 1
    comp_flat = torch.unique(comp_flat, return_inverse=True)[1]

    perm = torch.argsort(comp_flat)
    subg = dgl.reorder_graph(
        subg, "custom", permute_config={"nodes_perm": perm}
    )
    comp_flat = comp_flat[perm]
    batched_num_nodes = torch.bincount(comp_flat)
    comp_per_edge = comp_flat[subg.edges(order="eid")[0]]
    batched_num_edges = torch.bincount(comp_per_edge)
    subg.set_batch_num_nodes(batched_num_nodes)
    subg.set_batch_num_edges(batched_num_edges)

    job_idx_map = torch.repeat_interleave(
        torch.arange(K, device=device), sizes
    )

    def _scatter(src_t, reduce="sum"):
        out = torch.full(
            (comp_flat.max() + 2,), -1, device=device, dtype=src_t.dtype
        )
        out.scatter_reduce_(0, comp_for_cut, src_t, reduce, include_self=False)
        return out[1:]

    broken_bonds = torch.floor(_scatter(bond_types, reduce="sum").float() / 2)
    mapped_info = {
        k: _scatter(v[job_idx_map[kept_nodes]], reduce="max")
        for k, v in map_info.items()
    }

    return subg, broken_bonds, mapped_info


def msg_passing_frag_graph_hash(
    graph: dgl.DGLGraph, feat_dim: int = 32
) -> torch.Tensor:
    """Hash each fragment in a batched graph via one round of message passing.

    Parameters
    ----------
    graph : DGLGraph
        Batched fragment graph with ndata['h'] and edata['e_ind'].
    feat_dim : int
        Unused; kept for API compatibility.

    Returns
    -------
    Tensor[N_graphs, node_feat_dim]
        Per-graph hash vectors.
    """
    graph.apply_edges(
        lambda edges: {
            "e_ind_float": edges.data["e_ind"].to(dtype=torch.float32)
        }
    )
    graph.update_all(fn.u_mul_e("h", "e_ind_float", "m"), fn.sum("m", "h_new"))
    # Include node-feature sum so isolated-node fragments are differentiated
    hash_val = dgl.sum_nodes(graph, "h_new") + dgl.sum_nodes(graph, "h")
    del graph.ndata["h_new"]
    del graph.edata["e_ind_float"]
    return hash_val


def frag_to_form_vec(
    frag_graph: dgl.DGLGraph, add_hs: bool, embed_elem_group: bool
) -> torch.Tensor:
    """Sum atom one-hot features per fragment to obtain formula vectors.

    Parameters
    ----------
    frag_graph : DGLGraph
        Batched fragment graph with ndata['h'].
    add_hs : bool
        Whether H counts are encoded in node features.
    embed_elem_group : bool
        Whether element-group features are concatenated after element one-hots.

    Returns
    -------
    Tensor[N_frags, CHEM_ELEMENT_NUM]
        Formula vector per fragment.
    """
    from icicle.utils.chem.constants import (
        CHEM_ELEMENT_NUM,
        ELEMENT_GROUP_DIM,
        MAX_H,
        element_to_ind,
    )

    frag_h = frag_graph.ndata["h"]
    frag_graph.ndata["_h_heavy"] = frag_h[:, :CHEM_ELEMENT_NUM]
    form_vecs = dgl.sum_nodes(frag_graph, "_h_heavy")

    if add_hs:
        start = CHEM_ELEMENT_NUM + (
            ELEMENT_GROUP_DIM if embed_elem_group else 0
        )
        end = start + MAX_H
        h_counts = torch.sum(
            frag_h[:, start:end]
            * torch.arange(
                MAX_H, device=frag_graph.device, dtype=frag_h.dtype
            ),
            dim=1,
            keepdim=True,
        )
        frag_graph.ndata["_h_H"] = h_counts
        form_vecs[:, element_to_ind["H"]] = dgl.sum_nodes(
            frag_graph, "_h_H"
        ).squeeze(1)
        del frag_graph.ndata["_h_H"]

    del frag_graph.ndata["_h_heavy"]
    return form_vecs


def form_vec_to_mass(form_vecs: torch.Tensor) -> torch.Tensor:
    """Convert formula vectors to monoisotopic masses.

    Parameters
    ----------
    form_vecs : Tensor[N, CHEM_ELEMENT_NUM]
        Formula vectors as returned by frag_to_form_vec.

    Returns
    -------
    Tensor[N]
        Mass per fragment.
    """
    from icicle.utils.chem.constants import CHEM_MASSES

    device, dtype = form_vecs.device, form_vecs.dtype
    mass_vec = torch.tensor(CHEM_MASSES, device=device, dtype=dtype)
    return (form_vecs[:, : mass_vec.shape[0]] @ mass_vec).squeeze(-1)
