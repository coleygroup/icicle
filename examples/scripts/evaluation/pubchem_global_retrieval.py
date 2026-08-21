#!/usr/bin/env python
"""PubChem Global Retrieval Evaluation.

Evaluates exact retrieval ranks for every NIST test query against the full
PubChem prediction HDF5 (~87M spectra).  Two code paths:

  "all" level  — sequential GPU scan of the entire HDF5 (true global rank).
  RI windows   — binary-search on pre-sorted RI index to read only the window
                 for each query; O(log N + window_size) per query instead of O(N).

Requires sort_idx_{ri_type} and sorted_ri_{ri_type} datasets in the HDF5.
Build them first with:
    uv run examples/scripts/evaluation/add_sort_index_to_hdf5.py \
        --hdf5 data/PubChem/pubchem_predictions_with_ik14_master.hdf5

Usage
-----
uv run examples/scripts/evaluation/pubchem_global_retrieval.py
Override config values: output_dir=results/my_run  ri_types=[StdNP]  top_n_levels=[all,1000]
"""

import json
import logging
import time
from pathlib import Path

import h5py
import hydra
import joblib
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from icicle.analysis.metrics import (
    batch_composite_sim,
    batch_cosine_sim,
    batch_entropy_sim,
    batch_weighted_cosine_sim,
)

log = logging.getLogger(__name__)


METRIC_FNS = {
    "cosine": batch_cosine_sim,
    "entropy": batch_entropy_sim,
    "weighted_cosine": batch_weighted_cosine_sim,
    "composite_cosine": batch_composite_sim,
}


def _compute_sim(
    m: str,
    cands: torch.Tensor,
    queries: torch.Tensor,
    mz_t: torch.Tensor,
    entropy_cand_chunk: int,
    entropy_query_batch: int,
) -> torch.Tensor:
    """Route metric to the appropriate similarity function."""
    if m in ("entropy", "composite_cosine"):
        fn = batch_entropy_sim if m == "entropy" else batch_composite_sim
        parts = []
        for cs in range(0, cands.shape[0], entropy_cand_chunk):
            parts.append(
                fn(
                    cands[cs : cs + entropy_cand_chunk],
                    queries,
                    mz_weights=mz_t,
                    query_batch_size=entropy_query_batch,
                )
            )
        return torch.cat(parts, dim=0)
    kwargs = {"mz_weights": mz_t} if m == "weighted_cosine" else {}
    return METRIC_FNS[m](cands, queries, **kwargs)


def _bin_spectrum(
    masses: np.ndarray,
    intensities: np.ndarray,
    min_mz: float,
    max_mz: float,
    bin_width: float,
    n_bins: int,
) -> np.ndarray:
    spec = np.zeros(n_bins, dtype=np.float32)
    bins = np.floor((masses - min_mz) / bin_width).astype(int)
    valid = (bins >= 0) & (bins < n_bins)
    np.add.at(spec, bins[valid], intensities[valid])
    if spec.max() > 0:
        spec /= spec.max()
    return spec


def load_nist_queries(cfg: DictConfig, n_bins: int) -> pd.DataFrame:
    """Load NIST test queries with ground-truth spectra.

    Split membership (train/val/test) comes from the plain split TSV
    (nist_split_path) — the FULL test set, independent of RI availability.
    RI columns are then LEFT-joined on top so RI-window/MW retrieval can
    still filter to RI-having queries downstream, but "all"-only global
    retrieval is NOT silently restricted to the RI-matched subset (that was
    a real bug: nist_ri_path used to be the sole source of split
    membership via an INNER join, which meant "all" mode only ever scored
    ~14.8k RI-matched molecules instead of the true ~33.5k test-set size).
    """
    labels = pd.read_csv(
        cfg.nist_labels_path,
        sep="\t",
        usecols=["mol_id", "inchi_key", "standardized_smiles", "mw"],
    )
    split_df = pd.read_csv(cfg.nist_split_path, sep="\t")
    split_df = split_df[split_df["split"] == cfg.nist_split][["mol_id"]]
    merged = labels.merge(split_df, on="mol_id", how="inner")
    log.info(
        f"NIST {cfg.nist_split} split: {len(merged)} molecules (full test set)"
    )

    ri_df = pd.read_csv(cfg.nist_ri_path, sep="\t")
    ri_df = ri_df[ri_df["split"] == cfg.nist_split].copy()
    merged = merged.merge(
        ri_df[["mol_id", "ri_StdNP", "ri_SemiStdNP", "ri_StdPolar"]],
        on="mol_id",
        how="left",
    )
    n_with_ri = (
        merged[["ri_StdNP", "ri_SemiStdNP", "ri_StdPolar"]]
        .notna()
        .any(axis=1)
        .sum()
    )
    log.info(f"  {n_with_ri}/{len(merged)} have at least one RI value")

    merged["nominal_mass"] = merged["mw"].round().astype("Int64")
    merged["inchikey14"] = merged["inchi_key"].astype(str).str[:14]

    gt_map: dict[str, np.ndarray] = {}
    highest_peak_map: dict[str, float] = {}
    with h5py.File(cfg.nist_spectra_path, "r") as hf:
        for mol_id in merged["mol_id"].astype(str):
            if mol_id in hf:
                grp = hf[mol_id]
                masses = np.array(grp["masses"])
                intensities = np.array(grp["intensities"])
                spec = _bin_spectrum(
                    masses,
                    intensities,
                    cfg.min_mz,
                    cfg.max_mz,
                    cfg.bin_width,
                    n_bins,
                )
                if spec.sum() > 0:
                    gt_map[mol_id] = spec
                    nonzero = masses[intensities > 0]
                    if len(nonzero):
                        highest_peak_map[mol_id] = float(nonzero.max())

    merged["gt_spectrum"] = merged["mol_id"].astype(str).map(gt_map)
    merged["highest_peak_mz"] = (
        merged["mol_id"].astype(str).map(highest_peak_map)
    )
    n_missing = merged["gt_spectrum"].isna().sum()
    if n_missing:
        log.warning(
            f"{n_missing} molecules have no valid GT spectrum — excluded"
        )
    merged = merged[merged["gt_spectrum"].notna()].reset_index(drop=True)
    log.info(f"Queries with valid GT spectrum: {len(merged)}")
    return merged


def load_true_predicted_spectra(
    queries_df: pd.DataFrame,
    hdf5_files: list[str],
    cache_dir: str | None = None,
    scan_chunk: int = 500_000,
) -> tuple[dict[str, np.ndarray], list[str]]:
    """Find each query's predicted spectrum in the PubChem HDF5.

    Uses a sequential scan (not random access) because the HDF5 uses column-
    oriented chunking — random row access is slower than scanning. Results
    cached to <cache_dir>/true_pred_spectra_cache.npz; subsequent runs skip the
    scan entirely.
    """
    needed_iks = queries_df["inchikey14"].unique().tolist()
    needed_set = set(needed_iks)
    pred_map: dict[str, np.ndarray] = {}

    cache_path = (
        Path(cache_dir or Path(hdf5_files[0]).parent)
        / "true_pred_spectra_cache.npz"
    )
    if cache_path.exists():
        log.info(f"Loading spectra cache: {cache_path}")
        cached = np.load(cache_path, allow_pickle=False)
        for ik in needed_iks:
            if ik in cached:
                pred_map[ik] = cached[ik]
        still_needed = needed_set - set(pred_map.keys())
        log.info(
            f"  Cache hit: {len(pred_map)}/{len(needed_iks)}  still needed: {len(still_needed)}"
        )
    else:
        still_needed = needed_set

    for h5_path in hdf5_files:
        if not still_needed:
            break
        log.info(
            f"Scanning {Path(h5_path).name} for {len(still_needed)} spectra "
            f"(sequential, column-chunked HDF5)..."
        )
        with h5py.File(h5_path, "r") as f:
            total = f["intensities"].shape[0]
            for start in tqdm(
                range(0, total, scan_chunk), desc="Scan for true spectra"
            ):
                if not still_needed:
                    break
                end = min(start + scan_chunk, total)
                raw_ik = f["inchikey14"][start:end]
                chunk_ik14 = np.array(
                    [
                        ik.decode() if isinstance(ik, bytes) else ik
                        for ik in raw_ik
                    ]
                )
                hits = still_needed & set(chunk_ik14.tolist())
                if not hits:
                    continue
                hit_mask = np.isin(chunk_ik14, list(hits))
                spectra = f["intensities"][start:end][hit_mask]
                for ik, spec in zip(chunk_ik14[hit_mask], spectra):
                    if ik in still_needed:
                        pred_map[ik] = spec.astype(np.float32)
                        still_needed.discard(ik)

    missing = list(still_needed)
    log.info(
        f"Found {len(pred_map)}/{len(needed_iks)}  missing: {len(missing)}"
    )
    if missing:
        log.warning(
            f"{len(missing)} true molecules not in PubChem HDF5 — skipped"
        )

    existing: dict[str, np.ndarray] = {}
    if cache_path.exists():
        existing = dict(np.load(cache_path, allow_pickle=False))
    new_entries = {ik: v for ik, v in pred_map.items() if ik not in existing}
    if new_entries:
        existing.update(new_entries)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, **existing)
        log.info(f"Cache saved (+{len(new_entries)}) → {cache_path}")

    return pred_map, missing


def _true_scores(
    true_pred_t: torch.Tensor,
    gt_spectra_t: torch.Tensor,
    mz_t: torch.Tensor,
    metrics: list[str],
    entropy_query_batch: int,
) -> dict[str, torch.Tensor]:
    """Compute per-query similarity between predicted and GT spectrum."""
    scores: dict[str, torch.Tensor] = {}
    Q = true_pred_t.shape[0]
    for m in metrics:
        if m == "cosine":
            scores[m] = batch_cosine_sim(true_pred_t, gt_spectra_t).diagonal()
        elif m == "weighted_cosine":
            scores[m] = batch_weighted_cosine_sim(
                true_pred_t, gt_spectra_t, mz_t
            ).diagonal()
        elif m in ("entropy", "composite_cosine"):
            fn = batch_entropy_sim if m == "entropy" else batch_composite_sim
            parts = []
            for qi in range(0, Q, entropy_query_batch):
                qe = min(qi + entropy_query_batch, Q)
                sim = fn(
                    true_pred_t[qi:qe],
                    gt_spectra_t[qi:qe],
                    mz_weights=mz_t,
                    query_batch_size=entropy_query_batch,
                )
                parts.append(sim.diagonal())
            scores[m] = torch.cat(parts)
    return scores


def _count_beats(
    sim_mat: torch.Tensor,  # [N_cands, N_queries]
    true_score: torch.Tensor,  # [N_queries]
    active: torch.Tensor,  # [N_cands, N_queries] bool — in-window decoys
) -> torch.Tensor:
    """Count decoys that beat the true molecule per query."""
    return ((sim_mat > true_score.unsqueeze(0)) & active).sum(dim=0)


def _safe_query_batch(
    N_chunk: int, n_bins: int, budget_gb: float = 1.5
) -> int:
    """Max query batch so [N_chunk × Qb × (float32 + bool)] fits in budget_gb.

    The nominal 5 bytes/element only counts is_decoy (bool) + sim (float32). At
    high chunk density (e.g. an MW±80Da window covering ~100% of the database
    against the full ~27.5k-query test set) there are also in_window/active
    (bool) and a near-full-size spec_gpu/ active_gpu after row_mask filtering
    barely shrinks anything — real peak usage is well above this nominal
    estimate. A 3x safety margin (effectively ~15 bytes/element) keeps the true
    peak under budget_gb; confirmed via a real OOM at N_chunk=500_000,
    Q=27_525, ~100% density with the un-marginned estimate.
    """
    budget_bytes = budget_gb * 1024**3
    safety_margin = 3
    qb = int(budget_bytes / (N_chunk * 5 * safety_margin))
    return max(1, min(qb, 512))


def _connected_mask(smiles: np.ndarray) -> np.ndarray:
    """True where a candidate SMILES has no '.' (i.e. is a single connected
    component).

    Disconnected-component candidates (hydrates, salts) are excluded from the
    retrieval candidate pool, matching NIST which has none of these among the
    query molecules.
    """
    return np.array(
        [
            "." not in (s.decode() if isinstance(s, bytes) else s)
            for s in smiles
        ]
    )


def _connected_mask_chunk(f: h5py.File, start: int, end: int) -> np.ndarray:
    """_connected_mask for HDF5 rows [start:end], or an all-True (no-op) mask
    if the file has no 'smiles' dataset — some prediction files (e.g.
    MassFormer's columnar format) only store inchikey14, not the SMILES string,
    so the '.' filter can't be applied to them."""
    if "smiles" not in f:
        return np.ones(end - start, dtype=bool)
    return _connected_mask(f["smiles"][start:end])


def evaluate_full_scan(
    queries_df: pd.DataFrame,
    true_pred_spectra: np.ndarray,
    hdf5_files: list[str],
    ri_type: str | None,
    mz_values: np.ndarray,
    metrics: list[str],
    chunk_size: int,
    entropy_cand_chunk: int,
    entropy_query_batch: int,
    device: torch.device,
) -> tuple[dict, torch.Tensor, torch.Tensor]:
    """Sequential scan for the 'all' level.

    ri_type=None → no RI filter, score all valid candidates (true global rank).

    Returns
    -------
    beats    : metric -> [Q] beat counts (add 1 for final rank)
    found    : [Q] bool — true molecule seen in DB
    cand_cnt : [Q] total decoy count
    """
    Q = len(queries_df)
    gt_t = torch.tensor(
        np.stack(queries_df["gt_spectrum"].values),
        device=device,
        dtype=torch.float32,
    )
    pred_t = torch.tensor(
        true_pred_spectra, device=device, dtype=torch.float32
    )
    mz_t = torch.tensor(mz_values, device=device, dtype=torch.float32)
    q_ik14 = queries_df["inchikey14"].values

    ts = _true_scores(pred_t, gt_t, mz_t, metrics, entropy_query_batch)
    beats = {
        m: torch.zeros(Q, dtype=torch.long, device=device) for m in metrics
    }
    cand_cnt = torch.zeros(Q, dtype=torch.long, device=device)
    found = torch.zeros(Q, dtype=torch.bool, device=device)

    # Integer-code each unique query InChIKey once so the per-chunk decoy
    # check is a cheap GPU int comparison instead of a CPU string broadcast
    # (was: torch.tensor(chunk_ik14[:, None] != q_ik_batch[None, :], ...) —
    # a [chunk_size x Q] numpy string-compare array built on CPU per chunk,
    # the actual bottleneck; plus a per-query Python for-loop with a nested
    # tqdm for the "found" check, adding ~Q iterations of overhead per
    # chunk on top).
    unique_ik = np.unique(q_ik14)
    q_ik_codes_gpu = torch.tensor(
        np.searchsorted(unique_ik, q_ik14), device=device, dtype=torch.int64
    )

    for h5_path in hdf5_files:
        with h5py.File(h5_path, "r") as f:
            total = f["intensities"].shape[0]
            ri_key = f"ri_{ri_type}" if ri_type else None
            for start in tqdm(range(0, total, chunk_size), desc="[all] scan"):
                end = min(start + chunk_size, total)
                valid_mask = f["valid"][start:end].astype(bool)
                if valid_mask.any():
                    valid_mask &= _connected_mask_chunk(f, start, end)
                if not valid_mask.any():
                    continue

                if ri_key:
                    ri_np = f[ri_key][start:end][valid_mask]
                    ri_finite = np.isfinite(ri_np)
                    if not ri_finite.any():
                        continue
                    spec_np = f["intensities"][start:end][valid_mask][
                        ri_finite
                    ]
                    raw_ik = f["inchikey14"][start:end][valid_mask][ri_finite]
                else:
                    spec_np = f["intensities"][start:end][valid_mask]
                    raw_ik = f["inchikey14"][start:end][valid_mask]

                chunk_ik14 = np.array(
                    [
                        ik.decode() if isinstance(ik, bytes) else ik
                        for ik in raw_ik
                    ]
                )
                N = len(chunk_ik14)

                not_found_idx = np.nonzero(~found.cpu().numpy())[0]
                if len(not_found_idx):
                    chunk_set = set(chunk_ik14.tolist())
                    hit = np.fromiter(
                        (q_ik14[qi] in chunk_set for qi in not_found_idx),
                        dtype=bool,
                        count=len(not_found_idx),
                    )
                    if hit.any():
                        found[not_found_idx[hit]] = True

                pos = np.searchsorted(unique_ik, chunk_ik14)
                pos = np.clip(pos, 0, len(unique_ik) - 1)
                chunk_ik_codes = np.where(
                    unique_ik[pos] == chunk_ik14, pos, -1
                ).astype(np.int64)
                chunk_ik_codes_gpu = torch.tensor(
                    chunk_ik_codes, device=device
                )

                spec_chunk = torch.tensor(
                    spec_np, device=device, dtype=torch.float32
                )

                # Process queries in small batches to bound [N × Qb] GPU allocation.
                # Peak: N × Qb × (float32 sim + bool decoy) = N × Qb × 5 bytes.
                qb_size = _safe_query_batch(N, gt_t.shape[1])
                for qi0 in range(0, Q, qb_size):
                    qi1 = min(qi0 + qb_size, Q)
                    gt_batch = gt_t[qi0:qi1]
                    ts_batch = {m: ts[m][qi0:qi1] for m in metrics}
                    q_codes_batch = q_ik_codes_gpu[qi0:qi1]

                    is_decoy = (
                        chunk_ik_codes_gpu[:, None] != q_codes_batch[None, :]
                    )  # [N, Qb]
                    cand_cnt[qi0:qi1] += is_decoy.sum(dim=0)

                    for m in metrics:
                        sim = _compute_sim(
                            m,
                            spec_chunk,
                            gt_batch,
                            mz_t,
                            entropy_cand_chunk,
                            entropy_query_batch,
                        )  # [N, Qb]
                        beats[m][qi0:qi1] += _count_beats(
                            sim, ts_batch[m], is_decoy
                        )
                        del sim
                    del is_decoy
                del chunk_ik_codes_gpu, spec_chunk
                torch.cuda.empty_cache()

    return beats, found, cand_cnt


def _build_needed_mask(
    sort_idx: np.ndarray,
    sorted_vals: np.ndarray,
    q_vals: np.ndarray,
    top_n: int | None,
    window_lo: np.ndarray | None,
    window_hi: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Binary-search window → union row mask.

    Two modes:
    - top_n mode  (top_n is not None): take top_n nearest neighbors per query.
    - range mode  (window_lo/window_hi given): take all within [lo, hi] per query.

    Returns (needed_mask, q_lb, q_rb, window_lo_out, window_hi_out).
    """
    N_db = len(sorted_vals)
    Q = len(q_vals)
    q_lb = np.empty(Q, dtype=np.int64)
    q_rb = np.empty(Q, dtype=np.int64)

    if top_n is not None:
        half = top_n // 2
        for qi in range(Q):
            pos = int(np.searchsorted(sorted_vals, q_vals[qi]))
            lb = max(0, pos - half)
            rb = min(N_db, lb + top_n)
            lb = max(0, rb - top_n)
            q_lb[qi] = lb
            q_rb[qi] = rb
        wlo = sorted_vals[q_lb]
        whi = sorted_vals[(q_rb - 1).clip(0)]
    else:
        for qi in range(Q):
            pos = int(np.searchsorted(sorted_vals, q_vals[qi]))
            q_lb[qi] = int(np.searchsorted(sorted_vals, window_lo[qi]))
            q_rb[qi] = int(
                np.searchsorted(sorted_vals, window_hi[qi], side="right")
            )
        wlo = window_lo
        whi = window_hi

    pos_mask = np.zeros(N_db, dtype=bool)
    for qi in range(Q):
        pos_mask[q_lb[qi] : q_rb[qi]] = True
    needed_mask = np.zeros(N_db, dtype=bool)
    needed_mask[np.sort(sort_idx[pos_mask])] = True
    return needed_mask, wlo, whi


def _scan_window(
    hdf5_files: list[str],
    needed_mask: np.ndarray,
    val_key: str,
    window_lo: np.ndarray,
    window_hi: np.ndarray,
    gt_t: torch.Tensor,
    pred_t: torch.Tensor,
    mz_t: torch.Tensor,
    q_ik14: np.ndarray,
    ts: dict,
    metrics: list[str],
    chunk_size: int,
    entropy_cand_chunk: int,
    entropy_query_batch: int,
    device: torch.device,
    desc: str,
) -> tuple[dict, torch.Tensor, torch.Tensor]:
    """Scan HDF5 rows in needed_mask, score against queries, return
    beats/found/cand_cnt."""
    Q = len(q_ik14)
    beats = {
        m: torch.zeros(Q, dtype=torch.long, device=device) for m in metrics
    }
    cand_cnt = torch.zeros(Q, dtype=torch.long, device=device)
    found = torch.zeros(Q, dtype=torch.bool, device=device)

    # Integer-code each unique query InChIKey once so the per-chunk decoy
    # check is a cheap GPU int comparison instead of a CPU string broadcast.
    # unique_ik is sorted (np.unique), so chunk-side codes are looked up
    # via vectorized searchsorted rather than a per-row dict/python loop.
    unique_ik = np.unique(q_ik14)
    q_ik_codes_gpu = torch.tensor(
        np.searchsorted(unique_ik, q_ik14), device=device, dtype=torch.int64
    )

    for h5_path in hdf5_files:
        with h5py.File(h5_path, "r") as f:
            total = f["intensities"].shape[0]
            for start in tqdm(range(0, total, chunk_size), desc=desc):
                end = min(start + chunk_size, total)
                if not needed_mask[start:end].any():
                    continue

                chunk_needed = needed_mask[start:end]

                # Always read the chunk contiguously and mask in memory —
                # never fancy-index the HDF5 with a scattered integer array
                # (f[dataset][non_contiguous_idx]). h5py has no way to turn
                # that into one bulk sequential read; each selected row (or
                # short run of rows) becomes its own I/O op, which is
                # catastrophically slower than a single contiguous slice
                # read + boolean mask even when the needed fraction is
                # small. Measured directly: a density~0.42-0.46 chunk (the
                # common case for e.g. a 100k-per-query RI window with
                # ~1200 queries) took ~59 min via fancy-indexing vs low
                # single-digit seconds via contiguous-read-then-mask.
                keep_local = chunk_needed & f["valid"][start:end].astype(bool)
                if not keep_local.any():
                    continue
                keep_local &= _connected_mask_chunk(f, start, end)
                if not keep_local.any():
                    continue
                keep_abs = np.where(keep_local)[0] + start
                val_np = f[val_key][start:end][keep_local]
                finite = np.isfinite(val_np)
                if not finite.any():
                    continue
                chunk_val_np = val_np[finite]
                spec_np = f["intensities"][start:end][keep_local][finite]
                raw_ik = f["inchikey14"][start:end][keep_local][finite]

                chunk_ik14 = np.array(
                    [
                        ik.decode() if isinstance(ik, bytes) else ik
                        for ik in raw_ik
                    ]
                )
                # NOTE: "found" is set inside the per-query-batch loop below,
                # from (~is_decoy & in_window) — i.e. the query's OWN row
                # must itself be within THAT query's own window, not merely
                # present anywhere in needed_mask's union across all
                # queries. A prior version set found from chunk-membership
                # alone (qik in chunk_set), the same union-mask-membership
                # bug fixed in evaluate_ri_mw_union (see commit 498b6f7's
                # intent) — this shared helper was never actually corrected,
                # so every caller (evaluate_ri_window, evaluate_mw_window,
                # evaluate_heavy_atom_window, and the since-removed MW/HA
                # -then-RI funnel functions) inherited the bug until now.

                # Window/decoy membership computed once per chunk on GPU
                # with integer-coded InChIKeys (was dense CPU numpy
                # broadcasting on RAW STRINGS — chunk_ik14[:, None] !=
                # q_ik_batch[None, :] is a [chunk_rows x Q] string-compare
                # array rebuilt from scratch every chunk, single-threaded —
                # the actual bottleneck on high-density chunks).
                pos = np.searchsorted(unique_ik, chunk_ik14)
                pos = np.clip(pos, 0, len(unique_ik) - 1)
                chunk_ik_codes = np.where(
                    unique_ik[pos] == chunk_ik14, pos, -1
                ).astype(np.int64)
                chunk_ik_codes_gpu = torch.tensor(
                    chunk_ik_codes, device=device
                )
                chunk_val_gpu = torch.tensor(
                    chunk_val_np, device=device, dtype=torch.float32
                )
                window_lo_gpu = torch.tensor(
                    window_lo, device=device, dtype=torch.float32
                )
                window_hi_gpu = torch.tensor(
                    window_hi, device=device, dtype=torch.float32
                )
                spec_gpu_full = torch.tensor(
                    spec_np, device=device, dtype=torch.float32
                )
                N = len(chunk_ik14)
                qb_size = _safe_query_batch(N, gt_t.shape[1])

                for qi0 in range(0, Q, qb_size):
                    qi1 = min(qi0 + qb_size, Q)
                    gt_batch = gt_t[qi0:qi1]
                    ts_batch = {m: ts[m][qi0:qi1] for m in metrics}
                    q_codes_batch = q_ik_codes_gpu[qi0:qi1]
                    is_decoy = (
                        chunk_ik_codes_gpu[:, None] != q_codes_batch[None, :]
                    )
                    in_window = (
                        chunk_val_gpu[:, None] >= window_lo_gpu[None, qi0:qi1]
                    ) & (
                        chunk_val_gpu[:, None] <= window_hi_gpu[None, qi0:qi1]
                    )
                    active = is_decoy & in_window
                    row_mask = active.any(dim=1)
                    cand_cnt[qi0:qi1] += active.sum(dim=0)

                    # True molecule found for query qi iff its own row is
                    # present in this chunk AND within qi's own window (see
                    # the note above this loop).
                    own_row_in_window = (~is_decoy) & in_window
                    found[qi0:qi1] |= own_row_in_window.any(dim=0)

                    if not row_mask.any():
                        del (
                            is_decoy,
                            in_window,
                            active,
                            row_mask,
                            own_row_in_window,
                        )
                        continue
                    spec_gpu = spec_gpu_full[row_mask]
                    active_gpu = active[row_mask]
                    for m in metrics:
                        sim = _compute_sim(
                            m,
                            spec_gpu,
                            gt_batch,
                            mz_t,
                            entropy_cand_chunk,
                            entropy_query_batch,
                        )
                        beats[m][qi0:qi1] += (
                            (sim > ts_batch[m].unsqueeze(0)) & active_gpu
                        ).sum(dim=0)
                        del sim
                    del (
                        is_decoy,
                        in_window,
                        active,
                        row_mask,
                        own_row_in_window,
                        spec_gpu,
                        active_gpu,
                    )
                del chunk_val_gpu, spec_gpu_full, chunk_ik_codes_gpu
                torch.cuda.empty_cache()

    return beats, found, cand_cnt


def evaluate_ri_window(
    queries_df: pd.DataFrame,
    true_pred_spectra: np.ndarray,
    hdf5_files: list[str],
    ri_type: str,
    top_n: int,
    mz_values: np.ndarray,
    metrics: list[str],
    chunk_size: int,
    entropy_cand_chunk: int,
    entropy_query_batch: int,
    device: torch.device,
) -> tuple[dict, torch.Tensor, torch.Tensor]:
    """RI window evaluation via binary search on pre-sorted HDF5 index."""
    Q = len(queries_df)
    gt_t = torch.tensor(
        np.stack(queries_df["gt_spectrum"].values),
        device=device,
        dtype=torch.float32,
    )
    pred_t = torch.tensor(
        true_pred_spectra, device=device, dtype=torch.float32
    )
    mz_t = torch.tensor(mz_values, device=device, dtype=torch.float32)
    q_ik14 = queries_df["inchikey14"].values
    q_ri_np = queries_df["ri"].values.astype(np.float32)

    with h5py.File(hdf5_files[0], "r") as f:
        sort_idx = f[f"sort_idx_{ri_type}"][:]
        sorted_ri_db = f[f"sorted_ri_{ri_type}"][:]

    needed_mask, window_lo, window_hi = _build_needed_mask(
        sort_idx, sorted_ri_db, q_ri_np, top_n, None, None
    )
    log.info(
        f"  [RI top{top_n}] Union window: {needed_mask.sum():,} unique rows"
    )

    ts = _true_scores(pred_t, gt_t, mz_t, metrics, entropy_query_batch)
    return _scan_window(
        hdf5_files,
        needed_mask,
        f"ri_{ri_type}",
        window_lo,
        window_hi,
        gt_t,
        pred_t,
        mz_t,
        q_ik14,
        ts,
        metrics,
        chunk_size,
        entropy_cand_chunk,
        entropy_query_batch,
        device,
        desc=f"[RI top{top_n}] scan",
    )


def evaluate_mw_window(
    queries_df: pd.DataFrame,
    true_pred_spectra: np.ndarray,
    hdf5_files: list[str],
    mw_da: int,
    mz_values: np.ndarray,
    metrics: list[str],
    chunk_size: int,
    entropy_cand_chunk: int,
    entropy_query_batch: int,
    device: torch.device,
) -> tuple[dict, torch.Tensor, torch.Tensor]:
    """MW ±mw_da window evaluation via binary search on sort_idx_mw /
    sorted_mw."""
    Q = len(queries_df)
    gt_t = torch.tensor(
        np.stack(queries_df["gt_spectrum"].values),
        device=device,
        dtype=torch.float32,
    )
    pred_t = torch.tensor(
        true_pred_spectra, device=device, dtype=torch.float32
    )
    mz_t = torch.tensor(mz_values, device=device, dtype=torch.float32)
    q_ik14 = queries_df["inchikey14"].values
    # Window center is the highest observed m/z peak in the query's own
    # ground-truth spectrum — a realistic proxy for the (unknown) molecular
    # weight, not the true MW itself.
    q_mw = queries_df["highest_peak_mz"].values.astype(np.float32)

    with h5py.File(hdf5_files[0], "r") as f:
        sort_idx = f["sort_idx_mw"][:]
        sorted_mw_db = f["sorted_mw"][:]

    window_lo = q_mw - mw_da
    window_hi = q_mw + mw_da
    needed_mask, wlo, whi = _build_needed_mask(
        sort_idx, sorted_mw_db, q_mw, None, window_lo, window_hi
    )
    log.info(
        f"  [MW ±{mw_da}Da] Union window: {needed_mask.sum():,} unique rows"
    )

    ts = _true_scores(pred_t, gt_t, mz_t, metrics, entropy_query_batch)
    return _scan_window(
        hdf5_files,
        needed_mask,
        "mw",
        wlo,
        whi,
        gt_t,
        pred_t,
        mz_t,
        q_ik14,
        ts,
        metrics,
        chunk_size,
        entropy_cand_chunk,
        entropy_query_batch,
        device,
        desc=f"[MW ±{mw_da}Da] scan",
    )


def evaluate_ri_mw_union(
    queries_df: pd.DataFrame,
    true_pred_spectra: np.ndarray,
    hdf5_files: list[str],
    ri_type: str,
    top_n: int,
    mw_da: int,
    mz_values: np.ndarray,
    metrics: list[str],
    chunk_size: int,
    entropy_cand_chunk: int,
    entropy_query_batch: int,
    device: torch.device,
) -> tuple[dict, torch.Tensor, torch.Tensor]:
    """RI window ∪ MW window evaluation."""
    Q = len(queries_df)
    gt_t = torch.tensor(
        np.stack(queries_df["gt_spectrum"].values),
        device=device,
        dtype=torch.float32,
    )
    pred_t = torch.tensor(
        true_pred_spectra, device=device, dtype=torch.float32
    )
    mz_t = torch.tensor(mz_values, device=device, dtype=torch.float32)
    q_ik14 = queries_df["inchikey14"].values
    q_ri_np = queries_df["ri"].values.astype(np.float32)
    # Window center is the highest observed m/z peak in the query's own
    # ground-truth spectrum — a realistic proxy for the (unknown) molecular
    # weight, not the true MW itself.
    q_mw = queries_df["highest_peak_mz"].values.astype(np.float32)

    with h5py.File(hdf5_files[0], "r") as f:
        ri_sort_idx = f[f"sort_idx_{ri_type}"][:]
        sorted_ri_db = f[f"sorted_ri_{ri_type}"][:]
        mw_sort_idx = f["sort_idx_mw"][:]
        sorted_mw_db = f["sorted_mw"][:]

    ri_mask, ri_wlo, ri_whi = _build_needed_mask(
        ri_sort_idx, sorted_ri_db, q_ri_np, top_n, None, None
    )
    mw_mask, mw_wlo, mw_whi = _build_needed_mask(
        mw_sort_idx, sorted_mw_db, q_mw, None, q_mw - mw_da, q_mw + mw_da
    )

    # Union: a row is needed if in either window
    needed_mask = ri_mask | mw_mask
    log.info(
        f"  [RI top{top_n} ∪ MW ±{mw_da}Da] RI={ri_mask.sum():,}  MW={mw_mask.sum():,}  "
        f"union={needed_mask.sum():,} unique rows"
    )

    # For per-query in_window check: a candidate is active for query q if in RI window OR MW window
    # We pass both window bounds and combine them inside _scan_window via a custom path.
    # Simplest: re-implement the scan loop here with OR logic.
    ts = _true_scores(pred_t, gt_t, mz_t, metrics, entropy_query_batch)
    beats = {
        m: torch.zeros(Q, dtype=torch.long, device=device) for m in metrics
    }
    cand_cnt = torch.zeros(Q, dtype=torch.long, device=device)
    found = torch.zeros(Q, dtype=torch.bool, device=device)

    # Integer-code each unique query InChIKey once so the per-chunk decoy
    # check is a cheap GPU int comparison instead of a CPU string broadcast
    # (see evaluate_ri_window / _scan_window for the same fix + rationale).
    unique_ik = np.unique(q_ik14)
    q_ik_codes_gpu = torch.tensor(
        np.searchsorted(unique_ik, q_ik14), device=device, dtype=torch.int64
    )
    ri_wlo_gpu = torch.tensor(ri_wlo, device=device, dtype=torch.float32)
    ri_whi_gpu = torch.tensor(ri_whi, device=device, dtype=torch.float32)
    mw_wlo_gpu = torch.tensor(mw_wlo, device=device, dtype=torch.float32)
    mw_whi_gpu = torch.tensor(mw_whi, device=device, dtype=torch.float32)

    for h5_path in hdf5_files:
        with h5py.File(h5_path, "r") as f:
            total = f["intensities"].shape[0]
            for start in tqdm(
                range(0, total, chunk_size), desc=f"[RI∪MW top{top_n}] scan"
            ):
                end = min(start + chunk_size, total)
                if not needed_mask[start:end].any():
                    continue
                chunk_needed = needed_mask[start:end]
                ri_needed = ri_mask[start:end]
                mw_needed = mw_mask[start:end]

                # Always read contiguously + mask in memory (see the same
                # fix/comment in _scan_window above) — never fancy-index
                # the HDF5 with a scattered integer array.
                keep_local = chunk_needed & f["valid"][start:end].astype(bool)
                if not keep_local.any():
                    continue
                keep_local &= _connected_mask_chunk(f, start, end)
                if not keep_local.any():
                    continue
                ri_np = f[f"ri_{ri_type}"][start:end][keep_local]
                mw_np = (
                    f["mw"][start:end][keep_local]
                    if "mw" in f
                    else np.full(keep_local.sum(), np.nan, dtype=np.float32)
                )
                ri_finite = np.isfinite(ri_np)
                mw_finite = np.isfinite(mw_np)
                any_finite = ri_finite | mw_finite
                if not any_finite.any():
                    continue
                spec_np = f["intensities"][start:end][keep_local][any_finite]
                raw_ik = f["inchikey14"][start:end][keep_local][any_finite]
                chunk_ri = ri_np[any_finite]
                chunk_mw = mw_np[any_finite]
                chunk_ri_needed = ri_needed[np.where(keep_local)[0]][
                    any_finite
                ]
                chunk_mw_needed = mw_needed[np.where(keep_local)[0]][
                    any_finite
                ]

                chunk_ik14 = np.array(
                    [
                        ik.decode() if isinstance(ik, bytes) else ik
                        for ik in raw_ik
                    ]
                )
                # NOTE: "found" is set inside the per-query-batch loop below,
                # from (~is_decoy & (in_ri | in_mw)) — i.e. the query's OWN
                # row must itself be within THAT query's RI window OR MW
                # window, not merely present somewhere in needed_mask's
                # UNION across all queries. A prior version set found from
                # chunk-membership alone (qik in chunk_set), the same
                # union-mask-membership bug fixed in _scan_window /
                # evaluate_ri_window (see commit 498b6f7) — this function
                # was only vectorized for GPU speed by that commit, not
                # corrected for the found-flag bug, so its "union beats
                # unfiltered retrieval" result was unverified against this
                # class of bug until now.

                # Windowing/decoy membership computed on GPU with integer-
                # coded InChIKeys — was dense CPU numpy broadcasting on raw
                # strings and per-cs-subblock concatenation, the bottleneck
                # on high-density chunks (see evaluate_ri_window's fix).
                pos = np.searchsorted(unique_ik, chunk_ik14)
                pos = np.clip(pos, 0, len(unique_ik) - 1)
                chunk_ik_codes = np.where(
                    unique_ik[pos] == chunk_ik14, pos, -1
                ).astype(np.int64)
                chunk_ik_codes_gpu = torch.tensor(
                    chunk_ik_codes, device=device
                )
                chunk_ri_gpu = torch.tensor(
                    chunk_ri, device=device, dtype=torch.float32
                )
                chunk_mw_gpu = torch.tensor(
                    chunk_mw, device=device, dtype=torch.float32
                )
                chunk_ri_needed_gpu = torch.tensor(
                    chunk_ri_needed, device=device, dtype=torch.bool
                )
                chunk_mw_needed_gpu = torch.tensor(
                    chunk_mw_needed, device=device, dtype=torch.bool
                )
                spec_gpu_full = torch.tensor(
                    spec_np, device=device, dtype=torch.float32
                )
                N = len(chunk_ik14)
                qb_size = _safe_query_batch(N, gt_t.shape[1])

                for qi0 in range(0, Q, qb_size):
                    qi1 = min(qi0 + qb_size, Q)
                    gt_batch = gt_t[qi0:qi1]
                    ts_batch = {m: ts[m][qi0:qi1] for m in metrics}
                    q_codes_batch = q_ik_codes_gpu[qi0:qi1]
                    is_decoy = (
                        chunk_ik_codes_gpu[:, None] != q_codes_batch[None, :]
                    )

                    in_ri = (
                        (chunk_ri_gpu[:, None] >= ri_wlo_gpu[None, qi0:qi1])
                        & (chunk_ri_gpu[:, None] <= ri_whi_gpu[None, qi0:qi1])
                        & chunk_ri_needed_gpu[:, None]
                    )
                    in_mw = (
                        (chunk_mw_gpu[:, None] >= mw_wlo_gpu[None, qi0:qi1])
                        & (chunk_mw_gpu[:, None] <= mw_whi_gpu[None, qi0:qi1])
                        & chunk_mw_needed_gpu[:, None]
                    )
                    active = is_decoy & (in_ri | in_mw)
                    row_mask = active.any(dim=1)
                    cand_cnt[qi0:qi1] += active.sum(dim=0)

                    # True molecule found for query qi iff its own row is
                    # present in this chunk AND within qi's own RI window OR
                    # MW window (see the note above this loop).
                    own_row_in_window = (~is_decoy) & (in_ri | in_mw)
                    found[qi0:qi1] |= own_row_in_window.any(dim=0)

                    if not row_mask.any():
                        del (
                            is_decoy,
                            in_ri,
                            in_mw,
                            active,
                            row_mask,
                            own_row_in_window,
                        )
                        continue
                    spec_gpu = spec_gpu_full[row_mask]
                    active_gpu = active[row_mask]
                    for m in metrics:
                        sim = _compute_sim(
                            m,
                            spec_gpu,
                            gt_batch,
                            mz_t,
                            entropy_cand_chunk,
                            entropy_query_batch,
                        )
                        beats[m][qi0:qi1] += (
                            (sim > ts_batch[m].unsqueeze(0)) & active_gpu
                        ).sum(dim=0)
                        del sim
                    del (
                        is_decoy,
                        in_ri,
                        in_mw,
                        active,
                        row_mask,
                        own_row_in_window,
                        spec_gpu,
                        active_gpu,
                    )
                del (
                    chunk_ik_codes_gpu,
                    chunk_ri_gpu,
                    chunk_mw_gpu,
                    chunk_ri_needed_gpu,
                    chunk_mw_needed_gpu,
                    spec_gpu_full,
                )
                torch.cuda.empty_cache()

    return beats, found, cand_cnt


def evaluate_heavy_atom_window(
    queries_df: pd.DataFrame,
    true_pred_spectra: np.ndarray,
    hdf5_files: list[str],
    heavy_atom_pred: np.ndarray,
    window: float,
    mz_values: np.ndarray,
    metrics: list[str],
    chunk_size: int,
    entropy_cand_chunk: int,
    entropy_query_batch: int,
    device: torch.device,
) -> tuple[dict, torch.Tensor, torch.Tensor]:
    """Heavy-atom-count ±window evaluation via binary search on
    sort_idx_heavy_atom / sorted_heavy_atom.

    Query side: heavy_atom_pred, the trained GradientBoostingRegressor's
    prediction from the query's own PREDICTED spectrum (heavy-atom count
    isn't observable from a spectrum directly, unlike MW's highest-peak-mz
    proxy, so a query's own true structure can't be used here even though
    it's technically known — mirrors how a real deployment on an unknown
    molecule would only ever have a predicted/observed spectrum).
    Candidate side: heavy_atom_true in the HDF5 — RDKit-derived from each
    PubChem row's own SMILES, exact (see add_true_heavy_atom_index_to_hdf5.py).
    """
    Q = len(queries_df)
    gt_t = torch.tensor(
        np.stack(queries_df["gt_spectrum"].values),
        device=device,
        dtype=torch.float32,
    )
    pred_t = torch.tensor(
        true_pred_spectra, device=device, dtype=torch.float32
    )
    mz_t = torch.tensor(mz_values, device=device, dtype=torch.float32)
    q_ik14 = queries_df["inchikey14"].values

    with h5py.File(hdf5_files[0], "r") as f:
        sort_idx = f["sort_idx_heavy_atom"][:]
        sorted_ha_db = f["sorted_heavy_atom"][:]

    window_lo = heavy_atom_pred - window
    window_hi = heavy_atom_pred + window
    needed_mask, wlo, whi = _build_needed_mask(
        sort_idx, sorted_ha_db, heavy_atom_pred, None, window_lo, window_hi
    )
    log.info(
        f"  [HeavyAtom ±{window:g}] Union window: {needed_mask.sum():,} unique rows"
    )

    ts = _true_scores(pred_t, gt_t, mz_t, metrics, entropy_query_batch)
    return _scan_window(
        hdf5_files,
        needed_mask,
        "heavy_atom_true",
        wlo,
        whi,
        gt_t,
        pred_t,
        mz_t,
        q_ik14,
        ts,
        metrics,
        chunk_size,
        entropy_cand_chunk,
        entropy_query_batch,
        device,
        desc=f"[HeavyAtom ±{window:g}] scan",
    )


def evaluate_ri_heavy_atom_union(
    queries_df: pd.DataFrame,
    true_pred_spectra: np.ndarray,
    hdf5_files: list[str],
    ri_type: str,
    top_n: int,
    heavy_atom_pred: np.ndarray,
    window: float,
    mz_values: np.ndarray,
    metrics: list[str],
    chunk_size: int,
    entropy_cand_chunk: int,
    entropy_query_batch: int,
    device: torch.device,
) -> tuple[dict, torch.Tensor, torch.Tensor]:
    """RI window ∪ heavy-atom window evaluation.

    Mirrors evaluate_ri_mw_union exactly, with heavy-atom count (candidate-
    side: heavy_atom_true; query-side: heavy_atom_pred) in place of MW. Written
    with the CORRECT per-query-window found-flag logic from the start (see the
    note in evaluate_ri_mw_union about that function's found-flag bug, fixed
    separately) — a query's own row is "found" iff it is the non-decoy row AND
    falls in its own RI window OR its own heavy-atom window, not merely present
    anywhere in the union's scanned chunks.
    """
    Q = len(queries_df)
    gt_t = torch.tensor(
        np.stack(queries_df["gt_spectrum"].values),
        device=device,
        dtype=torch.float32,
    )
    pred_t = torch.tensor(
        true_pred_spectra, device=device, dtype=torch.float32
    )
    mz_t = torch.tensor(mz_values, device=device, dtype=torch.float32)
    q_ik14 = queries_df["inchikey14"].values
    q_ri_np = queries_df["ri"].values.astype(np.float32)

    with h5py.File(hdf5_files[0], "r") as f:
        ri_sort_idx = f[f"sort_idx_{ri_type}"][:]
        sorted_ri_db = f[f"sorted_ri_{ri_type}"][:]
        ha_sort_idx = f["sort_idx_heavy_atom"][:]
        sorted_ha_db = f["sorted_heavy_atom"][:]

    ri_mask, ri_wlo, ri_whi = _build_needed_mask(
        ri_sort_idx, sorted_ri_db, q_ri_np, top_n, None, None
    )
    ha_mask, ha_wlo, ha_whi = _build_needed_mask(
        ha_sort_idx,
        sorted_ha_db,
        heavy_atom_pred,
        None,
        heavy_atom_pred - window,
        heavy_atom_pred + window,
    )

    needed_mask = ri_mask | ha_mask
    log.info(
        f"  [RI top{top_n} ∪ HeavyAtom ±{window:g}] RI={ri_mask.sum():,}  "
        f"HeavyAtom={ha_mask.sum():,}  union={needed_mask.sum():,} unique rows"
    )

    ts = _true_scores(pred_t, gt_t, mz_t, metrics, entropy_query_batch)
    beats = {
        m: torch.zeros(Q, dtype=torch.long, device=device) for m in metrics
    }
    cand_cnt = torch.zeros(Q, dtype=torch.long, device=device)
    found = torch.zeros(Q, dtype=torch.bool, device=device)

    unique_ik = np.unique(q_ik14)
    q_ik_codes_gpu = torch.tensor(
        np.searchsorted(unique_ik, q_ik14), device=device, dtype=torch.int64
    )
    ri_wlo_gpu = torch.tensor(ri_wlo, device=device, dtype=torch.float32)
    ri_whi_gpu = torch.tensor(ri_whi, device=device, dtype=torch.float32)
    ha_wlo_gpu = torch.tensor(ha_wlo, device=device, dtype=torch.float32)
    ha_whi_gpu = torch.tensor(ha_whi, device=device, dtype=torch.float32)

    for h5_path in hdf5_files:
        with h5py.File(h5_path, "r") as f:
            total = f["intensities"].shape[0]
            for start in tqdm(
                range(0, total, chunk_size),
                desc=f"[RI top{top_n} ∪ HeavyAtom ±{window:g}] scan",
            ):
                end = min(start + chunk_size, total)
                if not needed_mask[start:end].any():
                    continue
                chunk_needed = needed_mask[start:end]
                ri_needed = ri_mask[start:end]
                ha_needed = ha_mask[start:end]

                keep_local = chunk_needed & f["valid"][start:end].astype(bool)
                if not keep_local.any():
                    continue
                keep_local &= _connected_mask_chunk(f, start, end)
                if not keep_local.any():
                    continue
                ri_np = f[f"ri_{ri_type}"][start:end][keep_local]
                ha_np = (
                    f["heavy_atom_true"][start:end][keep_local]
                    if "heavy_atom_true" in f
                    else np.full(keep_local.sum(), np.nan, dtype=np.float32)
                )
                ri_finite = np.isfinite(ri_np)
                ha_finite = np.isfinite(ha_np)
                any_finite = ri_finite | ha_finite
                if not any_finite.any():
                    continue
                spec_np = f["intensities"][start:end][keep_local][any_finite]
                raw_ik = f["inchikey14"][start:end][keep_local][any_finite]
                chunk_ri = ri_np[any_finite]
                chunk_ha = ha_np[any_finite]
                chunk_ri_needed = ri_needed[np.where(keep_local)[0]][
                    any_finite
                ]
                chunk_ha_needed = ha_needed[np.where(keep_local)[0]][
                    any_finite
                ]

                chunk_ik14 = np.array(
                    [
                        ik.decode() if isinstance(ik, bytes) else ik
                        for ik in raw_ik
                    ]
                )

                pos = np.searchsorted(unique_ik, chunk_ik14)
                pos = np.clip(pos, 0, len(unique_ik) - 1)
                chunk_ik_codes = np.where(
                    unique_ik[pos] == chunk_ik14, pos, -1
                ).astype(np.int64)
                chunk_ik_codes_gpu = torch.tensor(
                    chunk_ik_codes, device=device
                )
                chunk_ri_gpu = torch.tensor(
                    chunk_ri, device=device, dtype=torch.float32
                )
                chunk_ha_gpu = torch.tensor(
                    chunk_ha, device=device, dtype=torch.float32
                )
                chunk_ri_needed_gpu = torch.tensor(
                    chunk_ri_needed, device=device, dtype=torch.bool
                )
                chunk_ha_needed_gpu = torch.tensor(
                    chunk_ha_needed, device=device, dtype=torch.bool
                )
                spec_gpu_full = torch.tensor(
                    spec_np, device=device, dtype=torch.float32
                )
                N = len(chunk_ik14)
                qb_size = _safe_query_batch(N, gt_t.shape[1])

                for qi0 in range(0, Q, qb_size):
                    qi1 = min(qi0 + qb_size, Q)
                    gt_batch = gt_t[qi0:qi1]
                    ts_batch = {m: ts[m][qi0:qi1] for m in metrics}
                    q_codes_batch = q_ik_codes_gpu[qi0:qi1]
                    is_decoy = (
                        chunk_ik_codes_gpu[:, None] != q_codes_batch[None, :]
                    )

                    in_ri = (
                        (chunk_ri_gpu[:, None] >= ri_wlo_gpu[None, qi0:qi1])
                        & (chunk_ri_gpu[:, None] <= ri_whi_gpu[None, qi0:qi1])
                        & chunk_ri_needed_gpu[:, None]
                    )
                    in_ha = (
                        (chunk_ha_gpu[:, None] >= ha_wlo_gpu[None, qi0:qi1])
                        & (chunk_ha_gpu[:, None] <= ha_whi_gpu[None, qi0:qi1])
                        & chunk_ha_needed_gpu[:, None]
                    )
                    active = is_decoy & (in_ri | in_ha)
                    row_mask = active.any(dim=1)
                    cand_cnt[qi0:qi1] += active.sum(dim=0)

                    # True molecule found for query qi iff its own row is
                    # present in this chunk AND within qi's own RI window OR
                    # heavy-atom window.
                    own_row_in_window = (~is_decoy) & (in_ri | in_ha)
                    found[qi0:qi1] |= own_row_in_window.any(dim=0)

                    if not row_mask.any():
                        del (
                            is_decoy,
                            in_ri,
                            in_ha,
                            active,
                            row_mask,
                            own_row_in_window,
                        )
                        continue
                    spec_gpu = spec_gpu_full[row_mask]
                    active_gpu = active[row_mask]
                    for m in metrics:
                        sim = _compute_sim(
                            m,
                            spec_gpu,
                            gt_batch,
                            mz_t,
                            entropy_cand_chunk,
                            entropy_query_batch,
                        )
                        beats[m][qi0:qi1] += (
                            (sim > ts_batch[m].unsqueeze(0)) & active_gpu
                        ).sum(dim=0)
                        del sim
                    del (
                        is_decoy,
                        in_ri,
                        in_ha,
                        active,
                        row_mask,
                        own_row_in_window,
                        spec_gpu,
                        active_gpu,
                    )
                del (
                    chunk_ik_codes_gpu,
                    chunk_ri_gpu,
                    chunk_ha_gpu,
                    chunk_ri_needed_gpu,
                    chunk_ha_needed_gpu,
                    spec_gpu_full,
                )
                torch.cuda.empty_cache()

    return beats, found, cand_cnt


def build_results(
    beats: dict,
    found: torch.Tensor,
    cand_cnt: torch.Tensor,
    metrics: list[str],
) -> dict:
    """Convert beat counts to inject / autofail rank lists."""
    not_found = ~found
    results: dict = {"inject": {}, "autofail": {}}

    for m in metrics:
        rank_inject = (beats[m] + 1).cpu().numpy().tolist()
        rank_af = (
            beats[m].clone() + 1
        )  # beats -> 1-indexed rank, same as inject
        rank_af[not_found] = cand_cnt[not_found] + 1  # worst possible rank
        results["inject"][m] = rank_inject
        results["autofail"][m] = rank_af.cpu().numpy().tolist()

    results["true_mol_found"] = found.cpu().numpy().tolist()
    return results


def save_per_query_ranks(
    results: dict,
    queries_df: pd.DataFrame,
    cand_cnt: torch.Tensor,
    metrics: list[str],
    out_path: Path,
) -> None:
    """Dump per-query rank details (inject + autofail rank per metric, plus
    whether the true molecule was found and how many decoys were scored)
    alongside the holistic summary, for closer inspection of individual queries
    rather than just aggregate MRR/Top-k numbers."""
    df = queries_df[["mol_id", "inchi_key", "inchikey14"]].copy()
    if "standardized_smiles" in queries_df.columns:
        df["smiles"] = queries_df["standardized_smiles"]
    df["true_mol_found"] = results["true_mol_found"]
    df["n_candidates"] = cand_cnt.cpu().numpy()
    for m in metrics:
        df[f"rank_inject_{m}"] = results["inject"][m]
        df[f"rank_autofail_{m}"] = results["autofail"][m]
    df.to_csv(out_path, sep="\t", index=False)
    log.info(f"  Per-query ranks → {out_path}")


def summarize_ranks(
    results: dict,
    metrics: list[str],
    k_values: list[int] = [1, 5, 10, 20, 50],
) -> dict:
    summary: dict = {}
    for scenario in ["inject", "autofail"]:
        summary[scenario] = {}
        for m in metrics:
            arr = np.array(results[scenario][m])
            entry = {
                "mrr": float(np.mean(1.0 / arr)),
                "median_rank": float(np.median(arr)),
                "mean_rank": float(np.mean(arr)),
            }
            for k in k_values:
                entry[f"top_{k}_accuracy"] = float((arr <= k).mean())
            summary[scenario][m] = entry
    return summary


@hydra.main(
    version_base=None,
    config_path="../../../examples/configs/pubchem_retrieval",
    config_name="default",
)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    t0 = time.time()

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_bins = int((cfg.max_mz - cfg.min_mz) / cfg.bin_width)
    mz_values = np.linspace(
        cfg.min_mz, cfg.max_mz, n_bins, endpoint=False
    ).astype(np.float32)
    hdf5_paths = [str(p) for p in cfg.hdf5_files]
    top_n_levels = list(cfg.top_n_levels)
    ri_types = list(cfg.ri_types)
    metrics = list(cfg.ranking_metrics)
    entropy_cand_chunk = cfg.get("entropy_cand_chunk", 10000)
    entropy_query_batch = cfg.get("entropy_query_batch", 32)
    scan_chunk = cfg.scan_chunk_size

    # Load NIST queries
    queries_base = load_nist_queries(cfg, n_bins)

    # Find true predicted spectra
    spectra_cache_dir = cfg.get("spectra_cache_dir", str(output_dir))
    pred_map, missing_ik14 = load_true_predicted_spectra(
        queries_base, hdf5_paths, cache_dir=spectra_cache_dir
    )

    skipped_log = Path(
        cfg.get("skipped_log", str(output_dir / "skipped_queries.tsv"))
    )
    skipped_df = queries_base[queries_base["inchikey14"].isin(missing_ik14)][
        ["mol_id", "inchi_key", "inchikey14", "standardized_smiles"]
    ].drop_duplicates()
    skipped_df.to_csv(skipped_log, sep="\t", index=False)
    log.info(f"Skipped {len(skipped_df)} molecules → {skipped_log}")

    queries_base = queries_base[
        ~queries_base["inchikey14"].isin(missing_ik14)
    ].reset_index(drop=True)
    log.info(f"Evaluating {len(queries_base)} queries")

    all_results: dict = {}

    # Global-only mode: no RI filter, score all queries against all candidates.
    # Triggered when ri_types is empty or top_n_levels contains only "all".
    global_only = not ri_types or all(
        str(l).lower() == "all" for l in top_n_levels
    )
    if global_only:
        log.info(
            "Global-only mode: scoring all queries against all PubChem candidates (no RI filter)"
        )
        true_pred_mat = np.stack(
            [pred_map[ik] for ik in queries_base["inchikey14"]]
        )
        beats, found, cand_cnt = evaluate_full_scan(
            queries_df=queries_base,
            true_pred_spectra=true_pred_mat,
            hdf5_files=hdf5_paths,
            ri_type=None,
            mz_values=mz_values,
            metrics=metrics,
            chunk_size=scan_chunk,
            entropy_cand_chunk=entropy_cand_chunk,
            entropy_query_batch=entropy_query_batch,
            device=device,
        )
        raw_results = build_results(beats, found, cand_cnt, metrics)
        all_results["all"] = summarize_ranks(raw_results, metrics)
        global_json = output_dir / "retrieval_global_results.json"
        with open(global_json, "w") as fh:
            json.dump(all_results, fh, indent=2)
        log.info(f"Global results → {global_json}")
        save_per_query_ranks(
            raw_results,
            queries_base,
            cand_cnt,
            metrics,
            output_dir / "retrieval_global_per_query.tsv",
        )
        for scenario in ["inject", "autofail"]:
            print(f"\n[global] scenario={scenario}")
            for m in metrics:
                s = all_results["all"][scenario][m]
                print(
                    f"  {m:>16} | MRR={s['mrr']:.4f}  Top-1={s['top_1_accuracy']:.4f}  Top-10={s['top_10_accuracy']:.4f}  MedRank={s['median_rank']:.0f}"
                )
        log.info(f"Total elapsed: {(time.time() - t0) / 60:.1f} min")
        return

    # MW-only, full test set, no RI restriction at all — directly comparable
    # to retrieval_global_results.json's "all" scan (same queries_base, same
    # ~27.6k-query N), unlike the per-RI-type MW-only pass below which is
    # artificially restricted to whichever RI type's non-null subset it's
    # nested under. Answers "does the MW heuristic alone beat full-PubChem
    # random rank, on the actual full test set" rather than on a small
    # RI-having subset.
    mw_da_global = cfg.get("mw_window_da", None)
    if mw_da_global is not None and cfg.get("mw_global", False):
        with h5py.File(hdf5_paths[0], "r") as f:
            has_mw_index_g = "sort_idx_mw" in f and "mw" in f
        if not has_mw_index_g:
            log.warning(
                "sort_idx_mw / mw not in HDF5 — skipping MW-global eval."
            )
        else:
            queries_mw_global = queries_base.dropna(
                subset=["highest_peak_mz"]
            ).reset_index(drop=True)
            log.info(
                f"MW-global mode: {len(queries_mw_global)}/{len(queries_base)} "
                f"queries have highest_peak_mz — scoring MW±{mw_da_global}Da "
                "candidates, no RI filter, full test set"
            )
            true_pred_mw_global = np.stack(
                [pred_map[ik] for ik in queries_mw_global["inchikey14"]]
            )
            beats, found, cand_cnt = evaluate_mw_window(
                queries_df=queries_mw_global,
                true_pred_spectra=true_pred_mw_global,
                hdf5_files=hdf5_paths,
                mw_da=mw_da_global,
                mz_values=mz_values,
                metrics=metrics,
                chunk_size=scan_chunk,
                entropy_cand_chunk=entropy_cand_chunk,
                entropy_query_batch=entropy_query_batch,
                device=device,
            )
            raw_mw_global = build_results(beats, found, cand_cnt, metrics)
            mw_global_json = (
                output_dir / f"retrieval_mw{mw_da_global}_global_results.json"
            )
            with open(mw_global_json, "w") as fh:
                json.dump(
                    {"all": summarize_ranks(raw_mw_global, metrics)},
                    fh,
                    indent=2,
                )
            log.info(f"MW-global results → {mw_global_json}")
            save_per_query_ranks(
                raw_mw_global,
                queries_mw_global,
                cand_cnt,
                metrics,
                output_dir / f"retrieval_per_query_mw{mw_da_global}_all.tsv",
            )
            for scenario in ["inject", "autofail"]:
                print(f"\n[MW-global ±{mw_da_global}Da] scenario={scenario}")
                for m in metrics:
                    s = summarize_ranks(raw_mw_global, metrics)[scenario][m]
                    print(
                        f"  {m:>16} | MRR={s['mrr']:.4f}  Top-1={s['top_1_accuracy']:.4f}  "
                        f"Top-10={s['top_10_accuracy']:.4f}  MedRank={s['median_rank']:.0f}"
                    )

    # Heavy-atom-only, full test set, no RI restriction — mirrors the
    # MW-global block above but keyed on the model-predicted heavy-atom
    # count instead of the ground-truth-derived MW proxy.
    heavy_atom_window_global = cfg.get("heavy_atom_window", None)
    if heavy_atom_window_global is not None and cfg.get(
        "heavy_atom_global", False
    ):
        with h5py.File(hdf5_paths[0], "r") as f:
            has_ha_index_g = (
                "sort_idx_heavy_atom" in f and "heavy_atom_true" in f
            )
        if not has_ha_index_g:
            log.warning(
                "sort_idx_heavy_atom / heavy_atom_true not in HDF5 — skipping "
                "heavy-atom-global eval. Run add_true_heavy_atom_index_to_hdf5.py first."
            )
        else:
            heavy_atom_model_g = joblib.load(cfg.heavy_atom_model_path)
            true_pred_ha_global = np.stack(
                [pred_map[ik] for ik in queries_base["inchikey14"]]
            )
            heavy_atom_pred_g = heavy_atom_model_g.predict(
                true_pred_ha_global
            ).astype(np.float32)
            log.info(
                f"HeavyAtom-global mode: {len(queries_base)} queries — scoring "
                f"HeavyAtom ±{heavy_atom_window_global:g} candidates, no RI filter, full test set"
            )
            beats, found, cand_cnt = evaluate_heavy_atom_window(
                queries_df=queries_base,
                true_pred_spectra=true_pred_ha_global,
                hdf5_files=hdf5_paths,
                heavy_atom_pred=heavy_atom_pred_g,
                window=heavy_atom_window_global,
                mz_values=mz_values,
                metrics=metrics,
                chunk_size=scan_chunk,
                entropy_cand_chunk=entropy_cand_chunk,
                entropy_query_batch=entropy_query_batch,
                device=device,
            )
            raw_ha_global = build_results(beats, found, cand_cnt, metrics)
            ha_tag_g = f"{heavy_atom_window_global:g}"
            ha_global_json = (
                output_dir
                / f"retrieval_heavy_atom{ha_tag_g}_global_results.json"
            )
            with open(ha_global_json, "w") as fh:
                json.dump(
                    {"all": summarize_ranks(raw_ha_global, metrics)},
                    fh,
                    indent=2,
                )
            log.info(f"HeavyAtom-global results → {ha_global_json}")
            save_per_query_ranks(
                raw_ha_global,
                queries_base,
                cand_cnt,
                metrics,
                output_dir
                / f"retrieval_per_query_heavy_atom{ha_tag_g}_all.tsv",
            )
            for scenario in ["inject", "autofail"]:
                print(
                    f"\n[HeavyAtom-global ±{heavy_atom_window_global:g}] scenario={scenario}"
                )
                for m in metrics:
                    s = summarize_ranks(raw_ha_global, metrics)[scenario][m]
                    print(
                        f"  {m:>16} | MRR={s['mrr']:.4f}  Top-1={s['top_1_accuracy']:.4f}  "
                        f"Top-10={s['top_10_accuracy']:.4f}  MedRank={s['median_rank']:.0f}"
                    )

    # Plain RI-window ladder (writes retrieval_ablation_{ri_type}.json and
    # the combined all_ri_types.json). Skippable via skip_ri_ladder=true
    # when only the MW/heavy-atom funnel/union tracks further below are
    # wanted and the ladder itself was already computed in a prior run —
    # each block below rebuilds its own queries_ri/true_pred_mat
    # independently, so skipping this loop doesn't affect anything after it.
    if not cfg.get("skip_ri_ladder", False):
        for ri_type in ri_types:
            ri_col = f"ri_{ri_type}"
            log.info(f"\n{'=' * 60}\nRI type: {ri_type}\n{'=' * 60}")

            queries_ri = queries_base.copy()
            queries_ri["ri"] = queries_ri[ri_col]
            queries_ri = queries_ri.dropna(subset=["ri"]).reset_index(
                drop=True
            )
            if queries_ri.empty:
                log.warning(f"No queries with {ri_col} — skipping")
                continue
            log.info(f"  {len(queries_ri)} queries have {ri_col}")

            true_pred_mat = np.stack(
                [pred_map[ik] for ik in queries_ri["inchikey14"]]
            )

            ri_results: dict = {}

            for lvl in tqdm(
                top_n_levels, desc="Evaluating levels", leave=False
            ):
                lvl_str = str(lvl)
                log.info(f"  Level: {lvl_str}")

                if lvl_str.lower() == "all":
                    # True global rank on the RI-matched query subset — the
                    # full PubChem candidate pool, NOT filtered to candidates
                    # that happen to have this RI type (ri_type=None). This is
                    # the "ceiling" comparison point for the RI-window ladder:
                    # same queries as top1000/100k/1M/10M, but with no RI
                    # filter on the candidate side at all.
                    beats, found, cand_cnt = evaluate_full_scan(
                        queries_df=queries_ri,
                        true_pred_spectra=true_pred_mat,
                        hdf5_files=hdf5_paths,
                        ri_type=None,
                        mz_values=mz_values,
                        metrics=metrics,
                        chunk_size=scan_chunk,
                        entropy_cand_chunk=entropy_cand_chunk,
                        entropy_query_batch=entropy_query_batch,
                        device=device,
                    )
                else:
                    beats, found, cand_cnt = evaluate_ri_window(
                        queries_df=queries_ri,
                        true_pred_spectra=true_pred_mat,
                        hdf5_files=hdf5_paths,
                        ri_type=ri_type,
                        top_n=int(lvl),
                        mz_values=mz_values,
                        metrics=metrics,
                        chunk_size=scan_chunk,
                        entropy_cand_chunk=entropy_cand_chunk,
                        entropy_query_batch=entropy_query_batch,
                        device=device,
                    )

                raw_results = build_results(beats, found, cand_cnt, metrics)
                ri_results[lvl_str] = summarize_ranks(raw_results, metrics)
                save_per_query_ranks(
                    raw_results,
                    queries_ri,
                    cand_cnt,
                    metrics,
                    output_dir
                    / f"retrieval_per_query_{ri_type}_{lvl_str}.tsv",
                )

            all_results[ri_type] = ri_results

            ri_json = output_dir / f"retrieval_ablation_{ri_type}.json"
            with open(ri_json, "w") as fh:
                json.dump(ri_results, fh, indent=2)
            log.info(f"  Saved → {ri_json}")

            for lvl_str, lvl_data in ri_results.items():
                for scenario in ["inject", "autofail"]:
                    print(
                        f"\n[{ri_type}] top_n={lvl_str}  scenario={scenario}"
                    )
                    for m in metrics:
                        s = lvl_data[scenario][m]
                        print(
                            f"  {m:>16} | MRR={s['mrr']:.4f}  "
                            f"Top-1={s['top_1_accuracy']:.4f}  "
                            f"Top-10={s['top_10_accuracy']:.4f}  "
                            f"MedRank={s['median_rank']:.0f}"
                        )

        combined_json = output_dir / "retrieval_ablation_all_ri_types.json"
        with open(combined_json, "w") as fh:
            json.dump(all_results, fh, indent=2)
        log.info(f"\nAll results → {combined_json}")
    else:
        log.info(
            "skip_ri_ladder=true — skipping plain RI-window ladder (using existing retrieval_ablation_*.json)"
        )

    # MW-only and RI∪MW union tracks (requires sort_idx_mw/mw in HDF5)
    mw_da = cfg.get("mw_window_da", None)
    if mw_da is not None:
        with h5py.File(hdf5_paths[0], "r") as f:
            has_mw_index = "sort_idx_mw" in f and "mw" in f
        if not has_mw_index:
            log.warning(
                "sort_idx_mw / mw not in HDF5 — skipping MW eval. Run add_sort_index_to_hdf5.py --parquet first."
            )
        else:
            mw_results: dict = {}
            union_results: dict = {}

            for ri_type in ri_types:
                ri_col = f"ri_{ri_type}"
                queries_ri = queries_base.copy()
                queries_ri["ri"] = queries_ri[ri_col]
                queries_ri = queries_ri.dropna(
                    subset=["ri", "highest_peak_mz"]
                ).reset_index(drop=True)
                if queries_ri.empty:
                    log.warning(
                        f"No queries with {ri_col} + highest_peak_mz — skipping MW for {ri_type}"
                    )
                    continue

                true_pred_mat = np.stack(
                    [pred_map[ik] for ik in queries_ri["inchikey14"]]
                )

                log.info(
                    f"\n{'=' * 60}\n[MW ±{mw_da} Da]  RI type: {ri_type}\n{'=' * 60}"
                )
                beats, found, cand_cnt = evaluate_mw_window(
                    queries_df=queries_ri,
                    true_pred_spectra=true_pred_mat,
                    hdf5_files=hdf5_paths,
                    mw_da=mw_da,
                    mz_values=mz_values,
                    metrics=metrics,
                    chunk_size=scan_chunk,
                    entropy_cand_chunk=entropy_cand_chunk,
                    entropy_query_batch=entropy_query_batch,
                    device=device,
                )
                raw_mw_results = build_results(beats, found, cand_cnt, metrics)
                mw_results[ri_type] = summarize_ranks(raw_mw_results, metrics)
                save_per_query_ranks(
                    raw_mw_results,
                    queries_ri,
                    cand_cnt,
                    metrics,
                    output_dir
                    / f"retrieval_per_query_mw{mw_da}_{ri_type}.tsv",
                )

                log.info(
                    f"\n{'=' * 60}\n[RI∪MW top_n ±{mw_da} Da]  RI type: {ri_type}\n{'=' * 60}"
                )
                union_ri: dict = {}
                for lvl in [
                    l for l in top_n_levels if str(l).lower() != "all"
                ]:
                    beats, found, cand_cnt = evaluate_ri_mw_union(
                        queries_df=queries_ri,
                        true_pred_spectra=true_pred_mat,
                        hdf5_files=hdf5_paths,
                        ri_type=ri_type,
                        top_n=int(lvl),
                        mw_da=mw_da,
                        mz_values=mz_values,
                        metrics=metrics,
                        chunk_size=scan_chunk,
                        entropy_cand_chunk=entropy_cand_chunk,
                        entropy_query_batch=entropy_query_batch,
                        device=device,
                    )
                    raw_union_results = build_results(
                        beats, found, cand_cnt, metrics
                    )
                    union_ri[str(lvl)] = summarize_ranks(
                        raw_union_results, metrics
                    )
                    save_per_query_ranks(
                        raw_union_results,
                        queries_ri,
                        cand_cnt,
                        metrics,
                        output_dir
                        / f"retrieval_per_query_union_mw{mw_da}_{ri_type}_N{lvl}.tsv",
                    )
                union_results[ri_type] = union_ri

            mw_json = output_dir / f"retrieval_mw{mw_da}_results.json"
            with open(mw_json, "w") as fh:
                json.dump(mw_results, fh, indent=2)
            log.info(f"MW-only results → {mw_json}")

            union_json = output_dir / f"retrieval_union_mw{mw_da}_results.json"
            with open(union_json, "w") as fh:
                json.dump(union_results, fh, indent=2)
            log.info(f"RI∪MW union results → {union_json}")

    # Heavy-atom-only, heavy-atom→RI funnel, and RI∪heavy-atom union tracks
    # (requires sort_idx_heavy_atom/heavy_atom_true in HDF5). Mirrors the MW
    # block above exactly, with heavy-atom count in place of MW.
    heavy_atom_window = cfg.get("heavy_atom_window", None)
    if heavy_atom_window is not None:
        with h5py.File(hdf5_paths[0], "r") as f:
            has_ha_index = (
                "sort_idx_heavy_atom" in f and "heavy_atom_true" in f
            )
        if not has_ha_index:
            log.warning(
                "sort_idx_heavy_atom / heavy_atom_true not in HDF5 — skipping "
                "heavy-atom eval. Run add_true_heavy_atom_index_to_hdf5.py first."
            )
        else:
            heavy_atom_model = joblib.load(cfg.heavy_atom_model_path)
            ha_tag = f"{heavy_atom_window:g}"

            heavy_atom_results: dict = {}
            heavy_atom_union_results: dict = {}
            run_heavy_atom_union = cfg.get("heavy_atom_ri_union", False)

            for ri_type in ri_types:
                ri_col = f"ri_{ri_type}"
                queries_ri = queries_base.copy()
                queries_ri["ri"] = queries_ri[ri_col]
                queries_ri = queries_ri.dropna(subset=["ri"]).reset_index(
                    drop=True
                )
                if queries_ri.empty:
                    log.warning(
                        f"No queries with {ri_col} — skipping HeavyAtom for {ri_type}"
                    )
                    continue

                true_pred_mat = np.stack(
                    [pred_map[ik] for ik in queries_ri["inchikey14"]]
                )
                heavy_atom_pred = heavy_atom_model.predict(
                    true_pred_mat
                ).astype(np.float32)

                log.info(
                    f"\n{'=' * 60}\n[HeavyAtom ±{heavy_atom_window:g}]  RI type: {ri_type}\n{'=' * 60}"
                )
                beats, found, cand_cnt = evaluate_heavy_atom_window(
                    queries_df=queries_ri,
                    true_pred_spectra=true_pred_mat,
                    hdf5_files=hdf5_paths,
                    heavy_atom_pred=heavy_atom_pred,
                    window=heavy_atom_window,
                    mz_values=mz_values,
                    metrics=metrics,
                    chunk_size=scan_chunk,
                    entropy_cand_chunk=entropy_cand_chunk,
                    entropy_query_batch=entropy_query_batch,
                    device=device,
                )
                raw_ha_results = build_results(beats, found, cand_cnt, metrics)
                heavy_atom_results[ri_type] = summarize_ranks(
                    raw_ha_results, metrics
                )
                save_per_query_ranks(
                    raw_ha_results,
                    queries_ri,
                    cand_cnt,
                    metrics,
                    output_dir
                    / f"retrieval_per_query_heavy_atom{ha_tag}_{ri_type}.tsv",
                )

                if run_heavy_atom_union:
                    log.info(
                        f"\n{'=' * 60}\n[RI∪HeavyAtom top_n ±{heavy_atom_window:g}]  RI type: {ri_type}\n{'=' * 60}"
                    )
                    union_ri_ha: dict = {}
                    for lvl in [
                        l for l in top_n_levels if str(l).lower() != "all"
                    ]:
                        beats, found, cand_cnt = evaluate_ri_heavy_atom_union(
                            queries_df=queries_ri,
                            true_pred_spectra=true_pred_mat,
                            hdf5_files=hdf5_paths,
                            ri_type=ri_type,
                            top_n=int(lvl),
                            heavy_atom_pred=heavy_atom_pred,
                            window=heavy_atom_window,
                            mz_values=mz_values,
                            metrics=metrics,
                            chunk_size=scan_chunk,
                            entropy_cand_chunk=entropy_cand_chunk,
                            entropy_query_batch=entropy_query_batch,
                            device=device,
                        )
                        raw_ha_union_results = build_results(
                            beats, found, cand_cnt, metrics
                        )
                        union_ri_ha[str(lvl)] = summarize_ranks(
                            raw_ha_union_results, metrics
                        )
                        save_per_query_ranks(
                            raw_ha_union_results,
                            queries_ri,
                            cand_cnt,
                            metrics,
                            output_dir
                            / f"retrieval_per_query_union_heavy_atom{heavy_atom_window:g}_{ri_type}_N{lvl}.tsv",
                        )
                    heavy_atom_union_results[ri_type] = union_ri_ha

            ha_json = output_dir / f"retrieval_heavy_atom{ha_tag}_results.json"
            with open(ha_json, "w") as fh:
                json.dump(heavy_atom_results, fh, indent=2)
            log.info(f"HeavyAtom-only results → {ha_json}")

            if run_heavy_atom_union:
                ha_union_json = (
                    output_dir
                    / f"retrieval_union_heavy_atom{ha_tag}_results.json"
                )
                with open(ha_union_json, "w") as fh:
                    json.dump(heavy_atom_union_results, fh, indent=2)
                log.info(f"RI∪HeavyAtom union results → {ha_union_json}")

    log.info(f"Total elapsed: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
