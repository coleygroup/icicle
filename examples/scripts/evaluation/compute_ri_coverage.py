#!/usr/bin/env python
"""Compute RI-window coverage (no spectra, no GPU).

For each NIST test query and each top_n candidate window size, reports the
fraction of queries whose true molecule is present in the RI-nearest top_n
PubChem candidates.  This is the upper-bound recall for any retrieval method.

Usage
-----
uv run examples/scripts/evaluation/compute_ri_coverage.py
# Override: top_n_levels=[1000,10000,100000] ri_types=[StdNP]
"""

import json
import logging
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

log = logging.getLogger(__name__)


def load_queries(cfg: DictConfig) -> pd.DataFrame:
    """Load NIST test queries with RI values."""
    meta = pd.read_csv(cfg.nist_labels_path, sep="\t")
    meta["inchikey14"] = meta["inchi_key"].str[:14]
    ri = pd.read_csv(cfg.nist_ri_path, sep="\t")
    ri["inchikey14"] = ri["inchi_key"].str[:14]
    ri = ri[ri["split"] == cfg.nist_split]
    queries = meta.merge(
        ri[["inchikey14"] + [c for c in ri.columns if c.startswith("ri_")]],
        on="inchikey14",
        how="inner",
    ).drop_duplicates("inchikey14")
    log.info(f"Loaded {len(queries)} test queries")
    return queries


def required_ri_error_for_top_n(
    query_ik14: np.ndarray,
    query_ri: np.ndarray,
    pubchem_ik14: np.ndarray,
    pubchem_ri: np.ndarray,
    top_n: int,
) -> dict:
    """For a given top_n window, compute how accurate the RI predictor must be.

    The window takes the top_n PubChem molecules nearest to the predicted RI.
    For each query, the window spans some RI range [ri_low, ri_high]. The
    predictor can be off by at most (ri_high - ri_low) / 2 before the true
    molecule falls outside. Averaging this across queries gives the required
    predictor accuracy (in RI units) to achieve 100% coverage at this top_n.

    Returns dict with mean, median, p90, p95, p99 of per-query required accuracy.
    """
    sorted_pos = np.argsort(pubchem_ri, kind="stable")
    sorted_ri = pubchem_ri[sorted_pos]
    sorted_ik14 = pubchem_ik14[sorted_pos]

    ik14_to_sorted_pos: dict[str, list[int]] = {}
    for i, ik in enumerate(sorted_ik14):
        ik14_to_sorted_pos.setdefault(ik, []).append(i)

    valid_mask = np.isfinite(query_ri)
    q_ri_valid = query_ri[valid_mask]
    q_ik14_valid = query_ik14[valid_mask]

    half = top_n // 2
    N = len(sorted_ri)
    positions = np.searchsorted(sorted_ri, q_ri_valid)
    lbs = np.clip(positions - half, 0, N - top_n)
    lbs = np.where(positions - half < 0, 0, lbs)
    rbs = np.minimum(lbs + top_n, N)
    lbs = np.maximum(rbs - top_n, 0)

    window_half_spans = []
    n_missing = 0
    for qi in range(len(q_ri_valid)):
        ik = q_ik14_valid[qi]
        if ik not in ik14_to_sorted_pos:
            n_missing += 1
            continue
        lb, rb = int(lbs[qi]), int(rbs[qi])
        ri_low = float(sorted_ri[lb])
        ri_high = float(sorted_ri[rb - 1])
        window_half_spans.append((ri_high - ri_low) / 2.0)

    if not window_half_spans:
        return {}
    arr = np.array(window_half_spans)
    return {
        "n_queries": len(arr),
        "n_missing_from_pubchem": n_missing,
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }


def coverage_for_level(
    query_ik14: np.ndarray,
    query_ri: np.ndarray,
    pubchem_ik14: np.ndarray,
    pubchem_ri: np.ndarray,
    top_n: int,
) -> tuple[float, int, int]:
    """Return (coverage, n_found, n_with_ri) for one top_n window.

    Vectorized: binary-search all query positions at once, build a boolean
    membership mask, then check each query's window with a single mask slice.
    """
    sorted_pos = np.argsort(pubchem_ri, kind="stable")
    sorted_ri = pubchem_ri[sorted_pos]
    sorted_ik14 = pubchem_ik14[sorted_pos]

    # Build reverse map: original ik14 position → sorted position (for membership).
    # Use a dict for O(1) lookup per query window check.
    ik14_to_sorted_pos: dict[str, list[int]] = {}
    for i, ik in enumerate(sorted_ik14):
        ik14_to_sorted_pos.setdefault(ik, []).append(i)

    valid_mask = np.isfinite(query_ri)
    n_with_ri = int(valid_mask.sum())
    if n_with_ri == 0:
        return float("nan"), 0, 0

    q_ri_valid = query_ri[valid_mask]
    q_ik14_valid = query_ik14[valid_mask]

    half = top_n // 2
    N = len(sorted_ri)

    # Binary-search all query positions at once.
    positions = np.searchsorted(sorted_ri, q_ri_valid)
    lbs = np.clip(positions - half, 0, N - top_n)
    lbs = np.where(positions - half < 0, 0, lbs)
    rbs = np.minimum(lbs + top_n, N)
    lbs = np.maximum(rbs - top_n, 0)

    found = 0
    for qi in range(n_with_ri):
        lb, rb = int(lbs[qi]), int(rbs[qi])
        ik = q_ik14_valid[qi]
        for sp in ik14_to_sorted_pos.get(ik, []):
            if lb <= sp < rb:
                found += 1
                break

    coverage = found / n_with_ri
    return coverage, found, n_with_ri


@hydra.main(
    config_path="../../../examples/configs/pubchem_retrieval",
    config_name="default",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ri_types: list[str] = list(cfg.ri_types)
    top_n_levels: list = [
        l for l in cfg.top_n_levels if str(l).lower() != "all"
    ]
    top_n_levels = sorted([int(l) for l in top_n_levels])

    log.info(f"RI types: {ri_types}")
    log.info(f"top_n levels: {top_n_levels}")

    queries = load_queries(cfg)

    log.info("Loading PubChem parquet RI columns (this may take ~1 min)...")
    pubchem = pd.read_parquet(
        cfg.pubchem_ri_parquet,
        columns=["ik_prefix"] + [f"ri_{t}" for t in ri_types],
    )
    pubchem_ik14 = pubchem["ik_prefix"].values

    results: dict = {}

    for ri_type in ri_types:
        ri_col = f"ri_{ri_type}"
        if ri_col not in queries.columns:
            log.warning(f"No {ri_col} in queries — skipping")
            continue
        if ri_col not in pubchem.columns:
            log.warning(f"No {ri_col} in PubChem — skipping")
            continue

        query_ik14 = queries["inchikey14"].values
        query_ri = queries[ri_col].values.astype(np.float32)
        pubchem_ri = pubchem[ri_col].values.astype(np.float32)

        valid_pc = np.isfinite(pubchem_ri)
        log.info(
            f"[{ri_type}] PubChem rows with finite RI: {valid_pc.sum():,} / {len(pubchem_ri):,}"
        )

        results[ri_type] = {}
        for top_n in top_n_levels:
            log.info(f"  Computing coverage for top_n={top_n}...")
            cov, n_found, n_with_ri = coverage_for_level(
                query_ik14,
                query_ri,
                pubchem_ik14[valid_pc],
                pubchem_ri[valid_pc],
                top_n,
            )
            acc = required_ri_error_for_top_n(
                query_ik14,
                query_ri,
                pubchem_ik14[valid_pc],
                pubchem_ri[valid_pc],
                top_n,
            )
            results[ri_type][str(top_n)] = {
                "coverage": cov,
                "n_found": n_found,
                "n_with_ri": n_with_ri,
                "n_total_queries": len(queries),
                "required_ri_error": acc,
            }
            log.info(
                f"    coverage={cov:.4f}  ({n_found}/{n_with_ri})  required_error mean={acc.get('mean', float('nan')):.1f} p95={acc.get('p95', float('nan')):.1f} RI units"
            )

    out_path = output_dir / "ri_coverage.json"
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    log.info(f"\nSaved → {out_path}")

    print("\n=== RI Window Coverage ===")
    for ri_type, levels in results.items():
        print(f"\n{ri_type}")
        print(
            f"  {'top_n':>12}  {'coverage':>10}  {'n_found':>8}  {'req_error_mean':>16}  {'req_error_p95':>14}"
        )
        for lvl, v in levels.items():
            acc = v.get("required_ri_error", {})
            print(
                f"  {int(lvl):>12,}  {v['coverage']:>10.4f}  {v['n_found']:>8,}"
                f"  {acc.get('mean', float('nan')):>14.1f} RI  {acc.get('p95', float('nan')):>12.1f} RI"
            )


if __name__ == "__main__":
    main()
