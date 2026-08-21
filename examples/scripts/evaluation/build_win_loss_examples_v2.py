#!/usr/bin/env python
"""ICICLE vs. NEIMS win/loss examples across 6 outcome categories.

Categories (a-f), one example per regime where available:
  a/b/c  ICICLE ranks 1st (regardless of NEIMS's rank)
  d      ICICLE not 1st, but ranks better than NEIMS
  e      ICICLE not 1st, and ranks much worse than NEIMS
  f      both ICICLE and NEIMS rank badly

Regimes: random-formula, scaffold-formula, random-global, scaffold-global.

For formula regimes, "top decoy" (the best-scoring wrong candidate) is read
directly from the existing retrieval_with_formula_results.csv per-candidate
rows. For global regimes, per-query results only ever stored *rank*, never
which PubChem candidate was top-1 or its similarity -- so for the specific
molecules picked here (not the full ~33k test set) we re-scan the relevant
~93M-row PubChem prediction HDF5 once per query to find the actual top
candidate and its cosine similarity.

Plots reuse plot_three_spectra with smiles=None (no molecule structure
drawn over the spectrum, so peaks near the inset are visible).

Usage
-----
uv run examples/scripts/evaluation/build_win_loss_examples_v2.py
"""

import logging
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from icicle.analysis.metrics import cosine_similarity
from icicle.utils.visualization.eval_plots import model_color
from icicle.utils.visualization.mass_spectra import plot_three_spectra
from icicle.utils.visualization.style import save_fig, set_style

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

set_style("manuscript")

REPO_ROOT = Path("/home/magled/icicle-dev")
EVAL = REPO_ROOT / "results" / "eval"
OUTPUT_DIR = (
    REPO_ROOT / "examples" / "notebooks" / "retrieval_win_loss_examples_v2"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["ICICLE", "NEIMS"]
MODEL_COLORS = {
    m: model_color(m, fallback_index=i) for i, m in enumerate(MODELS)
}

FORMULA_DIRS = {
    "random": {
        "ICICLE": EVAL / "final_entropy_random_s1_retr",
        "NEIMS": EVAL / "neims_random_s1",
    },
    "scaffold": {
        "ICICLE": EVAL / "final_entropy_scaffold_s1_retr",
        "NEIMS": EVAL / "neims_scaffold_s1",
    },
}
SIM_DIRS = {
    "random": EVAL / "final_entropy_random_s1_sim",
    "scaffold": EVAL / "final_entropy_scaffold_s1_sim",
}
NEIMS_PRED_FILES = {
    "random": REPO_ROOT
    / "baselines"
    / "neims"
    / "results"
    / "predictions"
    / "neims_random_noqcxms2_s1_test.hdf5",
    "scaffold": REPO_ROOT
    / "baselines"
    / "neims"
    / "results"
    / "predictions"
    / "neims_scaffold_s1_test.hdf5",
}
GLOBAL_RETRIEVAL_DIRS = {
    "random": {
        "ICICLE": REPO_ROOT
        / "results"
        / "pubchem_retrieval_eval_icicle_rerun_260710",
        "NEIMS": REPO_ROOT / "results" / "pubchem_retrieval_eval_neims",
    },
    "scaffold": {
        "ICICLE": REPO_ROOT
        / "results"
        / "pubchem_retrieval_eval_icicle_scaffold_s1",
        "NEIMS": REPO_ROOT
        / "results"
        / "pubchem_retrieval_eval_neims_scaffold_s1",
    },
}
GLOBAL_PUBCHEM_HDF5 = {
    "ICICLE": REPO_ROOT
    / "results"
    / "inference"
    / "pubchem_predictions_rerun_260710.hdf5",
    "NEIMS": {
        "random": REPO_ROOT
        / "baselines"
        / "neims"
        / "results"
        / "pubchem_predictions"
        / "neims_random_s1_pubchem_full.hdf5",
        "scaffold": REPO_ROOT
        / "baselines"
        / "neims"
        / "results"
        / "pubchem_predictions"
        / "neims_scaffold_s1_pubchem_full.hdf5",
    },
}

metadata_full = pd.read_csv(
    REPO_ROOT / "data" / "NIST2023_GCMS_main" / "metadata.tsv",
    sep="\t",
    usecols=["mol_id", "standardized_smiles", "formula", "inchi_key"],
)
metadata_full["inchikey14"] = metadata_full["inchi_key"].astype(str).str[:14]


def load_icicle_formula_rows(regime: str, mol_id: int) -> pd.DataFrame:
    raw = pd.read_csv(
        FORMULA_DIRS[regime]["ICICLE"] / "retrieval_with_formula_results.csv"
    )
    return raw[raw["spec"] == mol_id]


def load_neims_formula_rows(regime: str, inchikey14: str) -> pd.DataFrame:
    raw = pd.read_csv(
        FORMULA_DIRS[regime]["NEIMS"] / "retrieval_with_formula_results.csv"
    )
    return raw[raw["query_inchikey14"] == inchikey14]


def formula_stats(rows: pd.DataFrame, true_mask: pd.Series) -> dict:
    true_row = rows[true_mask].iloc[0]
    decoys = rows[~true_mask]
    top_decoy = (
        decoys.loc[decoys["cosine_similarity"].idxmax()]
        if len(decoys)
        else None
    )
    return {
        "rank_cosine": int(true_row["rank_cosine_similarity"]),
        "cosine_similarity": float(true_row["cosine_similarity"]),
        "entropy_similarity": float(true_row["entropy_similarity"]),
        "n_decoys": int(len(decoys)),
        "top_decoy_formula": top_decoy["formula"]
        if top_decoy is not None
        else None,
        "top_decoy_cosine_similarity": float(top_decoy["cosine_similarity"])
        if top_decoy is not None
        else None,
    }


def load_icicle_eval_group(split: str, mol_id: int):
    full_inchikey = metadata_full.loc[
        metadata_full["mol_id"] == mol_id, "inchi_key"
    ].iloc[0]
    with h5py.File(SIM_DIRS[split] / "all_evaluation_spectra.hdf5", "r") as f:
        if full_inchikey not in f:
            return None, None, None
        grp = f[full_inchikey]
        return (
            grp["mz_bins"][:],
            grp["predicted_intensities"][:],
            grp["ground_truth_intensities"][:],
        )


def load_neims_predicted_spectrum(split: str, mol_id: int) -> np.ndarray:
    with h5py.File(NEIMS_PRED_FILES[split], "r") as f:
        key = str(mol_id)
        return f[key]["predicted_intensities"][:] if key in f else None


def scan_top_pubchem_candidate(
    hdf5_path: Path,
    query_spec: np.ndarray,
    exclude_inchikey14: str,
    chunk_size: int = 200_000,
) -> dict:
    """Stream through the full PubChem prediction HDF5 to find the candidate
    (excluding the true molecule itself) with highest cosine similarity to
    `query_spec`. Only used for the handful of specific.

    global-retrieval examples picked for this figure, not the full test set
    -- a full GPU-batched scan (as in pubchem_global_retrieval.py) would be
    overkill here.
    """
    query_norm = query_spec / (np.linalg.norm(query_spec) + 1e-10)
    best_sim, best_idx = -1.0, -1
    with h5py.File(hdf5_path, "r") as f:
        n = f["intensities"].shape[0]
        ik14 = f["inchikey14"]
        for start in tqdm(
            range(0, n, chunk_size), desc=f"scanning {hdf5_path.name}"
        ):
            end = min(start + chunk_size, n)
            cands = f["intensities"][start:end]
            norms = np.linalg.norm(cands, axis=1) + 1e-10
            sims = (cands @ query_norm) / norms
            keys = ik14[start:end]
            keys = np.array(
                [k.decode() if isinstance(k, bytes) else k for k in keys]
            )
            sims[keys == exclude_inchikey14] = -1.0
            local_best = np.argmax(sims)
            if sims[local_best] > best_sim:
                best_sim = float(sims[local_best])
                best_idx = start + local_best
    with h5py.File(hdf5_path, "r") as f:
        smiles = f["smiles"][best_idx]
        smiles = smiles.decode() if isinstance(smiles, bytes) else smiles
    return {
        "top_decoy_cosine_similarity": best_sim,
        "top_decoy_smiles": smiles,
    }


def plot_example(
    regime: str, mol_id: int, category: str, output_dir: Path
) -> dict:
    is_global = regime.endswith("_global")
    split = regime.split("_")[0]
    row = metadata_full[metadata_full["mol_id"] == mol_id].iloc[0]
    inchikey14 = row["inchikey14"]

    mz, icicle_pred, exp = load_icicle_eval_group(split, mol_id)
    neims_pred = load_neims_predicted_spectrum(split, mol_id)
    if icicle_pred is None or neims_pred is None:
        log.warning(f"Skipping mol_id={mol_id}: missing predicted spectrum")
        return None

    if is_global:
        icicle_ranks = pd.read_csv(
            GLOBAL_RETRIEVAL_DIRS[split]["ICICLE"]
            / "retrieval_global_per_query.tsv",
            sep="\t",
        )
        neims_ranks = pd.read_csv(
            GLOBAL_RETRIEVAL_DIRS[split]["NEIMS"]
            / "retrieval_global_per_query.tsv",
            sep="\t",
        )
        icicle_row = icicle_ranks[icicle_ranks["mol_id"] == mol_id].iloc[0]
        neims_row = neims_ranks[neims_ranks["mol_id"] == mol_id].iloc[0]
        icicle_stats = {
            "rank_cosine": int(icicle_row["rank_autofail_cosine"]),
            "n_decoys": int(icicle_row["n_candidates"]) - 1,
        }
        neims_stats = {
            "rank_cosine": int(neims_row["rank_autofail_cosine"]),
            "n_decoys": int(neims_row["n_candidates"]) - 1,
        }
        icicle_stats["cosine_similarity"] = float(
            cosine_similarity(icicle_pred, exp)
        )
        neims_stats["cosine_similarity"] = float(
            cosine_similarity(neims_pred, exp)
        )
        if icicle_stats["rank_cosine"] != 1:
            icicle_stats.update(
                scan_top_pubchem_candidate(
                    GLOBAL_PUBCHEM_HDF5["ICICLE"], icicle_pred, inchikey14
                )
            )
        if neims_stats["rank_cosine"] != 1:
            neims_stats.update(
                scan_top_pubchem_candidate(
                    GLOBAL_PUBCHEM_HDF5["NEIMS"][split], neims_pred, inchikey14
                )
            )
    else:
        icicle_rows = load_icicle_formula_rows(split, mol_id)
        neims_rows = load_neims_formula_rows(split, inchikey14)
        icicle_stats = formula_stats(icicle_rows, ~icicle_rows["is_decoy"])
        neims_stats = formula_stats(neims_rows, neims_rows["is_correct"])

    log.info(
        f"[{regime}/{category}] mol_id={mol_id} formula={row['formula']}\n"
        f"  ICICLE: {icicle_stats}\n  NEIMS: {neims_stats}"
    )

    fig = plot_three_spectra(
        exp_spec=exp,
        qcxms_spec=neims_pred,
        icicle_spec=icicle_pred,
        mz_values=mz,
        smiles=None,
        fade_unmatched=True,
        qcxms_label="NEIMS",
        icicle_color=MODEL_COLORS["ICICLE"],
        qcxms_color=MODEL_COLORS["NEIMS"],
        figsize=(3, 1.5),
    )
    stem = f"stacked_spectrum_{regime}_{category}_mol{mol_id}"
    save_fig(fig, stem, output_dir)
    log.info(f"  Saved {stem}.[svg|png]")

    return {
        "regime": regime,
        "category": category,
        "mol_id": mol_id,
        "inchikey14": inchikey14,
        "formula": row["formula"],
        **{f"icicle_{k}": v for k, v in icicle_stats.items()},
        **{f"neims_{k}": v for k, v in neims_stats.items()},
    }


def pick_examples_formula(regime: str) -> dict:
    """Pick one mol_id per category from the formula-retrieval CSVs."""
    icicle = pd.read_csv(
        FORMULA_DIRS[regime]["ICICLE"] / "retrieval_with_formula_results.csv"
    )
    neims = pd.read_csv(
        FORMULA_DIRS[regime]["NEIMS"] / "retrieval_with_formula_results.csv"
    )

    icicle_true = icicle[~icicle["is_decoy"]].rename(
        columns={"spec": "mol_id"}
    )
    icicle_true = icicle_true.merge(
        metadata_full[["mol_id", "inchikey14"]], on="mol_id", how="left"
    )
    neims_true = neims[neims["is_correct"]].rename(
        columns={"query_inchikey14": "inchikey14"}
    )

    merged = icicle_true.merge(
        neims_true, on="inchikey14", suffixes=("_icicle", "_neims")
    )
    merged = merged.dropna(subset=["mol_id"])

    picks = {}

    icicle_wins = merged[merged["rank_cosine_similarity_icicle"] == 1]
    if len(icicle_wins):
        picks["a_icicle_top1"] = int(icicle_wins.iloc[0]["mol_id"])

    icicle_not1_better = merged[
        (merged["rank_cosine_similarity_icicle"] != 1)
        & (
            merged["rank_cosine_similarity_icicle"]
            < merged["rank_cosine_similarity_neims"]
        )
    ].sort_values("rank_cosine_similarity_icicle")
    if len(icicle_not1_better):
        picks["d_icicle_better_not_top1"] = int(
            icicle_not1_better.iloc[0]["mol_id"]
        )

    much_worse = merged[
        (merged["rank_cosine_similarity_icicle"] != 1)
        & (
            merged["rank_cosine_similarity_icicle"]
            > merged["rank_cosine_similarity_neims"] * 5
        )
    ].sort_values(
        "rank_cosine_similarity_icicle"
    )  # least-extreme qualifying case, not the single worst outlier
    if len(much_worse):
        picks["e_icicle_much_worse"] = int(much_worse.iloc[0]["mol_id"])

    both_bad = merged[
        (merged["rank_cosine_similarity_icicle"] > 20)
        & (merged["rank_cosine_similarity_neims"] > 20)
    ].sort_values("rank_cosine_similarity_icicle")
    if len(both_bad):
        picks["f_both_bad"] = int(both_bad.iloc[0]["mol_id"])

    return picks


def pick_examples_global(split: str) -> dict:
    icicle = pd.read_csv(
        GLOBAL_RETRIEVAL_DIRS[split]["ICICLE"]
        / "retrieval_global_per_query.tsv",
        sep="\t",
    )
    neims = pd.read_csv(
        GLOBAL_RETRIEVAL_DIRS[split]["NEIMS"]
        / "retrieval_global_per_query.tsv",
        sep="\t",
    )
    merged = icicle.merge(neims, on="mol_id", suffixes=("_icicle", "_neims"))

    picks = {}

    icicle_wins = merged[merged["rank_autofail_cosine_icicle"] == 1]
    if len(icicle_wins):
        picks["b_icicle_top1"] = int(icicle_wins.iloc[0]["mol_id"])

    icicle_not1_better = merged[
        (merged["rank_autofail_cosine_icicle"] != 1)
        & (
            merged["rank_autofail_cosine_icicle"]
            < merged["rank_autofail_cosine_neims"]
        )
    ].sort_values("rank_autofail_cosine_icicle")
    if len(icicle_not1_better):
        picks["d_icicle_better_not_top1"] = int(
            icicle_not1_better.iloc[0]["mol_id"]
        )

    much_worse = merged[
        (merged["rank_autofail_cosine_icicle"] != 1)
        & (
            merged["rank_autofail_cosine_icicle"]
            > merged["rank_autofail_cosine_neims"] * 100
        )
    ].sort_values(
        "rank_autofail_cosine_icicle"
    )  # least-extreme qualifying case, not the single worst outlier
    if len(much_worse):
        picks["e_icicle_much_worse"] = int(much_worse.iloc[0]["mol_id"])

    both_bad = merged[
        (merged["rank_autofail_cosine_icicle"] > 100_000)
        & (merged["rank_autofail_cosine_neims"] > 100_000)
    ].sort_values("rank_autofail_cosine_icicle")
    if len(both_bad):
        picks["f_both_bad"] = int(both_bad.iloc[0]["mol_id"])

    return picks


def main():
    records = []
    for regime_name, split, picker in [
        ("random_formula", "random", lambda: pick_examples_formula("random")),
        (
            "scaffold_formula",
            "scaffold",
            lambda: pick_examples_formula("scaffold"),
        ),
        ("random_global", "random", lambda: pick_examples_global("random")),
        (
            "scaffold_global",
            "scaffold",
            lambda: pick_examples_global("scaffold"),
        ),
    ]:
        log.info(f"=== {regime_name} ===")
        picks = picker()
        log.info(f"  picks: {picks}")
        for category, mol_id in picks.items():
            record = plot_example(regime_name, mol_id, category, OUTPUT_DIR)
            if record is not None:
                records.append(record)

    df = pd.DataFrame(records)
    out_csv = OUTPUT_DIR / "win_loss_examples_rank_decoys.csv"
    df.to_csv(out_csv, index=False)
    log.info(f"Wrote {len(df)} examples to {out_csv}")


if __name__ == "__main__":
    main()
