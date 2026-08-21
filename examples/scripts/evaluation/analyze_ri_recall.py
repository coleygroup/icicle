"""Analyze recall vs top-N for RI-based candidate selection.

Compares how much various oracle pre-filters improve recall when combined
with RI-based ranking against the full PubChem reference database.

Scenarios
---------
All filters are *oracle* (they use ground-truth properties of the query
molecule).  They are upper-bound estimates of how much each piece of
structural/chemical information would improve retrieval if it were available.

1.  RI only               — pure RI ranking, no pre-filter (baseline)
2.  RI + exact formula    — filter ref to identical molecular formula
3.  RI + nominal mass     — filter ref to same integer Da mass
4.  RI + MW ±1 Da         — tight mass window
5.  RI + MW ±10 Da        — moderate mass window
6.  RI + MW ±80 Da        — broad mass window
7.  RI + DBE ±1           — degree-of-unsaturation window (±1 unit)
8.  RI + DBE ±2           — broader DBE window (±2 units)
9.  RI + compound class   — filter ref to same main-heteroatom class
                            (pure HC / O-containing / N-containing /
                             S-containing / halogenated / other)
10. RI + aromaticity      — filter ref to same aromatic/aliphatic class
"""

# %%
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from tqdm.auto import tqdm

from icicle.utils.visualization import palette, set_style

set_style()

HALOGEN_SYMBOLS = frozenset(("F", "Cl", "Br", "I"))

# %% [markdown]
# ## Helpers


# %%
def _calc_dbe_from_mol(mol) -> float:
    """Degree of unsaturation: 1 + (2C + N - H - X) / 2."""
    c = h = n = x = 0
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        h += atom.GetTotalNumHs()
        if sym == "C":
            c += 1
        elif sym == "N":
            n += 1
        elif sym in HALOGEN_SYMBOLS:
            x += 1
        elif sym == "H":
            h += 1
    return 1.0 + (2 * c + n - h - x) / 2.0


def _main_heteroatom_class(formula: str) -> str:
    """Coarse compound class based on main heteroatom in molecular formula."""
    if not isinstance(formula, str):
        return "unknown"
    has_halogen = bool(re.search(r"(F|Cl|Br|I)\d*", formula))
    has_n = "N" in formula
    has_s = "S" in formula
    has_o = "O" in formula
    if has_halogen:
        return "halogenated"
    if has_s:
        return "S-containing"
    if has_n and has_o:
        return "N+O-containing"
    if has_n:
        return "N-containing"
    if has_o:
        return "O-containing"
    return "pure HC"


# %% [markdown]
# ## Load Data

# %%
# RI_PREDICTIONS_FILE = "/home/magled/icicle-dev/data/PubChem/260303_full_PubChem_AIRI_with_RI_inchikey.tsv"

RI_PREDICTIONS_FILE = "/home/magled/icicle-dev/data/PubChem/PubChem_filtered_with_ri_new_random_inchikey.tsv"
RI_DATASET_FILE = (
    "/home/magled/icicle-dev/data/NIST2023_GCMS_main/retention_index/"
    "ri_dataset_random_split_no_xeno_aas.tsv"
)

RI_TYPES = ["StdNP"]  # , "SemiStdNP", "StdPolar"]

# %%
print("Loading test dataset...")
test_df = pd.read_csv(RI_DATASET_FILE, sep="\t")
test_df = test_df[test_df["split"] == "test"].copy()
test_df["ik_prefix"] = test_df["inchi_key"].str[:14]

print("  Computing descriptors for test molecules...")
desc_rows = []
for smi in test_df["standardized_smiles"]:
    if not isinstance(smi, str) or not smi:
        desc_rows.append((None, None, None, None, None))
        continue
    mol = MolFromSmiles(smi)
    if mol is None:
        desc_rows.append((None, None, None, None, None))
        continue
    mw = ExactMolWt(mol)
    formula = CalcMolFormula(mol)
    dbe = _calc_dbe_from_mol(mol)
    is_arom = any(a.GetIsAromatic() for a in mol.GetAtoms())
    desc_rows.append(
        (mw, formula, dbe, is_arom, _main_heteroatom_class(formula))
    )

test_df[
    ["mw", "molecular_formula", "dbe", "is_aromatic", "compound_class"]
] = pd.DataFrame(desc_rows, index=test_df.index)
test_df["nominal_mass"] = test_df["mw"].round().astype("Int64")

print(f"Test molecules: {len(test_df):,}")
for rt in RI_TYPES:
    print(f"  {rt}: {test_df[f'ri_{rt}'].notna().sum()} with experimental RI")

# %%
print("Loading PubChem reference database...")
_parquet_cache = RI_PREDICTIONS_FILE.replace(".tsv", "_cache.parquet")
if os.path.exists(_parquet_cache):
    print(f"  Loading parquet cache ({_parquet_cache})...")
    ref_df = pd.read_parquet(_parquet_cache)
    print("  Done.")
else:
    _needed_cols = [
        "InChIKey",
        "mw",
        "molecular_formula",
        "dbe",
        "is_aromatic",
        "ri_StdNP",
        "ri_SemiStdNP",
        "ri_StdPolar",
    ]
    ref_df = pd.read_csv(
        RI_PREDICTIONS_FILE, sep="\t", usecols=_needed_cols, engine="pyarrow"
    )
    ref_df["ik_prefix"] = ref_df["InChIKey"].str[:14]
    ref_df["nominal_mass"] = ref_df["mw"].round().astype("Int64")
    # compute compound_class on unique formulas only (much faster)
    unique_formulas = ref_df["molecular_formula"].dropna().unique()
    formula_class = {
        f: _main_heteroatom_class(f)
        for f in tqdm(unique_formulas, desc="compound_class")
    }
    ref_df["compound_class"] = ref_df["molecular_formula"].map(formula_class)
    ref_df.to_parquet(_parquet_cache, index=False)
    print("  (saved parquet cache for next run)")
print(f"Reference molecules: {len(ref_df):,}")

# %% [markdown]
# ## Ranking Engine


# %%
def compute_ground_truth_ranks(
    test_df, ref_df, ri_type, filter_type=None, filter_param=None
):
    """Rank ground-truth molecules by RI within an optional pre-filter.

    Parameters
    ----------
    filter_type : str or None
        One of: None, 'mw_window', 'nominal_mass', 'formula',
                'dbe_window', 'compound_class', 'aromaticity'
    filter_param : float or None
        Window half-width for 'mw_window' and 'dbe_window'.

    Returns
    -------
    ranks : list[float]   1-indexed; inf = ground truth not in (filtered) ref
    n_test : int
    n_in_ref : int
    """
    ri_col = f"ri_{ri_type}"
    test_with_ri = test_df[test_df[ri_col].notna()].copy()
    ref_with_ri = ref_df[
        ref_df[ri_col].notna() & ref_df["ik_prefix"].notna()
    ].copy()

    if len(test_with_ri) == 0 or len(ref_with_ri) == 0:
        return [], 0, 0

    ref_ri = ref_with_ri[ri_col].values.astype(np.float32)
    ref_ik = ref_with_ri["ik_prefix"].values

    # build filter-specific index structures
    if filter_type in ("mw_window", "dbe_window"):
        sort_col = "mw" if filter_type == "mw_window" else "dbe"
        raw_vals = ref_with_ri[sort_col].values.astype(np.float64)
        bin_keys = np.floor(raw_vals).astype(np.int32)
        tmp = pd.DataFrame({"bin": bin_keys, "ri": ref_ri, "ik": ref_ik})
        ri_bins: dict[int, np.ndarray] = {}
        ik_bins: dict[int, np.ndarray] = {}
        for bk, grp in tqdm(
            tmp.groupby("bin"),
            desc="Building index",
            total=tmp["bin"].nunique(),
        ):
            ri_arr = grp["ri"].values.astype(np.float32)
            ik_arr = grp["ik"].values
            order = np.argsort(ri_arr)
            ri_bins[int(bk)] = ri_arr[order]
            ik_bins[int(bk)] = ik_arr[order]

    elif filter_type in (
        "nominal_mass",
        "formula",
        "compound_class",
        "aromaticity",
    ):
        # dict: key -> (ri_array, ik_array)
        key_col = {
            "nominal_mass": "nominal_mass",
            "formula": "molecular_formula",
            "compound_class": "compound_class",
            "aromaticity": "is_aromatic",
        }[filter_type]
        group_ri: dict = {}
        group_ik: dict = {}
        keys = ref_with_ri[key_col].values
        for i, k in tqdm(
            enumerate(keys), total=len(keys), desc="Building index"
        ):
            if k not in group_ri:
                group_ri[k] = []
                group_ik[k] = []
            group_ri[k].append(ref_ri[i])
            group_ik[k].append(ref_ik[i])
        # convert to numpy
        group_ri = {
            k: np.array(v, dtype=np.float32) for k, v in group_ri.items()
        }
        group_ik = {k: np.array(v) for k, v in group_ik.items()}

    else:  # no filter — sort by RI once, encode ik as int codes for fast sort
        print(
            f"  [{ri_type}] Building RI-only index ({len(ref_with_ri):,} rows)…",
            flush=True,
        )
        s_ri = np.sort(ref_ri)
        # factorize converts strings -> int codes in O(N), then sort ints (fast)
        ik_codes, ik_uniques = pd.factorize(
            ref_with_ri["ik_prefix"], sort=True
        )
        ik_order = np.argsort(ik_codes, kind="stable")
        sorted_ik_codes = ik_codes[ik_order]
        sorted_ik_ri = ref_ri[ik_order]
        print(
            f"  [{ri_type}] Index ready. Starting per-query ranking…",
            flush=True,
        )

    # extract query arrays (avoid slow iterrows)
    q_ri = test_with_ri[ri_col].values.astype(np.float32)
    q_ik = test_with_ri["ik_prefix"].values
    if filter_type == "mw_window":
        q_filter = test_with_ri["mw"].values
    elif filter_type == "dbe_window":
        q_filter = test_with_ri["dbe"].values
    elif filter_type in (
        "nominal_mass",
        "formula",
        "compound_class",
        "aromaticity",
    ):
        _key_col = {
            "nominal_mass": "nominal_mass",
            "formula": "molecular_formula",
            "compound_class": "compound_class",
            "aromaticity": "is_aromatic",
        }[filter_type]
        q_filter = test_with_ri[_key_col].values
    else:
        q_filter = None

    # per-query ranking
    ranks = []
    n_in_ref = 0
    desc = f"{ri_type}" + (
        f" [{filter_type}={filter_param}]" if filter_type else ""
    )

    for idx in tqdm(range(len(test_with_ri)), desc=desc):
        query_ri = q_ri[idx]
        target_ik = q_ik[idx]

        if filter_type in ("mw_window", "dbe_window"):
            q_val = q_filter[idx]
            lo_bin = int(np.floor(q_val - filter_param))
            hi_bin = int(np.floor(q_val + filter_param))
            # find best RI diff for target across all bins in window
            best_diff = np.inf
            for bk in range(lo_bin, hi_bin + 1):
                if bk not in ri_bins:
                    continue
                mask = ik_bins[bk] == target_ik
                if mask.any():
                    d = float(np.min(np.abs(ri_bins[bk][mask] - query_ri)))
                    if d < best_diff:
                        best_diff = d
            if best_diff == np.inf:
                ranks.append(np.inf)
                continue
            n_in_ref += 1
            # count molecules with |ri - query_ri| < best_diff via binary search per bin
            count = 0
            for bk in range(lo_bin, hi_bin + 1):
                if bk not in ri_bins:
                    continue
                b_ri = ri_bins[bk]
                lo_r = int(
                    np.searchsorted(b_ri, query_ri - best_diff, side="right")
                )
                hi_r = int(
                    np.searchsorted(b_ri, query_ri + best_diff, side="left")
                )
                count += hi_r - lo_r
            rank = count + 1

        elif filter_type in (
            "nominal_mass",
            "formula",
            "compound_class",
            "aromaticity",
        ):
            q_key = q_filter[idx]
            if q_key not in group_ri:
                ranks.append(np.inf)
                continue
            cand_ri = group_ri[q_key]
            cand_ik = group_ik[q_key]
            mask = cand_ik == target_ik
            if not mask.any():
                ranks.append(np.inf)
                continue
            n_in_ref += 1
            ri_diffs = np.abs(cand_ri - query_ri)
            rank = int(np.sum(ri_diffs < np.min(ri_diffs[mask]))) + 1

        else:  # no filter
            target_code = int(np.searchsorted(ik_uniques, target_ik))
            if (
                target_code >= len(ik_uniques)
                or ik_uniques[target_code] != target_ik
            ):
                ranks.append(np.inf)
                continue
            lo_ik = int(
                np.searchsorted(sorted_ik_codes, target_code, side="left")
            )
            hi_ik = int(
                np.searchsorted(sorted_ik_codes, target_code, side="right")
            )
            n_in_ref += 1
            target_ri_vals = sorted_ik_ri[lo_ik:hi_ik]
            best = float(np.min(np.abs(target_ri_vals - query_ri)))
            # Count molecules with distance *strictly less than* best (open
            # interval), matching the strict-< convention in filtered branches.
            lo = int(np.searchsorted(s_ri, query_ri - best, side="right"))
            hi = int(np.searchsorted(s_ri, query_ri + best, side="left"))
            rank = (hi - lo) + 1

        ranks.append(rank)

    return ranks, len(test_with_ri), n_in_ref


def compute_recall_at_n(
    ranks: np.ndarray, top_n_values: np.ndarray
) -> np.ndarray:
    return np.array([np.mean(ranks <= n) for n in top_n_values])


# %%
top_n_values = np.unique(
    np.concatenate(
        [
            np.arange(1, 100, 10),
            np.arange(100, 1000, 100),
            np.arange(1000, 10000, 1000),
            np.arange(10000, 100001, 10000),
            np.arange(100000, 1000001, 100000),
            np.arange(1000000, 10000001, 1000000),
        ]
    )
).astype(int)

print(
    f"Top-N range: {top_n_values[0]:,} – {top_n_values[-1]:,}  ({len(top_n_values)} points)"
)

# %%
# (label, filter_type, filter_param, description for plot/report)
SCENARIOS: list[tuple[str, str | None, float | None, str]] = [
    (
        "RI only",
        None,
        None,
        "Baseline: rank all 95M PubChem molecules by |RI_pred − RI_exp|. "
        "No structural pre-filter.",
    ),
    (
        "RI + exact formula",
        "formula",
        None,
        "Oracle: keep only ref molecules with the *identical* molecular formula "
        "(e.g. C10H18O). Strongest structural filter; requires knowing the "
        "exact elemental composition, which is not obtainable from low-res EI alone.",
    ),
    (
        "RI + nominal mass",
        "nominal_mass",
        None,
        "Oracle: keep only ref molecules whose monoisotopic mass rounds to the "
        "same integer Da. Simulates detecting M⁺ at unit resolution. "
        "Note: M⁺ is often absent or weak in EI spectra.",
    ),
    # (
    #     "RI + MW ±1 Da",
    #     "mw_window",
    #     1.0,
    #     "Oracle mass window of ±1 Da. Tighter than nominal mass but still "
    #     "requires a reliable M⁺ measurement.",
    # ),
    # (
    #     "RI + MW ±10 Da",
    #     "mw_window",
    #     10.0,
    #     "Oracle mass window ±10 Da. Moderate constraint; useful if M⁺ is "
    #     "detectable but measurement is imprecise.",
    # ),
    # (
    #     "RI + MW ±80 Da",
    #     "mw_window",
    #     80.0,
    #     "Oracle mass window ±80 Da. Very broad constraint; represents a "
    #     "rough mass range estimate.",
    # ),
    # (
    #     "RI + DBE ±1",
    #     "dbe_window",
    #     1.0,
    #     "Oracle: filter ref to molecules within ±1 degree of unsaturation "
    #     "(DBE = rings + double bonds). Constrains saturation class without "
    #     "requiring exact mass.",
    # ),
    # (
    #     "RI + DBE ±2",
    #     "dbe_window",
    #     2.0,
    #     "Oracle: broader DBE window of ±2 units.",
    # ),
    # (
    #     "RI + compound class",
    #     "compound_class",
    #     None,
    #     "Oracle: filter ref to the same main-heteroatom class "
    #     "(pure HC / O-containing / N-containing / S-containing / "
    #     "halogenated / N+O-containing). "
    #     "Coarse structural class sometimes inferrable from EI fragment patterns.",
    # ),
    # (
    #     "RI + aromaticity",
    #     "aromaticity",
    #     None,
    #     "Oracle: filter ref to aromatic or non-aromatic molecules. "
    #     "Aromaticity is often inferrable from EI spectra "
    #     "(e.g. m/z 77/91 for phenyl/benzyl).",
    # ),
]

OUTPUT_DIR = "/home/magled/icicle-dev/examples/notebooks"

colors_ri = {
    "StdNP": palette[0],
    "SemiStdNP": palette[4],
    "StdPolar": palette[10],
}


def _plot_recall(ax, recall_curves, results, top_n_values, title):
    for ri_type in RI_TYPES:
        if ri_type not in recall_curves:
            continue
        n = results[ri_type]["n_in_ref"]
        ax.plot(
            top_n_values,
            recall_curves[ri_type] * 100,
            label=f"{ri_type} (n={n:,})",
            color=colors_ri[ri_type],
            linewidth=2,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Top-N candidates")
    ax.set_ylabel("Recall (%)")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7)
    ax.set_ylim(0, 105)
    ax.set_xlim(top_n_values[0], top_n_values[-1])
    for pct in [50, 90, 95]:
        ax.axhline(pct, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
        ax.text(
            top_n_values[-1] * 1.05,
            pct,
            f"{pct}%",
            va="center",
            fontsize=8,
            color="gray",
        )


# %% [markdown]
# ## Run All Scenarios

# %%
all_results: dict[str, dict] = {}
all_recall: dict[str, dict] = {}

for label, ftype, fparam, _ in SCENARIOS:
    print(f"\n{'=' * 60}\n{label}\n{'=' * 60}")
    results = {}
    for ri_type in RI_TYPES:
        ranks, n_test, n_in_ref = compute_ground_truth_ranks(
            test_df, ref_df, ri_type, filter_type=ftype, filter_param=fparam
        )
        ranks_arr = np.array(ranks, dtype=float)
        ranks_fin = ranks_arr[np.isfinite(ranks_arr)]
        results[ri_type] = {
            "ranks": ranks_fin,
            "n_test": n_test,
            "n_in_ref": n_in_ref,
        }

        pct = n_in_ref / n_test * 100 if n_test else 0
        print(f"  {ri_type}: {n_in_ref}/{n_test} in ref ({pct:.1f}%)", end="")
        if len(ranks_fin):
            coverage = pct / 100
            print(
                f"  median rank {np.median(ranks_fin):,.0f}"
                f"  (effective recall scaled by {coverage:.2f} coverage)"
            )
        else:
            print()

    all_results[label] = results
    all_recall[label] = {
        rt: compute_recall_at_n(results[rt]["ranks"], top_n_values)
        for rt in RI_TYPES
        if len(results[rt]["ranks"])
    }

    slug = label.lower().replace(" ", "_").replace("±", "pm").replace("/", "_")

    # save per-scenario recall curve
    fig, ax = plt.subplots(figsize=(3.25, 3.25))
    _plot_recall(
        ax, all_recall[label], all_results[label], top_n_values, label
    )
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/ri_recall_{slug}.svg"
    plt.savefig(path, bbox_inches="tight", format="svg")
    print(f"  Saved {path}")
    plt.close(fig)

    # save per-scenario CSV (matching old format)
    scenario_csv = pd.DataFrame({"top_n": top_n_values})
    for ri_type in RI_TYPES:
        if ri_type in all_recall[label]:
            scenario_csv[f"recall_{ri_type}"] = all_recall[label][ri_type]
    csv_path = f"{OUTPUT_DIR}/ri_recall_{slug}.csv"
    scenario_csv.to_csv(csv_path, index=False)
    print(f"  Saved {csv_path}")

    # save per-scenario rank distribution histogram
    n_types = sum(
        1 for rt in RI_TYPES if len(all_results[label][rt]["ranks"]) > 0
    )
    if n_types > 0:
        fig2, axes2 = plt.subplots(1, n_types, figsize=(3.25 * n_types, 3.25))
        if n_types == 1:
            axes2 = [axes2]
        ax_iter = iter(axes2)
        for ri_type in RI_TYPES:
            ranks = all_results[label][ri_type]["ranks"]
            if len(ranks) == 0:
                continue
            ax2 = next(ax_iter)
            ax2.hist(
                np.log10(ranks + 1),
                bins=50,
                color=colors_ri[ri_type],
            )
            ax2.set_xlabel("log₁₀(rank)")
            ax2.set_ylabel("Count")
            ax2.set_title(
                f"{ri_type}\nmedian: {np.median(ranks):,.0f}", fontsize=9
            )
        fig2.suptitle(label, fontsize=9)
        plt.tight_layout()
        path2 = f"{OUTPUT_DIR}/ri_ranks_hist_{slug}.svg"
        plt.savefig(path2, bbox_inches="tight", format="svg")
        print(f"  Saved {path2}")
        plt.close(fig2)

# %% [markdown]
# ## Plots

# all scenarios on one grid (per RI type)
scenario_colors = [
    palette[i % len(palette)] for i in range(0, len(SCENARIOS) * 2, 2)
]
scenario_styles = ["-", "--", "-.", ":", "-", "--", "-.", ":", "-", "--"]

for ri_type in RI_TYPES:
    fig, ax = plt.subplots(figsize=(4.5, 3.25))
    for i, (label, _, _, _) in enumerate(SCENARIOS):
        if ri_type not in all_recall[label]:
            continue
        ax.plot(
            top_n_values,
            all_recall[label][ri_type] * 100,
            label=label,
            color=scenario_colors[i % len(palette)],
            linewidth=1.5,
            linestyle=scenario_styles[i % len(scenario_styles)],
        )
    ax.set_xscale("log")
    ax.set_xlabel("Top-N candidates")
    ax.set_ylabel("Recall (%)")
    ax.set_title(f"{ri_type} — all scenarios", fontsize=9)
    ax.legend(fontsize=6, loc="upper left")
    ax.set_ylim(0, 105)
    ax.set_xlim(top_n_values[0], top_n_values[-1])
    for pct in [50, 90, 95]:
        ax.axhline(pct, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/ri_recall_all_scenarios_{ri_type}.svg"
    plt.savefig(path, bbox_inches="tight", format="svg")
    print(f"Saved {path}")
    plt.show()

# %% [markdown]
# ## Summary Table

# %%
key_n = [100, 1000, 10000, 100000, 1000000]

for ri_type in RI_TYPES:
    print(f"\n{'=' * 80}")
    print(f"Recall @ key Top-N  —  {ri_type}")
    print(f"{'=' * 80}")
    header = f"{'Scenario':<28}" + "".join(f"{n:>12,}" for n in key_n)
    print(header)
    print("-" * len(header))
    for label, _, _, desc in SCENARIOS:
        if ri_type not in all_recall[label]:
            continue
        row = f"{label:<28}"
        for n in key_n:
            idx = np.argmin(np.abs(top_n_values - n))
            row += f" {all_recall[label][ri_type][idx] * 100:>10.1f}%"
        print(row)

# %% [markdown]
# ## Scenario Descriptions

for label, ftype, fparam, desc in SCENARIOS:
    print(f"[{label}]")
    print(f"  {desc}")
    print()

# %%
# Save all recall curves to CSV
recall_df = pd.DataFrame({"top_n": top_n_values})
for label, _, _, _ in SCENARIOS:
    slug = label.lower().replace(" ", "_").replace("±", "pm").replace("/", "_")
    for ri_type in RI_TYPES:
        if ri_type in all_recall[label]:
            recall_df[f"{slug}__{ri_type}"] = all_recall[label][ri_type]

recall_df.to_csv(f"{OUTPUT_DIR}/ri_recall_all_scenarios.csv", index=False)
print(f"Saved {OUTPUT_DIR}/ri_recall_all_scenarios.csv")
