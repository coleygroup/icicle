"""Bond-breaking pattern analysis from ICICLE fragmentation predictions."""

import hashlib
import multiprocessing as mp
import os
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from tqdm import tqdm

from icicle.utils.visualization.style import get_palette, make_fig, set_style

_FRAGMENT_CACHE_DIR = "tmp/bond_breaking_fragment_cache"


def get_broken_bonds(
    mol: Chem.Mol, fragment_atoms: List[int]
) -> List[Dict[str, Any]]:
    """Return bonds broken in a fragment relative to the parent molecule.

    A bond is broken if exactly one of its atoms belongs to the fragment.

    Args:
        mol: Parent RDKit molecule.
        fragment_atoms: Atom indices present in the fragment.

    Returns:
        List of dicts with keys: begin_atom, end_atom, bond_type, bond_idx.
    """
    atom_set = set(fragment_atoms)
    broken = []
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        if (begin in atom_set) != (end in atom_set):
            broken.append(
                {
                    "begin_atom": begin,
                    "end_atom": end,
                    "bond_type": bond.GetBondType(),
                    "bond_idx": bond.GetIdx(),
                }
            )
    return broken


def bond_to_label(
    mol: Chem.Mol,
    begin_atom: int,
    end_atom: int,
    bond_type: Chem.rdchem.BondType,
) -> str:
    """Create a canonical label for a bond, e.g. 'C-N', 'C=O', 'C#N'.

    Atom symbols are sorted alphabetically so C-N and N-C map to the same label.

    Args:
        mol: Parent RDKit molecule.
        begin_atom: Begin atom index.
        end_atom: End atom index.
        bond_type: RDKit bond type.

    Returns:
        Canonical bond label string.
    """
    bond_symbol = {
        Chem.rdchem.BondType.SINGLE: "-",
        Chem.rdchem.BondType.DOUBLE: "=",
        Chem.rdchem.BondType.TRIPLE: "#",
        Chem.rdchem.BondType.AROMATIC: ":",
    }.get(bond_type, "~")

    sym1 = mol.GetAtomWithIdx(begin_atom).GetSymbol()
    sym2 = mol.GetAtomWithIdx(end_atom).GetSymbol()
    pair = sorted([sym1, sym2])
    return f"{pair[0]}{bond_symbol}{pair[1]}"


def bond_to_zeroth_label(mol: Chem.Mol, begin_atom: int, end_atom: int) -> str:
    """Bond-order-agnostic label: C-N, C=N, C#N all map to 'C~N'.

    Atom symbols are sorted alphabetically for canonical ordering.

    Args:
        mol: Parent RDKit molecule.
        begin_atom: Begin atom index.
        end_atom: End atom index.

    Returns:
        Label like ``'C~N'`` regardless of bond order.
    """
    sym1 = mol.GetAtomWithIdx(begin_atom).GetSymbol()
    sym2 = mol.GetAtomWithIdx(end_atom).GetSymbol()
    pair = sorted([sym1, sym2])
    return f"{pair[0]}~{pair[1]}"


def get_intensity_at_mz(
    mz_bins: np.ndarray, intensities: np.ndarray, mz: float
) -> float:
    """Return predicted intensity at the bin closest to the given m/z.

    Args:
        mz_bins: Array of m/z bin centres.
        intensities: Predicted intensity array aligned with mz_bins.
        mz: Target m/z value.

    Returns:
        Predicted intensity (float).
    """
    idx = int(np.argmin(np.abs(mz_bins - mz)))
    return float(intensities[idx])


def extract_bond_breaking_events(result: Dict[str, Any]) -> pd.DataFrame:
    """Extract all bond-breaking events from a single prediction result.

    Args:
        result: Output dict from EIMSPredictor.predict_from_smiles(), containing
            keys 'smiles', 'mz_bins', 'intensities', and 'fragments'.

    Returns:
        DataFrame with columns: smiles, mz, intensity, bond_label,
        begin_atom, end_atom.
    """
    smiles = result["smiles"]
    mz_bins = result["mz_bins"]
    intensities = result["intensities"]
    fragments = result["fragments"]

    records = []
    for mz, frag_info in fragments.items():
        mol = frag_info["structure"]
        fragment_atoms = frag_info["highlights"]["atoms"]
        intensity = get_intensity_at_mz(mz_bins, intensities, mz)

        for bond in get_broken_bonds(mol, fragment_atoms):
            label = bond_to_label(
                mol, bond["begin_atom"], bond["end_atom"], bond["bond_type"]
            )
            zeroth = bond_to_zeroth_label(
                mol, bond["begin_atom"], bond["end_atom"]
            )
            records.append(
                {
                    "smiles": smiles,
                    "mz": mz,
                    "intensity": intensity,
                    "bond_label": label,
                    "zeroth_label": zeroth,
                    "begin_atom": bond["begin_atom"],
                    "end_atom": bond["end_atom"],
                }
            )

    return pd.DataFrame(
        records,
        columns=[
            "smiles",
            "mz",
            "intensity",
            "bond_label",
            "zeroth_label",
            "begin_atom",
            "end_atom",
        ],
    )


def aggregate_bond_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate bond-breaking events across all molecules.

    Args:
        df: Concatenated DataFrames from extract_bond_breaking_events().

    Returns:
        DataFrame sorted by count (descending) with columns:
        bond_label, count, total_intensity, mean_intensity.
    """
    if df.empty:
        return pd.DataFrame(
            columns=[
                "bond_label",
                "count",
                "total_intensity",
                "mean_intensity",
            ]
        )

    return (
        df.groupby("bond_label")
        .agg(
            count=("bond_label", "count"),
            total_intensity=("intensity", "sum"),
            mean_intensity=("intensity", "mean"),
        )
        .reset_index()
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )


def aggregate_zeroth_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate bond-breaking events using bond-order-agnostic zeroth labels.

    C-N, C=N, and C#N are all counted together as C~N.

    Args:
        df: DataFrame from :func:`extract_bond_breaking_events` (must have
            ``zeroth_label`` column).

    Returns:
        DataFrame sorted by count (descending) with columns:
        bond_label, count, total_intensity, mean_intensity.
    """
    tmp = df.assign(bond_label=df["zeroth_label"])
    return aggregate_bond_patterns(tmp)


def count_zeroth_occurrences(results: Dict[str, Any]) -> Dict[str, int]:
    """Count how many times each zeroth-label bond occurs in parent structures.

    Bond order is ignored: C-N, C=N, C#N all increment the 'C~N' counter.

    Args:
        results: Dict mapping SMILES -> prediction result.

    Returns:
        Dict mapping zeroth label (e.g. ``'C~N'``) to occurrence count.
    """
    from collections import defaultdict

    counts: Dict[str, int] = defaultdict(int)
    for result in results.values():
        frags = result.get("fragments", {})
        if not frags:
            continue
        mol = next(iter(frags.values()))["structure"]
        for bond in mol.GetBonds():
            label = bond_to_zeroth_label(
                mol, bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            )
            counts[label] += 1
    return dict(counts)


def compute_zeroth_scores(
    events_df: pd.DataFrame, results: Dict[str, Any]
) -> Dict[str, Dict[str, float]]:
    """P-score and related metrics for zeroth-sphere (bond-order-agnostic)
    labels.

    Uses the same five metrics as :func:`compute_bond_scores` but groups C-N,
    C=N, and C#N together under 'C~N'.

    Args:
        events_df: DataFrame from :func:`extract_bond_breaking_events`.
        results: Dict mapping SMILES -> prediction result.

    Returns:
        Same dict structure as :func:`compute_bond_scores`.
    """
    return _compute_scores_from_events(
        events_df, "zeroth_label", count_zeroth_occurrences(results)
    )


def plot_zeroth_breaking_stats(
    agg_df: pd.DataFrame,
    top_n: int = 20,
    save_path: Optional[str] = None,
):
    """Plot zeroth-sphere bond-breaking statistics (bond-order agnostic).

    C-N, C=N, and C#N are merged into C~N before plotting.  Delegates to
    :func:`plot_bond_breaking_stats`.

    Args:
        agg_df: Aggregated DataFrame from :func:`aggregate_zeroth_patterns`.
        top_n: Number of most common bond types to display.
        save_path: Base path (without extension) for .svg / .png output.

    Returns:
        matplotlib Figure.
    """
    return plot_bond_breaking_stats(agg_df, top_n=top_n, save_path=save_path)


def get_atom_env_label(mol: Chem.Mol, atom_idx: int) -> str:
    """Return a coordination-sphere label for an atom, e.g. 'C.sp3', 'N.sp2'.

    The label encodes the atom symbol and its hybridization state, giving a
    concise descriptor of the local chemical environment around each end of a
    broken bond.

    Args:
        mol: Parent RDKit molecule.
        atom_idx: Atom index.

    Returns:
        Label string of the form ``"{Symbol}.{hybridization}"``.
    """
    atom = mol.GetAtomWithIdx(atom_idx)
    symbol = atom.GetSymbol()
    hyb_map = {
        Chem.rdchem.HybridizationType.SP3: "sp3",
        Chem.rdchem.HybridizationType.SP2: "sp2",
        Chem.rdchem.HybridizationType.SP: "sp",
        Chem.rdchem.HybridizationType.S: "s",
    }
    return f"{symbol}.{hyb_map.get(atom.GetHybridization(), 'other')}"


_BOND_TYPE_LABELS = {
    Chem.rdchem.BondType.SINGLE: "single",
    Chem.rdchem.BondType.DOUBLE: "double",
    Chem.rdchem.BondType.TRIPLE: "triple",
    Chem.rdchem.BondType.AROMATIC: "aromatic",
}


def _parse_bond_type_from_label(bond_label: str) -> str:
    """Infer bond type string from a bond label like 'C-N', 'C=O', 'C#N'."""
    if "=" in bond_label:
        return "double"
    if "#" in bond_label:
        return "triple"
    if ":" in bond_label:
        return "aromatic"
    return "single"


def enrich_bond_events(
    events_df: pd.DataFrame, results: Dict[str, Any]
) -> pd.DataFrame:
    """Enrich a bond-breaking events DataFrame with coordination-sphere labels.

    For each event, adds:
    - ``begin_env``: hybridization-resolved label for the begin atom (e.g. 'C.sp3').
    - ``end_env``: hybridization-resolved label for the end atom.
    - ``bond_type_str``: bond type as a string ('single', 'double', …).

    The mol object is taken from ``results`` rather than re-parsed from SMILES, so
    atom indices match exactly what the fragment engine produced.

    Args:
        events_df: DataFrame from :func:`extract_bond_breaking_events`.
        results: Dict mapping SMILES -> prediction result (output of
            ``EIMSPredictor.predict_from_smiles``).

    Returns:
        Copy of *events_df* with three additional columns.
    """
    # Pull the parent mol for each SMILES from the first available fragment.
    # All fragments for a molecule share the same parent mol object.
    mol_cache: Dict[str, Chem.Mol] = {}
    for smiles, result in results.items():
        fragments = result.get("fragments", {})
        if fragments:
            mol_cache[smiles] = next(iter(fragments.values()))["structure"]

    env_label_cache: Dict[tuple, str] = {}

    def _env_label(smiles: str, mol: Chem.Mol, atom_idx: int) -> str:
        key = (smiles, atom_idx)
        if key not in env_label_cache:
            env_label_cache[key] = get_atom_env_label(mol, atom_idx)
        return env_label_cache[key]

    keep_mask = []
    begin_envs = []
    end_envs = []
    bond_type_strs = []
    for row in events_df.itertuples(index=False):
        mol = mol_cache.get(row.smiles)
        if mol is None:
            keep_mask.append(False)
            continue
        keep_mask.append(True)
        begin_envs.append(_env_label(row.smiles, mol, int(row.begin_atom)))
        end_envs.append(_env_label(row.smiles, mol, int(row.end_atom)))
        bond_type_strs.append(_parse_bond_type_from_label(row.bond_label))

    out = events_df[keep_mask].reset_index(drop=True).copy()
    out["begin_env"] = begin_envs
    out["end_env"] = end_envs
    out["bond_type_str"] = bond_type_strs
    return out


def build_environment_matrix(
    enriched_df: pd.DataFrame,
    metric: str = "count",
    bond_type: Optional[str] = None,
) -> pd.DataFrame:
    """Build a symmetric co-occurrence matrix of coordination-sphere
    environments.

    Each broken bond contributes to the cell (env_A, env_B) and, symmetrically,
    (env_B, env_A). Diagonal entries (same environment on both sides) are counted
    once.

    Args:
        enriched_df: DataFrame from :func:`enrich_bond_events`.
        metric: ``"count"`` for raw event counts, ``"total_intensity"`` for
            intensity-weighted sums.
        bond_type: If given, filter to this bond type ('single', 'double', …).

    Returns:
        Square symmetric DataFrame indexed and columned by environment labels,
        sorted by total marginal count (most common first).
    """
    df = enriched_df.copy()
    if bond_type is not None:
        df = df[df["bond_type_str"] == bond_type]
    if df.empty:
        return pd.DataFrame()

    # Canonical pair ordering for aggregation
    df["env1"] = df.apply(lambda r: min(r["begin_env"], r["end_env"]), axis=1)
    df["env2"] = df.apply(lambda r: max(r["begin_env"], r["end_env"]), axis=1)

    if metric == "count":
        agg = df.groupby(["env1", "env2"]).size().reset_index(name="value")
    else:
        agg = (
            df.groupby(["env1", "env2"])["intensity"]
            .sum()
            .reset_index(name="value")
        )

    all_envs = sorted(set(agg["env1"]).union(agg["env2"]))
    matrix = pd.DataFrame(0.0, index=all_envs, columns=all_envs)
    for _, row in agg.iterrows():
        matrix.loc[row["env1"], row["env2"]] += row["value"]
        if row["env1"] != row["env2"]:
            matrix.loc[row["env2"], row["env1"]] += row["value"]

    # Sort rows/cols by marginal total (most common first)
    order = matrix.sum(axis=1).sort_values(ascending=False).index
    matrix = matrix.loc[order, order]
    return matrix


def plot_bond_environment_heatmap(
    enriched_df: pd.DataFrame,
    metric: str = "count",
    top_n_envs: int = 12,
    save_path: Optional[str] = None,
):
    """Plot coordination-sphere heatmaps, one panel per bond type.

    Each cell shows how often (or how intensely) two coordination-sphere
    environments are found at either side of a broken bond.

    Args:
        enriched_df: DataFrame from :func:`enrich_bond_events`.
        metric: ``"count"`` or ``"total_intensity"``.
        top_n_envs: Keep only the *top_n_envs* most common environment labels.
        save_path: Base path (no extension) to save .svg and .png outputs.

    Returns:
        matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    from icicle.utils.visualization import cmap

    set_style("manuscript")

    bond_types = sorted(enriched_df["bond_type_str"].unique())
    n = len(bond_types)
    size = max(3.0, 0.45 * top_n_envs)
    fig, axes = plt.subplots(1, n, figsize=(size * n, size))
    if n == 1:
        axes = [axes]

    label_str = metric.replace("_", " ").capitalize()

    for ax, bt in zip(axes, bond_types):
        matrix = build_environment_matrix(
            enriched_df, metric=metric, bond_type=bt
        )
        if matrix.empty:
            ax.set_visible(False)
            continue

        # Trim to top environments
        top_envs = matrix.sum(axis=1).nlargest(top_n_envs).index
        m = matrix.loc[top_envs, top_envs]

        im = ax.imshow(m.values, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(m.columns)))
        ax.set_xticklabels(m.columns, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(m.index)))
        ax.set_yticklabels(m.index, fontsize=8)
        ax.set_title(f"{bt.capitalize()} bonds", fontweight="bold")

        # Annotate non-zero cells
        vmax = m.values.max()
        for i in range(len(m.index)):
            for j in range(len(m.columns)):
                val = m.values[i, j]
                if val > 0:
                    text_color = "white" if val > 0.6 * vmax else "black"
                    fmt = f"{val:.0f}" if metric == "count" else f"{val:.2f}"
                    ax.text(
                        j,
                        i,
                        fmt,
                        ha="center",
                        va="center",
                        fontsize=7,
                        color=text_color,
                    )

        fig.colorbar(im, ax=ax, label=label_str, shrink=0.8)

    plt.tight_layout()

    if save_path:
        fig.savefig(f"{save_path}.svg", bbox_inches="tight")
        fig.savefig(f"{save_path}.png", bbox_inches="tight", dpi=300)

    return fig


def plot_bond_breaking_stats(
    agg_df: pd.DataFrame,
    top_n: int = 20,
    save_path: Optional[str] = None,
):
    """Plot bond-breaking pattern statistics as horizontal bar charts.

    Args:
        agg_df: Aggregated DataFrame from aggregate_bond_patterns().
        top_n: Number of most common bond types to display.
        save_path: Base path (without extension) to save .svg and .png outputs.
            If None, figures are only shown in-notebook.

    Returns:
        matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    set_style("manuscript")
    palette = get_palette()

    top = agg_df.head(top_n).iloc[::-1]  # reverse for bottom-to-top bar order

    fig, axes = plt.subplots(1, 2, figsize=(6.5, max(2.5, 0.3 * len(top))))

    axes[0].barh(top["bond_label"], top["count"], color=palette[0])
    axes[0].set_xlabel("Count", fontweight="bold")
    axes[0].set_ylabel("Bond type", fontweight="bold")
    axes[0].grid(True, linestyle="--", alpha=0.6, axis="x")

    axes[1].barh(top["bond_label"], top["total_intensity"], color=palette[2])
    axes[1].set_xlabel("Total predicted intensity", fontweight="bold")
    axes[1].grid(True, linestyle="--", alpha=0.6, axis="x")

    plt.tight_layout()

    if save_path:
        fig.savefig(f"{save_path}.svg", bbox_inches="tight")
        fig.savefig(f"{save_path}.png", bbox_inches="tight", dpi=300)

    return fig


# Bond scoring: P-score and related metrics


def count_bond_occurrences(results: Dict[str, Any]) -> Dict[str, int]:
    """Count how many times each bond label occurs across all unique parent
    molecules.

    Used as the denominator when computing occurrence-normalized scores such as
    the preferential score (P-score).

    Args:
        results: Dict mapping SMILES -> prediction result, as returned by the
            predictor.  The parent mol is taken from the first fragment entry.

    Returns:
        Dict mapping bond label (e.g. ``'C-N'``) to total count across all
        unique parent structures.
    """
    from collections import defaultdict

    counts: Dict[str, int] = defaultdict(int)
    for result in results.values():
        frags = result.get("fragments", {})
        if not frags:
            continue
        mol = next(iter(frags.values()))["structure"]
        for bond in mol.GetBonds():
            label = bond_to_label(
                mol,
                bond.GetBeginAtomIdx(),
                bond.GetEndAtomIdx(),
                bond.GetBondType(),
            )
            counts[label] += 1
    return dict(counts)


def _compute_scores_from_events(
    events_df: pd.DataFrame,
    label_col: str,
    bond_occurrences: Dict[str, int],
) -> Dict[str, Dict[str, float]]:
    """Shared scoring logic for any bond-label column.

    Args:
        events_df: Events DataFrame with at least *label_col* and ``intensity``.
        label_col: Column to group by (e.g. ``'bond_label'`` or ``'_s2_pair'``).
        bond_occurrences: Occurrence counts for each label in parent structures.

    Returns:
        Dict with keys ``absolute_counts``, ``intensity_score``,
        ``frequency_score``, ``preferential_score``, ``i_times_f_score``,
        ``bond_occurrences``.
    """
    break_counts = events_df.groupby(label_col).size().to_dict()
    break_intensities = (
        events_df.groupby(label_col)["intensity"].sum().to_dict()
    )

    total_breaks = sum(break_counts.values()) or 1
    total_intensity = sum(break_intensities.values()) or 1
    total_occurrences = sum(bond_occurrences.values()) or 1

    intensity_score = {
        b: (v / total_intensity) * 100 for b, v in break_intensities.items()
    }
    frequency_score = {
        b: (break_counts[b] / total_breaks)
        * (total_occurrences / bond_occurrences.get(b, 1))
        for b in break_counts
    }
    norm_intensities = {
        b: v / bond_occurrences.get(b, 1) for b, v in break_intensities.items()
    }
    total_norm = sum(norm_intensities.values()) or 1
    preferential_score = {
        b: (v / total_norm) * 100 for b, v in norm_intensities.items()
    }
    i_times_f_score = {
        b: intensity_score.get(b, 0) * frequency_score.get(b, 0)
        for b in break_counts
    }
    return {
        "absolute_counts": break_counts,
        "intensity_score": intensity_score,
        "frequency_score": frequency_score,
        "preferential_score": preferential_score,
        "i_times_f_score": i_times_f_score,
        "bond_occurrences": bond_occurrences,
    }


def compute_bond_scores(
    events_df: pd.DataFrame, results: Dict[str, Any]
) -> Dict[str, Dict[str, float]]:
    """Compute bond-breaking scoring metrics, including the P-score.

    Replicates the five metrics from the legacy ``analyze_bonds`` function, adapted
    for GC-EI spectra where no collision energy is available.  See
    :func:`_compute_scores_from_events` for metric definitions.

    Args:
        events_df: Bond-breaking events from :func:`extract_bond_breaking_events`.
        results: Dict mapping SMILES -> prediction result.

    Returns:
        Dict with keys ``absolute_counts``, ``intensity_score``,
        ``frequency_score``, ``preferential_score``, ``i_times_f_score``,
        ``bond_occurrences``.
    """
    return _compute_scores_from_events(
        events_df, "bond_label", count_bond_occurrences(results)
    )


def count_sphere2_occurrences(results: Dict[str, Any]) -> Dict[str, int]:
    """Count how many times each second-sphere bond-pair label occurs in parent
    structures.

    Uses the same canonical pair format as :func:`compute_sphere2_scores`:
    ``"min_label — max_label"``.

    Args:
        results: Dict mapping SMILES -> prediction result.

    Returns:
        Dict mapping sphere2 pair label -> occurrence count.
    """
    from collections import defaultdict

    counts: Dict[str, int] = defaultdict(int)
    for result in results.values():
        frags = result.get("fragments", {})
        if not frags:
            continue
        mol = next(iter(frags.values()))["structure"]
        for bond in mol.GetBonds():
            begin = bond.GetBeginAtomIdx()
            end = bond.GetEndAtomIdx()
            b_lbl = get_second_sphere_label(mol, begin, end)
            e_lbl = get_second_sphere_label(mol, end, begin)
            pair = f"{min(b_lbl, e_lbl)} — {max(b_lbl, e_lbl)}"
            counts[pair] += 1
    return dict(counts)


def compute_sphere2_scores(
    sphere2_df: pd.DataFrame, results: Dict[str, Any]
) -> Dict[str, Dict[str, float]]:
    """P-score and related metrics for second-sphere bond environments.

    Applies the same five metrics as :func:`compute_bond_scores` but uses the
    richer ``'Symbol.hyb(neighbors)'`` labels on both bond ends.  The P-score
    normalises by how many bonds of that *environment* exist in the parent
    structures, not just by atom-type bond counts.

    Args:
        sphere2_df: DataFrame from :func:`enrich_with_second_sphere`.
        results: Dict mapping SMILES -> prediction result.

    Returns:
        Same dict structure as :func:`compute_bond_scores`.
    """
    df = sphere2_df.copy()
    df["_s2_pair"] = df.apply(
        lambda r: f"{min(r['begin_sphere2'], r['end_sphere2'])} — {max(r['begin_sphere2'], r['end_sphere2'])}",
        axis=1,
    )
    return _compute_scores_from_events(
        df, "_s2_pair", count_sphere2_occurrences(results)
    )


_METRIC_LABELS = {
    "preferential_score": "P-score (%)",
    "intensity_score": "Intensity\nscore (%)",
    "frequency_score": "Frequency score",
    "i_times_f_score": "I × F score",
    "absolute_counts": "Absolute\ncounts (10\u00b3)",
    "bond_occurrences": "Bond occurrences",
}


def plot_scores_heatmap(
    scores_dict: Dict[str, Dict[str, float]],
    metrics: Optional[List[str]] = None,
    row_normalize: bool = True,
    bond_subset: Optional[List[str]] = None,
    top_n: Optional[int] = None,
    save_path: Optional[str] = None,
):
    """Heatmap of bond-scoring metrics with one row per metric.

    Bond types are ordered by P-score (preferential_score) descending so the
    most informative columns appear first.  With ``row_normalize=True`` (default)
    each row is scaled to [0, 1] so metrics on different absolute scales are
    visually comparable; raw values are still printed in each cell.

    Args:
        scores_dict: Output of :func:`compute_bond_scores`.
        metrics: Ordered list of metric keys to display.  Defaults to the four
            most informative ones.
        row_normalize: If True, normalize each row independently to [0, 1]
            before mapping to colour.
        bond_subset: If given, only show these bond-type columns (order
            preserved from P-score ranking).  Useful for slide figures with
            fewer columns.
        top_n: If given, keep only the top-N bond types by P-score rank.
            Applied after ``bond_subset`` filtering.
        save_path: Base path (no extension) for .svg / .png output.

    Returns:
        matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    from icicle.utils.visualization import cmap

    set_style("manuscript")

    if metrics is None:
        metrics = [
            "preferential_score",
            "intensity_score",
            "frequency_score",
            "i_times_f_score",
            "absolute_counts",
        ]

    # Order bond types by P-score (or first available metric) descending
    ref = scores_dict.get("preferential_score", scores_dict[metrics[0]])
    bond_order = sorted(ref, key=ref.get, reverse=True)
    extra = [
        b
        for b in scores_dict.get("bond_occurrences", {})
        if b not in bond_order
    ]
    bond_order = bond_order + sorted(extra)

    if bond_subset is not None:
        bond_order = [b for b in bond_order if b in bond_subset]

    if top_n is not None:
        bond_order = bond_order[:top_n]

    raw = pd.DataFrame(
        {b: [scores_dict[m].get(b, 0.0) for m in metrics] for b in bond_order},
        index=[_METRIC_LABELS.get(m, m) for m in metrics],
    )

    plot_data = (
        raw.div(raw.max(axis=1).replace(0, 1), axis=0)
        if row_normalize
        else raw
    )

    n_bonds = len(bond_order)
    n_metrics = len(metrics)
    row_h = 0.2
    # ponytail: wider per-column spacing (0.55in/bond) than the old 0.4in so
    # 20-column heatmaps aren't crowded; make_fig keeps the plot-standards
    # axes-area contract instead of a raw plt.subplots(figsize=...) call.
    fig, axes = make_fig(
        size=(max(6.8, 0.55 * n_bonds), row_h * n_metrics),
        nrows=n_metrics,
        ncols=1,
        sharex=True,
    )
    if n_metrics == 1:
        axes = [axes]

    for i, (ax, metric, row_label) in enumerate(zip(axes, metrics, raw.index)):
        vals_plot = plot_data.loc[row_label].values.reshape(1, -1)
        vals_raw = raw.loc[row_label].values
        ax.imshow(vals_plot, cmap=cmap, aspect="auto", vmin=0, vmax=1)
        ax.set_yticks([0])
        ax.set_yticklabels([row_label], fontsize=8)
        ax.tick_params(
            left=False, bottom=False, top=False, right=False, length=0
        )
        for spine in ax.spines.values():
            spine.set_visible(False)

        for j, (vraw, vplot) in enumerate(zip(vals_raw, vals_plot[0])):
            if metric in ("absolute_counts", "bond_occurrences"):
                fmt = f"{vraw / 1000:.1f}"
            else:
                fmt = f"{vraw:.1f}"
            color = "white" if vplot > 0.55 else "black"
            ax.text(
                j, 0, fmt, ha="center", va="center", fontsize=7, color=color
            )

        # cb = fig.colorbar(
        #     im,
        #     ax=ax,
        #     # shrink=0.7,
        #     pad=0.01,
        #     label="norm."
        #     if row_normalize
        #     else _METRIC_LABELS.get(metric, metric),
        # )
        # cb.ax.tick_params(length=0)

    axes[-1].set_xticks(range(n_bonds))
    axes[-1].set_xticklabels(bond_order, rotation=45, ha="right")
    axes[-1].tick_params(bottom=False, length=0)
    # axes[0].set_title(
    #     "Bond scoring metrics (columns ordered by P-score)",
    #     fontweight="bold",
    #     fontsize=9,
    # )

    plt.subplots_adjust(hspace=0)
    if save_path:
        fig.savefig(f"{save_path}.svg", bbox_inches="tight")
        fig.savefig(f"{save_path}.png", bbox_inches="tight", dpi=300)
    return fig


def plot_pscore_bar(
    scores_dict: Dict[str, Dict[str, float]],
    n_top: int = 3,
    n_bottom: int = 3,
    n_middle: int = 2,
    save_path: Optional[str] = None,
) -> "plt.Figure":
    """Slide-friendly horizontal bar chart showing P-score for a subset of bond
    types.

    Selects the ``n_top`` most abundant, ``n_bottom`` least abundant, and
    ``n_middle`` middle bond types (by P-score), then plots only P-score so the
    chart is uncluttered enough for a presentation slide.

    Args:
        scores_dict: Output of :func:`compute_bond_scores`.
        n_top: Number of highest-P-score bonds to include.
        n_bottom: Number of lowest-P-score bonds to include.
        n_middle: Number of middle-ranked bonds to include.
        save_path: Base path (no extension) for .svg / .png output.

    Returns:
        matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    set_style("manuscript")
    palette = get_palette()

    p = scores_dict.get("preferential_score", {})
    bond_order = sorted(p, key=p.get, reverse=True)  # high → low

    n = len(bond_order)
    mid_start = n // 2 - n_middle // 2
    mid_indices = list(range(mid_start, mid_start + n_middle))

    top_bonds = bond_order[:n_top]
    bottom_bonds = bond_order[n - n_bottom :]
    middle_bonds = [bond_order[i] for i in mid_indices]

    # Keep original P-score order (top → middle → bottom), deduplicate
    seen: set = set()
    selected: list = []
    for b in bond_order:
        if b in top_bonds or b in middle_bonds or b in bottom_bonds:
            if b not in seen:
                selected.append(b)
                seen.add(b)

    values = [p[b] for b in selected]

    # Color coding: top = palette[0], middle = palette[2], bottom = palette[4]
    colors = []
    for b in selected:
        if b in top_bonds:
            colors.append(palette[0])
        elif b in middle_bonds:
            colors.append(palette[2])
        else:
            colors.append(palette[4])

    fig, ax = plt.subplots(figsize=(3.3, 0.45 * len(selected) + 0.8))
    y = range(len(selected))
    bars = ax.barh(list(y), values, color=colors, edgecolor="none", height=0.6)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + 0.3,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%",
            va="center",
            ha="left",
            fontsize=8,
        )

    ax.set_yticks(list(y))
    ax.set_yticklabels(selected, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("P-score (%)", fontweight="bold")
    ax.set_title("Bond breaking preference (P-score)", fontweight="bold")
    ax.grid(True, axis="x", linestyle="--", alpha=0.6)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=palette[0], label=f"Top {n_top}"),
        Patch(facecolor=palette[2], label=f"Middle {n_middle}"),
        Patch(facecolor=palette[4], label=f"Bottom {n_bottom}"),
    ]
    ax.legend(
        handles=legend_elements, frameon=False, fontsize=7, loc="lower right"
    )

    plt.tight_layout()
    if save_path:
        fig.savefig(f"{save_path}.svg", bbox_inches="tight")
        fig.savefig(f"{save_path}.png", bbox_inches="tight", dpi=300)
    return fig


# Loading predictions from ICICLE eval HDF5 files


def _load_one_molecule(
    args: tuple,
) -> Optional[tuple]:
    """Worker for :func:`load_results_from_hdf5`: build one molecule's result
    dict.

    Args:
        args: Tuple of (smiles, mz_bins, intensities, threshold, max_nodes).

    Returns:
        Tuple of (smiles, result_dict), or None if no fragments passed threshold.
    """
    from icicle.data.fragmentation_engine import (
        FragmentationParams,
        FragmentEngine,
    )

    smiles, mz_bins, intensities, threshold, max_nodes = args

    # ponytail: fragment enumeration is the combinatorial bottleneck and is
    # identical across notebook re-runs for the same (smiles, max_nodes);
    # cache it on disk keyed by smiles so repeated runs skip re-fragmenting.
    cache_key = hashlib.sha1(f"{smiles}|{max_nodes}".encode()).hexdigest()
    cache_path = os.path.join(_FRAGMENT_CACHE_DIR, f"{cache_key}.joblib")

    if os.path.exists(cache_path):
        frag_to_entry, engine = joblib.load(cache_path)
    else:
        try:
            engine = FragmentEngine(
                mol_str=smiles,
                params=FragmentationParams(
                    max_tree_depth=3,
                    max_broken_bonds=6,
                    num_h_shifts=1,
                    detect_isotope_patterns=False,
                ),
            )
            engine.generate_fragments()
        except Exception:
            return None

        if len(engine.frag_to_entry) > max_nodes:
            top = sorted(
                engine.frag_to_entry.items(),
                key=lambda kv: (kv[1].tree_depth, kv[1].max_broken),
            )[:max_nodes]
            engine.frag_to_entry = dict(top)

        frag_to_entry = engine.frag_to_entry
        os.makedirs(_FRAGMENT_CACHE_DIR, exist_ok=True)
        joblib.dump((frag_to_entry, engine), cache_path)

    fragments: Dict[float, Any] = {}
    for frag_hash, frag_entry in frag_to_entry.items():
        base_mass = frag_entry.base_mass
        intensity = get_intensity_at_mz(mz_bins, intensities, base_mass)
        if intensity < threshold:
            continue
        try:
            draw_info = engine.get_draw_dict(frag_entry.frag)
            fragments[float(base_mass)] = {
                "structure": draw_info.mol,
                "form": frag_entry.form,
                "highlights": {
                    "atoms": draw_info.hatoms,
                    "bonds": draw_info.hbonds,
                },
                "frag_hash": frag_hash,
            }
        except Exception:
            continue

    if not fragments:
        return None

    return (
        smiles,
        {
            "smiles": smiles,
            "mz_bins": mz_bins,
            "intensities": intensities,
            "num_fragments": len(fragments),
            "fragments": fragments,
        },
    )


def load_results_from_hdf5(
    hdf5_path: str,
    metadata_path: str,
    max_molecules: Optional[int] = None,
    threshold: float = 0.01,
    max_nodes: int = 50,
    num_workers: Optional[int] = None,
) -> Dict[str, Any]:
    """Load predictions from an ICICLE eval HDF5 file into the results format.

    Builds a dict compatible with all bond-breaking analysis functions, using
    pre-computed predicted intensities from the HDF5 file instead of running
    the model.  Fragments are re-enumerated via :class:`FragmentEngine`; SMILES
    are looked up from the dataset ``metadata.tsv``.

    HDF5 keys are full InChIKeys; the first 14 characters are matched against
    the ``inchikey14`` column in the metadata file.

    Args:
        hdf5_path: Path to ``all_evaluation_spectra.hdf5`` (eval output).
        metadata_path: Path to ``metadata.tsv`` for the same dataset.
            Must have ``inchikey14`` and ``smiles`` columns.
        max_molecules: If set, process at most this many entries.
        threshold: Minimum predicted intensity for a fragment to be included.
        max_nodes: Maximum number of fragments per molecule.
        num_workers: Number of worker processes for parallel fragment
            enumeration. Defaults to all available CPUs.

    Returns:
        Dict mapping SMILES -> result dict with keys ``smiles``, ``mz_bins``,
        ``intensities``, ``num_fragments``, ``fragments`` — identical in
        structure to :meth:`EIMSPredictorFromFullEnumeration.predict_from_smiles`
        output.
    """
    import h5py

    meta = pd.read_csv(
        metadata_path, sep="\t", usecols=["inchikey", "standardized_smiles"]
    )
    ik14_to_smiles: Dict[str, str] = dict(
        zip(
            meta["inchikey"].astype(str),
            meta["standardized_smiles"].astype(str),
        )
    )
    # make the inchikeys ik14 by truncating to the first 14 chars
    ik14_to_smiles = {k[:14]: v for k, v in ik14_to_smiles.items()}

    tasks = []
    with h5py.File(hdf5_path, "r") as f:
        keys = list(f.keys())
        if max_molecules is not None:
            keys = keys[:max_molecules]

        for hdf5_key in keys:
            ik14 = hdf5_key[:14]
            smiles = ik14_to_smiles.get(ik14)
            if smiles is None:
                continue

            mz_bins = f[hdf5_key]["predicted_mz_bins"][()].astype(np.float32)
            intensities = f[hdf5_key]["predicted_intensities"][()].astype(
                np.float32
            )
            tasks.append((smiles, mz_bins, intensities, threshold, max_nodes))

    results: Dict[str, Any] = {}
    with mp.Pool(processes=num_workers) as pool:
        for out in tqdm(
            pool.imap_unordered(_load_one_molecule, tasks),
            total=len(tasks),
            desc="Loading + fragmenting molecules",
        ):
            if out is not None:
                smiles, result = out
                results[smiles] = result

    return results


# Second coordination sphere — neighbors beyond hybridization


def get_second_sphere_label(
    mol: Chem.Mol, atom_idx: int, exclude_idx: int
) -> str:
    """Return a second-sphere label including the atom's heavy-atom neighbors.

    Extends :func:`get_atom_env_label` by appending the sorted symbols of all
    heavy-atom neighbors (excluding the bond partner), e.g. ``'C.sp3(Cl,Br)'``
    or ``'N.sp3'`` when there are no other heavy neighbors.

    Args:
        mol: Parent RDKit molecule.
        atom_idx: Atom index of interest.
        exclude_idx: Bond-partner atom index to exclude from the neighbor list.

    Returns:
        Label string like ``'C.sp3(Cl,Br)'``.
    """
    atom = mol.GetAtomWithIdx(atom_idx)
    hyb_map = {
        Chem.rdchem.HybridizationType.SP3: "sp3",
        Chem.rdchem.HybridizationType.SP2: "sp2",
        Chem.rdchem.HybridizationType.SP: "sp",
        Chem.rdchem.HybridizationType.S: "s",
    }
    hyb = hyb_map.get(atom.GetHybridization(), "other")
    heavy_nbrs = sorted(
        mol.GetAtomWithIdx(n.GetIdx()).GetSymbol()
        for n in atom.GetNeighbors()
        if n.GetIdx() != exclude_idx and n.GetAtomicNum() > 1
    )
    nbr_str = f"({','.join(heavy_nbrs)})" if heavy_nbrs else ""
    return f"{atom.GetSymbol()}.{hyb}{nbr_str}"


def get_bond_neighborhood(
    mol: Chem.Mol, begin_atom: int, end_atom: int
) -> tuple:
    """Build a small RDKit mol covering the 2-hop neighborhood of a bond.

    Includes both bond-end atoms and all their immediate neighbors, plus every
    bond between atoms in that set.

    Args:
        mol: Parent RDKit molecule.
        begin_atom: Begin atom index of the broken bond.
        end_atom: End atom index of the broken bond.

    Returns:
        ``(submol, [begin_new_idx, end_new_idx])`` — the neighborhood mol and
        the new atom indices of the two bond-end atoms within it.
    """
    atom_indices = {begin_atom, end_atom}
    for nbr in mol.GetAtomWithIdx(begin_atom).GetNeighbors():
        atom_indices.add(nbr.GetIdx())
    for nbr in mol.GetAtomWithIdx(end_atom).GetNeighbors():
        atom_indices.add(nbr.GetIdx())

    atom_list = sorted(atom_indices)
    old_to_new = {old: i for i, old in enumerate(atom_list)}

    rw = Chem.RWMol()
    for old_idx in atom_list:
        a = mol.GetAtomWithIdx(old_idx)
        new_a = Chem.Atom(a.GetAtomicNum())
        new_a.SetNoImplicit(False)
        rw.AddAtom(new_a)

    for bond in mol.GetBonds():
        b, e = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if b in atom_indices and e in atom_indices:
            rw.AddBond(old_to_new[b], old_to_new[e], bond.GetBondType())

    Chem.SanitizeMol(rw)
    return rw.GetMol(), [old_to_new[begin_atom], old_to_new[end_atom]]


def get_bond_env_smarts(
    mol: Chem.Mol, begin_atom: int, end_atom: int
) -> Optional[str]:
    """Return a canonical SMILES key for the 2-hop bond environment.

    Bond-end atoms are annotated with atom-map numbers 1 and 2 so the key
    encodes which atoms sit at the broken bond.  Canonical SMILES is used for
    stable grouping across events.

    Args:
        mol: Parent RDKit molecule.
        begin_atom: Begin atom index.
        end_atom: End atom index.

    Returns:
        Canonical SMILES string, or ``None`` if sanitization fails.
    """
    submol, (b_new, e_new) = get_bond_neighborhood(mol, begin_atom, end_atom)
    em = Chem.RWMol(submol)
    em.GetAtomWithIdx(b_new).SetAtomMapNum(1)
    em.GetAtomWithIdx(e_new).SetAtomMapNum(2)
    return Chem.MolToSmiles(em.GetMol())


def enrich_with_second_sphere(
    enriched_df: pd.DataFrame, results: Dict[str, Any]
) -> pd.DataFrame:
    """Add second-sphere labels and bond-environment keys to an enriched events
    DataFrame.

    Adds three columns:
    - ``begin_sphere2``: second-sphere label for the begin atom, e.g. ``'C.sp3(Cl,Br)'``.
    - ``end_sphere2``: second-sphere label for the end atom.
    - ``bond_env_key``: canonical SMILES of the 2-hop neighborhood (atom-map
      annotated), used for grouping identical bond environments.

    Args:
        enriched_df: DataFrame from :func:`enrich_bond_events`.
        results: Dict mapping SMILES -> prediction result.

    Returns:
        DataFrame with three additional columns.
    """
    mol_cache: Dict[str, Chem.Mol] = {}
    for smiles, result in results.items():
        frags = result.get("fragments", {})
        if frags:
            mol_cache[smiles] = next(iter(frags.values()))["structure"]

    # ponytail: the same (smiles, begin_atom, end_atom) bond repeats across
    # fragments/mz peaks; the SMARTS/RWMol construction in get_bond_env_smarts
    # is the expensive part, so cache all three lookups per bond instead of
    # rebuilding the neighborhood submol for every duplicate row.
    sphere2_cache: Dict[tuple, str] = {}
    env_key_cache: Dict[tuple, Optional[str]] = {}

    def _sphere2(
        smiles: str, mol: Chem.Mol, atom_idx: int, exclude_idx: int
    ) -> str:
        key = (smiles, atom_idx, exclude_idx)
        if key not in sphere2_cache:
            sphere2_cache[key] = get_second_sphere_label(
                mol, atom_idx, exclude_idx
            )
        return sphere2_cache[key]

    def _env_key(
        smiles: str, mol: Chem.Mol, begin: int, end: int
    ) -> Optional[str]:
        key = (smiles, begin, end)
        if key not in env_key_cache:
            try:
                env_key_cache[key] = get_bond_env_smarts(mol, begin, end)
            except Exception:
                env_key_cache[key] = None
        return env_key_cache[key]

    keep_mask = []
    begin_sphere2s = []
    end_sphere2s = []
    bond_env_keys = []
    for row in enriched_df.itertuples(index=False):
        mol = mol_cache.get(row.smiles)
        if mol is None:
            keep_mask.append(False)
            continue
        keep_mask.append(True)
        begin, end = int(row.begin_atom), int(row.end_atom)
        begin_sphere2s.append(_sphere2(row.smiles, mol, begin, end))
        end_sphere2s.append(_sphere2(row.smiles, mol, end, begin))
        bond_env_keys.append(_env_key(row.smiles, mol, begin, end))

    out = enriched_df[keep_mask].reset_index(drop=True).copy()
    out["begin_sphere2"] = begin_sphere2s
    out["end_sphere2"] = end_sphere2s
    out["bond_env_key"] = bond_env_keys
    return out


def plot_second_sphere_heatmap(
    sphere2_df: pd.DataFrame,
    metric: str = "count",
    top_n_envs: int = 12,
    save_path: Optional[str] = None,
):
    """Heatmap of second-sphere environment co-occurrences, one panel per bond
    type.

    Axes show ``'Symbol.hyb(neighbors)'`` labels instead of bare hybridization,
    giving a richer view of the local chemical context around each broken bond.

    Args:
        sphere2_df: DataFrame from :func:`enrich_with_second_sphere`.
        metric: ``"count"`` or ``"total_intensity"``.
        top_n_envs: Keep only the *top_n_envs* most frequent environment labels.
        save_path: Base path (no extension) for .svg / .png output.

    Returns:
        matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    from icicle.utils.visualization import cmap

    set_style("manuscript")

    tmp = sphere2_df.drop(
        columns=["begin_env", "end_env"], errors="ignore"
    ).rename(columns={"begin_sphere2": "begin_env", "end_sphere2": "end_env"})
    bond_types = sorted(tmp["bond_type_str"].unique())
    n = len(bond_types)
    size = max(3.5, 0.5 * top_n_envs)
    fig, axes = plt.subplots(1, n, figsize=(size * n, size))
    if n == 1:
        axes = [axes]

    label_str = metric.replace("_", " ").capitalize()
    for ax, bt in zip(axes, bond_types):
        matrix = build_environment_matrix(tmp, metric=metric, bond_type=bt)
        if matrix.empty:
            ax.set_visible(False)
            continue
        top_envs = matrix.sum(axis=1).nlargest(top_n_envs).index
        m = matrix.loc[top_envs, top_envs]
        im = ax.imshow(m.values, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(m.columns)))
        ax.set_xticklabels(m.columns, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(m.index)))
        ax.set_yticklabels(m.index, fontsize=7)
        ax.set_title(
            f"{bt.capitalize()} bonds — 2nd sphere", fontweight="bold"
        )
        vmax = m.values.max()
        for i in range(len(m.index)):
            for j in range(len(m.columns)):
                val = m.values[i, j]
                if val > 0:
                    color = "white" if val > 0.6 * vmax else "black"
                    fmt = f"{val:.0f}" if metric == "count" else f"{val:.2f}"
                    ax.text(
                        j,
                        i,
                        fmt,
                        ha="center",
                        va="center",
                        fontsize=6,
                        color=color,
                    )
        fig.colorbar(im, ax=ax, label=label_str, shrink=0.8)

    plt.tight_layout()
    if save_path:
        fig.savefig(f"{save_path}.svg", bbox_inches="tight")
        fig.savefig(f"{save_path}.png", bbox_inches="tight", dpi=300)
    return fig


def plot_smarts_gallery(
    sphere2_df: pd.DataFrame,
    results: Dict[str, Any],
    top_n: int = 12,
    cols: int = 4,
    img_size: tuple = (280, 220),
    save_path: Optional[str] = None,
    sort_by: str = "count",
    scores_s2: Optional[Dict[str, Dict[str, float]]] = None,
):
    """Draw the most common 2-hop bond environments as a grid of molecule
    images.

    Each panel shows the substructure around a broken bond (2-hop neighborhood),
    with the two bond-end atoms highlighted.

    Args:
        sphere2_df: DataFrame from :func:`enrich_with_second_sphere`.
        results: Dict mapping SMILES -> prediction result (to recover the mol).
        top_n: Number of environments to show.
        cols: Number of columns in the grid.
        img_size: ``(width, height)`` in pixels per substructure panel.
        save_path: Base path (no extension) for .svg / .png output.
        sort_by: ``"count"`` (default) or ``"p_score"``. When ``"p_score"``,
            ``scores_s2`` must be provided.
        scores_s2: Output of :func:`compute_sphere2_scores`. Required when
            ``sort_by="p_score"``.

    Returns:
        matplotlib Figure, or ``None`` if no valid environments found.
    """
    import matplotlib.pyplot as plt
    from rdkit.Chem import AllChem, Draw

    set_style("manuscript")

    mol_cache: Dict[str, Chem.Mol] = {}
    for smiles, result in results.items():
        frags = result.get("fragments", {})
        if frags:
            mol_cache[smiles] = next(iter(frags.values()))["structure"]

    agg = (
        sphere2_df.dropna(subset=["bond_env_key"])
        .groupby("bond_env_key")
        .agg(
            count=("bond_env_key", "count"),
            total_intensity=("intensity", "sum"),
        )
        .reset_index()
    )

    if sort_by == "p_score":
        if scores_s2 is None:
            raise ValueError("scores_s2 required when sort_by='p_score'")
        p_map = scores_s2["preferential_score"]

        # bond_env_key uses "begin — end" format; p_map keys are normalized pairs
        def _lookup_pscore(key: str) -> float:
            parts = [p.strip() for p in key.split(" — ")]
            if len(parts) != 2:
                return 0.0
            norm = f"{min(parts)} — {max(parts)}"
            return p_map.get(norm, 0.0)

        agg["p_score"] = agg["bond_env_key"].apply(_lookup_pscore)
        ranked = agg.sort_values("p_score", ascending=False).head(top_n)
        sort_col = "p_score"
    else:
        ranked = agg.sort_values("count", ascending=False).head(top_n)

    panels = []
    for _, pat_row in ranked.iterrows():
        key = pat_row["bond_env_key"]
        rep = sphere2_df[sphere2_df["bond_env_key"] == key].iloc[0]
        mol = mol_cache.get(rep["smiles"])
        if mol is None:
            continue
        begin, end = int(rep["begin_atom"]), int(rep["end_atom"])
        try:
            submol, (b_new, e_new) = get_bond_neighborhood(mol, begin, end)
            AllChem.Compute2DCoords(submol)
            bond_idx = submol.GetBondBetweenAtoms(b_new, e_new).GetIdx()
            img = Draw.MolToImage(
                submol,
                size=img_size,
                highlightAtoms=[b_new, e_new],
                highlightBonds=[bond_idx],
            )
            panel_data = {
                "img": img,
                "count": int(pat_row["count"]),
                "intensity": pat_row["total_intensity"],
                "label": f"{rep['begin_sphere2']}  —  {rep['end_sphere2']}",
            }
            if sort_by == "p_score":
                panel_data["p_score"] = pat_row["p_score"]
            panels.append(panel_data)
        except Exception:
            continue

    if not panels:
        return None

    n_panels = len(panels)
    rows = (n_panels + cols - 1) // cols
    w_inch = img_size[0] / 100 * cols
    h_inch = img_size[1] / 100 * rows * 1.3
    fig, axes = plt.subplots(rows, cols, figsize=(w_inch, h_inch))
    axes = np.array(axes).reshape(-1)

    for ax, panel in zip(axes, panels):
        ax.imshow(panel["img"])
        ax.axis("off")
        if sort_by == "p_score":
            subtitle = f"P={panel['p_score']:.1f}%, n={panel['count']}\n{panel['label']}"
        else:
            subtitle = f"n={panel['count']}, I={panel['intensity']:.2f}\n{panel['label']}"
        ax.set_title(subtitle, fontsize=6, pad=3)
    for ax in axes[n_panels:]:
        ax.set_visible(False)

    title = (
        "Top 2-hop bond environments by P-score"
        if sort_by == "p_score"
        else "Most common 2-hop bond environments"
    )
    plt.suptitle(title, fontweight="bold", fontsize=9)
    plt.tight_layout()

    if save_path:
        fig.savefig(f"{save_path}.svg", bbox_inches="tight")
        fig.savefig(f"{save_path}.png", bbox_inches="tight", dpi=300)

    return fig
