"""Shared plotting functions for final evaluation results (similarity, formula
retrieval, RI-windowed PubChem retrieval).

All three ``fig_*_results.ipynb`` notebooks call into this module so that
figure style, CI methodology, and significance testing stay identical across
plots without duplicating logic per notebook.

Usage::

    from icicle.utils.visualization.eval_plots import (
        load_seed_csvs, mean_ci_t, plot_metric_bars, compute_topk_curve,
        plot_topk_curves, plot_ri_ladder, wilcoxon_stars,
    )
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statannotations.Annotator import Annotator

from icicle.utils.visualization.style import get_palette, make_fig

CI_LEVEL = 0.95
BOXPLOT_JITTER_SEED = 42

# Fixed palette indices per model, kept consistent across similarity,
# formula-match retrieval, RI-match retrieval, and QCxMS2 figures.
MODEL_PALETTE_INDICES = {
    "ICICLE": 0,
    "NEIMS": 3,
    "RASSP": 5,
    "MassFormer": 8,
}


def model_color(label: str, fallback_index: int = 0) -> str:
    """Fixed color for a model label, falling back to positional cycling.

    Args:
        label: Model label (e.g. ``"ICICLE"``, ``"NEIMS"``).
        fallback_index: Position to cycle the palette on for labels not in
            ``MODEL_PALETTE_INDICES``.

    Returns:
        Hex color string.
    """
    palette = get_palette()
    idx = MODEL_PALETTE_INDICES.get(label)
    if idx is None:
        idx = fallback_index % len(palette)
    return palette[idx]


def plot_boxplot_with_points(
    data: dict[str, np.ndarray],
    colors: dict[str, str],
    ylim_pad: float = 0.6,
    xlabel: str | None = None,
    xlim: tuple[float, float] | None = (0, 1),
) -> plt.Figure:
    """Horizontal boxplot with jittered per-point overlay, sorted by mean.

    Args:
        data: ``{label: values}``. Entries are sorted ascending by mean so
            the highest-mean group appears at the top of the horizontal plot.
        colors: ``{label: color}``, same keys as ``data``.
        ylim_pad: Padding added above/below the box range on the y-axis.
        xlabel: Metric name shown on the x-axis (e.g. "Entropy similarity").
        xlim: Fixed x-axis limits, matching ``plot_metric_bars`` so bar and
            boxplot figures stay visually uniform. Pass ``None`` to autoscale.

    Returns:
        The figure (not yet saved).
    """
    labels_sorted = sorted(data.keys(), key=lambda k: data[k].mean())
    values = [data[label] for label in labels_sorted]
    box_colors = [colors[label] for label in labels_sorted]

    fig, ax = make_fig("square")

    bp = ax.boxplot(
        values,
        patch_artist=True,
        widths=0.45,
        vert=False,
        medianprops=dict(color="black", linewidth=1.5),
        whiskerprops=dict(linewidth=1),
        capprops=dict(linewidth=1),
        flierprops=dict(marker=""),
    )
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    rng = np.random.default_rng(BOXPLOT_JITTER_SEED)
    for i, (vals, color) in enumerate(zip(values, box_colors), start=1):
        jitter = rng.uniform(-0.15, 0.15, size=len(vals))
        ax.scatter(vals, i + jitter, color=color, s=18, zorder=3, alpha=0.85)

    ax.set_yticks(range(1, len(labels_sorted) + 1))
    ax.set_yticklabels(labels_sorted)
    ax.set_ylim(1 - ylim_pad, len(labels_sorted) + ylim_pad)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if xlim is not None:
        ax.set_xlim(*xlim)
    return fig


def mean_ci_t(
    values: np.ndarray, level: float = CI_LEVEL
) -> tuple[float, float]:
    """Mean and 95% CI half-width via t-distribution over a small sample.

    Args:
        values: 1D array of per-seed (or per-run) values.
        level: Confidence level.

    Returns:
        ``(mean, half_width)`` such that the CI is ``mean ± half_width``.
        ``half_width`` is 0 if fewer than 2 values are given.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = values.mean()
    if n < 2:
        return mean, 0.0
    sem = stats.sem(values)
    half_width = sem * stats.t.ppf((1 + level) / 2, df=n - 1)
    return mean, half_width


def load_seed_csvs(paths: list[str | Path]) -> list[pd.DataFrame]:
    """Load one dataframe per seed, skipping paths that don't exist yet.

    Args:
        paths: Candidate ``results/eval/<run>/*.csv`` paths, one per seed.

    Returns:
        List of loaded dataframes (may be shorter than ``paths`` if some
        seeds haven't finished running yet).
    """
    dfs = []
    for p in paths:
        p = Path(p)
        if p.exists():
            dfs.append(pd.read_csv(p))
    return dfs


def paired_ttest_pvalue(a: np.ndarray, b: np.ndarray) -> float:
    """Paired t-test p-value over per-seed means.

    Args:
        a: Per-seed metric means, model A.
        b: Per-seed metric means, model B (paired with ``a`` by seed).

    Returns:
        Two-sided p-value.
    """
    _, p = stats.ttest_rel(a, b)
    return p


def wilcoxon_stars(a: np.ndarray, b: np.ndarray) -> str:
    """Paired t-test over per-seed means, rendered as significance stars.

    Args:
        a: Per-seed metric means, model A.
        b: Per-seed metric means, model B (paired with ``a`` by seed).

    Returns:
        ``"***"`` (p<0.001), ``"**"`` (p<0.01), ``"*"`` (p<0.05), or ``"ns"``.
    """
    p = paired_ttest_pvalue(a, b)
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def plot_metric_bars(
    model_dfs: dict[str, list[pd.DataFrame]],
    metric: str,
    reference: str,
    xlim: tuple[float, float] = (0, 1),
    ref_lines: dict[str, float] | None = None,
    ref_line_color: str = "0.6",
    show_significance: bool = True,
) -> plt.Figure:
    """Horizontal bar chart: mean ± 95% CI per model for one metric, with
    paired Wilcoxon significance stars vs. a reference model.

    Args:
        model_dfs: ``{model_label: [per_seed_df, ...]}``. Each df must have
            ``metric`` as a column.
        metric: Column name to plot (e.g. ``"cosine_similarity"``).
        reference: Key into ``model_dfs`` to compare every other model against.
        xlim: Fixed x-axis limits (metrics are similarities in [0, 1] by default).
        ref_lines: Optional ``{label: value}`` of extra vertical reference
            lines (e.g. isomer/stereoisomer/replicate similarity ceilings),
            drawn as dashed grey lines spanning the full plot height and
            labeled at the top.
        ref_line_color: Color for the reference lines and their labels.
        show_significance: If ``False``, skip the paired-test significance
            brackets vs. ``reference`` entirely.

    Returns:
        The figure (not yet saved).
    """
    labels = list(model_dfs.keys())
    fig, ax = make_fig("square")

    seed_means = {}
    for i, label in enumerate(labels):
        dfs = model_dfs[label]
        seed_means[label] = np.array([df[metric].mean() for df in dfs])
        mean, half_width = mean_ci_t(seed_means[label])
        ax.barh(
            i,
            mean,
            xerr=half_width,
            color=model_color(label, fallback_index=i),
            capsize=3,
        )

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlim(*xlim)
    ax.set_xlabel(metric)
    ax.invert_yaxis()

    all_ref_lines = dict(ref_lines) if ref_lines else {}
    if reference in seed_means:
        all_ref_lines[reference] = mean_ci_t(seed_means[reference])[0]
    for label, value in all_ref_lines.items():
        ax.axvline(
            value,
            color=ref_line_color,
            linestyle="--",
            linewidth=0.8,
            zorder=0,
        )
        ax.annotate(
            label,
            xy=(value, 1.0),
            xycoords=("data", "axes fraction"),
            xytext=(2, 2),
            textcoords="offset points",
            rotation=90,
            va="bottom",
            ha="left",
            color=ref_line_color,
            fontsize="x-small",
        )

    if show_significance and reference in model_dfs and len(labels) > 1:
        ref_mean = mean_ci_t(seed_means[reference])[0]

        long_df = pd.DataFrame(
            {
                metric: [v for label in labels for v in seed_means[label]],
                "model": [
                    label for label in labels for _ in seed_means[label]
                ],
            }
        )

        # One Annotator call per pair (not batched) so each bracket/star can be
        # colored independently: green = reference wins, red = reference loses,
        # gray = not significant. A star alone doesn't show direction, which is
        # misleading whenever the reference isn't the best model for a metric.
        # line_offset is staggered manually (axes-fraction units) since each
        # call is a fresh Annotator instance with no shared stacking state.
        comparisons = [label for label in labels if label != reference]
        for rank, label in enumerate(comparisons):
            other_mean = mean_ci_t(seed_means[label])[0]
            n = min(len(seed_means[label]), len(seed_means[reference]))
            pvalue = paired_ttest_pvalue(
                seed_means[label][:n], seed_means[reference][:n]
            )

            if pvalue >= 0.05:
                color = "0.5"
            elif ref_mean > other_mean:
                color = "tab:green"
            else:
                color = "tab:red"

            annotator = Annotator(
                ax,
                [(reference, label)],
                data=long_df,
                x=metric,
                y="model",
                order=labels,
                orient="h",
            )
            annotator.configure(
                test=None,
                text_format="star",
                loc="outside",
                color=color,
                verbose=False,
            )
            annotator.set_pvalues([pvalue])
            annotator.annotate(line_offset=0.03 + rank * 0.06)

    return fig


def _query_col(df: pd.DataFrame) -> str:
    """Return the per-query identifier column name."""
    for col in ("query_inchikey14", "query_mol_id", "spec"):
        if col in df.columns:
            return col
    raise KeyError(f"No query column found in {list(df.columns)}")


def correct_ranks_per_query(df: pd.DataFrame, rank_col: str) -> pd.Series:
    """Best (lowest) rank of the correct candidate, one value per query.

    A retrieval results dataframe has multiple candidate rows per query
    (correct + decoys); top-k accuracy must be computed per query, not
    over every candidate row.

    Args:
        df: Retrieval results with one row per (query, candidate) pair.
        rank_col: Column holding the candidate's rank for a given metric.

    Returns:
        Series indexed by query, one row per query.
    """
    correct = (
        df[~df["is_decoy"]]
        if "is_decoy" in df.columns
        else df[df["is_correct"]]
    )
    return correct.groupby(_query_col(df))[rank_col].min()


def compute_topk_curve(
    dfs: list[pd.DataFrame],
    rank_col: str,
    k_values: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Mean and 95% CI half-width of top-k accuracy across seeds.

    Args:
        dfs: One dataframe per seed, each with one row per (query, candidate)
            pair, a ``rank_col`` column, and either ``is_decoy`` or
            ``is_correct`` to identify the correct candidate per query.
        rank_col: Column name holding the rank.
        k_values: k values to evaluate top-k accuracy at.

    Returns:
        ``(mean_pct, half_width_pct)``, each shape ``(len(k_values),)``, in percent.
    """
    per_seed = np.array(
        [
            [
                100.0 * (correct_ranks_per_query(df, rank_col) <= k).mean()
                for k in k_values
            ]
            for df in dfs
        ]
    )
    means = per_seed.mean(axis=0)
    if per_seed.shape[0] < 2:
        return means, np.zeros_like(means)
    sem = stats.sem(per_seed, axis=0)
    half_width = sem * stats.t.ppf(
        (1 + CI_LEVEL) / 2, df=per_seed.shape[0] - 1
    )
    return means, half_width


def plot_topk_curves(
    model_dfs: dict[str, list[pd.DataFrame]],
    rank_col: str,
    k_values: list[int],
    ylim: tuple[float, float] = (0, 100),
) -> plt.Figure:
    """Top-k accuracy vs. k, one line per model, shaded 95% CI band.

    Args:
        model_dfs: ``{model_label: [per_seed_df, ...]}``.
        rank_col: Rank column to threshold at each k.
        k_values: k values (x-axis).
        ylim: Fixed y-axis limits (accuracy is a percentage).

    Returns:
        The figure (not yet saved).
    """
    fig, ax = make_fig("square")
    for i, (label, dfs) in enumerate(model_dfs.items()):
        means, half_widths = compute_topk_curve(dfs, rank_col, k_values)
        color = model_color(label, fallback_index=i)
        ax.plot(k_values, means, marker="o", label=label, color=color)
        ax.fill_between(
            k_values,
            means - half_widths,
            means + half_widths,
            color=color,
            alpha=0.2,
        )
    ax.set_xlabel("k")
    ax.set_ylabel("Top-k accuracy (%)")
    ax.set_ylim(*ylim)
    ax.legend()
    return fig


def plot_ri_ladder(
    model_results: dict[str, dict],
    metric: str = "cosine",
    mode: str = "autofail",
    k: int = 1,
) -> plt.Figure:
    """Top-k accuracy vs. candidate-set size, one line per model.

    Reads the ``retrieval_global_results.json`` structure:
    ``{top_n_level: {mode: {metric: {f"top_{k}_accuracy": ...}}}}``.
    Levels present vary by model/run — only shared/available levels are
    plotted per model, so this works before the full window ladder finishes.

    Args:
        model_results: ``{model_label: json_dict}`` loaded from each model's
            ``retrieval_global_results.json``.
        metric: Similarity metric key (e.g. ``"cosine"``).
        mode: ``"inject"`` or ``"autofail"``.
        k: top-k accuracy to plot (must match a ``top_{k}_accuracy`` key).

    Returns:
        The figure (not yet saved).
    """
    fig, ax = make_fig("default")
    for i, (label, results) in enumerate(model_results.items()):
        levels = [
            lvl
            for lvl in results
            if mode in results[lvl] and metric in results[lvl][mode]
        ]
        numeric_levels = sorted(
            (lvl for lvl in levels if lvl != "all"), key=lambda x: int(x)
        )
        ordered_levels = numeric_levels + (["all"] if "all" in levels else [])
        x = [lvl if lvl == "all" else int(lvl) for lvl in ordered_levels]
        y = [
            100.0 * results[lvl][mode][metric][f"top_{k}_accuracy"]
            for lvl in ordered_levels
        ]
        color = model_color(label, fallback_index=i)
        ax.plot(range(len(x)), y, marker="o", label=label, color=color)
    ax.set_xticks(range(len(x)))
    ax.set_xticklabels(
        [v if v == "all" else f"$10^{{{int(math.log10(v))}}}$" for v in x],
        # rotation=45,
        ha="right",
    )
    ax.set_xlabel("Candidate set size")
    ax.set_ylabel(f"Top-{k} accuracy (%)")
    ax.set_ylim(0, 100)
    ax.legend()
    return fig


def plot_ri_ladder_with_funnel(
    ri_only_results: dict,
    funnel_results: dict,
    metric: str = "cosine",
    mode: str = "autofail",
    k: int = 1,
) -> plt.Figure:
    """Top-k accuracy vs. candidate-set size for a single model: RI-only ladder
    vs. the MW±80Da-then-RI funnel, same x-axis (candidate set size) so the two
    strategies are directly comparable at each level.

    Unlike ``plot_ri_ladder`` (one line per model), this compares two
    *candidate-selection strategies* for one model — call once per model
    to see whether pre-filtering by MW before ranking by RI improves
    small-candidate-set retrieval versus RI-only.

    Args:
        ri_only_results: level -> {mode: {metric: {...}}}, e.g. loaded
            from ``retrieval_ablation_{ri_type}.json`` (plus optionally
            ``retrieval_global_results.json``'s "all" merged in).
        funnel_results: level -> {mode: {metric: {...}}}, loaded from
            ``retrieval_mw{da}_then_ri_results.json``'s ``[ri_type]`` key.
            No "all" level (the funnel requires a finite top_n).
        metric: Similarity metric key (e.g. ``"cosine"``).
        mode: ``"inject"`` or ``"autofail"``.
        k: top-k accuracy to plot (must match a ``top_{k}_accuracy`` key).

    Returns:
        The figure (not yet saved).
    """
    fig, ax = make_fig("default")
    palette = get_palette()

    def _levels_xy(results: dict) -> tuple[list, list]:
        levels = [
            lvl
            for lvl in results
            if mode in results[lvl] and metric in results[lvl][mode]
        ]
        numeric_levels = sorted(
            (lvl for lvl in levels if lvl != "all"), key=lambda x: int(x)
        )
        ordered_levels = numeric_levels + (["all"] if "all" in levels else [])
        x = [lvl if lvl == "all" else int(lvl) for lvl in ordered_levels]
        y = [
            100.0 * results[lvl][mode][metric][f"top_{k}_accuracy"]
            for lvl in ordered_levels
        ]
        return x, y

    x_ri, y_ri = _levels_xy(ri_only_results)
    ax.plot(
        range(len(x_ri)),
        y_ri,
        marker="o",
        label="RI-only",
        color=palette[0],
    )

    x_funnel, y_funnel = _levels_xy(funnel_results)
    ax.plot(
        range(len(x_funnel)),
        y_funnel,
        marker="s",
        label="MW±80Da → RI",
        color=palette[3],
    )

    x = x_ri if len(x_ri) >= len(x_funnel) else x_funnel
    ax.set_xticks(range(len(x)))
    ax.set_xticklabels(
        [v if v == "all" else f"$10^{{{int(math.log10(v))}}}$" for v in x],
        ha="right",
    )
    ax.set_xlabel("Candidate set size")
    ax.set_ylabel(f"Top-{k} accuracy (%)")
    ax.set_ylim(0, 100)
    ax.legend()
    return fig


def plot_ladder_multi_strategy(
    strategies: dict[str, dict],
    metric: str = "cosine",
    mode: str = "autofail",
    k: int = 1,
    base_color: str | None = None,
) -> plt.Figure:
    """Top-k accuracy vs. candidate-set size, multiple strategies on one model.

    Generalizes ``plot_ri_ladder_with_funnel`` to an arbitrary number of
    named strategies (e.g. "RI-only", "MW[-10,+80]->RI", "HA(w=3)->RI",
    "MW union", "HA union") sharing one candidate-set-size x-axis. The
    first strategy is drawn in ``base_color`` (solid, circle markers);
    later strategies reuse the same color at increasing transparency with
    a distinct marker each, so related traces read as "the same model,
    another filter" rather than competing colors.

    Args:
        strategies: ``{strategy_label: level_dict}`` where ``level_dict``
            is ``{level: {mode: {metric: {...}}}}``, e.g. loaded directly
            from ``retrieval_ablation_{ri_type}.json`` or
            ``retrieval_{mw,heavy_atom}*_then_ri_results.json``/
            ``retrieval_union_*_results.json``'s ``[ri_type]`` key.
        metric: Similarity metric key (e.g. ``"cosine"``).
        mode: ``"inject"`` or ``"autofail"``.
        k: top-k accuracy to plot (must match a ``top_{k}_accuracy`` key).
        base_color: Hex color for the first (reference) strategy. Defaults
            to the palette's first color.

    Returns:
        The figure (not yet saved).
    """
    fig, ax = make_fig("square")
    palette = get_palette()
    color = base_color or palette[0]
    markers = ["o", "s", "^", "D", "v", "P"]
    alphas = [1.0, 0.55, 0.55, 0.4, 0.4, 0.4]

    def _levels_xy(results: dict) -> tuple[list, list]:
        levels = [
            lvl
            for lvl in results
            if mode in results[lvl] and metric in results[lvl][mode]
        ]
        numeric_levels = sorted(
            (lvl for lvl in levels if lvl != "all"), key=lambda x: int(x)
        )
        ordered_levels = numeric_levels + (["all"] if "all" in levels else [])
        x = [lvl if lvl == "all" else int(lvl) for lvl in ordered_levels]
        y = [
            100.0 * results[lvl][mode][metric][f"top_{k}_accuracy"]
            for lvl in ordered_levels
        ]
        return x, y

    longest_x = []
    for i, (label, results) in enumerate(strategies.items()):
        x, y = _levels_xy(results)
        if len(x) > len(longest_x):
            longest_x = x
        ax.plot(
            range(len(x)),
            y,
            marker=markers[i % len(markers)],
            label=label,
            color=color,
            alpha=alphas[i % len(alphas)],
        )

    ax.set_xticks(range(len(longest_x)))
    ax.set_xticklabels(
        [
            v if v == "all" else f"$10^{{{int(math.log10(v))}}}$"
            for v in longest_x
        ],
        ha="right",
    )
    ax.set_xlabel("Candidate set size")
    ax.set_ylabel(f"Top-{k} accuracy (%)")
    ax.set_ylim(0, 100)
    ax.legend(fontsize="small")
    return fig


def plot_topk_vs_k_multi_setting(
    settings: dict[str, dict],
    k_values: list[int] = (1, 5, 10, 20, 50),
) -> plt.Figure:
    """Top-k accuracy vs. k (x=1 shows top-1 accuracy, x=5 shows top-5, etc.),
    one line per named candidate-selection setting, all on the same axes.

    Unlike ``plot_ri_ladder``/``plot_ri_ladder_with_funnel`` (x-axis =
    candidate-set size, one line per model/strategy at a FIXED k), this
    plots the standard top-k curve shape and instead uses one line per
    *setting* (e.g. "RI-only @1000", "RI-only @all", "MW-global
    [-10,+80]Da", "MW→RI funnel @1000") for a single model — so different
    candidate-selection strategies, including ones with no natural
    candidate-set-size axis (like MW-global), sit on the same plot.

    Args:
        settings: ``{setting_label: metric_dict}`` where ``metric_dict``
            is a single ``by_metric[metric]`` dict already extracted from
            a results JSON, i.e. has ``top_{k}_accuracy`` keys directly
            (not nested under level/mode/metric).
        k_values: k values to plot, must match available
            ``top_{k}_accuracy`` keys in every settings dict.

    Returns:
        The figure (not yet saved).
    """
    fig, ax = make_fig("default")
    palette = get_palette()
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]

    for i, (label, m) in enumerate(settings.items()):
        y = [
            100.0 * m[f"top_{k}_accuracy"]
            for k in k_values
            if f"top_{k}_accuracy" in m
        ]
        x = [k for k in k_values if f"top_{k}_accuracy" in m]
        if not y:
            continue
        ax.plot(
            x,
            y,
            marker=markers[i % len(markers)],
            label=label,
            color=palette[i % len(palette)],
        )

    ax.set_xlabel("k")
    ax.set_ylabel("Top-k accuracy (%)")
    ax.set_ylim(0, 100)
    ax.legend(fontsize="small", loc="lower right")
    return fig
