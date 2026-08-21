"""Shared histogram plotting helper for similarity analysis scripts."""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from icicle.utils.visualization.style import get_palette, set_style


def plot_similarity_histograms(
    df: pd.DataFrame,
    metric_cols: list[str],
    title: str,
    output_path: Path,
) -> None:
    """Plot one histogram per metric, save as .svg and .png.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing metric columns.
    metric_cols : list[str]
        Column names to plot.
    title : str
        Figure suptitle (typically includes N pairs).
    output_path : Path
        Stem path (no extension); .svg and .png are appended.
    """
    set_style("manuscript")
    palette = get_palette()
    bar_color = palette[0]

    ncols = min(4, len(metric_cols))
    nrows = math.ceil(len(metric_cols) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.3 * ncols, 2.5 * nrows))
    axes = np.array(axes).flatten()

    for ax, col in zip(axes, metric_cols):
        vals = df[col].dropna()
        ax.hist(vals, bins=50, color=bar_color)
        mean_val = vals.mean()
        median_val = vals.median()
        ax.axvline(
            mean_val,
            color="black",
            linestyle="-",
            linewidth=0.8,
            label=f"mean={mean_val:.3f}",
        )
        ax.axvline(
            median_val,
            color="black",
            linestyle="--",
            linewidth=0.8,
            label=f"median={median_val:.3f}",
        )
        ax.legend(fontsize=7)
        ax.set_xlabel(col.replace("_", " "), fontweight="bold")
        ax.set_ylabel("Count", fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.6)

    for ax in axes[len(metric_cols) :]:
        ax.set_visible(False)

    fig.suptitle(title, fontsize=11, fontweight="bold")
    plt.tight_layout()
    output_path = Path(output_path)
    fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
