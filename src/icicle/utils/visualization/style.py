"""Plotting utilities: colors, fonts, sizes, save helper.

Usage::

    from icicle.utils.visualization.style import set_style, get_palette, get_cmap, save_fig

    set_style("manuscript")          # call once at top of notebook/script
    palette = get_palette()
    fig, ax = plt.subplots()
    ...
    save_fig(fig, "my_figure")       # saves my_figure.svg + my_figure.png
"""

import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from pypalettes import load_cmap

# ---------------------------------------------------------------------------
# Palette (module-level, stable across calls)
# ---------------------------------------------------------------------------

_cmap = load_cmap("X90", cmap_type="continuous")
_palette: list[str] = list(_cmap.colors) + ["#D3D3D3", "#A9A9A9", "#808080"]
sns.set_palette(_palette)

warnings.filterwarnings("ignore")

spec_colors = {
    "true_spec": "#000000",
    "pred_spec": _palette[2],
}

# Color manipulation helpers (ported from foam/postprocessing reference)
import colorsys

import matplotlib.colors as mcolors
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


def lighten(color: str, amount: float = 0.6) -> tuple:
    """Blend color toward white by amount (0 = original, 1 = white)."""
    c = np.array(mcolors.to_rgb(color))
    return tuple((1 - amount) * c + amount * np.ones(3))


def darken(color: str, amount: float = 0.2) -> tuple:
    """Blend color toward black by amount (0 = original, 1 = black)."""
    c = np.array(mcolors.to_rgb(color))
    return tuple((1 - amount) * c)


def saturate(color: str, factor: float = 1.4) -> tuple:
    """Scale HSV saturation by factor (clamped to 1)."""
    r, g, b = mcolors.to_rgb(color)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    return colorsys.hsv_to_rgb(h, min(1.0, s * factor), v)


# Continuous colormap for density / hex plots (white → saturated green)
_intense_green = saturate(darken(_palette[2], 0.1), factor=1.3)
HEXPLOT_CMAP = LinearSegmentedColormap.from_list(
    "hexplot_cmap_green", ["white", _intense_green], N=256
)


# ---------------------------------------------------------------------------
# Public accessors
# ---------------------------------------------------------------------------


# Public aliases (kept for backward compat with __init__.py re-exports)
cmap = _cmap
palette = _palette


def set_size(w: float = 3.25, h: float = 3.5, ax=None) -> None:
    """Resize axis to exact dimensions in inches.

    Prefer FIGSIZE constants instead.
    """
    if ax is None:
        ax = plt.gca()
    sp = ax.figure.subplotpars
    figw = float(w) / (sp.right - sp.left)
    figh = float(h) / (sp.top - sp.bottom)
    ax.figure.set_size_inches(figw, figh)


def make_fig(
    size: str | tuple = "default",
    nrows: int = 1,
    ncols: int = 1,
    sharex: bool = False,
    sharey: bool = False,
    **subplot_kwargs,
) -> tuple:
    """Create a figure where the *axes area* matches the requested FIGSIZE.

    Unlike ``plt.subplots(figsize=FIGSIZE[...])`` — which sizes the whole figure
    including labels and whitespace — this function adds padding so the plot area
    itself is exactly the requested size.

    Args:
        size: FIGSIZE key (``"default"``, ``"square"``, ``"wide"``, ``"tall"``) or
              raw ``(w, h)`` tuple in inches.
        nrows: Number of subplot rows.
        ncols: Number of subplot cols.
        sharex: Share x-axis across subplots.
        sharey: Share y-axis across subplots.
        **subplot_kwargs: Passed to ``plt.subplots``.

    Returns:
        ``(fig, ax)`` — same as ``plt.subplots``.
    """
    plot_w, plot_h = FIGSIZE[size] if isinstance(size, str) else size

    # Empirical padding for manuscript style (inches): axes labels + tick labels
    pad_left = 0.55  # y-axis label + tick labels
    pad_right = 0.10  # right margin
    pad_bottom = 0.45  # x-axis label + tick labels
    pad_top = 0.15  # top margin (no title by default)

    fig_w = plot_w * ncols + pad_left + pad_right
    fig_h = plot_h * nrows + pad_bottom + pad_top

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(fig_w, fig_h),
        sharex=sharex,
        sharey=sharey,
        **subplot_kwargs,
    )

    left = pad_left / fig_w
    right = 1.0 - pad_right / fig_w
    bottom = pad_bottom / fig_h
    top = 1.0 - pad_top / fig_h
    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)

    return fig, axes


def get_palette() -> list[str]:
    """Return the active discrete color palette."""
    return _palette


def get_cmap():
    """Return the continuous colormap."""
    return _cmap


# ---------------------------------------------------------------------------
# Figure size constants
# ---------------------------------------------------------------------------

# Use these to pick a size; never pass figsize= directly in notebooks.
FIGSIZE = {
    "default": (3, 2.5),  # single panel, rectangle
    "square": (3, 3),  # parity / scatter
    "wide": (6, 2.5),  # wide two-panel (rare)
    "tall": (3, 4),  # tall single panel
}


# ---------------------------------------------------------------------------
# set_style
# ---------------------------------------------------------------------------


def set_style(style: str = "manuscript") -> None:
    """Set global matplotlib/seaborn style.

    Calling this once at the top of a notebook/script is sufficient.
    Do NOT override rcParams afterwards — let the style do its job.

    Args:
        style: One of ``"manuscript"``, ``"presentation"``, ``"poster"``.
    """
    _SIZE = {
        "manuscript": {
            "font": 10,
            "label": 10,
            "title": 10,
            "tick": 9,
            "legend": 9,
            "major_tick": 3,
        },
        "presentation": {
            "font": 12,
            "label": 12,
            "title": 12,
            "tick": 11,
            "legend": 11,
            "major_tick": 4,
        },
        "poster": {
            "font": 12,
            "label": 12,
            "title": 12,
            "tick": 11,
            "legend": 11,
            "major_tick": 4,
        },
    }
    if style not in _SIZE:
        raise KeyError(
            f"Style '{style}' not recognized. Choose: {list(_SIZE)}"
        )

    sz = _SIZE[style]

    settings: dict = {
        # --- reproducibility ---
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        # --- font (Arial via mathtext, no LaTeX dep) ---
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
        "mathtext.sf": "Arial",
        # --- font sizes (all uniform, no post-hoc fontweight/size) ---
        "font.size": sz["font"],
        "axes.labelsize": sz["label"],
        "axes.titlesize": sz["title"],
        "xtick.labelsize": sz["tick"],
        "ytick.labelsize": sz["tick"],
        "legend.fontsize": sz["legend"],
        "legend.title_fontsize": sz["legend"],
        # --- figure ---
        "figure.figsize": FIGSIZE["default"],
        "figure.dpi": 300,
        "figure.facecolor": "white",
        "figure.autolayout": False,
        # --- axes ---
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.linewidth": 1.4,
        "axes.edgecolor": "black",
        "axes.labelcolor": "black",
        "axes.axisbelow": True,
        "axes.xmargin": 0.02,
        "axes.ymargin": 0.02,
        # --- ticks ---
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": sz["major_tick"],
        "ytick.major.size": sz["major_tick"],
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.top": False,
        "ytick.right": False,
        "xtick.color": "black",
        "ytick.color": "black",
        # --- lines / markers ---
        "lines.linewidth": 1.0,
        "lines.markersize": 4,
        "hatch.linewidth": 0.5,
        "grid.linewidth": 0.5,
        # --- legend ---
        "legend.frameon": False,
        "legend.fancybox": False,
        "legend.facecolor": "none",
        "legend.edgecolor": "none",
        "legend.handlelength": 1.5,
        "legend.handletextpad": 0.4,
        # --- text ---
        "text.color": "black",
        # --- color cycle ---
        "axes.prop_cycle": plt.cycler("color", _palette[:8]),
    }

    for k, v in settings.items():
        mpl.rcParams[k] = v


# ---------------------------------------------------------------------------
# save_fig  — the only sanctioned way to save figures
# ---------------------------------------------------------------------------


def save_fig(
    fig: "plt.Figure", name: str, output_dir: str | Path = ".", dpi: int = 300
) -> None:
    """Save figure as .svg and .pdf (both vector, for LaTeX inclusion).

    Args:
        fig: The matplotlib figure to save.
        name: Filename stem (no extension).
        output_dir: Directory to write into (created if absent).
        dpi: Unused for vector formats; kept for backward compatibility.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    kwargs = dict(bbox_inches="tight", transparent=False)
    fig.savefig(out / f"{name}.svg", **kwargs)
    fig.savefig(out / f"{name}.pdf", **kwargs)
