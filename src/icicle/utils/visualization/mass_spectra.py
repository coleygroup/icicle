"""Visualization functions for mass spectra."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from rdkit import Chem
from rdkit.Chem import Draw

from .style import FIGSIZE, make_fig, spec_colors


def create_spectrum_figure(
    figsize: Tuple[int, int], n_subplots: int = 1, share_x: bool = True
) -> Tuple[plt.Figure, List[plt.Axes]]:
    """Creates a figure with the proper styling for mass spectra plots.

    ``figsize`` is the *axes area* per subplot, not the total figure size.
    """
    fig, axes = make_fig(figsize, nrows=n_subplots, sharex=share_x)
    if n_subplots == 1:
        axes = [axes]

    for ax in axes:
        ax.grid(False)
        ax.axhline(0, color="black", linewidth=1.0)
        ax.set_ylim(0, 1.1)
        ax.set_yticks([0, 0.5, 1.0])

        # Ensure spines are visible and black
        for spine in ax.spines.values():
            spine.set_color("black")
            spine.set_linewidth(1.4)

        ax.xaxis.set_minor_locator(plt.NullLocator())
        ax.yaxis.set_minor_locator(plt.NullLocator())
        ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=5))

    return fig, axes


def add_molecule_inset(
    ax: plt.Axes,
    smiles: str,
    position: Tuple[float, float] = (0.05, 0.65),
    size: Tuple[float, float] = (0.3, 0.3),
) -> None:
    """Adds a molecule visualization inset to the plot."""
    mol = Chem.MolFromSmiles(smiles)
    img = Draw.MolToImage(mol)
    img_array = np.asarray(img)
    ax_inset = ax.inset_axes([*position, *size])
    ax_inset.imshow(img_array)
    ax_inset.axis("off")


def plot_spectrum_stems(
    ax: plt.Axes,
    mz_values: np.ndarray,
    intensities: np.ndarray,
    color: str = "black",
    alpha: float = 0.7,
    linewidth: float = 1,
    label: Optional[str] = None,
    negative: bool = False,
) -> None:
    """Plots spectrum stems with consistent styling."""
    intensities_to_plot = -intensities if negative else intensities
    markerline, stemlines, baseline = ax.stem(
        mz_values,
        intensities_to_plot,
        markerfmt=" ",
        basefmt=" ",
        label=label if label else None,
    )
    plt.setp(stemlines, color=color, alpha=alpha, linewidth=linewidth)


def plot_mass_spectrum(
    mz_values: np.ndarray,
    intensities: np.ndarray,
    smiles: Optional[str] = None,
    title: Optional[str] = None,
    fragments: Optional[Dict] = None,
    comparison_intensities: Optional[np.ndarray] = None,
    output_path: Optional[str | Path] = None,
    max_fragments: int = 5,
    figsize: Tuple[int, int] = FIGSIZE["default"],
) -> plt.Figure:
    """[Original docstring remains the same]"""

    # Create figure using helper function
    n_subplots = 2 if comparison_intensities is not None else 1
    fig, axes = create_spectrum_figure(figsize, n_subplots)
    ax1 = axes[0]

    # Normalize intensities
    intensities = intensities / np.max(intensities)
    if comparison_intensities is not None:
        comparison_intensities = comparison_intensities / np.max(
            comparison_intensities
        )

    # Draw molecule if SMILES provided
    if smiles:
        add_molecule_inset(ax1, smiles)

    # Plot spectra using helper function with consistent colors
    for ax, intens, label in zip(
        axes,
        [intensities, comparison_intensities]
        if comparison_intensities is not None
        else [intensities],
        ["Predicted", "Experimental"]
        if comparison_intensities is not None
        else [""],
    ):
        # Use model_colors for consistent coloring
        color = (
            spec_colors["pred_spec"]
            if label in ["", "Predicted"]
            else spec_colors["true_spec"]
        )
        plot_spectrum_stems(ax, mz_values, intens, color=color, label=label)

        # Add fragments if provided and this is the first axis
        if fragments and ax == ax1:
            _add_fragments(ax, mz_values, fragments, intens, max_fragments)

        # Customize axis (same as before)
        ax.set_ylabel("Relative Intensity")
        if label:
            ax.set_title(f"{label} Spectrum")
            ax.legend(frameon=True, facecolor="white", edgecolor="none")

        # Set axis limits
        if smiles:
            precursor_mz = Chem.Descriptors.ExactMolWt(
                Chem.MolFromSmiles(smiles)
            )
            ax.set_xlim(0, precursor_mz + 10)

        # Grid customization
        ax.grid(False)

    # Set common labels and title
    axes[-1].set_xlabel("m/z")
    if title:
        fig.suptitle(title, y=0.95)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

    return fig


def plot_mirrored_spectra(
    true_spec: np.ndarray,
    pred_spec: np.ndarray,
    true_smiles: str = None,
    title: str = None,
    predicted_label: str = "Predicted",
    true_label: str = "True",
    figsize: Tuple[int, int] = FIGSIZE["default"],
    fade_unmatched: bool = False,
    pred_smiles: str = None,
    true_color: str = None,
    pred_color: str = None,
) -> plt.Figure:
    """Plot two mass spectra in a mirrored format."""
    mz = np.linspace(0, 750, len(true_spec))  # FIXME

    if true_color is None:
        true_color = spec_colors["true_spec"]
    if pred_color is None:
        pred_color = spec_colors["pred_spec"]

    # Use create_spectrum_figure helper
    fig, [ax] = create_spectrum_figure(figsize)
    ax.set_ylim(-1.1, 1.1)  # Override default y-limits for mirrored plot

    if fade_unmatched:
        # Find matching peaks
        matched_pred = np.where(true_spec > 0, pred_spec, 0)
        unmatched_pred = np.where(true_spec == 0, pred_spec, 0)
        matched_true = np.where(pred_spec > 0, true_spec, 0)
        unmatched_true = np.where(pred_spec == 0, true_spec, 0)

        # Plot matched and unmatched peaks using helper
        plot_spectrum_stems(ax, mz, matched_pred, color=pred_color, alpha=0.7)
        plot_spectrum_stems(
            ax, mz, unmatched_pred, color=pred_color, alpha=0.3
        )
        plot_spectrum_stems(
            ax,
            mz,
            matched_true,
            color=true_color,
            alpha=0.7,
            negative=True,
        )
        plot_spectrum_stems(
            ax,
            mz,
            unmatched_true,
            color=true_color,
            alpha=0.3,
            negative=True,
        )
    else:
        plot_spectrum_stems(ax, mz, pred_spec, color=pred_color, alpha=0.7)
        plot_spectrum_stems(ax, mz, true_spec, color=true_color, negative=True)

    if true_smiles:
        add_molecule_inset(ax, true_smiles)

    if pred_smiles:
        add_molecule_inset(ax, pred_smiles, position=(0.05, 0.05))

    # Add legend
    ax.plot(
        [],
        [],
        color=pred_color,
        label=predicted_label,
        alpha=0.7,
    )
    ax.plot([], [], color=true_color, label=true_label, alpha=0.7)
    ax.legend(
        loc="upper right", frameon=True, facecolor="white", edgecolor="none"
    )

    ax.set_xlabel("m/z")
    ax.set_ylabel("Relative Intensity")
    if title:
        ax.set_title(title)

    # Set x-axis limits
    max_idx = max(
        len(pred_spec) - 1 - np.argmax(pred_spec[::-1] > 0),
        len(true_spec) - 1 - np.argmax(true_spec[::-1] > 0),
    )

    if true_smiles:
        precursor_mz = Chem.Descriptors.ExactMolWt(
            Chem.MolFromSmiles(true_smiles)
        )
        ax.set_xlim(0, precursor_mz + 10)
    else:
        ax.set_xlim(0, mz[max_idx] + 10)

    plt.tight_layout()
    return fig


def _add_fragments(ax, mz_values, fragments, intensities, max_fragments):
    """Helper function to add fragment structures to the plot."""
    fragment_count = 0
    sorted_fragments = sorted(
        fragments.items(),
        key=lambda x: intensities[np.abs(x[0] - mz_values).argmin()],
        reverse=True,
    )

    for mz, frag_info in sorted_fragments:
        if fragment_count >= max_fragments:
            break

        try:
            # Create fragment image
            frag_img = Draw.MolToImage(
                frag_info["structure"],
                highlightAtoms=frag_info["highlights"].get("atoms", []),
                highlightBonds=frag_info["highlights"].get("bonds", []),
                size=(300, 300),
            )

            # Make background transparent
            frag_img = np.array(frag_img)
            rgba_img = np.concatenate(
                (
                    frag_img,
                    255
                    * np.ones(
                        (frag_img.shape[0], frag_img.shape[1], 1),
                        dtype=np.uint8,
                    ),
                ),
                axis=-1,
            )
            white_background = (frag_img[:, :, :3] == 255).all(axis=2)
            rgba_img[white_background] = [0, 0, 0, 0]

            # Get intensity at this m/z
            idx = np.abs(mz_values - mz).argmin()
            inten = intensities[idx]

            # Add fragment image
            imagebox = OffsetImage(rgba_img, zoom=0.15)
            ab = AnnotationBbox(imagebox, (mz, inten + 0.1), frameon=False)
            ax.add_artist(ab)

            # Add m/z label
            ax.text(
                mz,
                inten + 0.05,
                f"{mz:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

            fragment_count += 1

        except Exception as e:
            print(f"Failed to add fragment at m/z={mz:.1f}: {e}")


def plot_three_spectra(
    exp_spec: np.ndarray,
    qcxms_spec: np.ndarray,
    icicle_spec: np.ndarray,
    mz_values: np.ndarray = None,
    smiles: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (3, 1.5),
    fade_unmatched: bool = False,
    exp_label: str = "Experimental",
    qcxms_label: str = "QCxMS2",
    icicle_label: str = "ICICLE",
    icicle_color: Optional[str] = None,
    qcxms_color: Optional[str] = None,
) -> plt.Figure:
    """Plot three mass spectra stacked vertically in a single figure.

    Parameters
    ----------
    exp_spec : np.ndarray
        Experimental spectrum intensities (ground truth)
    qcxms_spec : np.ndarray
        QCxMS2 predicted spectrum intensities
    icicle_spec : np.ndarray
        ICICLE predicted spectrum intensities
    mz_values : np.ndarray, optional
        m/z values for all spectra (if None, uses range(len(exp_spec)))
    smiles : str, optional
        SMILES string of the molecule
    title : str, optional
        Title for the figure
    figsize : Tuple[int, int], optional
        Figure size
    fade_unmatched : bool, optional
        Whether to fade peaks that don't match the experimental spectrum
    exp_label : str, optional
        Label for experimental spectrum
    qcxms_label : str, optional
        Label for QCxMS2 spectrum
    icicle_label : str, optional
        Label for ICICLE spectrum
    icicle_color : str, optional
        Override color for the ICICLE spectrum (default: ``spec_colors["pred_spec"]``)
    qcxms_color : str, optional
        Override color for the QCxMS2/second-model spectrum (default: ``"#1C6090"``)

    Returns
    -------
    plt.Figure
        The created figure
    """

    # Create m/z values if not provided
    if mz_values is None:
        mz_values = np.arange(len(exp_spec))

    # Create figure with three subplots
    fig, axes = create_spectrum_figure(figsize, n_subplots=3, share_x=True)

    # Labels for the spectra
    labels = [icicle_label, qcxms_label, exp_label]
    specs = [icicle_spec, qcxms_spec, exp_spec]

    colors = [
        icicle_color or spec_colors["pred_spec"],
        qcxms_color or "#1C6090",
        spec_colors["true_spec"],  # Experimental color
    ]

    # Plot each spectrum
    for i, (ax, spec, label, color) in enumerate(
        zip(axes, specs, labels, colors)
    ):
        # Normalize the spectrum
        spec = spec / np.max(spec) if np.max(spec) > 0 else spec

        if fade_unmatched and i < 2:  # Only for ICICLE and QCxMS2 spectra
            # Find matching and non-matching peaks with experimental
            exp_normalized = (
                exp_spec / np.max(exp_spec)
                if np.max(exp_spec) > 0
                else exp_spec
            )

            # Define a threshold for considering a peak present
            threshold = 0.01
            exp_peaks = exp_normalized > threshold

            # Create matched and unmatched arrays
            matched = np.zeros_like(spec)
            unmatched = np.zeros_like(spec)

            # Fill arrays based on matching with experimental
            for j in range(len(spec)):
                if j < len(exp_peaks):
                    # If there's a peak in experimental, it's a match
                    if exp_peaks[j]:
                        matched[j] = spec[j]
                    # Otherwise it's unmatched
                    elif spec[j] > threshold:
                        unmatched[j] = spec[j]

            # Plot only the matched peaks with full opacity
            plot_spectrum_stems(
                ax,
                mz_values,
                matched,
                color=color,
                alpha=0.9,
                label=f"{label} (matched)",
            )

            # Plot only the unmatched peaks with reduced opacity
            plot_spectrum_stems(
                ax,
                mz_values,
                unmatched,
                color=color,
                alpha=0.3,
                label=f"{label} (unmatched)",
            )
        else:
            # Plot normal spectrum
            plot_spectrum_stems(
                ax, mz_values, spec, color=color, alpha=0.7, label=label
            )

        # Add molecule visualization for the top subplot
        if smiles and i == 0:
            add_molecule_inset(ax, smiles)

        ax.set_ylabel(label)

        # Only show legend if we have labels
        # if fade_unmatched and i < 2:
        #     # For faded spectra, show both matched and unmatched in legend
        #     handles, labels = ax.get_legend_handles_labels()
        #     # Limit to last two handles/labels (matched and unmatched)
        #     if len(handles) >= 2:
        #         ax.legend(
        #             handles=handles[-2:],
        #             labels=labels[-2:],
        #             loc="upper right",
        #             frameon=True,
        #             facecolor="white",
        #             edgecolor="none",
        #         )
        # elif label:
        #     ax.legend(
        #         loc="upper right",
        #         frameon=True,
        #         facecolor="white",
        #         edgecolor="none",
        #     )

        # Set x-axis limits based on molecule or data
        if smiles:
            precursor_mz = Chem.Descriptors.ExactMolWt(
                Chem.MolFromSmiles(smiles)
            )
            ax.set_xlim(0, precursor_mz + 10)
        else:
            # Find the highest m/z with significant intensity
            max_idx = (
                len(spec) - 1 - np.argmax(spec[::-1] > 0.01)
                if np.max(spec) > 0
                else len(spec) - 1
            )
            ax.set_xlim(0, mz_values[min(max_idx, len(mz_values) - 1)] + 10)

    # Only the bottom subplot shows x-axis ticks/label; others share it
    for ax in axes[:-1]:
        ax.set_xlabel("")
        ax.tick_params(axis="x", labelbottom=False, length=0)
    axes[-1].set_xlabel("m/z")

    plt.tight_layout()
    fig.subplots_adjust(hspace=0.05)  # subplots close together, shared x-axis

    return fig


def plot_four_spectra(
    exp_spec: np.ndarray,
    qcxms_spec_gfn2: np.ndarray,
    qcxms_spec_dft: np.ndarray,
    icicle_spec: np.ndarray,
    mz_values: np.ndarray = None,
    smiles: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = FIGSIZE["tall"],
    fade_unmatched: bool = False,
    exp_label: str = "Experimental",
    qcxms_gfn2_label: str = "QCxMS2 GFN2",
    qcxms_dft_label: str = "QCxMS2 wb97x3c",
    icicle_label: str = "ICICLE",
) -> plt.Figure:
    # Create m/z values if not provided
    if mz_values is None:
        mz_values = np.arange(len(exp_spec))

    # Create figure with three subplots
    fig, axes = create_spectrum_figure(figsize, n_subplots=4, share_x=True)

    # Labels for the spectra
    labels = [qcxms_dft_label, qcxms_gfn2_label, icicle_label, exp_label]
    specs = [qcxms_spec_dft, qcxms_spec_gfn2, icicle_spec, exp_spec]

    # Custom color for QCxMS2
    colors = [
        "#1C6090",  # QCxMS2 custom color
        "#1C6090",
        spec_colors["pred_spec"],  # ICICLE color
        spec_colors["true_spec"],  # Experimental color
    ]

    # Plot each spectrum
    for i, (ax, spec, label, color) in enumerate(
        zip(axes, specs, labels, colors)
    ):
        # Normalize the spectrum
        spec = spec / np.max(spec) if np.max(spec) > 0 else spec

        if fade_unmatched and i < 3:  # Only for ICICLE and QCxMS2 spectra
            # Find matching and non-matching peaks with experimental
            exp_normalized = (
                exp_spec / np.max(exp_spec)
                if np.max(exp_spec) > 0
                else exp_spec
            )

            # Define a threshold for considering a peak present
            threshold = 0.01
            exp_peaks = exp_normalized > threshold

            # Create matched and unmatched arrays
            matched = np.zeros_like(spec)
            unmatched = np.zeros_like(spec)

            # Fill arrays based on matching with experimental
            for j in range(len(spec)):
                if j < len(exp_peaks):
                    # If there's a peak in experimental, it's a match
                    if exp_peaks[j]:
                        matched[j] = spec[j]
                    # Otherwise it's unmatched
                    elif spec[j] > threshold:
                        unmatched[j] = spec[j]

            # Plot only the matched peaks with full opacity
            plot_spectrum_stems(
                ax,
                mz_values,
                matched,
                color=color,
                alpha=0.9,
                label=f"{label} (matched)",
            )

            # Plot only the unmatched peaks with reduced opacity
            plot_spectrum_stems(
                ax,
                mz_values,
                unmatched,
                color=color,
                alpha=0.3,
                label=f"{label} (unmatched)",
            )
        else:
            # Plot normal spectrum
            plot_spectrum_stems(
                ax, mz_values, spec, color=color, alpha=0.7, label=label
            )

        # Add molecule visualization for the top subplot
        if smiles and i == 0:
            add_molecule_inset(ax, smiles)

        ax.set_ylabel(label)

        # Set x-axis limits based on molecule or data
        if smiles:
            precursor_mz = Chem.Descriptors.ExactMolWt(
                Chem.MolFromSmiles(smiles)
            )
            ax.set_xlim(0, precursor_mz + 10)
        else:
            # Find the highest m/z with significant intensity
            max_idx = (
                len(spec) - 1 - np.argmax(spec[::-1] > 0.01)
                if np.max(spec) > 0
                else len(spec) - 1
            )
            ax.set_xlim(0, mz_values[min(max_idx, len(mz_values) - 1)] + 10)

    # Only add x-label to the bottom subplot
    for i in range(len(axes) - 1):
        axes[i].set_xlabel("")
    axes[-1].set_xlabel("m/z")

    # Add overall title if provided
    if title:
        fig.suptitle(title, y=0.98)

    plt.tight_layout()
    fig.subplots_adjust(hspace=0.3)  # Add some space between subplots

    return fig
