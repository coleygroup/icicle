"""Helper functions for computing spectral similarity metrics."""

import logging
from typing import Dict, List

import numpy as np

from icicle.analysis.metrics import (
    composite_weighted_cosine_similarity,
    cosine_similarity,
    entropy_distance,
    entropy_similarity,
    mean_squared_error,
    spectral_contrast_angle,
    weighted_cosine_similarity,
)


def compute_metrics_for_spectra(
    pred_spec: np.ndarray,
    true_spec: np.ndarray,
    ranking_metrics: List[str],
    weighted_cosine_schemes: List[str],
    composite_similarity_schemes: List[str],
    mz_values: np.ndarray,
) -> Dict[str, float]:
    """Compute all specified similarity metrics between two spectra.

    Parameters
    ----------
    pred_spec : np.ndarray
        Predicted spectrum intensities
    true_spec : np.ndarray
        Ground truth spectrum intensities
    ranking_metrics : List[str]
        List of base metric names to compute
    weighted_cosine_schemes : List[str]
        Weighting schemes for weighted cosine similarity
    composite_similarity_schemes : List[str]
        Weighting schemes for composite similarity
    mz_values : np.ndarray
        m/z values for weighted metrics

    Returns
    -------
    Dict[str, float]
        Dictionary mapping metric names to values
    """
    metrics = {}

    for metric in ranking_metrics:
        try:
            if metric == "cosine_similarity":
                metrics["cosine_similarity"] = float(
                    cosine_similarity(pred_spec, true_spec)
                )
            elif metric == "entropy_similarity":
                metrics["entropy_similarity"] = float(
                    entropy_similarity(pred_spec, true_spec)
                )
            elif metric == "entropy_distance":
                metrics["entropy_distance"] = float(
                    entropy_distance(pred_spec, true_spec)
                )
            elif metric == "spectral_contrast_angle":
                metrics["spectral_contrast_angle"] = float(
                    spectral_contrast_angle(pred_spec, true_spec)
                )
            elif metric == "mean_squared_error":
                metrics["mean_squared_error"] = float(
                    mean_squared_error(pred_spec, true_spec)
                )
            elif metric == "weighted_cosine":
                for scheme in weighted_cosine_schemes:
                    metrics[f"weighted_cosine_{scheme}"] = float(
                        weighted_cosine_similarity(
                            pred_spec,
                            true_spec,
                            mz_values,
                            weighting_scheme=scheme,
                        )
                    )
            elif metric == "composite_similarity":
                for scheme in composite_similarity_schemes:
                    metrics[f"composite_similarity_{scheme}"] = float(
                        composite_weighted_cosine_similarity(
                            pred_spec,
                            true_spec,
                            mz_values,
                            weighting_scheme=scheme,
                        )
                    )
        except Exception as e:
            logging.warning(f"Error computing {metric}: {e}")
            # Set defaults for failed metric computation
            if metric == "weighted_cosine":
                for scheme in weighted_cosine_schemes:
                    metrics[f"weighted_cosine_{scheme}"] = 0.0
            elif metric == "composite_similarity":
                for scheme in composite_similarity_schemes:
                    metrics[f"composite_similarity_{scheme}"] = 0.0
            elif metric == "spectral_contrast_angle":
                metrics["spectral_contrast_angle"] = float(np.pi / 2)
            elif metric == "mean_squared_error":
                metrics["mean_squared_error"] = 1.0
            else:
                metrics[metric] = 0.0

    return metrics


def get_default_metrics(
    ranking_metrics: List[str],
    weighted_cosine_schemes: List[str],
    composite_similarity_schemes: List[str],
) -> Dict[str, float]:
    """Get default metric values for failed predictions.

    Parameters
    ----------
    ranking_metrics : List[str]
        List of base metric names
    weighted_cosine_schemes : List[str]
        Weighting schemes for weighted cosine similarity
    composite_similarity_schemes : List[str]
        Weighting schemes for composite similarity

    Returns
    -------
    Dict[str, float]
        Dictionary mapping metric names to default values
    """
    defaults = {}

    for metric in ranking_metrics:
        if metric == "weighted_cosine":
            for scheme in weighted_cosine_schemes:
                defaults[f"weighted_cosine_{scheme}"] = 0.0
        elif metric == "composite_similarity":
            for scheme in composite_similarity_schemes:
                defaults[f"composite_similarity_{scheme}"] = 0.0
        elif metric == "spectral_contrast_angle":
            defaults["spectral_contrast_angle"] = float(np.pi / 2)
        elif metric == "mean_squared_error":
            defaults["mean_squared_error"] = 1.0
        elif metric == "entropy_distance":
            defaults["entropy_distance"] = 1.0  # worst distance = 1.0, not 0.0
        else:
            defaults[metric] = 0.0

    return defaults


def get_similarity_column_names(
    ranking_metrics: List[str],
    weighted_cosine_schemes: List[str],
    composite_similarity_schemes: List[str],
) -> List[str]:
    """Get list of all similarity column names from metrics and schemes.

    Parameters
    ----------
    ranking_metrics : List[str]
        List of base metric names
    weighted_cosine_schemes : List[str]
        Weighting schemes for weighted cosine similarity
    composite_similarity_schemes : List[str]
        Weighting schemes for composite similarity

    Returns
    -------
    List[str]
        List of all column names for similarity metrics
    """
    columns = []
    for metric in ranking_metrics:
        if metric == "weighted_cosine":
            for scheme in weighted_cosine_schemes:
                columns.append(f"weighted_cosine_{scheme}")
        elif metric == "composite_similarity":
            for scheme in composite_similarity_schemes:
                columns.append(f"composite_similarity_{scheme}")
        else:
            columns.append(metric)
    return columns
