"""Helper functions for ranking and retrieval metrics computation."""

import logging
from typing import Dict, List

import pandas as pd


# Metrics where lower values indicate more similar spectra
DISTANCE_METRICS = {
    "spectral_contrast_angle",
    "mean_squared_error",
    "entropy_distance",
}


def compute_retrieval_metrics(
    rankings_df: pd.DataFrame, k_max: int = 50
) -> Dict[str, float]:
    """Compute retrieval metrics from rankings.

    Parameters
    ----------
    rankings_df : pd.DataFrame
        DataFrame with columns: spec, inchikey, similarity_score, is_decoy, rank
    k_max : int
        Maximum k for top-k accuracy computation

    Returns
    -------
    Dict[str, float]
        Dictionary with retrieval metrics (top-1, top-5, top-10, etc. accuracy)
    """
    metrics = {}

    # Get rank of correct answer for each spectrum
    correct_ranks = (
        rankings_df[~rankings_df["is_decoy"]].groupby("spec")["rank"].min()
    )

    # Compute top-k accuracy for various k values
    k_values = [1, 2, 3, 4, 5, 10, 15, 20, 50]
    k_values = [k for k in k_values if k <= k_max]

    for k in k_values:
        top_k_acc = (correct_ranks <= k).mean()
        metrics[f"top_{k}_accuracy"] = float(top_k_acc)

    # Mean Reciprocal Rank (MRR)
    mrr = (1.0 / correct_ranks).mean()
    metrics["mrr"] = float(mrr)

    # Median rank
    metrics["median_rank"] = float(correct_ranks.median())
    metrics["mean_rank"] = float(correct_ranks.mean())

    return metrics


def compute_rankings_for_metrics(
    df: pd.DataFrame, similarity_columns: List[str]
) -> pd.DataFrame:
    """Compute rankings for each similarity metric.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing similarity scores
    similarity_columns : List[str]
        List of column names containing similarity metrics

    Returns
    -------
    pd.DataFrame
        DataFrame with added rank columns for each similarity metric
    """
    for sim_col in similarity_columns:
        if sim_col in df.columns:
            # Distance metrics (lower = more similar) rank ascending
            # Similarity metrics (higher = more similar) rank descending
            is_distance = sim_col in DISTANCE_METRICS
            df[f"rank_{sim_col}"] = (
                df.groupby("spec")[sim_col]
                .rank(ascending=is_distance, method="min")
                .astype(int)
            )
    return df
