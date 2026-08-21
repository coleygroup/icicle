"""Metrics for comparing spectra, similarity metrics and distances."""

from typing import List, Literal, TypedDict

import numpy as np
import torch
import torch.nn.functional as F


# Type definitions
class MetricsDict(TypedDict):
    jaccard: List[float]
    coverage: List[float]
    inten_coverage: List[float]
    digitized_coverage: List[float]
    num_pred: List[int]
    num_true: List[int]


class TableDict(TypedDict):
    formula: List[str]
    formula_mass_no_adduct: List[float]
    rel_inten: List[float]


def cosine_similarity(predicted_vector, true_vector) -> np.ndarray:
    """Calculate cosine similarity between predicted and true spectra.

    LaTeX equation:

        \\text{similarity} = \\frac{P \\cdot T}{||P|| \\cdot ||T||}

    with $P$ the predicted spectrum, $T$ the true spectrum, and $||\\cdot||$ the Euclidean norm.

    Args:
        predicted_vector (np.ndarray): Predicted spectrum.
        true_vector (np.ndarray): True spectrum.

    Returns:
        np.ndarray: Cosine similarity between predicted and true spectra.
    """
    norm_pred = np.linalg.norm(predicted_vector) + 1e-10
    norm_true = np.linalg.norm(true_vector) + 1e-10
    return np.dot(predicted_vector, true_vector) / (norm_pred * norm_true)


def weighted_cosine_similarity(
    predicted_vector: np.ndarray,
    true_vector: np.ndarray,
    mz_values: np.ndarray,
    weighting_scheme: Literal[
        "none", "sqrt", "massbank", "nist_lc", "nist_gc"
    ] = "none",
    exclude_molecular_ion: bool = False,
) -> float:
    """Calculate weighted cosine similarity between predicted and true spectra.

    Args:
        predicted_vector: Intensities of predicted spectrum
        true_vector: Intensities of true spectrum
        mz_values: m/z values corresponding to the intensities
        weighting_scheme: Weighting scheme to use for m/z and intensities
        exclude_molecular_ion: If True, excludes the highest m/z peak

    Returns:
        float: Weighted cosine similarity between the spectra
    """
    # Define weighting parameters based on scheme
    weights = {
        "none": (0, 1),  # m/z^0 * I^1
        "sqrt": (0, 0.5),  # m/z^0 * I^0.5
        "massbank": (2, 0.5),  # m/z^2 * I^0.5
        "nist_lc": (1.3, 0.53),  # m/z^1.3 * I^0.53
        "nist_gc": (3, 0.6),  # m/z^3 * I^0.6
    }

    a, b = weights[weighting_scheme]

    if exclude_molecular_ion:
        # Find highest m/z peak and create mask
        max_mz_idx = np.argmax(mz_values)
        mask = np.ones_like(mz_values, dtype=bool)
        mask[max_mz_idx] = False

        # Apply mask to all vectors
        mz_values = mz_values[mask]
        predicted_vector = predicted_vector[mask]
        true_vector = true_vector[mask]

    # Calculate weighted vectors
    mz_weights = np.power(mz_values, a)
    pred_weighted = np.multiply(np.power(predicted_vector, b), mz_weights)
    true_weighted = np.multiply(np.power(true_vector, b), mz_weights)

    # Calculate similarity
    num = np.dot(pred_weighted, true_weighted)
    denom = np.sqrt(np.sum(np.square(pred_weighted))) * np.sqrt(
        np.sum(np.square(true_weighted))
    )

    return num / (denom + 1e-10)  # Add small constant for numerical stability


def composite_weighted_cosine_similarity(
    predicted_vector: np.ndarray,
    true_vector: np.ndarray,
    mz_values: np.ndarray,
    weighting_scheme: str = "none",
) -> float:
    """Calculate composite weighted cosine similarity (identity) between
    spectra.

    Args:
        predicted_vector: Intensities of predicted spectrum
        true_vector: Intensities of true spectrum
        mz_values: m/z values corresponding to the intensities
        weighting_scheme: Weighting scheme for the cosine similarity component

    Returns:
        float: Composite weighted cosine similarity
    """
    # Calculate basic weighted cosine similarity
    cosine_sim = weighted_cosine_similarity(
        predicted_vector, true_vector, mz_values, weighting_scheme
    )

    # Calculate overlap (number of matching signals)
    overlap = np.sum((predicted_vector > 0) & (true_vector > 0))

    # Calculate ratio factors for adjacent peaks
    def get_intensity_ratios(spectrum: np.ndarray) -> np.ndarray:
        ratios = np.zeros_like(spectrum)
        mask = spectrum[:-1] > 0
        ratios[1:][mask] = spectrum[1:][mask] / spectrum[:-1][mask]
        return ratios

    pred_ratios = get_intensity_ratios(predicted_vector)
    true_ratios = get_intensity_ratios(true_vector)

    # Calculate ratio factor
    ratio_factor = np.zeros_like(predicted_vector)
    valid_mask = (pred_ratios > 0) & (true_ratios > 0)
    ratio_factor[valid_mask] = np.minimum(
        pred_ratios[valid_mask], true_ratios[valid_mask]
    ) / np.maximum(pred_ratios[valid_mask], true_ratios[valid_mask])

    # Calculate final composite similarity
    N = len(predicted_vector)
    composite_sim = (N * cosine_sim + overlap * np.mean(ratio_factor)) / (
        N + overlap
    )

    return composite_sim


def entropy_distance(
    predicted_vector: np.ndarray, true_vector: np.ndarray
) -> np.ndarray:
    """Calculate entropy-based distance between predicted and true spectra
    (with numerical stability).

    LaTeX equation:

        \\text{distance} = \\frac{2 \\cdot H(\\frac{P + T}{2}) - H(P) - H(T)}{\\log(4)}

    with $H$ the entropy function, $P$ the predicted spectrum, $T$ the true spectrum, and $\\log$ the natural logarithm.

    The entropy function is defined as:

        $H(x) = - \\sum_{i} x_i \\cdot \\log(x_i)$

    Args:
        predicted_vector (np.ndarray): Predicted spectrum.
        true_vector (np.ndarray): True spectrum.

    Returns:
        np.ndarray: Entropy-based distance between predicted and true spectra.
    """

    def normalize_peaks(prob: np.ndarray) -> np.ndarray:
        """Normalize peaks of a spectrum by dividing by the sum of all peaks.

        Args:
            prob (np.ndarray): Spectrum.

        Returns:
            np.ndarray: Normalized spectrum.
        """
        # Add small constant to avoid division by zero
        sum_prob = prob.sum(axis=-1, keepdims=True)
        sum_prob = np.where(sum_prob < 1e-10, 1e-10, sum_prob)
        return prob / sum_prob

    def entropy(prob: np.ndarray) -> np.ndarray:
        """Calculate entropy of a spectrum.

        Args:
            prob (np.ndarray): Spectrum.

        Returns:
            np.ndarray: Entropy of the spectrum.
        """
        # Avoid log(0) by masking zeros
        mask = prob > 1e-10
        safe_prob = np.where(mask, prob, 1.0)
        safe_log = np.where(mask, np.log(safe_prob), 0.0)
        return -np.sum(prob * safe_log, axis=-1)

    norm_pred = normalize_peaks(predicted_vector)
    norm_true = normalize_peaks(true_vector)
    entropy_pred = entropy(norm_pred)
    entropy_targ = entropy(norm_true)
    entropy_mix = entropy((norm_pred + norm_true) / 2)

    distance = (2 * entropy_mix - entropy_pred - entropy_targ) / np.log(4)
    # Ensure distance is non-negative and finite
    distance = np.where(np.isfinite(distance), distance, 1.0)
    distance = np.maximum(distance, 0.0)

    return distance


def entropy_similarity(
    predicted_vector: np.ndarray, true_vector: np.ndarray
) -> np.ndarray:
    """Calculate entropy-based similarity between predicted and true spectra.

    LaTeX equation:

        \\text{similarity} = 1 - \\frac{2 \\cdot H(\\frac{P + T}{2}) - H(P) - H(T)}{\\log(4)}

    with $H$ the entropy function, $P$ the predicted spectrum, $T$ the true spectrum, and $\\log$ the natural logarithm.

    The entropy function is defined as:

        $H(x) = - \\sum_{i} x_i \\cdot \\log(x_i)$

    Args:
        predicted_vector (np.ndarray): Predicted spectrum.
        true_vector (np.ndarray): True spectrum.

    Returns:
        np.ndarray: Entropy-based similarity between predicted and true spectra.
    """
    distance = entropy_distance(predicted_vector, true_vector)
    similarity = 1.0 - distance
    return similarity


def spectral_contrast_angle(
    predicted_vector: np.ndarray, true_vector: np.ndarray
) -> np.ndarray:
    """Calculate spectral contrast angle between predicted and true spectra.

    LaTeX equation:

        \\text{SCA} = \\arccos \\left( \\frac{P \\cdot T}{||P|| \\cdot ||T||} \\right)

    with $P$ the predicted spectrum, $T$ the true spectrum, and $||\\cdot||$ the Euclidean norm.

    Args:
        predicted_vector (np.ndarray): Predicted spectrum.
        true_vector (np.ndarray): True spectrum.

    Returns:
        np.ndarray: Spectral contrast angle between predicted and true spectra.
    """
    norm_pred = np.linalg.norm(predicted_vector) + 1e-10
    norm_true = np.linalg.norm(true_vector) + 1e-10
    cosine_angle = np.dot(predicted_vector, true_vector) / (
        norm_pred * norm_true
    )
    # Clip cosine value to avoid numerical issues with arccos
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    return np.arccos(cosine_angle)


def mean_squared_error(
    predicted_vector: np.ndarray, true_vector: np.ndarray
) -> np.ndarray:
    """Calculate mean squared error between predicted and true spectra.

    LaTeX equation:

        \\text{MSE} = \\frac{1}{n} \\sum_{i=1}^{n} (P_i - T_i)^2

    with $P$ the predicted spectrum, $T$ the true spectrum, and $n$ the number of peaks.

    Args:
        predicted_vector (np.ndarray): Predicted spectrum.
        true_vector (np.ndarray): True spectrum.

    Returns:
        np.ndarray: Mean squared error between predicted and true spectra.
    """
    mse = np.mean((predicted_vector - true_vector) ** 2)
    return mse


# ---------------------------------------------------------------------------
# Batched tensor variants for GPU retrieval (N_cands × N_queries)
# ---------------------------------------------------------------------------


def batch_cosine_sim(
    cands: torch.Tensor, queries: torch.Tensor, **_
) -> torch.Tensor:
    """Cosine similarity matrix [N_cands, N_queries].

    Args:
        cands: Candidate spectra [N_cands, n_bins].
        queries: Query spectra [N_queries, n_bins].

    Returns:
        torch.Tensor: Similarity matrix [N_cands, N_queries].
    """
    return F.normalize(cands, dim=-1) @ F.normalize(queries, dim=-1).T


def batch_entropy_sim(
    cands: torch.Tensor,
    queries: torch.Tensor,
    query_batch_size: int = 64,
    **_,
) -> torch.Tensor:
    """Entropy similarity matrix [N_cands, N_queries].

    Equivalent to 1 - entropy_distance, computed in batches to bound
    peak memory to O(N_cands × query_batch_size × n_bins).

    Args:
        cands: Candidate spectra [N_cands, n_bins].
        queries: Query spectra [N_queries, n_bins].
        query_batch_size: Number of queries to process at once.

    Returns:
        torch.Tensor: Similarity matrix [N_cands, N_queries].
    """
    N, Q = cands.shape[0], queries.shape[0]
    out = torch.empty(N, Q, device=cands.device, dtype=torch.float32)
    p_norm = cands / (cands.sum(-1, keepdim=True) + 1e-10)

    def _H(x: torch.Tensor) -> torch.Tensor:
        return -(x * x.clamp(min=1e-10).log()).sum(dim=-1)

    for qi in range(0, Q, query_batch_size):
        qe = min(qi + query_batch_size, Q)
        q_norm = queries[qi:qe] / (
            queries[qi:qe].sum(-1, keepdim=True) + 1e-10
        )
        p_exp = p_norm.unsqueeze(1).expand(-1, qe - qi, -1)
        q_exp = q_norm.unsqueeze(0).expand(N, -1, -1)
        m = 0.5 * (p_exp + q_exp)
        ed = (2 * _H(m) - _H(p_exp) - _H(q_exp)) / torch.tensor(4.0).log()
        out[:, qi:qe] = torch.clamp(1.0 - ed, 0.0, 1.0)

    return out


def batch_weighted_cosine_sim(
    cands: torch.Tensor, queries: torch.Tensor, mz_weights: torch.Tensor, **_
) -> torch.Tensor:
    """Weighted cosine similarity matrix [N_cands, N_queries].

    Applies sqrt intensity weighting and linear m/z weighting (mz^1 * I^0.5),
    matching the "massbank"-style scheme without the squared m/z factor.

    Args:
        cands: Candidate spectra [N_cands, n_bins].
        queries: Query spectra [N_queries, n_bins].
        mz_weights: m/z bin values [n_bins]; used as linear weights.

    Returns:
        torch.Tensor: Similarity matrix [N_cands, N_queries].
    """
    cw = F.normalize(cands.pow(0.5) * mz_weights, dim=-1)
    qw = F.normalize(queries.pow(0.5) * mz_weights, dim=-1)
    return cw @ qw.T


def batch_composite_sim(
    cands: torch.Tensor,
    queries: torch.Tensor,
    mz_weights: torch.Tensor,
    query_batch_size: int = 64,
    **_,
) -> torch.Tensor:
    """Composite similarity matrix [N_cands, N_queries].

    Matches the numpy ``composite_weighted_cosine_similarity`` formula::

        (N * weighted_cosine + overlap * mean_ratio_factor) / (N + overlap)

    where ``overlap`` = number of bins both spectra have non-zero intensity,
    and ``ratio_factor[b] = min(r_c[b], r_q[b]) / max(r_c[b], r_q[b])``
    for adjacent-bin intensity ratios ``r[b] = I[b] / I[b-1]``.

    Computed in query batches to bound peak memory to
    O(N_cands × query_batch_size × n_bins).

    Args:
        cands: Candidate spectra [N_cands, n_bins].
        queries: Query spectra [N_queries, n_bins].
        mz_weights: m/z bin values [n_bins]; used as linear weights.
        query_batch_size: Number of queries to process at once.

    Returns:
        torch.Tensor: Similarity matrix [N_cands, N_queries].
    """
    N, Q, B = cands.shape[0], queries.shape[0], cands.shape[1]
    out = torch.empty(N, Q, device=cands.device, dtype=torch.float32)

    cw = F.normalize(cands.pow(0.5) * mz_weights, dim=-1)  # [N, B]

    def _ratios(x: torch.Tensor) -> torch.Tensor:
        """Adjacent-bin intensity ratios; 0 where denominator is zero."""
        r = torch.zeros_like(x)
        mask = x[:, :-1] > 0
        r[:, 1:][mask] = x[:, 1:][mask] / x[:, :-1][mask]
        return r

    c_ratios = _ratios(cands)  # [N, B]

    for qi in range(0, Q, query_batch_size):
        qe = min(qi + query_batch_size, Q)
        q_batch = queries[qi:qe]  # [Qb, B]
        Qb = qe - qi

        qw_batch = F.normalize(q_batch.pow(0.5) * mz_weights, dim=-1)
        wcos = cw @ qw_batch.T  # [N, Qb]

        # overlap: number of bins both spectra are non-zero [N, Qb]
        c_nz = (cands > 0).unsqueeze(1).expand(-1, Qb, -1)  # [N, Qb, B]
        q_nz = (q_batch > 0).unsqueeze(0).expand(N, -1, -1)  # [N, Qb, B]
        overlap = (c_nz & q_nz).sum(dim=-1).float()  # [N, Qb]

        # ratio factor: min/max of adjacent-bin ratios where both are positive
        q_ratios = _ratios(q_batch)  # [Qb, B]
        c_r = c_ratios.unsqueeze(1).expand(-1, Qb, -1)  # [N, Qb, B]
        q_r = q_ratios.unsqueeze(0).expand(N, -1, -1)  # [N, Qb, B]
        valid = (c_r > 0) & (q_r > 0)
        rf = torch.zeros(N, Qb, B, device=cands.device, dtype=torch.float32)
        rf[valid] = torch.minimum(c_r[valid], q_r[valid]) / torch.maximum(
            c_r[valid], q_r[valid]
        )
        mean_rf = rf.mean(dim=-1)  # [N, Qb]

        out[:, qi:qe] = (B * wcos + overlap * mean_rf) / (B + overlap + 1e-10)

    return out
