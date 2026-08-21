"""Helper for binomial coefficient (if torch.special.binom is not available)"""

import torch


def safe_log_binom(n: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Computes log(C(n, k)) robustly using torch.lgamma.

    Handles n < k cases by returning -inf (log of 0). n and k can be tensors.
    """
    # Ensure inputs are float for lgamma
    n = n.float()
    k = k.float()

    # Create a mask for valid (n >= k and n >= 0 and k >= 0) cases
    # For binomial, n and k must be non-negative integers
    valid_mask = (
        (n >= k) & (n >= 0) & (k >= 0) & (k == k.int()) & (n == n.int())
    )

    # Initialize result with -inf (log of 0 for invalid combinations)
    log_binom = torch.full_like(n, float("-inf"))

    # Compute lgamma only for valid cases to avoid NaNs
    n_plus_1 = n + 1
    k_plus_1 = k + 1
    n_minus_k_plus_1 = n - k + 1

    # Ensure inputs to lgamma are at least 1 for non-negative factorial (0! = 1)
    # lgamma(0) is inf, lgamma(1) is 0.
    # For non-integer k, this can be complex. Assuming integer k as per binomial def.

    # Calculate lgamma terms for valid cases
    lgamma_n_plus_1 = torch.where(
        valid_mask, torch.lgamma(n_plus_1), torch.tensor(0.0, device=n.device)
    )
    lgamma_k_plus_1 = torch.where(
        valid_mask, torch.lgamma(k_plus_1), torch.tensor(0.0, device=k.device)
    )
    lgamma_n_minus_k_plus_1 = torch.where(
        valid_mask,
        torch.lgamma(n_minus_k_plus_1),
        torch.tensor(0.0, device=n.device),
    )

    # Compute log_binom for valid cases
    log_binom_valid = (
        lgamma_n_plus_1 - lgamma_k_plus_1 - lgamma_n_minus_k_plus_1
    )

    # Update log_binom with values for valid cases
    log_binom = torch.where(valid_mask, log_binom_valid, log_binom)

    # Handle cases where lgamma(0) might lead to inf.
    # For example, C(0,0) = 1, log(1) = 0.
    # lgamma(1) is 0, so lgamma(0+1) - lgamma(0+1) - lgamma(0+1) = 0 - 0 - 0 = 0.
    # This should work for C(0,0).

    return log_binom


def safe_binom(n: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Computes C(n, k) = n!

    / (k! * (n-k)!) robustly. Handles n < k by returning 0.
    """
    return torch.exp(safe_log_binom(n, k))
