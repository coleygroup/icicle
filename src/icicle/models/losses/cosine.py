"""Cosine similarity loss functions.

Uses different weighting schemes for the cosine similarity loss, and also a
lower m/z cutoff (ignoring low m/z peaks).
"""

import torch
import torch.nn as nn


class CosineSimilarityLoss(nn.Module):
    def __init__(self, lower_mz_cutoff: float = 0.0, **kwargs):
        super().__init__()
        self.lower_mz_cutoff = lower_mz_cutoff

    def forward(self, pred, targ, inten_buckets=None):
        """Simplified cosine similarity loss."""
        # Apply m/z cutoff if specified
        if self.lower_mz_cutoff > 0 and inten_buckets is not None:
            mz_mask = inten_buckets >= self.lower_mz_cutoff
            pred = pred * mz_mask
            targ = targ * mz_mask

        # Add small epsilon to prevent zero vectors
        epsilon = 1e-10
        pred = pred + epsilon
        targ = targ + epsilon

        # Compute cosine similarity manually with proper handling
        pred_norm = pred / torch.norm(pred, dim=1, keepdim=True)
        targ_norm = targ / torch.norm(targ, dim=1, keepdim=True)

        similarity = torch.sum(pred_norm * targ_norm, dim=1)
        # Clamp to valid cosine range
        similarity = torch.clamp(similarity, -1.0, 1.0)

        loss = 1 - similarity

        return {"loss": loss}


class WeightedCosineSimilarityLoss(nn.Module):
    def __init__(
        self, weighting: str = "sqrt", lower_mz_cutoff: float = 0.0, **kwargs
    ):
        super().__init__()
        self.weighting = weighting
        self.lower_mz_cutoff = lower_mz_cutoff

        # Define weighting parameters
        self.weights = {
            "sqrt": (0, 0.5),  # m/z^0 * I^0.5
            "massbank": (2, 0.5),  # m/z^2 * I^0.5
            "nist_lc": (1.3, 0.53),  # m/z^1.3 * I^0.53
            "nist_gc": (3, 0.6),  # m/z^3 * I^0.6
        }

        if weighting not in self.weights:
            raise ValueError(f"Unknown weighting scheme: {weighting}")

    def forward(self, pred, targ, inten_buckets=None):
        """Weighted cosine similarity loss with different weighting schemes."""
        if inten_buckets is None:
            raise ValueError("inten_buckets required for weighted cosine loss")

        # Apply m/z cutoff
        mz_mask = inten_buckets >= self.lower_mz_cutoff

        # Ensure non-negative inputs
        pred = torch.relu(pred)
        targ = torch.relu(targ)

        masked_pred = pred * mz_mask
        masked_targ = targ * mz_mask

        # Add small epsilon to avoid zero inputs
        epsilon = 1e-10
        masked_pred = masked_pred + epsilon
        masked_targ = masked_targ + epsilon

        # Get weighting parameters
        a, b = self.weights[self.weighting]

        # Calculate m/z weights
        mz_values = inten_buckets
        mz_weights = torch.pow(mz_values, a)

        # Apply intensity weighting with safe power operation
        pred_weighted = torch.mul(torch.pow(masked_pred, b), mz_weights)
        targ_weighted = torch.mul(torch.pow(masked_targ, b), mz_weights)

        # Calculate similarity with numerical stability
        num = torch.sum(pred_weighted * targ_weighted, dim=1)
        denom = torch.sqrt(
            torch.sum(torch.square(pred_weighted), dim=1) + epsilon
        ) * torch.sqrt(torch.sum(torch.square(targ_weighted), dim=1) + epsilon)

        similarity = num / denom
        loss = 1 - similarity

        return {"loss": loss}


class CompositeWeightedCosineSimilarityLoss(nn.Module):
    def __init__(
        self, weighting: str = "sqrt", lower_mz_cutoff: float = 0.0, **kwargs
    ):
        super().__init__()
        self.weighting = weighting
        self.lower_mz_cutoff = lower_mz_cutoff

        # Use the weighted cosine loss as a component
        self.weighted_cosine = WeightedCosineSimilarityLoss(
            weighting=weighting, lower_mz_cutoff=lower_mz_cutoff
        )

    def forward(self, pred, targ, inten_buckets=None):
        """Composite similarity loss combining weighted cosine and ratio
        factors."""
        if inten_buckets is None:
            raise ValueError(
                "inten_buckets required for composite weighted cosine loss"
            )

        # Ensure non-negative inputs and add small epsilon
        epsilon = 1e-10
        pred = torch.relu(pred) + epsilon
        targ = torch.relu(targ) + epsilon

        # Get basic weighted cosine similarity
        cosine_result = self.weighted_cosine(pred, targ, inten_buckets)
        cosine_sim = 1 - cosine_result["loss"]

        # Calculate overlap with threshold
        overlap = torch.sum((pred > epsilon) & (targ > epsilon), dim=1).float()

        # Calculate intensity ratios
        def get_intensity_ratios(spectrum):
            ratios = torch.zeros_like(spectrum)
            # Use epsilon threshold for masking
            mask = spectrum[:, :-1] > epsilon
            # Add epsilon to denominator for stability
            ratios[:, 1:][mask] = spectrum[:, 1:][mask] / (
                spectrum[:, :-1][mask] + epsilon
            )
            return ratios

        pred_ratios = get_intensity_ratios(pred)
        targ_ratios = get_intensity_ratios(targ)

        # Calculate ratio factor with numerical stability
        ratio_factor = torch.zeros_like(pred)
        valid_mask = (pred_ratios > epsilon) & (targ_ratios > epsilon)
        min_ratios = torch.minimum(
            pred_ratios[valid_mask], targ_ratios[valid_mask]
        )
        max_ratios = (
            torch.maximum(pred_ratios[valid_mask], targ_ratios[valid_mask])
            + epsilon
        )
        ratio_factor[valid_mask] = min_ratios / max_ratios

        # Calculate final composite similarity
        N = pred.shape[1]
        # Add epsilon to denominator for stability
        composite_sim = (
            N * cosine_sim + overlap * torch.mean(ratio_factor, dim=1)
        ) / (N + overlap + epsilon)
        loss = 1 - composite_sim

        return {"loss": loss}
