"""Entropy-based distance loss."""

import torch
import torch.nn as nn


class EntropyLoss(nn.Module):
    def __init__(self, lower_mz_cutoff: float = 0.0, **kwargs):
        super().__init__()
        self.lower_mz_cutoff = lower_mz_cutoff

    def forward(self, pred, targ, inten_buckets=None):
        """Entropy-based distance loss."""
        if inten_buckets is None:
            raise ValueError("inten_buckets required for entropy loss")

        def normalize_peaks(prob):
            sum_prob = torch.sum(prob, dim=1, keepdim=True)
            sum_prob = torch.where(
                sum_prob < 1e-10, torch.ones_like(sum_prob) * 1e-10, sum_prob
            )
            return prob / sum_prob

        def entropy(prob):
            mask = prob > 1e-10
            safe_prob = torch.where(mask, prob, torch.ones_like(prob))
            safe_log = torch.where(
                mask, torch.log(safe_prob), torch.zeros_like(prob)
            )
            return -torch.sum(prob * safe_log, dim=1)

        mz_mask = inten_buckets >= self.lower_mz_cutoff
        masked_pred = pred * mz_mask
        masked_targ = targ * mz_mask

        norm_pred = normalize_peaks(masked_pred)
        norm_targ = normalize_peaks(masked_targ)

        entropy_pred = entropy(norm_pred)
        entropy_targ = entropy(norm_targ)
        entropy_mix = entropy((norm_pred + norm_targ) / 2)

        distance = (2 * entropy_mix - entropy_pred - entropy_targ) / torch.log(
            torch.tensor(4.0)
        )

        # Ensure distance is non-negative and finite
        distance = torch.where(
            torch.isfinite(distance), distance, torch.ones_like(distance)
        )
        distance = torch.maximum(distance, torch.zeros_like(distance))

        return {"loss": distance}
