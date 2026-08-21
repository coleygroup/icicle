"""Jensen-Shannon similarity loss."""

import torch
import torch.nn as nn


class JensenShannonLoss(nn.Module):
    def __init__(self, lower_mz_cutoff: float = 0.0, **kwargs):
        super().__init__()
        self.lower_mz_cutoff = lower_mz_cutoff

    def forward(self, pred, targ, inten_buckets=None):
        """Jensen-Shannon similarity loss."""
        if inten_buckets is None:
            raise ValueError("inten_buckets required for Jensen-Shannon loss")

        mz_mask = inten_buckets >= self.lower_mz_cutoff
        masked_pred = pred * mz_mask
        masked_targ = targ * mz_mask

        # Normalize to probability distributions
        pred_norm = masked_pred / (
            torch.sum(masked_pred, dim=1, keepdim=True) + 1e-10
        )
        targ_norm = masked_targ / (
            torch.sum(masked_targ, dim=1, keepdim=True) + 1e-10
        )

        # Calculate mean distribution
        m = 0.5 * (pred_norm + targ_norm)

        # Calculate KL divergences
        def kl_div(p, q):
            mask = p > 1e-10
            p_safe = torch.where(mask, p, torch.ones_like(p))
            q_safe = torch.where(mask, q + 1e-10, torch.ones_like(q))
            return torch.sum(
                torch.where(
                    mask, p * torch.log(p_safe / q_safe), torch.zeros_like(p)
                ),
                dim=1,
            )

        js_div = 0.5 * (kl_div(pred_norm, m) + kl_div(targ_norm, m))
        similarity = 1 / (1 + torch.sqrt(js_div + 1e-10))
        loss = 1 - similarity

        return {"loss": loss}
