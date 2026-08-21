"""Mean Squared Error loss."""

import torch.nn as nn


class MSELoss(nn.Module):
    def __init__(self, lower_mz_cutoff: float = 0.0, **kwargs):
        super().__init__()
        self.lower_mz_cutoff = lower_mz_cutoff
        self.mse = nn.MSELoss(reduction="none")

    def forward(self, pred, targ, inten_buckets=None):
        """Mean Squared Error loss with optional m/z cutoff."""
        # Apply m/z cutoff if specified and buckets available
        if self.lower_mz_cutoff > 0 and inten_buckets is not None:
            mz_mask = inten_buckets >= self.lower_mz_cutoff
            pred = pred * mz_mask
            targ = targ * mz_mask

        # Compute MSE loss
        loss = self.mse(pred, targ).mean(dim=1)

        return {"loss": loss}
