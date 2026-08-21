"""Random spectrum baseline."""

from ..base_model import BaseSpectrumPredictor
import numpy as np
import torch
from typing import Any, Dict


class RandomSpectrumBaseline(BaseSpectrumPredictor):
    """Simple baseline that predicts random spectra."""

    def __init__(
        self, min_mz: float, max_mz: float, bin_width: float, **kwargs
    ):
        super().__init__(
            min_mz=min_mz, max_mz=max_mz, bin_width=bin_width, **kwargs
        )
        self.num_bins = int((max_mz - min_mz) / bin_width)
        self.mz_bins = np.linspace(
            min_mz, max_mz, self.num_bins, endpoint=False
        ).astype(np.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return random spectrum for any input."""
        batch_size = x.shape[0] if x.dim() > 1 else 1
        return torch.abs(torch.randn(batch_size, self.num_bins))

    def predict_mol(self, smi: str | list[str], device: str = "cpu", **kwargs):
        """Predict random spectrum for any molecule."""
        if isinstance(smi, str):
            smi = [smi]
            batched = False
        else:
            batched = True

        predictions = torch.abs(torch.randn(len(smi), self.num_bins)).to(
            device
        )
        return self._format_output(predictions, batched)

    def predict_from_smiles(
        self, smiles: str, device: str = "cpu", **kwargs
    ) -> Dict[str, Any]:
        """Predict random spectrum for a SMILES string.

        Matches the interface expected by eval.py.
        """
        # Generate random positive intensities and normalize
        intensities = np.abs(np.random.randn(self.num_bins)).astype(np.float32)
        intensities = (
            intensities / intensities.max()
            if intensities.max() > 0
            else intensities
        )

        return {
            "smiles": smiles,
            "mz_bins": self.mz_bins,
            "intensities": intensities,
            "num_fragments": 0,
            "fragments": {},
        }

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Dummy training step - this baseline doesn't actually train."""
        return torch.tensor(0.0, requires_grad=True)

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Dummy validation step."""
        return torch.tensor(0.0)

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Dummy test step."""
        return torch.tensor(0.0)

    def configure_optimizers(self):
        # Configure dummy optimizer, no parameters to optimize
        return None
