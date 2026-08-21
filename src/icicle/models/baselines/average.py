"""Average spectrum baseline.

Predicts the average of all training spectra for any input molecule.
"""

import logging
from typing import Any, Dict, Optional

import h5py
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from icicle.data.transforms.spectrum import SpecBinner
from icicle.models.base_model import BaseSpectrumPredictor


class AverageSpectrumBaseline(BaseSpectrumPredictor):
    """Baseline that always predicts the training set average spectrum.

    Can be initialized with a pre-computed average spectrum tensor, or compute
    it from training data paths.
    """

    def __init__(
        self,
        min_mz: float,
        max_mz: float,
        bin_width: float,
        average_spectrum: Optional[torch.Tensor] = None,
        spectra_path: Optional[str] = None,
        labels_path: Optional[str] = None,
        splits_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            min_mz=min_mz, max_mz=max_mz, bin_width=bin_width, **kwargs
        )
        self.num_bins = int((max_mz - min_mz) / bin_width)
        self.mz_bins = np.linspace(
            min_mz, max_mz, self.num_bins, endpoint=False
        ).astype(np.float32)

        if average_spectrum is not None:
            self.register_buffer("average_spectrum", average_spectrum)
        elif spectra_path and labels_path and splits_path:
            avg_spec = self._compute_average_spectrum(
                spectra_path, labels_path, splits_path
            )
            self.register_buffer("average_spectrum", avg_spec)
        else:
            # Initialize with zeros - will need to be set later
            self.register_buffer(
                "average_spectrum",
                torch.zeros(self.num_bins, dtype=torch.float32),
            )
            logging.warning(
                "AverageSpectrumBaseline initialized without average spectrum. "
                "Provide average_spectrum tensor or paths to compute from data."
            )

    def _compute_average_spectrum(
        self, spectra_path: str, labels_path: str, splits_path: str
    ) -> torch.Tensor:
        """Compute average spectrum from training data."""
        logging.info("Computing average spectrum from training data...")

        # Load splits to get training mol_ids
        splits_df = pd.read_csv(splits_path, sep="\t")
        train_mol_ids = set(
            splits_df[splits_df["split"] == "train"]["mol_id"]
            .astype(str)
            .tolist()
        )

        # Load labels to get mol_id mapping
        labels_df = pd.read_csv(labels_path, sep="\t")
        labels_df["mol_id"] = labels_df["mol_id"].astype(str)

        # Filter to training molecules
        train_labels = labels_df[labels_df["mol_id"].isin(train_mol_ids)]

        # Create binner
        binner = SpecBinner(
            min_mz=self.min_mz, max_mz=self.max_mz, bin_width=self.bin_width
        )

        # Accumulate spectra
        spectra_sum = np.zeros(self.num_bins, dtype=np.float64)
        count = 0

        with h5py.File(spectra_path, "r") as hf:
            for _, row in tqdm(
                train_labels.iterrows(), total=len(train_labels)
            ):
                mol_id = str(row["mol_id"])
                if mol_id in hf:
                    group = hf[mol_id]
                    raw_mz = group["masses"][:]
                    raw_intensities = group["intensities"][:]

                    # Bin the spectrum
                    binned_result = binner(raw_mz, raw_intensities)
                    binned_intensities = binned_result["spectrum"].numpy()

                    # Normalize to max=1 before averaging
                    if binned_intensities.max() > 0:
                        binned_intensities = (
                            binned_intensities / binned_intensities.max()
                        )

                    spectra_sum += binned_intensities
                    count += 1

        if count > 0:
            avg_spectrum = spectra_sum / count
            # Normalize final average
            if avg_spectrum.max() > 0:
                avg_spectrum = avg_spectrum / avg_spectrum.max()
        else:
            avg_spectrum = np.zeros(self.num_bins)
            logging.warning("No training spectra found!")

        logging.info(f"Computed average from {count} training spectra")
        return torch.from_numpy(avg_spectrum.astype(np.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return average spectrum for any input."""
        batch_size = x.shape[0] if x.dim() > 1 else 1
        return self.average_spectrum.unsqueeze(0).repeat(batch_size, 1)

    def predict_mol(self, smi: str | list[str], device: str = "cpu", **kwargs):
        """Predict average spectrum for any molecule."""
        if isinstance(smi, str):
            smi = [smi]
            batched = False
        else:
            batched = True

        predictions = (
            self.average_spectrum.unsqueeze(0).repeat(len(smi), 1).to(device)
        )
        return self._format_output(predictions, batched)

    def predict_from_smiles(
        self, smiles: str, device: str = "cpu", **kwargs
    ) -> Dict[str, Any]:
        """Predict average spectrum for a SMILES string.

        Matches the interface expected by eval.py.
        """
        intensities = self.average_spectrum.cpu().numpy()

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
        """No optimizer needed."""
        return None
