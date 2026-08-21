"""Baseline: Full Enumeration Bar Code.

Uses MAGMa full enumeration to get fragment masses, then predicts intensity=1
for each fragment mass bin (barcode spectrum).
"""

import logging
from typing import Any, Dict

import numpy as np
import torch

from icicle.data.fragmentation_engine import (
    FragmentEngine,
    FragmentationParams,
)
from icicle.models.base_model import BaseSpectrumPredictor


class FullEnumerationBarCode(BaseSpectrumPredictor):
    """Baseline that uses MAGMa fragmentation with uniform intensity=1.

    For each molecule:
    1. Run MAGMa full enumeration to get all possible fragments
    2. For each fragment mass, set intensity=1 in the corresponding bin
    3. Normalize so max intensity = 1
    """

    def __init__(
        self,
        min_mz: float,
        max_mz: float,
        bin_width: float,
        max_tree_depth: int = 3,
        max_broken_bonds: int = 6,
        num_h_shifts: int = 6,
        **kwargs,
    ):
        super().__init__(
            min_mz=min_mz, max_mz=max_mz, bin_width=bin_width, **kwargs
        )
        self.num_bins = int((max_mz - min_mz) / bin_width)
        self.mz_bins = np.linspace(
            min_mz, max_mz, self.num_bins, endpoint=False
        ).astype(np.float32)
        self.max_tree_depth = max_tree_depth
        self.max_broken_bonds = max_broken_bonds
        self.num_h_shifts = num_h_shifts

    def _mass_to_bin(self, mass: float) -> int:
        """Convert a mass value to its bin index."""
        if mass < self.min_mz or mass >= self.max_mz:
            return -1
        return int((mass - self.min_mz) / self.bin_width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Not used for this baseline - use predict_from_smiles instead."""
        raise NotImplementedError("Use predict_from_smiles for this baseline")

    def predict_mol(self, smi: str | list[str], device: str = "cpu", **kwargs):
        """Predict barcode spectrum for molecules."""
        if isinstance(smi, str):
            smi = [smi]
            batched = False
        else:
            batched = True

        predictions = []
        for s in smi:
            result = self.predict_from_smiles(s, device=device, **kwargs)
            predictions.append(torch.from_numpy(result["intensities"]))

        predictions = torch.stack(predictions).to(device)
        return self._format_output(predictions, batched)

    def predict_from_smiles(
        self, smiles: str, device: str = "cpu", **kwargs
    ) -> Dict[str, Any]:
        """Predict barcode spectrum for a SMILES string.

        Matches the interface expected by eval.py.
        """
        intensities = np.zeros(self.num_bins, dtype=np.float32)
        num_fragments = 0
        fragments_info = {}

        try:
            # Initialize fragment engine with MAGMa
            engine = FragmentEngine(
                mol_str=smiles,
                params=FragmentationParams(
                    max_tree_depth=self.max_tree_depth,
                    max_broken_bonds=self.max_broken_bonds,
                    num_h_shifts=self.num_h_shifts,
                    detect_isotope_patterns=False,
                ),
            )
            engine.generate_fragments()

            # Get all fragment entries
            frag_entries = engine.frag_to_entry
            num_fragments = len(frag_entries)

            # For each fragment, set intensity=1 in its mass bin
            for frag_hash, entry in frag_entries.items():
                base_mass = entry.base_mass

                # Apply H-shifts to get all possible masses for this fragment
                for h_shift in range(
                    -self.num_h_shifts, self.num_h_shifts + 1
                ):
                    mass = base_mass + h_shift
                    bin_idx = self._mass_to_bin(mass)
                    if 0 <= bin_idx < self.num_bins:
                        intensities[bin_idx] = 1.0

                fragments_info[frag_hash] = {
                    "base_mass": base_mass,
                    "form": entry.form,
                }

            # Normalize so max = 1 (already 1 for barcode, but just in case)
            if intensities.max() > 0:
                intensities = intensities / intensities.max()

        except Exception as e:
            logging.warning(f"Fragmentation failed for {smiles}: {e}")
            # Return empty spectrum on failure

        return {
            "smiles": smiles,
            "mz_bins": self.mz_bins,
            "intensities": intensities,
            "num_fragments": num_fragments,
            "fragments": fragments_info,
        }

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Dummy training step - this baseline doesn't train."""
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
