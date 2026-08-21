"""Spectrum transforms."""

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch

from icicle.data.transforms.base import SpecTransform


class SpecBinner(SpecTransform):
    """Bin spectrum into fixed-width bins."""

    def __init__(self, min_mz: float, max_mz: float, bin_width: float):
        self.min_mz = min_mz
        self.max_mz = max_mz
        self.bin_width = bin_width
        self.num_bins = int((max_mz - min_mz) / bin_width)

    def __call__(
        self, mz: np.ndarray, intensities: np.ndarray, **metadata
    ) -> Dict[str, torch.Tensor]:
        spectrum = np.zeros(self.num_bins, dtype=np.float32)

        for m, intensity in zip(mz, intensities):
            if m >= self.min_mz and m < self.max_mz:
                bin_idx = int(m / self.bin_width)
                if 0 <= bin_idx < self.num_bins:
                    spectrum[bin_idx] += intensity

        # Always normalize
        if spectrum.max() > 0:
            spectrum = spectrum / spectrum.max()

        return {"spectrum": torch.tensor(spectrum)}


class SpecTokenizer(SpecTransform):
    """Represent spectrum as (mz, intensity) pairs."""

    def __init__(self, n_peaks, mz_range: Tuple[float, float]):
        self.n_peaks = n_peaks
        self.mz_range = mz_range

    def __call__(
        self, mz: np.ndarray, intensities: np.ndarray, **metadata
    ) -> Dict[str, torch.Tensor]:
        # Filter and sort by intensity
        mask = (mz >= self.mz_range[0]) & (mz <= self.mz_range[1])
        mz_filtered = mz[mask]
        int_filtered = intensities[mask]

        if len(mz_filtered) > self.n_peaks:
            top_indices = np.argsort(int_filtered)[-self.n_peaks :]
            mz_filtered = mz_filtered[top_indices]
            int_filtered = int_filtered[top_indices]

        # Normalize
        if len(int_filtered) > 0:
            int_filtered = int_filtered / int_filtered.max()
            spec_matrix = np.column_stack([mz_filtered, int_filtered])
        else:
            spec_matrix = np.empty((0, 2))

        # Pad
        padded = np.zeros((self.n_peaks, 2), dtype=np.float32)
        if len(spec_matrix) > 0:
            copy_size = min(len(spec_matrix), self.n_peaks)
            padded[:copy_size] = spec_matrix[:copy_size]

        return {"spectrum": torch.tensor(padded)}


class SpecSparse(SpecTransform):
    """Sparse spectrum representation."""

    def __init__(
        self,
        n_peaks: Optional[int],
        mz_range: Tuple[float, float],
    ):
        self.n_peaks = n_peaks
        self.mz_range = mz_range

    def __call__(
        self, mz: np.ndarray, intensities: np.ndarray, **metadata
    ) -> Dict[str, torch.Tensor]:
        # Filter
        mask = (mz >= self.mz_range[0]) & (mz <= self.mz_range[1])
        mz_filtered = mz[mask]
        int_filtered = intensities[mask]

        # Limit peaks
        if self.n_peaks and len(mz_filtered) > self.n_peaks:
            top_indices = np.argsort(int_filtered)[-self.n_peaks :]
            mz_filtered = mz_filtered[top_indices]
            int_filtered = int_filtered[top_indices]

        # Normalize
        if len(int_filtered) > 0:
            int_filtered = int_filtered / int_filtered.max()

        return {
            "spec_mzs": torch.tensor(mz_filtered, dtype=torch.float32),
            "spec_ints": torch.tensor(int_filtered, dtype=torch.float32),
        }

    def collate_fn(
        self, batch_data: Dict[str, List]
    ) -> Dict[str, torch.Tensor]:
        """Custom collate for sparse data."""
        collated = {}

        if "spec_mzs" in batch_data:
            # Concatenate all mzs/ints and create batch indices
            all_mzs = torch.cat(batch_data["spec_mzs"])
            all_ints = torch.cat(batch_data["spec_ints"])

            # Create batch indices
            batch_indices = []
            for batch_idx, mzs in enumerate(batch_data["spec_mzs"]):
                batch_indices.extend([batch_idx] * len(mzs))

            collated.update(
                {
                    "spec_mzs": all_mzs,
                    "spec_ints": all_ints,
                    "spec_batch_idx": torch.tensor(
                        batch_indices, dtype=torch.long
                    ),
                }
            )

        return collated
