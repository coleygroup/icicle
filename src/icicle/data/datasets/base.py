"""Complete dataset classes for all mass spectrometry model types."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class BaseMassSpecDataset(Dataset):
    """Base class for mass spectrometry datasets."""

    def __init__(
        self,
        labels_path: str,
        spectra_path: str,  # Might not always be spectra, but keep for compatibility
        transforms: Dict[str, Any],
        split_specs: Optional[List[str]] = None,
    ):
        """Initialize base dataset."""
        self.labels_path = Path(labels_path)
        self.spectra_path = Path(spectra_path)
        self.transforms = transforms
        self.split_specs = split_specs

        # Load labels
        self.df = pd.read_csv(labels_path, sep="\t")
        if split_specs is not None:
            self.df = self.df[self.df["mol_id"].isin(split_specs)].reset_index(
                drop=True
            )

        # Validate entries - implemented by subclasses
        self.valid_indices = self._validate_entries()

    def _validate_entries(self) -> List[int]:
        """Validate dataset entries.

        Override in subclasses.
        """
        raise NotImplementedError(
            "Subclasses must implement _validate_entries()"
        )

    def _load_spectrum(self, spec_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load spectrum data.

        Override in subclasses.
        """
        raise NotImplementedError("Subclasses must implement _load_spectrum()")

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get dataset item.

        Override in subclasses.
        """
        raise NotImplementedError("Subclasses must implement __getitem__()")

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate function for batching.

        Override in subclasses.
        """
        raise NotImplementedError("Subclasses must implement collate_fn()")
