"""Base transforms."""

from abc import ABC, abstractmethod
from typing import Dict, List
import torch
import numpy as np


class Transform(ABC):
    """Base transform class with collate support."""

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

    def collate_fn(
        self, batch_data: Dict[str, List]
    ) -> Dict[str, torch.Tensor]:
        """Default collate - stack tensors."""
        collated = {}
        for key, values in batch_data.items():
            if isinstance(values[0], torch.Tensor):
                collated[key] = torch.stack(values)
        return collated


class SpecTransform(Transform):
    """Spectrum transform base class."""

    @abstractmethod
    def __call__(
        self, mz: np.ndarray, intensities: np.ndarray, **metadata
    ) -> Dict[str, torch.Tensor]:
        pass


class MolTransform(Transform):
    """Molecule transform base class."""

    @abstractmethod
    def __call__(self, smiles: str) -> Dict[str, torch.Tensor]:
        pass


class MetaTransform(Transform):
    """Metadata transform base class."""

    @abstractmethod
    def __call__(self, **metadata) -> Dict[str, torch.Tensor]:
        pass
