"""Dataset for Molecule -> Spectrum prediction (ECFP-based models)."""

import torch
import pandas as pd
import numpy as np
import h5py
import json
import logging
from torch.utils.data import Dataset
from rdkit import Chem
from rdkit.Chem import Descriptors
from typing import Optional, List, Dict, Tuple, Any

from icicle.data.datasets.base import BaseMassSpecDataset
from icicle.data.transforms import SpecTransform, MolTransform
from icicle.utils import smiles_from_inchi

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MassSpecDataset(BaseMassSpecDataset):
    """Dataset for Molecule -> Spectrum prediction (ECFP-based models).

    This class adapts the dataset to go from a molecule (SMILES) to its
    fingerprint (ECFP) and then to its binned mass spectrum.
    """

    def __init__(
        self,
        labels_path: str,
        spectra_path: str,  # Path to HDF5 file with experimental spectra
        transforms: Dict[str, Any],
        split_specs: Optional[List[str]] = None,
    ):
        """Initialize the molecule-to-spectrum dataset."""
        self.spec_transform: SpecTransform = transforms["spec_transform"]
        self.mol_transform: MolTransform = transforms["mol_transform"]

        self.spectra_h5 = None

        super().__init__(labels_path, spectra_path, transforms, split_specs)

    def _validate_entries(self) -> List[int]:
        """Validate entries to ensure they have a valid SMILES string and an
        available spectrum in the HDF5 file.

        This runs in the main process.
        """
        valid_indices = []
        try:
            with h5py.File(self.spectra_path, "r") as spectra_file:
                available_spectra_keys = set(spectra_file.keys())

                for idx in range(len(self.df)):
                    row = self.df.iloc[idx]
                    mol_id = str(row["mol_id"])
                    smiles = row["standardized_smiles"]

                    if (
                        not smiles
                        or not isinstance(smiles, str)
                        or Chem.MolFromSmiles(smiles) is None
                    ):
                        logger.debug(
                            f"Skipping entry {mol_id}: Invalid or missing SMILES."
                        )
                        continue

                    if mol_id in available_spectra_keys:
                        valid_indices.append(idx)
                    else:
                        logger.debug(
                            f"Skipping entry {mol_id}: Spectrum not found in HDF5 file."
                        )

        except Exception as e:
            logger.error(
                f"Failed to validate entries for {self.spectra_path}: {e}"
            )
            raise  # Fail fast if the HDF5 file is inaccessible or corrupt

        logger.info(
            f"Found {len(valid_indices)} valid entries with SMILES and spectra."
        )
        return valid_indices

    def _load_spectrum(self, mol_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load an experimental spectrum (m/z and intensity arrays) for a given
        mol_id.

        This is called by workers and manages its own HDF5 file handle.
        """
        # Open the HDF5 file if it's not already open for this worker
        if self.spectra_h5 is None:
            try:
                self.spectra_h5 = h5py.File(self.spectra_path, "r", swmr=True)
            except Exception as e:
                logger.error(
                    f"Worker failed to open HDF5 file {self.spectra_path}: {e}"
                )
                return np.array([]), np.array([])  # Return empty on failure

        try:
            # Assumes the mol_id is the key in the HDF5 file
            if mol_id in self.spectra_h5:
                group = self.spectra_h5[mol_id]
                mz = group["masses"][:]
                intensities = group["intensities"][:]
                return mz, intensities
            else:
                logger.warning(
                    f"Spectrum for mol_id '{mol_id}' not found inside HDF5 file."
                )
                return np.array([]), np.array([])

        except Exception as e:
            logger.error(f"Error loading spectrum for mol_id '{mol_id}': {e}")
            return np.array([]), np.array([])

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Retrieves a single processed data point: a molecule's fingerprint,
        its binned spectrum, and its normalized mass.
        """
        df_idx = self.valid_indices[idx]
        row = self.df.iloc[df_idx]

        mol_id = str(row["mol_id"])
        smiles = row["standardized_smiles"]

        # 1. Transform SMILES to fingerprint tensor using the mol_transform
        mol_data = self.mol_transform(
            smiles
        )  # Returns {"fingerprints": tensor}

        # 2. Load raw spectrum and bin it using the spec_transform
        mz_raw, intensity_raw = self._load_spectrum(mol_id)
        spec_data = self.spec_transform(
            mz_raw, intensity_raw
        )  # Returns {"spectrum": tensor}
        spectrum_tensor = spec_data["spectrum"]

        # 3. Calculate and normalize molecular mass as metadata
        mass_tensor = torch.tensor([0.0], dtype=torch.float32)
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            mass = Descriptors.ExactMolWt(mol)
            mass_tensor = torch.tensor([mass], dtype=torch.float32)

        # 4. Assemble the final dictionary
        item = {
            "mol_id": mol_id,
            "spectrum": spectrum_tensor,
            "mass": mass_tensor,
            **mol_data,
        }

        return item

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collates a list of data points into a single batch dictionary."""
        mol_ids = [item["mol_id"] for item in batch]
        fingerprints = torch.stack([item["fingerprints"] for item in batch])
        spectra = torch.stack([item["spectrum"] for item in batch])
        masses = torch.stack([item["mass"] for item in batch])

        return {
            "mol_id": mol_ids,
            "fingerprints": fingerprints,
            "spectrum": spectra,
            "mass": masses,
        }

    def __del__(self):
        """Ensures the HDF5 file handle is closed when the dataset object is
        destroyed."""
        if hasattr(self, "spectra_h5") and self.spectra_h5 is not None:
            self.spectra_h5.close()
            logger.info("HDF5 file handle closed.")
