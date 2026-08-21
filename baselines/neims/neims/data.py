"""Data loading for NEIMS."""

import logging

import h5py
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NEIMSDataset(Dataset):
    """Dataset for NEIMS training."""

    def __init__(
        self,
        metadata_path: str,
        spectra_path: str,
        split_specs: list,
        fp_radius: int = 2,
        fp_length: int = 4096,
        min_mz: float = 0.0,
        max_mz: float = 750.0,
        bin_width: float = 1.0,
    ):
        """Initialize dataset.

        Args:
            metadata_path: Path to metadata TSV file
            spectra_path: Path to spectra HDF5 file
            split_specs: List of spectrum IDs for this split
            fp_radius: Morgan fingerprint radius
            fp_length: Morgan fingerprint length
            min_mz: Minimum m/z value
            max_mz: Maximum m/z value
            bin_width: Bin width for spectrum
        """
        self.metadata_path = metadata_path
        self.spectra_path = spectra_path
        self.split_specs = split_specs
        self.fp_radius = fp_radius
        self.fp_length = fp_length
        self.min_mz = min_mz
        self.max_mz = max_mz
        self.bin_width = bin_width

        # Load metadata
        self.metadata = pd.read_csv(metadata_path, sep="\t")
        self.metadata = self.metadata[
            self.metadata["mol_id"].isin(split_specs)
        ]
        self.metadata = self.metadata.reset_index(drop=True)

        # Calculate output size
        self.output_size = int((max_mz - min_mz) / bin_width)

        # Initialize Morgan fingerprint generator (modern API, counting mode)
        self.mfpgen = rdFingerprintGenerator.GetMorganGenerator(
            radius=fp_radius, fpSize=fp_length
        )

        # HDF5 file handle (opened lazily per worker to avoid multiprocessing issues)
        self.spectra_h5 = None

        print(f"Loaded {len(self.metadata)} samples")

    def __len__(self):
        return len(self.metadata)

    def _compute_fingerprint(self, mol: Chem.Mol) -> np.ndarray:
        """Compute counting Morgan fingerprint from RDKit mol object.

        Uses MorganGenerator with GetCountFingerprintAsNumPy to produce counting
        fingerprints that record how many times each substructure occurs, matching
        the reference NEIMS implementation's use of counting circular fingerprints.
        """
        if mol is None:
            raise ValueError("Invalid molecule object")

        fp_array = self.mfpgen.GetCountFingerprintAsNumPy(mol)
        return fp_array.astype(np.float32)

    def _bin_spectrum(
        self, mz: np.ndarray, intensity: np.ndarray
    ) -> np.ndarray:
        """Bin spectrum to fixed size.

        Returns raw intensities (not normalized) to match the reference
        implementation. The ratio-based loss is scale-invariant but requires
        raw intensities for numerically stable gradients.
        """
        binned = np.zeros(self.output_size, dtype=np.float32)

        for m, i in zip(mz, intensity):
            if self.min_mz <= m < self.max_mz:
                bin_idx = int((m - self.min_mz) / self.bin_width)
                if 0 <= bin_idx < self.output_size:
                    binned[bin_idx] = max(binned[bin_idx], i)

        return binned

    def _load_spectrum(self, mol_id: str) -> tuple[np.ndarray, np.ndarray]:
        """Load spectrum from HDF5 file.

        Opens HDF5 file lazily (once per worker) to avoid multiprocessing issues.
        """
        # Open the HDF5 file if it's not already open for this worker
        if self.spectra_h5 is None:
            try:
                # Open in read-only mode without SWMR to avoid potential hangs
                self.spectra_h5 = h5py.File(self.spectra_path, "r")
                logger.info(f"Opened HDF5 file in worker: {self.spectra_path}")
            except Exception as e:
                logger.error(
                    f"Worker failed to open HDF5 file {self.spectra_path}: {e}"
                )
                return np.array([]), np.array([])

        try:
            if mol_id in self.spectra_h5:
                spec_group = self.spectra_h5[mol_id]
                # Support both naming conventions
                if "masses" in spec_group:
                    mz = spec_group["masses"][:]
                    intensity = spec_group["intensities"][:]
                else:
                    mz = spec_group["mz"][:]
                    intensity = spec_group["intensity"][:]
                return mz, intensity
            else:
                logger.warning(
                    f"Spectrum for mol_id '{mol_id}' not found in HDF5 file."
                )
                return np.array([]), np.array([])
        except Exception as e:
            logger.error(f"Error loading spectrum for mol_id '{mol_id}': {e}")
            return np.array([]), np.array([])

    def __getitem__(self, idx):
        """Get a single sample.

        Opens HDF5 file lazily (once per worker) to properly handle multiprocessing.
        """
        row = self.metadata.iloc[idx]

        # Convert mol_id to string (important for HDF5 access)
        mol_id = str(row["mol_id"])

        # Get SMILES and parse to mol (try both column names)
        smiles = row.get("standardized_smiles", row.get("smiles", None))
        if smiles is None:
            raise ValueError("No SMILES column found in metadata")

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        # Compute fingerprint from mol object
        fingerprint = self._compute_fingerprint(mol)

        # Get molecular mass from the mol object
        mass = Descriptors.ExactMolWt(mol)

        # Load spectrum using lazy HDF5 file handle
        mz, intensity = self._load_spectrum(mol_id)
        spectrum = self._bin_spectrum(mz, intensity)

        return {
            "fingerprint": torch.from_numpy(fingerprint),
            "mass": torch.tensor([mass], dtype=torch.float32),
            "spectrum": torch.from_numpy(spectrum),
            "mol_id": mol_id,
        }

    def __del__(self):
        """Close HDF5 file handle when dataset is destroyed."""
        if hasattr(self, "spectra_h5") and self.spectra_h5 is not None:
            self.spectra_h5.close()
            logger.info("HDF5 file handle closed.")


def create_dataloaders(
    metadata_path: str,
    spectra_path: str,
    splits_path: str,
    batch_size: int = 64,
    num_workers: int = 4,
    fp_radius: int = 2,
    fp_length: int = 4096,
    min_mz: float = 0.0,
    max_mz: float = 750.0,
    bin_width: float = 1.0,
):
    """Create train, val, and test dataloaders.

    Args:
        metadata_path: Path to metadata TSV file
        spectra_path: Path to spectra HDF5 file
        splits_path: Path to splits TSV file
        batch_size: Batch size
        num_workers: Number of workers
        fp_radius: Morgan fingerprint radius
        fp_length: Morgan fingerprint length
        min_mz: Minimum m/z value
        max_mz: Maximum m/z value
        bin_width: Bin width

    Returns:
        Tuple of (train_loader, val_loader, test_loader, output_size)
    """
    # Load splits
    splits_df = pd.read_csv(splits_path, sep="\t")

    train_specs = splits_df[splits_df["split"] == "train"]["mol_id"].tolist()
    val_specs = splits_df[splits_df["split"] == "val"]["mol_id"].tolist()
    test_specs = splits_df[splits_df["split"] == "test"]["mol_id"].tolist()

    print(
        f"Splits: {len(train_specs)} train, {len(val_specs)} val, {len(test_specs)} test"
    )

    # Create datasets
    train_dataset = NEIMSDataset(
        metadata_path,
        spectra_path,
        train_specs,
        fp_radius,
        fp_length,
        min_mz,
        max_mz,
        bin_width,
    )
    val_dataset = NEIMSDataset(
        metadata_path,
        spectra_path,
        val_specs,
        fp_radius,
        fp_length,
        min_mz,
        max_mz,
        bin_width,
    )
    test_dataset = NEIMSDataset(
        metadata_path,
        spectra_path,
        test_specs,
        fp_radius,
        fp_length,
        min_mz,
        max_mz,
        bin_width,
    )

    # Create dataloaders
    # Use persistent_workers to avoid worker recreation and HDF5 file reopening issues
    # Set prefetch_factor explicitly (2 is default) because it must be None when num_workers=0
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    return train_loader, val_loader, test_loader, train_dataset.output_size
