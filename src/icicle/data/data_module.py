"""Unified data module that handles all MS model types via dataset
selection."""

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from icicle.data.datasets import (
    DAGDataset,
    FragmentSpecDataset,
    MassSpecDataset,
)
from icicle.data.transforms import MolFingerprints, SpecBinner
from icicle.data.tree_processing import TreeProcessingConfig, TreeProcessor


class MassSpecDataModule(pl.LightningDataModule):
    """Data module for all mass spectrometry model types.

    Handles different model types by creating appropriate datasets:
    - "molecule_to_spectrum": SMILES -> Spectrum (baseline)
    - "molecule_to_dag": SMILES -> DAG (fragmentation)
    - "dag_to_spectrum": DAG -> Spectrum (intensity)
    """

    def __init__(
        self,
        labels_path: str,
        splits_path: str,
        dataset_type: str,
        dataset_config: Dict[str, Any],
        batch_size: int = 32,
        num_workers: int = 4,
        training_data_fraction: float = 1.0,
        prefetch_factor: int = 4,
    ):
        """Initialize unified data module.

        Args:
            labels_path: Path to labels/metadata file
            splits_path: Path to train/val/test splits
            dataset_type: Type of dataset to create
            dataset_config: Configuration specific to the dataset type
            batch_size: Batch size for dataloaders
            num_workers: Number of workers for dataloaders
            prefetch_factor: Number of batches to prefetch per worker
        """
        super().__init__()
        self.save_hyperparameters()

        self.labels_path = labels_path
        self.splits_path = splits_path
        self.dataset_type = dataset_type
        self.dataset_config = dataset_config
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.training_data_fraction = training_data_fraction
        self.prefetch_factor = prefetch_factor

        # Validate dataset type
        valid_types = [
            "molecule_to_spectrum",
            "molecule_to_dag",
            "dag_to_spectrum",
        ]
        if dataset_type not in valid_types:
            raise ValueError(
                f"dataset_type must be one of {valid_types}, got {dataset_type}"
            )

        # Initialize datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        # Setup dataset-specific components
        self._setup_dataset_components()

    def _setup_dataset_components(self):
        """Setup components specific to dataset type."""
        if self.dataset_type in ["molecule_to_dag", "dag_to_spectrum"]:
            self._setup_dag_components()

        if self.dataset_type == "molecule_to_spectrum":
            mol_transforms = self._create_mol_transforms()
            spec_transforms = self._create_spectrum_transforms()
            self.transforms = {**mol_transforms, **spec_transforms}

        self._setup_spectrum_components()

    def _setup_spectrum_components(self):
        """Setup components for spectrum prediction."""
        if self.dataset_type == "dag_to_spectrum":
            self.transforms = self._create_spectrum_transforms()

    def _setup_dag_components(self):
        """Setup components for DAG-based models."""

        tree_config = TreeProcessingConfig(
            **self.dataset_config["tree_processor_config"]
        )
        self.tree_processor = TreeProcessor(config=tree_config)
        self.tree_processor.config.min_mz = self.dataset_config["min_mz"]
        self.tree_processor.config.max_mz = self.dataset_config["max_mz"]
        self.tree_processor.config.bin_width = self.dataset_config["bin_width"]
        num_bins = int(
            (self.dataset_config["max_mz"] - self.dataset_config["min_mz"])
            / self.dataset_config["bin_width"]
        )
        self.tree_processor.bins = np.linspace(
            self.dataset_config["min_mz"],
            self.dataset_config["max_mz"],
            num_bins,
        )

    def _create_spectrum_transforms(self):
        """Create transforms for spectrum prediction."""
        config = self.dataset_config
        transforms = {}

        # Spectrum transform
        if config["spec_transform"] == "binner":
            transforms["spec_transform"] = SpecBinner(
                min_mz=config["min_mz"],
                max_mz=config["max_mz"],
                bin_width=config["bin_width"],
            )
        else:
            raise ValueError(
                f"Unknown spec transform: {config['spec_transform']}"
            )

        return transforms

    def _create_mol_transforms(self):
        """Create transforms for molecule prediction."""
        config = self.dataset_config
        transforms = {}

        # Molecule transform
        if config["mol_transform"] == "morgan":
            transforms["mol_transform"] = MolFingerprints(
                fp_types=config["fp_types"],
                morgan_bits=config["morgan_bits"],
                morgan_radius=config["morgan_radius"],
            )
        else:
            raise ValueError(
                f"Unknown mol transform: {config['mol_transform']}"
            )

        return transforms

    def prepare_data(self):
        """Build disk cache on rank 0 only.

        Lightning guarantees this runs on a single process before setup().
        Other ranks wait at an implicit barrier until this completes.
        """
        if self.dataset_type not in ("molecule_to_dag", "dag_to_spectrum"):
            return

        splits_df = pd.read_csv(self.splits_path, sep="\t")
        all_specs = splits_df["mol_id"].tolist()

        ds = self._create_dataset(all_specs)
        if hasattr(ds, "preprocess_cache"):
            ds.preprocess_cache()
        del ds

    def setup(self, stage: Optional[str] = None):
        """Setup datasets on each rank.

        Cache is already populated by prepare_data().
        """
        # Load splits
        splits_df = pd.read_csv(self.splits_path, sep="\t")

        # Get spec lists for each split
        train_specs = splits_df[splits_df["split"] == "train"][
            "mol_id"
        ].tolist()

        if self.training_data_fraction < 1.0:
            train_specs = np.random.choice(
                train_specs,
                size=int(len(train_specs) * self.training_data_fraction),
                replace=False,
            )

        val_specs = splits_df[splits_df["split"] == "val"]["mol_id"].tolist()
        test_specs = splits_df[splits_df["split"] == "test"]["mol_id"].tolist()

        print(
            f"Splits: {len(train_specs)} train, {len(val_specs)} val, {len(test_specs)} test"
        )

        # Create datasets based on type
        if stage in ("fit", None):
            self.train_dataset = self._create_dataset(train_specs)
            self.val_dataset = self._create_dataset(val_specs)

        if stage in ("test", None):
            self.test_dataset = self._create_dataset(test_specs)

    def _create_dataset(self, split_specs: List[str]):
        """Create appropriate dataset based on dataset_type."""
        if self.dataset_type == "molecule_to_dag":
            return DAGDataset(
                labels_path=self.labels_path,
                magma_trees_path=Path(self.dataset_config["magma_trees_path"]),
                tree_processor=self.tree_processor,
                split_specs=split_specs,
            )
        elif self.dataset_type == "dag_to_spectrum":
            return FragmentSpecDataset(
                labels_path=self.labels_path,
                magma_trees_path=Path(self.dataset_config["magma_trees_path"]),
                spectra_path=Path(self.dataset_config["spectra_path"]),
                tree_processor=self.tree_processor,
                split_specs=split_specs,
            )
        elif self.dataset_type == "molecule_to_spectrum":
            return MassSpecDataset(
                labels_path=self.labels_path,
                spectra_path=Path(self.dataset_config["spectra_path"]),
                transforms=self.transforms,
                split_specs=split_specs,
            )
        else:
            raise ValueError(f"Unknown dataset_type: {self.dataset_type}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.train_dataset.collate_fn,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=self.prefetch_factor
            if self.num_workers > 0
            else None,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.val_dataset.collate_fn,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=self.prefetch_factor
            if self.num_workers > 0
            else None,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.test_dataset.collate_fn,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=self.prefetch_factor
            if self.num_workers > 0
            else None,
        )

    @property
    def input_size(self) -> int:
        """Get input size for molecule->spectrum models."""
        if self.dataset_type == "molecule_to_spectrum":
            return self.transforms["mol_transform"].output_size
        else:
            raise AttributeError(
                f"input_size not available for dataset_type: {self.dataset_type}"
            )

    @property
    def node_input_size(self) -> int:
        """Get node input size for graph-based models."""
        if self.dataset_type in ["molecule_to_dag", "dag_to_spectrum"]:
            return self.tree_processor.get_node_feats()
        else:
            raise AttributeError(
                f"node_input_size not available for dataset_type: {self.dataset_type}"
            )

    @property
    def output_size(self) -> int:
        """Get output size for models."""
        if self.dataset_type == "molecule_to_spectrum":
            # Calculate number of bins for spectrum
            config = self.dataset_config
            num_bins = int(
                (config["max_mz"] - config["min_mz"]) / config["bin_width"]
            )
            return num_bins
        elif self.dataset_type == "molecule_to_dag":
            return 1  # Binary decision per atom
        elif self.dataset_type == "dag_to_spectrum":
            # Calculate number of bins for spectrum
            config = self.dataset_config
            num_bins = int(
                (config["max_mz"] - config["min_mz"]) / config["bin_width"]
            )
            return num_bins
        else:
            raise AttributeError(
                f"output_size not available for dataset_type: {self.dataset_type}"
            )

    def get_model_args(self) -> Dict[str, Any]:
        """Get arguments needed for model initialization."""
        if self.dataset_type == "molecule_to_spectrum":
            return {
                "input_size": self.input_size,
                "output_size": self.output_size,
                "min_mz": self.dataset_config["min_mz"],
                "max_mz": self.dataset_config["max_mz"],
                "bin_width": self.dataset_config["bin_width"],
            }
        elif self.dataset_type == "molecule_to_dag":
            return {
                "node_input_size": self.node_input_size,
                "tree_processor": self.tree_processor,
            }
        elif self.dataset_type == "dag_to_spectrum":
            return {
                "node_input_size": self.node_input_size,
                "output_size": self.output_size,
                "tree_processor": self.tree_processor,
                "min_mz": self.dataset_config["min_mz"],
                "max_mz": self.dataset_config["max_mz"],
                "bin_width": self.dataset_config["bin_width"],
            }
        else:
            raise ValueError(f"Unknown dataset_type: {self.dataset_type}")
