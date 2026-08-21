"""Data loading for NEIMS-GNN.

Provides a PyTorch Geometric dataset that converts molecules to graphs
with atom/bond features matching Zhu et al. 2020 (arXiv:2010.04661).
"""

import logging

import h5py
import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

RDLogger.DisableLog("rdApp.*")
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from neims.gnn_model import NUM_BOND_FEATURES, atom_features, bond_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def mol_to_pyg_graph(mol, use_edge_features=False):
    """Convert an RDKit Mol to a PyTorch Geometric Data object.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        RDKit molecule object (must not be None).
    use_edge_features : bool
        Whether to include bond features as edge attributes.

    Returns
    -------
    torch_geometric.data.Data
        Graph with node features (x), edge indices (edge_index),
        and optionally edge attributes (edge_attr).
    """
    # Node features
    node_feats = []
    for atom in mol.GetAtoms():
        node_feats.append(atom_features(atom))
    x = torch.tensor(node_feats, dtype=torch.float32)

    # Edge index (COO format) - self-loops added by GATConv/GCNConv
    edge_list = []
    edge_attrs = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_list.append([i, j])
        edge_list.append([j, i])
        if use_edge_features:
            bf = bond_features(bond)
            edge_attrs.append(bf)
            edge_attrs.append(bf)  # same features for both directions

    if len(edge_list) > 0:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index)

    if use_edge_features:
        if len(edge_attrs) > 0:
            data.edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)
        else:
            data.edge_attr = torch.zeros(
                (0, NUM_BOND_FEATURES), dtype=torch.float32
            )

    return data


class NEIMSGNNDataset(Dataset):
    """Dataset for NEIMS-GNN training with molecular graphs.

    Each sample is a PyG Data object containing:
        - x: atom features [num_atoms, 15]
        - edge_index: bond connectivity [2, num_edges]
        - edge_attr: bond features [num_edges, 6] (optional)
        - y: binned spectrum [1, output_size]
        - mass: molecular mass [1]
    """

    def __init__(
        self,
        metadata_path: str,
        spectra_path: str,
        split_specs: list,
        min_mz: float = 0.0,
        max_mz: float = 750.0,
        bin_width: float = 1.0,
        use_edge_features: bool = False,
    ):
        self.metadata_path = metadata_path
        self.spectra_path = spectra_path
        self.split_specs = split_specs
        self.min_mz = min_mz
        self.max_mz = max_mz
        self.bin_width = bin_width
        self.use_edge_features = use_edge_features

        # Load metadata
        self.metadata = pd.read_csv(metadata_path, sep="\t")
        self.metadata = self.metadata[
            self.metadata["mol_id"].isin(split_specs)
        ].reset_index(drop=True)

        # Calculate output size
        self.output_size = int((max_mz - min_mz) / bin_width)

        # HDF5 file handle (opened lazily per worker)
        self.spectra_h5 = None

        print(f"Loaded {len(self.metadata)} samples (GNN mode)")

    def __len__(self):
        return len(self.metadata)

    def _bin_spectrum(
        self, mz: np.ndarray, intensity: np.ndarray
    ) -> np.ndarray:
        """Bin spectrum and max-normalize to [0, 1].

        No log/sqrt transform — matches eval pipeline in eval_from_predictions.py
        which also does bin + max-normalize on ground truth.
        """
        binned = np.zeros(self.output_size, dtype=np.float32)
        for m, i in zip(mz, intensity):
            if self.min_mz <= m < self.max_mz:
                bin_idx = int((m - self.min_mz) / self.bin_width)
                if 0 <= bin_idx < self.output_size:
                    binned[bin_idx] = max(binned[bin_idx], i)
        max_val = binned.max()
        if max_val > 0:
            binned /= max_val
        return binned

    def _load_spectrum(self, mol_id: str) -> tuple:
        """Load spectrum from HDF5 file (lazy open per worker)."""
        if self.spectra_h5 is None:
            try:
                self.spectra_h5 = h5py.File(self.spectra_path, "r")
            except Exception as e:
                logger.error(
                    f"Worker failed to open HDF5 file {self.spectra_path}: {e}"
                )
                return np.array([]), np.array([])

        try:
            if mol_id in self.spectra_h5:
                spec_group = self.spectra_h5[mol_id]
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
        row = self.metadata.iloc[idx]
        mol_id = str(row["mol_id"])

        smiles = row.get("standardized_smiles", row.get("smiles", None))
        if smiles is None:
            raise ValueError("No SMILES column found in metadata")

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        # Build molecular graph
        graph = mol_to_pyg_graph(mol, use_edge_features=self.use_edge_features)

        # Molecular mass
        mass = Descriptors.ExactMolWt(mol)
        graph.mass = torch.tensor([mass], dtype=torch.float32)

        # Load and bin spectrum - unsqueeze to [1, output_size] for proper
        # batching (PyG concatenates along dim 0)
        mz, intensity = self._load_spectrum(mol_id)
        spectrum = self._bin_spectrum(mz, intensity)
        graph.y = torch.from_numpy(spectrum).unsqueeze(0)

        return graph

    def __del__(self):
        if hasattr(self, "spectra_h5") and self.spectra_h5 is not None:
            self.spectra_h5.close()
            logger.info("HDF5 file handle closed.")


def create_gnn_dataloaders(
    metadata_path: str,
    spectra_path: str,
    splits_path: str,
    batch_size: int = 64,
    num_workers: int = 4,
    min_mz: float = 0.0,
    max_mz: float = 750.0,
    bin_width: float = 1.0,
    use_edge_features: bool = False,
):
    """Create train, val, and test dataloaders for NEIMS-GNN.

    Uses PyTorch Geometric's DataLoader which handles batching of
    variable-size graphs via the Batch.from_data_list mechanism.

    Returns
    -------
    tuple
        (train_loader, val_loader, test_loader, output_size)
    """
    splits_df = pd.read_csv(splits_path, sep="\t")

    train_specs = splits_df[splits_df["split"] == "train"]["mol_id"].tolist()
    val_specs = splits_df[splits_df["split"] == "val"]["mol_id"].tolist()
    test_specs = splits_df[splits_df["split"] == "test"]["mol_id"].tolist()

    print(
        f"Splits: {len(train_specs)} train, {len(val_specs)} val, {len(test_specs)} test"
    )

    common_kwargs = dict(
        metadata_path=metadata_path,
        spectra_path=spectra_path,
        min_mz=min_mz,
        max_mz=max_mz,
        bin_width=bin_width,
        use_edge_features=use_edge_features,
    )

    train_dataset = NEIMSGNNDataset(split_specs=train_specs, **common_kwargs)
    val_dataset = NEIMSGNNDataset(split_specs=val_specs, **common_kwargs)
    test_dataset = NEIMSGNNDataset(split_specs=test_specs, **common_kwargs)

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader, train_dataset.output_size


class NEIMSGNNInferenceDataset(Dataset):
    """Lightweight dataset for NEIMS-GNN inference (no spectra loading).

    Each sample is a PyG Data object with atom/bond features, mass, and mol_id.
    """

    def __init__(
        self,
        mol_ids: list,
        smiles_list: list,
        output_size: int = 750,
        use_edge_features: bool = False,
    ):
        self.mol_ids = mol_ids
        self.smiles_list = smiles_list
        self.output_size = output_size
        self.use_edge_features = use_edge_features

    def __len__(self):
        return len(self.mol_ids)

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        mol_id = self.mol_ids[idx]

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            # Fallback: single-atom carbon graph so batching doesn't break
            mol = Chem.MolFromSmiles("C")

        graph = mol_to_pyg_graph(mol, use_edge_features=self.use_edge_features)
        graph.mass = torch.tensor([Descriptors.ExactMolWt(mol)], dtype=torch.float32)
        graph.mol_id = mol_id
        return graph


def create_inference_dataloader(
    mol_ids: list,
    smiles_list: list,
    output_size: int = 750,
    use_edge_features: bool = False,
    batch_size: int = 256,
    num_workers: int = 4,
):
    """Create a DataLoader for NEIMS-GNN inference over a list of SMILES.

    Parameters
    ----------
    mol_ids : list of str
        Identifiers for each molecule (used to map predictions back).
    smiles_list : list of str
        SMILES strings to run inference on.
    output_size : int
        Number of m/z bins (must match the trained model).
    use_edge_features : bool
        Whether to include bond features (must match the trained model config).
    batch_size : int
        Inference batch size.
    num_workers : int
        DataLoader worker processes.

    Returns
    -------
    torch_geometric.loader.DataLoader
    """
    ds = NEIMSGNNInferenceDataset(
        mol_ids=mol_ids,
        smiles_list=smiles_list,
        output_size=output_size,
        use_edge_features=use_edge_features,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )
