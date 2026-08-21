"""DAG dataset for fragmentation prediction."""

import collections
import hashlib
import json
import logging
import multiprocessing as mp
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import dgl
import h5py
import numpy as np
import torch
from tqdm import tqdm

from icicle.data.datasets.base import BaseMassSpecDataset
from icicle.data.transforms.spectrum import SpecBinner
from icicle.data.tree_processing import TreeProcessor
from icicle.utils import HDF5Dataset, smiles_from_inchi
from icicle.utils.chem import ELEMENT_DIM

_cache_worker_h5 = None
_cache_worker_processor = None


def _config_hash(tree_processor: TreeProcessor, mode: str) -> str:
    """Create a short hash from tree processor config and processing mode."""
    config_dict = asdict(tree_processor.config)
    config_str = f"{mode}:{sorted(config_dict.items())}"
    return hashlib.md5(config_str.encode()).hexdigest()[:8]


def _cache_worker_init(magma_trees_path, tree_processor):
    """Initialize per-worker state for parallel cache building."""
    global _cache_worker_h5, _cache_worker_processor
    _cache_worker_h5 = HDF5Dataset(magma_trees_path)
    _cache_worker_processor = tree_processor


def _cache_worker_fn(args):
    """Process a single molecule for caching (runs in worker process)."""
    global _cache_worker_h5, _cache_worker_processor
    mol_id, cache_path_str, mode = args
    try:
        tree_data = json.loads(_cache_worker_h5.read_str(mol_id))
        tree_data["root_canonical_smiles"] = smiles_from_inchi(
            tree_data["root_inchi"]
        )
        if mode == "gen":
            processed_tree = _cache_worker_processor.process_tree_gen(
                tree_data
            )
        else:
            processed_tree = _cache_worker_processor.process_tree_inten(
                tree_data
            )
        cache_data = {
            "smiles": tree_data["root_canonical_smiles"],
            **processed_tree["dgl_tree"],
        }
        torch.save(cache_data, cache_path_str)
        return mol_id, True
    except Exception as e:
        logging.error(f"Error caching {mol_id}: {e}")
        return mol_id, False


class DAGDataset(BaseMassSpecDataset):
    """Simple dataset for Molecule -> DAG prediction (fragmentation)."""

    def __init__(
        self,
        labels_path: str,
        magma_trees_path: Path,
        tree_processor: TreeProcessor,
        split_specs: Optional[List[str]] = None,
    ):
        """Initialize dataset."""
        self.magma_trees_path = magma_trees_path
        self.tree_processor = tree_processor

        # HDF5 file handle - will be opened per worker
        self.trees_h5 = None

        # Setup cache directory next to the HDF5 file
        chash = _config_hash(tree_processor, "gen")
        self.cache_dir = (
            Path(magma_trees_path).parent / f".tree_cache_gen_{chash}"
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Per-worker LRU cache (bounded to prevent OOM, persists with persistent_workers)
        _MEM_CACHE_MAX = 5000
        self._mem_cache = collections.OrderedDict()
        self._mem_cache_max = _MEM_CACHE_MAX

        # Initialize base class
        super().__init__(
            labels_path=labels_path,
            spectra_path=str(magma_trees_path),
            transforms={"tree_processor": tree_processor},
            split_specs=split_specs,
        )

    def _cache_put(self, idx: int, result: Dict[str, Any]):
        """Insert into bounded LRU in-memory cache, evicting oldest if full."""
        if idx in self._mem_cache:
            self._mem_cache.move_to_end(idx)
            return
        if len(self._mem_cache) >= self._mem_cache_max:
            self._mem_cache.popitem(last=False)
        self._mem_cache[idx] = result

    def _get_cache_path(self, mol_id: str) -> Path:
        """Get cache file path for a molecule."""
        return self.cache_dir / f"{mol_id}.pt"

    def preprocess_cache(self):
        """Pre-compute and cache all tree processing results in parallel."""
        uncached = []
        for i in range(len(self.valid_indices)):
            df_idx = self.valid_indices[i]
            mol_id = str(self.df.iloc[df_idx]["mol_id"])
            if not self._get_cache_path(mol_id).exists():
                uncached.append((i, mol_id))

        if not uncached:
            logging.info(
                f"DAGDataset cache fully populated "
                f"({len(self.valid_indices)} entries) at {self.cache_dir}"
            )
            return

        logging.info(
            f"Preprocessing {len(uncached)} entries "
            f"({len(self.valid_indices) - len(uncached)} already cached)"
        )

        num_workers = min(os.cpu_count() or 4, len(uncached))
        args_list = [
            (mol_id, str(self._get_cache_path(mol_id)), "gen")
            for _, mol_id in uncached
        ]
        chunksize = max(1, len(args_list) // (num_workers * 4))

        ctx = mp.get_context("spawn")
        with ctx.Pool(
            processes=num_workers,
            initializer=_cache_worker_init,
            initargs=(self.magma_trees_path, self.tree_processor),
        ) as pool:
            results = list(
                tqdm(
                    pool.imap_unordered(
                        _cache_worker_fn, args_list, chunksize=chunksize
                    ),
                    total=len(args_list),
                    desc="Building DAG cache",
                )
            )

        succeeded = sum(1 for _, ok in results if ok)
        logging.info(
            f"DAG cache complete: {succeeded}/{len(uncached)} succeeded"
        )

    def _validate_entries(self) -> List[int]:
        """Find entries that have corresponding MAGMA tree files."""
        # Open the file for this specific check, then close it.
        # This part runs in the main process.
        with HDF5Dataset(self.magma_trees_path) as trees_h5:
            tree_files = set(trees_h5.get_all_names())
            valid_indices = []

            for idx in tqdm(range(len(self.df))):
                mol_id = str(self.df.iloc[idx]["mol_id"])
                tree_file = f"{mol_id}"

                if tree_file in tree_files:
                    valid_indices.append(idx)

        return valid_indices

    def _load_magma_tree(self, mol_id: str) -> Dict[str, Any]:
        """Load MAGMA tree data."""
        # Check if the file is open for this worker. If not, open it.
        # This part runs in the worker processes.
        if self.trees_h5 is None:
            self.trees_h5 = HDF5Dataset(self.magma_trees_path)

        try:
            tree_file = f"{mol_id}"
            tree_data = json.loads(self.trees_h5.read_str(tree_file))
            return tree_data
        except Exception as e:
            logging.error(f"Error loading tree for {mol_id}: {e}")
            raise

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get dataset item."""
        # Check in-memory cache first (fastest path)
        if idx in self._mem_cache:
            self._mem_cache.move_to_end(idx)
            return self._mem_cache[idx]

        df_idx = self.valid_indices[idx]
        row = self.df.iloc[df_idx]
        mol_id = str(row["mol_id"])

        # Try loading from disk cache
        cache_path = self._get_cache_path(mol_id)
        if cache_path.exists():
            try:
                cached = torch.load(cache_path, weights_only=False)
                result = {"name": mol_id, **cached}
                self._cache_put(idx, result)
                return result
            except Exception:
                pass  # Fall through to recompute

        # Fallback: compute from scratch and cache the result
        try:
            tree_data = self._load_magma_tree(mol_id)
            tree_data["root_canonical_smiles"] = smiles_from_inchi(
                tree_data["root_inchi"]
            )
            processed_tree = self.tree_processor.process_tree_gen(tree_data)

            result = {
                "name": mol_id,
                "smiles": tree_data["root_canonical_smiles"],
                **processed_tree["dgl_tree"],
            }

            # Lazy-cache to disk for next run
            try:
                cache_data = {
                    "smiles": tree_data["root_canonical_smiles"],
                    **processed_tree["dgl_tree"],
                }
                torch.save(cache_data, cache_path)
            except Exception:
                pass

            self._cache_put(idx, result)
            return result

        except Exception as e:
            logging.error(f"Error processing {mol_id}: {e}")
            return None

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate function for batching."""
        batch = [item for item in batch if item is not None]
        if len(batch) == 0:
            return None
        names = [item["name"] for item in batch]
        smiles = [item["smiles"] for item in batch]

        # Fragment graphs (input)
        frag_graphs = [item["dgl_frags"] for item in batch]
        frag_graphs_flat = [
            graph for graphs in frag_graphs for graph in graphs
        ]

        # Number of fragments per item
        num_frags = torch.tensor(
            [len(graphs) for graphs in frag_graphs], dtype=torch.long
        )

        # Fragment atom counts
        frag_atoms = torch.tensor(
            [graph.num_nodes() for graphs in frag_graphs for graph in graphs],
            dtype=torch.long,
        )

        # Target atoms (binary decisions - what should fragment)
        targets = [target for item in batch for target in item["targs"]]
        targets_padded = torch.nn.utils.rnn.pad_sequence(
            targets, batch_first=True
        )

        # Root representations
        root_reprs = self._collate_root_representations(batch)

        # Fragment graphs batch
        frag_batch = dgl.batch(frag_graphs_flat) if frag_graphs_flat else None

        # Index mapping for root to fragments
        root_inds = torch.arange(
            len(frag_graphs), dtype=torch.long
        ).repeat_interleave(num_frags)

        # Broken bonds
        max_num_frags = max(len(item["max_broken"]) for item in batch)
        padded_max_broken = torch.zeros(
            len(batch), max_num_frags, dtype=torch.long
        )
        for i, item in enumerate(batch):
            n = len(item["max_broken"])
            padded_max_broken[i, :n] = torch.tensor(
                item["max_broken"], dtype=torch.long
            )
        max_broken = padded_max_broken

        # Formula vectors
        form_vecs = torch.tensor(
            np.array([form for item in batch for form in item["form_vecs"]]),
            dtype=torch.long,
        )
        root_vecs = torch.tensor(
            np.array([item["root_form_vec"] for item in batch]),
            dtype=torch.long,
        )

        return {
            "names": names,
            "smiles": smiles,
            "root_reprs": root_reprs,
            "frag_graphs": frag_batch,
            "targ_atoms": targets_padded,
            "frag_atoms": frag_atoms,
            "inds": root_inds,
            "broken_bonds": max_broken,
            "root_form_vecs": root_vecs,
            "frag_form_vecs": form_vecs,
        }

    def _collate_root_representations(self, batch: List[Dict[str, Any]]):
        """Collate root representations."""
        root_reprs = [item["root_repr"] for item in batch]

        if isinstance(root_reprs[0], dgl.DGLGraph):
            return dgl.batch(root_reprs)
        elif isinstance(root_reprs[0], np.ndarray):
            return torch.tensor(np.vstack(root_reprs), dtype=torch.float)
        else:
            raise NotImplementedError(
                f"Unsupported root representation: {type(root_reprs[0])}"
            )

    def __del__(self):
        """Cleanup HDF5 files."""
        if hasattr(self, "trees_h5") and self.trees_h5 is not None:
            self.trees_h5.close()


class FragmentSpecDataset(BaseMassSpecDataset):
    """Dataset for DAG -> Spectrum prediction (fragments to intensities)."""

    def __init__(
        self,
        labels_path: str,
        magma_trees_path: Path,  # HDF5 with MAGMA JSON trees (input)
        spectra_path: Path,  # HDF5 with experimental spectra (target)
        tree_processor: TreeProcessor,
        split_specs: Optional[List[str]] = None,
    ):
        """Initialize DAG->spectrum dataset."""
        self.tree_processor = tree_processor
        self.magma_trees_path = magma_trees_path
        self.spectra_path = spectra_path

        # Infer mass range parameters from tree processor config
        self.min_mz = self.tree_processor.config.min_mz
        self.max_mz = self.tree_processor.config.max_mz
        self.num_bins = int(
            (self.max_mz - self.min_mz) / self.tree_processor.config.bin_width
        )
        self.bin_width = self.tree_processor.config.bin_width

        # File handles will be opened by workers
        self.trees_h5 = None
        self.spectra_h5 = None

        self.binner = SpecBinner(
            min_mz=self.min_mz, max_mz=self.max_mz, bin_width=self.bin_width
        )

        # Setup cache directory next to the HDF5 file
        chash = _config_hash(tree_processor, "inten")
        self.cache_dir = (
            Path(magma_trees_path).parent / f".tree_cache_inten_{chash}"
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Per-worker LRU cache (bounded to prevent OOM, persists with persistent_workers)
        _MEM_CACHE_MAX = 5000
        self._mem_cache = collections.OrderedDict()
        self._mem_cache_max = _MEM_CACHE_MAX

        super().__init__(
            labels_path=labels_path,
            spectra_path=str(spectra_path),
            transforms={"tree_processor": tree_processor},
            split_specs=split_specs,
        )

    def _cache_put(self, idx: int, result: Dict[str, Any]):
        """Insert into bounded LRU in-memory cache, evicting oldest if full."""
        if idx in self._mem_cache:
            self._mem_cache.move_to_end(idx)
            return
        if len(self._mem_cache) >= self._mem_cache_max:
            self._mem_cache.popitem(last=False)
        self._mem_cache[idx] = result

    def _get_cache_path(self, mol_id: str) -> Path:
        """Get cache file path for a molecule."""
        return self.cache_dir / f"{mol_id}.pt"

    def preprocess_cache(self):
        """Pre-compute and cache all tree processing results in parallel."""
        uncached = []
        for i in range(len(self.valid_indices)):
            df_idx = self.valid_indices[i]
            mol_id = str(self.df.iloc[df_idx]["mol_id"])
            if not self._get_cache_path(mol_id).exists():
                uncached.append((i, mol_id))

        if not uncached:
            logging.info(
                f"FragmentSpecDataset cache fully populated "
                f"({len(self.valid_indices)} entries) at {self.cache_dir}"
            )
            # self._filter_uncached_entries()
            return

        logging.info(
            f"Preprocessing {len(uncached)} entries "
            f"({len(self.valid_indices) - len(uncached)} already cached)"
        )

        num_workers = min(os.cpu_count() or 4, len(uncached))
        args_list = [
            (mol_id, str(self._get_cache_path(mol_id)), "inten")
            for _, mol_id in uncached
        ]
        chunksize = max(1, len(args_list) // (num_workers * 4))

        ctx = mp.get_context("spawn")
        with ctx.Pool(
            processes=num_workers,
            initializer=_cache_worker_init,
            initargs=(self.magma_trees_path, self.tree_processor),
        ) as pool:
            results = list(
                tqdm(
                    pool.imap_unordered(
                        _cache_worker_fn, args_list, chunksize=chunksize
                    ),
                    total=len(args_list),
                    desc="Building intensity cache",
                )
            )

        succeeded = sum(1 for _, ok in results if ok)
        logging.info(
            f"Intensity cache complete: {succeeded}/{len(uncached)} succeeded"
        )

    def _load_magma_tree(self, mol_id: str) -> Dict[str, Any]:
        """Load MAGMA tree data (input)."""
        if self.trees_h5 is None:
            self.trees_h5 = HDF5Dataset(self.magma_trees_path)

        try:
            tree_data = json.loads(self.trees_h5.read_str(mol_id))
            return tree_data
        except Exception as e:
            logging.error(f"Error loading tree for {mol_id}: {e}")
            raise

    def _validate_entries(self) -> List[int]:
        """Find entries that have both MAGMA trees AND experimental spectra."""
        # Open both files to check for matching mol_ids (not InChI keys!)
        with HDF5Dataset(self.magma_trees_path) as trees_h5:
            tree_mol_ids = set(trees_h5.get_all_names())

        with h5py.File(self.spectra_path, "r") as spectra_h5:
            spectra_mol_ids = set(spectra_h5.keys())

        # Find intersection of mol_ids that exist in both datasets
        common_mol_ids = tree_mol_ids.intersection(spectra_mol_ids)
        logging.info(
            f"Found {len(common_mol_ids)} common mol_ids between trees and spectra"
        )

        # Find valid indices where the mol_id exists in both datasets
        valid_indices = []
        for idx in range(len(self.df)):
            mol_id = str(
                self.df.iloc[idx]["mol_id"]
            )  # Use mol_id instead of inchi_key
            if mol_id in common_mol_ids:
                valid_indices.append(idx)

        logging.info(
            f"Found {len(valid_indices)} valid entries with both trees and spectra"
        )
        return valid_indices

    def _load_spectrum(self, mol_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load experimental spectrum using mol_id (not inchi_key)."""
        if self.spectra_h5 is None:
            self.spectra_h5 = h5py.File(self.spectra_path, "r")

        try:
            if mol_id in self.spectra_h5:
                mol_group = self.spectra_h5[mol_id]

                # Handle both scalar and array data
                masses_dataset = mol_group["masses"]
                intensities_dataset = mol_group["intensities"]

                if masses_dataset.shape == ():  # Scalar
                    mz = np.array([masses_dataset[()]])
                    intensities = np.array([intensities_dataset[()]])
                else:  # Array
                    mz = masses_dataset[:]
                    intensities = intensities_dataset[:]

                return mz, intensities
            else:
                return np.array([]), np.array([])

        except Exception as e:
            logging.error(f"Error loading spectrum for {mol_id}: {e}")
            return np.array([]), np.array([])

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get dataset item."""
        # Check in-memory cache first (fastest path)
        if idx in self._mem_cache:
            self._mem_cache.move_to_end(idx)
            return self._mem_cache[idx]

        df_idx = self.valid_indices[idx]
        row = self.df.iloc[df_idx]
        mol_id = str(row["mol_id"])

        # Try loading tree data from disk cache
        cache_path = self._get_cache_path(mol_id)
        if cache_path.exists():
            try:
                cached = torch.load(cache_path, weights_only=False)

                mz, intensities = self._load_spectrum(mol_id)
                binned_intensities = self.binner(mz, intensities)[
                    "spectrum"
                ].numpy()

                result = {
                    "name": mol_id,
                    "inten_targs": binned_intensities,
                    **cached,
                }
                self._cache_put(idx, result)
                return result
            except Exception:
                pass  # Fall through to recompute

        # Fallback: compute from scratch
        try:
            tree_data = self._load_magma_tree(mol_id)
            tree_data["root_canonical_smiles"] = smiles_from_inchi(
                tree_data["root_inchi"]
            )
            processed_tree = self.tree_processor.process_tree_inten(tree_data)

            mz, intensities = self._load_spectrum(mol_id)
            binned_intensities = self.binner(mz, intensities)[
                "spectrum"
            ].numpy()

            result = {
                "name": mol_id,
                "smiles": tree_data["root_canonical_smiles"],
                "inten_targs": binned_intensities,
                **processed_tree["dgl_tree"],
            }

            # Lazy-cache tree data to disk for next run
            try:
                cache_data = {
                    "smiles": tree_data["root_canonical_smiles"],
                    **processed_tree["dgl_tree"],
                }
                torch.save(cache_data, cache_path)
            except Exception:
                pass

            self._cache_put(idx, result)
            return result

        except Exception as e:
            logging.error(f"Error processing {mol_id}: {e}")
            return None

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate function for intensity prediction."""
        batch = [item for item in batch if item is not None]
        if len(batch) == 0:
            return None
        names = [item["name"] for item in batch]
        smiles = [item["smiles"] for item in batch]

        # Fragment graphs (input)
        frag_graphs = [item["dgl_frags"] for item in batch]
        frag_graphs_flat = [
            graph for graphs in frag_graphs for graph in graphs
        ]

        # Number of fragments per item
        num_frags = torch.tensor(
            [len(graphs) for graphs in frag_graphs], dtype=torch.long
        )

        # Root representations
        root_reprs = self._collate_root_representations(batch)

        # Fragment graphs batch
        frag_batch = dgl.batch(frag_graphs_flat) if frag_graphs_flat else None

        # Index mapping for root to fragments
        root_inds = torch.arange(
            len(frag_graphs), dtype=torch.long
        ).repeat_interleave(num_frags)

        # Broken bonds - pad to 2D tensor [batch, max_frags]
        max_num_frags = max(len(item["max_broken"]) for item in batch)
        padded_max_broken = torch.zeros(
            len(batch), max_num_frags, dtype=torch.long
        )
        for i, item in enumerate(batch):
            n = len(item["max_broken"])
            padded_max_broken[i, :n] = torch.tensor(
                np.array(item["max_broken"]), dtype=torch.long
            )
        max_broken = padded_max_broken

        # Intensity targets (binned spectra)
        inten_targets = torch.tensor(
            np.array([item["inten_targs"] for item in batch]),
            dtype=torch.float,
        )

        # Optional: Formula vectors if available
        form_data = {}
        if "form_vecs" in batch[0]:  # Checking `form_vecs` key
            # Formula vectors have fixed dimension (ELEMENT_DIM = 18)
            max_num_frags_form = max(len(item["form_vecs"]) for item in batch)
            form_dim = (
                batch[0]["form_vecs"][0].shape[0]
                if len(batch[0]["form_vecs"]) > 0
                else ELEMENT_DIM
            )

            padded_form_vecs = torch.zeros(
                len(batch), max_num_frags_form, form_dim, dtype=torch.long
            )
            for i, item in enumerate(batch):
                n = len(item["form_vecs"])
                if n > 0:
                    padded_form_vecs[i, :n] = torch.tensor(
                        np.array(item["form_vecs"]), dtype=torch.long
                    )

            # Also include formulae as strings for isotope generation
            # This requires that 'formulae_str' is generated by your TreeProcessor
            # Let's assume you add a key like 'frag_formulae_str_list' in your dataset item.
            # If your TreeProcessor outputs formula strings directly in `item["frag_formulae_str_list"]`
            # then you collect it similarly to other padded lists.
            # For this context, let's pass a list of lists of strings, padding with empty strings
            # to match `max_num_frags`.
            padded_frag_formulae_str = []
            for item in batch:
                current_mol_formulas = item.get(
                    "frag_formulae_str_list", []
                )  # Assuming this key exists
                padded_mol_formulas = current_mol_formulas + [""] * (
                    max_num_frags_form - len(current_mol_formulas)
                )
                padded_frag_formulae_str.append(padded_mol_formulas)

            root_vecs = torch.tensor(
                np.array([item["root_form_vec"] for item in batch]),
                dtype=torch.long,
            )
            form_data = {
                "root_form_vecs": root_vecs,
                "frag_form_vecs": padded_form_vecs,
                "formulae": padded_frag_formulae_str,  # Add this for isotope generation
            }

        # Optional: Masses if available
        mass_data = {}
        if "masses" in batch[0]:
            # Pad masses to same shape
            masses_list = [item["masses"] for item in batch]
            max_frags_mass = max(m.shape[0] for m in masses_list)
            max_shifts = max(m.shape[-1] for m in masses_list)

            # NOTE: Use `max_frags_mass` to ensure mass tensor is correctly padded
            padded_masses = torch.zeros(
                len(batch), max_frags_mass, 1, max_shifts
            )
            for i, masses in enumerate(masses_list):
                h, c, w = masses.shape  # c should be 1
                padded_masses[i, :h, :c, :w] = torch.tensor(masses)

            mass_data = {"masses": padded_masses}

        # `max_add_hs` and `max_remove_hs`
        # These need to be padded like `broken_bonds` to `[batch_size, max_frags]`
        additional_data = {}

        # Get the maximum number of fragments across all molecules in this batch
        # This `num_frags` tensor has shape [batch_size], where each element is the
        # actual number of fragments for that molecule.
        # So, we take the max from this tensor.
        max_num_frags_global = torch.max(
            num_frags
        ).item()  # Use torch.max() on the tensor

        for key in ["max_add_hs", "max_remove_hs"]:
            # Check if key exists in the first item AND if the list it holds is not empty/None
            if (
                key in batch[0]
                and batch[0][key] is not None
                and len(batch[0][key]) > 0
            ):
                padded_tensor = torch.zeros(
                    len(batch), max_num_frags_global, dtype=torch.long
                )
                for i, item_in_batch in enumerate(
                    batch
                ):  # Renamed 'item' to 'item_in_batch' for clarity
                    n_frags_current_mol = num_frags[
                        i
                    ].item()  # Actual number of fragments for current molecule
                    val_list = item_in_batch[
                        key
                    ]  # This is a list of values for each fragment in this molecule

                    if (
                        val_list is not None
                    ):  # Double check val_list is not None
                        current_vals = torch.tensor(val_list, dtype=torch.long)
                        # Pad up to the actual number of fragments for this molecule, then the rest is 0 for padded fragments
                        padded_tensor[i, :n_frags_current_mol] = current_vals[
                            :n_frags_current_mol
                        ]  # Slice current_vals to match n_frags_current_mol if it's too long

                additional_data[key] = padded_tensor
            else:  # If not provided in batch[0] or is None/empty, provide default zero tensor
                additional_data[key] = torch.zeros(
                    len(batch), max_num_frags_global, dtype=torch.long
                )

        return {
            "names": names,
            "smiles": smiles,
            "root_reprs": root_reprs,
            "frag_graphs": frag_batch,
            "inds": root_inds,
            "num_frags": num_frags,
            "broken_bonds": max_broken,
            "inten_targs": inten_targets,
            **form_data,
            **mass_data,
            **additional_data,  # This now contains padded max_add_hs and max_remove_hs
        }

    def _collate_root_representations(self, batch: List[Dict[str, Any]]):
        """Collate root representations."""
        root_reprs = [item["root_repr"] for item in batch]

        if isinstance(root_reprs[0], dgl.DGLGraph):
            return dgl.batch(root_reprs)
        elif isinstance(root_reprs[0], np.ndarray):
            return torch.tensor(np.vstack(root_reprs), dtype=torch.float)
        else:
            raise NotImplementedError(
                f"Unsupported root representation: {type(root_reprs[0])}"
            )

    def __del__(self):
        """Cleanup HDF5 files."""
        if hasattr(self, "trees_h5") and self.trees_h5 is not None:
            self.trees_h5.close()
        if hasattr(self, "spectra_h5") and self.spectra_h5 is not None:
            self.spectra_h5.close()
