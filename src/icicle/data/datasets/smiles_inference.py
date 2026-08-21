"""Inference-time datasets for SMILES-to-spectrum prediction.

These datasets are used during inference (not training).  They take a plain
list of SMILES strings and produce the root DGL graph for each molecule.
Fragment enumeration is deferred to the GPU inside
``_gpu_enumerate_fragments_batched``, keeping CPU workers lightweight.
"""

import csv
import logging
import os
from typing import List, Optional

import dgl
import torch
import torch.utils.data
from rdkit import Chem
from rdkit.Chem.inchi import MolToInchiKey

from icicle.data.fragmentation_engine import FragmentEngine


class RootGraphInferenceDataset(torch.utils.data.Dataset):
    """Lightweight CPU dataset for GPU-accelerated inference.

    Each worker only parses the SMILES and builds the root DGL graph
    (one ``featurize_frag`` call, no PE).  Fragment enumeration is deferred
    to the GPU inside ``_gpu_enumerate_fragments``, keeping CPU workers fast
    and the GPU saturated.

    Each item contains the root DGL graph (``ndata['n_id']`` set to arange)
    and the raw ``FragmentEngine`` (needed for visualisation in
    ``_format_spectrum_result``).

    If ``failure_log_path`` is given, failed molecules are appended to a CSV
    (one file per DataLoader worker, suffixed with the worker pid).
    """

    def __init__(
        self,
        smiles_list: List[str],
        predictor,
        failure_log_path: Optional[str] = None,
    ):
        self.smiles_list = smiles_list
        # Store only tree_processor (a pure-CPU object) so DataLoader workers
        # can be pickled without capturing any CUDA tensors or model weights.
        self.tree_processor = predictor.tree_processor
        self.failure_log_path = failure_log_path

    def _log_failure(self, idx: int, smiles: str, reason: str) -> None:
        if self.failure_log_path is None:
            return
        # Per-worker file avoids cross-process locking.
        path = f"{self.failure_log_path}.{os.getpid()}.csv"
        write_header = not os.path.exists(path)
        mol = Chem.MolFromSmiles(smiles)
        inchi = Chem.MolToInchi(mol) if mol is not None else ""
        inchikey = MolToInchiKey(mol) if mol is not None else ""
        with open(path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(
                    ["idx", "smiles", "inchi", "inchikey", "reason"]
                )
            writer.writerow([idx, smiles, inchi, inchikey, reason])

    def __len__(self) -> int:
        return len(self.smiles_list)

    def __getitem__(self, idx: int) -> dict:
        smiles = self.smiles_list[idx]
        try:
            engine = FragmentEngine(mol_str=smiles)
            root_frag = engine.get_root_frag()
            root_info = self.tree_processor.featurize_frag(
                frag=root_frag, engine=engine, add_random_walk=False
            )
            g = root_info["graph"]
            g.ndata["n_id"] = torch.arange(g.num_nodes(), dtype=torch.long)
            return {
                "idx": idx,
                "valid": True,
                "smiles": smiles,
                "engine": engine,
                "root_graph": g,
            }
        except Exception as e:
            logging.warning(f"Root graph build failed for {smiles!r}: {e}")
            self._log_failure(idx, smiles, str(e))
            return {"idx": idx, "valid": False}


def root_only_collate_fn(batch: List[dict]) -> dict:
    """Collate root-only items for the GPU enumeration path.

    ``batch_idx_range`` covers every item in this DataLoader batch
    (valid and invalid), not just the surviving ``valid`` items. Callers
    that reconstruct a batch's index span from ``min/max(item["idx"] for
    item in valid)`` alone silently shrink that span whenever an invalid
    item sits at the first or last position of the batch, permanently
    shifting every subsequent batch's positional alignment.
    """
    batch_idx_range = (batch[0]["idx"], batch[-1]["idx"])
    valid = [item for item in batch if item["valid"]]
    if not valid:
        return {
            "valid": [],
            "root_graphs": None,
            "batch_idx_range": batch_idx_range,
        }
    return {
        "valid": valid,
        "root_graphs": dgl.batch([item["root_graph"] for item in valid]),
        "batch_idx_range": batch_idx_range,
    }
