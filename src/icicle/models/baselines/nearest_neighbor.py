"""Nearest-neighbor spectrum baseline.

Predicts the (binned, max-normalized) ground-truth spectrum of the training
molecule with the highest Morgan/Tanimoto fingerprint similarity to the query.
"""

import logging
import time
from typing import Any, Dict

import h5py
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import DataStructs, rdFingerprintGenerator
from tqdm import tqdm

from icicle.data.transforms.spectrum import SpecBinner
from icicle.models.base_model import BaseSpectrumPredictor


class NearestNeighborBaseline(BaseSpectrumPredictor):
    """Baseline that predicts the nearest training molecule's true spectrum.

    Nearest neighbor is found via Tanimoto similarity of Morgan fingerprints
    over all training molecules. Fingerprint radius/nBits match the model's
    own molecule featurization (icicle.data.transforms.mol.MolTransform
    defaults: radius=2, nBits=4096), for consistency across the paper.
    """

    def __init__(
        self,
        min_mz: float,
        max_mz: float,
        bin_width: float,
        spectra_path: str,
        labels_path: str,
        splits_path: str,
        radius: int = 2,
        n_bits: int = 4096,
        **kwargs,
    ):
        super().__init__(
            min_mz=min_mz, max_mz=max_mz, bin_width=bin_width, **kwargs
        )
        self.num_bins = int((max_mz - min_mz) / bin_width)
        self.mz_bins = np.linspace(
            min_mz, max_mz, self.num_bins, endpoint=False
        ).astype(np.float32)
        self.radius = radius
        self.n_bits = n_bits
        self.mfpgen = rdFingerprintGenerator.GetMorganGenerator(
            radius=radius, fpSize=n_bits
        )

        self.train_fps, self.train_spectra = self._build_train_index(
            spectra_path, labels_path, splits_path
        )

    def _build_train_index(
        self, spectra_path: str, labels_path: str, splits_path: str
    ):
        """Compute Morgan fingerprints and binned spectra for training set."""
        logging.info("Building nearest-neighbor training index...")

        splits_df = pd.read_csv(splits_path, sep="\t")
        train_mol_ids = set(
            splits_df[splits_df["split"] == "train"]["mol_id"]
            .astype(str)
            .tolist()
        )

        labels_df = pd.read_csv(labels_path, sep="\t")
        labels_df["mol_id"] = labels_df["mol_id"].astype(str)
        train_labels = labels_df[labels_df["mol_id"].isin(train_mol_ids)]

        binner = SpecBinner(
            min_mz=self.min_mz, max_mz=self.max_mz, bin_width=self.bin_width
        )

        fps = []
        spectra = []
        t0 = time.perf_counter()
        with h5py.File(spectra_path, "r") as hf:
            for _, row in tqdm(
                train_labels.iterrows(),
                total=len(train_labels),
                desc="NN baseline: indexing training fingerprints+spectra",
            ):
                mol_id = str(row["mol_id"])
                if mol_id not in hf:
                    continue
                mol = Chem.MolFromSmiles(row["standardized_smiles"])
                if mol is None:
                    continue
                fp = self.mfpgen.GetFingerprint(mol)

                group = hf[mol_id]
                binned_result = binner(
                    group["masses"][:], group["intensities"][:]
                )
                binned_intensities = binned_result["spectrum"].numpy()
                if binned_intensities.max() > 0:
                    binned_intensities = (
                        binned_intensities / binned_intensities.max()
                    )

                fps.append(fp)
                spectra.append(binned_intensities.astype(np.float32))

        elapsed = time.perf_counter() - t0
        logging.info(
            f"Indexed {len(fps)} training spectra for NN lookup in "
            f"{elapsed:.1f}s ({elapsed / max(len(fps), 1) * 1000:.2f} ms/mol)"
        )
        return fps, np.stack(spectra) if spectra else np.zeros(
            (0, self.num_bins), dtype=np.float32
        )

    def _nearest_spectrum(self, smiles: str) -> np.ndarray:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or len(self.train_fps) == 0:
            return np.zeros(self.num_bins, dtype=np.float32)
        query_fp = self.mfpgen.GetFingerprint(mol)
        similarities = DataStructs.BulkTanimotoSimilarity(
            query_fp, self.train_fps
        )
        best_idx = int(np.argmax(similarities))
        return self.train_spectra[best_idx]

    def predict_from_smiles(
        self, smiles: str, device: str = "cpu", **kwargs
    ) -> Dict[str, Any]:
        """Predict nearest training neighbor's spectrum for a SMILES string."""
        return {
            "smiles": smiles,
            "mz_bins": self.mz_bins,
            "intensities": self._nearest_spectrum(smiles),
            "num_fragments": 0,
            "fragments": {},
        }

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Dummy training step - this baseline doesn't actually train."""
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
