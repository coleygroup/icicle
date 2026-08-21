"""Molecule transforms."""

from typing import Dict, List
import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Descriptors

from icicle.data.transforms.base import MolTransform


class MolFingerprints(MolTransform):
    """Generate molecular fingerprints."""

    def __init__(
        self,
        fp_types: List[str] = ["morgan"],
        morgan_radius: int = 2,
        morgan_bits: int = 4096,
    ):
        self.fp_types = fp_types
        self.morgan_radius = morgan_radius
        self.morgan_bits = morgan_bits

    def __call__(self, smiles: str) -> Dict[str, torch.Tensor]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            total_size = self._get_total_size()
            return {
                "fingerprints": torch.zeros(total_size, dtype=torch.float32)
            }

        fps = []
        for fp_type in self.fp_types:
            if fp_type == "morgan":
                from rdkit.Chem import rdFingerprintGenerator

                mfpgen = rdFingerprintGenerator.GetMorganGenerator(
                    radius=self.morgan_radius, fpSize=self.morgan_bits
                )
                fp = mfpgen.GetFingerprint(mol)
                fps.append(np.array(fp, dtype=np.float32))
            elif fp_type == "maccs":
                fp = rdMolDescriptors.GetMACCSKeysFingerprint(mol)
                fps.append(np.array(fp, dtype=np.float32))
            elif fp_type == "rdkit":
                fp = Chem.RDKFingerprint(mol)
                fps.append(np.array(fp, dtype=np.float32))

        combined = np.concatenate(fps) if len(fps) > 1 else fps[0]
        return {"fingerprints": torch.tensor(combined)}

    def _get_total_size(self) -> int:
        size = 0
        for fp_type in self.fp_types:
            if fp_type == "morgan":
                size += self.morgan_bits
            elif fp_type == "maccs":
                size += 167
            elif fp_type == "rdkit":
                size += 2048
        return size

    @property
    def output_size(self) -> int:
        return self._get_total_size()


class MolDescriptors(MolTransform):
    """Generate molecular descriptors."""

    def __init__(self, descriptors: List[str] = None):
        if descriptors is None:
            self.descriptors = [
                "MolWt",
                "LogP",
                "NumHDonors",
                "NumHAcceptors",
                "TPSA",
            ]
        else:
            self.descriptors = descriptors

    def __call__(self, smiles: str) -> Dict[str, torch.Tensor]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {
                "descriptors": torch.zeros(
                    len(self.descriptors), dtype=torch.float32
                )
            }

        values = []
        for desc_name in self.descriptors:
            try:
                if hasattr(Descriptors, desc_name):
                    value = getattr(Descriptors, desc_name)(mol)
                else:
                    value = 0.0

                if np.isnan(value) or np.isinf(value):
                    value = 0.0
                values.append(float(value))
            except:
                values.append(0.0)

        return {"descriptors": torch.tensor(values, dtype=torch.float32)}

    @property
    def output_size(self) -> int:
        return len(self.descriptors)
