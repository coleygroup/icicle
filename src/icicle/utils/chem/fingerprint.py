"""Utility functions for generating fingerprints from molecules."""

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs


def get_morgan_fp_from_mol(
    mol: Chem.Mol, nbits: int = 2048, radius: int = 3
) -> np.ndarray:
    """Get Morgan fingerprint from a molecule.

    Args:
        mol (Chem.Mol): RDKit molecule.
        nbits (int): Number of bits in the fingerprint.
        radius (int): Radius of the fingerprint.

    Returns:
        np.ndarray: Morgan fingerprint as a numpy array.
    """

    if mol is None:
        raise ValueError("Molecule is None")

    curr_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)

    fingerprint = np.zeros((0,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(curr_fp, fingerprint)
    return fingerprint


def get_morgan_fp_from_smi(
    smi: str, nbits: int = 2048, radius: int = 3
) -> np.ndarray:
    """Get Morgan fingerprint from a SMILES string.

    Args:
        smi (str): SMILES string of the molecule.
        nbits (int): Number of bits in the fingerprint.
        radius (int): Radius of the fingerprint.

    Returns:
        np.ndarray: Morgan fingerprint as a numpy array.
    """
    return get_morgan_fp_from_mol(
        Chem.MolFromSmiles(smi), nbits=nbits, radius=radius
    )


def get_morgan_fp_from_inchi(
    inchi: str, nbits: int = 2048, radius: int = 3
) -> np.ndarray:
    """Get Morgan fingerprint from a InChI string.

    Args:
        inchi (str): InChI string of the molecule.
        nbits (int): Number of bits in the fingerprint.
        radius (int): Radius of the fingerprint.

    Returns:
        np.ndarray: Morgan fingerprint as a numpy array.
    """
    return get_morgan_fp_from_mol(
        Chem.MolFromInchi(inchi), nbits=nbits, radius=radius
    )
