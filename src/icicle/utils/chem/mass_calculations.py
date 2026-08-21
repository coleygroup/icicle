"""Molecular mass calculation utilities."""

from rdkit import Chem
from rdkit.Chem.Descriptors import ExactMolWt


def mass_from_inchi(inchi: str) -> float:
    """Get the mass of a molecule from an InChI string.

    Args:
        inchi (str): InChI string of the molecule.

    Returns:
        float: Mass of the molecule.
    """
    mol = Chem.MolFromInchi(inchi)
    if mol is None:
        return 0
    else:
        return ExactMolWt(mol)


def mass_from_smi(smi: str) -> float:
    """Get the mass of a molecule from a SMILES string.

    Args:
        smi (str): SMILES string of the molecule.

    Return:
        float: Mass of the molecule.
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return 0
    else:
        return ExactMolWt(mol)
