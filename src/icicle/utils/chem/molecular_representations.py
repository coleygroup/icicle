"""Tools for handling different molecular representations (SMILES, InChI,
etc.).

TODO
"""

from rdkit import Chem

from .constants import FIX_H_IN_INCHI


def inchikey_from_smiles(smi: str) -> str:
    """Get InChIKey from SMILES.

    Args:
        smi (str): SMILES

    Returns
    -------
        str: InChIKey
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return ""
    else:
        return Chem.MolToInchiKey(mol)


def inchi_from_smiles(smi: str) -> str:
    """Get InChI from SMILES.

    Args:
        smi (str): SMILES

    Returns
    -------
        str: InChI
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return ""
    elif FIX_H_IN_INCHI:
        return Chem.MolToInchi(mol, options="/FIX_H")
    else:
        return Chem.MolToInchi(mol)


def smi_inchi_round_mol(smi: str) -> Chem.Mol:
    """Do a round trip from SMILES to InChI to RDKit mol.

    Args:
        smi (str): SMILES

    Returns
    -------
        Chem.Mol: RDKit mol object
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None

    inchi = (
        Chem.MolToInchi(mol, options="/FIX_H")
        if FIX_H_IN_INCHI
        else Chem.MolToInchi(mol)
    )
    if inchi is None:
        return None

    mol = Chem.MolFromInchi(inchi)
    return mol


def smiles_from_inchi(inchi: str) -> str:
    """Get SMILES from InChI.

    Args:
        inchi (str): InChI

    Returns
    -------
        str: SMILES
    """
    mol = Chem.MolFromInchi(inchi)
    if mol is None:
        return ""
    else:
        return Chem.MolToSmiles(mol)


def remove_stereochemistry(mol: str, mol_type: str = "smi") -> str:
    """Remove stereochemistry from a molecule.

    Args:
        mol (str): molecule
        mol_type (str): molecule type (smi, inchi, mol)

    Returns
    -------
        str: molecule without stereochemistry
    """
    if mol_type == "smi":
        mol = Chem.MolFromSmiles(mol)
    elif mol_type == "inchi":
        mol = Chem.MolFromInchi(mol)
    elif mol_type == "mol":
        mol = mol
    else:
        raise ValueError(f"Unknown mol_type={mol_type}")

    if mol is None:
        return
    else:
        Chem.RemoveStereochemistry(mol)

    if mol_type == "smi":
        return Chem.MolToSmiles(mol)
    elif mol_type == "inchi":
        if FIX_H_IN_INCHI:
            return Chem.MolToInchi(mol, options="/FIX_H")
        else:
            return Chem.MolToInchi(mol)
    else:
        return mol


def get_mol_from_structure_string(
    structure_string: str, structure_type: str
) -> Chem.Mol:
    """Get RDKit mol object from structure string.

    Args:
        structure_string (str): structure_string
        structure_type (str): structure_type

    Returns
    -------
        mol: RDKit mol object
    """
    if structure_type == "InChI":
        return Chem.MolFromInchi(structure_string)
    else:
        return Chem.MolFromSmiles(structure_string)


def inchi_key_from_inchi(inchi: str) -> str:
    """Get InChIKey from InChI.

    Args:
        inchi (str): InChI

    Returns
    -------
        str: InChIKey
    """

    mol = Chem.MolFromInchi(inchi)
    if mol is None:
        raise ValueError(f"Invalid InChI: {inchi}")

    return Chem.MolToInchiKey(mol)
