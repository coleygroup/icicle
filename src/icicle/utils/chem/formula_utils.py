"""Chemical formula manipulation and analysis tools."""

import re
from typing import Any, Optional

import numpy as np
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

from .constants import (
    CHEM_FORMULA_SIZE,
    ELEMENT_TO_MASS,
    NORM_VEC_MASS,
    VALID_ELEMENTS,
    element_to_position,
    element_to_position_mass,
)


def formula_to_dense(chem_formula: str) -> np.ndarray:
    """Convert a chemical formula to a dense vector. A dense vector is a vector
    of zeros with a 1 at the position of the element. The position is the index
    of the element in the VALID_ELEMENTS list.

    Args:
        chem_formula (str): Input chemical formula
    Return:
        np.ndarray: Dense vector of the chemical formula
    """
    # Pre-allocate output array
    dense_vec = np.zeros(len(element_to_position))

    # Early return for empty formula
    if not chem_formula:
        return dense_vec

    # Process formula directly without building intermediate lists
    for chem_symbol, num in re.findall(CHEM_FORMULA_SIZE, chem_formula):
        num = 1 if num == "" else int(num)
        position = element_to_position[chem_symbol]
        dense_vec += position * num

    return dense_vec


def formula_to_dense_mass(chem_formula: str) -> np.ndarray:
    """Convert a chemical formula to a dense vector including full compound
    mass. A dense vector is a vector of zeros with a 1 at the position of the
    element. The position is the index of the element in the VALID_ELEMENTS
    list.

    Args:
        chem_formula (str): Input chemical formula
    Return:
        np.ndarray: Dense vector of the chemical formula including full compound mass
    """
    total_onehot = []
    for chem_symbol, num in re.findall(CHEM_FORMULA_SIZE, chem_formula):
        # Convert num to int
        num = 1 if num == "" else int(num)
        one_hot = element_to_position_mass[chem_symbol].reshape(1, -1)
        one_hot_repeats = np.repeat(one_hot, repeats=num, axis=0)
        total_onehot.append(one_hot_repeats)

    # Check if null
    if len(total_onehot) == 0:
        dense_vec = np.zeros(len(element_to_position_mass["H"]))
    else:
        dense_vec = np.vstack(total_onehot).sum(0)

    return dense_vec


def formula_to_dense_mass_norm(chem_formula: str) -> np.ndarray:
    """Convert a chemical formula to a dense vector including full compound
    mass and normalized. A dense vector is a vector of zeros with a 1 at the
    position of the element. The position is the index of the element in the
    VALID_ELEMENTS list. The vector is normalized by the mass of the compound.

    Args:
        chem_formula (str): Input chemical formula
    Return:
        np.ndarray: Dense vector of the chemical formula including full compound mass and normalized
    """
    dense_vec = formula_to_dense_mass(chem_formula)
    dense_vec = dense_vec / NORM_VEC_MASS

    return dense_vec


def formula_mass(chem_formula: str) -> float:
    """Get formula mass from a chemical formula. Formula mass is the sum of the
    mass of the elements in the formula.

    Args:
        chem_formula (str): Input chemical formula
    Return:
        float: Formula mass
    """
    mass = 0
    for chem_symbol, num in re.findall(CHEM_FORMULA_SIZE, chem_formula):
        # Convert num to int
        num = 1 if num == "" else int(num)
        mass += ELEMENT_TO_MASS[chem_symbol] * num
    return mass


def formula_difference(formula_1: Any, formula_2: Any) -> Any:
    """Calculate the difference between two chemical formulae.

    Args:
        formula_1 (Any): First chemical formula
        formula_2 (Any): Second chemical formula
    Return:
        Any: Difference between the two chemical formulae
    """
    form_1 = {
        chem_symbol: (int(num) if num != "" else 1)
        for chem_symbol, num in re.findall(CHEM_FORMULA_SIZE, formula_1)
    }
    form_2 = {
        chem_symbol: (int(num) if num != "" else 1)
        for chem_symbol, num in re.findall(CHEM_FORMULA_SIZE, formula_2)
    }

    for k, v in form_2.items():
        if k in form_1:
            form_1[k] = form_1[k] - form_2[k]
        else:
            form_1[k] = -form_2[k]

    out_formula = "".join([f"{k}{v}" for k, v in form_1.items() if v != 0])
    return out_formula


def standardize_formula(formula: Any) -> Any:
    """Standardize a chemical formula. A standardized formula is a formula with
    the elements in alphabetical order.

    Args:
        formula (Any): Input chemical formula
    Return:
        Any: Standardized chemical formula
    """
    return vec_to_formula(formula_to_dense(formula))


def vec_to_formula(form_vec: np.ndarray) -> str:
    """Convert a dense vector to a chemical formula.

    Args:
        form_vec (np.ndarray): Input dense vector
    Return:
        str: Chemical formula
    """
    build_str = ""
    if hasattr(form_vec, "device") and str(form_vec.device) != "cpu":
        form_vec = form_vec.cpu()

    for i in np.argwhere(form_vec > 0).flatten():
        el = VALID_ELEMENTS[i]
        ct = int(form_vec[i])
        new_item = f"{el}{ct}" if ct > 1 else f"{el}"
        build_str = build_str + new_item
    return build_str


def uncharged_formula(mol: Chem.Mol, mol_type: str = "mol") -> Optional[str]:
    """Compute uncharged formula from a molecule.

    Args:
        mol (Chem.Mol): Input molecule
        mol_type (str): Type of molecule. Can be "mol", "smiles", or "inchi".
    Return:
        Optional[str]: Uncharged chemical formula
    """
    if mol_type == "mol":
        chem_formula = CalcMolFormula(mol)
    elif mol_type == "smiles":
        mol = Chem.MolFromSmiles(mol)
        if mol is None:
            return None
        chem_formula = CalcMolFormula(mol)
    elif mol_type == "inchi":
        mol = Chem.MolFromInchi(mol)
        if mol is None:
            return None
        chem_formula = CalcMolFormula(mol)
    else:
        raise ValueError()

    return re.findall(r"^([^\+,^\-]*)", chem_formula)[0]


def formula_from_smi(smi: str) -> Optional[str]:
    """Get chemical formula from a SMILES string.

    Args:
        smi (str): SMILES string

    Return:
        Optional[str]: Chemical formula
    """
    return uncharged_formula(smi, mol_type="smiles")


def formula_from_inchi(inchi: str) -> Optional[str]:
    """Get chemical formula from a InChI string.

    Args:
        inchi (str): InChI string

    Return:
        Optional[str]: Chemical formula
    """
    return uncharged_formula(inchi, mol_type="inchi")
