"""Chemistry-related constants and lookup tables."""

import os
from typing import Dict, Set

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Atom

P_TBL = Chem.GetPeriodicTable()
FIX_H_IN_INCHI = False

if os.environ.get("FIX_H_IN_INCHI") == "True":
    FIX_H_IN_INCHI = True

ELECTRON_MASS = 0.00054858
CHEM_FORMULA_SIZE = "([A-Z][a-z]*)([0-9]*)"

VALID_ELEMENTS = [
    "C",
    "N",
    "P",
    "O",
    "S",
    "Si",
    "I",
    "H",
    "Cl",
    "F",
    "Br",
    "B",
    "Se",
    "Fe",
    "Co",
    "As",
    "Na",
    "K",
]

ELEMENT_TO_GROUP = {
    "C": 4,  # group 5
    "N": 3,  # group 4
    "P": 3,
    "O": 5,  # group 6
    "S": 5,
    "Si": 4,
    "I": 6,  # group 7 / halogens
    "H": 0,
    "Cl": 6,
    "F": 6,
    "Br": 6,
    "B": 2,  # group 3
    "Se": 5,
    "Fe": 7,  # transition metals
    "Co": 7,
    "As": 3,
    "Na": 1,  # alkali metals
    "K": 1,
}

ELEMENT_GROUP_DIM = len(set(ELEMENT_TO_GROUP.values()))
ELEMENT_GROUP_VECTORS = np.eye(ELEMENT_GROUP_DIM)

# Set the exact molecular weight?
# Use this to define an element priority queue
VALID_ATOM_NUM = [Atom(i).GetAtomicNum() for i in VALID_ELEMENTS]
CHEM_ELEMENT_NUM = len(VALID_ELEMENTS)

BINARY_BITS = 8
# Convert to onehot
ATOM_NUM_TO_ONEHOT = torch.zeros((max(VALID_ATOM_NUM) + 1, CHEM_ELEMENT_NUM))
ATOM_NUM_TO_ONEHOT[VALID_ATOM_NUM, torch.arange(CHEM_ELEMENT_NUM)] = 1

# Use Monoisotopic
VALID_MONO_MASSES = np.array(
    [P_TBL.GetMostCommonIsotopeMass(i) for i in VALID_ELEMENTS]
)
CHEM_MASSES = VALID_MONO_MASSES[:, None]

ELEMENT_VECTORS = np.eye(len(VALID_ELEMENTS))
ELEMENT_VECTORS_MASS = np.hstack([ELEMENT_VECTORS, CHEM_MASSES])
ELEMENT_TO_MASS = dict(zip(VALID_ELEMENTS, CHEM_MASSES.squeeze()))

ELEMENT_DIM = len(ELEMENT_VECTORS[0])

# Reasonable normalization vector for elements
# TODO; what does that mean?
# Estimated by max counts (+ 1 when zero)
NORM_VEC_MASS = np.array(
    [81, 19, 6, 34, 6, 6, 6, 158, 10, 17, 3, 1, 2, 1, 1, 2, 1, 1, 1471]
)

NORM_VEC = np.array(
    [81, 19, 6, 34, 6, 6, 6, 158, 10, 17, 3, 1, 2, 1, 1, 2, 1, 1]
)

# Hydrogen featurizer
MAX_H = 6

element_to_ind = dict(zip(VALID_ELEMENTS, np.arange(len(VALID_ELEMENTS))))
element_to_position = dict(zip(VALID_ELEMENTS, ELEMENT_VECTORS))
element_to_position_mass = dict(zip(VALID_ELEMENTS, ELEMENT_VECTORS_MASS))
element_to_group = {
    k: ELEMENT_GROUP_VECTORS[v] for k, v in ELEMENT_TO_GROUP.items()
}
