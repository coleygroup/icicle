"""Substructure utilities.

TODO
"""

from typing import Tuple

import numpy as np

from icicle.utils.chem.formula_utils import formula_to_dense
from icicle.utils.chem.constants import ELEMENT_VECTORS, VALID_MONO_MASSES


def get_all_subsets(chem_formula: str) -> Tuple[np.ndarray, np.ndarray]:
    """get_all_subsets.

    Args:
        chem_formula (str): Chem formula
    Return:
        Tuple of vecs and their masses
    """
    dense_formula = formula_to_dense(chem_formula)
    non_zero = np.argwhere(dense_formula > 0).flatten()

    vectorized_formula = [
        ELEMENT_VECTORS[nonzero_ind]
        * np.arange(0, dense_formula[nonzero_ind] + 1)[:, None]
        for nonzero_ind in non_zero
    ]

    cross_prod = reduce(cross_sum, vectorized_formula)
    cross_prod_inds = rdbe_filter(cross_prod)
    cross_prod = cross_prod[cross_prod_inds]

    all_masses = np.einsum("ij,j->i", cross_prod, VALID_MONO_MASSES)
    return cross_prod, all_masses
