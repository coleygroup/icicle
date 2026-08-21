"""Common functions and utilities for ICICLE."""

# suppress RDKit warnings
from rdkit import RDLogger

from .chem import (
    # Constants
    ELECTRON_MASS,
    ELEMENT_DIM,
    ELEMENT_GROUP_DIM,
    ELEMENT_TO_MASS,
    ELEMENT_VECTORS,
    MAX_H,
    NORM_VEC,
    VALID_ELEMENTS,
    ISOTOPE_PATTERNS_SIMPLIFIED,
    ISOTOPE_PATTERNS,
    # Mass spec utilities
    element_to_group,
    # Element utilities
    element_to_ind,
    element_to_position,
    formula_from_inchi,
    formula_from_smi,
    # Formula utilities
    formula_to_dense,
    get_all_subsets,
    # Fingerprints
    get_morgan_fp_from_inchi,
    inchi_key_from_inchi,
    get_morgan_fp_from_smi,
    inchi_from_smiles,
    inchikey_from_smiles,
    remove_stereochemistry,
    # Mass calculations
    mass_from_smi,
    parse_spectra,
    process_common_spec_file,
    filter_spectra_by_intensity,
    # Molecular structure handling
    smi_inchi_round_mol,
    smiles_from_inchi,
    standardize_formula,
    uncharged_formula,
    vec_to_formula,
)
from .data import HDF5Dataset, load_ckpt
from .caching import batch_func, chunked_parallel
from .visualization import palette, set_style, cmap
from .graph_utils import random_walk_pe, pad_packed_tensor
from .torch_utils import safe_binom
from .external_services import get_compound_class
