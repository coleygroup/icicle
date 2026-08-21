"""Molecular Fragmentation Engine.

This module provides tools for combinatorial fragmentation of molecules by pulling atoms.
It implements a WL (Weisfeiler-Lehman) based approach to generate unique hashes for
molecular fragments and handles both single and multiple bond breaking scenarios.

Key components:
- FragmentEngine: Main class for molecular fragmentation
- Supporting functions for fragment manipulation and hashing
"""

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import IntEnum
from hashlib import blake2b
from typing import Any, DefaultDict, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from rdkit import Chem

from icicle.utils import (
    ELEMENT_TO_MASS,
    ELEMENT_VECTORS,
    VALID_ELEMENTS,
    element_to_ind,
    formula_to_dense,
    vec_to_formula,
)
from icicle.data.isotope_distribution import IsotopePatternCalculator


class BondType(IntEnum):
    """Bond types and their weights for fragmentation."""

    AROMATIC = 2
    DOUBLE = 2
    TRIPLE = 3
    SINGLE = 1


BOND_WEIGHTS = {
    Chem.rdchem.BondType.names["AROMATIC"]: BondType.AROMATIC,
    Chem.rdchem.BondType.names["DOUBLE"]: BondType.DOUBLE,
    Chem.rdchem.BondType.names["TRIPLE"]: BondType.TRIPLE,
    Chem.rdchem.BondType.names["SINGLE"]: BondType.SINGLE,
}

MAX_BONDS = max(BondType) + 1
MAX_ATOM_BONDS = 6


@dataclass
class HeteroWeights:
    """Bond weights for different types."""

    cc_bond: int = 2  # Weight for C-C bonds
    other_bond: int = 1  # Weight for other bonds


@dataclass
class FragmentationParams:
    """Parameters for molecular fragmentation."""

    max_tree_depth: int = 3
    max_broken_bonds: int = 6
    mol_str_type: str = "smiles"
    num_h_shifts: int = 1
    hetero_weights: HeteroWeights = field(default_factory=HeteroWeights)
    detect_isotope_patterns: bool = True
    min_isotope_intensity: float = 0.01


@dataclass
class Fragment:
    """Represents a molecular fragment with its properties."""

    frag: int
    id: int
    sibling_hashes: List[str]
    parents: List[str]
    parent_hashes: List[str]
    parent_ind_removed: List[str]
    max_broken: int
    tree_depth: int
    score: float
    base_mass: float
    form: str
    frag_hs: int
    max_remove_hs: int
    max_add_hs: int


@dataclass
class FragmentInfo:
    """Information about a generated fragment."""

    new_frag: int
    new_hash: str
    removed_atom: int
    rm_bond_t: int


@dataclass
class DrawInfo:
    """Information for drawing a fragment."""

    hatoms: List[int]
    hbonds: List[int]
    mol: Chem.Mol
    smiles: str


class FragmentEngine:
    """Engine for molecular fragmentation analysis.

    This class handles the fragmentation of molecules through bond breaking and
    provides tools for analyzing the resulting fragments, including mass calculation,
    formula generation, and fragment tree construction.

    Attributes
    ----------
        smiles (str): SMILES representation of the molecule
        inchi (str): InChI representation of the molecule
        mol (Chem.Mol): RDKit molecule object
        natoms (int): Number of atoms in the molecule
        atom_symbols (List[str]): List of atomic symbols
        atom_hs (np.ndarray): Number of hydrogens per atom
        total_hs (int): Total number of hydrogens
        atom_weights (np.ndarray): Atomic weights without hydrogens
        atom_weights_h (np.ndarray): Atomic weights including hydrogens
        full_weight (float): Total molecular weight
        frag_to_entry (Dict[str, Fragment]): Maps fragment hashes to Fragment objects
    """

    def __init__(
        self, mol_str: str, params: Optional[FragmentationParams] = None
    ):
        """Initialize FragmentEngine.

        Args:
            mol_str: SMILES or InChI string of the molecule
            params: Fragmentation parameters, uses defaults if None

        Raises
        ------
            RuntimeError: If molecule cannot be parsed
            ValueError: If mol_str_type is invalid
        """
        if params is None:
            params = FragmentationParams()

        self.params = params
        self._initialize_molecule(mol_str)
        self._setup_fragmentation_params()
        self._setup_atoms()
        self._setup_bonds()
        self._initialize_isotope_calculator()

    def _initialize_isotope_calculator(self) -> None:
        """Initialize isotope calculator."""
        self.isotope_calculator = IsotopePatternCalculator(
            min_intensity_threshold=self.params.min_isotope_intensity
        )

    def _initialize_molecule(self, mol_str: str) -> None:
        """Initialize molecule from string representation."""
        if self.params.mol_str_type == "smiles":
            self.smiles = mol_str
            self.mol = Chem.MolFromSmiles(self.smiles)
            if self.mol is None:
                raise RuntimeError(f"Could not parse SMILES: {self.smiles}")
            self.inchi = Chem.MolToInchi(self.mol)
            self.mol = Chem.MolFromInchi(self.inchi)
        elif self.params.mol_str_type == "inchi":
            self.inchi = mol_str
            self.mol = Chem.MolFromInchi(self.inchi)
            if self.mol is None:
                raise RuntimeError(f"Could not parse InChI: {self.inchi}")
            self.smiles = Chem.MolToSmiles(self.mol)
        else:
            raise ValueError(
                f"Invalid mol_str_type: {self.params.mol_str_type}"
            )

        if self.mol is None:
            raise RuntimeError(
                f"Invalid molecule encountered. SMILES: {self.smiles}, InChI: {self.inchi}"
            )

        self.natoms = self.mol.GetNumAtoms()
        Chem.Kekulize(self.mol, clearAromaticFlags=True)

    def _setup_atoms(self) -> None:
        """Setup atomic properties and weights."""
        self.atom_symbols = [atom.GetSymbol() for atom in self.mol.GetAtoms()]
        self.atom_symbols_ar = np.array(self.atom_symbols)
        self.atom_hs = np.array(
            [
                atom.GetNumImplicitHs() + atom.GetNumExplicitHs()
                for atom in self.mol.GetAtoms()
            ]
        )
        self.total_hs = self.atom_hs.sum()
        self.atom_weights = np.array(
            [ELEMENT_TO_MASS[symbol] for symbol in self.atom_symbols]
        )
        self.atom_weights_h = (
            self.atom_hs * ELEMENT_TO_MASS["H"] + self.atom_weights
        )
        self.full_weight = np.sum(self.atom_weights_h)
        # Integer initial hashes for WL — stable element index, no string needed
        # Cast to plain Python int to avoid numpy int64 overflow in hash arithmetic
        self._atom_init_hashes: List[int] = [
            int(element_to_ind.get(s, 0)) for s in self.atom_symbols
        ]

    def _setup_bonds(self) -> None:
        """Setup bond information and scoring."""
        # Initialize bond storage structures
        self._initialize_bond_arrays()

        # Process all bonds
        for bond in self.mol.GetBonds():
            self._process_bond(bond)

    def _process_bond(self, bond: Chem.rdchem.Bond) -> None:
        """Process a single bond and update bond-related data structures.

        Args:
            bond: RDKit bond object

        Updates:
            - bonded_atoms: Lists of bonded atoms
            - bonded_types: Lists of bond types
            - bonded_atoms_np: NumPy array of bonded atoms
            - bonded_types_np: NumPy array of bond types
            - num_bonds_np: NumPy array of bond counts
            - bond_to_type: Dictionary mapping bond bits to types
            - bonds: Set of bond bits
            - bonds_list: List of bond bits
            - bond_types_list: List of bond types
            - bond_inds_list: List of bond atom indices
            - bondscore: Dictionary mapping bond bits to scores
        """
        # Get atoms involved in the bond
        atom1, atom2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()

        # Update bonded atoms lists
        self.bonded_atoms[atom1].append(atom2)
        self.bonded_atoms[atom2].append(atom1)

        # Update numpy arrays for bonded atoms
        self.bonded_atoms_np[atom1, self.num_bonds_np[atom1]] = atom2
        self.bonded_atoms_np[atom2, self.num_bonds_np[atom2]] = atom1

        # Create bond bits (binary representation of bond)
        bond_bits = 1 << atom1 | 1 << atom2

        # Calculate bond score based on bond type and atoms involved
        bond_score = (
            BOND_WEIGHTS[bond.GetBondType()]
            * self.hetero_weights[
                self.atom_symbols[atom1] != "C"
                or self.atom_symbols[atom2] != "C"
            ]
        )

        # Get bond type
        bond_type = BOND_WEIGHTS[bond.GetBondType()]

        # Update bonded types lists
        self.bonded_types[atom1].append(bond_type)
        self.bonded_types[atom2].append(bond_type)

        # Update numpy arrays for bond types
        self.bonded_types_np[atom1, self.num_bonds_np[atom1]] = bond_type
        self.bonded_types_np[atom2, self.num_bonds_np[atom2]] = bond_type

        # Increment number of bonds for both atoms
        self.num_bonds_np[atom1] += 1
        self.num_bonds_np[atom2] += 1

        # Update bond type mapping
        self.bond_to_type[bond_bits] = bond_type

        # Update bond score mapping
        self.bondscore[bond_bits] = bond_score

        # Update bond collections if this is a new bond
        if bond_bits not in self.bonds:
            self.bonds_list.append(bond_bits)
            self.bond_types_list.append(bond_type)
            self.bond_inds_list.append((atom1, atom2))

        # Add to set of bonds
        self.bonds.add(bond_bits)

    def _initialize_bond_arrays(self) -> None:
        """Initialize arrays for storing bond information."""
        self.bonded_atoms: List[List[int]] = [[] for _ in self.atom_symbols]
        self.bonded_types: List[List[int]] = [[] for _ in self.atom_symbols]
        self.bonded_atoms_np = np.zeros(
            (self.natoms, MAX_ATOM_BONDS), dtype=int
        )
        self.bonded_types_np = np.zeros(
            (self.natoms, MAX_ATOM_BONDS), dtype=int
        )
        self.num_bonds_np = np.zeros(self.natoms, dtype=int)
        self.bond_to_type: Dict[int, int] = {}
        self.bonds: Set[int] = set()
        self.bonds_list: List[int] = []
        self.bond_types_list: List[int] = []
        self.bond_inds_list: List[Tuple[int, int]] = []
        self.bondscore: Dict[int, float] = {}

    def _setup_fragmentation_params(self) -> None:
        """Setup parameters for fragmentation."""
        self.shift_buckets = (
            np.arange(self.params.num_h_shifts * 2 + 1)
            - self.params.num_h_shifts
        )
        self.shift_bucket_inds = np.arange(self.params.num_h_shifts * 2 + 1)
        self.shift_bucket_masses = self.shift_buckets * ELEMENT_TO_MASS["H"]
        self.frag_to_entry: Dict[str, Fragment] = {}
        self._hash_cache: Dict[int, str] = {}

        self.hetero_weights = {
            False: self.params.hetero_weights.cc_bond,
            True: self.params.hetero_weights.other_bond,
        }

    def _create_root_frag(self) -> Any:
        return (1 << self.natoms) - 1

    def _create_root_fragment(self, frag: int, cur_id: int) -> Fragment:
        """Create the root fragment of the fragmentation tree."""
        score = self.score_fragment(frag)[1]
        stats = self.atom_pass_stats(frag, depth=0)

        return Fragment(
            frag=frag,
            id=cur_id,
            sibling_hashes=[],
            parents=[],
            parent_hashes=[],
            parent_ind_removed=[],
            max_broken=0,
            tree_depth=0,
            score=float(score),
            base_mass=float(stats["base_mass"]),
            form=str(stats["form"]),
            frag_hs=int(stats["frag_hs"]),
            max_remove_hs=int(stats["max_remove_hs"]),
            max_add_hs=int(stats["max_add_hs"]),
        )

    def _process_fragment(
        self, frag_hash: int, cur_id: int, new_fragments: List[str]
    ) -> int:
        """Process a fragment to generate its sub-fragments."""
        parent_entry = self.frag_to_entry[frag_hash]  # type: ignore
        fragment = parent_entry.frag
        parent_broken = parent_entry.max_broken

        for atom in range(self.natoms):
            extended_fragments = self.remove_atom(fragment, atom)

            sibling_hashes = {
                frag_info.new_hash for frag_info in extended_fragments
            }

            for frag_info in extended_fragments:
                removed_atom = frag_info.removed_atom
                new_frag_hash = frag_info.new_hash
                rm_bond_t = frag_info.rm_bond_t
                new_frag = frag_info.new_frag

                temp_sibs = list(sibling_hashes - {new_frag_hash})
                max_broken = parent_broken + rm_bond_t

                old_entry = self.frag_to_entry.get(new_frag_hash)

                if old_entry is None:
                    cur_id += 1
                    score = self.score_fragment(new_frag)[1]
                    stats = self.atom_pass_stats(new_frag, depth=max_broken)

                    new_entry = Fragment(
                        frag=new_frag,
                        id=cur_id,
                        sibling_hashes=([temp_sibs[0]] if temp_sibs else []),
                        parents=[parent_entry.id],  # type: ignore
                        parent_hashes=[frag_hash],  # type: ignore
                        parent_ind_removed=[removed_atom],  # type: ignore
                        max_broken=max_broken,
                        tree_depth=parent_entry.tree_depth + 1,
                        score=float(score),
                        base_mass=float(stats["base_mass"]),
                        form=str(stats["form"]),
                        frag_hs=int(stats["frag_hs"]),
                        max_remove_hs=int(stats["max_remove_hs"]),
                        max_add_hs=int(stats["max_add_hs"]),
                    )

                    self.frag_to_entry[new_frag_hash] = new_entry
                    new_fragments.append(new_frag_hash)

                elif old_entry.max_broken == max_broken:
                    old_entry.parent_ind_removed.append(removed_atom)  # type: ignore
                    old_entry.parents.append(parent_entry.id)  # type: ignore
                    old_entry.parent_hashes.append(frag_hash)  # type: ignore
                    if temp_sibs:
                        old_entry.sibling_hashes.append(
                            temp_sibs[0]
                        )  # Keep as int

        return cur_id

    def get_root_frag(self) -> int:
        """get_root_frag."""
        return (1 << self.natoms) - 1

    def get_frag_masses(self):
        """Enhanced version that includes isotope variants."""
        frag_hashes, frag_inds, shift_inds, masses, scores = [], [], [], [], []

        for frag_hash, fragment_entry in self.frag_to_entry.items():
            base_mass = fragment_entry.base_mass
            score = fragment_entry.score
            formula_str = fragment_entry.form
            max_remove_h = fragment_entry.max_remove_hs
            max_add_h = fragment_entry.max_add_hs

            # Calculate isotope distribution if enabled
            if self.params.detect_isotope_patterns and self.isotope_calculator:
                formula_dict = self.isotope_calculator.parse_formula(
                    formula_str
                )
                isotope_dist = (
                    self.isotope_calculator.calculate_isotope_distribution(
                        formula_dict
                    )
                )
            else:
                isotope_dist = {0: 1.0}  # Only monoisotopic

            # Generate H-shift variants
            for h_shift in range(-max_remove_h, max_add_h + 1):
                h_mass_shift = h_shift * 1.007825
                h_shifted_mass = base_mass + h_mass_shift

                # For each H-shift, add isotope variants
                for iso_shift, rel_abundance in isotope_dist.items():
                    if rel_abundance >= self.params.min_isotope_intensity:
                        final_mass = h_shifted_mass + iso_shift
                        unit_mass = round(final_mass)

                        frag_hashes.append(frag_hash)
                        frag_inds.append(fragment_entry.frag)
                        shift_inds.append(h_shift)
                        masses.append(unit_mass)
                        scores.append(score)

        return (
            np.array(frag_hashes),
            np.array(frag_inds),
            np.array(shift_inds),
            np.array(masses),
            np.array(scores),
        )

    def get_frag_forms(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get fragment forms and their masses."""
        masses, form_vecs = [], []
        form_set: Set[str] = set()

        for k, v in self.frag_to_entry.items():
            max_remove, max_add = v.max_remove_hs, v.max_add_hs
            base_mass = v.base_mass
            base_form_str = v.form
            base_form_vec = formula_to_dense(base_form_str)

            for num_shift, shift_ind, shift_mass in zip(
                self.shift_buckets,
                self.shift_bucket_inds,
                self.shift_bucket_masses,
            ):
                if (num_shift >= -max_remove) and (num_shift <= max_add):
                    new_form_vec = (
                        base_form_vec
                        + num_shift * ELEMENT_VECTORS[element_to_ind["H"]]
                    )
                    str_code = str(new_form_vec)
                    if str_code in form_set:
                        continue

                    masses.append(base_mass + shift_mass)
                    form_vecs.append(new_form_vec)
                    form_set.add(str_code)

        return np.array(form_vecs), np.array(masses)

    def atom_pass_stats(
        self, frag: int, depth: Optional[int] = None
    ) -> Dict[str, Union[str, float, int]]:
        """Calculate atom-based statistics for a fragment.

        Args:
            frag: Integer representation of fragment
            depth: Maximum depth to consider for hydrogen shifts

        Returns
        -------
            Dictionary containing:
                - form: Molecular formula string
                - base_mass: Fragment mass
                - frag_hs: Number of hydrogens
                - max_remove_hs: Maximum number of removable hydrogens
                - max_add_hs: Maximum number of addable hydrogens
        """
        fragment_mass = 0.0
        form_vec = np.zeros(len(VALID_ELEMENTS))
        h_pos = element_to_ind["H"]

        # Calculate mass and element counts
        for atom in range(self.natoms):
            if frag & (1 << atom):
                fragment_mass += self.atom_weights_h[atom]
                dense_pos = element_to_ind[self.atom_symbols[atom]]
                form_vec[dense_pos] += 1
                form_vec[h_pos] += self.atom_hs[atom]

        form = vec_to_formula(form_vec)
        frag_hs = int(form_vec[h_pos])

        # Calculate hydrogen shift limits
        max_remove = int(min(frag_hs, self.params.num_h_shifts))
        max_add = int(min(self.total_hs - frag_hs, self.params.num_h_shifts))

        if depth is not None:
            max_remove = int(min(depth, max_remove))
            max_add = int(min(depth, max_add))

        return {
            "form": form,
            "base_mass": float(fragment_mass),
            "frag_hs": frag_hs,
            "max_remove_hs": max_remove,
            "max_add_hs": max_add,
        }

    def generate_fragments(self) -> None:
        cur_id = 0
        frag = self._create_root_frag()

        # Initialize root fragment
        root = self._create_root_fragment(frag, cur_id)
        frag_hash = self.wl_hash(frag)

        self.frag_to_entry[frag_hash] = root

        current_fragments: List[str] = [frag_hash]
        new_fragments: List[str] = []

        # Generate fragments for max_tree_depth steps
        for step in range(self.params.max_tree_depth):
            for frag_hash in current_fragments:
                parent_entry = self.frag_to_entry[frag_hash]

                if parent_entry.max_broken >= self.params.max_broken_bonds:
                    continue

                cur_id = self._process_fragment(
                    frag_hash,  # type: ignore
                    cur_id,
                    new_fragments,
                )

            current_fragments = new_fragments.copy()
            new_fragments = []

    def remove_atom(self, fragment: int, atom: int) -> List[FragmentInfo]:
        if not ((1 << atom) & fragment):
            return []

        template_fragment = fragment ^ (1 << atom)

        list_ext_atoms: Set[int] = set()
        ext_atom_to_bo: Dict[int, int] = {}

        # Get neighboring atoms
        for a in self.bonded_atoms[atom]:
            if (1 << a) & template_fragment:
                list_ext_atoms.add(a)
                bond_num = (1 << atom) | (1 << a)
                bond_type = self.bond_to_type[bond_num]
                ext_atom_to_bo[a] = bond_type

        # Handle single bond case
        if len(list_ext_atoms) == 1:
            if template_fragment == 0:
                return []
            bo = next(iter(ext_atom_to_bo.values()))
            new_frag_hash = self.wl_hash(template_fragment)
            return [
                FragmentInfo(
                    new_frag=template_fragment,
                    new_hash=new_frag_hash,
                    removed_atom=atom,
                    rm_bond_t=bo,
                )
            ]

        # Handle multi-bond case
        extended_fragments: List[FragmentInfo] = []
        for a in list_ext_atoms:
            is_ring = any(
                (1 << a) & frag.new_frag for frag in extended_fragments
            )
            if not is_ring:
                rm_bond_t = ext_atom_to_bo[a]
                new_fragment = self._extend_atom(a, template_fragment)

                if new_fragment == 0:
                    continue

                new_frag_hash = self.wl_hash(new_fragment)
                extended_fragments.append(
                    FragmentInfo(
                        new_frag=new_fragment,
                        new_hash=new_frag_hash,
                        removed_atom=atom,
                        rm_bond_t=rm_bond_t,
                    )
                )

        return extended_fragments

    def _extend_atom(self, atom: int, template_fragment: int) -> int:
        """Direct port of legacy extend() implementation."""
        stack = [atom]
        new_fragment = 0

        while len(stack) > 0:  # Match legacy len() check
            atom = stack.pop()  # Use atom to match legacy naming
            for a in self.bonded_atoms[atom]:
                atombit = 1 << a
                # Exact same condition order as legacy
                if (not (atombit & template_fragment)) or (
                    atombit & new_fragment
                ):
                    continue
                new_fragment = new_fragment | atombit  # Use | instead of |=
                stack.append(a)

        return new_fragment

    def export_edges(self, frag_hashes: List[int]) -> List[Tuple[int, int]]:
        """Get edges between fragments in the fragment tree.

        Args:
            frag_hashes: List of fragment hashes to consider

        Returns
        -------
            List of (parent_hash, child_hash) tuples representing edges
        """
        explored = set(frag_hashes)
        return [
            (p, i)  # type: ignore
            for i in frag_hashes
            for p in self.frag_to_entry[i].parent_hashes  # type: ignore
            if p in explored
        ]

    def export_edges_dict(
        self, frag_hashes: List[int]
    ) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
        """Get incoming and outgoing edges for each fragment."""
        incoming = defaultdict(list)
        outgoing = defaultdict(list)
        explored = set(frag_hashes)

        for i in frag_hashes:
            for p in self.frag_to_entry[i].parent_hashes:  # type: ignore
                if p in explored:
                    incoming[i].append(p)
                    outgoing[p].append(i)

        return dict(incoming), dict(outgoing)  # type: ignore

    def get_present_atoms(self, frag: int) -> Tuple[List[int], List[str]]:
        """Get atoms present in a fragment.

        Args:
            frag: Integer representation of fragment

        Returns
        -------
            Tuple of (atom_indices, atom_symbols)
        """
        indices = []
        symbols = []

        for atom in range(self.natoms):
            if (1 << atom) & frag:
                indices.append(atom)
                symbols.append(self.atom_symbols[atom])

        return indices, symbols

    def get_present_edges(
        self, frag: int
    ) -> Tuple[List[int], List[Tuple[int, int]]]:
        """Get bonds present in a fragment.

        Args:
            frag: Integer representation of fragment

        Returns
        -------
            Tuple of (bond_types, bond_indices)
        """
        bond_types = []
        bond_indices = []

        for bond, bond_inds in zip(self.bonds_list, self.bond_inds_list):
            if (frag & bond) == bond:
                bond_types.append(self.bond_to_type[bond])
                bond_indices.append(bond_inds)

        return bond_types, bond_indices

    def get_atoms_hash(self, frag_hash: int) -> Tuple[List[int], List[str]]:
        """Get atoms for a fragment specified by hash.

        Args:
            frag_hash: Hash identifying the fragment

        Returns
        -------
            Tuple of (atom_indices, atom_symbols)
        """
        frag = self.frag_to_entry[frag_hash].frag  # type: ignore
        return self.get_present_atoms(frag)

    def get_draw_dict(self, frag: int) -> DrawInfo:
        """Get information needed to draw a fragment.

        Args:
            frag: Integer representation of fragment

        Returns
        -------
            DrawInfo object with drawing information
        """
        keep_atoms, _ = self.get_present_atoms(frag)
        _, keep_bonds = self.get_present_edges(frag)
        bond_indices = [
            self.mol.GetBondBetweenAtoms(*i).GetIdx() for i in keep_bonds
        ]

        return DrawInfo(
            hatoms=keep_atoms,
            hbonds=bond_indices,
            mol=self.mol,
            smiles=self.smiles,
        )

    def frags_to_intens(
        self, frags: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Convert fragments to intensity data."""
        mass_to_obj: DefaultDict[float, Dict[str, Any]] = defaultdict(dict)

        # Collect intensities for each mass
        for frag_hash, frag_data in frags.items():
            masses = frag_data["base_mass"] + self.shift_bucket_masses
            intensities = frag_data["intens"]

            for mass, intensity in zip(masses, intensities):
                if intensity <= 0:
                    continue

                current = mass_to_obj[mass]
                if current.get("inten", 0) > 0:
                    if current["inten"] < intensity:
                        current["frag_hash"] = frag_hash
                    current["inten"] += intensity
                else:
                    current.update(
                        {"inten": intensity, "frag_hash": frag_hash}
                    )

        # Normalize intensities
        max_inten = max(
            (obj["inten"] for obj in mass_to_obj.values()), default=1e-9
        )

        # Convert to regular dict with properly typed values
        mass_to_obj_dict: Dict[float, Dict[str, Any]] = {
            mass: {
                "inten": float(data["inten"]) / max_inten,
                "frag_hash": str(data["frag_hash"]),
            }
            for mass, data in mass_to_obj.items()
        }

        return [
            {"mz": mass, **data} for mass, data in mass_to_obj_dict.items()
        ]

    def wl_hash(
        self,
        template_fragment: int,
    ) -> str:
        """Compute WL hash of a fragment using integer polynomial hashing.

        Same WL neighborhood-aggregation logic as before, but uses integer
        arithmetic instead of blake2b + string formatting.  Results are cached
        per engine instance keyed on the bitmask integer, so the same fragment
        reached via different BFS paths is only hashed once.

        Args:
            template_fragment: Int defining template fragment (bitmask over atoms)

        Return:
            Hex string uniquely identifying the fragment graph
        """
        if template_fragment in self._hash_cache:
            return self._hash_cache[template_fragment]

        cur_hashes = list(self._atom_init_hashes)

        graph_hash = _get_graph_hash_int(
            [
                cur_hashes[a]
                for a in range(self.natoms)
                if template_fragment & (1 << a)
            ]
        )
        iterations = self.natoms
        changed = True
        ct = 0

        while ct <= iterations and changed:
            new_hashes = []
            temp_atoms = 0
            for atom in range(self.natoms):
                atombit = 1 << atom
                cur_hash = cur_hashes[atom]

                if not atombit & template_fragment:
                    new_hashes.append(cur_hash)
                    continue

                temp_atoms += 1

                neighbor_labels = []
                for targind in self.bonded_atoms[atom]:
                    targbit = 1 << targind
                    if not targbit & template_fragment:
                        continue
                    bondbit = targbit | atombit
                    bondtype = self.bond_to_type[bondbit]
                    neighbor_labels.append((bondtype, cur_hashes[targind]))

                neighbor_labels.sort()
                vals = [cur_hash] + [
                    v for pair in neighbor_labels for v in pair
                ]
                new_hashes.append(_poly_hash_ints(vals))

            iterations = temp_atoms
            new_graph_hash = _get_graph_hash_int(
                [
                    new_hashes[a]
                    for a in range(self.natoms)
                    if template_fragment & (1 << a)
                ]
            )
            changed = new_graph_hash != graph_hash
            graph_hash = new_graph_hash
            cur_hash = new_hashes  # type: ignore  # matches original convergence behaviour

        result = hex(graph_hash)
        self._hash_cache[template_fragment] = result
        return result

    def score_fragment(self, fragment: int) -> Tuple[int, float]:
        """Score a fragment based on broken bonds.

        Args:
            fragment: Integer representation of fragment

        Returns
        -------
            Tuple of (number of breaks, score)
        """
        score, breaks = 0.0, 0
        for bond in self.bonds:
            if 0 < (fragment & bond) < bond:
                score += self.bondscore[bond]
                breaks += 1
        return breaks, score

    def single_mass(self, frag: int) -> float:
        """Calculate mass of fragment."""
        fragment_mass = 0.0
        for atom in range(self.natoms):
            if frag & (1 << atom):
                fragment_mass += self.atom_weights_h[atom]
        return fragment_mass

    def formula_from_frag(self, frag: int, h_shift: int = 0) -> str:
        """Generate molecular formula for fragment."""
        form_vec = np.zeros(len(VALID_ELEMENTS))
        h_pos = element_to_ind["H"]
        for atom in range(self.natoms):
            if frag & (1 << atom):
                dense_pos = element_to_ind[self.atom_symbols[atom]]
                form_vec[dense_pos] += 1
                form_vec[h_pos] += self.atom_hs[atom]

        form_vec[h_pos] += h_shift
        return vec_to_formula(form_vec)

    def formula_from_kept_inds(self, kept_inds: np.ndarray) -> str:
        """Generate molecular formula from kept atom indices."""
        form_vec = np.zeros(len(VALID_ELEMENTS))
        h_count = self.atom_hs[kept_inds].sum()
        h_pos = element_to_ind["H"]
        form_vec[h_pos] = h_count
        atom_cts = Counter(self.atom_symbols_ar[kept_inds])
        for atom_type, atom_ct in atom_cts.items():
            form_vec[element_to_ind[atom_type]] = atom_ct
        return vec_to_formula(form_vec)


_WL_PRIME = 1000003  # small prime with good avalanche for polynomial hashing


_WL_MASK = 0xFFFFFFFFFFFFFFFF  # 64-bit mask


def _poly_hash_ints(vals) -> int:
    """Deterministic polynomial hash over a sequence of integers."""
    h = 0
    for v in vals:
        h = (h * _WL_PRIME ^ v) & _WL_MASK
    return h


def _get_graph_hash_int(atom_hashes) -> int:
    """Order-invariant (multiset) hash of integer atom hashes."""
    counter = Counter(atom_hashes)
    h = 0
    for val, cnt in sorted(counter.items()):
        h = ((h * _WL_PRIME ^ val) * _WL_PRIME ^ cnt) & _WL_MASK
    return h


def _hash_label(label, digest_size=32):
    """Create hash from label string."""
    return blake2b(label.encode("ascii"), digest_size=digest_size).hexdigest()


def extend(atom: int, bonded_atoms: list, template_fragment: int) -> int:
    """DFS extension of atom through template fragment.

    Args:
        atom: Starting atom position
        bonded_atoms: List mapping atoms to bonded atoms
        template_fragment: Binary template of fragment

    Returns
    -------
        int: New fragment
    """
    stack = [atom]
    new_fragment = 0
    while len(stack) > 0:
        atom = stack.pop()
        for a in bonded_atoms[atom]:
            atombit = 1 << a
            if (not (atombit & template_fragment)) or (atombit & new_fragment):
                continue
            new_fragment = new_fragment | atombit
            stack.append(a)
    return new_fragment


def bit_array(num: int) -> np.ndarray:
    """Convert a positive integer into a bit vector.

    Args:
        num: Positive integer to convert

    Returns
    -------
        Numpy array of bits
    """
    return np.array(list(f"{num:b}")).astype(float)


def create_new_ids(
    frags: Dict[str, Fragment],
) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Create mappings between fragment hashes and sequential IDs.

    Args:
        frags: Dictionary of fragments

    Returns
    -------
        Tuple of (hash->id, id->hash) mappings
    """
    frag_to_id = {
        i: id
        for id, i in enumerate(
            sorted(frags, key=lambda x: frags[x]["tree_depth"])
        )
    }
    id_to_frag = {id: i for i, id in frag_to_id.items()}
    return frag_to_id, id_to_frag
