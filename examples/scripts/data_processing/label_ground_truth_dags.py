"""MAGMA Mass Spectrometry Analysis.

This module computes directed acyclic graphs (DAGs) for each molecule in the
dataset and assigns subformulae to peaks.
"""

import argparse
import json
import h5py
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple, Union, cast

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from tqdm import tqdm
import multiprocessing as mp

try:
    mp.set_start_method("spawn")
except RuntimeError:
    pass

from icicle.data.fragmentation_engine import (
    FragmentationParams,
    FragmentEngine,
)
from icicle.data.isotope_distribution import IsotopePatternCalculator
from icicle.utils import (
    HDF5Dataset,
    chunked_parallel,
    filter_spectra_by_intensity,
    parse_spectra,
    process_common_spec_file,
)

from rdkit import Chem, RDLogger

# SUPPRESS ALL RDKit warnings and info messages
RDLogger.DisableLog("rdApp.*")  # Disable all RDKit logging


# Type aliases
SpectrumDict = Dict[str, Union[str, float]]
MetaDict = Mapping[str, Union[str, float]]
ResultDict = Dict[str, Any]


@dataclass
class ProcessingConfig:
    """Configuration for MAGMA processing."""

    spectra_dir: Path
    output_dir: Path
    spec_labels: Path
    max_peaks: int
    num_h_shifts: int
    max_tree_depth: int
    max_broken_bonds: int
    workers: int
    debug: bool = False
    batch_size: int = 100
    detect_isotope_patterns: bool = False
    # Optimization settings
    enable_caching: bool = True
    cache_size: int = 1000
    optimize_memory: bool = True
    cuda_memory_fraction: float = 0.8

    # Error handling
    log_errors: bool = True
    continue_on_error: bool = True


class SpectrumProcessor:
    """Unified processor for MAGMA mass spectrometry analysis.

    This class combines all functionality from legacy implementations into a
    single, optimized, and maintainable processor.
    """

    def __init__(self, config: ProcessingConfig):
        """Initialize the MAGMA processor.

        Args:
            config: Processing configuration object
        """

        self.config = config
        self._setup_logging()
        self._setup_error_handling()
        self._setup_memory_management()
        self._setup_caching()

    def _setup_logging(self) -> None:
        """Configure logging system."""
        logging.basicConfig(
            level=logging.DEBUG if self.config.debug else logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)],
        )
        self.logger = logging.getLogger(__name__)

        # Suppress RDKit warnings
        RDLogger.DisableLog("rdApp.*")

    def _setup_error_handling(self) -> None:
        """Setup error logging infrastructure."""
        if self.config.log_errors:
            self.error_file = self.config.output_dir / "magma_errors.log"
            self.config.output_dir.mkdir(exist_ok=True, parents=True)

            if not self.error_file.exists():
                with open(self.error_file, "w") as f:
                    f.write("timestamp\tspec_name\tsmiles\terror_message\n")
        else:
            self.error_file = None

    def _setup_memory_management(self) -> None:
        """Configure memory management and optimization settings."""
        if not self.config.optimize_memory:
            return

        # CUDA optimization
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.set_per_process_memory_fraction(
                self.config.cuda_memory_fraction
            )
            torch.backends.cudnn.benchmark = True
            self.logger.info("CUDA memory optimization enabled")

        # NumPy optimization
        np.set_printoptions(threshold=10000)

    def _setup_caching(self) -> None:
        """Setup caching mechanisms."""
        if self.config.enable_caching:
            # Initialize manual cache
            self._cache = {}
            self.logger.info(
                f"Caching enabled with size {self.config.cache_size}"
            )

    def clear_cache(self) -> None:
        """Clear processing cache."""
        if self.config.enable_caching and hasattr(self, "_cache"):
            self._cache.clear()
            self.logger.info("Processing cache cleared")

    # CORE PROCESSING METHODS

    def process_all_spectra(self) -> None:
        """Main entry point for processing all spectra."""
        self.logger.info("Starting MAGMA spectrum processing")

        # Setup output directory
        self.config.output_dir.mkdir(exist_ok=True, parents=True)

        # Load metadata
        params = self._load_spectrum_parameters()

        # Process spectra
        results = self._execute_processing(params)

        # Save results
        self._save_results(results)

        self.logger.info("MAGMA processing completed successfully")

    def _load_spectrum_parameters(self) -> List[Dict[str, Any]]:
        """Load and prepare spectrum processing parameters."""
        self.logger.info("Loading spectrum metadata")

        # Read metadata files
        df = pd.read_csv(self.config.spec_labels, sep="\t")
        # Use mol_id directly as index for faster lookup
        df.set_index("inchikey", inplace=True)

        params = []
        try:
            # Correctly load the HDF5 file and iterate over the top-level keys
            with h5py.File(self.config.spectra_dir, "r") as spec_h5:
                for spectrum_key in tqdm(
                    spec_h5.keys(), desc="Loading spectra"
                ):
                    required_keys = [
                        "inchi_key",
                        "masses",
                        "intensities",
                        "standardized_smiles",
                    ]
                    if not all(
                        key in spec_h5[spectrum_key] for key in required_keys
                    ):
                        self.logger.warning(
                            f"Spectrum {spectrum_key} is missing required keys"
                        )
                        continue

                    inchikey = spec_h5[f"{spectrum_key}/inchi_key"][()].decode(
                        "utf-8"
                    )
                    # Ensure the mol_id from HDF5 exists in the metadata
                    if inchikey in df.index:
                        masses = spec_h5[f"{spectrum_key}/masses"][:]
                        intensities = spec_h5[f"{spectrum_key}/intensities"][:]
                        smiles = spec_h5[
                            f"{spectrum_key}/standardized_smiles"
                        ][()].decode("utf-8")
                        spec_data = np.vstack([masses, intensities]).transpose(
                            1, 0
                        )
                        mol_id_from_metadata = df.loc[inchikey, "mol_id"]
                        params.append(
                            {
                                "mol_id": mol_id_from_metadata,
                                "inchikey": inchikey,
                                "smiles": smiles,
                                "spec_data": spec_data,
                            }
                        )

        except Exception as e:
            self.logger.error(f"Error loading spectrum parameters: {e}")
            raise e

        self.logger.info(f"Loaded {len(params)} spectra for processing")
        return params

    def _execute_processing(
        self, params: List[Dict[str, Any]]
    ) -> List[Tuple[str, Dict[str, str], Dict[str, str]]]:
        """Execute spectrum processing with parallelization."""

        self.logger.info(
            f"Processing {len(params)} spectra with {self.config.workers} workers"
        )
        results = chunked_parallel(
            params,
            self.process_single_spectrum,
            max_cpu=self.config.workers,
            chunks=self.config.batch_size,
        )

        valid_results = [r for r in results if r is not None]

        self.logger.info(
            f"Successfully processed {len(valid_results)} spectra"
        )

        results = valid_results

        # Print the first result
        self.logger.info(f"Results for tsv")
        key = list(results[0][0].keys())[0]
        tsv_data = results[0][0][key]
        self.logger.info(f"TSV data: {tsv_data}")

        # Get just the tree part
        self.logger.info(f"Results for tree")
        key = list(results[0][1].keys())[0]
        tree_data = results[0][1][key]
        tree_content = json.loads(tree_data)  # Parse the JSON

        self.logger.info(f"Tree data: {tree_content.keys()}")

        self.logger.info(f"Root molecule: {tree_content['root_inchi']}")
        self.logger.info(f"Number of fragments: {len(tree_content['frags'])}")

        # Show observed fragments (ones that appear in your spectrum)
        observed_frags = [
            frag
            for frag in tree_content["frags"].values()
            if frag["is_observed"]
        ]
        self.logger.info(f"Observed fragments: {len(observed_frags)}")

        # Show a specific fragment
        first_frag = list(tree_content["frags"].values())[0]
        self.logger.info(f"Example fragment mass: {first_frag['base_mass']}")
        return results

    def process_single_spectrum(
        self, param: Dict[str, Any]
    ) -> Optional[Tuple[Dict[str, str], Dict[str, str]]]:
        """Process a single spectrum with comprehensive error handling."""

        mol_id = param["mol_id"]
        smiles = param["smiles"]

        try:
            # Validate inputs
            if not self._validate_inputs(param):
                return None

            # Generate molecular fragments
            fragment_engine = self._create_fragment_engine(smiles)
            if fragment_engine is None:
                return None

            # Analyze spectrum against fragments
            tsv_dict, tree_dict = self._analyze_spectrum(
                spectrum=param["spec_data"],
                fragment_engine=fragment_engine,
                mol_id=mol_id,
            )

            return tsv_dict, tree_dict

        except Exception as e:
            error_msg = f"Processing error: {str(e)}"
            self.logger.error(f"Error processing {mol_id}: {error_msg}")
            self._log_error(mol_id, smiles, error_msg)

            if not self.config.continue_on_error:
                raise
            return None

    def calculate_pattern_confidence(
        self,
        fragment_formula: str,
        base_mass: int,
        spectrum_lookup: Dict[int, float],
    ) -> float:
        """Calculate confidence that observed peaks match expected isotope
        pattern."""
        if not self.isotope_calculator:
            return 1.0

        formula_dict = self.isotope_calculator.parse_formula(fragment_formula)
        expected_dist = self.isotope_calculator.calculate_isotope_distribution(
            formula_dict
        )

        base_intensity = spectrum_lookup.get(base_mass, 0)
        if base_intensity <= 0:
            return 0.0

        confidence = 0.0
        num_comparisons = 0

        for iso_shift, expected_rel in expected_dist.items():
            peak_mass = base_mass + iso_shift
            observed_intensity = spectrum_lookup.get(peak_mass, 0)

            if expected_rel >= 0.05:  # Check significant peaks
                if observed_intensity > 0:
                    observed_rel = observed_intensity / base_intensity
                    ratio_error = (
                        abs(observed_rel - expected_rel) / expected_rel
                    )
                    match_score = max(0, 1 - ratio_error)
                    confidence += match_score
                else:
                    confidence -= 0.5  # Penalty for missing peak

                num_comparisons += 1

        return (
            max(0.0, confidence / num_comparisons)
            if num_comparisons > 0
            else 1.0
        )

    def _validate_inputs(self, param: Dict[str, Any]) -> bool:
        """Validate spectrum processing inputs."""
        mol_id = param["mol_id"]
        smiles = param["smiles"]

        if not smiles or not isinstance(smiles, str):
            self._log_error(mol_id, smiles, "Missing or invalid SMILES")
            return False

        if not self._validate_smiles(smiles):
            self._log_error(mol_id, smiles, "Invalid SMILES structure")
            return False

        return True

    @staticmethod
    def _validate_smiles(smiles: str) -> bool:
        """Validate SMILES string structure."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None
        except Exception:
            return False

    def _create_fragment_engine(self, smiles: str) -> Optional[FragmentEngine]:
        """Create and initialize fragment engine."""
        try:
            fragment_engine = FragmentEngine(
                mol_str=smiles,
                params=FragmentationParams(
                    max_tree_depth=self.config.max_tree_depth,
                    max_broken_bonds=self.config.max_broken_bonds,
                    num_h_shifts=self.config.num_h_shifts,
                    detect_isotope_patterns=self.config.detect_isotope_patterns,
                    min_isotope_intensity=0.01,
                ),
            )
            fragment_engine.generate_fragments()
            return fragment_engine

        except Exception as e:
            self.logger.error(
                f"Fragment generation failed for {smiles}: {str(e)}"
            )
            return None

    # SPECTRUM ANALYSIS METHODS

    def _analyze_spectrum(
        self,
        spectrum: np.ndarray,
        fragment_engine: FragmentEngine,
        mol_id: str,
    ) -> Tuple[Dict[str, str], Dict[str, str]]:
        """Analyze a single spectrum against molecular fragments."""

        # Scale spectrum so that the highest intensity is 1
        spectrum[:, 1] = spectrum[:, 1] / np.max(spectrum[:, 1])

        # Prepare spectrum
        spectrum = filter_spectra_by_intensity(
            spectrum,
            max_num_inten=self.config.max_peaks,
            inten_thresh=0.01,
        )

        # Get spectrum masses
        spectrum_masses = spectrum[:, 0]

        # Get fragment information
        frag_hashes, frag_inds, shift_inds, masses, scores = (
            fragment_engine.get_frag_masses()
        )

        # Sort by bond breaking scores (lower is better)
        sort_idx = np.argsort(scores)
        frag_hashes = frag_hashes[sort_idx]
        frag_inds = frag_inds[sort_idx]
        shift_inds = shift_inds[sort_idx]
        masses = masses[sort_idx]
        scores = scores[sort_idx]

        # Perform mass comparison
        peak_mask, min_diffs = self._compare_masses(masses, spectrum_masses)

        # Generate outputs
        spec_data = {
            "mol_id": str(mol_id),
        }

        return self._generate_outputs(
            spec_data=spec_data,
            fragment_engine=fragment_engine,
            spectrum=spectrum,
            spectrum_masses=spectrum_masses,
            frag_hashes=frag_hashes,
            frag_inds=frag_inds,
            shift_inds=shift_inds,
            masses=masses,
            scores=scores,
            peak_mask=peak_mask,
            min_diffs=min_diffs,
        )

    def _compare_masses(
        self, masses: np.ndarray, adjusted_masses: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compare fragment masses with observed masses."""

        if self.config.enable_caching:
            # Use cached comparison for better performance
            cache_key = (
                f"{hash(masses.tobytes())}_{hash(adjusted_masses.tobytes())}"
            )
            return self._process_spectrum_cached(
                cache_key, masses, adjusted_masses
            )
        else:
            return self._unit_mass_comparison(masses, adjusted_masses)

    def _process_spectrum_cached(
        self,
        cache_key: str,
        masses: np.ndarray,
        adjusted_masses: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Cached mass comparison for improved performance."""
        # Create a proper cache key from the arrays
        masses_key = hash(masses.tobytes())
        adjusted_masses_key = hash(adjusted_masses.tobytes())
        full_cache_key = f"{cache_key}_{masses_key}_{adjusted_masses_key}"

        # Check if we have a cached result
        if hasattr(self, "_cache") and full_cache_key in self._cache:
            return self._cache[full_cache_key]

        # Compute result
        result = self._unit_mass_comparison(masses, adjusted_masses)

        # Cache the result
        if not hasattr(self, "_cache"):
            self._cache = {}

        # Limit cache size
        if len(self._cache) >= self.config.cache_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]

        self._cache[full_cache_key] = result
        return result

    def _unit_mass_comparison(
        self, masses: np.ndarray, adjusted_masses: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Perform unit mass-based comparison."""
        rounded_masses = np.round(masses)
        rounded_observed = np.round(adjusted_masses)

        mass_diffs = np.abs(
            rounded_masses[None, :] - rounded_observed[:, None]
        )
        min_diffs = mass_diffs.min(axis=1)
        peak_mask = min_diffs < 1

        return peak_mask, min_diffs

    # OUTPUT GENERATION METHODS

    def _generate_outputs(
        self,
        spec_data: Dict[str, str],
        fragment_engine: FragmentEngine,
        spectrum: np.ndarray,
        spectrum_masses: np.ndarray,
        frag_hashes: np.ndarray,
        frag_inds: np.ndarray,
        shift_inds: np.ndarray,
        masses: np.ndarray,
        scores: np.ndarray,
        peak_mask: np.ndarray,
        min_diffs: np.ndarray,
    ) -> Tuple[Dict[str, str], Dict[str, str]]:
        """Generate TSV and tree outputs for spectrum analysis."""

        # Generate TSV output
        tsv_output = self._generate_tsv_output(
            spec_data,
            spectrum,
            spectrum_masses,
            frag_hashes,
            frag_inds,
            shift_inds,
            masses,
            scores,
            peak_mask,
            min_diffs,
            fragment_engine,
        )

        # Generate tree output
        tree_output = self._generate_tree_output(
            spec_data, fragment_engine, tsv_output
        )

        return tsv_output, tree_output

    def _generate_tsv_output(
        self,
        spec_data: Dict[str, str],
        spectrum: np.ndarray,
        spectrum_masses: np.ndarray,
        frag_hashes: np.ndarray,
        frag_inds: np.ndarray,
        shift_inds: np.ndarray,
        masses: np.ndarray,
        scores: np.ndarray,
        peak_mask: np.ndarray,
        min_diffs: np.ndarray,
        fragment_engine: FragmentEngine,
    ) -> Dict[str, str]:
        """Generate TSV format output."""

        tsv_export_list = []
        spec_masses, spec_intensities = spectrum[:, 0], spectrum[:, 1]

        for ind, was_assigned in enumerate(peak_mask):
            new_entry = {
                "mz_observed": spec_masses[ind],
                "mz_corrected": spectrum_masses[ind],
                "inten": spec_intensities[ind],
                "difference": "",
                "frag_inds": "",
                "frag_mass": "",
                "frag_h_shift": "",
                "frag_base_form": "",
                "frag_hashes": "",
            }

            if was_assigned:
                # Find matching fragments
                rounded_mass = np.round(spec_masses[ind])
                rounded_masses = np.round(masses)
                matched_peaks = np.abs(rounded_masses - rounded_mass) < 1
                min_inds = np.where(matched_peaks)[0]

                # Select best scoring fragments
                if len(min_inds) > 0:
                    # Simple scoring: just use bond breaking scores
                    min_score = np.min(scores[min_inds])
                    best_inds = min_inds[scores[min_inds] == min_score]
                    best_inds = best_inds[:5]  # Limit to top 5

                # Extract fragment information
                frag_info = self._extract_fragment_info(
                    best_inds,
                    frag_hashes,
                    frag_inds,
                    shift_inds,
                    masses,
                    fragment_engine,
                )

                # Update entry
                new_entry.update(
                    {
                        "difference": min_diffs[ind],
                        "frag_inds": ",".join(
                            map(str, frag_info["frag_inds"])
                        ),
                        "frag_hashes": ",".join(frag_info["frag_hashes"]),
                        "frag_mass": ",".join(
                            map(str, frag_info["frag_masses"])
                        ),
                        "frag_h_shift": ",".join(
                            map(str, frag_info["shift_inds"])
                        ),
                        "frag_base_form": ",".join(frag_info["frag_forms"]),
                    }
                )

            tsv_export_list.append(new_entry)

        # Create DataFrame and format output
        df = pd.DataFrame(tsv_export_list)
        df.sort_values(by="mz_observed", inplace=True)

        tsv_filename = spec_data["mol_id"]

        return {tsv_filename: df.to_csv(sep="\t", index=False)}

    def _extract_fragment_info(
        self,
        indices: np.ndarray,
        frag_hashes: np.ndarray,
        frag_inds: np.ndarray,
        shift_inds: np.ndarray,
        masses: np.ndarray,
        fragment_engine: FragmentEngine,
    ) -> Dict[str, List]:
        """Extract fragment information for given indices."""

        return {
            "frag_inds": [int(frag_inds[i]) for i in indices],
            "frag_masses": [masses[i] for i in indices],
            "frag_hashes": [frag_hashes[i] for i in indices],
            "shift_inds": [shift_inds[i] for i in indices],
            "frag_forms": [
                fragment_engine.frag_to_entry[frag_hashes[i]].form
                for i in indices
            ],
        }

    def _generate_tree_output(
        self,
        spec_data: Dict[str, str],
        fragment_engine: FragmentEngine,
        tsv_output: Dict[str, str],
    ) -> Dict[str, str]:
        """Generate tree format output."""

        # Extract observed fragments from TSV data
        tree_nodes = self._extract_tree_nodes(tsv_output)
        if not tree_nodes:
            return {}

        # Build and prune fragment tree
        pruned_nodes = self._build_and_prune_tree(fragment_engine, tree_nodes)
        if not pruned_nodes:
            return {}

        # Create tree structure
        tree_data = self._create_tree_structure(
            fragment_engine, pruned_nodes, tree_nodes, spec_data
        )

        tree_filename = spec_data["mol_id"]

        return {tree_filename: json.dumps(tree_data, indent=2)}

    def _extract_tree_nodes(self, tsv_output: Dict[str, str]) -> List[str]:
        """Extract tree nodes from TSV output."""
        tree_nodes = []

        for tsv_content in tsv_output.values():
            # Parse TSV content to extract fragment hashes
            lines = tsv_content.strip().split("\n")[1:]  # Skip header
            for line in lines:
                fields = line.split("\t")
                if len(fields) > 8:  # Ensure we have frag_hashes field
                    frag_hashes = fields[8]  # frag_hashes column
                    if frag_hashes:
                        tree_nodes.extend(
                            hash.strip()
                            for hash in frag_hashes.split(",")
                            if hash.strip()
                        )

        return list(set(tree_nodes))

    def _build_and_prune_tree(
        self, fragment_engine: FragmentEngine, tree_nodes: List[str]
    ) -> List[str]:
        """Build complete tree and prune using greedy algorithm."""

        # Get all fragments
        all_fragments = list(fragment_engine.frag_to_entry.keys())

        # Explore backwards from tree nodes to find all ancestors
        explore_queue = tree_nodes.copy()
        explored = set(tree_nodes)

        while explore_queue:
            current_node = explore_queue.pop()
            if current_node in fragment_engine.frag_to_entry:
                entry = fragment_engine.frag_to_entry[current_node]
                for parent in entry.parent_hashes:
                    if parent not in explored:
                        explored.add(parent)
                        explore_queue.append(parent)

        # Prune the tree
        included_nodes = sorted(explored)
        return self._greedy_prune(fragment_engine, included_nodes, tree_nodes)

    def _greedy_prune(
        self,
        fragment_engine: FragmentEngine,
        included_nodes: List[str],
        tree_nodes: List[str],
    ) -> List[str]:
        """Prune fragmentation tree using greedy set cover algorithm."""

        tree_set = set(tree_nodes)
        if not tree_set:
            return []

        # Initialize data structures
        included_nodes = sorted(included_nodes)
        hash_to_pos = dict(zip(included_nodes, range(len(included_nodes))))

        node_priorities = np.zeros(len(included_nodes))
        output_mask = np.array([node in tree_set for node in included_nodes])
        node_priorities += output_mask

        # Get graph structure
        incoming_edges, outgoing_edges = fragment_engine.export_edges_dict(
            included_nodes
        )

        # Get scoring information
        entries = [
            fragment_engine.frag_to_entry[node] for node in included_nodes
        ]
        node_scores = [entry.score for entry in entries]
        tree_depths = np.array([entry.tree_depth for entry in entries])

        # Process each depth level
        highest_depth = max(tree_depths) if len(tree_depths) > 0 else 0

        for depth in range(highest_depth, 0, -1):
            cur_layer_inds = tree_depths == depth
            cover_options = np.zeros(len(included_nodes), dtype=bool)
            to_cover = np.logical_and(output_mask, cur_layer_inds)

            # Update parent priorities
            for node_idx in np.where(to_cover)[0]:
                hash_key = included_nodes[node_idx]
                for parent in incoming_edges.get(hash_key, []):
                    if parent in hash_to_pos:
                        parent_pos = hash_to_pos[parent]
                        node_priorities[parent_pos] += 1
                        cover_options[parent_pos] = True

            # Greedy selection
            num_to_cover = np.sum(to_cover)
            while num_to_cover > 0:
                if not np.any(cover_options):
                    break

                max_score = np.max(node_priorities[cover_options])
                best_mask = (node_priorities == max_score) & cover_options
                best_candidates = np.where(best_mask)[0]

                # Break ties using node scores
                best_node = min(best_candidates, key=lambda x: node_scores[x])
                output_mask[best_node] = True

                # Update coverage
                node_hash = included_nodes[best_node]
                for child in outgoing_edges.get(node_hash, []):
                    if child in hash_to_pos:
                        child_pos = hash_to_pos[child]
                        to_cover[child_pos] = False

                        # Update parent priorities
                        for parent in incoming_edges.get(child, []):
                            if parent in hash_to_pos:
                                parent_pos = hash_to_pos[parent]
                                if not output_mask[parent_pos]:
                                    node_priorities[parent_pos] -= 1

                cover_options[best_node] = False
                num_to_cover = np.sum(to_cover)

        return [included_nodes[i] for i in np.where(output_mask)[0]]

    def _create_tree_structure(
        self,
        fragment_engine: FragmentEngine,
        pruned_nodes: List[str],
        tree_nodes: List[str],
        spec_data: Dict[str, str],
    ) -> Dict[str, Any]:
        """Create the final tree data structure with correct atoms_pulled."""

        out_frags = {}
        pruned_node_set = set(pruned_nodes)
        tree_node_set = set(tree_nodes)

        # Use the same approach as the older version
        node_to_pulled: Dict[str, Set[int]] = defaultdict(set)
        node_to_parents: Dict[str, Set[str]] = defaultdict(set)

        # First pass: Collect relationship information from FragmentEngine
        for frag_hash in pruned_nodes:
            entry = fragment_engine.frag_to_entry[frag_hash]

            # Extract atoms_pulled from FragmentEngine's parent_ind_removed
            for parent_hash, pulled_atom in zip(
                entry.parent_hashes, entry.parent_ind_removed
            ):
                if parent_hash in pruned_node_set:
                    node_to_parents[frag_hash].add(parent_hash)
                    node_to_pulled[parent_hash].add(pulled_atom)

        # Second pass: Create fragment entries with correct atoms_pulled
        for frag_hash in pruned_nodes:
            entry = fragment_engine.frag_to_entry[frag_hash]

            # Create fragment entry with REAL atoms_pulled data
            fragment_entry = {
                "frag_hash": frag_hash,
                "frag": entry.frag,
                "is_observed": frag_hash in tree_node_set,
                "atoms_pulled": list(
                    node_to_pulled[frag_hash]
                ),  # ← REAL DATA!
                "parents": list(node_to_parents[frag_hash]),  # ← REAL PARENTS!
                "base_mass": float(entry.base_mass),
                "intens": np.zeros(
                    len(fragment_engine.shift_bucket_inds)
                ).tolist(),
                "id": entry.id,
                "sib": False,
                "max_broken": entry.max_broken,
                "tree_depth": entry.tree_depth,
                "max_remove_hs": entry.max_remove_hs,
                "max_add_hs": entry.max_add_hs,
            }

            out_frags[frag_hash] = fragment_entry

        # Create tree data
        return {
            "root_inchi": fragment_engine.inchi,
            "frags": out_frags,
        }

    def _log_error(self, spec_name: str, smiles: str, error_msg: str) -> None:
        """Log processing errors to file."""
        if not self.error_file:
            return

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            with open(self.error_file, "a") as f:
                f.write(f"{timestamp}\t{spec_name}\t{smiles}\t{error_msg}\n")
        except Exception as e:
            self.logger.error(f"Failed to log error: {e}")

    def _save_results(
        self,
        results: List[Tuple[Dict[str, str], Dict[str, str]]],
    ) -> None:
        """Save processing results to HDF5 files."""
        self.logger.info("Saving results to HDF5 files")

        try:
            tsv_h5 = HDF5Dataset(
                self.config.output_dir / "magma_tsv.hdf5", mode="w"
            )
            tree_h5 = HDF5Dataset(
                self.config.output_dir / "magma_tree.hdf5", mode="w"
            )

            for tsv_dict, tree_dict in tqdm(results, desc="Saving results"):
                if tsv_dict:
                    tsv_h5.write_dict(tsv_dict)
                if tree_dict:
                    tree_h5.write_dict(tree_dict)

        finally:
            tsv_h5.close()
            tree_h5.close()

        self.logger.info("Results saved successfully")

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics and cache information."""
        stats = {
            "config": self.config,
            "cache_enabled": self.config.enable_caching,
        }

        if self.config.enable_caching and hasattr(self, "_cache"):
            stats["cache_stats"] = {
                "cache_size": len(self._cache),
                "max_size": self.config.cache_size,
            }

        return stats

    def clear_cache(self) -> None:
        """Clear processing cache."""
        if self.config.enable_caching:
            self._process_spectrum_cached.cache_clear()
            self.logger.info("Processing cache cleared")


def main():
    """Main entry point for the unified MAGMA processor."""
    parser = argparse.ArgumentParser(
        description="Label ground truth DAGs with MAGMa"
    )

    # Required arguments
    parser.add_argument(
        "--data-dir", required=True, help="Directory with spectra files"
    )

    # Processing parameters
    parser.add_argument(
        "--max-peaks",
        type=int,
        default=50,
        help="Maximum number of peaks to process",
    )
    parser.add_argument(
        "--num-h-shifts",
        type=int,
        default=1,
        help="Maximum number of H shifts",
    )
    parser.add_argument(
        "--detect-isotope-patterns",
        action="store_true",
        help="Detect isotope patterns in the spectrum",
    )
    parser.add_argument(
        "--max-tree-depth", type=int, default=3, help="Maximum tree depth"
    )
    parser.add_argument(
        "--max-broken-bonds",
        type=int,
        default=6,
        help="Maximum number of broken bonds",
    )
    parser.add_argument(
        "--workers", type=int, default=16, help="Number of parallel workers"
    )

    # Optimization options
    parser.add_argument(
        "--disable-caching",
        action="store_true",
        help="Disable caching for memory-constrained environments",
    )
    parser.add_argument(
        "--cache-size",
        type=int,
        default=1000,
        help="Maximum cache size for spectrum processing",
    )
    parser.add_argument(
        "--disable-memory-optimization",
        action="store_true",
        help="Disable memory optimization features",
    )

    # Debugging and error handling
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode with limited data",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for parallel processing",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    spectra_dir = data_dir / "spectra.hdf5"
    spec_labels = data_dir / "metadata.tsv"
    output_dir = (
        data_dir
        / f"processed_{args.num_h_shifts}_h_shifts_{args.max_peaks}_peaks_{args.max_tree_depth}_tree_depth_{args.max_broken_bonds}_broken_bonds_{args.detect_isotope_patterns}_isotope_patterns"
    )
    output_dir.mkdir(exist_ok=True, parents=True)
    # add a config file to the output directory
    config_file = output_dir / "config.yaml"
    import yaml

    with open(config_file, "w") as f:
        yaml.dump(args, f)

    # Create configuration
    config = ProcessingConfig(
        spectra_dir=spectra_dir,
        output_dir=output_dir,
        spec_labels=spec_labels,
        max_peaks=args.max_peaks,
        num_h_shifts=args.num_h_shifts,
        max_tree_depth=args.max_tree_depth,
        max_broken_bonds=args.max_broken_bonds,
        workers=args.workers,
        debug=args.debug,
        batch_size=args.batch_size,
        enable_caching=not args.disable_caching,
        cache_size=args.cache_size,
        optimize_memory=not args.disable_memory_optimization,
        detect_isotope_patterns=args.detect_isotope_patterns,
    )

    # Initialize and run processor
    processor = SpectrumProcessor(config)
    processor.process_all_spectra()

    # Print final statistics
    stats = processor.get_processing_stats()
    logging.info("\nProcessing Statistics:")
    logging.info(json.dumps(stats, indent=2, default=str))


if __name__ == "__main__":
    main()
