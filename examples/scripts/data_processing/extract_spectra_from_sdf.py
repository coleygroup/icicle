"""Extract spectra from SDF file and save as HDF5 file."""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
import h5py
import re
from pathlib import Path
from tqdm import tqdm
import logging
import warnings
import argparse

# Suppress ALL RDKit warnings and errors for clean output
warnings.filterwarnings("ignore", category=UserWarning)
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")  # Disable all RDKit logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SDFToMLDataset:
    """Convert SDF files with mass spectra to ML-ready datasets."""

    def __init__(self, sdf_path, output_dir="./processed_data"):
        self.sdf_path = Path(sdf_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Initialize storage
        self.metadata_records = []
        self.spectra_data = {}

    def parse_spectrum_peaks(self, peaks_text):
        """
        Parse mass spectral peaks from text format
        Returns: (masses, intensities) as numpy arrays
        """
        if not peaks_text or peaks_text.strip() == "":
            return None, None

        lines = peaks_text.strip().split("\n")
        masses, intensities = [], []

        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    mass = float(parts[0])
                    intensity = float(parts[1])
                    masses.append(mass)
                    intensities.append(intensity)
                except ValueError:
                    continue

        return np.array(masses), np.array(intensities)

    def standardize_molecule(self, mol):
        """
        Standardize molecule using RDKit
        Returns: standardized SMILES
        """
        if mol is None:
            return None

        try:
            # Basic standardization
            mol = Chem.RemoveHs(mol)

            # Get canonical SMILES
            smiles = Chem.MolToSmiles(mol, canonical=True)
            return smiles
        except:
            return None

    def extract_metadata(self, mol):
        """Extract all metadata from molecule object."""
        metadata = {}

        # Extract all properties
        prop_names = mol.GetPropNames()
        for prop in prop_names:
            try:
                value = mol.GetProp(prop)
                # Clean up property names (remove spaces, special chars)
                clean_prop = re.sub(r"[^\w]", "_", prop.lower())
                metadata[clean_prop] = value
            except:
                continue

        return metadata

    def process_sdf(self):
        """Extract spectra from SDF file and save as HDF5 file."""
        logger.info(f"Processing SDF file: {self.sdf_path}")

        supplier = Chem.SDMolSupplier(str(self.sdf_path))

        valid_count = 0
        skipped_count = 0

        # Ensure mol_id is handled correctly and consistently
        for i, mol in enumerate(tqdm(supplier, desc="Processing molecules")):
            if mol is None:
                skipped_count += 1
                continue

            # 1. Determine the canonical mol_id for this entry
            # This is the single source of truth for the ID
            mol_id = None
            if mol.HasProp("ID"):
                mol_id = mol.GetProp("ID")
            elif mol.HasProp("NISTNO"):
                mol_id = mol.GetProp("NISTNO")
            else:
                logger.warning(
                    f"Skipping molecule {i} as no ID or NISTNO was found."
                )
                skipped_count += 1
                continue

            try:
                # 2. Process molecule and its properties
                std_smiles = self.standardize_molecule(mol)
                if std_smiles is None:
                    skipped_count += 1
                    continue

                inchi_key = (
                    mol.GetProp("INCHIKEY")
                    if mol.HasProp("INCHIKEY")
                    else None
                )
                metadata = self.extract_metadata(mol)

                record = {
                    "mol_id": mol_id,  # Use the canonical mol_id here
                    "inchi_key": inchi_key,
                    "standardized_smiles": std_smiles,
                    "mol_weight": Descriptors.MolWt(mol),
                    "num_atoms": mol.GetNumAtoms(),
                    "num_bonds": mol.GetNumBonds(),
                    **metadata,
                }

                # 3. Apply all filters
                if record["mol_weight"] > 750:
                    logger.warning(
                        f"Skipping molecule {mol_id} because it has a molecular weight greater than 750"
                    )
                    skipped_count += 1
                    continue

                if (
                    "spectrum_type" in record
                    and record["spectrum_type"] != "MS1"
                ):
                    logger.warning(
                        f"Skipping molecule {mol_id} because it is not an MS1 spectrum"
                    )
                    skipped_count += 1
                    continue

                if (
                    "collision_energy" in record
                    and record["collision_energy"] is not None
                ):
                    logger.warning(
                        f"Skipping molecule {mol_id} because it has a collision energy"
                    )
                    skipped_count += 1
                    continue

                if "ion_mode" in record and record["ion_mode"] == "N":
                    logger.warning(
                        f"Skipping molecule {mol_id} because it has a negative ion mode"
                    )
                    skipped_count += 1
                    continue

                if (
                    "precursor_m_z" in record
                    and record["precursor_m_z"] is not None
                ):
                    logger.warning(
                        f"Skipping molecule {mol_id} because it has a precursor m/z"
                    )
                    skipped_count += 1
                    continue

                if any(
                    atom.GetSymbol()
                    not in [
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
                    for atom in mol.GetAtoms()
                ):
                    logger.warning(
                        "Skipping molecule because it contains atoms that are not in the list"
                    )
                    skipped_count += 1
                    continue

                # 4. Process and filter spectrum
                masses, intensities = None, None
                if mol.HasProp("MASS SPECTRAL PEAKS"):
                    peaks_text = mol.GetProp("MASS SPECTRAL PEAKS")
                    masses, intensities = self.parse_spectrum_peaks(peaks_text)

                    if masses is not None and len(masses) > 0:
                        highest_peak = masses[np.argmax(intensities)]
                        if highest_peak > (record["mol_weight"] + 5):
                            logger.warning(
                                f"Skipping molecule {mol_id} because the highest peak is higher than the molecular weight"
                            )
                            skipped_count += 1
                            continue

                        record["has_spectrum"] = True
                        record["num_peaks"] = len(masses)

                        # Normalize intensities so that the highest intensity is 1
                        intensities = intensities / np.max(intensities)

                        # 5. Save spectrum data using the canonical mol_id
                        self.spectra_data[mol_id] = {
                            "masses": masses,
                            "intensities": intensities,
                        }
                    else:
                        record["has_spectrum"] = False
                        record["num_peaks"] = 0
                else:
                    record["has_spectrum"] = False
                    record["num_peaks"] = 0

                # 6. Append metadata record with the canonical mol_id
                self.metadata_records.append(record)
                valid_count += 1

            except Exception as e:
                logger.warning(f"Error processing molecule {mol_id}: {e}")
                skipped_count += 1
                continue

        logger.info(
            f"Processed: {valid_count} valid molecules, {skipped_count} skipped"
        )
        return valid_count, skipped_count

    def save_metadata_df(self):
        """Save metadata as tab-separated file."""
        df = pd.DataFrame(self.metadata_records)

        if "comment" in df.columns:
            df["MoNA_rating"] = (
                df["comment"]
                .str.extract(r"MoNA Rating=(\d+\.\d+)")
                .astype(float)
            )

            # Get mol_ids before deduplication
            mol_ids_before = set(df["mol_id"].tolist())

            # for entries with same inchi_key, keep the one with the highest MoNA_rating
            df = df.sort_values(
                "MoNA_rating", ascending=False
            ).drop_duplicates(subset="inchi_key")

            # Get mol_ids after deduplication
            mol_ids_after = set(df["mol_id"].tolist())
            removed_mol_ids = mol_ids_before - mol_ids_after
        else:
            removed_mol_ids = []

        logger.info(f"After deduplication: {len(df)} molecules")
        logger.info(f"Removed {len(removed_mol_ids)} duplicates")

        # Remove spectra for molecules that were deduplicated out
        for mol_id in removed_mol_ids:
            if mol_id in self.spectra_data:
                del self.spectra_data[mol_id]

        logger.info(
            f"Spectra data now contains {len(self.spectra_data)} entries"
        )

        # Reorder columns to put important ones first
        important_cols = [
            "mol_id",
            "inchi_key",
            "standardized_smiles",
            "has_spectrum",
            "num_peaks",
        ]
        other_cols = [col for col in df.columns if col not in important_cols]
        df = df[important_cols + other_cols]

        columns_to_drop = [
            "mass_spectral_peaks",
            "collision_energy",
            "precursor_m_z",
            "ion_mode",
            "spectrum_type",
        ]
        # Only drop columns that actually exist
        existing_columns_to_drop = [
            col for col in columns_to_drop if col in df.columns
        ]
        if existing_columns_to_drop:
            df = df.drop(columns=existing_columns_to_drop)
            logger.info(f"Dropped columns: {existing_columns_to_drop}")

        output_path = self.output_dir / "metadata.tsv"
        df.to_csv(output_path, sep="\t", index=False)
        logger.info(f"Saved metadata to: {output_path}")
        return df

    def save_spectra_hdf5(self):
        """
        Save spectra with mol_ids as top-level keys - much better access pattern!
        """
        if not self.spectra_data:
            logger.warning("No spectra data to save")
            return

        output_path = self.output_dir / "spectra.hdf5"

        with h5py.File(output_path, "w") as f:
            # Store each mol_id as a top-level group
            for mol_id, spectrum_data in self.spectra_data.items():
                # Create group for this mol_id
                mol_group = f.create_group(mol_id)

                # Store spectrum data
                mol_group.create_dataset(
                    "masses",
                    data=spectrum_data["masses"],
                    compression="lzf",
                    dtype=np.float32,
                )

                mol_group.create_dataset(
                    "intensities",
                    data=spectrum_data["intensities"],
                    compression="lzf",
                    dtype=np.float32,
                )

                # Find and store metadata
                mol_metadata = next(
                    (
                        record
                        for record in self.metadata_records
                        if record["mol_id"] == mol_id
                    ),
                    None,
                )

                if mol_metadata:
                    if mol_metadata.get("inchi_key"):
                        mol_group.create_dataset(
                            "inchi_key",
                            data=mol_metadata["inchi_key"].encode(),
                            dtype="S100",
                        )

                    if mol_metadata.get("standardized_smiles"):
                        mol_group.create_dataset(
                            "standardized_smiles",
                            data=mol_metadata["standardized_smiles"].encode(),
                            dtype="S200",
                        )

        logger.info(
            f"Saved {len(self.spectra_data)} spectra to: {output_path}"
        )

    def compare_storage_formats(self):
        """Save data in all formats and compare file sizes."""
        logger.info("Comparing storage formats...")

        # Save in all formats
        self.save_spectra_hdf5()

        # Compare file sizes
        formats = {
            "HDF5 Optimized": self.output_dir / "spectra.hdf5",
            "Metadata TSV": self.output_dir / "metadata.tsv",
        }

        print("\n" + "=" * 50)
        print("FILE SIZE COMPARISON")
        print("=" * 50)

        sizes = {}
        for name, path in formats.items():
            if path.exists():
                size_mb = path.stat().st_size / 1024**2
                sizes[name] = size_mb
                print(f"{name:15}: {size_mb:8.2f} MB")

        if len(sizes) > 1:
            smallest = min(sizes.values())
            print("\nCompression ratios (vs smallest):")
            for name, size in sizes.items():
                ratio = size / smallest
                print(f"{name:15}: {ratio:6.1f}x")

    def process_and_save(self):
        """Complete processing pipeline."""
        # Process SDF
        valid_count, skipped_count = self.process_sdf()

        # Save metadata
        df = self.save_metadata_df()

        # Save spectra in requested format(s)
        self.save_spectra_hdf5()

        # Print summary
        logger.info(f"""
        Processing Summary:
        Total valid molecules: {valid_count}
        Molecules with spectra: {len(self.spectra_data)}
        Molecules skipped: {skipped_count}
        
        Output files:
        - Metadata: {self.output_dir}/metadata.tsv
        - Spectra: {self.output_dir}/spectra.*
        """)

        return df


def load_metadata(data_dir):
    """Load metadata DataFrame."""
    path = Path(data_dir) / "metadata.tsv"
    return pd.read_csv(path, sep="\t")


def load_spectrum(mol_id, data_dir):
    """Load spectrum from new HDF5 format with mol_ids as top-level keys."""
    path = Path(data_dir) / "spectra.hdf5"

    try:
        with h5py.File(path, "r") as f:
            # Check if mol_id exists as a top-level key
            if mol_id not in f:
                logger.warning(f"mol_id {mol_id} not found in dataset")
                return None, None, None, None

            # Access the group for this mol_id
            mol_group = f[mol_id]

            # Load spectrum data
            masses = mol_group["masses"][:]
            intensities = mol_group["intensities"][:]

            # Load metadata if available
            inchi_key = ""
            if "inchi_key" in mol_group:
                inchi_key = mol_group["inchi_key"][()].decode()

            standardized_smiles = ""
            if "standardized_smiles" in mol_group:
                standardized_smiles = mol_group["standardized_smiles"][
                    ()
                ].decode()

            return masses, intensities, inchi_key, standardized_smiles

    except Exception as e:
        logger.error(f"Error loading spectrum for {mol_id}: {e}")
        return None, None, None, None


def load_all_spectra_optimized(data_dir="./processed_data"):
    """Load all spectra from new HDF5 format."""
    path = Path(data_dir) / "spectra.hdf5"
    spectra = {}

    try:
        with h5py.File(path, "r") as f:
            # Get list of all mol_ids
            mol_ids_bytes = f["mol_ids"][:]
            mol_ids = [mid.decode() for mid in mol_ids_bytes]

            # Load each spectrum
            for mol_id in mol_ids:
                if mol_id in f:  # Check if the group exists
                    mol_group = f[mol_id]

                    spectra[mol_id] = {
                        "masses": mol_group["masses"][:],
                        "intensities": mol_group["intensities"][:],
                    }

                    # Add metadata if available
                    if "inchi_key" in mol_group:
                        spectra[mol_id]["inchi_key"] = mol_group["inchi_key"][
                            ()
                        ].decode()

                    if "standardized_smiles" in mol_group:
                        spectra[mol_id]["standardized_smiles"] = mol_group[
                            "standardized_smiles"
                        ][()].decode()

        logger.info(f"Loaded {len(spectra)} spectra from {path}")
        return spectra

    except Exception as e:
        logger.error(f"Error loading all spectra: {e}")
        return {}


# Example usage
if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "--sdf-path",
        type=str,
        required=True,
        help="Path to the input SDF file (e.g. NIST2023_GCMS_main.sdf).",
    )
    args.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for spectra.hdf5 and metadata.tsv (e.g. data/NIST2023_GCMS_main/).",
    )
    args = args.parse_args()

    # Initialize processor
    processor = SDFToMLDataset(args.sdf_path, output_dir=args.output_dir)

    # Process and save with comparison of all formats
    df = processor.process_and_save()

    # # Show some stats
    print(f"\nProcessed {len(df)} molecules")
    print(f"With spectra: {df['has_spectrum'].sum()}")
    print("\nSample of data:")
    print(
        df[
            [
                "mol_id",
                "inchi_key",
                "standardized_smiles",
                "has_spectrum",
                "num_peaks",
            ]
        ].head()
    )

    mol_id = "UO000024"
    masses, intensities, inchi_key, standardized_smiles = load_spectrum(
        mol_id, args.output_dir
    )
    print(f"Spectrum peaks: {len(masses)}")
    print(f"Spectrum masses: {masses}")
    print(f"Spectrum intensities: {intensities}")
    print(f"Spectrum inchi_key: {inchi_key}")
    print(f"Spectrum standardized_smiles: {standardized_smiles}")
    print(f"Spectrum mol_id: {mol_id}")
