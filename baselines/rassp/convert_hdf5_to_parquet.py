#!/usr/bin/env python3
"""
Convert HDF5 spectral data to RASSP Parquet format.

Usage:
    python convert_hdf5_to_rassp_parquet.py /path/to/spectra.hdf5 output.parquet

    # With split file to create separate train/val/test parquet files:
    python convert_hdf5_to_rassp_parquet.py /path/to/spectra.hdf5 output.parquet \\
        --split-file /path/to/splits.tsv
"""

import logging
import os
import zlib

import h5py
import pandas as pd
from rassp.msutil import masscompute
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors
from tqdm import tqdm

RDLogger.DisableLog("rdApp.*")


def get_actual_formulae_count(
    mol, allowed_elements=[1, 6, 7, 8, 9, 15, 16, 17]
):
    """
    Use RASSP's actual formula enumerator to count unique formulae.
    This is what they used for filtering the NIST dataset!
    """
    try:
        # Use the same enumerator RASSP uses
        ffe = masscompute.FragmentFormulaPeakEnumerator(
            allowed_elements,
            use_highres=True,  # Match RASSP config
            max_peak_num=12,  # Match RASSP config
        )
        formulae, masses = ffe.get_frag_formulae(mol)
        return len(formulae)
    except Exception as e:
        raise e


def compute_morgan_fingerprint_crc32(mol, radius=2, nbits=2048):
    """Compute Morgan fingerprint and its CRC32 checksum."""
    if mol is None:
        return None

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    fp_bytes = fp.ToBitString().encode()
    crc32 = zlib.crc32(fp_bytes) & 0xFFFFFFFF
    return crc32


def spectrum_to_list(masses, intensities, min_intensity=0.001):
    """
    Convert mass/intensity arrays to list of tuples format.

    RASSP expects: List[Tuple[Float, Float]] where each tuple is (mass, intensity)
    """
    # Filter out very low intensity peaks
    mask = intensities >= min_intensity
    masses_filtered = masses[mask]
    intensities_filtered = intensities[mask]

    # Normalize intensities to sum to 1.0
    if len(intensities_filtered) > 0:
        intensities_filtered = (
            intensities_filtered / intensities_filtered.sum()
        )

    # Convert to list of tuples
    spectrum = [
        (float(m), float(i))
        for m, i in zip(masses_filtered, intensities_filtered)
    ]

    return spectrum


def convert_hdf5_to_rassp_parquet(
    hdf5_path,
    output_parquet_path,
    split_file=None,
    max_molecules=None,
    filter_max_mass=511,
    filter_max_atoms=48,
    filter_max_formulae=4096,
    min_intensity=0.001,
    allowed_elements={1, 6, 7, 8, 9, 15, 16, 17},
):
    """
    Convert HDF5 spectral database to RASSP Parquet format.

    Args:
        hdf5_path: Path to input HDF5 file
        output_parquet_path: Path to output Parquet file
        split_file: Optional path to TSV file with mol_id and split columns.
            If provided, creates separate {name}_{train|val|test}.parquet files.
        max_molecules: Optional limit on number of molecules to process
        filter_max_mass: Maximum molecular mass (filter out larger molecules)
        filter_max_atoms: Maximum number of atoms (filter out larger molecules)
        min_intensity: Minimum intensity threshold for peaks
    """

    logging.info(f"Reading HDF5 file: {hdf5_path}")

    # Load split file if provided
    split_mapping = None
    if split_file is not None:
        logging.info(f"Loading split file: {split_file}")
        split_df = pd.read_csv(split_file, sep="\t", dtype={"mol_id": str})
        split_mapping = dict(zip(split_df["mol_id"], split_df["split"]))
        logging.info(f"  Loaded {len(split_mapping)} mol_id -> split mappings")

    allowed_elements_list = sorted(list(allowed_elements))

    records = []
    skipped = {
        "no_smiles": 0,
        "invalid_smiles": 0,
        "too_large": 0,
        "too_heavy": 0,
        "no_spectrum": 0,
        "unsupported_elements": 0,
        "too_many_formulae": 0,
    }

    with h5py.File(hdf5_path, "r") as f:
        mol_ids = list(f.keys())

        if max_molecules is not None:
            mol_ids = mol_ids[:max_molecules]

        logging.info(f"Processing {len(mol_ids)} molecules...")

        for mol_id in tqdm(mol_ids):
            try:
                mol_data = f[mol_id]

                # Extract data
                smiles = (
                    mol_data["standardized_smiles"][()].decode("utf-8")
                    if isinstance(mol_data["standardized_smiles"][()], bytes)
                    else str(mol_data["standardized_smiles"][()])
                )

                inchi_key = (
                    mol_data["inchi_key"][()].decode("utf-8")
                    if isinstance(mol_data["inchi_key"][()], bytes)
                    else str(mol_data["inchi_key"][()])
                )

                masses = mol_data["masses"][:]
                intensities = mol_data["intensities"][:]

                # Skip if no SMILES
                if not smiles or smiles == "":
                    skipped["no_smiles"] += 1
                    continue

                # Create RDKit molecule
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    skipped["invalid_smiles"] += 1
                    continue

                # Add hydrogens
                mol = Chem.AddHs(mol)

                mol_elements = set(
                    atom.GetAtomicNum() for atom in mol.GetAtoms()
                )
                if not mol_elements.issubset(allowed_elements):
                    skipped["unsupported_elements"] += 1
                    continue

                # Filter by size
                n_atoms = mol.GetNumAtoms()
                if n_atoms > filter_max_atoms:
                    skipped["too_large"] += 1
                    continue

                # Filter by mass
                mol_mass = Descriptors.ExactMolWt(mol)
                if mol_mass > filter_max_mass:
                    skipped["too_heavy"] += 1
                    continue

                # Check spectrum
                if len(masses) == 0 or len(intensities) == 0:
                    skipped["no_spectrum"] += 1
                    continue

                actual_formulae_count = get_actual_formulae_count(
                    mol, allowed_elements_list
                )
                if actual_formulae_count > filter_max_formulae:
                    skipped["too_many_formulae"] += 1
                    continue

                # Compute Morgan fingerprint for CV splitting
                morgan_crc32 = compute_morgan_fingerprint_crc32(
                    mol, radius=2, nbits=2048
                )

                # Convert spectrum to RASSP format
                spectrum = spectrum_to_list(
                    masses, intensities, min_intensity=min_intensity
                )

                # Get InChI
                try:
                    inchi = Chem.MolToInchi(mol)
                except:
                    inchi = ""

                # Serialize molecule for storage
                rdmol_binary = mol.ToBinary()

                # Create record
                record = {
                    "mol_id": mol_id,
                    "smiles": smiles,
                    "inchi": inchi,
                    "inchi_key": inchi_key,
                    "rdmol": rdmol_binary,
                    "spect": spectrum,
                    "morgan4_crc32": morgan_crc32,
                    "n_atoms": n_atoms,
                    "mol_mass": mol_mass,
                    "n_peaks": len(spectrum),
                    "n_formulae": actual_formulae_count,
                }

                records.append(record)

            except Exception as e:
                logging.info(f"\nError processing {mol_id}: {e}")
                continue

    logging.info(f"\nProcessed {len(records)} molecules successfully")
    logging.info("Skipped molecules:")
    for reason, count in skipped.items():
        logging.info(f"  {reason}: {count}")

    # Create DataFrame
    df = pd.DataFrame(records)

    # Save to Parquet
    if split_mapping is not None:
        # Add split column based on mol_id
        df["split"] = df["mol_id"].map(split_mapping)

        # Check for molecules without split assignment
        missing_splits = df["split"].isna().sum()
        if missing_splits > 0:
            logging.info(
                f"\nWarning: {missing_splits} molecules not found in split file"
            )
            # Filter out molecules without split assignment
            df = df[df["split"].notna()]

        # Get base name and extension for output files
        base_path, ext = os.path.splitext(output_parquet_path)

        # Save separate files for each split
        splits = df["split"].unique()
        logging.info("\nSaving split files...")
        for split_name in sorted(splits):
            split_df = df[df["split"] == split_name].copy()
            split_df = split_df.drop(columns=["split"])  # Remove split column
            split_path = f"{base_path}_{split_name}{ext}.parquet"
            split_df.to_parquet(split_path, index=False, engine="pyarrow")
            logging.info(f"  Saved {len(split_df)} molecules to {split_path}")

        logging.info(
            f"\nDone! Saved {len(df)} molecules across {len(splits)} split files"
        )
    else:
        logging.info(f"\nSaving to {output_parquet_path}.parquet ...")
        df.to_parquet(
            f"{output_parquet_path}.parquet", index=False, engine="pyarrow"
        )
        logging.info(
            f"Done! Saved {len(df)} molecules to {output_parquet_path}"
        )

    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert HDF5 spectral data to RASSP Parquet format"
    )
    parser.add_argument(
        "--hdf5-path", type=str, required=True, help="Path to input HDF5 file"
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Path to output Parquet file (without split or .parquet suffix)",
    )
    parser.add_argument(
        "--split-file",
        type=str,
        default=None,
        help="Path to TSV file with mol_id and split columns. "
        "If provided, creates {output}_{train|val|test}.parquet files",
    )
    parser.add_argument(
        "--max-molecules",
        type=int,
        default=None,
        help="Maximum number of molecules to process (default: all)",
    )

    args = parser.parse_args()

    log_file = f"{args.name}_conversion.log"
    os.makedirs(
        os.path.dirname(log_file) if os.path.dirname(log_file) else ".",
        exist_ok=True,
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )

    # Convert HDF5 to Parquet
    df = convert_hdf5_to_rassp_parquet(
        args.hdf5_path,
        args.name,
        split_file=args.split_file,
        max_molecules=args.max_molecules,
    )
