#!/usr/bin/env python
"""
Convert NIST23 GC-MS data from HDF5 + TSV splits to MassFormer format.

This script creates spec_df.pkl and mol_df.pkl files needed by MassFormer
from the existing ICICLE data format (HDF5 spectra + TSV splits).

For EI/GC-MS data, we set default values for missing metadata:
- prec_type: "EI" (electron ionization)
- col_energy: None (not applicable for EI)
- inst_type: "EI-GC" (default instrument type)
- frag_mode: "EI" (electron ionization fragmentation)
- spec_type: "EI"
- ion_mode: "EI"
"""

import argparse
import os
import sys

import h5py
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from massformer.data_utils import rdkit_import
from rdkit.Chem import Descriptors


def load_spectra_from_hdf5(hdf5_path, mol_ids):
    """Load spectra from HDF5 file for given mol_ids.

    Args:
        hdf5_path: Path to spectra.hdf5 file
        mol_ids: List of molecule IDs to load

    Returns:
        dict: Mapping from mol_id to spectrum data
    """
    print(f"Loading spectra from {hdf5_path}...")
    spec_data = {}

    with h5py.File(hdf5_path, "r") as f:
        for mol_id in tqdm(mol_ids, desc="Loading spectra"):
            mol_id_str = str(mol_id)
            if mol_id_str not in f:
                print(f"Warning: mol_id {mol_id} not found in HDF5 file")
                continue

            group = f[mol_id_str]
            masses = group["masses"][:]
            intensities = group["intensities"][:]
            smiles = group["standardized_smiles"][()].decode("utf-8")
            inchi_key = group["inchi_key"][()].decode("utf-8")

            # Format peaks as list of [mz, intensity] pairs
            peaks = [[float(m), float(i)] for m, i in zip(masses, intensities)]

            spec_data[mol_id] = {
                "smiles": smiles,
                "inchi_key": inchi_key,
                "peaks": peaks,
            }

    return spec_data


def create_spec_and_mol_dfs(spec_data, splits_df):
    """Create spec_df and mol_df dataframes.

    Args:
        spec_data: Dict mapping mol_id to spectrum data
        splits_df: DataFrame with mol_id, inchi_key, split columns

    Returns:
        tuple: (spec_df, mol_df)
    """
    print("Creating spec_df and mol_df...")

    # Import RDKit
    Chem, rdinchi, AllChem = rdkit_import(
        "rdkit.Chem", "rdkit.Chem.rdinchi", "rdkit.Chem.AllChem"
    )

    # Create a mapping from mol_id to split
    mol_id_to_split = dict(zip(splits_df["mol_id"], splits_df["split"]))

    # Create spec_df entries
    spec_entries = []
    for mol_id, data in tqdm(spec_data.items(), desc="Processing spectra"):
        spec_entry = {
            "spec_id": mol_id,  # Use mol_id as spec_id (one spectrum per molecule)
            "mol_id": mol_id,
            "prec_type": "EI",  # Electron ionization
            "inst_type": "EI-GC",  # GC-MS instrument
            "frag_mode": "EI",  # EI fragmentation
            "spec_type": "EI",  # EI spectrum type
            "ion_mode": "EI",  # EI ionization mode
            "dset": "nist23",
            "col_gas": None,  # Not applicable for EI
            "res": 1,  # Low resolution (unit mass)
            "ace": None,  # No collision energy for EI
            "nce": None,  # No normalized collision energy for EI
            "prec_mz": None,  # No precursor for EI (will be computed from MW)
            "peaks": data["peaks"],
            "ri": None,  # Retention index not provided
            "split": mol_id_to_split.get(mol_id, None),  # Add split assignment
        }
        spec_entries.append(spec_entry)

    spec_df = pd.DataFrame(spec_entries)

    # Create mol_df from unique molecules
    print("Creating mol_df...")
    mol_entries = []
    seen_smiles = set()

    for mol_id, data in tqdm(spec_data.items(), desc="Processing molecules"):
        smiles = data["smiles"]
        if smiles in seen_smiles:
            continue
        seen_smiles.add(smiles)

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(
                f"Warning: Could not parse SMILES for mol_id {mol_id}: {smiles}"
            )
            continue

        # Compute molecular properties
        try:
            # MolToInchi can return a tuple (code, inchi) or just inchi
            inchi_result = rdinchi.MolToInchi(mol)
            if isinstance(inchi_result, tuple):
                inchi = inchi_result[1] if inchi_result[0] == 0 else None
            else:
                inchi = inchi_result
            inchikey = rdinchi.InchiToInchiKey(inchi) if inchi else None
            inchikey_s = inchikey.split("-")[0] if inchikey else None

            # Get Murcko scaffold
            from rdkit.Chem.Scaffolds import MurckoScaffold

            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            scaffold_smiles = Chem.MolToSmiles(scaffold) if scaffold else None

            # Get molecular formula
            formula = Chem.rdMolDescriptors.CalcMolFormula(mol)

            # Get molecular weight
            mw = Descriptors.MolWt(mol)
            exact_mw = Descriptors.ExactMolWt(mol)

            mol_entry = {
                "mol_id": mol_id,
                "smiles": smiles,
                "mol": mol,
                "inchikey_s": inchikey_s,
                "scaffold": scaffold_smiles,
                "formula": formula,
                "inchi": inchi,
                "mw": mw,
                "exact_mw": exact_mw,
                "split": mol_id_to_split.get(
                    mol_id, None
                ),  # Add split assignment
            }
            mol_entries.append(mol_entry)
        except Exception as e:
            print(f"Warning: Error processing mol_id {mol_id}: {e}")
            continue

    mol_df = pd.DataFrame(mol_entries)

    # Set precursor m/z to molecular weight for EI spectra
    # Merge mol_df to get exact_mw for each spectrum
    spec_df = spec_df.merge(
        mol_df[["mol_id", "exact_mw"]], on="mol_id", how="left"
    )
    spec_df["prec_mz"] = spec_df["exact_mw"]
    spec_df = spec_df.drop(columns=["exact_mw"])

    return spec_df, mol_df


def main():
    parser = argparse.ArgumentParser(
        description="Convert NIST23 GC-MS data to MassFormer format"
    )
    parser.add_argument(
        "--hdf5-path",
        type=str,
        default="data/NIST2023_GCMS_main/spectra.hdf5",
        help="Path to spectra.hdf5 file",
    )
    parser.add_argument(
        "--splits-path",
        type=str,
        default="data/NIST2023_GCMS_main/splits/scaffold_no_xeno_aas_deduplicated.tsv",
        help="Path to splits TSV file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/proc/nist23",
        help="Output directory for spec_df.pkl and mol_df.pkl",
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load splits file
    print(f"Loading splits from {args.splits_path}...")
    splits_df = pd.read_csv(args.splits_path, sep="\t")
    print(f"Loaded {len(splits_df)} molecules from splits file")

    # Load spectra
    mol_ids = splits_df["mol_id"].unique()
    spec_data = load_spectra_from_hdf5(args.hdf5_path, mol_ids)
    print(f"Loaded {len(spec_data)} spectra")

    # Create dataframes
    spec_df, mol_df = create_spec_and_mol_dfs(spec_data, splits_df)

    print(f"\nCreated spec_df with {len(spec_df)} entries")
    print(f"Created mol_df with {len(mol_df)} entries")

    # Verify splits are present
    print("\nSplit distribution in spec_df:")
    print(spec_df["split"].value_counts())
    print("\nSplit distribution in mol_df:")
    print(mol_df["split"].value_counts())

    # Save to pickle files
    spec_df_path = os.path.join(args.output_dir, "spec_df.pkl")
    mol_df_path = os.path.join(args.output_dir, "mol_df.pkl")

    print(f"\nSaving spec_df to {spec_df_path}...")
    spec_df.to_pickle(spec_df_path)

    print(f"Saving mol_df to {mol_df_path}...")
    mol_df.to_pickle(mol_df_path)

    print("\nDataFrame info:")
    print("\nspec_df columns:", spec_df.columns.tolist())
    print("spec_df shape:", spec_df.shape)
    print("\nspec_df sample:")
    print(spec_df.head())
    print("\nspec_df dtypes:")
    print(spec_df.dtypes)
    print("\nmol_df columns:", mol_df.columns.tolist())
    print("mol_df shape:", mol_df.shape)
    print("\nmol_df sample:")
    print(mol_df.head())

    print("\n✓ Conversion complete!")


if __name__ == "__main__":
    main()
