#!/usr/bin/env python
"""Prepare NIST data for AIRI model training.

This script converts NIST metadata with retention index values into the parquet
format required by the AIRI model.
"""

import logging
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import click
import jsonpickle
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# Shortest Path Computation (from masskit)


def ordered_pair(a1, a2):
    """Return atom indices as ordered pair."""
    if a1 > a2:
        return (a2, a1)
    else:
        return (a1, a2)


def get_ring_paths(rd_mol):
    """Get ring information for atom pairs."""
    rings_dict = {}
    ssr = [list(x) for x in Chem.GetSymmSSSR(rd_mol)]
    for ring in ssr:
        ring_sz = len(ring)
        is_aromatic = True
        for atom_idx in ring:
            if not rd_mol.GetAtoms()[atom_idx].GetIsAromatic():
                is_aromatic = False
                break
        for ring_idx, atom_idx in enumerate(ring):
            for other_idx in ring[ring_idx:]:
                atom_pair = ordered_pair(atom_idx, other_idx)
                if atom_pair not in rings_dict:
                    rings_dict[atom_pair] = [(ring_sz, is_aromatic)]
                else:
                    if (ring_sz, is_aromatic) not in rings_dict[atom_pair]:
                        rings_dict[atom_pair].append((ring_sz, is_aromatic))
    return rings_dict


def get_shortest_paths(rd_mol, max_path_length=5):
    """Compute shortest paths for all atom pairs in a molecule.

    Returns tuple of (paths_dict, pointer_dict, ring_dict) containing shortest
    path information required by AIRI model.
    """
    fragments = Chem.rdmolops.GetMolFrags(rd_mol)

    def get_atom_frag(fragments, atom_idx):
        for frag in fragments:
            if atom_idx in frag:
                return frag
        assert False

    n_atoms = rd_mol.GetNumAtoms()
    paths_dict = {}
    pointer_dict = {}

    for atom_idx in range(n_atoms):
        atom_frag = get_atom_frag(fragments, atom_idx)

        for other_idx in range(atom_idx + 1, n_atoms, 1):
            if other_idx not in atom_frag:
                continue

            shortest_path = Chem.rdmolops.GetShortestPath(
                rd_mol, atom_idx, other_idx
            )
            path_length = len(shortest_path) - 1

            if path_length > max_path_length:
                pointer_dict[(atom_idx, other_idx)] = shortest_path[
                    max_path_length
                ]
                pointer_dict[(other_idx, atom_idx)] = shortest_path[
                    -1 - max_path_length
                ]
            else:
                paths_dict[(atom_idx, other_idx)] = shortest_path

    ring_dict = get_ring_paths(rd_mol)
    return paths_dict, pointer_dict, ring_dict


def mol_to_shortest_path(mol, max_path_length=5):
    """Convert RDKit mol to shortest path JSON string."""
    if mol is None:
        return None
    try:
        return jsonpickle.encode(
            get_shortest_paths(mol, max_path_length), keys=True
        )
    except Exception as e:
        logger.warning(f"Failed to compute shortest paths: {e}")
        return None


def smiles_to_mol_and_path(smiles, max_path_length=5):
    """Convert SMILES to (mol, shortest_paths) tuple."""
    if pd.isna(smiles) or not smiles:
        return None, None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, None
        shortest_paths = mol_to_shortest_path(mol, max_path_length)
        return mol, shortest_paths
    except Exception as e:
        logger.warning(f"Failed to process SMILES '{smiles}': {e}")
        return None, None


# Data Loading and Processing


def parse_retention_index(ri_string: str) -> dict:
    """Parse retention index string and extract values for each column type."""
    if pd.isna(ri_string):
        return {}

    result = {}
    parts = ri_string.split()

    for part in parts:
        if "=" in part:
            col_type, value_str = part.split("=", 1)
            primary_value = value_str.split("/")[0]
            try:
                result[col_type] = float(primary_value)
            except ValueError:
                continue

    return result


def get_preferred_ri(row: pd.Series) -> float:
    """Get preferred RI value from available types."""
    for col in ["ri_StdNP", "ri_SemiStdNP", "ri_StdPolar", "ri_Any"]:
        if col in row and pd.notna(row[col]):
            return row[col]
    return np.nan


def load_nist_data(
    data_dir: str,
    splits_file: str = "scaffold_no_xeno_aas_deduplicated.tsv",
    ri_type: str = None,
) -> pd.DataFrame:
    """Load NIST metadata and splits, filter for RI data.

    Parameters
    ----------
    data_dir : str
        Path to NIST data directory
    splits_file : str
        Name of splits file in splits/ subdirectory
    ri_type : str, optional
        Specific RI type to filter for (StdNP, SemiStdNP, StdPolar)
        If None, uses unified RI (preferred available type)

    Returns
    -------
    pd.DataFrame
        DataFrame with mol_id, smiles, ri value, and split assignment
    """
    data_path = Path(data_dir)

    logger.info(f"Loading data from {data_dir}")

    # Load metadata
    metadata = pd.read_csv(data_path / "metadata.tsv", sep="\t")
    logger.info(f"Metadata shape: {metadata.shape}")

    # Load splits - handle both filename and full path
    splits_path = Path(splits_file)
    if splits_path.is_absolute() or splits_path.exists():
        # Full path provided
        splits = pd.read_csv(splits_path, sep="\t")
    else:
        # Just filename - look in splits/ subdirectory
        splits = pd.read_csv(data_path / "splits" / splits_file, sep="\t")
    logger.info(f"Splits shape: {splits.shape}")

    # Parse retention indices
    parsed_ri = metadata["retention_index"].apply(parse_retention_index)

    ri_types = ["StdNP", "SemiStdNP", "StdPolar", "Any"]
    for rt in ri_types:
        metadata[f"ri_{rt}"] = parsed_ri.apply(
            lambda x, rt=rt: x.get(rt, np.nan)
        )

    # Add unified RI
    metadata["ri_unified"] = metadata.apply(get_preferred_ri, axis=1)

    # Log RI coverage
    logger.info("Retention Index Coverage by Type:")
    for rt in ri_types:
        count = metadata[f"ri_{rt}"].notna().sum()
        logger.info(f"  {rt}: {count:,} ({100 * count / len(metadata):.1f}%)")

    # Merge with splits
    df = splits.merge(
        metadata[
            ["mol_id", "standardized_smiles", "inchikey"]
            + [f"ri_{rt}" for rt in ri_types]
            + ["ri_unified"]
        ],
        left_on="mol_id",
        right_on="mol_id",
        how="inner",
    )

    # Determine which RI column to use
    if ri_type:
        ri_col = f"ri_{ri_type}"
        if ri_col not in df.columns:
            raise ValueError(f"Unknown RI type: {ri_type}")
    else:
        ri_col = "ri_unified"

    # Filter to molecules with RI data
    df = df[df[ri_col].notna()].copy()
    df = df[(df[ri_col] > 0) & (df[ri_col] <= 10000)].copy()

    # Rename for consistency with AIRI format
    df = df.rename(
        columns={
            "standardized_smiles": "smiles",
            ri_col: "experimental_ri",
            "split": "set",
        }
    )

    logger.info(f"Molecules with valid RI data: {len(df):,}")
    logger.info(f"  Train: {(df['set'] == 'train').sum():,}")
    logger.info(f"  Valid: {(df['set'] == 'val').sum():,}")
    logger.info(f"  Test: {(df['set'] == 'test').sum():,}")

    return df[["mol_id", "smiles", "experimental_ri", "set", "inchikey"]]


def process_molecules(
    df: pd.DataFrame, max_path_length: int = 5, num_workers: int = 8
) -> pd.DataFrame:
    """Process SMILES to mol objects and compute shortest paths.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'smiles' column
    max_path_length : int
        Maximum path length for shortest path computation
    num_workers : int
        Number of parallel workers

    Returns
    -------
    pd.DataFrame
        DataFrame with additional 'mol' and 'shortest_paths' columns
    """
    logger.info(f"Processing {len(df):,} molecules...")

    smiles_list = df["smiles"].tolist()

    # Process in parallel
    with Pool(num_workers) as p:
        results = list(
            tqdm(
                p.imap(
                    partial(
                        smiles_to_mol_and_path, max_path_length=max_path_length
                    ),
                    smiles_list,
                ),
                total=len(smiles_list),
                desc="Processing molecules",
            )
        )

    mols, shortest_paths = zip(*results)
    df = df.copy()
    df["mol"] = list(mols)
    df["shortest_paths"] = list(shortest_paths)

    # Filter out failed molecules
    valid_mask = df["mol"].notna() & df["shortest_paths"].notna()
    n_failed = (~valid_mask).sum()
    if n_failed > 0:
        logger.warning(
            f"Failed to process {n_failed:,} molecules, removing them"
        )
        df = df[valid_mask].copy()

    logger.info(f"Successfully processed {len(df):,} molecules")

    return df


def mol_to_json(mol):
    """Convert RDKit mol to JSON string and verify it can be read back."""
    if mol is None:
        return None
    try:
        # 1. Ensure molecule is in a valid state
        Chem.SanitizeMol(mol)
        json_str = Chem.rdMolInterchange.MolToJSON(mol)

        # 2. VALIDATION: Try to hydrate it back.
        # If this fails or returns None, the training script WILL crash.
        test_mols = Chem.rdMolInterchange.JSONToMols(json_str)
        if not test_mols or test_mols[0] is None:
            logger.warning(
                f"Molecule JSON validation failed, SMILES {Chem.MolToSmiles(mol)}"
            )
            return None

        return json_str
    except Exception:
        return None


def save_to_parquet(df: pd.DataFrame, output_path: Path, split: str = None):
    """Save DataFrame to parquet format compatible with AIRI/masskit.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with mol, shortest_paths, experimental_ri columns
    output_path : Path
        Output file path
    split : str, optional
        If provided, filter to this split before saving
    """
    from masskit.data_specs.arrow_types import MolArrowType, PathArrowType

    if split:
        df = df[df["set"] == split].copy()

    if len(df) == 0:
        logger.warning(f"No data for split '{split}', skipping")
        return

    logger.info(f"Saving {len(df):,} records to {output_path}")

    # Convert mols to JSON strings for storage
    logger.info("Converting molecules to JSON...")
    mol_jsons = [
        mol_to_json(m) for m in tqdm(df["mol"].values, desc="Converting mols")
    ]

    # Build extension arrays in batches to avoid memory issues
    batch_size = 10000
    n_records = len(df)
    writer = None

    try:
        for start_idx in tqdm(
            range(0, n_records, batch_size), desc="Writing batches"
        ):
            end_idx = min(start_idx + batch_size, n_records)

            batch_mol_jsons = mol_jsons[start_idx:end_idx]
            batch_paths = df["shortest_paths"].iloc[start_idx:end_idx].tolist()
            batch_mol_ids = df["mol_id"].iloc[start_idx:end_idx].values
            batch_ri = df["experimental_ri"].iloc[start_idx:end_idx].values
            batch_set = df["set"].iloc[start_idx:end_idx].values

            # Create storage arrays
            mol_storage = pa.array(batch_mol_jsons, type=pa.string())
            mol_array = pa.ExtensionArray.from_storage(
                MolArrowType(), mol_storage
            )

            path_storage = pa.array(batch_paths, type=pa.string())
            path_array = pa.ExtensionArray.from_storage(
                PathArrowType(), path_storage
            )

            # Build batch table
            batch_table = pa.table(
                {
                    "id": pa.array(batch_mol_ids, type=pa.int64()),
                    "mol": mol_array,
                    "experimental_ri": pa.array(batch_ri, type=pa.float64()),
                    "shortest_paths": path_array,
                    "set": pa.array(batch_set, type=pa.string()),
                }
            )

            # Initialize writer on first batch
            if writer is None:
                writer = pq.ParquetWriter(output_path, batch_table.schema)

            writer.write_table(batch_table)
    finally:
        if writer is not None:
            writer.close()

    logger.info(f"Saved to {output_path}")


@click.command()
@click.option(
    "--data-dir",
    required=True,
    type=click.Path(exists=True),
    help="Path to NIST data directory (e.g., NIST2023_GCMS_main)",
)
@click.option(
    "--output-dir",
    default="./airi_data",
    type=click.Path(),
    help="Directory to save processed parquet files",
)
@click.option(
    "--splits-file",
    default="random_no_xeno_aas_deduplicated.tsv",
    help="Name of splits file in splits/ directory",
)
@click.option(
    "--ri-type",
    type=click.Choice(["StdNP", "SemiStdNP", "StdPolar"]),
    default=None,
    help="Specific RI type to use (default: unified/preferred)",
)
@click.option(
    "--max-path-length",
    default=5,
    type=int,
    help="Maximum path length for shortest path computation",
)
@click.option(
    "--num-workers",
    default=8,
    type=int,
    help="Number of parallel workers for processing",
)
@click.option(
    "--single-file",
    is_flag=True,
    help="Save all splits to a single file instead of separate files",
)
def main(
    data_dir: str,
    output_dir: str,
    splits_file: str,
    ri_type: str,
    max_path_length: int,
    num_workers: int,
    single_file: bool,
):
    """Prepare NIST data for AIRI model training.

    Converts NIST metadata with retention index values into parquet format with
    shortest path features required by the AIRI model.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("AIRI Data Preparation")
    logger.info("=" * 60)
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Splits file: {splits_file}")
    logger.info(f"RI type: {ri_type or 'unified'}")
    logger.info(f"Max path length: {max_path_length}")

    # Load NIST data
    df = load_nist_data(data_dir, splits_file, ri_type)

    # Process molecules
    df = process_molecules(df, max_path_length, num_workers)

    # Save to parquet
    ri_suffix = f"_{ri_type}" if ri_type else ""

    if single_file:
        save_to_parquet(df, output_path / f"airi_data{ri_suffix}.parquet")
    else:
        # Save separate files for each split
        for split, masskit_split in [
            ("train", "train"),
            ("val", "valid"),
            ("test", "test"),
        ]:
            # Rename split for masskit compatibility
            df_split = df[df["set"] == split].copy()
            df_split["set"] = masskit_split
            save_to_parquet(
                df_split,
                output_path / f"airi_{masskit_split}{ri_suffix}.parquet",
                split=masskit_split,
            )

    logger.info("\n" + "=" * 60)
    logger.info("Data preparation complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
