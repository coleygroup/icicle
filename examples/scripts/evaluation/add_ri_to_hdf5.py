"""Add RI columns (ri_StdNP, ri_SemiStdNP, ri_StdPolar) to the master PubChem
HDF5.

RI values are matched by a SMILES-keyed join (TSV rows are not assumed to be
row-aligned with the HDF5).
Safe to re-run: skips datasets that already exist unless --overwrite is passed.

Usage:
    uv run examples/scripts/evaluation/add_ri_to_hdf5.py --hdf5 <path/to/pubchem_predictions.hdf5>
    uv run examples/scripts/evaluation/add_ri_to_hdf5.py --hdf5 <path/to/pubchem_predictions.hdf5> --overwrite
"""

import argparse
import logging
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

log = logging.getLogger(__name__)

# Placeholder example path only, not a real file on disk -- callers should
# always pass --hdf5 explicitly.
HDF5_PATH = "data/PubChem/pubchem_predictions_with_ik14_master.hdf5"
TSV_PATH = "data/PubChem/AIRI_inference_output_full.tsv"
RI_COLS = ["ri_StdNP", "ri_SemiStdNP", "ri_StdPolar"]
CHUNK_SIZE = 500_000


def build_smiles_lookup(tsv_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load TSV, return (smiles array, ri_values array [N, 3]) for vectorized
    lookup."""
    import pyarrow.csv as pa_csv

    log.info("Loading RI TSV via pyarrow (fast columnar read)...")
    table = pa_csv.read_csv(
        tsv_path,
        parse_options=pa_csv.ParseOptions(delimiter="\t"),
        convert_options=pa_csv.ConvertOptions(
            include_columns=["smiles"] + RI_COLS
        ),
    )
    df = (
        table.to_pandas()
        .drop_duplicates(subset="smiles")
        .reset_index(drop=True)
    )
    smi_arr = df["smiles"].values
    ri_arr = df[RI_COLS].values.astype(np.float32)
    log.info(f"Loaded {len(smi_arr):,} unique SMILES entries.")
    return smi_arr, ri_arr


def add_ri_datasets(hdf5_path: str, tsv_path: str, overwrite: bool) -> None:
    """Write ri_* float32 datasets into the HDF5 file."""
    with h5py.File(hdf5_path, "a") as f:
        cols_to_write = (
            RI_COLS if overwrite else [c for c in RI_COLS if c not in f]
        )

        if not cols_to_write:
            log.info(
                "All RI datasets already present. Use --overwrite to replace."
            )
            return

        n_rows = f["intensities"].shape[0]
        log.info(f"HDF5 has {n_rows:,} rows. Writing: {cols_to_write}")

        smi_arr, ri_arr = build_smiles_lookup(tsv_path)
        smi_to_idx = {smi: i for i, smi in enumerate(smi_arr)}
        col_idx = {col: j for j, col in enumerate(RI_COLS)}
        ri_arrays = {
            col: np.full(n_rows, np.nan, dtype=np.float32)
            for col in cols_to_write
        }

        for start in tqdm(range(0, n_rows, CHUNK_SIZE), desc="Mapping RI"):
            end = min(start + CHUNK_SIZE, n_rows)
            raw_smiles = f["smiles"][start:end]
            smiles_chunk = [
                x.decode() if isinstance(x, bytes) else x for x in raw_smiles
            ]
            tsv_indices = np.array(
                [smi_to_idx.get(s, -1) for s in smiles_chunk]
            )
            found = tsv_indices >= 0
            for col in cols_to_write:
                ri_arrays[col][start:end][found] = ri_arr[
                    tsv_indices[found], col_idx[col]
                ]

        for col in cols_to_write:
            if col in f:
                del f[col]
            f.create_dataset(
                col, data=ri_arrays[col], compression="lzf", chunks=True
            )
            coverage = np.isfinite(ri_arrays[col]).mean() * 100
            log.info(f"  {col}: written ({coverage:.1f}% coverage)")

        f.flush()
    log.info("Done.")


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5", default=HDF5_PATH)
    parser.add_argument("--tsv", default=TSV_PATH)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not Path(args.hdf5).exists():
        raise FileNotFoundError(f"HDF5 not found: {args.hdf5}")
    if not Path(args.tsv).exists():
        raise FileNotFoundError(f"TSV not found: {args.tsv}")

    add_ri_datasets(args.hdf5, args.tsv, args.overwrite)


if __name__ == "__main__":
    main()
