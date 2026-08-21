"""Add a mw column (+ sort_idx_mw / sorted_mw) to a PubChem prediction HDF5.

Joins by SMILES (matches add_ri_to_hdf5.py's join key) when the HDF5 has a
smiles column, otherwise falls back to InChIKey-14 (e.g. MassFormer's
columnar conversion, which only carries inchikey14).
Safe to re-run: skips if mw/sort_idx_mw already exist unless --overwrite.

Usage:
    uv run examples/scripts/evaluation/add_mw_to_hdf5.py \
        --hdf5 results/inference/pubchem_predictions_rerun_260630.hdf5
"""

import argparse
import logging
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

log = logging.getLogger(__name__)

TSV_PATH = "data/PubChem/PubChem_filtered.tsv"
CHUNK_SIZE = 500_000


def build_key_to_mw(tsv_path: str, key_col: str) -> dict[str, float]:
    """Load PubChem_filtered.tsv, return {key: mw} for lookup by SMILES or
    InChIKey-14 (matches add_ri_to_hdf5.py's join key, robust to row-order
    drift)."""
    import pyarrow.csv as pa_csv

    log.info(f"Loading MW TSV via pyarrow, joining on {key_col}...")
    table = pa_csv.read_csv(
        tsv_path,
        parse_options=pa_csv.ParseOptions(delimiter="\t"),
        convert_options=pa_csv.ConvertOptions(
            include_columns=["SMILES", "InChIKey", "MW"]
        ),
    )
    df = table.to_pandas()
    if key_col == "inchikey14":
        df["inchikey14"] = df["InChIKey"].str[:14]
        join_col = "inchikey14"
    else:
        join_col = "SMILES"
    df = df.drop_duplicates(subset=join_col)
    log.info(f"Loaded {len(df):,} unique {key_col} entries.")
    return dict(zip(df[join_col].values, df["MW"].values.astype(np.float32)))


def add_mw_dataset(hdf5_path: str, tsv_path: str, overwrite: bool) -> None:
    with h5py.File(hdf5_path, "a") as f:
        if "mw" in f and "sort_idx_mw" in f and not overwrite:
            log.info(
                "mw + sort_idx_mw already present. Use --overwrite to redo."
            )
            return

        n_rows = f["intensities"].shape[0]
        log.info(f"HDF5 has {n_rows:,} rows.")

        # Join on SMILES if present (row-order-independent, exact match
        # already verified against PubChem_filtered.tsv); fall back to
        # InChIKey-14 for HDF5s that don't carry a smiles column (e.g.
        # MassFormer's columnar conversion, which only has inchikey14).
        has_smiles = "smiles" in f
        src_key = "smiles" if has_smiles else "inchikey14"
        key_to_mw = build_key_to_mw(tsv_path, src_key)
        mw = np.full(n_rows, np.nan, dtype=np.float32)

        for start in tqdm(range(0, n_rows, CHUNK_SIZE), desc="Mapping MW"):
            end = min(start + CHUNK_SIZE, n_rows)
            raw_keys = f[src_key][start:end]
            keys_chunk = [
                x.decode() if isinstance(x, bytes) else x for x in raw_keys
            ]
            mw[start:end] = [key_to_mw.get(k, np.nan) for k in keys_chunk]

        coverage = np.isfinite(mw).mean() * 100
        log.info(f"mw: {coverage:.1f}% coverage")

        mw_for_sort = mw.copy()
        mw_for_sort[np.isnan(mw_for_sort)] = np.inf
        idx = np.argsort(mw_for_sort, kind="stable").astype(np.int32)
        sorted_mw = mw[idx]

        for key, data in [
            ("mw", mw),
            ("sort_idx_mw", idx),
            ("sorted_mw", sorted_mw),
        ]:
            if key in f:
                del f[key]
            f.create_dataset(
                key,
                data=data,
                compression="gzip",
                compression_opts=1,
                chunks=(min(1_000_000, data.shape[0]),),
            )
            log.info(f"  Wrote {key}  shape={data.shape}  dtype={data.dtype}")

        f.flush()
    log.info("Done.")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5", required=True)
    parser.add_argument("--tsv", default=TSV_PATH)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not Path(args.hdf5).exists():
        raise FileNotFoundError(f"HDF5 not found: {args.hdf5}")
    if not Path(args.tsv).exists():
        raise FileNotFoundError(f"TSV not found: {args.tsv}")

    add_mw_dataset(args.hdf5, args.tsv, args.overwrite)


if __name__ == "__main__":
    main()
