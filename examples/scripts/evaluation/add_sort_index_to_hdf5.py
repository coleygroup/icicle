#!/usr/bin/env python
"""Add sorted RI and MW index datasets to the PubChem master HDF5.

For each RI column type (StdNP, SemiStdNP, StdPolar), writes two new datasets:
  - sort_idx_{ri_type}  : int32 argsort indices (NaNs pushed to end)
  - sorted_ri_{ri_type} : float32 sorted RI values (NaNs at end)

Also writes a nominal-mass index (from the PubChem parquet, joined on inchikey14):
  - sort_idx_mw  : int32 argsort indices
  - sorted_mw    : int32 sorted nominal_mass values (missing → 0 at end)

This enables O(log N) binary-search window extraction in the eval script,
replacing the full sequential scan for RI-windowed and MW-windowed candidate sets.

Usage
-----
uv run examples/scripts/evaluation/add_sort_index_to_hdf5.py \
    --hdf5 <path/to/pubchem_predictions.hdf5> \
    --parquet data/PubChem/PubChem_filtered_with_ri_new_random_inchikey_cache.parquet

Add --force to recompute and overwrite existing index datasets.
"""

import argparse
import logging

import h5py
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
RI_TYPES = ["StdNP", "SemiStdNP", "StdPolar"]


def add_sort_indices(
    hdf5_path: str, parquet_path: str | None = None, force: bool = False
) -> None:
    """Compute and store argsort + sorted arrays for all RI types and MW."""
    with h5py.File(hdf5_path, "a") as f:
        for ri_type in RI_TYPES:
            ri_key = f"ri_{ri_type}"
            idx_key = f"sort_idx_{ri_type}"
            sorted_ri_key = f"sorted_ri_{ri_type}"

            if ri_key not in f:
                log.warning(f"{ri_key} not in HDF5 — skipping")
                continue

            if idx_key in f and not force:
                log.info(
                    f"{idx_key} already exists — skipping (use --force to recompute)"
                )
                continue

            log.info(
                f"Computing argsort for {ri_key} ({f[ri_key].shape[0]:,} rows)..."
            )
            ri = f[ri_key][:]

            # NaN-safe argsort: put NaNs at the end by replacing with +inf
            ri_for_sort = ri.copy()
            ri_for_sort[np.isnan(ri_for_sort)] = np.inf
            idx = np.argsort(ri_for_sort, kind="stable").astype(np.int32)
            sorted_ri = ri[idx]

            for key, data in [(idx_key, idx), (sorted_ri_key, sorted_ri)]:
                if key in f:
                    del f[key]
                f.create_dataset(
                    key,
                    data=data,
                    compression="gzip",
                    compression_opts=1,
                    chunks=(min(1_000_000, data.shape[0]),),
                )
                log.info(
                    f"  Wrote {key}  shape={data.shape}  dtype={data.dtype}"
                )

        if parquet_path is not None:
            _add_mw_index(f, parquet_path, force)

    log.info("Done.")


def _add_mw_index(f: h5py.File, parquet_path: str, force: bool) -> None:
    """Join nominal_mass from parquet onto HDF5 inchikey14, write sort_idx_mw /
    sorted_mw."""
    if "sort_idx_mw" in f and not force:
        log.info(
            "sort_idx_mw already exists — skipping (use --force to recompute)"
        )
        return

    log.info(f"Loading MW from parquet: {parquet_path}")
    ref = pd.read_parquet(parquet_path, columns=["InChIKey", "nominal_mass"])
    ref["ik14"] = ref["InChIKey"].str[:14]
    ref = ref.drop_duplicates("ik14").set_index("ik14")["nominal_mass"]
    log.info(f"  {len(ref):,} unique inchikey14 in parquet")

    log.info(
        f"Reading inchikey14 from HDF5 ({f['inchikey14'].shape[0]:,} rows)..."
    )
    raw_ik = f["inchikey14"][:]
    ik14 = np.array(
        [v.decode() if isinstance(v, bytes) else v for v in raw_ik]
    )

    mw = ref.reindex(ik14).values.astype("float32")  # NaN where not found
    log.info(f"  MW assigned: {np.isfinite(mw).sum():,} / {len(mw):,}")

    # NaN-safe argsort: NaNs at end
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


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hdf5", required=True, help="Path to master PubChem HDF5"
    )
    parser.add_argument(
        "--parquet",
        default=None,
        help="PubChem parquet with nominal_mass (for MW index)",
    )
    parser.add_argument(
        "--force", action="store_true", help="Recompute even if indices exist"
    )
    args = parser.parse_args()
    add_sort_indices(args.hdf5, parquet_path=args.parquet, force=args.force)


if __name__ == "__main__":
    main()
