#!/usr/bin/env python
"""Add a TRUE (structure-derived) heavy-atom-count index to a PubChem
prediction HDF5, replacing the model-predicted-from-spectrum version.

Motivation: unlike a spectrum (unmeasured for ~all of PubChem), the
heavy-atom count IS exactly knowable for every PubChem candidate directly
from its own structure (the SMILES is already stored in the HDF5's `smiles`
dataset) -- there's no reason to use a noisy spectrum-based prediction on
the candidate side when the ground truth is one RDKit call away. This
mirrors add_sort_index_to_hdf5.py's MW index, which joins the TRUE
nominal_mass from PubChem's own parquet (not a value predicted from any
spectrum).

The query side (NIST test molecules) still uses the model's prediction from
the query's own PREDICTED spectrum (see evaluate_heavy_atom_window /
evaluate_heavy_atom_then_ri in pubchem_global_retrieval.py) -- a query's
true spectrum-derived proxy (highest_peak_mz) is used for MW because NIST
queries have real measured spectra, but heavy-atom count isn't directly
observable from a spectrum at all, so the trained predictor is the query
side's only option, exactly mirroring how MW's real deployment scenario
would only ever have a predicted/observed spectrum for a truly unknown
molecule.

Writes (overwrites any previous model-predicted heavy_atom_pred/sort_idx_
heavy_atom/sorted_heavy_atom from the superseded spectrum-based predictor):
  - heavy_atom_true   : float32 true heavy-atom count per row (NaN if SMILES
                        unparseable)
  - sort_idx_heavy_atom  : int32 argsort indices (NaNs pushed to end)
  - sorted_heavy_atom    : float32 sorted heavy_atom_true values

Usage
-----
uv run examples/scripts/evaluation/add_true_heavy_atom_index_to_hdf5.py \
    --hdf5 results/inference/pubchem_predictions_rerun_260710.hdf5

Add --force to recompute and overwrite existing index datasets.
"""

import argparse
import logging
import multiprocessing as mp

import h5py
import numpy as np
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

log = logging.getLogger(__name__)

CHUNK_SIZE = 500_000
NUM_WORKERS = 48


def _true_heavy_atom_counts(smiles_chunk: np.ndarray) -> np.ndarray:
    out = np.full(len(smiles_chunk), np.nan, dtype=np.float32)
    for i, smi in enumerate(smiles_chunk):
        smi = smi.decode() if isinstance(smi, bytes) else smi
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            out[i] = mol.GetNumHeavyAtoms()
    return out


def add_true_heavy_atom_index(
    hdf5_path: str,
    force: bool = False,
    chunk_size: int = CHUNK_SIZE,
    num_workers: int = NUM_WORKERS,
) -> None:
    with h5py.File(hdf5_path, "a") as f:
        if "heavy_atom_true" in f and not force:
            log.info(
                "heavy_atom_true already exists — skipping (use --force to recompute)"
            )
            return

        total = f["smiles"].shape[0]
        log.info(
            f"Computing true heavy-atom count for {total:,} rows "
            f"({num_workers} worker processes)..."
        )
        true_ha = np.empty(total, dtype=np.float32)
        sub_chunk = chunk_size // num_workers
        with mp.Pool(num_workers) as pool:
            for start in range(0, total, chunk_size):
                end = min(start + chunk_size, total)
                smi_chunk = f["smiles"][start:end]
                sub_chunks = [
                    smi_chunk[i : i + sub_chunk]
                    for i in range(0, len(smi_chunk), sub_chunk)
                ]
                results = pool.map(_true_heavy_atom_counts, sub_chunks)
                true_ha[start:end] = np.concatenate(results)
                if start % (chunk_size * 10) == 0:
                    log.info(f"  {end:,}/{total:,}")

        n_finite = np.isfinite(true_ha).sum()
        log.info(
            f"True heavy-atom count: {n_finite:,}/{total:,} parsed  "
            f"mean={np.nanmean(true_ha):.2f}  std={np.nanstd(true_ha):.2f}  "
            f"min={np.nanmin(true_ha):.2f}  max={np.nanmax(true_ha):.2f}"
        )

        # NaN-safe argsort: NaNs (unparseable SMILES) pushed to the end.
        ha_for_sort = true_ha.copy()
        ha_for_sort[np.isnan(ha_for_sort)] = np.inf
        idx = np.argsort(ha_for_sort, kind="stable").astype(np.int32)
        sorted_ha = true_ha[idx]

        for key, data in [
            ("heavy_atom_true", true_ha),
            ("sort_idx_heavy_atom", idx),
            ("sorted_heavy_atom", sorted_ha),
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

        # Drop the old model-predicted column so nothing accidentally reads
        # stale spectrum-derived values under a similar name.
        if "heavy_atom_pred" in f:
            del f["heavy_atom_pred"]
            log.info(
                "  Removed stale heavy_atom_pred (spectrum-predicted, superseded)"
            )

    log.info("Done.")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hdf5", required=True, help="Path to PubChem prediction HDF5"
    )
    parser.add_argument(
        "--force", action="store_true", help="Recompute even if indices exist"
    )
    args = parser.parse_args()
    add_true_heavy_atom_index(args.hdf5, force=args.force)


if __name__ == "__main__":
    main()
