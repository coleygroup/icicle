"""Add an inchikey14 dataset to a PubChem prediction HDF5, in place.

Single file handle (unlike add_inchikeys_to_hdf5.py, which merges multiple
input files into a new output file) — streams the smiles column in chunks,
computes InChIKey-14 with RDKit (multiprocessing), and appends the result
directly into the source file. No second copy of the file is created.
Safe to re-run: skips if inchikey14 already exists unless --overwrite.

Usage:
    uv run examples/scripts/evaluation/add_inchikeys_inplace.py \
        --hdf5 results/inference/pubchem_predictions_rerun_260630.hdf5
"""

import argparse
import logging
import multiprocessing
from pathlib import Path

import h5py
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import inchi
from tqdm import tqdm

RDLogger.DisableLog("rdApp.*")
log = logging.getLogger(__name__)

CHUNK_SIZE = 200_000


def compute_ik14(smi) -> bytes:
    """Convert a SMILES (bytes or str) to an InChIKey-14 byte string."""
    try:
        if isinstance(smi, (bytes, bytearray)):
            smi = smi.decode("utf-8")
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return b""
        ik = inchi.MolToInchiKey(mol)
        return ik[:14].encode("ascii") if ik else b""
    except Exception:
        return b""


def add_inchikeys(hdf5_path: str, overwrite: bool) -> None:
    num_cores = max(1, multiprocessing.cpu_count() - 1)
    with h5py.File(hdf5_path, "a") as f:
        n = f["smiles"].shape[0]
        if "inchikey14" in f and not overwrite:
            written = f["inchikey14"].attrs.get("rows_written", n)
            if written >= n:
                log.info(
                    "inchikey14 already fully present. Use --overwrite to redo."
                )
                return
            start = int(written)
            log.info(f"Resuming from row {start:,}/{n:,}")
        else:
            if "inchikey14" in f:
                del f["inchikey14"]
            f.create_dataset(
                "inchikey14",
                shape=(n,),
                dtype="S14",
                compression="lzf",
                chunks=True,
            )
            start = 0

        pool = multiprocessing.Pool(num_cores)
        try:
            for chunk_start in tqdm(
                range(start, n, CHUNK_SIZE), desc="Computing IK14"
            ):
                chunk_end = min(chunk_start + CHUNK_SIZE, n)
                smiles_chunk = f["smiles"][chunk_start:chunk_end]
                ik14 = pool.map(compute_ik14, smiles_chunk, chunksize=1000)
                f["inchikey14"][chunk_start:chunk_end] = np.array(
                    ik14, dtype="S14"
                )
                f["inchikey14"].attrs["rows_written"] = chunk_end
                f.flush()
        finally:
            pool.terminate()
            pool.join()

    log.info("Done.")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not Path(args.hdf5).exists():
        raise FileNotFoundError(f"HDF5 not found: {args.hdf5}")

    add_inchikeys(args.hdf5, args.overwrite)


if __name__ == "__main__":
    main()
