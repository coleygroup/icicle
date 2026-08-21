"""Convert MassFormer PubChem chunk HDF5s to columnar format for retrieval
eval.

MassFormer chunks: {CID: {mz_bins: [750], predicted_intensities: [750]}}
Target format (matches NEIMS/ICICLE): columnar intensities[N,750], inchikey14[N],
ri_StdNP[N], ri_SemiStdNP[N], ri_StdPolar[N], valid[N].

Usage:
    uv run examples/scripts/evaluation/convert_massformer_pubchem_chunks.py
    uv run examples/scripts/evaluation/convert_massformer_pubchem_chunks.py \
        --chunk-glob "massformer_scaffold_s1_pubchem_full.chunk*.hdf5" \
        --output baselines/massformer/results/predictions/massformer_scaffold_s1_pubchem_full_columnar.hdf5 \
        --temp-dir baselines/massformer/results/predictions/tmp_chunks_scaffold
"""

import argparse
import multiprocessing as mp
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

CHUNK_DIR = Path("baselines/massformer/results/predictions")
CID_MAP_TSV = Path(
    "data/PubChem/PubChem_filtered_with_ri_new_random_inchikey.tsv"
)
N_BINS = 750
RI_COLS = ["ri_StdNP", "ri_SemiStdNP", "ri_StdPolar"]
N_WORKERS = 16

# Shared numpy arrays — set in main before fork, inherited by workers via CoW
_CID_INDEX: np.ndarray = None  # int64 CID values
_IK14: np.ndarray = None  # S14 inchikey14
_RI_STDNP: np.ndarray = None
_RI_SEMI: np.ndarray = None
_RI_POLAR: np.ndarray = None
TEMP_DIR: Path = None  # set in main before fork, inherited by workers via CoW


def _sorted_chunks(chunk_dir: Path, glob: str) -> list[Path]:
    files = list(chunk_dir.glob(glob))
    return sorted(files, key=lambda p: int(p.stem.split("chunk")[-1]))


def _process_chunk(chunk_path: Path) -> Path:
    """Worker: uses inherited numpy arrays (CoW), no IO for CID map."""
    temp_out = TEMP_DIR / (chunk_path.stem + "_col.hdf5")
    if temp_out.exists():
        return temp_out

    with h5py.File(chunk_path, "r") as fin:
        cids = np.array([int(k) for k in fin.keys()], dtype=np.int64)
        intensities = np.stack(
            [fin[str(cid)]["predicted_intensities"][:] for cid in cids], axis=0
        ).astype(np.float32)

    # Binary search into sorted CID index
    idx = np.searchsorted(_CID_INDEX, cids)
    idx = np.clip(idx, 0, len(_CID_INDEX) - 1)
    match = _CID_INDEX[idx] == cids

    ik14 = np.where(match, _IK14[idx], b"")
    ri_stdnp = np.where(match, _RI_STDNP[idx], np.nan).astype(np.float32)
    ri_semi = np.where(match, _RI_SEMI[idx], np.nan).astype(np.float32)
    ri_polar = np.where(match, _RI_POLAR[idx], np.nan).astype(np.float32)
    valid = match & (_IK14[idx] != b"")

    with h5py.File(temp_out, "w") as f:
        f.create_dataset("intensities", data=intensities, compression="lzf")
        f.create_dataset(
            "inchikey14", data=ik14, dtype="S14", compression="lzf"
        )
        f.create_dataset("ri_StdNP", data=ri_stdnp, compression="lzf")
        f.create_dataset("ri_SemiStdNP", data=ri_semi, compression="lzf")
        f.create_dataset("ri_StdPolar", data=ri_polar, compression="lzf")
        f.create_dataset("valid", data=valid, dtype=bool, compression="lzf")

    return temp_out


def merge_temp_files(temp_files: list[Path], output: Path) -> None:
    """Concatenate temp columnar HDF5s into final output."""
    output.parent.mkdir(parents=True, exist_ok=True)
    keys = [
        "intensities",
        "inchikey14",
        "ri_StdNP",
        "ri_SemiStdNP",
        "ri_StdPolar",
        "valid",
    ]

    with h5py.File(output, "w") as fout:
        with h5py.File(temp_files[0], "r") as f0:
            total = sum(
                h5py.File(p, "r")["intensities"].shape[0] for p in temp_files
            )
            fout.create_dataset(
                "intensities",
                shape=(total, N_BINS),
                dtype=np.float32,
                compression="lzf",
                chunks=(10000, N_BINS),
            )
            for k in keys[1:]:
                fout.create_dataset(
                    k,
                    shape=(total,),
                    dtype=f0[k].dtype,
                    compression="lzf",
                    chunks=True,
                )

        offset = 0
        for p in tqdm(temp_files, desc="Merging"):
            with h5py.File(p, "r") as f:
                n = f["intensities"].shape[0]
                for k in keys:
                    fout[k][offset : offset + n] = f[k][:]
                offset += n

    print(f"\nDone. Output: {output}  ({offset:,} rows total)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chunk-glob",
        default="massformer_random_s2_pubchem_full.chunk*.hdf5",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "baselines/massformer/results/predictions/massformer_random_s2_pubchem_full_columnar.hdf5"
        ),
    )
    parser.add_argument(
        "--temp-dir",
        type=Path,
        default=Path("baselines/massformer/results/predictions/tmp_chunks"),
    )
    args = parser.parse_args()

    os.chdir(Path(__file__).parents[3])
    TEMP_DIR = args.temp_dir

    chunks = _sorted_chunks(CHUNK_DIR, args.chunk_glob)
    print(f"Found {len(chunks)} chunks.")
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading CID map once ...")
    df = pd.read_csv(
        CID_MAP_TSV,
        sep="\t",
        usecols=["ID", "InChIKey"] + RI_COLS,
        dtype={"ID": np.int64},
    )
    df["inchikey14"] = df["InChIKey"].str[:14]
    df = df.drop(columns=["InChIKey"]).sort_values("ID")

    # Store as module-level numpy arrays — fork workers inherit via CoW, zero serialization
    _CID_INDEX = df["ID"].values
    _IK14 = (
        df["inchikey14"]
        .str.encode("ascii")
        .str[:14]
        .fillna(b"")
        .values.astype("S14")
    )
    _RI_STDNP = df["ri_StdNP"].values.astype(np.float32)
    _RI_SEMI = df["ri_SemiStdNP"].values.astype(np.float32)
    _RI_POLAR = df["ri_StdPolar"].values.astype(np.float32)
    del df
    print("  Done. Spawning workers ...")

    with mp.Pool(N_WORKERS) as pool:
        temp_files = list(
            tqdm(
                pool.imap(_process_chunk, chunks, chunksize=1),
                total=len(chunks),
                desc="Chunks",
            )
        )

    print("Merging ...")
    merge_temp_files(sorted(temp_files), args.output)
