"""Large-scale batched EIMS inference across multiple GPUs.

Usage
-----
# Minimal (full enumeration, 8 GPUs):
uv run src/icicle/batch_infer.py \
    --intensity-predictor /path/to/ckpt.ckpt \
    --input compounds.csv \
    --smiles-col smiles \
    --output predictions.hdf5 \
    --num-gpus 8

Output HDF5 layout
-----
results.hdf5
├── smiles          (N,)         variable-length UTF-8 string
├── intensities     (N, n_bins)  float32   -- zeros for invalid SMILES
├── num_fragments   (N,)         int32     -- 0 for invalid
└── valid           (N,)         bool
"""

import argparse
import json
import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional

import h5py
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
from tqdm import tqdm

from icicle.models.eims_predictor import (
    EIMSPredictorFromFullEnumeration,
    EIMSPredictorWithFragmentGenerator,
)
from icicle.models.intensity_model import IntensityPredictor

logging.basicConfig(
    level=logging.ERROR,  # only show errors on stdout; warnings go to per-worker log files
    format="%(asctime)s [rank %(rank)s] %(levelname)s - %(message)s",
)

# Set by the preemption signal handler inside each worker process.
_shutdown_requested = threading.Event()


def _handle_preemption_signal(signum, frame):
    _shutdown_requested.set()


def _infer_sep(path: str, explicit_sep: Optional[str]) -> str:
    if explicit_sep is not None:
        return explicit_sep
    if path.endswith(".tsv") or path.endswith(".txt"):
        return "\t"
    return ","


def _load_smiles(
    path: str,
    smiles_col: str,
    no_header: bool = False,
    sep: Optional[str] = None,
    smiles_col_idx: int = 0,
) -> List[str]:
    sep = _infer_sep(path, sep)
    if no_header:
        df = pd.read_csv(path, sep=sep, header=None, dtype=str)
        smiles = df.iloc[:, smiles_col_idx].fillna("").tolist()
    elif not smiles_col:
        df = pd.read_csv(path, sep=sep, dtype=str)
        smiles = df.iloc[:, smiles_col_idx].fillna("").tolist()
    else:
        df = pd.read_csv(path, sep=sep, usecols=[smiles_col], dtype=str)
        smiles = df[smiles_col].fillna("").tolist()
    logging.info(
        f"Loaded {len(smiles):,} SMILES from {path}", extra={"rank": "main"}
    )
    return smiles


def _load_smiles_slice(
    path: str,
    sep: str,
    no_header: bool,
    smiles_col: str,
    smiles_col_idx: int,
    row_start: int,
    row_end: int,
) -> List[str]:
    """Load rows [row_start, row_end) without reading the full file."""
    nrows = row_end - row_start
    if no_header:
        df = pd.read_csv(
            path,
            sep=sep,
            header=None,
            dtype=str,
            skiprows=row_start,
            nrows=nrows,
        )
        return df.iloc[:, smiles_col_idx].fillna("").tolist()
    # File has header — read column names, then skip to row_start
    if row_start > 0:
        col_names = pd.read_csv(path, sep=sep, nrows=0).columns.tolist()
        df = pd.read_csv(
            path,
            sep=sep,
            header=None,
            names=col_names,
            dtype=str,
            skiprows=row_start + 1,
            nrows=nrows,
        )
    else:
        df = pd.read_csv(path, sep=sep, dtype=str, nrows=nrows)
    if smiles_col and smiles_col in df.columns:
        return df[smiles_col].fillna("").tolist()
    return df.iloc[:, smiles_col_idx].fillna("").tolist()


def _checkpoint_progress_path(shard_path: str) -> str:
    return shard_path + ".progress.json"


def _read_resume_state(shard_path: str) -> int:
    """Return number of molecules already durably written to shard (0 if none).

    Reads the row count directly from the HDF5 file itself rather than a
    separately-written progress.json. A prior version wrote the HDF5 append and
    the progress.json update as two separate steps; a crash between them
    (observed routinely in production, with ~1,000 restarts/GPU on an 8-GPU
    multi-day run) left progress.json under-reporting how many rows were
    actually committed. On resume, the worker would then re-process and re-
    append already-written molecules, permanently shifting every subsequent
    row's smiles-to-intensity alignment for the rest of the shard. The HDF5's
    own row count is always consistent with the data that's actually there the
    instant the write completes -- there's no second file that can fall out of
    sync with it.
    """
    if not os.path.exists(shard_path):
        return 0
    with h5py.File(shard_path, "r") as f:
        if "smiles" not in f:
            return 0
        return int(f["smiles"].shape[0])


def _write_checkpoint(
    shard_path: str,
    smiles_buf: List[str],
    intensities_buf: List[np.ndarray],
    num_frags_buf: List[int],
    valid_buf: List[bool],
    n_bins: int,
    already_processed: int,
    fragment_masses_buf: Optional[List[Optional[np.ndarray]]] = None,
):
    """Append buffered results to the shard HDF5 and update the progress
    file."""
    buf_size = len(smiles_buf)
    if buf_size == 0:
        return

    intensities_arr = np.zeros((buf_size, n_bins), dtype=np.float32)
    for i, arr in enumerate(intensities_buf):
        if arr is not None:
            intensities_arr[i] = arr

    str_dt = h5py.special_dtype(vlen=str)
    save_frags = fragment_masses_buf is not None
    if save_frags:
        vlen_f32 = h5py.vlen_dtype(np.float32)
        frag_arr = np.empty(buf_size, dtype=object)
        for i, fm in enumerate(fragment_masses_buf):
            frag_arr[i] = (
                fm if fm is not None else np.array([], dtype=np.float32)
            )

    mode = "a" if os.path.exists(shard_path) else "w"
    with h5py.File(shard_path, mode) as f:
        if "smiles" not in f:
            # First checkpoint: create resizable datasets
            f.create_dataset(
                "smiles",
                data=np.array(smiles_buf, dtype=object),
                dtype=str_dt,
                maxshape=(None,),
            )
            f.create_dataset(
                "intensities",
                data=intensities_arr,
                compression="gzip",
                compression_opts=4,
                chunks=(min(1000, buf_size), n_bins),
                maxshape=(None, n_bins),
            )
            f.create_dataset(
                "num_fragments",
                data=np.array(num_frags_buf, dtype=np.int32),
                maxshape=(None,),
            )
            f.create_dataset(
                "valid",
                data=np.array(valid_buf, dtype=bool),
                maxshape=(None,),
            )
            if save_frags:
                f.create_dataset(
                    "fragment_masses",
                    data=frag_arr,
                    dtype=vlen_f32,
                    maxshape=(None,),
                )
        else:
            # Subsequent checkpoints: resize and append
            cur = f["smiles"].shape[0]
            new = cur + buf_size
            f["smiles"].resize((new,))
            f["smiles"][cur:new] = np.array(smiles_buf, dtype=object)
            f["intensities"].resize((new, n_bins))
            f["intensities"][cur:new] = intensities_arr
            f["num_fragments"].resize((new,))
            f["num_fragments"][cur:new] = np.array(
                num_frags_buf, dtype=np.int32
            )
            f["valid"].resize((new,))
            f["valid"][cur:new] = np.array(valid_buf, dtype=bool)
            if save_frags:
                f["fragment_masses"].resize((new,))
                f["fragment_masses"][cur:new] = frag_arr

    # Heartbeat file for the workstation script's stall watchdog (detects a
    # wedged process by file size no longer growing) and as a "shard still
    # in progress" marker for the merge script. NOT used for resume
    # correctness -- that reads the row count directly from the HDF5 itself
    # (see _read_resume_state) so it can never fall out of sync with what
    # was actually durably written.
    total_processed = already_processed + buf_size
    with open(_checkpoint_progress_path(shard_path), "w") as f:
        json.dump({"processed": total_processed}, f)


def _worker(
    rank: int,
    input_path: str,
    input_sep: str,
    no_header: bool,
    smiles_col: str,
    smiles_col_idx: int,
    row_start: int,
    row_end: int,
    intensity_ckpt: str,
    fragment_ckpt: Optional[str],
    shard_path: str,
    batch_size: int,
    num_workers: int,
    max_nodes: int,
    threshold: float,
    checkpoint_every: int,
    save_fragments: bool = False,
):
    # Register preemption signal handlers in this spawned process.
    signal.signal(signal.SIGUSR1, _handle_preemption_signal)
    signal.signal(signal.SIGTERM, _handle_preemption_signal)

    # Avoid pidfd_getfd permission errors on restricted systems (e.g. containers).
    # Must be set in each spawned process, not just the parent.
    import torch.multiprocessing as _mp

    _mp.set_sharing_strategy("file_system")

    # Restrict this process (and all its DataLoader workers) to one GPU only.
    # Must happen before any CUDA call.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Suppress RDKit warnings in this process.
    try:
        from rdkit import RDLogger

        RDLogger.DisableLog("rdApp.*")
    except Exception:
        pass

    log = logging.getLogger()
    # Per-worker log file so crashes are readable without screen noise.
    _log_path = str(Path(shard_path).with_suffix(f".gpu{rank}.log"))
    _fh = logging.FileHandler(_log_path, mode="a")
    _fh.setLevel(logging.DEBUG)
    _fh.setFormatter(
        logging.Formatter(
            "%(asctime)s [GPU %(rank)s] %(levelname)s %(message)s",
            defaults={"rank": rank},
        )
    )
    log.addHandler(_fh)
    extra = {"rank": rank}
    device = "cuda:0"  # only one GPU visible in this process

    chunk_total = row_end - row_start

    # Shard exists but no progress (heartbeat) file means the worker
    # finished cleanly on a previous run (progress file is deleted on clean
    # completion in the success path below). Skip -- distinct from
    # already_done == 0, which can also legitimately mean "shard file
    # exists but crashed before its first checkpoint ever landed."
    if os.path.exists(shard_path) and not os.path.exists(
        _checkpoint_progress_path(shard_path)
    ):
        log.info(
            f"Shard {shard_path} already complete, skipping.", extra=extra
        )
        return

    # Resume support: advance row_start past already-processed rows. Read
    # directly from the HDF5's own row count, never from the (now
    # non-authoritative) progress heartbeat file -- see _read_resume_state.
    already_done = _read_resume_state(shard_path)

    effective_start = row_start + already_done
    if already_done > 0:
        log.info(
            f"Resuming from row {effective_start:,} (skipping {already_done:,} already done).",
            extra=extra,
        )

    smiles_chunk = _load_smiles_slice(
        input_path,
        input_sep,
        no_header,
        smiles_col,
        smiles_col_idx,
        effective_start,
        row_end,
    )

    n_total = len(smiles_chunk)
    if n_total == 0:
        log.info("Nothing left to process.", extra=extra)
        return

    log.info(
        f"Loading model on {device} for {n_total:,} molecules.", extra=extra
    )

    if fragment_ckpt:
        model = EIMSPredictorWithFragmentGenerator()
        model.load_from_checkpoint(
            fragment_generator_checkpoint=fragment_ckpt,
            intensity_predictor_checkpoint=intensity_ckpt,
        )
    else:
        ip = IntensityPredictor.load_from_checkpoint(
            intensity_ckpt, map_location="cpu"
        )
        model = EIMSPredictorFromFullEnumeration(
            min_mz=ip.min_mz,
            max_mz=ip.max_mz,
            bin_width=ip.bin_width,
            intensity_predictor=ip,
        )

    model.to(device)
    model.eval()

    # Buffers for current checkpoint window
    smiles_buf: List[str] = []
    intensities_buf: List[Optional[np.ndarray]] = []
    num_frags_buf: List[int] = []
    valid_buf: List[bool] = []
    fragment_masses_buf: List[Optional[np.ndarray]] = (
        [] if save_fragments else None
    )
    n_bins: Optional[int] = None
    batches_since_ckpt = 0

    # Background checkpoint thread.
    # Inference results accumulate in smiles_buf / intensities_buf / … until
    # checkpoint_every batches have been processed, then _flush_checkpoint_async
    # writes them to the shard HDF5 on a background thread so the GPU is never
    # blocked waiting for disk I/O.  The next flush waits for the previous
    # thread to finish before starting a new one (back-pressure if disk is slow).
    _ckpt_thread: Optional[threading.Thread] = None

    def _flush_checkpoint_async(
        s_buf, i_buf, nf_buf, v_buf, n_b, a_done, fm_buf=None
    ):
        """Start a background thread to append buffered results to the shard.

        Waits for any previous checkpoint thread to finish first, providing
        back-pressure if disk writes are slower than inference.  The caller
        passes copies of the buffers so inference can continue immediately
        while the thread writes.
        """
        nonlocal _ckpt_thread
        if _ckpt_thread is not None:
            _ckpt_thread.join()
        _ckpt_thread = threading.Thread(
            target=_write_checkpoint,
            args=(
                shard_path,
                s_buf,
                i_buf,
                nf_buf,
                v_buf,
                n_b,
                a_done,
                fm_buf,
            ),
            daemon=True,
        )
        _ckpt_thread.start()

    t0 = time.time()
    done = 0

    pbar = tqdm(
        total=chunk_total,
        initial=already_done,
        desc=f"GPU {rank}",
        position=rank,
        leave=True,
        unit="mol",
        dynamic_ncols=True,
    )

    # Stream through the ENTIRE chunk with ONE DataLoader — workers are forked
    # once and stay alive (persistent_workers=True) for all batches.
    for batch_results in model.stream_predict_from_smiles(
        smiles_list=smiles_chunk,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        max_nodes=max_nodes,
        threshold=threshold,
        failure_log_path=shard_path + ".failures",
    ):
        batch_smiles = smiles_chunk[done : done + len(batch_results)]
        for smi, res in zip(batch_smiles, batch_results):
            if res is None or res.get("intensities") is None:
                smiles_buf.append(smi)
                intensities_buf.append(None)
                num_frags_buf.append(0)
                valid_buf.append(False)
                if fragment_masses_buf is not None:
                    fragment_masses_buf.append(None)
            else:
                arr = np.array(res["intensities"], dtype=np.float32)
                if n_bins is None:
                    n_bins = arr.shape[0]
                smiles_buf.append(smi)
                intensities_buf.append(arr)
                num_frags_buf.append(int(res.get("num_fragments", 0)))
                valid_buf.append(True)
                if fragment_masses_buf is not None:
                    fm = res.get("fragment_masses")
                    fragment_masses_buf.append(
                        np.asarray(fm, dtype=np.float32)
                        if fm is not None
                        else None
                    )

        done += len(batch_results)
        pbar.update(len(batch_results))
        batches_since_ckpt += 1

        if _shutdown_requested.is_set():
            tqdm.write(
                f"[GPU {rank}] Preemption signal — flushing buffer and exiting."
            )
            if _ckpt_thread is not None:
                _ckpt_thread.join()
            if smiles_buf and n_bins is not None:
                _write_checkpoint(
                    shard_path,
                    smiles_buf,
                    intensities_buf,
                    num_frags_buf,
                    valid_buf,
                    n_bins,
                    already_done,
                    fragment_masses_buf,
                )
            pbar.close()
            sys.exit(2)  # 2 = gracefully preempted; main will skip merge

        if batches_since_ckpt >= checkpoint_every and n_bins is not None:
            log.info(
                f"Checkpointing {len(smiles_buf):,} molecules to {shard_path}",
                extra=extra,
            )
            _flush_checkpoint_async(
                list(smiles_buf),
                list(intensities_buf),
                list(num_frags_buf),
                list(valid_buf),
                n_bins,
                already_done,
                list(fragment_masses_buf)
                if fragment_masses_buf is not None
                else None,
            )
            already_done += len(smiles_buf)
            smiles_buf.clear()
            intensities_buf.clear()
            num_frags_buf.clear()
            valid_buf.clear()
            if fragment_masses_buf is not None:
                fragment_masses_buf.clear()
            batches_since_ckpt = 0
            torch.cuda.empty_cache()

    # Final flush — wait for any background checkpoint first.
    if _ckpt_thread is not None:
        _ckpt_thread.join()
    if smiles_buf and n_bins is not None:
        log.info(
            f"Final checkpoint: {len(smiles_buf):,} remaining molecules.",
            extra=extra,
        )
        _write_checkpoint(
            shard_path,
            smiles_buf,
            intensities_buf,
            num_frags_buf,
            valid_buf,
            n_bins,
            already_done,
            fragment_masses_buf,
        )

    pbar.close()

    # Remove progress file — shard is complete
    progress_path = _checkpoint_progress_path(shard_path)
    if os.path.exists(progress_path):
        os.remove(progress_path)

    total_time = time.time() - t0
    tqdm.write(
        f"[GPU {rank}] Done. {total_time / 3600:.2f} h  "
        f"({total_time / n_total * 1000:.1f} ms/mol avg)"
    )


def _forward_signal_to_workers(processes):
    """Return a signal handler that forwards SIGUSR1/SIGTERM to all workers."""

    def _handler(signum, frame):
        print(
            f"[main] Signal {signum} received — forwarding to {len(processes)} workers.",
            flush=True,
        )
        for p in processes:
            if p.is_alive():
                try:
                    os.kill(p.pid, signal.SIGUSR1)
                except ProcessLookupError:
                    pass

    return _handler


def _merge_failure_logs(shard_paths: List[str], output_path: str) -> None:
    """Merge per-worker failure CSVs into a single file next to the output
    HDF5."""
    import csv
    import glob

    output_dir = Path(output_path).parent
    pattern = str(output_dir / "_shard_*.hdf5.failures.*.csv")
    parts = sorted(glob.glob(pattern))
    if not parts:
        return

    merged_path = str(Path(output_path).with_suffix("")) + "_failures.csv"
    header_written = False
    with open(merged_path, "w", newline="") as out:
        writer = csv.writer(out)
        for part in parts:
            with open(part, newline="") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if header is None:
                    continue
                if not header_written:
                    writer.writerow(header)
                    header_written = True
                writer.writerows(reader)
            os.remove(part)

    logging.info(
        f"Failure log written to {merged_path}", extra={"rank": "main"}
    )


def _merge_shards(shard_paths: List[str], output_path: str):
    log = logging.getLogger()
    extra = {"rank": "main"}
    log.info(
        f"Merging {len(shard_paths)} shards -> {output_path}", extra=extra
    )

    # Peek at first shard to get n_bins and check for fragment_masses
    with h5py.File(shard_paths[0], "r") as f0:
        n_bins = f0["intensities"].shape[1]
        has_frag_masses = "fragment_masses" in f0

    str_dt = h5py.special_dtype(vlen=str)
    # Stream shards one at a time — never holds more than one shard in RAM
    with h5py.File(output_path, "w") as out:
        ds_smiles = out.create_dataset(
            "smiles", shape=(0,), maxshape=(None,), dtype=str_dt
        )
        ds_int = out.create_dataset(
            "intensities",
            shape=(0, n_bins),
            maxshape=(None, n_bins),
            compression="gzip",
            compression_opts=4,
            chunks=(1000, n_bins),
        )
        ds_nf = out.create_dataset(
            "num_fragments", shape=(0,), maxshape=(None,), dtype=np.int64
        )
        ds_valid = out.create_dataset(
            "valid", shape=(0,), maxshape=(None,), dtype=bool
        )
        if has_frag_masses:
            ds_fm = out.create_dataset(
                "fragment_masses",
                shape=(0,),
                maxshape=(None,),
                dtype=h5py.vlen_dtype(np.float32),
            )

        n_total = 0
        valid_total = 0
        for path in shard_paths:
            with h5py.File(path, "r") as f:
                n = f["intensities"].shape[0]
                smiles = f["smiles"][:]
                intensities = f["intensities"][:]
                num_frags = f["num_fragments"][:]
                valid = f["valid"][:]

                ds_smiles.resize(n_total + n, axis=0)
                ds_int.resize(n_total + n, axis=0)
                ds_nf.resize(n_total + n, axis=0)
                ds_valid.resize(n_total + n, axis=0)

                ds_smiles[n_total : n_total + n] = smiles
                ds_int[n_total : n_total + n] = intensities
                ds_nf[n_total : n_total + n] = num_frags
                ds_valid[n_total : n_total + n] = valid

                if has_frag_masses and "fragment_masses" in f:
                    fm = f["fragment_masses"][:]
                    ds_fm.resize(n_total + n, axis=0)
                    ds_fm[n_total : n_total + n] = fm

                n_total += n
                valid_total += int(valid.sum())
            log.info(f"Merged shard {path} ({n:,} molecules)", extra=extra)

        out.attrs["n_molecules"] = n_total
        out.attrs["n_bins"] = n_bins
        out.attrs["valid_count"] = valid_total

    log.info(
        f"Saved {n_total:,} spectra ({valid_total:,} valid) to {output_path}",
        extra=extra,
    )

    # Clean up shards
    for path in shard_paths:
        os.remove(path)


def main():
    parser = argparse.ArgumentParser(
        description="Large-scale batched EIMS inference across multiple GPUs."
    )
    parser.add_argument(
        "--intensity-predictor",
        required=True,
        help="Path to intensity predictor checkpoint.",
    )
    parser.add_argument(
        "--fragment-generator",
        default=None,
        help="Path to fragment generator checkpoint (optional).",
    )
    parser.add_argument(
        "--input", required=True, help="Path to input CSV or TSV file."
    )
    parser.add_argument(
        "--smiles-col",
        default="smiles",
        help="Column name containing SMILES (default: smiles). Pass empty string to auto-use first column.",
    )
    parser.add_argument(
        "--no-header",
        action="store_true",
        help="File has no header row — just one SMILES per line.",
    )
    parser.add_argument(
        "--smiles-col-idx",
        type=int,
        default=0,
        help="0-based column index for SMILES in headerless files (default: 0).",
    )
    parser.add_argument(
        "--sep",
        type=str,
        default=None,
        help="Column separator. Auto-detected from extension (.tsv/.txt->tab, .csv->comma) if not set.",
    )
    parser.add_argument(
        "--output", required=True, help="Output HDF5 file path."
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=8,
        help="Number of GPUs to use (default: 8).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size per GPU (default: 64).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader workers per GPU for preprocessing (default: 4).",
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=50,
        help="Max nodes for fragment generation (default: 50).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.01,
        help="Fragment generation threshold (default: 0.01).",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Comma-separated GPU indices to use (e.g. '0,1,2,3'). Overrides --num-gpus.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=100,
        help="Save a checkpoint to the shard HDF5 every N batches (default: 100). Enables resume on crash.",
    )
    parser.add_argument(
        "--start-idx",
        type=int,
        default=None,
        help="Start index into the SMILES list (inclusive). Useful for splitting across machines.",
    )
    parser.add_argument(
        "--end-idx",
        type=int,
        default=None,
        help="End index into the SMILES list (exclusive). Useful for splitting across machines.",
    )
    parser.add_argument(
        "--shard-id",
        type=int,
        default=None,
        help=(
            "Explicit shard index for the output shard filename "
            "(_shard_<id>.hdf5). Defaults to the 0-based rank within --gpus. "
            "Set explicitly when launching one independent process per GPU "
            "(each with --num-gpus 1) so shards don't collide."
        ),
    )
    parser.add_argument(
        "--save-fragments",
        action="store_true",
        default=False,
        help=(
            "Save per-molecule fragment base masses to the output HDF5 as a "
            "variable-length float32 dataset 'fragment_masses'. Increases output "
            "file size. Each entry is an array of monoisotopic masses (one per "
            "enumerated fragment) for valid molecules, or empty for invalid ones."
        ),
    )
    args = parser.parse_args()

    # Resolve GPU indices — use nvidia-smi to avoid initializing a CUDA context
    # in the parent process (which would be inherited by spawned workers).
    if args.gpus:
        gpu_ids = [int(g) for g in args.gpus.split(",")]
    else:
        import subprocess

        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                check=True,
            )
            available = len(result.stdout.strip().splitlines())
        except Exception:
            available = args.num_gpus  # fallback
        num = min(args.num_gpus, available)
        if num == 0:
            logging.error("No GPUs available.", extra={"rank": "main"})
            sys.exit(1)
        gpu_ids = list(range(num))

    logging.info(f"Using GPUs: {gpu_ids}", extra={"rank": "main"})

    # Skip entirely if output already exists from a previous completed run.
    if os.path.exists(args.output):
        print(
            f"Output {args.output} already exists — nothing to do.", flush=True
        )
        sys.exit(0)

    # Resolve row range — avoid loading the full file in the parent process.
    sep = _infer_sep(args.input, args.sep)
    start = args.start_idx if args.start_idx is not None else 0
    if args.end_idx is not None:
        end = args.end_idx
    else:
        print("Counting lines in input file (no --end-idx given)...")
        with open(args.input, "rb") as fh:
            total_lines = sum(1 for _ in fh)
        end = total_lines if args.no_header else total_lines - 1
    n = end - start
    print(f"Row range [{start}:{end}] -> {n:,} molecules")

    # Estimate time
    ms_per_mol = 100
    total_s = n * ms_per_mol / 1000 / len(gpu_ids)
    print(
        f"Rough time estimate: {n:,} mol / {len(gpu_ids)} GPUs @ {ms_per_mol}ms/mol"
        f" = {total_s / 3600:.1f} h ({total_s / 86400:.2f} days)"
    )

    # Split row range across GPUs
    chunk_size = (n + len(gpu_ids) - 1) // len(gpu_ids)
    chunk_ranges = [
        (start + i * chunk_size, min(start + (i + 1) * chunk_size, end))
        for i in range(len(gpu_ids))
    ]

    output_dir = Path(args.output).parent
    shard_ids = (
        [args.shard_id]
        if args.shard_id is not None
        else list(range(len(gpu_ids)))
    )
    shard_paths = [
        str(output_dir / f"_shard_{shard_id}.hdf5") for shard_id in shard_ids
    ]

    # Suppress C++ deprecation warnings (e.g. lazyInitCUDA) in all child
    # processes.  Must be set before torch is imported in each worker, so set
    # it here in the parent — spawned processes inherit the environment.
    os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"
    os.environ["TORCH_SHOW_CPP_STACKTRACES"] = "0"

    # Use file_system sharing strategy to avoid pidfd_getfd permission errors
    # on cluster nodes where that syscall is restricted.
    mp.set_sharing_strategy("file_system")

    # Launch one process per GPU — pass file path + row range, not the data,
    # so nothing large or CUDA-related needs to be pickled across the IPC pipe.
    ctx = mp.get_context("spawn")
    processes = []
    for rank, (row_start, row_end), shard in zip(
        gpu_ids, chunk_ranges, shard_paths
    ):
        p = ctx.Process(
            target=_worker,
            args=(
                rank,
                args.input,
                sep,
                args.no_header,
                args.smiles_col,
                args.smiles_col_idx,
                row_start,
                row_end,
                args.intensity_predictor,
                args.fragment_generator,
                shard,
                args.batch_size,
                args.num_workers,
                args.max_nodes,
                args.threshold,
                args.checkpoint_every,
                args.save_fragments,
            ),
        )
        p.start()
        processes.append(p)

    # Forward preemption signals from the batch script to all worker processes.
    _handler = _forward_signal_to_workers(processes)
    signal.signal(signal.SIGUSR1, _handler)
    signal.signal(signal.SIGTERM, _handler)

    # Poll workers. As soon as ANY worker dies (crash or preemption), stop the
    # rest and exit — a dead GPU has its progress checkpointed, and the outer
    # restart loop will relaunch the whole job so that GPU resumes immediately
    # instead of sitting idle for the rest of the multi-day run.
    exited_bad = []
    while True:
        all_done = True
        for i, p in enumerate(processes):
            if p.is_alive():
                all_done = False
                continue
            if i not in exited_bad and p.exitcode not in (0, 2):
                exited_bad.append(i)
        if exited_bad or all_done:
            break
        time.sleep(5)

    if exited_bad:
        logging.error(
            f"Workers {exited_bad} died — restarting whole job so they can "
            f"resume from checkpoint.",
            extra={"rank": "main"},
        )
        for p in processes:
            if p.is_alive():
                try:
                    os.kill(p.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for p in processes:
            p.join(timeout=30)
        sys.exit(1)

    preempted = [i for i, p in enumerate(processes) if p.exitcode == 2]
    if preempted:
        print(
            f"[main] Workers {preempted} exited for preemption — job will be requeued.",
            flush=True,
        )
        sys.exit(0)

    if args.shard_id is not None:
        # One-shard-per-process mode (e.g. one independent OS process per
        # GPU): this process only owns a single shard, not the full output.
        # Merging is a separate manual step run once after all shards finish
        # -- see examples/scripts/evaluation/merge_pubchem_shards.py.
        logging.info(
            f"Shard {shard_paths[0]} complete. Run the merge script once all "
            f"shards are done.",
            extra={"rank": "main"},
        )
        return

    _merge_shards(shard_paths, args.output)
    _merge_failure_logs(shard_paths, args.output)
    logging.info("Done.", extra={"rank": "main"})


if __name__ == "__main__":
    main()
