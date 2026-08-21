#!/usr/bin/env python
"""Large-scale NEIMS inference over filtered PubChem for global retrieval evaluation.

Reads PubChem_filtered.tsv (tab-sep, header: ID, SMILES, InChIKey, MW — the same
~93M-molecule file used for ICICLE batch inference), runs NEIMS (ECFP) or NEIMS-GNN
inference, and writes a flat columnar HDF5 matching the format expected by
pubchem_global_retrieval.py:
    intensities  (N, n_bins)  float32
    smiles       (N,)         variable-length UTF-8
    inchikey14   (N,)         variable-length UTF-8  (first 14 chars of InChIKey)
    valid        (N,)         bool

InChIKey14 is read directly from the file (no recomputation needed).

Supports slicing (--start-idx / --end-idx) for multi-node parallelism, and
resumes from the last checkpoint if the output file already exists.

Usage
-----
# Full filtered PubChem (~93M molecules, single GPU):
python batch_infer_pubchem_neims.py \\
    --checkpoint outputs/neims_random_s1/best_model.pt \\
    --input ../../data/PubChem/PubChem_filtered.tsv \\
    --output results/pubchem_predictions/neims_random_s1_pubchem_full.hdf5 \\
    --batch-size 2048 --num-workers 8

# Sliced for two nodes:
python batch_infer_pubchem_neims.py ... --start-idx 0       --end-idx 47000000
python batch_infer_pubchem_neims.py ... --start-idx 47000000
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import pandas as pd
import torch
from rdkit import RDLogger
from tqdm import tqdm

RDLogger.DisableLog("rdApp.*")

FLUSH_EVERY = 50_000


def _iter_chunks(
    path: str, start: int, end: Optional[int], chunk_size: int
):
    """Yield (smiles_list, ik14_list) chunks from PubChem_filtered.tsv.

    Streams the file to avoid loading all SMILES into RAM.
    """
    nrows = (end - start) if end is not None else None
    reader = pd.read_csv(
        path,
        sep="\t",
        header=None,
        dtype=str,
        skiprows=start + 1,
        nrows=nrows,
        usecols=[1, 2],
        chunksize=chunk_size,
    )
    for chunk in reader:
        smiles = chunk.iloc[:, 0].fillna("").tolist()
        ik14 = [ik[:14] if isinstance(ik, str) and ik else "" for ik in chunk.iloc[:, 1].fillna("")]
        yield smiles, ik14


def _count_rows(path: str, start: int, end: Optional[int]) -> int:
    """Count rows in slice without loading data."""
    nrows = (end - start) if end is not None else None
    total = 0
    reader = pd.read_csv(
        path,
        sep="\t",
        header=None,
        dtype=str,
        skiprows=start + 1,
        nrows=nrows,
        usecols=[1],
        chunksize=500_000,
    )
    for chunk in reader:
        total += len(chunk)
    return total


def _load_checkpoint(out_path: str) -> int:
    """Return rows already written (0 if no checkpoint)."""
    prog = out_path + ".progress.json"
    if not os.path.exists(prog) or not os.path.exists(out_path):
        return 0
    with open(prog) as f:
        return json.load(f).get("processed", 0)


def _flush(
    out_path: str,
    smiles_buf: list[str],
    intensities_buf: list[np.ndarray],
    ik14_buf: list[str],
    valid_buf: list[bool],
    n_bins: int,
    already: int,
):
    buf_size = len(smiles_buf)
    if buf_size == 0:
        return

    int_arr = np.zeros((buf_size, n_bins), dtype=np.float32)
    for i, arr in enumerate(intensities_buf):
        if arr is not None:
            int_arr[i] = arr

    str_dt = h5py.special_dtype(vlen=str)
    mode = "a" if os.path.exists(out_path) else "w"
    with h5py.File(out_path, mode) as f:
        if "intensities" not in f:
            f.create_dataset(
                "intensities",
                data=int_arr,
                compression="gzip",
                compression_opts=4,
                chunks=(min(1000, buf_size), n_bins),
                maxshape=(None, n_bins),
            )
            f.create_dataset(
                "smiles",
                data=np.array(smiles_buf, dtype=object),
                dtype=str_dt,
                maxshape=(None,),
            )
            f.create_dataset(
                "inchikey14",
                data=np.array(ik14_buf, dtype=object),
                dtype=str_dt,
                maxshape=(None,),
            )
            f.create_dataset(
                "valid",
                data=np.array(valid_buf, dtype=bool),
                maxshape=(None,),
            )
        else:
            n = f["intensities"].shape[0]
            for ds_name, arr in [
                ("intensities", int_arr),
                ("smiles", np.array(smiles_buf, dtype=object)),
                ("inchikey14", np.array(ik14_buf, dtype=object)),
                ("valid", np.array(valid_buf, dtype=bool)),
            ]:
                f[ds_name].resize(n + buf_size, axis=0)
                f[ds_name][n:] = arr

    prog = out_path + ".progress.json"
    with open(prog, "w") as f:
        json.dump({"processed": already + buf_size}, f)


def main():
    parser = argparse.ArgumentParser(
        description="NEIMS-GNN batch inference over PubChem → flat HDF5"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--input",
        required=True,
        help="pubchem_full.txt: tab-sep, no header, col0=CID, col1=SMILES",
    )
    parser.add_argument("--output", "-o", required=True)
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--end-idx", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    out_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt["model_config"]
    model_type = cfg.get("model_type", "neims")
    print(f"  model_type: {model_type}")

    if model_type == "neims_gnn":
        from neims.gnn_model import NEIMSGNN
        model = NEIMSGNN(
            output_size=cfg["output_size"],
            gnn_type=cfg.get("gnn_type", "GAT"),
            gnn_hidden_size=cfg.get("gnn_hidden_size", 64),
            gnn_num_layers=cfg.get("gnn_num_layers", 10),
            gnn_num_heads=cfg.get("gnn_num_heads", 8),
            gnn_dropout=cfg.get("gnn_dropout", 0.5),
            pool_type=cfg.get("pool_type", "max"),
            use_edge_features=cfg.get("use_edge_features", False),
            ffnn_hidden_sizes=cfg.get("ffnn_hidden_sizes", []),
            ffnn_dropout=cfg.get("ffnn_dropout", 0.25),
            resnet_bottleneck=cfg.get("resnet_bottleneck", 0.5),
            use_glu=cfg.get("use_glu", True),
            bidirectional=cfg.get("bidirectional", True),
            gate_bidirectional=cfg.get("gate_bidirectional", False),
            max_mass_offset=cfg.get("max_mass_offset", 5),
        )
    else:
        from neims.model import NEIMS
        model = NEIMS(
            input_size=cfg.get("fp_length", 4096),
            output_size=cfg["output_size"],
            hidden_sizes=cfg.get("hidden_sizes", [2000] * 8),
            dropout=cfg.get("dropout", 0.25),
            bidirectional=cfg.get("bidirectional", True),
            gate_bidirectional=cfg.get("gate_bidirectional", False),
            resnet_bottleneck=cfg.get("resnet_bottleneck", 0.5),
            max_mass_offset=cfg.get("max_mass_offset", 5),
            fp_radius=cfg.get("fp_radius", 2),
            fp_length=cfg.get("fp_length", 4096),
        )

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(args.device)
    model.eval()
    print(f"  Model on {args.device}, output_size={cfg['output_size']}")

    n_bins = cfg["output_size"]

    already = _load_checkpoint(out_path)
    effective_start = args.start_idx + already
    if already:
        print(f"Resuming from row {effective_start} ({already} already written)")

    print(f"Streaming SMILES [{effective_start}, {args.end_idx}) from {args.input}")

    READ_CHUNK = 2_000_000  # rows per TSV chunk — fits comfortably in RAM

    if model_type == "neims_gnn":
        from neims.gnn_data import create_inference_dataloader
    else:
        from predict import InferenceDataset
        from torch.utils.data import DataLoader

    smiles_buf: list[str] = []
    intensities_buf: list[np.ndarray] = []
    ik14_buf: list[str] = []
    valid_buf: list[bool] = []

    chunk_iter = _iter_chunks(args.input, effective_start, args.end_idx, READ_CHUNK)
    chunk_offset = 0

    for smiles_chunk, ik14_chunk in chunk_iter:
        chunk_size = len(smiles_chunk)

        if model_type == "neims_gnn":
            mol_ids = [str(i) for i in range(chunk_size)]
            dl = create_inference_dataloader(
                mol_ids=mol_ids,
                smiles_list=smiles_chunk,
                output_size=n_bins,
                use_edge_features=cfg.get("use_edge_features", False),
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
        else:
            mol_ids = [str(i) for i in range(chunk_size)]
            ds = InferenceDataset(
                mol_ids=mol_ids,
                smiles_list=smiles_chunk,
                fp_radius=cfg.get("fp_radius", 2),
                fp_length=cfg.get("fp_length", 4096),
                output_size=n_bins,
            )
            dl = DataLoader(
                ds,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                shuffle=False,
                persistent_workers=args.num_workers > 0,
                prefetch_factor=2 if args.num_workers > 0 else None,
            )

        idx = 0
        with torch.no_grad():
            for batch in tqdm(dl, desc=f"Inference chunk+{chunk_offset}", total=len(dl)):
                if model_type == "neims_gnn":
                    batch = batch.to(args.device)
                    preds = model(batch, batch.mass).cpu().numpy()
                    batch_size_actual = len(batch.mol_id)
                else:
                    fps = batch["fingerprint"].to(args.device)
                    masses = batch["mass"].to(args.device)
                    preds = model(fps, masses).cpu().numpy()
                    batch_size_actual = fps.shape[0]

                batch_smiles = smiles_chunk[idx : idx + batch_size_actual]
                batch_ik14 = ik14_chunk[idx : idx + batch_size_actual]
                for smi, ik14, pred in zip(batch_smiles, batch_ik14, preds):
                    row_max = pred.max()
                    if row_max > 0:
                        pred = pred / row_max
                        is_valid = True
                    else:
                        pred = np.zeros(n_bins, dtype=np.float32)
                        is_valid = False
                    smiles_buf.append(smi)
                    intensities_buf.append(pred.astype(np.float32))
                    ik14_buf.append(ik14)
                    valid_buf.append(is_valid and ik14 != "")

                idx += batch_size_actual

                if len(smiles_buf) >= FLUSH_EVERY:
                    _flush(
                        out_path, smiles_buf, intensities_buf, ik14_buf,
                        valid_buf, n_bins, already,
                    )
                    already += len(smiles_buf)
                    smiles_buf, intensities_buf, ik14_buf, valid_buf = [], [], [], []

        chunk_offset += chunk_size

    if smiles_buf:
        _flush(
            out_path, smiles_buf, intensities_buf, ik14_buf,
            valid_buf, n_bins, already,
        )
        already += len(smiles_buf)

    print(f"Done. {already:,} predictions in {out_path}")

    prog = out_path + ".progress.json"
    if os.path.exists(prog):
        os.remove(prog)


if __name__ == "__main__":
    main()
