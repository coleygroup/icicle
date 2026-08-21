#!/usr/bin/env python
"""Run MassFormer inference and save predictions for evaluation.

This script runs inference on a set of SMILES and saves predicted spectra
to an HDF5 file. The predictions can then be evaluated using ICICLE's
unified evaluation script.

Usage:
    # Predict test-split molecules (for similarity eval)
    python scripts/run_inference_for_eval.py \
        --config config/inference_nist23.yml \
        --output predictions/massformer_nist23_scaffold.hdf5 \
        --split test

    # Predict all PubChem isomers top-50 (for formula retrieval eval)
    python scripts/run_inference_for_eval.py \
        --config config/inference_nist23.yml \
        --output predictions/massformer_pubchem_cands_50.hdf5 \
        --candidates-pickle ../../data/NIST2023_GCMS_main/retrieval/cands_pickled_scaffold_50.pkl

    # Predict all PubChem isomers (no limit)
    # Predict test-split molecules (for similarity eval)
    python scripts/run_inference_for_eval.py \
        --config config/inference_nist23.yml \
        --output predictions/massformer_nist23_scaffold.hdf5 \
        --split test

    # Predict all PubChem isomers top-50 (for formula retrieval eval)
    python scripts/run_inference_for_eval.py \
        --config config/inference_nist23.yml \
        --output predictions/massformer_pubchem_cands_50.hdf5 \
        --candidates-pickle ../../data/NIST2023_GCMS_main/retrieval/cands_pickled_scaffold_50.pkl

    # Predict all PubChem isomers (no limit)
    python scripts/run_inference_for_eval.py \
        --config config/inference_nist23.yml \
        --output predictions/massformer_pubchem_cands_all.hdf5 \
        --candidates-pickle ../../data/NIST2023_GCMS_main/retrieval/cands_pickled_scaffold_None.pkl

The output HDF5 file has the structure:
    /<mol_id>/
        - smiles: str (attribute)
        - inchi_key: str (attribute)
        - predicted_intensities: float32 array of shape (num_bins,)
        - mz_bins: float32 array of shape (num_bins,)
"""

import argparse
import logging
import os
import shutil
import sys
import tempfile
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict

import h5py
import numpy as np
import pandas as pd
import torch as th
import yaml
from tqdm import tqdm

# Add massformer to path
script_dir = Path(__file__).parent
massformer_root = script_dir.parent
sys.path.insert(0, str(massformer_root / "src"))
sys.path.insert(0, str(script_dir))

import massformer.data_utils as _mf_data_utils
from massformer.dataset import data_to_device
from massformer.runner import get_ds_model, get_pbar, load_config
from massformer.spec_utils import process_spec, unprocess_spec
from run_inference import init_from_smiles

# Patch get_murcko_scaffold to not crash on invalid valence SMILES from PubChem
_orig_get_murcko_scaffold = _mf_data_utils.get_murcko_scaffold


def _safe_murcko_scaffold(mol, **kwargs):
    try:
        return _orig_get_murcko_scaffold(mol, **kwargs)
    except Exception:
        return float("nan")


_mf_data_utils.get_murcko_scaffold = _safe_murcko_scaffold

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_smiles_from_splits(
    labels_path: str,
    splits_path: str,
    split: str = "test",
) -> pd.DataFrame:
    """Load SMILES for a given split from ICICLE's splits format.

     Parameters
    -------
     labels_path : str
         Path to metadata.tsv with mol_id, standardized_smiles, inchi_key
     splits_path : str
         Path to splits TSV with mol_id, inchi_key, split columns
     split : str
         Which split to load: "train", "val", or "test"

     Returns
    ----
     pd.DataFrame
         DataFrame with mol_id, smiles, inchi_key for the requested split
     Returns
    ----
     pd.DataFrame
         DataFrame with mol_id, smiles, inchi_key for the requested split
    """
    labels_df = pd.read_csv(labels_path, sep="\t")
    splits_df = pd.read_csv(splits_path, sep="\t")

    # Normalise column names for non-NIST datasets (VGWD uses spec/smiles/inchikey)
    if "spec" in labels_df.columns and "mol_id" not in labels_df.columns:
        labels_df = labels_df.rename(columns={"spec": "mol_id"})
    if (
        "smiles" in labels_df.columns
        and "standardized_smiles" not in labels_df.columns
    ):
        labels_df = labels_df.rename(columns={"smiles": "standardized_smiles"})
    if (
        "inchikey" in labels_df.columns
        and "inchi_key" not in labels_df.columns
    ):
        labels_df = labels_df.rename(columns={"inchikey": "inchi_key"})

    split_mol_ids = splits_df[splits_df["split"] == split]["mol_id"].tolist()
    filtered = labels_df[labels_df["mol_id"].isin(split_mol_ids)].copy()

    logging.info(f"Loaded {len(filtered)} molecules from {split} split")

    return filtered[["mol_id", "standardized_smiles", "inchi_key"]]


def _smiles_to_row(smi: str):
    """Convert a SMILES string to (smiles, inchi_key). Returns (smi, None) on failure."""
    from rdkit import Chem
    from rdkit.Chem.inchi import InchiToInchiKey, MolToInchi

    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return smi, None
    inchi = MolToInchi(mol)
    if inchi is None:
        return smi, None
    return smi, InchiToInchiKey(inchi)


def load_smiles_from_tsv(
    tsv_path: str,
    smiles_col: int = 1,
    id_col: int = 0,
    chunk_start: int = 0,
    chunk_size=None,
) -> pd.DataFrame:
    """Load SMILES from a tab-separated file (e.g. PubChem_filtered.tsv).

    Parameters
    ----------
    tsv_path : str
        Path to TSV with at minimum an ID column and a SMILES column.
    smiles_col : int
        0-indexed column index for SMILES.
    id_col : int
        0-indexed column index for molecule ID.
    chunk_start : int
        Row offset (0-indexed, after header) to start reading from.
    chunk_size : int, optional
        Number of rows to read; None reads all remaining rows.

    Returns
    -------
    pd.DataFrame
        DataFrame with mol_id, standardized_smiles, inchi_key columns.
    """
    df = pd.read_csv(
        tsv_path,
        sep="\t",
        skiprows=range(1, chunk_start + 1) if chunk_start > 0 else None,
        nrows=chunk_size,
        header=0,
        usecols=[id_col, smiles_col],
    )
    df.columns = ["mol_id", "standardized_smiles"]
    df["mol_id"] = df["mol_id"].astype(str)
    df["inchi_key"] = ""
    logging.info(f"Loaded {len(df)} SMILES from {tsv_path} (start={chunk_start})")
    return df


def load_smiles_from_candidates_pickle(
    candidates_pickle_path: str,
) -> pd.DataFrame:
    """Extract all unique candidate non-stereo SMILES from a retrieval candidates pickle.

     The pickle is keyed by target InChIKey (full) and each entry has a 'cands'
     array of non-stereo SMILES ranked by Tanimoto similarity. This function
     collects all unique candidate SMILES across all targets (excluding the
     targets themselves, which should be predicted separately via the splits path).

     Parameters
    -------
     candidates_pickle_path : str
         Path to cands_pickled_scaffold_50.pkl or cands_pickled_scaffold_None.pkl

     Returns
    ----
     pd.DataFrame
         DataFrame with mol_id (0-indexed int), smiles, inchi_key columns.
         inchi_key is empty string since PubChem candidates have no NIST mol_id;
         eval_from_predictions.py matches them via InChIKey-14 from the formula map.
    """
    import pickle

    with open(candidates_pickle_path, "rb") as f:
        cands_dict = pickle.load(f)

    unique_smiles = set()
    for entry in cands_dict.values():
        cands = entry["cands"]
        unique_smiles.update(
            cands.tolist() if hasattr(cands, "tolist") else cands
        )

    logging.info(
        f"Loaded {len(unique_smiles)} unique candidate SMILES from {candidates_pickle_path}"
    )

    smiles_list = sorted(unique_smiles)
    n_workers = cpu_count()
    logging.info(
        f"Computing InChIKeys using {n_workers} workers for {len(smiles_list)} SMILES..."
    )

    chunk_size = max(1000, len(smiles_list) // (n_workers * 10))
    with Pool(n_workers) as pool:
        results = list(
            tqdm(
                pool.imap(_smiles_to_row, smiles_list, chunksize=chunk_size),
                total=len(smiles_list),
                desc="Computing InChIKeys",
            )
        )

    rows = [
        (i, smi, ik) for i, (smi, ik) in enumerate(results) if ik is not None
    ]
    skipped = len(results) - len(rows)
    if skipped:
        logging.warning(
            f"Skipped {skipped} SMILES that could not be parsed or converted to InChIKey"
        )
    logging.info(f"Prepared {len(rows)} candidate molecules for inference")
    return pd.DataFrame(
        rows, columns=["mol_id", "standardized_smiles", "inchi_key"]
    )


def run_inference(
    model,
    smiles_df: pd.DataFrame,
    dataset,
    run_d: dict,
    data_d: dict,
) -> Dict[str, Dict]:
    """Run MassFormer inference on SMILES.

    Returns dict mapping mol_id to prediction info.
    """
    smiles_list = smiles_df["standardized_smiles"].tolist()
    mol_ids = smiles_df["mol_id"].tolist()
    inchi_keys = smiles_df["inchi_key"].tolist()

    # Create input dataframe for MassFormer
    input_df = pd.DataFrame(
        {"mol_id": list(range(len(smiles_list))), "smiles": smiles_list}
    )

    # Initialize dataset from SMILES
    new_ds, new_dl = init_from_smiles(
        dataset, input_df, prec_types=[], nces=[], run_d=run_d
    )

    # Run inference
    dev = th.device(run_d["device"])
    nb = run_d["non_blocking"]
    model.to(dev)
    model.eval()

    all_preds = []
    all_input_mol_ids = []

    with th.no_grad():
        for b_idx, b in get_pbar(
            enumerate(new_dl), run_d, desc="> inference", total=len(new_dl)
        ):
            b = data_to_device(b, dev, nb)
            b_pred = model(data=b, amp=run_d["amp"])["pred"]
            b_mol_id = b["mol_id"]
            all_preds.append(b_pred.detach().cpu())
            all_input_mol_ids.append(b_mol_id.detach().cpu())

    all_preds = th.cat(all_preds, dim=0)
    all_input_mol_ids = th.cat(all_input_mol_ids, dim=0)

    # Untransform predictions (reverse log10over3)
    all_preds = unprocess_spec(all_preds, data_d["transform"])
    # Normalize to L1
    all_preds = process_spec(all_preds, "none", "l1")
    all_preds_np = all_preds.numpy()

    # Normalize to max=1 (ICICLE convention)
    for i in range(len(all_preds_np)):
        if all_preds_np[i].max() > 0:
            all_preds_np[i] = all_preds_np[i] / all_preds_np[i].max()

    # Build output dict mapping original mol_id to predictions
    predictions = {}
    for i, input_mol_id in enumerate(all_input_mol_ids.numpy()):
        orig_mol_id = mol_ids[input_mol_id]
        predictions[orig_mol_id] = {
            "smiles": smiles_list[input_mol_id],
            "inchi_key": inchi_keys[input_mol_id],
            "intensities": all_preds_np[i],
        }

    return predictions


def save_predictions_hdf5(
    predictions: Dict[str, Dict],
    output_path: Path,
    min_mz: float,
    max_mz: float,
    bin_width: float,
):
    """Save predictions to HDF5 file.

    Format matches what ICICLE's eval expects.

    Writes to a local-disk temp file first, then moves the completed file to
    output_path (typically on NFS) only after the write succeeds. output_path
    lives on a shared NFS mount, and many GPU workers writing large HDF5s to
    it concurrently has been observed to fail mid-write with EIO, which
    leaves a structurally corrupt (but existing) file at output_path -- the
    resume logic elsewhere only checks existence, not validity, so a
    half-written file gets silently trusted. Writing locally avoids the
    concurrent-NFS-write failure window entirely, and the final move only
    ever produces a complete file or no file at all.
    """
    num_bins = int((max_mz - min_mz) / bin_width)
    mz_bins = np.linspace(min_mz, max_mz, num_bins, endpoint=False).astype(
        np.float32
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(
        suffix=".hdf5", prefix=output_path.stem + "_", dir="/tmp"
    )
    os.close(fd)
    tmp_path = Path(tmp_name)

    try:
        with h5py.File(tmp_path, "w") as hf:
            # Store metadata
            hf.attrs["min_mz"] = min_mz
            hf.attrs["max_mz"] = max_mz
            hf.attrs["bin_width"] = bin_width
            hf.attrs["num_predictions"] = len(predictions)

            for mol_id, data in tqdm(
                predictions.items(), desc="Saving predictions"
            ):
                grp = hf.create_group(str(mol_id))
                grp.attrs["smiles"] = data["smiles"]
                grp.attrs["inchi_key"] = data["inchi_key"]
                grp.create_dataset(
                    "predicted_intensities",
                    data=data["intensities"],
                    compression="gzip",
                )
                grp.create_dataset(
                    "mz_bins", data=mz_bins, compression="gzip"
                )

        shutil.move(str(tmp_path), str(output_path))
    finally:
        tmp_path.unlink(missing_ok=True)

    logging.info(f"Saved {len(predictions)} predictions to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run MassFormer inference for evaluation"
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Path to inference config YAML",
    )
    parser.add_argument(
        "--output", "-o", type=str, required=True, help="Output HDF5 file path"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Which split to run inference on (ignored when --candidates-pickle is set)",
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device (overrides config)"
    )
    parser.add_argument(
        "--candidates-pickle",
        type=str,
        default=None,
        help=(
            "Path to prebuilt candidates pickle "
            "(e.g. data/NIST2023_GCMS_main/retrieval/cands_pickled_scaffold_50.pkl or "
            "cands_pickled_scaffold_None.pkl). When set, runs inference on all unique "
            "PubChem candidate SMILES instead of a dataset split."
        ),
    )
    parser.add_argument(
        "--smiles-tsv",
        type=str,
        default=None,
        help=(
            "Path to a tab-separated file of SMILES (e.g. PubChem_filtered.tsv). "
            "When set, runs inference on all rows in the file, optionally in chunks."
        ),
    )
    parser.add_argument(
        "--smiles-tsv-smiles-col",
        type=int,
        default=1,
        help="0-indexed column index for SMILES in --smiles-tsv (default: 1).",
    )
    parser.add_argument(
        "--smiles-tsv-id-col",
        type=int,
        default=0,
        help="0-indexed column index for molecule ID in --smiles-tsv (default: 0).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help=(
            "Number of molecules to process per chunk when using --smiles-tsv. "
            "Writes one HDF5 per chunk named <output>.chunk<N>.hdf5. "
            "Default: process all at once (only feasible for small inputs)."
        ),
    )
    parser.add_argument(
        "--chunk-start",
        type=int,
        default=0,
        help="Row offset (0-indexed) to start reading from --smiles-tsv (default: 0).",
    )
    parser.add_argument(
        "--chunk-end",
        type=int,
        default=None,
        help=(
            "Row offset (exclusive) to stop at when using --smiles-tsv with "
            "--chunk-size — stops once the next chunk's start would reach this "
            "row, without processing it. Needed to partition the file across "
            "multiple parallel workers without them racing on each other's "
            "chunk files near shared boundaries. Default: process to EOF."
        ),
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Load MassFormer model (shared across all chunks)
    template_fp = config["massformer"]["template_config"]
    custom_fp = config["massformer"]["custom_config"]
    device_id = config["massformer"].get("device_id", 0)

    if args.device:
        device_id = (
            int(args.device.replace("cuda:", ""))
            if "cuda" in args.device
            else -1
        )

    entity_name, project_name, run_name, data_d, model_d, run_d = load_config(
        template_fp, custom_fp, device_id, None
    )

    if args.device:
        run_d["device"] = args.device

    dataset, model, _, _, _ = get_ds_model(data_d, model_d, run_d)

    checkpoint_path = config["massformer"]["checkpoint_path"]
    logging.info(f"Loading checkpoint from {checkpoint_path}")
    chkpt_d = th.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(chkpt_d["best_model_sd"])

    min_mz = config["data"]["min_mz"]
    max_mz = config["data"]["max_mz"]
    bin_width = config["data"]["bin_width"]
    output_path = Path(args.output)

    if args.smiles_tsv:
        if args.chunk_size is None:
            logging.info(f"Running inference on full TSV {args.smiles_tsv} (no chunking)")
            smiles_df = load_smiles_from_tsv(
                args.smiles_tsv,
                smiles_col=args.smiles_tsv_smiles_col,
                id_col=args.smiles_tsv_id_col,
                chunk_start=args.chunk_start,
            )
            predictions = run_inference(model, smiles_df, dataset, run_d, data_d)
            save_predictions_hdf5(predictions, output_path, min_mz, max_mz, bin_width)
        else:
            chunk_idx = args.chunk_start // args.chunk_size
            offset = args.chunk_start
            while True:
                if args.chunk_end is not None and offset >= args.chunk_end:
                    break
                smiles_df = load_smiles_from_tsv(
                    args.smiles_tsv,
                    smiles_col=args.smiles_tsv_smiles_col,
                    id_col=args.smiles_tsv_id_col,
                    chunk_start=offset,
                    chunk_size=args.chunk_size,
                )
                if smiles_df.empty:
                    break
                chunk_path = output_path.with_suffix(f".chunk{chunk_idx}.hdf5")
                if chunk_path.exists():
                    logging.info(f"Skipping existing chunk {chunk_path}")
                else:
                    predictions = run_inference(model, smiles_df, dataset, run_d, data_d)
                    save_predictions_hdf5(predictions, chunk_path, min_mz, max_mz, bin_width)
                offset += args.chunk_size
                chunk_idx += 1
                if len(smiles_df) < args.chunk_size:
                    break
    elif args.candidates_pickle:
        logging.info(f"Running inference on PubChem candidates from {args.candidates_pickle}")
        smiles_df = load_smiles_from_candidates_pickle(args.candidates_pickle)
        predictions = run_inference(model, smiles_df, dataset, run_d, data_d)
        save_predictions_hdf5(predictions, output_path, min_mz, max_mz, bin_width)
    else:
        logging.info(f"Running inference for {args.split} split")
        smiles_df = load_smiles_from_splits(
            labels_path=config["data"]["labels_path"],
            splits_path=config["data"]["splits_path"],
            split=args.split,
        )
        predictions = run_inference(model, smiles_df, dataset, run_d, data_d)
        save_predictions_hdf5(predictions, output_path, min_mz, max_mz, bin_width)

    logging.info("Inference complete!")


if __name__ == "__main__":
    main()
