#!/usr/bin/env python
"""Run RASSP inference and save predictions as HDF5 for unified ICICLE evaluation.

Reads ICICLE's metadata.tsv + splits.tsv, runs RASSP FormulaNet inference on
the requested split, converts sparse (mz, intensity) output to a dense binned
spectrum, and writes an HDF5 file that eval_from_predictions.py can consume.

Output HDF5 format
------------------
/<mol_id>/
    attrs:
        smiles    (str)
        inchi_key (str)
    datasets:
        predicted_intensities  float32[num_bins]
        mz_bins                float32[num_bins]
file-level attrs:
    min_mz, max_mz, bin_width, model, eval_split

Usage
-----
conda activate rassp
cd baselines/rassp

# Find your trained checkpoint (naming: <config>.<timestamp>.<epoch>.model)
ls checkpoints/

python scripts/run_inference_for_eval.py \\
    --checkpoint checkpoints/best_config.68094069.00000000.model \\
    --meta       checkpoints/best_config.68094069.meta \\
    --metadata   ../../data/NIST2023_GCMS_main/metadata.tsv \\
    --splits     ../../data/NIST2023_GCMS_main/splits/scaffold.tsv \\
    --output     results/rassp_nist23_scaffold.hdf5 \\
    --eval-split test \\
    --gpu

Then from the repo root (ICICLE env):
    uv run src/icicle/eval_from_predictions.py \\
        --predictions baselines/rassp/results/rassp_nist23_scaffold.hdf5 \\
        --ground-truth data/NIST2023_GCMS_main/spectra.hdf5 \\
        --labels  data/NIST2023_GCMS_main/metadata.tsv \\
        --splits  data/NIST2023_GCMS_main/splits/scaffold.tsv \\
        --output  results/eval/rassp_scaffold \\
        --mode all
"""

import argparse
import os
import pickle
from multiprocessing import Pool, cpu_count
from typing import Optional

import h5py
import numpy as np
import pandas as pd
import torch

# rassp must be installed (pip install -e .) before running this script
from rassp import netutil
from rassp.run_rassp import RenameUnpickler
from rassp.util import num_unique_frag_formulae
from rdkit import Chem
from rdkit.Chem import InchiToInchiKey
from rdkit.Chem.inchi import MolToInchi
from tqdm import tqdm


def _smiles_to_row(smi: str):
    """Convert a SMILES string to (smiles, inchi_key). Returns (smi, None) on failure."""
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return smi, None
        inchi = MolToInchi(mol)
        if inchi is None:
            return smi, None
        return smi, InchiToInchiKey(inchi)
    except Exception:
        return smi, None


def load_smiles_from_candidates_pickle(
    candidates_pickle_path: str,
    query_fraction: float = 1.0,
) -> tuple:
    """Extract all unique candidate SMILES from a retrieval candidates pickle.

    Parameters
    ----------
    query_fraction : float
        Fraction of test query molecules to include (default 1.0 = all).
        Values <1.0 subsample queries, reducing the candidate SMILES set
        proportionally — useful for quick end-to-end testing.

    Returns
    -------
    tuple of (mol_ids, smiles_list, inchikey_list)
    """
    with open(candidates_pickle_path, "rb") as f:
        cands_dict = pickle.load(f)

    if query_fraction < 1.0:
        import random
        keys = sorted(cands_dict.keys())
        n_keep = max(1, int(len(keys) * query_fraction))
        keys = random.sample(keys, n_keep)
        cands_dict = {k: cands_dict[k] for k in keys}
        n_total = int(len(keys) / query_fraction)
        print(f"  Subsampled to {n_keep}/{n_total} test queries ({query_fraction:.0%})")

    unique_smiles = set()
    for entry in cands_dict.values():
        cands = entry["cands"]
        unique_smiles.update(
            cands.tolist() if hasattr(cands, "tolist") else cands
        )

    smiles_list = sorted(unique_smiles)
    n_workers = cpu_count()
    print(
        f"Computing InChIKeys for {len(smiles_list)} unique SMILES ({n_workers} workers)…"
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
        print(f"Skipped {skipped} unparseable SMILES")
    print(f"  {len(rows)} candidate molecules ready for inference")

    mol_ids = [str(r[0]) for r in rows]
    smiles_out = [r[1] for r in rows]
    inchikeys = [r[2] for r in rows]
    return mol_ids, smiles_out, inchikeys


def _smiles_to_inchikey(smiles: str) -> str:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ""
        inchi = MolToInchi(mol)
        if inchi is None:
            return ""
        return InchiToInchiKey(inchi) or ""
    except Exception:
        return ""


def _is_valid_mol(mol: Chem.Mol, max_n_atoms: int, max_n_formula: int):
    """Return (mol_with_Hs, reason_str). reason is '' if valid."""
    atom_nums = {1, 6, 7, 8, 9, 15, 16, 17}  # H C N O F P S Cl
    try:
        anum_set = {
            mol.GetAtomWithIdx(i).GetAtomicNum()
            for i in range(mol.GetNumAtoms())
        }
        if not anum_set.issubset(atom_nums):
            return None, "atom type constraint violated"
        mol = Chem.AddHs(mol)
        Chem.SanitizeMol(mol)
        n_atoms = mol.GetNumAtoms()
        if n_atoms > max_n_atoms:
            return None, f"{n_atoms} atoms > {max_n_atoms}"
        n_formula = num_unique_frag_formulae(mol)
        if n_formula > max_n_formula:
            return None, f"{n_formula} formulae > {max_n_formula}"
        if len(Chem.GetMolFrags(mol)) > 1:
            return None, "multiple fragments"
        return mol, ""
    except Exception as e:
        return None, str(e)


# Module-level config for parallel validation (avoids pickling issues)
_VALIDATE_MAX_N_ATOMS = 48
_VALIDATE_MAX_N_FORMULA = 4096


def _validate_smiles(smi: str):
    """Return (mol_with_Hs | None) for use in parallel validation."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    result, _ = _is_valid_mol(mol, _VALIDATE_MAX_N_ATOMS, _VALIDATE_MAX_N_FORMULA)
    return result


def validate_smiles_parallel(
    smiles_list: list,
    max_n_atoms: int,
    max_n_formula: int,
    n_workers=None,  # type: Optional[int]
) -> tuple:
    """Validate SMILES in parallel. Returns (valid_mols, validity) lists."""
    global _VALIDATE_MAX_N_ATOMS, _VALIDATE_MAX_N_FORMULA
    _VALIDATE_MAX_N_ATOMS = max_n_atoms
    _VALIDATE_MAX_N_FORMULA = max_n_formula

    if n_workers is None:
        n_workers = cpu_count()

    chunk_size = max(500, len(smiles_list) // (n_workers * 20))
    print(f"Validating {len(smiles_list)} molecules ({n_workers} workers)…")
    with Pool(n_workers) as pool:
        valid_mols = list(
            tqdm(
                pool.imap(_validate_smiles, smiles_list, chunksize=chunk_size),
                total=len(smiles_list),
                desc="Validating molecules",
            )
        )
    validity = [m is not None for m in valid_mols]
    return valid_mols, validity


def _sparse_to_dense(
    spect,
    n_bins: int,
    bin_width: float,
    min_mz: float = 0.0,
) -> np.ndarray:
    """Convert RASSP sparse [(mz, intensity), ...] list to a dense binned array.

     Parameters
    ----------
     spect:     list of (mz, intensity) tuples from RASSP, or a numpy array
     n_bins:    number of output bins
     bin_width: bin width in Da
     min_mz:    lowest m/z

     Returns
    -------
     float32 array [n_bins], normalised to sum=1 (all-zeros if no peaks)
    """
    dense = np.zeros(n_bins, dtype=np.float32)

    if isinstance(spect, np.ndarray) and spect.ndim == 2:
        # Sparse (N, 2) array: columns are [mz, intensity]
        for mz, intensity in spect:
            bin_idx = int(round((mz - min_mz) / bin_width))
            if 0 <= bin_idx < n_bins:
                dense[bin_idx] += float(intensity)
    elif isinstance(spect, np.ndarray):
        # Dense 1D array — resize/copy
        length = min(len(spect), n_bins)
        dense[:length] = spect[:length].astype(np.float32)
    else:
        if not spect:
            return dense
        for mz, intensity in spect:
            bin_idx = int(round((mz - min_mz) / bin_width))
            if 0 <= bin_idx < n_bins:
                dense[bin_idx] += float(intensity)

    total = dense.sum()
    if total > 0:
        dense /= total

    return dense


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RASSP inference -> HDF5 (ICICLE-compatible format)"
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to RASSP .model checkpoint file",
    )
    parser.add_argument(
        "--meta",
        required=True,
        help="Path to RASSP .meta file (same basename as checkpoint)",
    )
    parser.add_argument(
        "--metadata",
        default=None,
        help="Path to ICICLE metadata.tsv (required when not using --candidates-pickle)",
    )
    parser.add_argument(
        "--splits",
        default=None,
        help="Path to ICICLE splits TSV (required when not using --candidates-pickle)",
    )
    parser.add_argument(
        "--candidates-pickle",
        type=str,
        default=None,
        help=(
            "Path to prebuilt candidates pickle "
            "(e.g. data/NIST2023_GCMS_main/retrieval/cands_pickled_scaffold_None_compat.pkl). "
            "When set, runs inference on all unique PubChem candidate SMILES instead of a split."
        ),
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output .hdf5 file path",
    )
    parser.add_argument(
        "--query-fraction",
        type=float,
        default=1.0,
        help="Fraction of test queries to use from --candidates-pickle (default: 1.0). "
             "Set <1.0 to subsample queries and their candidate SMILES for quick testing.",
    )
    parser.add_argument(
        "--eval-split",
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to run inference on (default: test)",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        default=False,
        help="Use GPU for inference",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Inference batch size (default: 32)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader workers (default: 4)",
    )
    parser.add_argument(
        "--data-parallel",
        action="store_true",
        default=False,
        help="Use nn.DataParallel across all visible GPUs",
    )
    parser.add_argument(
        "--max-n-atoms",
        type=int,
        default=48,
        help="Max atoms per molecule (default: 48 for FormulaNet)",
    )
    parser.add_argument(
        "--max-n-formula",
        type=int,
        default=4096,
        help="Max unique fragment formulae (default: 4096 for FormulaNet)",
    )
    parser.add_argument(
        "--max-n-subset",
        type=int,
        default=12288,
        help="Max vertex subset samples (default: 12288 for FormulaNet)",
    )
    parser.add_argument(
        "--min-mz",
        type=float,
        default=0.0,
        help="Minimum m/z for output bins (default: 0.0)",
    )
    parser.add_argument(
        "--max-mz",
        type=float,
        default=1000.0,
        help="Maximum m/z for output bins (default: 1000.0)",
    )
    parser.add_argument(
        "--bin-width",
        type=float,
        default=1.0,
        help="Bin width in Da (default: 1.0)",
    )
    args = parser.parse_args()

    print(f"Loading RASSP model from {args.checkpoint}")
    assert os.path.exists(args.checkpoint), (
        f"Checkpoint not found: {args.checkpoint}"
    )
    assert os.path.exists(args.meta), f"Meta file not found: {args.meta}"

    with open(args.meta, "rb") as f:
        meta = RenameUnpickler(f).load()

    feat_config = meta["featurize_config"]
    feat_config["MAX_N"] = args.max_n_atoms
    feat_config["explicit_formulae_config"]["max_formulae"] = (
        args.max_n_formula
    )
    feat_config["vert_subset_samples_n"] = args.max_n_subset

    use_gpu = args.gpu and torch.cuda.is_available()
    if args.gpu and not torch.cuda.is_available():
        print("WARNING: --gpu requested but CUDA not available, using CPU")

    use_data_parallel = use_gpu and args.data_parallel and torch.cuda.device_count() > 1
    predictor = netutil.PredModel(
        args.meta,
        args.checkpoint,
        USE_CUDA=use_gpu,
        data_parallel=use_data_parallel,
        featurize_config_update=feat_config,
    )
    n_gpus = torch.cuda.device_count() if use_data_parallel else (1 if use_gpu else 0)
    print(f"  Model loaded ({'GPU x' + str(n_gpus) if use_gpu else 'CPU'})")

    if args.candidates_pickle:
        print(
            f"Running inference on PubChem candidates from {args.candidates_pickle}"
        )
        mol_ids, smiles_list, inchikey_list = (
            load_smiles_from_candidates_pickle(args.candidates_pickle, args.query_fraction)
        )
        eval_split_label = "candidates"
    else:
        print(f"Loading {args.eval_split} split …")
        metadata_df = pd.read_csv(args.metadata, sep="\t")
        splits_df = pd.read_csv(args.splits, sep="\t")

        if "mol_id" not in metadata_df.columns and "spec" in metadata_df.columns:
            metadata_df = metadata_df.rename(columns={"spec": "mol_id"})

        split_mol_ids = set(
            splits_df[splits_df["split"] == args.eval_split]["mol_id"].astype(
                str
            )
        )
        mol_df = metadata_df[
            metadata_df["mol_id"].astype(str).isin(split_mol_ids)
        ].copy()
        print(f"  {len(mol_df)} molecules in {args.eval_split} split")

        smiles_col = (
            "standardized_smiles"
            if "standardized_smiles" in mol_df.columns
            else "smiles"
        )
        mol_ids = mol_df["mol_id"].astype(str).tolist()
        smiles_list = mol_df[smiles_col].tolist()
        if "inchi_key" in mol_df.columns:
            inchikey_list = mol_df["inchi_key"].tolist()
        elif "inchikey" in mol_df.columns:
            inchikey_list = mol_df["inchikey"].tolist()
        else:
            inchikey_list = [_smiles_to_inchikey(s) for s in smiles_list]
        eval_split_label = args.eval_split

    # Parse and validate molecules (parallel for large sets)
    valid_mols, validity = validate_smiles_parallel(
        smiles_list, args.max_n_atoms, args.max_n_formula
    )
    n_valid = sum(validity)
    print(f"  {n_valid}/{len(mol_ids)} molecules pass RASSP constraints")

    # Run inference only on valid mols
    valid_mol_list = [m for m, ok in zip(valid_mols, validity) if ok]

    print("Running RASSP inference …")
    preds = predictor.pred(
        valid_mol_list,
        progress_bar=True,
        normalize_pred=True,
        output_hist_bins=True,
        batch_size=args.batch_size,
        dataloader_config={
            "pin_memory": False,
            "num_workers": args.num_workers,
            "persistent_workers": False,
        },
    )
    pred_binned = preds[
        "pred_binned"
    ]  # list of arrays or (mz, intensity) tuples

    n_bins = int((args.max_mz - args.min_mz) / args.bin_width)
    mz_bins = (
        np.arange(n_bins) * args.bin_width + args.min_mz + 0.5 * args.bin_width
    ).astype(np.float32)

    out_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    print(f"Writing HDF5 -> {out_path}")

    pred_idx = 0
    n_nonzero = 0
    with h5py.File(out_path, "w") as hf:
        hf.attrs["min_mz"] = args.min_mz
        hf.attrs["max_mz"] = args.max_mz
        hf.attrs["bin_width"] = args.bin_width
        hf.attrs["model"] = "rassp"
        hf.attrs["eval_split"] = eval_split_label
        hf.attrs["n_samples"] = len(mol_ids)

        for i, (mol_id, smiles, inchi_key, is_valid) in enumerate(
            tqdm(
                zip(mol_ids, smiles_list, inchikey_list, validity),
                total=len(mol_ids),
                desc="Writing samples",
            )
        ):
            if is_valid:
                dense = _sparse_to_dense(
                    pred_binned[pred_idx], n_bins, args.bin_width, args.min_mz
                )
                pred_idx += 1
                if dense.sum() > 0:
                    n_nonzero += 1
            else:
                dense = np.zeros(n_bins, dtype=np.float32)

            grp = hf.create_group(mol_id)
            grp.attrs["smiles"] = smiles
            grp.attrs["inchi_key"] = (
                inchi_key
                if inchi_key
                and not (isinstance(inchi_key, float) and np.isnan(inchi_key))
                else ""
            )
            grp.create_dataset(
                "predicted_intensities", data=dense, compression="gzip"
            )
            grp.create_dataset("mz_bins", data=mz_bins, compression="gzip")

    print(
        f"Done. {len(mol_ids)} entries written "
        f"({n_nonzero} non-zero, {len(mol_ids) - n_valid} zero due to constraints)"
    )


if __name__ == "__main__":
    main()
