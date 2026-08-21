#!/usr/bin/env python
"""Run AIRI inference on SMILES data - direct model loading, streaming inference.

Loads AIRI model checkpoints directly (bypassing masskit predict.py subprocess)
and runs batched GPU inference. No intermediate parquet files are written, so
disk usage is minimal and inference starts immediately after each batch of
molecules is preprocessed.

Key improvements over the old subprocess approach:
- No intermediate parquet files (no disk space issues)
- Model loaded once per run (not once per RI type per chunk)
- Inference starts as soon as the first batch is ready (streaming pipeline)
- CPU preprocessing (shortest paths) runs in parallel with GPU inference

Usage:
    # Single RI type
    python infer_airi.py --input smiles.csv --output predictions.csv \\
        --model-path ./model.ckpt --ri-type StdNP

    # All 3 RI types in one pass (models run sequentially per batch)
    python infer_airi.py --input smiles.csv --output predictions.csv \\
        --model-path stdnp.ckpt --ri-type StdNP \\
        --model-path semistdnp.ckpt --ri-type SemiStdNP \\
        --model-path stdpolar.ckpt --ri-type StdPolar

    # With TMS derivatization
    python infer_airi.py --input smiles.csv --output predictions.csv \\
        --model-path ./model.ckpt --ri-type StdNP --derivatize
"""

import logging
import sys
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse
from rdkit import RDLogger
from scipy.sparse.csgraph import shortest_path as _scipy_sp

# This silences ALL warnings, including the hydrogen ones
lg = RDLogger.logger()
lg.setLevel(RDLogger.ERROR)
import click
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm

try:
    from masskit_ai.mol.small.models import path_utils
    from masskit_ai.mol.small.models.mol_graph import MolGraph
    from masskit_ai.spectrum.spectrum_lightning import SpectrumLightningModule
except ImportError as _err:
    print(
        f"Error: masskit_ai not found ({_err}).\n"
        "Please activate the masskit_ai conda environment before running this script.",
        file=sys.stderr,
    )
    sys.exit(1)

_worker_prop_args: Optional[dict] = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# Shortest path computation


def _ordered_pair(a1, a2):
    return (a2, a1) if a1 > a2 else (a1, a2)


def _get_ring_paths(rd_mol):
    rings_dict = {}
    ssr = [list(x) for x in Chem.GetSymmSSSR(rd_mol)]
    for ring in ssr:
        ring_sz = len(ring)
        is_aromatic = all(rd_mol.GetAtoms()[i].GetIsAromatic() for i in ring)
        for ring_idx, atom_idx in enumerate(ring):
            for other_idx in ring[ring_idx:]:
                pair = _ordered_pair(atom_idx, other_idx)
                entry = (ring_sz, is_aromatic)
                if pair not in rings_dict:
                    rings_dict[pair] = [entry]
                elif entry not in rings_dict[pair]:
                    rings_dict[pair].append(entry)
    return rings_dict


def _get_shortest_paths(rd_mol, max_path_length=5):
    n_atoms = rd_mol.GetNumAtoms()
    if n_atoms == 0:
        return {}, {}, {}

    # Build sparse adjacency matrix
    rows, cols = [], []
    for bond in rd_mol.GetBonds():
        a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        rows.extend([a1, a2])
        cols.extend([a2, a1])

    if not rows:
        return {}, {}, _get_ring_paths(rd_mol)

    graph = scipy.sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(n_atoms, n_atoms),
    )

    dist_matrix, predecessors = _scipy_sp(
        graph, directed=False, return_predecessors=True, unweighted=True
    )

    paths_dict = {}
    pointer_dict = {}

    for atom_idx in range(n_atoms):
        for other_idx in range(atom_idx + 1, n_atoms):
            d = dist_matrix[atom_idx, other_idx]
            if np.isinf(d):  # disconnected (different fragment)
                continue
            d = int(d)

            if d <= max_path_length:
                # MUST use GetShortestPath here: scipy predecessors choose a different
                # valid path for ring systems (verified), breaking feature consistency
                # with the training-time preprocessing that used GetShortestPath.
                sp = Chem.rdmolops.GetShortestPath(rd_mol, atom_idx, other_idx)
                paths_dict[(atom_idx, other_idx)] = sp
            else:
                # Far pairs — only need one pointer atom at max_path_length steps.
                # scipy predecessor traversal avoids a GetShortestPath call per pair.
                fwd = other_idx
                for _ in range(d - max_path_length):
                    fwd = predecessors[atom_idx, fwd]
                pointer_dict[(atom_idx, other_idx)] = int(fwd)

                rev = atom_idx
                for _ in range(d - max_path_length):
                    rev = predecessors[other_idx, rev]
                pointer_dict[(other_idx, atom_idx)] = int(rev)

    return paths_dict, pointer_dict, _get_ring_paths(rd_mol)


# TMS Derivatization TODO

_TMS_SMARTS = [
    ("[OH:1]", "[O:1][Si](C)(C)C"),
    ("[CX3:1](=[OX1:2])[OX2H1:3]", "[C:1](=[O:2])[O:3][Si](C)(C)C"),
    ("[NX3H2:1]", "[N:1]([Si](C)(C)C)[Si](C)(C)C"),
    ("[NX3H1:1]", "[N:1][Si](C)(C)C"),
    ("[SH:1]", "[S:1][Si](C)(C)C"),
]


def _derivatize_tms(mol):
    if mol is None:
        return None
    result = mol
    for smarts_r, smarts_p in _TMS_SMARTS:
        try:
            rxn = AllChem.ReactionFromSmarts(f"{smarts_r}>>{smarts_p}")
            while True:
                products = rxn.RunReactants((result,))
                if not products:
                    break
                result = products[0][0]
                try:
                    Chem.SanitizeMol(result)
                except Exception:
                    break
        except Exception:
            continue
    return result


# Input Reading


def _looks_like_smiles(text: str) -> bool:
    if not text or not isinstance(text, str):
        return False
    smiles_chars = set("CNOSPFIBrcnopsfibl[]()=#@+-./\\0123456789")
    text_chars = set(text.replace(" ", ""))
    if len(text_chars - smiles_chars) <= 2 and len(text) > 2:
        headers = {
            "smiles",
            "smile",
            "molecule",
            "compound",
            "id",
            "name",
            "index",
        }
        return text.lower() not in headers
    return False


def read_smiles_input(
    input_file: str, smiles_column: str = None
) -> pd.DataFrame:
    """Read SMILES from CSV or plain text file.

    The SMILES column is always renamed to 'input_smiles'.
    """
    path = Path(input_file)
    if path.suffix.lower() in (".csv", ".tsv"):
        sep = "\t" if path.suffix.lower() == ".tsv" else ","
        df = pd.read_csv(path, sep=sep)

        if len(df.columns) == 1 and _looks_like_smiles(df.columns[0]):
            df = pd.read_csv(
                path, sep=sep, header=None, names=["input_smiles"]
            )
            logger.info(
                f"Detected headerless SMILES file ({len(df):,} molecules)"
            )
            return df

        if smiles_column and smiles_column in df.columns:
            pass
        elif "smiles" in df.columns:
            smiles_column = "smiles"
        elif "SMILES" in df.columns:
            smiles_column = "SMILES"
        elif "molecules" in df.columns:
            smiles_column = "molecules"
        else:
            smiles_column = df.columns[0]
            logger.warning(
                f"SMILES column not found, using first column: '{smiles_column}'"
            )

        return df.rename(columns={smiles_column: "input_smiles"})

    with open(path) as f:
        smiles_list = [l.strip() for l in f if l.strip()]
    return pd.DataFrame({"input_smiles": smiles_list})


# molecue processing


def _process_smiles_worker(args: tuple) -> dict:
    """Parse SMILES -> RDKit mol -> shortest paths -> path feature matrix.

    Runs entirely in worker processes (no CUDA). The O(n²) path feature
    assembly (path_utils.get_path_input) happens here in parallel across all
    worker cores, rather than serially in the main thread.
    """
    idx, smiles, derivatize, max_path_length = args
    result = {
        "id": idx,
        "original_smiles": smiles,
        "smiles": smiles,
        "mol": None,
        "n_atoms": 0,
        "path_input": None,  # [n_atoms, n_atoms, n_features] numpy array
        "path_mask": None,  # [n_atoms, n_atoms] numpy array
        "error": None,
    }

    if pd.isna(smiles) or not str(smiles).strip():
        result["error"] = "Empty SMILES"
        return result

    smiles = str(smiles).strip()
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            result["error"] = "Invalid SMILES"
            return result

        if derivatize:
            mol = _derivatize_tms(mol)
            if mol is None:
                result["error"] = "Derivatization failed"
                return result
            result["smiles"] = Chem.MolToSmiles(mol)

        shortest_paths = _get_shortest_paths(mol, max_path_length)
        n_atoms = mol.GetNumAtoms()

        # Assemble the [n_atoms × n_atoms × n_features] path feature matrix
        # here in the worker, in parallel with all other workers.
        # _worker_prop_args is inherited from the parent via fork.
        pi, pm = path_utils.get_path_input(
            [mol],
            [shortest_paths],
            n_atoms,
            _worker_prop_args,
            output_tensor=False,
        )
        result["mol"] = mol
        result["n_atoms"] = n_atoms
        result["path_input"] = pi[0]  # drop leading batch dim -> [n, n, f]
        result["path_mask"] = pm[0]

    except Exception as exc:
        result["error"] = str(exc)

    return result


# model loading


def load_airi_model(checkpoint_path: str) -> Tuple:
    """Load an AIRI checkpoint with SpectrumLightningModule.

    Returns (model, prop_predictor_args_dict, normalization).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"  Loading {Path(checkpoint_path).name} -> {device}")

    model = SpectrumLightningModule.load_from_checkpoint(
        checkpoint_path, map_location=device
    )
    model.to(device)
    model.eval()

    prop_args = dict(model.config.ml.model.PropPredictor)
    normalization = float(model.config.ms.get("normalization", 10000.0))
    logger.info(
        f"    max_path_length={prop_args.get('max_path_length', '?')} | "
        f"normalization={normalization}"
    )
    return model, prop_args, normalization


# Batch Embedding + GPU Inference


def _infer_batch(
    batch: List[dict],
    models_info: Dict[str, Tuple],
    prop_predictor_args: dict,
    ri_results: Dict[str, np.ndarray],
) -> None:
    """Collate pre-embedded molecules and run GPU inference.

    path_input / path_mask arrays were already computed by workers, so this
    function only pads to a uniform atom count, creates MolGraph, and runs
    each model's forward pass. No O(n²) Python work here.

    Args:
        batch: list of successful _process_smiles_worker results.
        models_info: ri_type -> (model, prop_predictor_args, normalization).
        prop_predictor_args: shared embedding config.
        ri_results: ri_type -> ndarray[n_total], updated in-place.
    """
    if not batch:
        return

    batch_mols = [item["mol"] for item in batch]
    batch_path_inputs = [item["path_input"] for item in batch]
    batch_path_masks = [item["path_mask"] for item in batch]
    batch_n_atoms = [item["n_atoms"] for item in batch]
    valid_ids = [item["id"] for item in batch]

    max_atoms = max(batch_n_atoms)
    # merge_path_inputs pads each [n_i, n_i, f] array to [max_atoms, max_atoms, f]
    # and stacks into CPU tensors (avoids CUDA in forked-process context).
    padded_inputs, padded_masks = path_utils.merge_path_inputs(
        batch_path_inputs, batch_path_masks, max_atoms, prop_predictor_args
    )
    mol_graph = MolGraph(batch_mols, padded_inputs, padded_masks)

    for ri_type, (model, _, normalization) in models_info.items():
        try:
            with torch.no_grad():
                output = model([mol_graph])
                preds = output.y_prime.detach().cpu().numpy() * normalization
            for gid, pred in zip(valid_ids, preds):
                ri_results[ri_type][gid] = float(pred)
        except Exception as exc:
            logger.error(
                f"Inference error for {ri_type} (batch size {len(batch)}): {exc}"
            )


# Main Inference Pipeline


def run_inference(
    df_input: pd.DataFrame,
    checkpoints: List[Tuple[str, str]],
    derivatize: bool,
    max_path_length: int,
    batch_size: int,
    num_workers: int,
) -> pd.DataFrame:
    """Stream SMILES through parallel CPU preprocessing and batched GPU
    inference.

    Pipeline (avoids CUDA + fork deadlock):
        1. Load first checkpoint on CPU -> extract PropPredictor config (no CUDA)
        2. Set _worker_prop_args global (inherited by forked workers)
        3. Pool(num_workers) — fork workers; they inherit masskit_ai imports
                               and _worker_prop_args via copy-on-write
        4. pool.imap_unordered — workers run: SMILES -> mol -> shortest_paths
                                  -> path_utils.get_path_input (O(n²), parallel!)
        5. load_airi_model()  — CUDA initialised HERE, after fork
        6. iterate imap       — main process: merge_path_inputs + GPU forward pass

    The key improvement: get_path_input (O(n²) nested Python loops) now runs in
    parallel across all worker cores instead of serially in the main thread.

    Memory is bounded: at most batch_size mol objects live in the buffer at once.
    No intermediate files are written.
    """
    global _worker_prop_args

    n = len(df_input)

    results_smiles = list(df_input["input_smiles"])
    results_errors: List[Optional[str]] = [None] * n

    # Extract PropPredictor config without initialising CUDA
    # load_from_checkpoint with map_location='cpu' puts all tensors on CPU and
    # never calls .to('cuda'), so no CUDA context is created. Workers inherit
    # _worker_prop_args via fork and use it inside _process_smiles_worker to
    # call path_utils.get_path_input, parallelising the O(n²) feature assembly.
    first_ckpt = checkpoints[0][1]
    logger.info(
        f"Extracting model config from {Path(first_ckpt).name} (CPU)..."
    )
    _tmp = SpectrumLightningModule.load_from_checkpoint(
        first_ckpt, map_location="cpu"
    )
    _worker_prop_args = dict(_tmp.config.ml.model.PropPredictor)
    # Override max_path_length from model config (takes precedence over CLI default)
    max_path_length = _worker_prop_args.get("max_path_length", max_path_length)
    logger.info(
        f"  max_path_length={max_path_length} "
        f"| ring_embed={_worker_prop_args.get('ring_embed')} "
        f"| p_embed={_worker_prop_args.get('p_embed')}"
    )
    del _tmp  # free ~500MB of CPU-resident model weights

    args_gen = (
        (i, smiles, derivatize, max_path_length)
        for i, smiles in enumerate(df_input["input_smiles"])
    )

    n_failed = 0
    buffer: List[dict] = []

    with Pool(num_workers) as pool:
        # Fork workers here. They inherit _worker_prop_args and all masskit_ai
        # imports via copy-on-write, so they can call path_utils.get_path_input
        # without any re-import overhead. No CUDA context exists yet in the
        # parent process (map_location='cpu' above kept everything on CPU),
        # so workers are fork-safe.
        imap_iter = pool.imap_unordered(
            _process_smiles_worker, args_gen, chunksize=200
        )

        # Load models onto GPU — CUDA is initialised HERE, after fork.
        logger.info("Loading model checkpoint(s)...")
        models_info: Dict[str, Tuple] = {}
        for label, ckpt in checkpoints:
            models_info[label] = load_airi_model(ckpt)

        if len(models_info) > 1:
            all_args = [info[1] for info in models_info.values()]
            if not all(a == all_args[0] for a in all_args[1:]):
                logger.warning(
                    "Checkpoints have different PropPredictor configs. "
                    "Using first checkpoint's config for embedding — results may be inaccurate."
                )

        ri_types = list(models_info.keys())
        shared_prop_args = next(iter(models_info.values()))[1]
        ri_results = {
            rt: np.full(n, np.nan, dtype=np.float64) for rt in ri_types
        }

        # Step 4: consume preprocessing results + GPU inference
        for r in tqdm(imap_iter, total=n, desc="Preprocessing + inference"):
            idx = r["id"]
            if r["error"]:
                results_errors[idx] = r["error"]
                n_failed += 1
            else:
                results_smiles[idx] = r["smiles"]
                buffer.append(r)

            if len(buffer) >= batch_size:
                _infer_batch(buffer, models_info, shared_prop_args, ri_results)
                buffer.clear()

        if buffer:
            _infer_batch(buffer, models_info, shared_prop_args, ri_results)

    if n_failed:
        logger.warning(
            f"Failed to preprocess {n_failed:,} molecules (left as NaN in output)"
        )

    df_out = df_input.copy()
    df_out["smiles"] = results_smiles
    df_out["error"] = results_errors
    for rt in ri_types:
        df_out[f"ri_{rt}"] = ri_results[rt]

    if derivatize:
        df_out = df_out.rename(columns={"smiles": "derivatized_smiles"})

    return df_out, models_info


# CLI


@click.command()
@click.option(
    "--input",
    "-i",
    "input_file",
    required=True,
    type=click.Path(exists=True),
    help="Input file with SMILES (CSV or plain text).",
)
@click.option(
    "--output",
    "-o",
    "output_file",
    required=True,
    type=click.Path(),
    help="Output CSV file for predictions.",
)
@click.option(
    "--smiles-column",
    default=None,
    help="SMILES column name in CSV (auto-detected if not given).",
)
@click.option(
    "--model-path",
    multiple=True,
    help="Path to a model checkpoint (.ckpt). Repeat for multiple RI types.",
)
@click.option(
    "--ri-type",
    multiple=True,
    type=click.Choice(["StdNP", "SemiStdNP", "StdPolar"]),
    help="RI type label for each --model-path (same order). Produces ri_<type> column.",
)
@click.option(
    "--derivatize",
    is_flag=True,
    help="Apply TMS derivatization before prediction.",
)
@click.option(
    "--max-path-length",
    default=5,
    type=int,
    help="Maximum shortest-path length for atom-pair features (default: 5).",
)
@click.option(
    "--num-workers",
    default=8,
    type=int,
    help="CPU workers for parallel SMILES preprocessing (default: 8).",
)
@click.option(
    "--batch-size",
    default=256,
    type=int,
    help="Molecules per GPU inference batch (default: 256). "
    "Larger = better GPU utilisation but more VRAM.",
)
def main(
    input_file: str,
    output_file: str,
    smiles_column: str,
    model_path: Tuple[str],
    ri_type: Tuple[str],
    derivatize: bool,
    max_path_length: int,
    num_workers: int,
    batch_size: int,
):
    """Predict retention index values for SMILES using AIRI neural network
    models.

    Models are loaded directly from checkpoints — no subprocess, no parquet
    files. Molecules are preprocessed in parallel on CPU and inference runs in
    batches on GPU.
    """
    logger.info("=" * 60)
    logger.info("AIRI Retention Index Prediction (direct inference)")
    logger.info("=" * 60)
    logger.info(f"Input:       {input_file}")
    logger.info(f"Output:      {output_file}")
    logger.info(f"Batch size:  {batch_size}")
    logger.info(f"CPU workers: {num_workers}")
    logger.info(f"Derivatize:  {derivatize}")
    # NOTE: do NOT call torch.cuda.is_available() here — that initialises CUDA
    # before workers are forked, which causes them to hang. Device is logged
    # inside load_airi_model() after the Pool is created.

    if not model_path:
        raise click.UsageError("At least one --model-path is required.")

    if ri_type and len(ri_type) != len(model_path):
        raise click.UsageError(
            f"--ri-type count ({len(ri_type)}) must match "
            f"--model-path count ({len(model_path)})"
        )

    # Build (label, path) pairs — do NOT load models yet (no CUDA before fork)
    checkpoints: List[Tuple[str, str]] = [
        (ri_type[i] if ri_type else f"model_{i}", ckpt)
        for i, ckpt in enumerate(model_path)
    ]

    # Read input
    logger.info("\nReading input...")
    df_input = read_smiles_input(input_file, smiles_column)
    logger.info(f"  {len(df_input):,} molecules")

    # Run streaming inference.
    # Pool is created inside, BEFORE model loading, so workers are forked
    # with no CUDA context (avoids CUDA+fork deadlock).
    logger.info("\nRunning inference...")
    df_output, models_info = run_inference(
        df_input,
        checkpoints,
        derivatize=derivatize,
        max_path_length=max_path_length,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # Save output
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    df_output.to_csv(output_file, index=False)
    logger.info(f"\nSaved -> {output_file}")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Summary")
    logger.info("=" * 60)
    logger.info(f"Total: {len(df_output):,}")
    for rt in models_info:
        col = f"ri_{rt}"
        if col in df_output.columns:
            vals = df_output[col].dropna()
            if len(vals):
                logger.info(
                    f"  {rt}: {len(vals):,} predicted | "
                    f"range=[{vals.min():.0f}, {vals.max():.0f}] | mean={vals.mean():.0f}"
                )
            else:
                logger.info(f"  {rt}: 0 predictions")

    logger.info("\nDone!")


if __name__ == "__main__":
    main()
