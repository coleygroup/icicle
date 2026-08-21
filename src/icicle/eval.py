"""Evaluation script for trained models."""

import logging
import os
import time
import warnings

warnings.filterwarnings("ignore", message=".*lazyInitCUDA.*")
warnings.filterwarnings("ignore", message=".*lazy_init.*")
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"
import pickle
from datetime import timedelta
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional

import h5py
import hydra
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import wandb
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import WandbLogger
from rdkit import RDLogger
from tqdm import tqdm

from icicle.data.data_module import MassSpecDataModule
from icicle.utils.eval.data_loading import (
    get_retrieval_candidates_from_labels,
    retrieve_ground_truth,
)
from icicle.utils.eval.eval_parallel import (
    EvaluationCheckpoint,
    gather_pickled_data,
    gather_results,
    split_workload_by_rank,
)
from icicle.utils.eval.hdf5_io import save_predictions_as_hdf5
from icicle.utils.eval.metrics_computation import (
    compute_metrics_for_spectra,
    get_default_metrics,
    get_similarity_column_names,
)
from icicle.utils.eval.model_utils import get_or_predict_spectrum
from icicle.utils.eval.ranking_utils import (
    DISTANCE_METRICS,
    compute_rankings_for_metrics,
    compute_retrieval_metrics,
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

RDLogger.DisableLog("rdApp.*")


def _precompute_candidate_spectra(
    model: Any,
    candidates_df: pd.DataFrame,
    smiles_column: str,
    model_inference_params: Dict,
    num_gpu_workers: int = 0,
    batch_size: int = 64,
    cache_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Pre-compute spectra for all unique candidate SMILES.

    Uses batched GPU inference (predict_from_smiles_parallel) when available,
    falling back to the serial path for models that don't support it.

    Args:
        model: The model to use for predictions
        candidates_df: DataFrame containing candidate molecules
        smiles_column: Name of the column containing SMILES strings
        model_inference_params: Parameters for model inference
        num_gpu_workers: DataLoader workers for CPU preprocessing
        batch_size: Molecules per GPU forward pass
        cache_path: If given, load predictions from this pickle if it exists
            (skipping computation entirely), otherwise compute and save here.
            Callers must pick a path unique to the (checkpoint, candidate set)
            pair -- the cache is not otherwise invalidated.

    Returns:
        Dictionary mapping SMILES to predicted spectra
    """
    if cache_path is not None and cache_path.exists():
        logging.info(f"Loading cached candidate spectra from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    unique_smiles = candidates_df[smiles_column].unique().tolist()
    logging.info(
        f"Pre-computing spectra for {len(unique_smiles)} unique candidates "
        f"(batch_size={batch_size}, num_workers={num_gpu_workers})"
    )

    device = model_inference_params.get("device", torch.device("cpu"))
    extra_kwargs = {
        k: v for k, v in model_inference_params.items() if k != "device"
    }

    if hasattr(model, "stream_predict_from_smiles"):
        predictions_map = {}
        idx = 0
        with tqdm(
            total=len(unique_smiles),
            desc="Pre-computing candidate spectra",
            unit="mol",
        ) as pbar:
            for batch in model.stream_predict_from_smiles(
                unique_smiles,
                device=str(device),
                batch_size=batch_size,
                num_workers=num_gpu_workers,
                populate_fragments=False,
                **extra_kwargs,
            ):
                for r in batch:
                    # Key by the input SMILES (batch position), not
                    # r["smiles"] -- failed predictions always come back as
                    # _empty_spectrum_result(), whose "smiles" field is
                    # hardcoded to "" regardless of the input. Keying on
                    # that collapses every failed candidate onto a single
                    # "" cache entry, so the real SMILES is never cached and
                    # every later lookup falls through to a slow, uncached,
                    # single-molecule predict_from_smiles call instead.
                    predictions_map[unique_smiles[idx]] = (
                        r["intensities"]
                        if r.get("intensities") is not None
                        and r.get("intensities").any()
                        else None
                    )
                    idx += 1
                pbar.update(len(batch))
    else:
        predictions_map = {}
        for smiles in tqdm(unique_smiles, desc="Computing candidate spectra"):
            try:
                predicted_result = model.predict_from_smiles(
                    smiles=smiles, **model_inference_params
                )
                if (
                    predicted_result.get("smiles") != ""
                    and predicted_result.get("intensities") is not None
                ):
                    predictions_map[smiles] = predicted_result["intensities"]
                else:
                    predictions_map[smiles] = None
            except Exception as e:
                logging.warning(
                    f"Failed to predict spectrum for {smiles}: {e}"
                )
                predictions_map[smiles] = None

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(predictions_map, f)
        logging.info(f"Cached candidate spectra to {cache_path}")

    return predictions_map


def load_model(
    config: DictConfig, device: torch.device, compile_model: bool = True
):
    """Helper function to load the model.

    Args:
        config: Configuration object
        device: Device to load model on
        compile_model: Whether to use torch.compile for speedup (if available)
    """

    model = hydra.utils.instantiate(config.eval.model.architecture)

    model.to(device)
    model.eval()

    # Apply torch.compile if available and enabled (PyTorch 2.0+)
    if (
        compile_model
        and hasattr(torch, "compile")
        and config.system.get("compile_mode")
    ):
        try:
            model = torch.compile(model, mode=config.system.compile_mode)
            logging.info(
                f"Model compiled with mode: {config.system.compile_mode}"
            )
        except Exception as e:
            logging.warning(
                f"Failed to compile model: {e}. Continuing without compilation."
            )

    return model


def run_similarity_evaluation(
    model: pl.LightningModule,
    ground_truth_map: Dict,
    eval_smiles_list: List[str],
    config: DictConfig,
    save_dir: Path,
    device: torch.device,
    checkpoint_manager: Optional[EvaluationCheckpoint] = None,
    rank: int = 0,
    world_size: int = 1,
) -> pd.DataFrame:
    """Runs the similarity evaluation for the model.

    Computes all specified similarity metrics between predicted and ground
    truth spectra. Supports multi-GPU parallel evaluation and checkpoint/resume.

    Args:
        model: Model to evaluate
        ground_truth_map: Dictionary mapping SMILES to ground truth data
        eval_smiles_list: List of SMILES to evaluate
        config: Configuration object
        save_dir: Directory to save results
        device: Device to run on
        checkpoint_manager: Optional checkpoint manager for resume support
        rank: Process rank for multi-GPU
        world_size: Total number of processes
    """
    logging.info(
        f"Starting similarity evaluation on rank {rank}/{world_size}..."
    )

    # Load checkpoint if available
    completed_smiles = []
    partial_results_df = None
    if checkpoint_manager is not None:
        completed_smiles, partial_results_df = (
            checkpoint_manager.load_checkpoint("similarity")
        )

    # Filter out already completed SMILES
    if completed_smiles:
        remaining_smiles = [
            s for s in eval_smiles_list if s not in completed_smiles
        ]
        logging.info(
            f"Resuming from checkpoint: {len(completed_smiles)} already completed, "
            f"{len(remaining_smiles)} remaining"
        )
    else:
        remaining_smiles = eval_smiles_list

    # Split workload across ranks
    rank_smiles = split_workload_by_rank(remaining_smiles, rank, world_size)

    # This list will hold dictionaries, each representing a row in the final CSV DataFrame
    similarity_results_df_rows = []
    # This list will hold data for HDF5 saving (predicted_spec, true_spec, metadata, metrics)
    spectra_data_for_hdf5 = []

    min_mz = config.eval.data_module.dataset_config.min_mz
    max_mz = config.eval.data_module.dataset_config.max_mz
    bin_width = config.eval.data_module.dataset_config.bin_width
    num_bins = int((max_mz - min_mz) / bin_width)
    mz_values_for_metrics = np.linspace(
        min_mz, max_mz, num_bins, endpoint=False
    ).astype(np.float32)

    # Extract fragmentation kwargs (device passed separately to parallel method)
    fragmentation_kwargs = {}
    if config.eval.model.architecture.get("max_nodes"):
        fragmentation_kwargs["max_nodes"] = (
            config.eval.model.architecture.max_nodes
        )
    if config.eval.model.architecture.get("threshold"):
        fragmentation_kwargs["threshold"] = (
            config.eval.model.architecture.threshold
        )

    # Batch-predict all SMILES at once using parallel CPU preprocessing + batched GPU forward
    batch_size = config.eval.similarity.get("batch_size", 64)
    num_workers = config.eval.similarity.get("num_workers", 0)

    t0 = time.perf_counter()
    if hasattr(model, "predict_from_smiles_parallel"):
        logging.info(
            f"Rank {rank}: batch-predicting {len(rank_smiles)} spectra "
            f"(batch_size={batch_size}, num_workers={num_workers})"
        )
        all_preds = model.predict_from_smiles_parallel(
            rank_smiles,
            device=str(device),
            batch_size=batch_size,
            num_workers=num_workers,
            **fragmentation_kwargs,
        )
    else:
        model_inference_params = {"device": device, **fragmentation_kwargs}
        all_preds = [
            model.predict_from_smiles(smiles=smi, **model_inference_params)
            for smi in tqdm(
                rank_smiles,
                desc=f"Rank {rank} - Predicting",
                disable=(rank != 0),
            )
        ]
    inference_wall_time = time.perf_counter() - t0
    n_predicted = len(rank_smiles)
    ms_per_mol = (
        (inference_wall_time / n_predicted * 1000) if n_predicted > 0 else 0.0
    )
    logging.info(
        f"Rank {rank}: inference done in {inference_wall_time:.1f}s "
        f"({ms_per_mol:.2f} ms/mol)"
    )

    # Checkpoint frequency (save every N items)
    checkpoint_frequency = config.eval.get("checkpoint_frequency", 100)
    processed_count = 0
    completed_smiles_list = list(completed_smiles) if completed_smiles else []

    for smiles, predicted_result in tqdm(
        zip(rank_smiles, all_preds),
        total=len(rank_smiles),
        desc=f"Rank {rank} - Calculating Similarity",
        disable=(rank != 0),
    ):
        ground_truth_data = ground_truth_map.get(smiles)

        if predicted_result["smiles"] == "" or ground_truth_data is None:
            logging.warning(
                f"Skipping similarity for {smiles}: missing prediction or ground truth. Predicted result: {predicted_result['smiles'] != ''}, GT: {ground_truth_data is not None}"
            )
            continue

        pred_spec = predicted_result[
            "intensities"
        ]  # Assuming 'intensities' key from _format_spectrum_result
        true_spec = ground_truth_data["intensities"]
        mol_id = ground_truth_data["mol_id"]
        inchi_key = ground_truth_data["inchi_key"]
        # Check for empty spectra or shape mismatch
        if (
            pred_spec is None
            or true_spec is None
            or pred_spec.sum() == 0
            or true_spec.sum() == 0
            or pred_spec.shape != true_spec.shape
        ):
            logging.warning(
                f"Skipping similarity for {smiles}: Invalid or empty spectra or shape mismatch. Predicted sum: {pred_spec.sum()}, True sum: {true_spec.sum()}, Shapes: {pred_spec.shape} vs {true_spec.shape}"
            )
            continue

        current_metrics = {
            "smiles": smiles,
            "mol_id": mol_id,
        }

        # Dynamically compute metrics based on config
        metrics_to_compute = config.eval.similarity.get(
            "metrics_to_compute", []
        )
        weighted_cosine_schemes = config.eval.similarity.get(
            "weighted_cosine_schemes", []
        )
        composite_similarity_schemes = config.eval.similarity.get(
            "composite_similarity_schemes", []
        )

        # Use helper function to compute all metrics
        computed_metrics = compute_metrics_for_spectra(
            pred_spec,
            true_spec,
            metrics_to_compute,
            weighted_cosine_schemes,
            composite_similarity_schemes,
            mz_values_for_metrics,
        )
        current_metrics.update(computed_metrics)

        similarity_results_df_rows.append(current_metrics)

        # Prepare data for HDF5 saving
        spectra_data_for_hdf5.append(
            {
                "smiles": smiles,
                "mol_id": mol_id,
                "inchi_key": inchi_key,
                "predicted_mz_bins": predicted_result["mz_bins"],
                "predicted_intensities": predicted_result["intensities"],
                "ground_truth_mz_bins": ground_truth_data["mz_bins"],
                "ground_truth_intensities": ground_truth_data["intensities"],
                "metrics": current_metrics,
            }
        )

        # Track completion and checkpoint periodically
        completed_smiles_list.append(smiles)
        processed_count += 1

        if (
            checkpoint_manager is not None
            and processed_count % checkpoint_frequency == 0
        ):
            # Save intermediate checkpoint
            temp_df = pd.DataFrame(similarity_results_df_rows)
            checkpoint_manager.save_checkpoint(
                completed_smiles_list, temp_df, "similarity"
            )

    # Save final checkpoint for this rank
    if checkpoint_manager is not None and processed_count > 0:
        temp_df = pd.DataFrame(similarity_results_df_rows)
        checkpoint_manager.save_checkpoint(
            completed_smiles_list, temp_df, "similarity"
        )

    # Convert to DataFrame (local results for this rank)
    df_results = pd.DataFrame(similarity_results_df_rows)

    # Gather results from all ranks if multi-GPU (file-based, no NCCL barrier
    # -- an imbalanced/slow rank can never trigger a watchdog timeout here)
    if world_size > 1:
        df_results = gather_results(
            df_results, rank, world_size, save_dir=save_dir, label="similarity"
        )
        spectra_data_for_hdf5 = gather_pickled_data(
            spectra_data_for_hdf5,
            rank,
            world_size,
            save_dir=save_dir,
            label="similarity_spectra",
        )
        # Only rank 0 continues with saving
        if rank != 0:
            return pd.DataFrame()

    # Combine with partial results if resuming
    if partial_results_df is not None and len(partial_results_df) > 0:
        df_results = pd.concat(
            [partial_results_df, df_results], ignore_index=True
        )
        logging.info(
            f"Combined with {len(partial_results_df)} previous results"
        )

    output_path_csv = save_dir / "similarity_results.csv"
    df_results.to_csv(output_path_csv, index=False)
    logging.info(f"Similarity results saved to CSV: {output_path_csv}")

    # Save all spectra and metrics to HDF5
    hdf5_output_path = save_dir / "all_evaluation_spectra.hdf5"
    save_predictions_as_hdf5(spectra_data_for_hdf5, hdf5_output_path)

    # Calculate and log summary statistics to Wandb and a text file
    summary_metrics = {}
    for col in df_results.columns:
        if col not in ["smiles", "mol_id"]:
            valid_scores = df_results[col].dropna()
            if not valid_scores.empty:
                mean_val = valid_scores.mean()
                std_val = valid_scores.std()
                summary_metrics[f"avg_{col}"] = mean_val
                summary_metrics[f"std_{col}"] = std_val
                wandb.log({f"eval/similarity/{col}/mean": mean_val})
                # wandb.log({f"eval/similarity/{col}/std": std_val})
            else:
                logging.warning(f"No valid scores for {col}.")

    # Save summary metrics to a text file
    n_total = len(df_results)
    effective_ms_per_mol = (
        (inference_wall_time / n_total * 1000) if n_total > 0 else 0.0
    )
    with open(save_dir / "similarity_summary_metrics.txt", "w") as f:
        f.write(f"n_molecules_total: {n_total}\n")
        f.write(f"n_molecules_per_rank: {n_predicted}\n")
        f.write(f"world_size: {world_size}\n")
        f.write(f"inference_wall_time_s: {inference_wall_time:.2f}\n")
        f.write(f"inference_ms_per_mol_per_rank: {ms_per_mol:.2f}\n")
        f.write(
            f"inference_ms_per_mol_effective: {effective_ms_per_mol:.2f}\n"
        )
        for key, value in summary_metrics.items():
            f.write(f"{key}: {value:.4f}\n")
    wandb.log({"eval/inference_ms_per_mol_effective": effective_ms_per_mol})
    wandb.log({"eval/inference_ms_per_mol_per_rank": ms_per_mol})
    logging.info("Similarity summary metrics calculated and logged.")

    return df_results


def _get_pubchem_retrieval_candidates(
    test_labels: pd.DataFrame,
    candidates_pickle_path: str,
) -> pd.DataFrame:
    """Build retrieval candidate DataFrame from prebuilt PubChem pickle.

    Uses the same pickle format produced by retrieval_create_retrieval_lists.py
    (target InChIKey -> {smiles, formula, cands: array of non-stereo SMILES}).
    Returns a DataFrame in the same format as get_retrieval_candidates_from_labels:
    columns spec, inchikey, standardized_smiles, formula, is_decoy.
    """
    logging.info(
        f"Loading PubChem retrieval candidates from {candidates_pickle_path}"
    )
    with open(candidates_pickle_path, "rb") as f:
        cands_dict = pickle.load(f)

    # Build lookup from inchi_key -> test row
    ik_to_test_row = {}
    for _, row in test_labels.iterrows():
        ik_to_test_row[row["inchikey"]] = row

    rows = []
    missing = 0
    for target_ik, test_row in ik_to_test_row.items():
        if target_ik not in cands_dict:
            missing += 1
            continue
        entry = cands_dict[target_ik]
        spec = test_row["spec"]
        formula = entry.get("formula", test_row.get("formula", ""))

        # Correct answer (the target itself, non-stereo SMILES)
        rows.append(
            {
                "spec": spec,
                "inchikey": target_ik,
                "standardized_smiles": entry["smiles"],
                "formula": formula,
                "is_decoy": False,
            }
        )

        # Decoy candidates from PubChem (non-stereo SMILES)
        for cand_ns_smi in entry["cands"]:
            rows.append(
                {
                    "spec": spec,
                    "inchikey": "",  # not available for non-stereo PubChem candidates
                    "standardized_smiles": cand_ns_smi,
                    "formula": formula,
                    "is_decoy": True,
                }
            )

    if missing > 0:
        logging.warning(
            f"{missing} test targets not found in candidates pickle, skipped"
        )

    result = pd.DataFrame(rows)
    if len(result) > 0:
        per_spec = result.groupby("spec").size()
        logging.info(
            f"PubChem candidates: {len(per_spec)} queries, "
            f"min={per_spec.min()} max={per_spec.max()} mean={per_spec.mean():.1f} candidates"
        )
    return result


def run_retrieval_with_formula_evaluation(
    model: pl.LightningModule,
    ground_truth_map: Dict,
    eval_smiles_list: List[str],
    config: DictConfig,
    save_dir: Path,
    device: torch.device,
    checkpoint_manager: Optional[EvaluationCheckpoint] = None,
    rank: int = 0,
    world_size: int = 1,
) -> pd.DataFrame:
    """Runs the retrieval evaluation for the model using molecular formula
    filtering.

    For each test spectrum, retrieves all candidates with the same molecular formula,
    predicts spectra for each candidate, and ranks them by similarity to the
    experimental spectrum. Supports multi-GPU parallel evaluation and checkpoint/resume.

    Parameters
    ----------
    model : pl.LightningModule
        The trained model for spectrum prediction
    ground_truth_map : Dict
        Dictionary mapping SMILES to ground truth spectrum data
    eval_smiles_list : List[str]
        List of SMILES strings to evaluate
    config : DictConfig
        Hydra configuration object
    save_dir : Path
        Directory to save results
    device : torch.device
        Device to run inference on
    checkpoint_manager : Optional[EvaluationCheckpoint]
        Optional checkpoint manager for resume support
    rank : int
        Process rank for multi-GPU
    world_size : int
        Total number of processes

    Returns
    -------
    pd.DataFrame
        DataFrame with retrieval results and rankings
    """
    logging.info(
        f"Starting retrieval with formula evaluation on rank {rank}/{world_size}..."
    )

    # Load labels to get formula information
    labels_df = pd.read_csv(config.eval.data_module.labels_path, sep="\t")

    # Check required columns exist
    required_cols = ["mol_id", "standardized_smiles", "inchi_key", "formula"]
    missing_cols = [c for c in required_cols if c not in labels_df.columns]
    if missing_cols:
        raise ValueError(
            f"Labels file missing required columns: {missing_cols}"
        )

    # Drop duplicate 'inchikey' column if it exists (some datasets have both inchi_key and inchikey)
    if "inchikey" in labels_df.columns and "inchi_key" in labels_df.columns:
        labels_df = labels_df.drop(columns=["inchikey"])

    # Rename columns for consistency with legacy code
    labels_df = labels_df.rename(
        columns={"mol_id": "spec", "inchi_key": "inchikey"}
    )

    # Create test labels from eval_smiles_list
    test_labels = labels_df[
        labels_df["standardized_smiles"].isin(eval_smiles_list)
    ].copy()

    if len(test_labels) == 0:
        logging.warning("No test labels found matching eval_smiles_list")
        return pd.DataFrame()

    logging.info(
        f"Found {len(test_labels)} test spectra for retrieval evaluation"
    )

    # Use prebuilt PubChem candidates when configured (same pool as baselines)
    candidates_pickle = config.eval.retrieval_with_formula.get(
        "candidates_pickle", None
    )
    if candidates_pickle is not None:
        retrieval_candidates = _get_pubchem_retrieval_candidates(
            test_labels, candidates_pickle
        )
    else:
        logging.warning(
            "No candidates_pickle configured — falling back to within-dataset "
            "formula matching. Candidate pool will be much smaller than baselines."
        )
        retrieval_candidates = get_retrieval_candidates_from_labels(
            test_labels, labels_df
        )

    if len(retrieval_candidates) == 0:
        logging.warning("No retrieval candidates found")
        return pd.DataFrame()

    # Get spectrum parameters
    min_mz = config.eval.data_module.dataset_config.min_mz
    max_mz = config.eval.data_module.dataset_config.max_mz
    bin_width = config.eval.data_module.dataset_config.bin_width
    num_bins = int((max_mz - min_mz) / bin_width)
    mz_values = np.linspace(min_mz, max_mz, num_bins, endpoint=False).astype(
        np.float32
    )

    # Extract model inference parameters
    model_inference_params = {"device": device}
    if config.eval.model.architecture.get("max_nodes"):
        model_inference_params["max_nodes"] = (
            config.eval.model.architecture.max_nodes
        )
    if config.eval.model.architecture.get("threshold"):
        model_inference_params["threshold"] = (
            config.eval.model.architecture.threshold
        )

    # Get ranking metrics to compute - use ALL metrics by default for comprehensive evaluation
    ranking_metrics = config.eval.retrieval_with_formula.get(
        "ranking_metrics",
        [
            "cosine_similarity",
            "entropy_similarity",
            "entropy_distance",
            "spectral_contrast_angle",
            "mean_squared_error",
            "weighted_cosine",
            "composite_similarity",
        ],
    )
    # Support legacy single metric config
    if not ranking_metrics:
        legacy_metric = config.eval.retrieval_with_formula.get(
            "ranking_metric", "cosine_similarity"
        )
        ranking_metrics = [legacy_metric]

    k_max = config.eval.retrieval_with_formula.get("k_max", 50)
    weighted_cosine_schemes = config.eval.retrieval_with_formula.get(
        "weighted_cosine_schemes", ["nist_gc"]
    )
    composite_similarity_schemes = config.eval.retrieval_with_formula.get(
        "composite_similarity_schemes", ["nist_gc"]
    )

    logging.info(f"Computing retrieval metrics for: {ranking_metrics}")

    num_gpu_workers = config.eval.get("num_gpu_workers", 4)
    retrieval_batch_size = config.eval.retrieval_with_formula.get(
        "batch_size", 64
    )

    predicted_spectra_cache = _precompute_candidate_spectra(
        model=model,
        candidates_df=retrieval_candidates,
        smiles_column="standardized_smiles",
        model_inference_params=model_inference_params,
        num_gpu_workers=num_gpu_workers,
        batch_size=retrieval_batch_size,
        cache_path=save_dir
        / "candidate_spectra_cache"
        / f"rank_{rank}_formula.pkl",
    )

    # Load checkpoint if available
    completed_specs = []
    partial_results_df = None
    if checkpoint_manager is not None:
        completed_specs, partial_results_df = (
            checkpoint_manager.load_checkpoint("retrieval_formula")
        )

    # Load checkpoint if available
    completed_specs = []
    partial_results_df = None
    if checkpoint_manager is not None:
        completed_specs, partial_results_df = (
            checkpoint_manager.load_checkpoint("retrieval_formula")
        )

    # Process each spectrum and its candidates
    all_results = []
    unique_specs = retrieval_candidates["spec"].unique()

    # Filter out already completed specs
    if completed_specs:
        remaining_specs = [s for s in unique_specs if s not in completed_specs]
        logging.info(
            f"Resuming from checkpoint: {len(completed_specs)} already completed, "
            f"{len(remaining_specs)} remaining"
        )
    else:
        remaining_specs = list(unique_specs)

    # Split workload across ranks
    rank_specs = split_workload_by_rank(remaining_specs, rank, world_size)

    # Checkpoint frequency
    checkpoint_frequency = config.eval.get("checkpoint_frequency", 50)
    processed_count = 0
    completed_specs_list = list(completed_specs) if completed_specs else []

    for spec_id in tqdm(
        rank_specs,
        desc=f"Rank {rank} - Retrieval Evaluation",
        disable=(rank != 0),
    ):
        # Get ground truth spectrum for this query
        spec_candidates = retrieval_candidates[
            retrieval_candidates["spec"] == spec_id
        ]
        query_row = spec_candidates[~spec_candidates["is_decoy"]].iloc[0]
        query_smiles = query_row["standardized_smiles"]

        if query_smiles not in ground_truth_map:
            logging.warning(
                f"Ground truth not found for {query_smiles}, skipping"
            )
            continue

        ground_truth_spec = ground_truth_map[query_smiles]["intensities"]

        # Score all candidates for this spectrum
        candidate_scores = []

        for _, candidate_row in spec_candidates.iterrows():
            candidate_smiles = candidate_row["standardized_smiles"]
            is_decoy = candidate_row["is_decoy"]

            # Get or compute predicted spectrum using helper
            pred_spec = get_or_predict_spectrum(
                candidate_smiles,
                model,
                model_inference_params,
                predicted_spectra_cache,
            )

            # Compute all similarity metrics
            score_dict = {
                "spec": spec_id,
                "inchikey": candidate_row["inchikey"],
                "smiles": candidate_smiles,
                "formula": candidate_row["formula"],
                "is_decoy": is_decoy,
            }

            if pred_spec is None or pred_spec.sum() == 0:
                # Use helper to get default metrics for failed predictions
                defaults = get_default_metrics(
                    ranking_metrics,
                    weighted_cosine_schemes,
                    composite_similarity_schemes,
                )
                score_dict.update(defaults)
            else:
                # Use helper to compute all metrics
                metrics = compute_metrics_for_spectra(
                    pred_spec,
                    ground_truth_spec,
                    ranking_metrics,
                    weighted_cosine_schemes,
                    composite_similarity_schemes,
                    mz_values,
                )
                score_dict.update(metrics)

            candidate_scores.append(score_dict)

        all_results.extend(candidate_scores)

        # Track completion and checkpoint periodically
        completed_specs_list.append(spec_id)
        processed_count += 1

        if (
            checkpoint_manager is not None
            and processed_count % checkpoint_frequency == 0
        ):
            # Save intermediate checkpoint
            temp_df = pd.DataFrame(all_results)
            checkpoint_manager.save_checkpoint(
                completed_specs_list, temp_df, "retrieval_formula"
            )

    # Save final checkpoint for this rank
    if checkpoint_manager is not None and processed_count > 0:
        temp_df = pd.DataFrame(all_results)
        checkpoint_manager.save_checkpoint(
            completed_specs_list, temp_df, "retrieval_formula"
        )

    # Convert to DataFrame (local results for this rank)
    results_df = pd.DataFrame(all_results)

    # Gather results from all ranks if multi-GPU (file-based, no NCCL barrier
    # -- an imbalanced/slow rank can never trigger a watchdog timeout here)
    if world_size > 1:
        results_df = gather_results(
            results_df,
            rank,
            world_size,
            save_dir=save_dir,
            label="retrieval_formula",
        )
        predicted_spectra_cache = gather_pickled_data(
            predicted_spectra_cache,
            rank,
            world_size,
            save_dir=save_dir,
            label="retrieval_spectra_cache",
        )
        # Only rank 0 continues with saving
        if rank != 0:
            return pd.DataFrame()

    # Combine with partial results if resuming
    if partial_results_df is not None and len(partial_results_df) > 0:
        results_df = pd.concat(
            [partial_results_df, results_df], ignore_index=True
        )
        logging.info(
            f"Combined with {len(partial_results_df)} previous results"
        )

    if len(results_df) == 0:
        logging.warning("No retrieval results computed")
        return pd.DataFrame()

    # Get all similarity columns for ranking using helper
    similarity_columns = get_similarity_column_names(
        ranking_metrics,
        weighted_cosine_schemes,
        composite_similarity_schemes,
    )

    # Compute ranks for each similarity metric using helper
    results_df = compute_rankings_for_metrics(results_df, similarity_columns)

    # Save detailed results
    output_path_csv = save_dir / "retrieval_with_formula_results.csv"
    results_df.to_csv(output_path_csv, index=False)
    logging.info(f"Retrieval results saved to CSV: {output_path_csv}")

    # Save spectra to HDF5 for post-hoc analysis (success/failure cases)
    hdf5_path = save_dir / "retrieval_with_formula_spectra.hdf5"
    logging.info(f"Saving retrieval spectra to: {hdf5_path}")
    str_dtype = h5py.string_dtype(encoding="utf-8")

    with h5py.File(hdf5_path, "w") as hf:
        hf.create_dataset("mz_bins", data=mz_values, compression="gzip")

        for spec_id in tqdm(
            results_df["spec"].unique(),
            desc="Saving spectra (formula)",
        ):
            # Find the correct (non-decoy) row to get the query SMILES
            spec_rows = results_df[results_df["spec"] == spec_id]
            correct_rows = spec_rows[~spec_rows["is_decoy"]]
            if len(correct_rows) == 0:
                continue
            query_smiles = correct_rows.iloc[0]["smiles"]

            if query_smiles not in ground_truth_map:
                continue

            gt_spec = ground_truth_map[query_smiles]["intensities"]

            grp = hf.create_group(str(spec_id))
            grp.attrs["target_smiles"] = query_smiles
            grp.create_dataset(
                "ground_truth_intensities",
                data=gt_spec,
                compression="gzip",
            )

            n_cands = len(spec_rows)

            # Candidate metadata
            cand_smiles = spec_rows["smiles"].tolist()
            cand_iks = [
                str(v) if pd.notna(v) else "" for v in spec_rows["inchikey"]
            ]
            cand_is_decoy = spec_rows["is_decoy"].values

            grp.create_dataset(
                "candidate_smiles", data=cand_smiles, dtype=str_dtype
            )
            grp.create_dataset(
                "candidate_inchikeys", data=cand_iks, dtype=str_dtype
            )
            grp.create_dataset("candidate_is_decoy", data=cand_is_decoy)

            if "formula" in spec_rows.columns:
                formulas = [
                    str(v) if pd.notna(v) else "" for v in spec_rows["formula"]
                ]
                grp.create_dataset(
                    "candidate_formulas", data=formulas, dtype=str_dtype
                )

            # Predicted spectra matrix
            pred_matrix = np.zeros((n_cands, num_bins), dtype=np.float32)
            for i, smi in enumerate(cand_smiles):
                pred = predicted_spectra_cache.get(smi)
                if pred is not None:
                    pred_matrix[i] = pred
            grp.create_dataset(
                "candidate_predicted_intensities",
                data=pred_matrix,
                compression="gzip",
            )

            # Similarity scores and ranks
            for sim_col in similarity_columns:
                if sim_col in spec_rows.columns:
                    grp.create_dataset(
                        f"candidate_{sim_col}",
                        data=spec_rows[sim_col].values.astype(np.float32),
                    )
                rank_col = f"rank_{sim_col}"
                if rank_col in spec_rows.columns:
                    grp.create_dataset(
                        f"candidate_{rank_col}",
                        data=spec_rows[rank_col].values.astype(np.int32),
                    )

    logging.info(f"Retrieval spectra (formula) saved to: {hdf5_path}")

    # Compute and log retrieval metrics for each similarity metric
    all_retrieval_metrics = {}

    with open(save_dir / "retrieval_with_formula_summary.txt", "w") as f:
        f.write(f"Number of query spectra: {len(unique_specs)}\n")
        f.write(f"Total candidates evaluated: {len(results_df)}\n\n")

        for sim_col in similarity_columns:
            rank_col = f"rank_{sim_col}"
            if rank_col not in results_df.columns:
                continue

            # Create a temporary df with standard column names for compute_retrieval_metrics
            temp_df = results_df[["spec", "is_decoy", rank_col]].copy()
            temp_df = temp_df.rename(columns={rank_col: "rank"})

            retrieval_metrics = compute_retrieval_metrics(temp_df, k_max=k_max)

            # Store with metric prefix
            for metric_name, metric_value in retrieval_metrics.items():
                full_key = f"{sim_col}/{metric_name}"
                all_retrieval_metrics[full_key] = metric_value
                wandb.log(
                    {f"eval/retrieval_with_formula/{full_key}": metric_value}
                )

            # Write to summary file
            f.write(f"{sim_col}\n")
            for metric_name, metric_value in retrieval_metrics.items():
                f.write(f"  {metric_name}: {metric_value:.4f}\n")
            f.write("\n")

    logging.info("Retrieval with formula evaluation complete.")

    # Log summary for first metric
    if similarity_columns:
        first_metric = similarity_columns[0]
        logging.info(f"Results for {first_metric}:")
        logging.info(
            f"  Top-1 accuracy: {all_retrieval_metrics.get(f'{first_metric}/top_1_accuracy', 0):.4f}"
        )
        logging.info(
            f"  Top-10 accuracy: {all_retrieval_metrics.get(f'{first_metric}/top_10_accuracy', 0):.4f}"
        )
        logging.info(
            f"  MRR: {all_retrieval_metrics.get(f'{first_metric}/mrr', 0):.4f}"
        )

    return results_df


def run_retrieval_with_ri_evaluation(
    model: pl.LightningModule,
    ground_truth_map: Dict,
    eval_smiles_list: List[str],
    config: DictConfig,
    save_dir: Path,
    device: torch.device,
) -> pd.DataFrame:
    """Runs retrieval evaluation using pre-computed RI-based candidate sets.

    For each RI type (StdNP, SemiStdNP, StdPolar), loads the pre-computed
    candidate TSV, predicts spectra for each candidate, and ranks them by
    similarity to the experimental spectrum.

    Parameters
    ----------
    model : pl.LightningModule
        The trained model for spectrum prediction
    ground_truth_map : Dict
        Dictionary mapping SMILES to ground truth spectrum data
    eval_smiles_list : List[str]
        List of SMILES strings to evaluate
    config : DictConfig
        Hydra configuration object
    save_dir : Path
        Directory to save results
    device : torch.device
        Device to run inference on

    Returns
    -------
    pd.DataFrame
        DataFrame with retrieval results from the last RI type processed
    """
    logging.info("Starting retrieval with RI evaluation...")

    ri_config = config.eval.retrieval_with_ri
    candidates_dir = Path(ri_config.candidates_dir)
    candidates_prefix = ri_config.get(
        "candidates_prefix", "pubchem_ri_candidates"
    )
    ri_source = ri_config.get("ri_source", "from_exp")
    ri_types = ri_config.get("ri_types", ["StdNP", "SemiStdNP", "StdPolar"])
    k_max = ri_config.get("k_max", 50)
    # Use ALL metrics by default for comprehensive evaluation
    ranking_metrics = ri_config.get(
        "ranking_metrics",
        [
            "cosine_similarity",
            "entropy_similarity",
            "entropy_distance",
            "spectral_contrast_angle",
            "mean_squared_error",
            "weighted_cosine",
            "composite_similarity",
        ],
    )
    weighted_cosine_schemes = ri_config.get(
        "weighted_cosine_schemes", ["nist_gc"]
    )
    composite_similarity_schemes = ri_config.get(
        "composite_similarity_schemes", ["nist_gc"]
    )

    logging.info(f"Retrieval with RI mode: {ri_source}")

    # Build InChIKey prefix -> SMILES lookup from ground truth map
    ik_prefix_to_smiles = {}
    for smiles, gt_data in ground_truth_map.items():
        ik = gt_data.get("inchi_key", "")
        if ik and len(ik) >= 14:
            ik_prefix_to_smiles[ik[:14]] = smiles

    # Get spectrum parameters
    min_mz = config.eval.data_module.dataset_config.min_mz
    max_mz = config.eval.data_module.dataset_config.max_mz
    bin_width = config.eval.data_module.dataset_config.bin_width
    num_bins = int((max_mz - min_mz) / bin_width)
    mz_values = np.linspace(min_mz, max_mz, num_bins, endpoint=False).astype(
        np.float32
    )

    # Extract model inference parameters
    model_inference_params = {"device": device}
    if config.eval.model.architecture.get("max_nodes"):
        model_inference_params["max_nodes"] = (
            config.eval.model.architecture.max_nodes
        )
    if config.eval.model.architecture.get("threshold"):
        model_inference_params["threshold"] = (
            config.eval.model.architecture.threshold
        )

    # Get all similarity columns for ranking (shared across RI types) using helper
    similarity_columns = get_similarity_column_names(
        ranking_metrics,
        weighted_cosine_schemes,
        composite_similarity_schemes,
    )

    logging.info(f"Computing retrieval metrics for: {ranking_metrics}")

    num_gpu_workers = config.eval.get("num_gpu_workers", 4)
    retrieval_batch_size = config.eval.retrieval_with_ri.get("batch_size", 64)

    # Load all candidates across all RI types first to pre-compute their spectra
    all_candidates_dfs = []
    for ri_type in ri_types:
        candidates_path = candidates_dir / f"{candidates_prefix}_{ri_type}.tsv"
        if candidates_path.exists():
            df = pd.read_csv(candidates_path, sep="\t")
            all_candidates_dfs.append(df)

    if all_candidates_dfs:
        combined_candidates = pd.concat(all_candidates_dfs, ignore_index=True)
        predicted_spectra_cache = _precompute_candidate_spectra(
            model=model,
            candidates_df=combined_candidates,
            smiles_column="candidate_smiles",
            model_inference_params=model_inference_params,
            num_gpu_workers=num_gpu_workers,
            batch_size=retrieval_batch_size,
            cache_path=save_dir / "candidate_spectra_cache" / "ri.pkl",
        )
    else:
        predicted_spectra_cache = {}

    last_results_df = pd.DataFrame()
    ri_results = {}  # Store per-RI-type results for combined evaluation

    # Process each RI type separately
    for ri_type in ri_types:
        candidates_path = (
            candidates_dir / f"{candidates_prefix}_{ri_source}_{ri_type}.tsv"
        )
        if not candidates_path.exists():
            logging.warning(
                f"RI candidates file not found: {candidates_path}, "
                f"skipping {ri_type}"
            )
            continue

        logging.info(
            f"Loading RI candidates for {ri_type} from {candidates_path}"
        )
        candidates_df = pd.read_csv(candidates_path, sep="\t")

        # Get unique target molecules and filter to those in ground truth
        unique_targets = candidates_df["target_inchikey"].unique()
        valid_targets = []
        for target_ik in unique_targets:
            ik_prefix = target_ik[:14] if isinstance(target_ik, str) else ""
            if ik_prefix in ik_prefix_to_smiles:
                valid_targets.append(target_ik)

        logging.info(
            f"  {ri_type}: {len(valid_targets)} targets with ground truth "
            f"(out of {len(unique_targets)} total)"
        )

        if len(valid_targets) == 0:
            logging.warning(f"No valid targets for {ri_type}, skipping")
            continue

        all_results = []
        n_correct_injected = 0

        for target_ik in tqdm(valid_targets, desc=f"RI Retrieval ({ri_type})"):
            ik_prefix = target_ik[:14]
            query_smiles = ik_prefix_to_smiles[ik_prefix]
            ground_truth_spec = ground_truth_map[query_smiles]["intensities"]

            # Get candidates for this target
            target_candidates = candidates_df[
                candidates_df["target_inchikey"] == target_ik
            ]

            # Check if correct answer is already in candidate set
            correct_in_set = any(
                isinstance(cik, str) and cik[:14] == ik_prefix
                for cik in target_candidates["candidate_inchikey"]
            )

            # If correct answer is not in the candidate set, add it
            if not correct_in_set:
                n_correct_injected += 1
                correct_entry = {
                    "target_inchikey": target_ik,
                    "target_smiles": query_smiles,
                    "candidate_smiles": query_smiles,
                    "candidate_inchikey": ground_truth_map[query_smiles][
                        "inchi_key"
                    ],
                    "ri_diff": 0.0,
                    "candidate_rank": 0,
                }
                target_candidates = pd.concat(
                    [pd.DataFrame([correct_entry]), target_candidates],
                    ignore_index=True,
                )

            # Score all candidates
            for _, cand_row in target_candidates.iterrows():
                candidate_smiles = cand_row["candidate_smiles"]
                candidate_ik = cand_row["candidate_inchikey"]
                is_decoy = (
                    not isinstance(candidate_ik, str)
                    or candidate_ik[:14] != ik_prefix
                )

                # Get or compute predicted spectrum using helper
                pred_spec = get_or_predict_spectrum(
                    candidate_smiles,
                    model,
                    model_inference_params,
                    predicted_spectra_cache,
                )

                # Build score dict
                score_dict = {
                    "spec": target_ik,
                    "inchikey": candidate_ik,
                    "smiles": candidate_smiles,
                    "ri_diff": cand_row.get("ri_diff", np.nan),
                    "is_decoy": is_decoy,
                }

                if pred_spec is None or pred_spec.sum() == 0:
                    # Use helper to get default metrics for failed predictions
                    defaults = get_default_metrics(
                        ranking_metrics,
                        weighted_cosine_schemes,
                        composite_similarity_schemes,
                    )
                    score_dict.update(defaults)
                else:
                    # Use helper to compute all metrics
                    metrics = compute_metrics_for_spectra(
                        pred_spec,
                        ground_truth_spec,
                        ranking_metrics,
                        weighted_cosine_schemes,
                        composite_similarity_schemes,
                        mz_values,
                    )
                    score_dict.update(metrics)

                all_results.append(score_dict)

        # Convert to DataFrame
        results_df = pd.DataFrame(all_results)

        if len(results_df) == 0:
            logging.warning(f"No retrieval results for {ri_type}")
            continue

        logging.info(
            f"  {ri_type}: injected correct answer for "
            f"{n_correct_injected}/{len(valid_targets)} targets "
            f"(not originally in RI candidate set)"
        )

        # Compute ranks for each similarity metric using helper
        results_df = compute_rankings_for_metrics(
            results_df, similarity_columns
        )

        # Save per-RI-type results
        output_path_csv = save_dir / f"retrieval_with_ri_{ri_type}_results.csv"
        results_df.to_csv(output_path_csv, index=False)
        logging.info(
            f"Retrieval results for {ri_type} saved to: {output_path_csv}"
        )

        # Save spectra to HDF5 for post-hoc analysis (success/failure cases)
        hdf5_path = save_dir / f"retrieval_with_ri_{ri_type}_spectra.hdf5"
        logging.info(f"Saving retrieval spectra to: {hdf5_path}")
        str_dtype = h5py.string_dtype(encoding="utf-8")

        with h5py.File(hdf5_path, "w") as hf:
            hf.create_dataset("mz_bins", data=mz_values, compression="gzip")

            for target_ik in tqdm(
                results_df["spec"].unique(),
                desc=f"Saving spectra ({ri_type})",
            ):
                ik_prefix = (
                    target_ik[:14] if isinstance(target_ik, str) else ""
                )
                if ik_prefix not in ik_prefix_to_smiles:
                    continue

                query_smiles = ik_prefix_to_smiles[ik_prefix]
                gt_spec = ground_truth_map[query_smiles]["intensities"]

                grp = hf.create_group(str(target_ik))
                grp.attrs["target_smiles"] = query_smiles
                grp.create_dataset(
                    "ground_truth_intensities",
                    data=gt_spec,
                    compression="gzip",
                )

                # Get this target's candidates from results_df
                target_rows = results_df[
                    results_df["spec"] == target_ik
                ].copy()
                n_cands = len(target_rows)

                # Candidate metadata
                cand_smiles = target_rows["smiles"].tolist()
                cand_iks = [
                    str(v) if pd.notna(v) else ""
                    for v in target_rows["inchikey"]
                ]
                cand_is_decoy = target_rows["is_decoy"].values

                grp.create_dataset(
                    "candidate_smiles",
                    data=cand_smiles,
                    dtype=str_dtype,
                )
                grp.create_dataset(
                    "candidate_inchikeys",
                    data=cand_iks,
                    dtype=str_dtype,
                )
                grp.create_dataset("candidate_is_decoy", data=cand_is_decoy)

                if "ri_diff" in target_rows.columns:
                    grp.create_dataset(
                        "candidate_ri_diff",
                        data=target_rows["ri_diff"]
                        .fillna(0.0)
                        .values.astype(np.float32),
                    )

                # Predicted spectra matrix
                pred_matrix = np.zeros((n_cands, num_bins), dtype=np.float32)
                for i, smi in enumerate(cand_smiles):
                    pred = predicted_spectra_cache.get(smi)
                    if pred is not None:
                        pred_matrix[i] = pred
                grp.create_dataset(
                    "candidate_predicted_intensities",
                    data=pred_matrix,
                    compression="gzip",
                )

                # Similarity scores and ranks
                for sim_col in similarity_columns:
                    if sim_col in target_rows.columns:
                        grp.create_dataset(
                            f"candidate_{sim_col}",
                            data=target_rows[sim_col].values.astype(
                                np.float32
                            ),
                        )
                    rank_col = f"rank_{sim_col}"
                    if rank_col in target_rows.columns:
                        grp.create_dataset(
                            f"candidate_{rank_col}",
                            data=target_rows[rank_col].values.astype(np.int32),
                        )

        logging.info(f"Retrieval spectra for {ri_type} saved to: {hdf5_path}")

        # Compute and log retrieval metrics
        unique_specs = results_df["spec"].unique()
        all_retrieval_metrics = {}

        # Calculate average number of candidates per spectrum
        avg_candidates_per_spectrum = (
            len(results_df) / len(unique_specs) if len(unique_specs) > 0 else 0
        )

        with open(
            save_dir / f"retrieval_with_ri_{ri_type}_summary.txt", "w"
        ) as f:
            f.write(f"RI Type: {ri_type}\n")
            f.write(f"RI Source: {ri_source}\n")
            f.write(f"Number of query spectra: {len(unique_specs)}\n")
            f.write(f"Total candidates evaluated: {len(results_df)}\n")
            f.write(
                f"Average candidates per spectrum: {avg_candidates_per_spectrum:.2f}\n"
            )
            f.write(
                f"Correct answer injected: {n_correct_injected}/"
                f"{len(valid_targets)}\n\n"
            )

            for sim_col in similarity_columns:
                rank_col = f"rank_{sim_col}"
                if rank_col not in results_df.columns:
                    continue

                temp_df = results_df[["spec", "is_decoy", rank_col]].copy()
                temp_df = temp_df.rename(columns={rank_col: "rank"})

                retrieval_metrics = compute_retrieval_metrics(
                    temp_df, k_max=k_max
                )

                for metric_name, metric_value in retrieval_metrics.items():
                    full_key = f"{ri_source}/{ri_type}/{sim_col}/{metric_name}"
                    all_retrieval_metrics[full_key] = metric_value
                    wandb.log(
                        {f"eval/retrieval_with_ri/{full_key}": metric_value}
                    )

                f.write(f"{sim_col}\n")
                for metric_name, metric_value in retrieval_metrics.items():
                    f.write(f"  {metric_name}: {metric_value:.4f}\n")
                f.write("\n")

        # Log average candidates per spectrum to wandb
        wandb.log(
            {
                f"eval/retrieval_with_ri/{ri_source}/{ri_type}/avg_candidates_per_spectrum": avg_candidates_per_spectrum
            }
        )

        # Log summary for first metric
        if similarity_columns:
            first_metric = similarity_columns[0]
            logging.info(
                f"Results for {ri_source} / {ri_type} / {first_metric}:"
            )
            logging.info(
                f"  Average candidates per spectrum: {avg_candidates_per_spectrum:.2f}"
            )
            logging.info(
                f"  Top-1 accuracy: {all_retrieval_metrics.get(f'{ri_source}/{ri_type}/{first_metric}/top_1_accuracy', 0):.4f}"
            )
            logging.info(
                f"  Top-10 accuracy: {all_retrieval_metrics.get(f'{ri_source}/{ri_type}/{first_metric}/top_10_accuracy', 0):.4f}"
            )
            logging.info(
                f"  MRR: {all_retrieval_metrics.get(f'{ri_source}/{ri_type}/{first_metric}/mrr', 0):.4f}"
            )

        ri_results[ri_type] = results_df
        last_results_df = results_df

    # Compute combined RI type evaluations (intersections of candidate sets)
    if len(ri_results) >= 2:
        processed_types = list(ri_results.keys())
        for combo_size in range(2, len(processed_types) + 1):
            for combo in combinations(processed_types, combo_size):
                combo_name = "+".join(combo)
                logging.info(
                    f"Computing combined RI retrieval for: {combo_name}"
                )

                # Start with first RI type's results, intersect with others
                # Drop rank columns since we'll recompute after intersection
                rank_cols_to_drop = [
                    c
                    for c in ri_results[combo[0]].columns
                    if c.startswith("rank_")
                ]
                combined_df = ri_results[combo[0]].drop(
                    columns=rank_cols_to_drop
                )

                for other_type in combo[1:]:
                    other_pairs = ri_results[other_type][
                        ["spec", "smiles"]
                    ].drop_duplicates()
                    combined_df = combined_df.merge(
                        other_pairs, on=["spec", "smiles"], how="inner"
                    )

                if len(combined_df) == 0:
                    logging.warning(
                        f"No candidates in intersection for {combo_name}"
                    )
                    continue

                # Re-rank within the intersected candidate set
                for sim_col in similarity_columns:
                    if sim_col in combined_df.columns:
                        is_distance = sim_col in DISTANCE_METRICS
                        combined_df[f"rank_{sim_col}"] = (
                            combined_df.groupby("spec")[sim_col]
                            .rank(ascending=is_distance, method="min")
                            .astype(int)
                        )

                # Save results
                output_csv = (
                    save_dir / f"retrieval_with_ri_{combo_name}_results.csv"
                )
                combined_df.to_csv(output_csv, index=False)

                # Compute and log retrieval metrics
                unique_specs = combined_df["spec"].unique()
                combo_retrieval_metrics = {}
                avg_cand_per_spec = (
                    len(combined_df) / len(unique_specs)
                    if len(unique_specs) > 0
                    else 0
                )

                with open(
                    save_dir / f"retrieval_with_ri_{combo_name}_summary.txt",
                    "w",
                ) as f:
                    f.write(f"RI Types: {combo_name}\n")
                    f.write(f"RI Source: {ri_source}\n")
                    f.write(f"Number of query spectra: {len(unique_specs)}\n")
                    f.write(
                        f"Total candidates after intersection: "
                        f"{len(combined_df)}\n"
                    )
                    if len(unique_specs) > 0:
                        f.write(
                            f"Average candidates per spectrum: "
                            f"{avg_cand_per_spec:.1f}\n"
                        )
                    f.write("\n")

                    for sim_col in similarity_columns:
                        rank_col = f"rank_{sim_col}"
                        if rank_col not in combined_df.columns:
                            continue

                        temp_df = combined_df[
                            ["spec", "is_decoy", rank_col]
                        ].copy()
                        temp_df = temp_df.rename(columns={rank_col: "rank"})

                        retrieval_metrics = compute_retrieval_metrics(
                            temp_df, k_max=k_max
                        )

                        for (
                            metric_name,
                            metric_value,
                        ) in retrieval_metrics.items():
                            full_key = f"{ri_source}/{combo_name}/{sim_col}/{metric_name}"
                            combo_retrieval_metrics[full_key] = metric_value
                            wandb.log(
                                {
                                    f"eval/retrieval_with_ri/{full_key}": metric_value
                                }
                            )

                        f.write(f"{sim_col}\n")
                        for (
                            metric_name,
                            metric_value,
                        ) in retrieval_metrics.items():
                            f.write(f"  {metric_name}: {metric_value:.4f}\n")
                        f.write("\n")

                # Log average candidates per spectrum for combined set
                wandb.log(
                    {
                        f"eval/retrieval_with_ri/{ri_source}/{combo_name}/avg_candidates_per_spectrum": avg_cand_per_spec
                    }
                )

                # Log summary for first metric
                if similarity_columns:
                    first_metric = similarity_columns[0]
                    logging.info(
                        f"Results for {ri_source} / {combo_name} / {first_metric}:"
                    )
                    logging.info(
                        f"  Top-1 accuracy: {combo_retrieval_metrics.get(f'{ri_source}/{combo_name}/{first_metric}/top_1_accuracy', 0):.4f}"
                    )
                    logging.info(
                        f"  Top-10 accuracy: {combo_retrieval_metrics.get(f'{ri_source}/{combo_name}/{first_metric}/top_10_accuracy', 0):.4f}"
                    )
                    logging.info(
                        f"  MRR: {combo_retrieval_metrics.get(f'{ri_source}/{combo_name}/{first_metric}/mrr', 0):.4f}"
                    )

    logging.info("Retrieval with RI evaluation complete.")
    return last_results_df


def run_predict_only(
    model: Any,
    config: DictConfig,
    save_dir: Path,
    device: torch.device,
) -> None:
    """Generate predictions for a test split and save as HDF5 for later
    evaluation.

    Reads SMILES and InChIKey from labels + splits (no spectra.hdf5 needed).
    Output HDF5 is compatible with eval_from_predictions.py.
    """
    labels_df = pd.read_csv(config.eval.data_module.labels_path, sep="\t")
    splits_df = pd.read_csv(config.eval.data_module.splits_path, sep="\t")

    # Normalise column names across possible naming conventions
    for old, new in [
        ("spec", "mol_id"),
        ("smiles", "standardized_smiles"),
        ("inchikey", "inchi_key"),
    ]:
        if old in labels_df.columns and new not in labels_df.columns:
            labels_df = labels_df.rename(columns={old: new})
    if "spec" in splits_df.columns and "mol_id" not in splits_df.columns:
        splits_df = splits_df.rename(columns={"spec": "mol_id"})

    test_mol_ids = set(
        splits_df[splits_df["split"] == "test"]["mol_id"].astype(str).tolist()
    )
    test_df = labels_df[labels_df["mol_id"].astype(str).isin(test_mol_ids)]

    min_mz = config.eval.data_module.dataset_config.get("min_mz", 0)
    max_mz = config.eval.data_module.dataset_config.get("max_mz", 750)
    bin_width = config.eval.data_module.dataset_config.get("bin_width", 1.0)
    num_bins = int((max_mz - min_mz) / bin_width)
    mz_bins = np.linspace(min_mz, max_mz, num_bins, endpoint=False).astype(
        np.float32
    )

    inference_params = {
        "device": device,
        "min_mz": min_mz,
        "max_mz": max_mz,
        "bin_width": bin_width,
    }

    predictions_data = []
    for _, row in tqdm(
        test_df.iterrows(), total=len(test_df), desc="Predicting"
    ):
        smiles = row["standardized_smiles"]
        inchi_key = row["inchi_key"]
        mol_id = str(row["mol_id"])
        with torch.no_grad():
            result = model.predict_from_smiles(
                smiles=smiles, **inference_params
            )
        predictions_data.append(
            {
                "smiles": smiles,
                "mol_id": mol_id,
                "inchi_key": inchi_key,
                "predicted_mz_bins": mz_bins,
                "predicted_intensities": result["intensities"].astype(
                    np.float32
                ),
                "ground_truth_mz_bins": mz_bins,
                "ground_truth_intensities": np.zeros(
                    num_bins, dtype=np.float32
                ),
                "metrics": {},
            }
        )

    hdf5_path = save_dir / "predictions.hdf5"
    save_predictions_as_hdf5(predictions_data, hdf5_path)
    logging.info(f"Predictions saved to {hdf5_path}")


def evaluate(config: DictConfig) -> None:
    """Evaluate a trained model using a checkpoint with multi-GPU and
    checkpoint support."""

    # Determine rank and world_size for multi-GPU evaluation
    # Only enable multi-GPU if a distributed launcher (torchrun) is active
    rank = 0
    world_size = 1
    if "LOCAL_RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    elif config.system.get("devices") and len(config.system.devices) > 1:
        logging.info(
            "Multiple devices configured but no distributed launcher detected. "
            "Running on a single GPU. Use `torchrun --nproc_per_node=N` for multi-GPU."
        )

    # Set device for this rank (must happen before any CUDA operations)
    if config.system.accelerator == "gpu" and torch.cuda.is_available():
        device_id = config.system.devices[rank % len(config.system.devices)]
        torch.cuda.set_device(device_id)
        device = torch.device(f"cuda:{device_id}")
    else:
        device = torch.device("cpu")

    # Initialize distributed process group for multi-GPU
    if world_size > 1:
        import torch.distributed as dist

        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=20))

    # Get save_dir from Hydra, then broadcast rank 0's dir to all ranks
    # (each torchrun process may get a different Hydra timestamp)
    save_dir = Path(
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    )
    if world_size > 1:
        import torch.distributed as dist

        save_dir_list = [str(save_dir)]
        dist.broadcast_object_list(save_dir_list, src=0)
        save_dir = Path(save_dir_list[0])
    save_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging (only main process writes to file)
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(save_dir / "evaluation.log"),
                logging.StreamHandler(),
            ],
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format=f"%(asctime)s - [Rank {rank}] %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler()],
        )

    logging.info(f"Rank {rank}/{world_size} using device: {device}")
    logging.info(f"Starting evaluation. Results will be saved to: {save_dir}")

    # Initialize Weights & Biases if configured (only on main process)
    if rank == 0:
        wandb_logger = initiate_wandb(config, save_dir)
        wandb.config.update({"hydra_cwd": os.getcwd()})
        wandb.config.update(OmegaConf.to_container(config, resolve=True))
        logging.info("Weights & Biases initialized.")

    # Load model; disable torch.compile in multi-GPU mode (CUDA graphs conflict)
    model = load_model(config, device, compile_model=(world_size == 1))

    # Initialize checkpoint manager
    checkpoint_manager = None
    if config.eval.get("enable_checkpointing", True):
        checkpoint_dir = save_dir / "checkpoints"
        checkpoint_manager = EvaluationCheckpoint(checkpoint_dir)
        logging.info(
            f"Checkpointing enabled. Checkpoint dir: {checkpoint_dir}"
        )

    # Predict-only mode: generate predictions HDF5 without ground truth spectra
    if config.eval.get("predict_only", False):
        if rank == 0:
            run_predict_only(model, config, save_dir, device)
        if rank == 0:
            wandb.finish()
        if world_size > 1:
            import torch.distributed as dist

            dist.destroy_process_group()
        return

    # Prepare data module (only if full dataset_config is provided)
    if config.eval.data_module.get("dataset_type"):
        data_module = MassSpecDataModule(
            labels_path=config.eval.data_module.labels_path,
            splits_path=config.eval.data_module.splits_path,
            dataset_type=config.eval.data_module.dataset_type,
            dataset_config=config.eval.data_module.dataset_config,
            batch_size=config.eval.data_module.batch_size,
            num_workers=config.eval.data_module.num_workers,
            training_data_fraction=config.eval.data_module.training_data_fraction,
        )
        data_module.setup("test")

    # Get test set mol_ids
    splits_df = pd.read_csv(config.eval.data_module.splits_path, sep="\t")
    test_mol_ids = splits_df[splits_df["split"] == "test"]["mol_id"].tolist()

    # Randomly sample a fraction of the test set (use same seed on all ranks for consistency)
    if config.eval.fraction_of_spectra_to_compute < 1:
        np.random.seed(config.system.seed)  # Same seed on all ranks
        test_mol_ids = list(
            np.random.choice(
                test_mol_ids,
                size=int(
                    len(test_mol_ids)
                    * config.eval.fraction_of_spectra_to_compute
                ),
                replace=False,
            )
        )

    # Retrieve all ground truth spectra that are relevant to the test set
    ground_truth_map, eval_smiles_list = retrieve_ground_truth(
        labels_path=config.eval.data_module.labels_path,
        spectra_path=config.eval.data_module.dataset_config.spectra_path,
        mol_ids=test_mol_ids,
        min_mz=config.eval.data_module.dataset_config.min_mz,
        max_mz=config.eval.data_module.dataset_config.max_mz,
        bin_width=config.eval.data_module.dataset_config.bin_width,
    )

    # Run evaluations based on config flags
    if config.eval.similarity.enable:
        run_similarity_evaluation(
            model,
            ground_truth_map,
            eval_smiles_list,
            config,
            save_dir,
            device,
            checkpoint_manager=checkpoint_manager,
            rank=rank,
            world_size=world_size,
        )

    if config.eval.retrieval_with_formula.enable:
        run_retrieval_with_formula_evaluation(
            model,
            ground_truth_map,
            eval_smiles_list,
            config,
            save_dir,
            device,
            checkpoint_manager=checkpoint_manager,
            rank=rank,
            world_size=world_size,
        )

    if config.eval.retrieval_with_ri.enable:
        run_retrieval_with_ri_evaluation(
            model, ground_truth_map, eval_smiles_list, config, save_dir, device
        )

    if rank == 0:
        wandb.finish()

    # Clean up distributed process group
    if world_size > 1:
        import torch.distributed as dist

        dist.destroy_process_group()

    logging.info("Evaluation complete.")


def initiate_wandb(config: DictConfig, save_dir: Path):
    """Initiates Weights & Biases run and returns a WandbLogger instance."""

    wandb.init(
        project=config.wandb.project,
        entity=config.wandb.entity,
        mode=config.wandb.mode,
        dir=save_dir,
        config=OmegaConf.to_container(config, resolve=True),
    )
    wandb.run.summary["output_directory"] = str(save_dir.resolve())
    logging.info(f"Wandb run initialized: {wandb.run.url}")
    return WandbLogger(log_model=False)


@hydra.main(
    config_path="../../examples/configs",
    config_name="config",
    version_base=None,
)
def main(config: DictConfig) -> None:
    """Main entry point for evaluation."""
    pl.seed_everything(config.system.seed)

    evaluate(config)


if __name__ == "__main__":
    main()
