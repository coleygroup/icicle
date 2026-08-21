"""Generate DAG predictions for train/val/test splits using a trained
fragmentation model.

Simple script that works with hydra run directories and saves predictions
directly there.

GPU Parallelization:
This script supports parallel inference using multiple GPU workers to speed up
prediction generation. Key features:

* --num_gpu_workers: Number of parallel GPU workers (default: 0, sequential)
* --gpu: Enable GPU inference (required for GPU workers)
* --debug: Limit to 10 batches for testing

Usage Examples:
# Sequential inference on CPU:
python get_dag_predictions.py --run_dir /path/to/run --checkpoint best.ckpt

# Sequential inference on single GPU:
python get_dag_predictions.py --run_dir /path/to/run --checkpoint best.ckpt --gpu

# Parallel inference with 4 GPU workers:
python get_dag_predictions.py --run_dir /path/to/run --checkpoint best.ckpt \\
    --gpu --num_gpu_workers 4

# Debug mode (process only 10 batches):
python get_dag_predictions.py --run_dir /path/to/run --checkpoint best.ckpt \\
    --gpu --num_gpu_workers 2 --debug

GPU Worker Configuration:
When using multiple GPU workers:
- Each worker loads its own copy of the model
- Workers are assigned to GPUs in round-robin fashion (worker_id % num_gpus)
- If num_gpu_workers > num_gpus, multiple workers share the same GPU
  (may cause memory issues with large models)
- Best practice: Set num_gpu_workers = num_gpus for optimal performance

Performance Notes:
- GPU workers significantly speed up inference for large datasets
- Each worker processes batches independently in parallel
- Memory usage scales with number of workers (each loads full model)
- For debug/testing, use --debug flag to process only 10 batches
"""

import argparse
import json
import logging
import multiprocessing as mp
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
from omegaconf import OmegaConf
from rdkit import RDLogger
from tqdm import tqdm

from icicle.data.data_module import MassSpecDataModule
from icicle.models.fragmentation_model import FragmentGenerator
from icicle.utils import HDF5Dataset

RDLogger.DisableLog("rdApp.*")


def _process_batches_sequential(
    dataloader: Any,
    model: FragmentGenerator,
    max_nodes: int,
    threshold: float,
    logger: logging.Logger,
    debug: bool = False,
) -> Dict[str, Any]:
    """Process batches sequentially without parallelization."""
    predictions = {}

    with torch.no_grad():
        for batch_idx, batch in enumerate(
            tqdm(dataloader, desc="Processing batches")
        ):
            if debug and batch_idx >= 10:
                break

            try:
                batch_preds = model.predict_mol(
                    smi=batch["smiles"],
                    max_nodes=max_nodes,
                    threshold=threshold,
                )

                for name, pred in zip(batch["names"], batch_preds):
                    predictions[str(name)] = pred

            except Exception as e:
                logger.warning(f"Error processing batch {batch_idx}: {e}")
                continue

    return predictions


def _process_batch_worker(
    batch: Any,
    checkpoint_path: str,
    max_nodes: int,
    threshold: float,
    worker_id: int,
    num_gpus: int,
) -> List[tuple]:
    """Worker function to process a single batch on a specific GPU."""
    # Limit CPU threads per worker to avoid oversubscription when multiple
    # workers are running in parallel
    torch.set_num_threads(1)

    # Load model in worker process
    model = FragmentGenerator.load_from_checkpoint(str(checkpoint_path))
    model.eval()

    # Assign GPU to worker
    if num_gpus > 0:
        gpu_id = worker_id % num_gpus
        device = f"cuda:{gpu_id}"
        model = model.to(device)
    else:
        device = "cpu"

    results = []

    with torch.no_grad():
        try:
            batch_preds = model.predict_mol(
                smi=batch["smiles"],
                max_nodes=max_nodes,
                threshold=threshold,
            )

            for name, pred in zip(batch["names"], batch_preds):
                results.append((str(name), pred))

        except Exception as e:
            # Use logging module for proper error handling in multiprocessing context
            import logging

            logging.error(f"Error in worker {worker_id}: {e}")

    return results


def _process_batches_parallel(
    dataloader: Any,
    checkpoint_path: str,
    max_nodes: int,
    threshold: float,
    num_workers: int,
    num_gpus: int,
    logger: logging.Logger,
    debug: bool = False,
) -> Dict[str, Any]:
    """Process batches in parallel using multiple GPU workers.

    Note: Each worker loads its own copy of the model. When num_workers > num_gpus,
    multiple workers will share the same GPU which may cause memory issues.
    """
    from functools import partial

    # Collect batches from dataloader
    all_batches = []
    for batch in tqdm(dataloader, desc="Loading batches"):
        all_batches.append(batch)
        if debug and len(all_batches) >= 10:
            break

    logger.info(f"Loaded {len(all_batches)} batches")

    # Create worker function with fixed parameters
    worker_func = partial(
        _gpu_worker_wrapper,
        checkpoint_path=checkpoint_path,
        max_nodes=max_nodes,
        threshold=threshold,
        num_gpus=num_gpus,
    )

    predictions = {}

    # Use multiprocessing pool for parallel processing
    logger.info(f"Starting pool with {num_workers} workers")

    with mp.Pool(processes=num_workers) as pool:
        # Process batches in parallel
        results_list = list(
            tqdm(
                pool.imap(worker_func, enumerate(all_batches)),
                total=len(all_batches),
                desc="Processing batches with GPU workers",
            )
        )

        # Collect results
        for batch_results in results_list:
            for name, pred in batch_results:
                predictions[name] = pred

    return predictions


def _gpu_worker_wrapper(
    batch_with_index: tuple,
    checkpoint_path: str,
    max_nodes: int,
    threshold: float,
    num_gpus: int,
) -> List[tuple]:
    """Wrapper to extract batch and index for worker processing."""
    batch_idx, batch = batch_with_index
    # Use batch_idx as worker_id for consistent GPU assignment
    return _process_batch_worker(
        batch, checkpoint_path, max_nodes, threshold, batch_idx, num_gpus
    )


def generate_dag_predictions(
    run_dir: str,
    checkpoint_path: str,
    max_nodes: int = 100,
    threshold: float = 0.01,
    splits: list = ["train", "val", "test"],
    batch_size: int = 256,
    num_workers: int = 24,
    num_cpu_workers: int = 0,
    num_gpu_workers: int = 0,
    gpu: bool = False,
    debug: bool = False,
):
    """Generate DAG predictions and save them in the hydra run directory.

    Args:
        run_dir: Hydra run directory containing config and checkpoint
        checkpoint_path: Path to model checkpoint (can be relative to run_dir)
        max_nodes: Maximum number of nodes in generated DAGs
        threshold: Probability threshold for fragment selection
        splits: Which splits to generate predictions for
        batch_size: Batch size for prediction generation
        num_workers: Number of dataloader workers
        num_cpu_workers: Number of CPU workers for parallel processing
        num_gpu_workers: Number of GPU workers for parallel processing
        gpu: Whether to use GPU for inference
        debug: Debug mode (limits to 10 samples)
    """
    run_dir = Path(run_dir)

    # Handle relative checkpoint path
    if not Path(checkpoint_path).is_absolute():
        checkpoint_path = run_dir / checkpoint_path

    # Load config from hydra run
    config_path = run_dir / ".hydra" / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found at {config_path}")

    cfg = OmegaConf.load(config_path)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(run_dir / "prediction_generation.log"),
        ],
    )
    logger = logging.getLogger(__name__)

    logger.info(f"Loading model from {checkpoint_path}")
    logger.info(
        f"Using parameters: max_nodes={max_nodes}, threshold={threshold}"
    )
    logger.info(
        f"GPU parallelization: gpu={gpu}, num_gpu_workers={num_gpu_workers}, num_cpu_workers={num_cpu_workers}"
    )

    # Detect available GPUs
    num_available_gpus = torch.cuda.device_count() if gpu else 0
    logger.info(f"Available GPUs: {num_available_gpus}")

    # Validate GPU worker configuration
    if num_gpu_workers > 0 and not gpu:
        logger.warning(
            "num_gpu_workers > 0 but --gpu flag not set. GPU workers will use CPU instead."
        )
    if num_gpu_workers > num_available_gpus > 0:
        logger.warning(
            f"num_gpu_workers ({num_gpu_workers}) exceeds available GPUs ({num_available_gpus}). "
            f"Multiple workers will share GPUs, which may cause memory issues."
        )

    # Create data module using the original config
    data_module = MassSpecDataModule(
        labels_path=cfg.model.data_module.labels_path,
        splits_path=cfg.model.data_module.splits_path,
        dataset_type=cfg.model.data_module.dataset_type,
        dataset_config=cfg.model.data_module.dataset_config,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    data_module.setup()

    loaders = {
        "train": data_module.train_dataloader(),
        "val": data_module.val_dataloader(),
        "test": data_module.test_dataloader(),
    }

    # Generate predictions for requested splits
    prediction_files = {}
    prediction_counts = {}

    for split_name in splits:
        if split_name not in loaders:
            logger.warning(f"Split '{split_name}' not available, skipping")
            continue

        logger.info(f"Generating predictions for {split_name} split...")

        dataloader = loaders[split_name]

        # Process batches with GPU workers if enabled
        if num_gpu_workers > 0:
            logger.info(
                f"Using {num_gpu_workers} GPU workers for parallel processing"
            )
            predictions = _process_batches_parallel(
                dataloader,
                checkpoint_path,
                max_nodes,
                threshold,
                num_gpu_workers,
                num_available_gpus,
                logger,
                debug,
            )
        else:
            logger.info("Processing batches sequentially")
            # Load model from checkpoint
            model = FragmentGenerator.load_from_checkpoint(
                str(checkpoint_path)
            )
            model.eval()

            # Move model to GPU if available
            if gpu and num_available_gpus > 0:
                model = model.cuda()

            predictions = _process_batches_sequential(
                dataloader, model, max_nodes, threshold, logger, debug
            )

        pred_file = run_dir / f"{split_name}_dag_predictions.hdf5"

        if pred_file.exists():
            pred_file.unlink()

        with HDF5Dataset(str(pred_file), mode="w") as h5:
            for name, pred in predictions.items():
                h5.write_str(f"pred_{name}", json.dumps(pred))

        prediction_files[split_name] = pred_file
        prediction_counts[split_name] = len(predictions)
        logger.info(
            f"Saved {len(predictions)} {split_name} predictions to {pred_file}"
        )

        time.sleep(0.1)

    metadata = {
        "prediction_params": {
            "max_nodes": max_nodes,
            "threshold": threshold,
        },
        "model_checkpoint": str(checkpoint_path),
        "prediction_files": {k: str(v) for k, v in prediction_files.items()},
        "total_predictions": sum(prediction_counts.values()),
        "prediction_counts_by_split": prediction_counts,
    }

    metadata_file = run_dir / "dag_prediction_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("DAG prediction generation complete!")
    logger.info(f"Generated files: {list(prediction_files.keys())}")
    logger.info(f"Total predictions: {sum(prediction_counts.values())}")
    logger.info(f"Metadata saved to: {metadata_file}")

    return prediction_files


def _load_prediction_names(file_path: Path, max_retries: int = 3) -> list:
    """Load prediction names from HDF5 file with retry logic."""
    for attempt in range(max_retries):
        try:
            time.sleep(0.1 * attempt)

            with HDF5Dataset(str(file_path), mode="r") as h5:
                names = h5.get_all_names()
                return names

        except (ValueError, OSError, IOError) as e:
            if attempt < max_retries - 1:
                logging.warning(
                    f"Attempt {attempt + 1} failed to read {file_path}: {e}. Retrying..."
                )
                continue
            else:
                logging.error(
                    f"Failed to read {file_path} after {max_retries} attempts: {e}"
                )
                return []
        except Exception as e:
            logging.error(f"Unexpected error reading {file_path}: {e}")
            return []

    return []


def verify_predictions(run_dir: str) -> Dict[str, int]:
    """Verify prediction files and return counts."""
    run_dir = Path(run_dir)

    prediction_files = list(run_dir.glob("*_dag_predictions.hdf5"))

    counts = {}
    for pred_file in prediction_files:
        split_name = pred_file.stem.replace("_dag_predictions", "")
        names = _load_prediction_names(pred_file)
        counts[split_name] = len(names)
        print(f"{split_name}: {len(names)} predictions")

    return counts


def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(
        description="Generate DAG predictions for hydra run"
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Hydra run directory containing config and checkpoint",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint (absolute or relative to run_dir)",
    )
    parser.add_argument(
        "--max_nodes",
        type=int,
        default=100,
        help="Maximum nodes in DAG (default: 100)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.01,
        help="Fragment probability threshold (default: 0.01)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Which splits to generate predictions for",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for prediction generation",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=24,
        help="Number of dataloader workers (default: 24)",
    )
    parser.add_argument(
        "--num_cpu_workers",
        type=int,
        default=0,
        help="Number of CPU workers for parallel processing (default: 0)",
    )
    parser.add_argument(
        "--num_gpu_workers",
        type=int,
        default=0,
        help="Number of GPU workers for parallel processing (default: 0)",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        default=False,
        help="Use GPU for inference (default: False)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Debug mode - process only 10 batches (default: False)",
    )
    parser.add_argument(
        "--verify_only",
        action="store_true",
        help="Only verify existing prediction files without generating new ones",
    )

    args = parser.parse_args()

    if args.verify_only:
        print("Verifying existing prediction files...")
        counts = verify_predictions(args.run_dir)
        print(f"Total predictions across all splits: {sum(counts.values())}")
    else:
        generate_dag_predictions(
            run_dir=args.run_dir,
            checkpoint_path=args.checkpoint,
            max_nodes=args.max_nodes,
            threshold=args.threshold,
            splits=args.splits,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            num_cpu_workers=args.num_cpu_workers,
            num_gpu_workers=args.num_gpu_workers,
            gpu=args.gpu,
            debug=args.debug,
        )


if __name__ == "__main__":
    main()
