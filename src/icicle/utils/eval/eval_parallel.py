"""Parallel evaluation utilities for multi-GPU support.

This module provides utilities for parallelizing evaluation across multiple
GPUs and adding checkpoint/resume support for preemptable nodes.
"""

import json
import logging
import os
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist


class EvaluationCheckpoint:
    """Manages checkpoint saving and loading for evaluation."""

    def __init__(self, checkpoint_dir: Path):
        """Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to save checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(
        self,
        completed_items: List[str],
        partial_results: pd.DataFrame,
        stage: str,
    ):
        """Save checkpoint of completed work.

        Args:
            completed_items: List of completed item IDs (SMILES or spec IDs)
            partial_results: DataFrame with partial results
            stage: Name of evaluation stage (similarity, retrieval_formula, retrieval_ri)
        """
        checkpoint_path = self.checkpoint_dir / f"{stage}_checkpoint.json"
        results_path = self.checkpoint_dir / f"{stage}_partial_results.parquet"

        # Convert items to Python native types for JSON serialization
        # (handles numpy int64, int32, etc. from pandas DataFrames)
        completed_items_native = [
            int(item) if isinstance(item, np.integer) else str(item)
            for item in completed_items
        ]

        # Save completed items list
        checkpoint_data = {
            "completed_items": completed_items_native,
            "stage": stage,
        }

        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint_data, f)

        # Save partial results
        if len(partial_results) > 0:
            partial_results.to_parquet(results_path, index=False)

        logging.info(
            f"Checkpoint saved: {len(completed_items)} items completed for {stage}"
        )

    def load_checkpoint(
        self, stage: str
    ) -> Tuple[List[str], Optional[pd.DataFrame]]:
        """Load checkpoint for resuming evaluation.

        Args:
            stage: Name of evaluation stage

        Returns:
            Tuple of (completed_items, partial_results)
        """
        checkpoint_path = self.checkpoint_dir / f"{stage}_checkpoint.json"
        results_path = self.checkpoint_dir / f"{stage}_partial_results.parquet"

        if not checkpoint_path.exists():
            logging.info(f"No checkpoint found for {stage}, starting fresh")
            return [], None

        # Load completed items
        with open(checkpoint_path, "r") as f:
            checkpoint_data = json.load(f)

        completed_items = checkpoint_data.get("completed_items", [])

        # Load partial results if available
        partial_results = None
        if results_path.exists():
            partial_results = pd.read_parquet(results_path)

        logging.info(
            f"Checkpoint loaded: {len(completed_items)} items already completed for {stage}"
        )

        return completed_items, partial_results

    def clear_checkpoint(self, stage: str):
        """Clear checkpoint files for a stage.

        Args:
            stage: Name of evaluation stage
        """
        checkpoint_path = self.checkpoint_dir / f"{stage}_checkpoint.json"
        results_path = self.checkpoint_dir / f"{stage}_partial_results.parquet"

        if checkpoint_path.exists():
            checkpoint_path.unlink()
        if results_path.exists():
            results_path.unlink()

        logging.info(f"Checkpoint cleared for {stage}")


def setup_distributed() -> Tuple[int, int, int]:
    """Setup distributed environment and return rank, world_size, local_rank.

    Returns:
        Tuple of (rank, world_size, local_rank)
    """
    if not dist.is_available():
        return 0, 1, 0

    if not dist.is_initialized():
        # Try to initialize if environment variables are set
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            dist.init_process_group(backend="nccl")
        else:
            return 0, 1, 0

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    return rank, world_size, local_rank


def split_workload_by_rank(
    items: List[str], rank: int, world_size: int
) -> List[str]:
    """Split workload across ranks for parallel processing.

    Args:
        items: List of items to process
        rank: Current process rank
        world_size: Total number of processes

    Returns:
        List of items assigned to this rank
    """
    if world_size == 1:
        return items

    # Split items evenly across ranks
    items_per_rank = len(items) // world_size
    remainder = len(items) % world_size

    start_idx = rank * items_per_rank + min(rank, remainder)
    end_idx = start_idx + items_per_rank + (1 if rank < remainder else 0)

    assigned_items = items[start_idx:end_idx]

    logging.info(
        f"Rank {rank}/{world_size}: Processing {len(assigned_items)} items "
        f"(indices {start_idx} to {end_idx})"
    )

    return assigned_items


def gather_results(
    local_results: pd.DataFrame,
    rank: int,
    world_size: int,
    save_dir: Optional[Path] = None,
    label: str = "results",
    poll_interval_s: float = 5.0,
    max_wait_s: float = 6 * 3600,
) -> Optional[pd.DataFrame]:
    """Gather results from all ranks to rank 0 without any NCCL
    synchronization.

    Each rank writes its local results to its own file under ``save_dir``.
    Every rank then polls (plain filesystem, no ``dist.barrier()``) until all
    ranks' files exist. Ranks finish independently and at their own pace --
    a slow/imbalanced rank can never trigger a NCCL watchdog timeout here,
    since no collective op is used.

    Args:
        local_results: DataFrame with results from this rank
        rank: Current process rank
        world_size: Total number of processes
        save_dir: Directory to write per-rank result files to (persists on
            disk, unlike a temp dir, so a killed run can be inspected/resumed)
        label: Label for the per-rank files (must be unique per call site)
        poll_interval_s: Seconds between filesystem polls while waiting
        max_wait_s: Give up waiting for sibling ranks after this long

    Returns:
        Combined DataFrame on rank 0, None on other ranks
    """
    if world_size == 1:
        return local_results

    assert save_dir is not None, "save_dir is required when world_size > 1"
    gather_dir = Path(save_dir) / "rank_gather"
    gather_dir.mkdir(parents=True, exist_ok=True)

    local_path = gather_dir / f"rank_{rank}_{label}.parquet"
    local_results.to_parquet(local_path, index=False)
    done_path = gather_dir / f"rank_{rank}_{label}.done"
    done_path.touch()

    if rank != 0:
        return None

    # Rank 0 waits for every rank's .done sentinel, then merges.
    import time

    rank_paths = {
        r: gather_dir / f"rank_{r}_{label}.parquet" for r in range(world_size)
    }
    done_paths = {
        r: gather_dir / f"rank_{r}_{label}.done" for r in range(world_size)
    }
    waited_s = 0.0
    while waited_s < max_wait_s:
        if all(p.exists() for p in done_paths.values()):
            break
        time.sleep(poll_interval_s)
        waited_s += poll_interval_s
    else:
        missing = [r for r, p in done_paths.items() if not p.exists()]
        logging.warning(
            f"Gave up waiting for ranks {missing} after {max_wait_s}s; "
            "merging whatever finished."
        )

    all_results = [
        pd.read_parquet(p) for p in rank_paths.values() if p.exists()
    ]
    combined_results = pd.concat(all_results, ignore_index=True)
    logging.info(
        f"Gathered {len(combined_results)} total results from "
        f"{len(all_results)}/{world_size} ranks"
    )

    for p in list(rank_paths.values()) + list(done_paths.values()):
        if p.exists():
            p.unlink()

    return combined_results


def gather_pickled_data(
    local_data,
    rank: int,
    world_size: int,
    save_dir: Optional[Path] = None,
    label: str = "data",
    poll_interval_s: float = 5.0,
    max_wait_s: float = 6 * 3600,
):
    """Gather arbitrary picklable data from all ranks to rank 0.

    Each rank writes its data to its own pickle file under ``save_dir`` and
    a ``.done`` sentinel; rank 0 polls the filesystem (no ``dist.barrier()``)
    until all ranks are done, then reads and merges. No NCCL collective is
    used, so an imbalanced/slow rank can never trigger a watchdog timeout.

    Args:
        local_data: Data to gather (list or dict)
        rank: Current process rank
        world_size: Total number of processes
        save_dir: Directory to write per-rank files to
        label: Label for per-rank files (must be unique per call site)
        poll_interval_s: Seconds between filesystem polls while waiting
        max_wait_s: Give up waiting for sibling ranks after this long

    Returns:
        Merged data on rank 0, None on other ranks
    """
    if world_size == 1:
        return local_data

    assert save_dir is not None, "save_dir is required when world_size > 1"
    gather_dir = Path(save_dir) / "rank_gather"
    gather_dir.mkdir(parents=True, exist_ok=True)

    local_path = gather_dir / f"rank_{rank}_{label}.pkl"
    with open(local_path, "wb") as f:
        pickle.dump(local_data, f)
    done_path = gather_dir / f"rank_{rank}_{label}.done"
    done_path.touch()

    if rank != 0:
        return None

    import time

    rank_paths = {
        r: gather_dir / f"rank_{r}_{label}.pkl" for r in range(world_size)
    }
    done_paths = {
        r: gather_dir / f"rank_{r}_{label}.done" for r in range(world_size)
    }
    waited_s = 0.0
    while waited_s < max_wait_s:
        if all(p.exists() for p in done_paths.values()):
            break
        time.sleep(poll_interval_s)
        waited_s += poll_interval_s
    else:
        missing = [r for r, p in done_paths.items() if not p.exists()]
        logging.warning(
            f"Gave up waiting for ranks {missing} after {max_wait_s}s; "
            "merging whatever finished."
        )

    all_data = []
    for p in rank_paths.values():
        if p.exists():
            with open(p, "rb") as f:
                all_data.append(pickle.load(f))

    if not all_data:
        merged = None
    elif isinstance(all_data[0], list):
        merged = []
        for d in all_data:
            merged.extend(d)
    elif isinstance(all_data[0], dict):
        merged = {}
        for d in all_data:
            merged.update(d)
    else:
        merged = all_data

    for p in list(rank_paths.values()) + list(done_paths.values()):
        if p.exists():
            p.unlink()

    return merged


def is_main_process() -> bool:
    """Check if this is the main process (rank 0).

    Returns:
        True if main process, False otherwise
    """
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def get_device_for_rank(rank: int, devices: List[int]) -> torch.device:
    """Get the appropriate device for a given rank.

    Args:
        rank: Process rank
        devices: List of device IDs

    Returns:
        torch.device for this rank
    """
    if len(devices) == 0:
        return torch.device("cpu")

    # Assign device in round-robin fashion
    device_idx = devices[rank % len(devices)]
    return torch.device(f"cuda:{device_idx}")
