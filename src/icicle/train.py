"""Training script for EIMS Predictor."""

import logging
import os
import signal
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"
import hydra
import pytorch_lightning as pl
import torch
import wandb
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from rdkit import RDLogger

from icicle.data.data_module import MassSpecDataModule

RDLogger.DisableLog("rdApp.*")

os.environ["TORCH_WARN_ONCE"] = "1"


load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_callbacks(config: DictConfig) -> List[pl.Callback]:
    """Create training callbacks.

    Args:
        config: Configuration dictionary

    Returns:
        List of callbacks
    """
    callbacks: List[pl.Callback] = [
        ModelCheckpoint(
            dirpath="checkpoints",
            filename="best-model-{val_loss:.4f}-{epoch:02d}",
            monitor="val_loss",
            mode="min",
            save_last=True,
            verbose=True,
        ),
        # Step-based checkpointing for preemptable job resume
        ModelCheckpoint(
            dirpath="checkpoints",
            filename="step-{step:06d}",
            every_n_train_steps=500,
            save_top_k=1,
            save_last=False,
        ),
    ]

    if not config.system.debug:
        callbacks.extend(
            [
                EarlyStopping(
                    monitor="val_loss",
                    patience=10,
                    mode="min",
                    verbose=True,
                ),
                # Add learning rate monitoring
                pl.callbacks.LearningRateMonitor(logging_interval="step"),
                # Add model summary
                pl.callbacks.ModelSummary(max_depth=2),
            ]
        )

    return callbacks


def create_trainer(
    config: DictConfig,
    callbacks: List[pl.Callback],
    logger: Optional[pl.loggers.Logger] = None,
) -> pl.Trainer:
    """Create PyTorch Lightning trainer.

    Args:
        config: Configuration dictionary
        callbacks: List of callbacks
        logger: Optional logger

    Returns:
        PyTorch Lightning trainer
    """
    return pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        accelerator=config.system.accelerator,
        devices=config.system.devices,
        **config.trainer,
    )


def create_logger(
    config: DictConfig, resume_run_id: Optional[str] = None
) -> pl.loggers.Logger:
    """Create logger.

    Args:
        config: Configuration dictionary
        resume_run_id: Optional WandB run ID to resume logging to.
    """
    wandb_kwargs = dict(
        project=config.wandb.project,
        entity=config.wandb.entity,
        mode=config.wandb.mode,
    )
    if resume_run_id:
        wandb_kwargs["id"] = resume_run_id
        wandb_kwargs["resume"] = "allow"
        logger.info(f"Resuming WandB run: {resume_run_id}")

    wandb_kwargs["config"] = {
        "hydra_cwd": os.getcwd(),
        "config": OmegaConf.to_container(config, resolve=True),
    }

    wandb_logger = WandbLogger(**wandb_kwargs)

    return wandb_logger


def create_data_module(config: DictConfig) -> pl.LightningDataModule:
    """Create data module.

    Args:
        config: Configuration dictionary
    """
    return MassSpecDataModule(
        labels_path=config.model.data_module.labels_path,
        splits_path=config.model.data_module.splits_path,
        dataset_type=config.model.data_module.dataset_type,
        dataset_config=config.model.data_module.dataset_config,
        batch_size=config.model.data_module.batch_size,
        num_workers=config.model.data_module.num_workers,
        training_data_fraction=config.model.data_module.training_data_fraction,
    )


def log_training_summary(
    trainer: pl.Trainer,
    val_results: List[Dict[str, Any]],
    test_results: List[Dict[str, Any]],
    wandb_logger: WandbLogger,
) -> Dict[str, float]:
    """Log training summary and return final metrics.

    Args:
        trainer: PyTorch Lightning trainer
        val_results: Validation results
        test_results: Test results
        wandb_logger: W&B logger

    Returns:
        Dictionary with final metrics
    """
    # Extract final metrics
    final_metrics = {}

    if val_results:
        final_metrics.update(val_results[0])

    if test_results:
        final_metrics.update(test_results[0])

    # Log to W&B
    wandb_logger.experiment.log(
        {
            "final_val_loss": final_metrics.get("val_loss", 0.0),
            "final_test_loss": final_metrics.get("test_loss", 0.0),
            "training_completed": True,
            "best_model_path": trainer.checkpoint_callback.best_model_path
            if trainer.checkpoint_callback
            else None,
        }
    )

    # Log summary
    logger.info("=" * 50)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 50)
    logger.info(
        f"Best validation loss: {final_metrics.get('val_loss', 'N/A'):.6f}"
    )
    logger.info(f"Test loss: {final_metrics.get('test_loss', 'N/A'):.6f}")
    logger.info(
        f"Best model saved at: {trainer.checkpoint_callback.best_model_path if trainer.checkpoint_callback else 'N/A'}"
    )
    logger.info("=" * 50)

    return final_metrics


def resolve_ckpt_path(config: DictConfig) -> Optional[str]:
    # 1. Explicit override
    ckpt_path = getattr(config.system, "ckpt_path", None)
    if ckpt_path:
        return ckpt_path

    # 2. Get the Hydra Output Dir (The trial-specific folder)
    # This ensures we find the checkpoint for THIS specific trial
    cwd = Path(os.getcwd())
    last_ckpt = cwd / "checkpoints" / "last.ckpt"

    if last_ckpt.exists():
        logger.info(f"Auto-resuming trial from {last_ckpt}")
        return str(last_ckpt)

    return None


def get_wandb_resume_id() -> Optional[str]:
    """Try to find a previous WandB run ID for resuming.

    Looks for the run ID in the ``wandb/latest-run`` symlink.

    Returns:
        Previous WandB run ID, or None.
    """
    latest_run = Path("wandb/latest-run")
    if latest_run.exists():
        # The latest-run symlink points to run-<YYYYMMDD>_<HHMMSS>-<run_id>
        target = Path(os.readlink(latest_run)).name
        # Format: run-YYYYMMDD_HHMMSS-<run_id>
        parts = target.split("-")
        if len(parts) >= 3:
            run_id = parts[-1]
            logger.info(f"Found previous WandB run ID: {run_id}")
            return run_id
    return None


@hydra.main(
    config_path="../../examples/configs",
    config_name="config",
    version_base=None,
)
def main(config: DictConfig) -> float:
    """Training entrypoint.

    Args:
        config: Configuration dictionary

    Returns:
        Validation loss
    """
    # Reproducibility
    torch.multiprocessing.set_start_method("spawn", force=True)
    torch.set_float32_matmul_precision("medium")
    pl.seed_everything(config.system.seed)

    # Graceful shutdown on SIGTERM (sent by SLURM before preemption/cancel)
    def _sigterm_handler(signum, frame):
        logger.warning("Received SIGTERM — shutting down gracefully")
        try:
            wandb.finish(quiet=True)
        except Exception:
            pass
        raise SystemExit(1)

    signal.signal(signal.SIGTERM, _sigterm_handler)

    try:
        # Check for checkpoint to resume from
        ckpt_path = resolve_ckpt_path(config)

        # Setup logging (with WandB resume if resuming)
        prev_run_id = get_wandb_resume_id() if ckpt_path else None
        wandb_logger = create_logger(config, resume_run_id=prev_run_id)

        # Setup data
        data_module = create_data_module(config)

        # Create model (get_model_args uses only config and tree_processor, set in __init__)
        model_args = data_module.get_model_args()
        model = hydra.utils.instantiate(
            config.model.architecture, **model_args
        )

        # Compile model with PyTorch 2.0+ for additional speedup
        # Only compile if torch.compile is available (PyTorch >= 2.0)
        if hasattr(torch, "compile") and not config.system.debug:
            compile_mode = getattr(
                config.system, "compile_mode", "reduce-overhead"
            )
            logger.info(
                f"Compiling model with torch.compile (mode={compile_mode}) for optimized GPU execution..."
            )
            try:
                # Use configured mode (default: "reduce-overhead" for training workloads)
                model = torch.compile(model, mode=compile_mode)
                logger.info("Model compiled successfully")
            except Exception as e:
                logger.warning(
                    f"Failed to compile model: {e}. Proceeding without compilation."
                )

        # Setup training components
        callbacks = create_callbacks(config)
        trainer = create_trainer(config, callbacks, wandb_logger)

        # Training with proper validation (resume from checkpoint if available)
        trainer.fit(model=model, datamodule=data_module, ckpt_path=ckpt_path)

        # Final validation on validation set
        val_results = trainer.validate(model=model, datamodule=data_module)

        # Test on test set
        test_results = trainer.test(model=model, datamodule=data_module)

        # Log final results
        final_metrics = log_training_summary(
            trainer, val_results, test_results, wandb_logger
        )

        # Close W&B
        wandb.finish()

        return float(final_metrics.get("val_loss", 0.0))

    except Exception as e:
        # Log error and cleanup
        if "wandb_logger" in locals():
            wandb_logger.experiment.log(
                {"training_error": str(e), "training_completed": False}
            )
            wandb.finish()

        # Re-raise for proper error handling
        raise RuntimeError(f"Training failed: {str(e)}") from e


if __name__ == "__main__":
    main()
