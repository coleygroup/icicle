"""Helper functions for model operations."""

import logging
from typing import Dict, Optional

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig


def load_model(config: DictConfig, device: torch.device) -> pl.LightningModule:
    """Load and initialize model from configuration.

    Parameters
    ----------
    config : DictConfig
        Configuration containing model architecture and checkpoint path
    device : torch.device
        Device to load model onto

    Returns
    -------
    pl.LightningModule
        Loaded model in evaluation mode
    """
    model = hydra.utils.instantiate(config.eval.model.architecture)

    # Load checkpoint if specified
    if (
        hasattr(config.eval.model, "checkpoint_path")
        and config.eval.model.checkpoint_path
    ):
        try:
            model = model.load_from_checkpoint(
                config.eval.model.checkpoint_path,
                **config.eval.model.get("load_args", {}),
            )
            logging.info(
                f"Model loaded successfully from {config.eval.model.checkpoint_path}"
            )
        except Exception as e:
            logging.error(
                f"Failed to load model from checkpoint {config.eval.model.checkpoint_path}: {e}"
            )
            raise

    model.to(device)
    model.eval()
    return model


def get_or_predict_spectrum(
    smiles: str,
    model: pl.LightningModule,
    model_inference_params: Dict,
    predicted_spectra_cache: Dict[str, Optional[object]],
) -> Optional[object]:
    """Get predicted spectrum from cache or compute it.

    Parameters
    ----------
    smiles : str
        SMILES string of molecule to predict
    model : pl.LightningModule
        Model to use for prediction
    model_inference_params : Dict
        Parameters for model inference
    predicted_spectra_cache : Dict[str, Optional[object]]
        Cache mapping SMILES to predicted spectra

    Returns
    -------
    Optional[object]
        Predicted spectrum intensities, or None if prediction failed
    """
    if smiles not in predicted_spectra_cache:
        try:
            predicted_result = model.predict_from_smiles(
                smiles=smiles,
                **model_inference_params,
            )
            if (
                predicted_result["smiles"] != ""
                and predicted_result["intensities"] is not None
            ):
                predicted_spectra_cache[smiles] = predicted_result[
                    "intensities"
                ]
            else:
                predicted_spectra_cache[smiles] = None
        except Exception as e:
            logging.warning(f"Failed to predict spectrum for {smiles}: {e}")
            predicted_spectra_cache[smiles] = None

    return predicted_spectra_cache.get(smiles)
