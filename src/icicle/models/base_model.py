"""Base model class for spectrum prediction.

This base model is used for all models and baselines. It sets the output size
(the binned spectrum) and declares some basic functions.
"""

from typing import Any, Dict, List, Optional, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn


class BaseSpectrumPredictor(pl.LightningModule):
    """Base class for spectrum prediction models.

    This class provides a flexible interface for different types of spectrum
    prediction models. Subclasses should implement the core methods with their
    specific logic. Methods are:
    - forward
    - predict_mol
    - training_step
    - validation_step
    - test_step
    - configure_optimizers
    """

    def __init__(
        self,
        min_mz: float,
        max_mz: float,
        bin_width: float,
        **kwargs,
    ):
        """Initialize base model.

        Args:
            min_mz: Minimum m/z value
            max_mz: Maximum m/z value
            bin_width: Width of m/z bins
            **kwargs: Additional model-specific parameters
        """
        super().__init__()
        self.min_mz = min_mz
        self.max_mz = max_mz
        self.bin_width = bin_width

        # Save all hyperparameters for Lightning
        self.save_hyperparameters()

        self._init_weights()

    def _init_weights(self):
        """Proper weight initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            batch: Dictionary containing model inputs

        Returns:
            Model predictions
        """
        raise NotImplementedError("Subclasses must implement forward()")

    def predict_mol(
        self, smi: Union[str, List[str]], device: str = "cpu", **kwargs
    ) -> Dict[str, Any]:
        """High-level prediction interface for molecules.

        Args:
            smi: SMILES string(s) of molecule(s) to predict
            device: Device to run prediction on
            **kwargs: Additional model-specific arguments

        Returns:
            Dictionary containing predictions
        """
        raise NotImplementedError("Subclasses must implement predict_mol()")

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Training step.

        Args:
            batch: Dictionary containing model inputs and targets
            batch_idx: Index of current batch

        Returns:
            Loss tensor
        """
        raise NotImplementedError("Subclasses must implement training_step()")

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Optional[torch.Tensor]:
        """Validation step.

        Args:
            batch: Dictionary containing model inputs and targets
            batch_idx: Index of current batch

        Returns:
            Optional loss tensor
        """
        raise NotImplementedError(
            "Subclasses must implement validation_step()"
        )

    def test_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> None:
        """Test step.

        Args:
            batch: Dictionary containing model inputs and targets
            batch_idx: Index of current batch
        """
        raise NotImplementedError("Subclasses must implement test_step()")

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        raise NotImplementedError(
            "Subclasses must implement configure_optimizers()"
        )

    def _format_output(
        self, predictions: torch.Tensor, batched: bool
    ) -> Dict[str, Any]:
        """Helper method to format output consistently.

        Args:
            predictions: Model predictions tensor
            batched: Whether input was batched

        Returns:
            Formatted output dictionary
        """
        if batched:
            return {"spec": [pred.cpu() for pred in predictions]}
        else:
            return {"spec": predictions[0].cpu()}


class BaseFragmentGenerator(BaseSpectrumPredictor):
    """Base class for fragmentation models.

    This class extends BaseSpectrumPredictor with fragmentation-specific
    functionality.

    Methods are:
    - predict_fragmentation
    """

    def predict_fragmentation(
        self,
        root_smiles: str | List[str],
        **kwargs,
    ) -> Dict[str, Any] | List[Dict[str, Any]]:
        """Predict fragmentation tree/pattern.

        Args:
            root_smiles: SMILES string(s) of molecule(s) to predict
            **kwargs: Additional model-specific arguments

        Returns:
            Fragmentation predictions
        """
        raise NotImplementedError(
            "Subclasses must implement predict_fragmentation()"
        )


class BaseIntensityModel(BaseSpectrumPredictor):
    """Base class for intensity prediction models.

    This class extends BaseSpectrumPredictor with intensity-specific
    functionality.

    Methods are:
    - predict_intensities
    """

    def predict_intensities(
        self,
        graphs: Any,
        root_reprs: Any,
        ind_maps: torch.Tensor,
        num_frags: torch.Tensor,
        **kwargs,
    ) -> Dict[str, List[torch.Tensor]]:
        """Predict intensities for given fragments.

        Args:
            graphs: Fragment graphs
            root_reprs: Root molecule representations
            ind_maps: Index mappings
            num_frags: Number of fragments
            **kwargs: Additional model-specific arguments

        Returns:
            Intensity predictions
        """
        raise NotImplementedError(
            "Subclasses must implement predict_intensities()"
        )
