"""NEIMS: Mass spectrum prediction models.

Contains the base class NEIMSBase with shared prediction logic, and the
NEIMS ECFP-based MLP implementation. Based on the model from
ACS Cent. Sci. 2019, 5, 700-708.
"""

from typing import List

import torch
import torch.nn as nn


def generalized_mse_loss(
    pred: torch.Tensor, target: torch.Tensor, mass_power: float = 0.5
) -> torch.Tensor:
    """Compute generalized MSE loss with mass-weighted error.

    Scale-invariant ratio loss: sum(w * (t-p)^2) / (sum(w * p^2) + sum(w * t^2)).
    Denominator uses both pred and target to avoid explosion when pred ~ 0 at
    the start of training (especially for GNN models that initialize near zero).
    """
    mass_indices = torch.arange(
        pred.shape[1], device=pred.device, dtype=torch.float32
    )
    mass_weights = torch.pow(mass_indices + 1, mass_power)

    weighted_squared_error = mass_weights[None, :] * torch.pow(target - pred, 2)
    weighted_pred_squared = mass_weights[None, :] * torch.pow(pred, 2)
    weighted_target_squared = mass_weights[None, :] * torch.pow(target, 2)

    numerator = torch.sum(weighted_squared_error, dim=1)
    denominator = torch.sum(weighted_pred_squared + weighted_target_squared, dim=1)
    denominator = torch.maximum(denominator, torch.tensor(1e-6, device=pred.device))

    loss = numerator / denominator
    return torch.mean(loss)


class NEIMSBase(nn.Module):
    """Base class for NEIMS spectrum prediction models.

    Provides the shared prediction head logic (bidirectional, GLU, or simple
    dense) and utility methods for mass masking and reverse prediction.
    Subclasses implement their own encoder (fingerprint MLP or GNN) and call
    ``_predict()`` to apply the prediction head.
    """

    def __init__(
        self,
        output_size: int,
        max_mass_offset: int,
        bidirectional: bool = False,
        gate_bidirectional: bool = False,
        use_glu: bool = False,
    ):
        """Initialize base model.

        Parameters
        ----------
        output_size : int
            Number of output m/z bins.
        max_mass_offset : int
            Mass masking tolerance in Da.
        bidirectional : bool
            Use forward + backward prediction heads.
        gate_bidirectional : bool
            Learn a gate between forward and backward predictions.
        use_glu : bool
            Use Gated Linear Unit output (Zhu et al. 2020).
        """
        super().__init__()
        self.output_size = output_size
        self.max_mass_offset = max_mass_offset
        self.bidirectional = bidirectional
        self.gate_bidirectional = gate_bidirectional
        self.use_glu = use_glu

    @staticmethod
    def _make_residual_block(size: int, bottleneck: float, dropout: float):
        """Create a residual block with BatchNorm, ReLU, dropout, bottleneck."""
        return nn.Sequential(
            nn.BatchNorm1d(size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(size, int(size * bottleneck)),
            nn.BatchNorm1d(int(size * bottleneck)),
            nn.ReLU(),
            nn.Linear(int(size * bottleneck), size),
        )

    def _build_prediction_head(self, pred_input_size: int):
        """Build prediction head modules.

        Must be called by subclasses after determining ``pred_input_size``.
        Creates the appropriate Linear layers as attributes on ``self``.
        """
        if self.use_glu:
            self.glu_linear = nn.Linear(pred_input_size, self.output_size)
            self.glu_gate = nn.Linear(pred_input_size, self.output_size)
        elif self.bidirectional:
            self.forward_predictor = nn.Linear(pred_input_size, self.output_size)
            self.backward_predictor = nn.Linear(pred_input_size, self.output_size)
            if self.gate_bidirectional:
                self.gate_predictor = nn.Linear(pred_input_size, self.output_size)
        else:
            self.predictor = nn.Linear(pred_input_size, self.output_size)

    def _mask_by_mass(
        self, pred: torch.Tensor, masses: torch.Tensor
    ) -> torch.Tensor:
        """Mask predictions beyond molecule mass + tolerance."""
        batch_size = pred.shape[0]
        mass_indices = torch.arange(pred.shape[1], device=pred.device).expand(
            batch_size, -1
        )

        masses = masses.view(-1, 1)
        mask = mass_indices <= (masses + self.max_mass_offset)

        return pred * mask.float()

    def _reverse_prediction(
        self, raw_prediction: torch.Tensor, masses: torch.Tensor
    ) -> torch.Tensor:
        """Mirror prediction around molecular mass position.

        Implements scatter_by_anchor_indices from the reference:
        output[i][j] = data[i][anchor[i] - j + max_mass_offset]

        This maps predictions from "offset from molecular mass" coordinates
        back to absolute m/z coordinates.
        """
        num_columns = raw_prediction.shape[1]
        device = raw_prediction.device

        anchor_indices = torch.round(masses.squeeze(-1)).long()

        j_indices = torch.arange(num_columns, device=device)

        source_indices = (
            anchor_indices.unsqueeze(1)
            - j_indices.unsqueeze(0)
            + self.max_mass_offset
        )

        valid_mask = (source_indices >= 0) & (source_indices < num_columns)
        safe_indices = source_indices.clamp(0, num_columns - 1)

        output = torch.gather(raw_prediction, 1, safe_indices)
        output = output * valid_mask.float()

        return output

    def _predict(self, features: torch.Tensor, masses: torch.Tensor) -> torch.Tensor:
        """Apply prediction head to encoder features.

        Parameters
        ----------
        features : torch.Tensor
            Encoder output, shape [batch_size, pred_input_size].
        masses : torch.Tensor
            Molecular masses, shape [batch_size] or [batch_size, 1].

        Returns
        -------
        torch.Tensor
            Predicted spectrum, shape [batch_size, output_size].
        """
        if self.use_glu:
            linear_out = self.glu_linear(features)
            gate_out = torch.sigmoid(self.glu_gate(features))
            prediction = linear_out * gate_out
            prediction = self._mask_by_mass(prediction, masses)
        elif self.bidirectional:
            forward_pred = self.forward_predictor(features)
            forward_pred = self._mask_by_mass(forward_pred, masses)

            backward_pred = self.backward_predictor(features)
            backward_pred = self._reverse_prediction(backward_pred, masses)

            if self.gate_bidirectional:
                gate = torch.sigmoid(self.gate_predictor(features))
                prediction = gate * forward_pred + (1 - gate) * backward_pred
            else:
                prediction = forward_pred + backward_pred
        else:
            prediction = self.predictor(features)
            prediction = self._mask_by_mass(prediction, masses)

        prediction = torch.relu(prediction)
        return prediction


class NEIMS(NEIMSBase):
    """NEIMS: ECFP-based MLP for mass spectrum prediction."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_sizes: List[int],
        dropout: float,
        bidirectional: bool,
        gate_bidirectional: bool,
        resnet_bottleneck: float,
        max_mass_offset: int,
        fp_radius: int = 2,
        fp_length: int = 4096,
    ):
        """Initialize NEIMS model.

        Args:
            input_size: Size of input fingerprint
            output_size: Size of output spectrum
            hidden_sizes: List of hidden layer sizes
            dropout: Dropout rate
            bidirectional: Whether to use bidirectional prediction
            gate_bidirectional: Whether to use gated bidirectional prediction
            resnet_bottleneck: Bottleneck factor for residual blocks
            max_mass_offset: Maximum mass offset for masking
            fp_radius: Morgan fingerprint radius
            fp_length: Morgan fingerprint length
        """
        super().__init__(
            output_size=output_size,
            max_mass_offset=max_mass_offset,
            bidirectional=bidirectional,
            gate_bidirectional=gate_bidirectional,
        )

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.resnet_bottleneck = resnet_bottleneck
        self.fp_radius = fp_radius
        self.fp_length = fp_length

        if input_size != fp_length:
            raise ValueError(
                f"input_size ({input_size}) must match fp_length ({fp_length})"
            )

        # Input layer
        self.input_layer = nn.Linear(input_size, hidden_sizes[0])
        self.input_bn = nn.BatchNorm1d(hidden_sizes[0])

        # Residual blocks
        self.residual_blocks = nn.ModuleList()
        for i in range(len(hidden_sizes) - 1):
            self.residual_blocks.append(
                self._make_residual_block(hidden_sizes[i], resnet_bottleneck, dropout)
            )

        final_hidden_size = hidden_sizes[-1]
        self.final_bn = nn.BatchNorm1d(final_hidden_size)

        # Prediction head
        self._build_prediction_head(final_hidden_size)

    def forward(
        self, fingerprints: torch.Tensor, masses: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            fingerprints: ECFP fingerprints [batch_size, fp_length]
            masses: Molecular masses [batch_size, 1]

        Returns:
            Predicted spectrum [batch_size, output_size]
        """
        # Input layer
        x = torch.relu(self.input_bn(self.input_layer(fingerprints)))

        # Residual blocks
        for block in self.residual_blocks:
            x = x + block(x)

        features = torch.relu(self.final_bn(x))

        return self._predict(features, masses)
