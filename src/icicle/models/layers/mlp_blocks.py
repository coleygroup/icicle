"""Multi-layer perceptron (MLP) blocks."""

import copy
from typing import Optional

import torch.nn as nn


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class MLPBlocks(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dropout: float,
        num_layers: int,
        output_size: Optional[int] = None,
        use_residuals: bool = False,
        use_batchnorm: bool = False,
    ):
        super().__init__()
        self.activation = nn.ReLU()
        self.dropout_layer = nn.Dropout(p=dropout)
        self.input_layer = nn.Linear(input_size, hidden_size)
        middle_layer = nn.Linear(hidden_size, hidden_size)
        self.layers = get_clones(middle_layer, num_layers - 1)

        self.output_layer = None
        self.output_size = output_size
        if self.output_size is not None:
            self.output_layer = nn.Linear(hidden_size, self.output_size)

        self.use_residuals = use_residuals
        self.use_batchnorm = use_batchnorm
        if self.use_batchnorm:
            self.bn_input = nn.BatchNorm1d(hidden_size)
            bn = nn.BatchNorm1d(hidden_size)
            self.bn_mids = get_clones(bn, num_layers - 1)

    def safe_apply_bn(self, x, bn):
        """Transpose and untranspose after linear for 3 dim items to us
        batchnorm."""
        temp_shape = x.shape
        if len(x.shape) == 2:
            return bn(x)
        elif len(x.shape) == 3:
            return bn(x.transpose(-1, -2)).transpose(-1, -2)
        else:
            raise NotImplementedError()

    def forward(self, x):
        output = x
        output = self.input_layer(x)
        output = self.activation(output)
        output = self.dropout_layer(output)

        if self.use_batchnorm:
            output = self.safe_apply_bn(output, self.bn_input)

        old_op = output
        for layer_index, layer in enumerate(self.layers):
            output = layer(output)
            output = self.activation(output)
            output = self.dropout_layer(output)

            if self.use_batchnorm:
                output = self.safe_apply_bn(output, self.bn_mids[layer_index])

            if self.use_residuals:
                output = output + old_op
                old_op = output

        if self.output_layer is not None:
            output = self.output_layer(output)

        return output
