"""NEIMS-GNN: Graph Neural Network for mass spectrum prediction.

Based on Zhu, Liu, Hassoun. "Using Graph Neural Networks for Mass Spectrometry
Prediction." arXiv:2010.04661 (2020).

Architecture (Section 2.3):
    1. Multiple GNN layers (GCN or GAT) with ReLU activation
    2. Global pooling (max, mean, or attention)
    3. Feed-forward prediction: Dense layer or GLU, activated by ReLU
    4. Mass masking: zero bins above molecular weight bin (ms-pred style)
"""

from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GATConv,
    GCNConv,
    GlobalAttention,
    global_max_pool,
    global_mean_pool,
)

VALID_ELEMENTS = [
    "C", "N", "P", "O", "S", "Si", "I", "H", "Cl", "F",
    "Br", "B", "Se", "Fe", "Co", "As", "Na", "K",
]

# 1 (weight) + len(VALID_ELEMENTS) + 1 (other) + 1 (degree) + 1 (num Hs)
# + 1 (in ring) + 1 (aromatic)
NUM_ATOM_FEATURES = 1 + len(VALID_ELEMENTS) + 1 + 1 + 1 + 1 + 1

BOND_TYPES = {1: 0, 2: 1, 3: 2, 12: 3}  # single, double, triple, aromatic
NUM_BOND_FEATURES = len(BOND_TYPES) + 1 + 1  # type one-hot + conjugated + in_ring


def atom_features(atom):
    """Compute atom features (Section 2.2 of Zhu et al. 2020).

    Features: standard atomic weight, atom type one-hot, number of bonds,
    number of neighboring hydrogens, is in ring, is in aromatic ring.
    """
    weight = atom.GetMass()
    symbol = atom.GetSymbol()
    type_encoding = [1.0 if symbol == t else 0.0 for t in VALID_ELEMENTS]
    type_encoding.append(1.0 if symbol not in VALID_ELEMENTS else 0.0)
    num_bonds = float(atom.GetDegree())
    num_hs = float(atom.GetTotalNumHs())
    in_ring = 1.0 if atom.IsInRing() else 0.0
    in_aromatic = 1.0 if atom.GetIsAromatic() else 0.0
    return [weight] + type_encoding + [num_bonds, num_hs, in_ring, in_aromatic]


def bond_features(bond):
    """Compute bond features (6-dim): type one-hot + conjugated + in_ring."""
    bt = int(bond.GetBondTypeAsDouble())
    if bond.GetIsAromatic():
        bt = 12
    type_encoding = [0.0] * len(BOND_TYPES)
    if bt in BOND_TYPES:
        type_encoding[BOND_TYPES[bt]] = 1.0
    conjugated = 1.0 if bond.GetIsConjugated() else 0.0
    in_ring = 1.0 if bond.IsInRing() else 0.0
    return type_encoding + [conjugated, in_ring]


class NEIMSGNN(nn.Module):
    """NEIMS-GNN: faithful implementation of Zhu et al. 2020.

    GNN encoder (GCN or GAT) -> global pooling -> GLU or dense output -> ReLU
    -> mass mask (zero bins above molecular weight + offset).
    """

    def __init__(
        self,
        output_size: int = 1000,
        gnn_type: str = "GAT",
        gnn_hidden_size: int = 64,
        gnn_num_layers: int = 10,
        gnn_num_heads: int = 8,
        gnn_dropout: float = 0.5,
        pool_type: str = "max",
        use_edge_features: bool = False,
        use_glu: bool = True,
        node_feat_dim: int = NUM_ATOM_FEATURES,
        max_mz: float = 750.0,
        # Unused kwargs kept for CLI compat
        ffnn_hidden_sizes: Optional[List[int]] = None,
        ffnn_dropout: float = 0.25,
        resnet_bottleneck: float = 0.5,
        bidirectional: bool = False,
        gate_bidirectional: bool = False,
        max_mass_offset: int = 5,
    ):
        super().__init__()

        self.gnn_type = gnn_type
        self.gnn_hidden_size = gnn_hidden_size
        self.gnn_num_layers = gnn_num_layers
        self.pool_type = pool_type
        self.use_edge_features = use_edge_features
        self.use_glu = use_glu
        self.output_size = output_size

        # Bin centers matching the dataset: linspace(min_mz, max_mz, output_size)
        # Used to find the bin index of each molecule's MW (ms-pred style masking).
        bin_masses = torch.from_numpy(
            np.linspace(0, max_mz, output_size).astype(np.float32)
        )
        self.register_buffer("bin_masses", bin_masses)

        edge_dim = NUM_BOND_FEATURES if use_edge_features else None

        self.gnn_layers = nn.ModuleList()

        if gnn_type == "GAT":
            assert gnn_hidden_size % gnn_num_heads == 0, (
                f"gnn_hidden_size ({gnn_hidden_size}) must be divisible by "
                f"gnn_num_heads ({gnn_num_heads})"
            )
            head_dim = gnn_hidden_size // gnn_num_heads
            self.gnn_layers.append(
                GATConv(node_feat_dim, head_dim, heads=gnn_num_heads,
                        dropout=gnn_dropout, edge_dim=edge_dim)
            )
            for _ in range(gnn_num_layers - 1):
                self.gnn_layers.append(
                    GATConv(gnn_hidden_size, head_dim, heads=gnn_num_heads,
                            dropout=gnn_dropout, edge_dim=edge_dim)
                )

        elif gnn_type == "GCN":
            self.gnn_layers.append(GCNConv(node_feat_dim, gnn_hidden_size))
            for _ in range(gnn_num_layers - 1):
                self.gnn_layers.append(GCNConv(gnn_hidden_size, gnn_hidden_size))

        else:
            raise ValueError(f"Unsupported gnn_type: {gnn_type}. Use 'GAT' or 'GCN'.")

        self.dropout = nn.Dropout(gnn_dropout)

        if pool_type == "attention":
            self.pool = GlobalAttention(nn.Linear(gnn_hidden_size, 1))

        # Prediction head
        if use_glu:
            self.glu_linear = nn.Linear(gnn_hidden_size, output_size)
            self.glu_gate = nn.Linear(gnn_hidden_size, output_size)
        else:
            self.predictor = nn.Linear(gnn_hidden_size, output_size)

    def forward(self, data, masses: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        data : torch_geometric.data.Batch
        masses : torch.Tensor
            Molecular masses [batch_size] or [batch_size, 1] in Da.
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = data.edge_attr if self.use_edge_features else None

        for conv in self.gnn_layers:
            if self.gnn_type == "GAT":
                x = conv(x, edge_index, edge_attr=edge_attr)
            else:
                x = conv(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)

        if self.pool_type == "max":
            x = global_max_pool(x, batch)
        elif self.pool_type == "mean":
            x = global_mean_pool(x, batch)
        elif self.pool_type == "attention":
            x = self.pool(x, batch)

        if self.use_glu:
            out = self.glu_linear(x) * torch.sigmoid(self.glu_gate(x))
        else:
            out = self.predictor(x)

        out = F.relu(out)

        # Mass masking: find the bin index of each molecule's MW, zero above it.
        # Equivalent to ms-pred ForwardGNN: argmax of (full_weight < bin_masses).
        masses_flat = masses.view(-1)
        full_mass_bin = (masses_flat[:, None] < self.bin_masses[None, :]).int().argmax(dim=-1)
        bin_arange = torch.arange(self.output_size, device=out.device)
        is_valid = bin_arange[None, :] <= full_mass_bin[:, None]
        return out * is_valid.float()
