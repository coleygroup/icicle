"""Module that encodes a molecule graph into a latent space."""

from typing import Any

from torch import nn

from ..layers import GGNN
from ..set_transformer.set_transformer_encoder import SetTransformerEncoder


class GNNEncoder(nn.Module):
    """GNNEncoder Module."""

    def __init__(
        self,
        hidden_size: int,
        num_step_message_passing: int = 4,
        gnn_node_feats: int = 74,
        gnn_edge_feats: int = 4,
        mpnn_type: str = "GGNN",
        node_feat_symbol: str = "h",
        set_transform_layers: int = 2,
        dropout: float = 0,
        **kwargs: Any,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.gnn_edge_feats = gnn_edge_feats
        self.gnn_node_feats = gnn_node_feats
        self.node_feat_symbol = node_feat_symbol
        self.dropout = dropout

        self.mpnn_type = mpnn_type
        self.hidden_size = hidden_size
        self.num_step_message_passing = num_step_message_passing
        self.input_project = nn.Linear(self.gnn_node_feats, self.hidden_size)

        self.gnn = GGNN(
            hidden_size=self.hidden_size,
            edge_feats=self.gnn_edge_feats,
            node_feats=self.gnn_node_feats,
            num_step_message_passing=num_step_message_passing,
            **kwargs,
        )

        self.set_transformer = SetTransformerEncoder(
            d_model=self.hidden_size,
            n_heads=4,
            d_head=self.hidden_size // 8,
            d_ff=hidden_size // 2,
            n_layers=set_transform_layers,
        )

    def forward(self, g):
        """Encode batch of molecule graph."""
        with g.local_scope():
            # Set initial hidden
            ndata = g.ndata[self.node_feat_symbol]
            edata = g.edata["e"]
            h_init = self.input_project(ndata)
            g.ndata.update({"_h": h_init})
            g.edata.update({"_e": edata})
            output = self.gnn(g, "_h", "_e")
        output = self.set_transformer(g, output)
        return output
