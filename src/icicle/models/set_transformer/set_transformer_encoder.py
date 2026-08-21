"""Set transformer encoder."""

from torch import nn

from .set_attention_block import SetAttentionBlock


class SetTransformerEncoder(nn.Module):
    r"""

    Description
    -----------
    The Encoder module in `Set Transformer: A Framework for Attention-based
    Permutation-Invariant Neural Networks <https://arxiv.org/pdf/1810.00825.pdf>`__.

    Parameters
    ----------
    d_model : int
        The hidden size of the model.
    n_heads : int
        The number of heads.
    d_head : int
        The hidden size of each head.
    d_ff : int
        The kernel size in FFN (Positionwise Feed-Forward Network) layer.
    n_layers : int
        The number of layers.
    block_type : str
        Building block type: 'sab' (Set Attention Block) or 'isab' (Induced
        Set Attention Block).
    m : int or None
        The number of induced vectors in ISAB Block. Set to None if block type
        is 'sab'.
    dropouth : float
        The dropout rate of each sublayer.
    dropouta : float
        The dropout rate of attention heads.

    Examples
    --------
    >>> import dgl
    >>> import torch as th
    >>> from dgl.nn import SetTransformerEncoder
    >>>
    >>> g1 = dgl.rand_graph(3, 4)  # g1 is a random graph with 3 nodes and 4 edges
    >>> g1_node_feats = torch.rand(3, 5)  # feature size is 5
    >>> g1_node_feats
    tensor([[0.8948, 0.0699, 0.9137, 0.7567, 0.3637],
            [0.8137, 0.8938, 0.8377, 0.4249, 0.6118],
            [0.5197, 0.9030, 0.6825, 0.5725, 0.4755]])
    >>>
    >>> g2 = dgl.rand_graph(4, 6)  # g2 is a random graph with 4 nodes and 6 edges
    >>> g2_node_feats = torch.rand(4, 5)  # feature size is 5
    >>> g2_node_feats
    tensor([[0.2053, 0.2426, 0.4111, 0.9028, 0.5658],
            [0.5278, 0.6365, 0.9990, 0.2351, 0.8945],
            [0.3134, 0.0580, 0.4349, 0.7949, 0.3891],
            [0.0142, 0.2709, 0.3330, 0.8521, 0.6925]])
    >>>
    >>> set_trans_enc = SetTransformerEncoder(5, 4, 4, 20)  # create a settrans encoder.

    Case 1: Input a single graph

    >>> set_trans_enc(g1, g1_node_feats)
    tensor([[ 0.1262, -1.9081,  0.7287,  0.1678,  0.8854],
            [-0.0634, -1.1996,  0.6955, -0.9230,  1.4904],
            [-0.9972, -0.7924,  0.6907, -0.5221,  1.6211]],
           grad_fn=<NativeLayerNormBackward>)

    Case 2: Input a batch of graphs

    Build a batch of DGL graphs and concatenate all graphs' node features into one tensor.

    >>> batch_g = dgl.batch([g1, g2])
    >>> batch_f = torch.cat([g1_node_feats, g2_node_feats])
    >>>
    >>> set_trans_enc(batch_g, batch_f)
    tensor([[ 0.1262, -1.9081,  0.7287,  0.1678,  0.8854],
            [-0.0634, -1.1996,  0.6955, -0.9230,  1.4904],
            [-0.9972, -0.7924,  0.6907, -0.5221,  1.6211],
            [-0.7973, -1.3203,  0.0634,  0.5237,  1.5306],
            [-0.4497, -1.0920,  0.8470, -0.8030,  1.4977],
            [-0.4940, -1.6045,  0.2363,  0.4885,  1.3737],
            [-0.9840, -1.0913, -0.0099,  0.4653,  1.6199]],
           grad_fn=<NativeLayerNormBackward>)

    See Also
    --------
    SetTransformerDecoder

    Notes
    -----
    SetTransformerEncoder is not a readout layer, the tensor it returned is nodewise
    representation instead out graphwise representation, and the SetTransformerDecoder
    would return a graph readout tensor.
    """

    def __init__(
        self,
        d_model,
        n_heads,
        d_head,
        d_ff,
        n_layers=1,
        block_type="sab",
        m=None,
        dropouth=0.0,
        dropouta=0.0,
    ):
        super(SetTransformerEncoder, self).__init__()
        self.n_layers = n_layers
        self.block_type = block_type
        self.m = m
        layers = []
        if block_type == "isab" and m is None:
            raise KeyError(
                "The number of inducing points is not specified in ISAB block."
            )

        for _ in range(n_layers):
            if block_type == "sab":
                layers.append(
                    SetAttentionBlock(
                        d_model,
                        n_heads,
                        d_head,
                        d_ff,
                        dropouth=dropouth,
                        dropouta=dropouta,
                    )
                )
            else:
                raise KeyError(
                    "Unrecognized block type {}: we only support sab/isab"
                )

        self.layers = nn.ModuleList(layers)

    def forward(self, graph, feat):
        """Compute the Encoder part of Set Transformer.

        Parameters
        ----------
        graph : DGLGraph
            The input graph.
        feat : torch.Tensor
            The input feature with shape :math:`(N, D)`, where :math:`N` is the
            number of nodes in the graph.

        Returns
        -------
        torch.Tensor
            The output feature with shape :math:`(N, D)`.
        """
        lengths = graph.batch_num_nodes()
        for layer in self.layers:
            feat = layer(feat, lengths)
        return feat
