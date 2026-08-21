"""Set attention block."""

import torch.nn as nn

from .multi_head_attention import MultiHeadAttention


class SetAttentionBlock(nn.Module):
    r"""SAB block introduced in Set-Transformer paper.

    Parameters
    ----------
    d_model : int
        The feature size (input and output) in Multi-Head Attention layer.
    num_heads : int
        The number of heads.
    d_head : int
        The hidden size per head.
    d_ff : int
        The inner hidden size in the Feed-Forward Neural Network.
    dropouth : float
        The dropout rate of each sublayer.
    dropouta : float
        The dropout rate of attention heads.

    Notes
    -----
    This module was used in SetTransformer layer.
    """

    def __init__(
        self, d_model, num_heads, d_head, d_ff, dropouth=0.0, dropouta=0.0
    ):
        super(SetAttentionBlock, self).__init__()
        self.mha = MultiHeadAttention(
            d_model,
            num_heads,
            d_head,
            d_ff,
            dropouth=dropouth,
            dropouta=dropouta,
        )

    def forward(self, feat, lengths):
        """Compute a Set Attention Block.

        Parameters
        ----------
        feat : torch.Tensor
            The input feature.
        lengths : list
            The array of node numbers, used to segment feat tensor.
        """
        return self.mha(feat, feat, lengths, lengths)
