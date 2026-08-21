"""Multi-head attention block."""

import numpy as np
import torch
from torch import nn
import torch as th

# DGL Models
# https://docs.dgl.ai/en/0.6.x/_modules/dgl/nn/pytorch/glob.html#SetTransformerDecoder

from icicle.utils.graph_utils import pad_packed_tensor


def is_tensor(obj):
    return isinstance(obj, th.Tensor)


def as_scalar(data):
    return data.item()


def pack_padded_tensor(input, lengths):
    max_len = input.shape[1]
    device = input.device
    if not is_tensor(lengths):
        lengths = th.tensor(lengths, dtype=th.int64, device=device)
    else:
        lengths = lengths.to(device)
    input = input.view(-1, *input.shape[2:])
    out_len = lengths.sum().item()
    index = th.ones(out_len, dtype=th.int64, device=device)
    cum_lengths = th.cumsum(lengths, 0)
    index[cum_lengths[:-1]] += max_len - lengths[:-1]
    index = th.cumsum(index, 0) - 1
    return input[index]


class MultiHeadAttention(nn.Module):
    r"""Multi-Head Attention block, used in Transformer, Set Transformer and so
    on.

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
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_head
        self.d_ff = d_ff
        self.proj_q = nn.Linear(d_model, num_heads * d_head, bias=False)
        self.proj_k = nn.Linear(d_model, num_heads * d_head, bias=False)
        self.proj_v = nn.Linear(d_model, num_heads * d_head, bias=False)
        self.proj_o = nn.Linear(num_heads * d_head, d_model, bias=False)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropouth),
            nn.Linear(d_ff, d_model),
        )
        self.droph = nn.Dropout(dropouth)
        self.dropa = nn.Dropout(dropouta)
        self.norm_in = nn.LayerNorm(d_model)
        self.norm_inter = nn.LayerNorm(d_model)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def self_attention(self, x, mem, lengths_x, lengths_mem):
        batch_size = len(lengths_x)
        max_len_x = max(lengths_x)
        max_len_mem = max(lengths_mem)
        device = x.device

        lengths_x = lengths_x.clone().detach().long().to(device)
        lengths_mem = lengths_mem.clone().detach().long().to(device)

        queries = self.proj_q(x).view(-1, self.num_heads, self.d_head)
        keys = self.proj_k(mem).view(-1, self.num_heads, self.d_head)
        values = self.proj_v(mem).view(-1, self.num_heads, self.d_head)

        # padding to (B, max_len_x/mem, num_heads, d_head)
        queries = pad_packed_tensor(queries, lengths_x, 0)
        keys = pad_packed_tensor(keys, lengths_mem, 0)
        values = pad_packed_tensor(values, lengths_mem, 0)

        # attention score with shape (B, num_heads, max_len_x, max_len_mem)
        e = torch.einsum("bxhd,byhd->bhxy", queries, keys)
        # normalize
        e = e / np.sqrt(self.d_head)

        # generate mask
        mask = _gen_mask(lengths_x, lengths_mem, max_len_x, max_len_mem)
        e = e.masked_fill(mask == 0, -float("inf"))

        # apply softmax
        alpha = torch.softmax(e, dim=-1)
        # the following line addresses the NaN issue, see
        # https://github.com/dmlc/dgl/issues/2657
        alpha = alpha.masked_fill(mask == 0, 0.0)

        # sum of value weighted by alpha
        out = torch.einsum("bhxy,byhd->bxhd", alpha, values)
        # project to output
        out = self.proj_o(
            out.contiguous().view(
                batch_size, max_len_x, self.num_heads * self.d_head
            )
        )
        # pack tensor
        out = pack_padded_tensor(out, lengths_x)
        return out

    def forward(self, x, mem, lengths_x, lengths_mem):
        """Compute multi-head self-attention.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor used to compute queries.
        mem : torch.Tensor
            The memory tensor used to compute keys and values.
        lengths_x : list
            The array of node numbers, used to segment x.
        lengths_mem : list
            The array of node numbers, used to segment mem.
        """
        ### Following a _pre_ transformer

        # intra norm
        x = x + self.self_attention(
            self.norm_in(x), mem, lengths_x, lengths_mem
        )

        # inter norm
        x = x + self.ffn(self.norm_inter(x))

        ## intra norm
        # x = self.norm_in(x + out)

        ## inter norm
        # x = self.norm_inter(x + self.ffn(x))
        return x


def _gen_mask(lengths_x, lengths_y, max_len_x, max_len_y):
    """Generate binary mask array for given x and y input pairs.

    Parameters
    ----------
    lengths_x : Tensor
        The int tensor indicates the segment information of x.
    lengths_y : Tensor
        The int tensor indicates the segment information of y.
    max_len_x : int
        The maximum element in lengths_x.
    max_len_y : int
        The maximum element in lengths_y.

    Returns
    -------
    Tensor
        the mask tensor with shape (batch_size, 1, max_len_x, max_len_y)
    """
    device = lengths_x.device
    # x_mask: (batch_size, max_len_x)
    x_mask = torch.arange(max_len_x, device=device).unsqueeze(
        0
    ) < lengths_x.unsqueeze(1)
    # y_mask: (batch_size, max_len_y)
    y_mask = torch.arange(max_len_y, device=device).unsqueeze(
        0
    ) < lengths_y.unsqueeze(1)
    # mask: (batch_size, 1, max_len_x, max_len_y)
    mask = (x_mask.unsqueeze(-1) & y_mask.unsqueeze(-2)).unsqueeze(1)
    return mask
