import math
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional


def scaled_dot_product_attention(
    query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None
) -> Tensor:
    """Computes Scaled Dot-Product Attention.

    Args:
        query (Tensor): Query tensor; shape (batch_size, num_heads, seq_len_q, d_k).
        key (Tensor): Key tensor; shape (batch_size, num_heads, seq_len_k, d_k).
        value (Tensor): Value tensor; shape (batch_size, num_heads, seq_len_v, d_v).
                        Usually seq_len_k == seq_len_v.
        mask (Optional[Tensor]): Mask tensor; shape must be broadcastable to
                                (batch_size, num_heads, seq_len_q, seq_len_k).
                                Values should be False (0) for tokens to attend to and True (1) for masked tokens.

    Returns:
        Tensor: Output tensor after attention; shape (batch_size, num_heads, seq_len_q, d_v).
    """
    d_k = query.size(-1)
    # scores shape: (batch_size, num_heads, seq_len_q, seq_len_k)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        # Mask shape needs to be broadcastable to scores shape
        # Where mask is True, fill with -inf
        scores = scores.masked_fill(mask, float("-inf"))

    # attn_weights shape: (batch_size, num_heads, seq_len_q, seq_len_k)
    attn_weights = torch.softmax(scores, dim=-1)

    # output shape: (batch_size, num_heads, seq_len_q, d_v)
    output = torch.matmul(attn_weights, value)
    return output


class MultiHeadAttention(nn.Module):
    """Implements Multi-Head Attention.

    Projects queries, keys, and values multiple times ('heads'), applies
    scaled dot-product attention independently on each head, and then
    concatenates the results, followed by a final linear layer.
    """

    def __init__(self, d_model: int, h: int, dropout: float = 0.1):
        """
        Args:
            d_model (int): Total dimension of the model.
            h (int): Number of attention heads.
            dropout (float): Dropout probability. Not used in the original paper's
                             attention mechanism itself, but kept for potential extension.
                             Currently unused in the forward pass.
        """
        super().__init__()
        if d_model % h != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by h ({h})")

        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h  # Dimension of keys/queries per head
        self.d_v = d_model // h  # Dimension of values per head

        # Linear layers for Q, K, V projections (input)
        # Bias is set to False as per the original implementation
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)

        # Final linear layer (output)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        # Store dropout value, though not currently applied in fwd
        self.dropout = dropout

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None
    ) -> Tensor:
        """Performs the multi-head attention forward pass.

        Args:
            query (Tensor): Query tensor; shape (batch_size, seq_len_q, d_model).
            key (Tensor): Key tensor; shape (batch_size, seq_len_k, d_model).
            value (Tensor): Value tensor; shape (batch_size, seq_len_v, d_model).
                            Usually seq_len_k == seq_len_v.
            mask (Optional[Tensor]): Mask tensor. The mask should identify positions to be ignored.
                                    In scaled_dot_product_attention, positions where the mask
                                    is True will be filled with -inf.
                                    Expected shape is broadcastable to (batch_size, h, seq_len_q, seq_len_k).

        Returns:
            Tensor: Output tensor after multi-head attention; shape (batch_size, seq_len_q, d_model).
        """
        batch_size = query.size(0)

        # 1. Linear projections: (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model)
        q = self.w_q(query)
        k = self.w_k(key)
        v = self.w_v(value)

        # 2. Reshape for multi-head processing
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, h, d_k/d_v)
        # -> (batch_size, h, seq_len, d_k/d_v)
        q = q.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        k = k.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        v = v.view(batch_size, -1, self.h, self.d_v).transpose(1, 2)
        # q shape: (batch_size, h, seq_len_q, d_k)
        # k shape: (batch_size, h, seq_len_k, d_k)
        # v shape: (batch_size, h, seq_len_v, d_v)

        # 3. Apply scaled dot-product attention
        # attn_output shape: (batch_size, h, seq_len_q, d_v)
        # The mask is passed directly; scaled_dot_product_attention handles broadcasting
        attn_output = scaled_dot_product_attention(q, k, v, mask=mask)

        # 4. Concatenate heads and reshape back
        # (batch_size, h, seq_len_q, d_v) -> (batch_size, seq_len_q, h, d_v)
        # -> (batch_size, seq_len_q, d_model)
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        )

        # 5. Final linear layer
        # output shape: (batch_size, seq_len_q, d_model)
        output = self.w_o(attn_output)

        return output
