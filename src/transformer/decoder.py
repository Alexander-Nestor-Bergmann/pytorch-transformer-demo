import copy

import torch
import torch.nn as nn

from .attention import MultiHeadAttention
from .layers import PositionwiseFeedForward


class DecoderLayer(nn.Module):
    """Implements a single layer of the Transformer Decoder.

    Each layer consists of three sub-layers:
    1. Masked Multi-Head Self-Attention mechanism (looks only at previous positions).
    2. Multi-Head Cross-Attention mechanism (attends to the encoder output).
    3. Position-wise Fully Connected Feed-Forward Network.

    Residual connections and Layer Normalisation are applied after each sub-layer.
    """

    def __init__(self, d_model: int, h: int, d_ff: int, dropout: float = 0.1):
        """
        Args:
            d_model (int): The dimension of the model (embeddings and layers).
            h (int): The number of attention heads.
            d_ff (int): The dimension of the inner layer of the FFN.
            dropout (float): The dropout probability.
        """
        super().__init__()
        self.d_model = d_model  # Store d_model for Decoder's final LayerNorm
        self.self_attn = MultiHeadAttention(d_model, h, dropout=dropout)
        self.cross_attn = MultiHeadAttention(d_model, h, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout=dropout)
        # LayerNorm layers applied *after* the additions
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass for the Decoder layer.

        Args:
            tgt (torch.Tensor): Input tensor for the target sequence
                               (batch_size, tgt_seq_len, d_model).
            memory (torch.Tensor): Output tensor from the encoder stack
                                  (batch_size, src_seq_len, d_model).
            tgt_mask (torch.Tensor | None): Mask for the target sequence
                                          (handles look-ahead and padding).
            memory_mask (torch.Tensor | None): Mask for the source sequence
                                             (handles encoder padding).

        Returns:
            torch.Tensor: Output tensor (batch_size, tgt_seq_len, d_model).
        """
        # 1. Masked Multi-Head Self-Attention + Add & Norm
        # Query, Key, Value are all the target input 'tgt'
        # The 'tgt_mask' ensures causality (no peeking ahead)
        attn_output = self.self_attn(tgt, tgt, tgt, mask=tgt_mask)
        tgt = self.norm1(tgt + self.dropout(attn_output))

        # 2. Multi-Head Cross-Attention + Add & Norm
        # Query comes from the decoder's current state ('tgt').
        # Key and Value come from the encoder's output ('memory').
        # 'memory_mask' masks padding in the encoder output.
        attn_output = self.cross_attn(tgt, memory, memory, mask=memory_mask)
        tgt = self.norm2(tgt + self.dropout(attn_output))

        # 3. Position-wise Feed-Forward + Add & Norm
        ff_output = self.feed_forward(tgt)
        tgt = self.norm3(tgt + self.dropout(ff_output))

        return tgt


class Decoder(nn.Module):
    """Implements the full Transformer Decoder stack.

    Consists of N identical DecoderLayers stacked on top of each other.
    """

    def __init__(self, layer: DecoderLayer, num_layers: int):
        """
        Args:
            layer (DecoderLayer): An instance of the DecoderLayer.
                                 All layers in the stack will be copies of this.
            num_layers (int): The number of layers in the decoder stack.
        """
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])
        # Final LayerNorm applied after the stack
        self.norm = nn.LayerNorm(layer.d_model)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass for the entire Decoder stack.

        Args:
            tgt (torch.Tensor): Input tensor for the target sequence
                               (batch_size, tgt_seq_len, d_model).
                               Should be the sum of token embeddings and
                               positional encodings for the target.
            memory (torch.Tensor): Output tensor from the encoder stack
                                  (batch_size, src_seq_len, d_model).
            tgt_mask (torch.Tensor | None): Mask for the target sequence.
            memory_mask (torch.Tensor | None): Mask for the source sequence.

        Returns:
            torch.Tensor: Output tensor from the final layer
                          (batch_size, tgt_seq_len, d_model).
        """
        output = tgt
        # Pass the input through each layer in the stack
        for layer in self.layers:
            output = layer(output, memory, tgt_mask, memory_mask)

        # Apply final Layer Normalisation
        return self.norm(output)
