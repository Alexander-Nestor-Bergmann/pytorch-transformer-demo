import copy

import torch
import torch.nn as nn

from .attention import MultiHeadAttention
from .layers import PositionwiseFeedForward


class EncoderLayer(nn.Module):
    """Implements a single layer of the Transformer Encoder.

    Each layer consists of two sub-layers:
    1. Multi-Head Self-Attention mechanism.
    2. Position-wise Fully Connected Feed-Forward Network.

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
        # Note: Accessing layer.d_model in Encoder init assumes layer is passed
        # and might be slightly less clean than passing d_model directly.
        # However, it simplifies EncoderLayer initialization if creating Encoder first.
        self.d_model = d_model
        self.self_attn = MultiHeadAttention(d_model, h, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout=dropout)
        # LayerNorm layers applied *after* the additions
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, src: torch.Tensor, src_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Forward pass for the Encoder layer.

        Args:
            src (torch.Tensor): Input tensor (batch_size, seq_len, d_model).
            src_mask (torch.Tensor | None): Mask for the source sequence.

        Returns:
            torch.Tensor: Output tensor (batch_size, seq_len, d_model).
        """
        # 1. Multi-Head Self-Attention + Add & Norm
        # The attention mechanism needs query, key, value. In self-attention,
        # they are all the same: the input 'src'.
        attn_output = self.self_attn(src, src, src, mask=src_mask)
        # Residual connection: Add the input 'src' to the attention output.
        # Dropout is applied to the output of the sublayer *before* adding.
        # Then, Layer Normalisation is applied.
        src = self.norm1(src + self.dropout(attn_output))

        # 2. Position-wise Feed-Forward + Add & Norm
        ff_output = self.feed_forward(src)
        # Residual connection: Add the input 'src' (from previous step)
        # to the feed-forward output. Dropout applied before adding.
        # Then, Layer Normalisation is applied.
        src = self.norm2(src + self.dropout(ff_output))

        return src


class Encoder(nn.Module):
    """Implements the full Transformer Encoder stack.

    Consists of N identical EncoderLayers stacked on top of each other.
    """

    def __init__(self, layer: EncoderLayer, num_layers: int):
        """
        Args:
            layer (EncoderLayer): An instance of the EncoderLayer.
                                 All layers in the stack will be copies of this.
            num_layers (int): The number of layers in the encoder stack.
        """
        super().__init__()
        # Create N identical layers using deep copies
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])
        # Final LayerNorm applied after the stack
        self.norm = nn.LayerNorm(layer.d_model)  # Access d_model from the passed layer

    def forward(
        self, src: torch.Tensor, src_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Forward pass for the entire Encoder stack.

        Args:
            src (torch.Tensor): Input tensor (batch_size, seq_len, d_model).
                                Should be the sum of token embeddings and
                                positional encodings.
            src_mask (torch.Tensor | None): Mask for the source sequence.

        Returns:
            torch.Tensor: Output tensor from the final layer
                          (batch_size, seq_len, d_model).
        """
        output = src
        # Pass the input through each layer in the stack
        for layer in self.layers:
            output = layer(output, src_mask)

        # Apply final Layer Normalisation
        return self.norm(output)
