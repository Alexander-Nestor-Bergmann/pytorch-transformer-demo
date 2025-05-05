import torch
import torch.nn as nn
from torch import Tensor

# Note: LayerNorm (nn.LayerNorm) is used directly within Encoder/Decoder layers.


class PositionwiseFeedForward(nn.Module):
    """Implements the Position-wise Feed-Forward Network (FFN).

    This network is applied to each position separately and identically.
    It consists of two linear transformations with a ReLU activation in between.
    FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """
        Args:
            d_model (int): The dimension of the input and output (usually 512).
            d_ff (int): The inner dimension of the feed-forward network (usually 2048).
            dropout (float): Dropout probability, applied after the first linear layer and ReLU.
        """
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for the FFN.

        Args:
            x (Tensor): Input tensor; shape (batch_size, seq_len, d_model).

        Returns:
            Tensor: Output tensor; shape (batch_size, seq_len, d_model).
        """
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        # (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)
        x = self.linear2(x)
        return x
