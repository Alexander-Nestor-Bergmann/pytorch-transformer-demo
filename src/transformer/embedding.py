import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Injects positional information into the input embeddings.

    By position, we literally mean the position of the token in the sequence
    (e.g. the first token, the second token, etc.)

    The positional encodings have the same dimension as the embeddings
    so that the two can be summed. Here, we use sine and cosine
    functions of different frequencies.
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Args:
            d_model (int): The embedding dimension.
            max_len (int): The maximum sequence length.
            dropout (float): The dropout probability.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Adds positional encoding to the input tensor.

        Args:
            x (torch.Tensor): The input tensor. Expected shape: (seq_len, batch_size, embedding_dim).

        Returns:
            torch.Tensor: The output tensor with positional encoding added.
        """
        # x shape: [seq_len, batch_size, embedding_dim]
        # self.pe[: x.size(0)] shape: [seq_len, 1, embedding_dim]
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class TokenEmbedding(nn.Module):
    """Converts token indices to embeddings and scales them.

    Multiplies the embedding by sqrt(d_model) as per the paper.
    """

    def __init__(self, vocab_size: int, d_model: int):
        """
        Args:
            vocab_size (int): The size of the vocabulary.
            d_model (int): The embedding dimension.
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Performs the embedding lookup and scaling.

        Args:
            tokens (torch.Tensor): Input tensor of token indices.

        Returns:
            torch.Tensor: The resulting embeddings, scaled.
        """
        return self.embedding(tokens.long()) * math.sqrt(self.d_model)


class TransformerEmbedding(nn.Module):
    """Combines token embedding and positional encoding.

    Applies token embedding, scales the embeddings, adds positional encoding,
    and applies dropout.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_len: int = 5000,
        dropout: float = 0.1,
    ):
        """
        Args:
            vocab_size (int): The size of the vocabulary.
            d_model (int): The embedding dimension.
            max_len (int): The maximum sequence length for positional encoding.
            dropout (float): The dropout probability for positional encoding.
        """
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Forward pass for the Transformer Embedding layer.

        Args:
            tokens (torch.Tensor): Input tensor of token indices. Expected shape: (seq_len, batch_size).

        Returns:
            torch.Tensor: Output tensor after embedding and positional encoding. Shape: (seq_len, batch_size, embedding_dim).
        """
        # Get token embeddings and scale
        x = self.token_embedding(tokens)
        # Add positional encoding and dropout
        x = self.positional_encoding(x)
        return x
