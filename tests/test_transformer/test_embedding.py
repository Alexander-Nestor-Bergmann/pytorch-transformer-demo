import math
import pytest
import torch

from src.transformer.embedding import (
    PositionalEncoding,
    TokenEmbedding,
    TransformerEmbedding,
)

# Constants for testing
VOCAB_SIZE = 1000
D_MODEL = 512
SEQ_LEN = 20
BATCH_SIZE = 32
MAX_LEN = 100
DROPOUT = 0.1


def test_token_embedding_output_shape():
    """Tests if TokenEmbedding returns the correct output shape and scaling."""
    embedding = TokenEmbedding(vocab_size=VOCAB_SIZE, d_model=D_MODEL)
    # Assume input tokens have shape (seq_len, batch_size)
    tokens = torch.randint(0, VOCAB_SIZE, (SEQ_LEN, BATCH_SIZE))
    output = embedding(tokens)

    assert output.shape == (
        SEQ_LEN,
        BATCH_SIZE,
        D_MODEL,
    ), f"Expected shape {(SEQ_LEN, BATCH_SIZE, D_MODEL)}, but got {output.shape}"

    # Check scaling factor (magnitude)
    # Get the norm of the embedding for the first token
    # Compare it to the norm of the raw embedding vector scaled
    raw_embedding_norm = torch.norm(embedding.embedding.weight[tokens[0, 0]], p=2)
    expected_norm = raw_embedding_norm * math.sqrt(D_MODEL)
    output_norm = torch.norm(output[0, 0], p=2)
    assert torch.allclose(
        output_norm, expected_norm
    ), f"Expected norm ~{expected_norm}, but got {output_norm}"


def test_positional_encoding_output_shape_and_values():
    """Tests if PositionalEncoding returns the correct shape and adds values."""
    pos_encoding = PositionalEncoding(d_model=D_MODEL, max_len=MAX_LEN, dropout=0.0)
    # Input tensor shape: (seq_len, batch_size, d_model)
    input_tensor = torch.zeros(SEQ_LEN, BATCH_SIZE, D_MODEL)
    output = pos_encoding(input_tensor)

    assert output.shape == (
        SEQ_LEN,
        BATCH_SIZE,
        D_MODEL,
    ), f"Expected shape {(SEQ_LEN, BATCH_SIZE, D_MODEL)}, but got {output.shape}"

    # Check if positional encoding was added (output should not be all zeros)
    assert not torch.all(output == 0.0), "Positional encoding was not added."

    # Check if dropout is applied when p > 0
    pos_encoding_dropout = PositionalEncoding(
        d_model=D_MODEL, max_len=MAX_LEN, dropout=0.9  # High dropout
    )
    output_dropout = pos_encoding_dropout(input_tensor)
    # With high dropout, some values should likely be zeroed out
    # Note: This is a probabilistic check, might fail rarely
    assert torch.sum(output_dropout == 0.0) > 0, "Dropout did not seem to be applied."


def test_positional_encoding_uniqueness():
    """Tests if positional encodings are unique across sequence positions."""
    pos_encoding = PositionalEncoding(d_model=D_MODEL, max_len=MAX_LEN, dropout=0.0)
    # Get PE values directly (requires access, or use forward on zeros)
    pe_values = pos_encoding.pe  # Shape: [max_len, 1, d_model]
    assert pe_values.shape[0] == MAX_LEN
    assert pe_values.shape[2] == D_MODEL

    # Check that PE for position 0 is different from PE for position 1
    assert not torch.allclose(
        pe_values[0], pe_values[1]
    ), "Positional encodings for position 0 and 1 are the same."
    # Check that PE for position 1 is different from PE for position 2
    assert not torch.allclose(
        pe_values[1], pe_values[2]
    ), "Positional encodings for position 1 and 2 are the same."


def test_transformer_embedding_output_shape():
    """Tests if the combined TransformerEmbedding returns the correct shape."""
    transformer_embedding = TransformerEmbedding(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        max_len=MAX_LEN,
        dropout=DROPOUT,
    )
    # Input tokens shape: (seq_len, batch_size)
    tokens = torch.randint(0, VOCAB_SIZE, (SEQ_LEN, BATCH_SIZE))
    output = transformer_embedding(tokens)

    assert output.shape == (
        SEQ_LEN,
        BATCH_SIZE,
        D_MODEL,
    ), f"Expected shape {(SEQ_LEN, BATCH_SIZE, D_MODEL)}, but got {output.shape}"
