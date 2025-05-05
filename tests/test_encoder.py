import torch
import pytest

from src.transformer.encoder import EncoderLayer, Encoder
from src.transformer.masks import create_padding_mask

# Test parameters
BATCH_SIZE = 4
SEQ_LEN = 12  # Sequence length
D_MODEL = 512  # Model dimension
H = 8  # Number of heads
D_FF = 2048  # Feed-forward dimension
DROPOUT = 0.1
NUM_LAYERS = 6  # Number of encoder layers
PAD_IDX = 0  # Padding index


@pytest.fixture
def encoder_layer():
    """Fixture to create an EncoderLayer."""
    return EncoderLayer(d_model=D_MODEL, h=H, d_ff=D_FF, dropout=DROPOUT)


@pytest.fixture
def encoder(encoder_layer):
    """Fixture to create an Encoder stack."""
    return Encoder(layer=encoder_layer, num_layers=NUM_LAYERS)


@pytest.fixture
def input_tensor():
    """Fixture to create a sample input tensor (embeddings + positional encodings)."""
    # Simulate some padding
    tensor = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL)
    return tensor


@pytest.fixture
def input_sequence_indices():
    """Fixture to create sample input sequence indices with padding."""
    # Create sequences with varying lengths and padding
    seq = torch.randint(1, 100, (BATCH_SIZE, SEQ_LEN))  # Values between 1 and 99
    # Add padding (PAD_IDX=0) to some sequences
    for i in range(BATCH_SIZE):
        pad_len = torch.randint(
            0, SEQ_LEN // 2, (1,)
        ).item()  # Pad up to half the length
        if pad_len > 0:
            seq[i, -pad_len:] = PAD_IDX
    return seq


@pytest.fixture
def src_mask(input_sequence_indices):
    """Fixture to create a source padding mask."""
    return create_padding_mask(input_sequence_indices, PAD_IDX)


def test_encoder_layer_output_shape(encoder_layer, input_tensor, src_mask):
    """Test if the EncoderLayer returns the correct output shape."""
    encoder_layer.eval()  # Disable dropout
    with torch.no_grad():
        output = encoder_layer(input_tensor, src_mask)

    assert output.shape == (
        BATCH_SIZE,
        SEQ_LEN,
        D_MODEL,
    ), f"Expected output shape {(BATCH_SIZE, SEQ_LEN, D_MODEL)}, but got {output.shape}"


def test_encoder_output_shape(encoder, input_tensor, src_mask):
    """Test if the full Encoder stack returns the correct output shape."""
    encoder.eval()  # Disable dropout
    with torch.no_grad():
        output = encoder(input_tensor, src_mask)

    assert output.shape == (
        BATCH_SIZE,
        SEQ_LEN,
        D_MODEL,
    ), f"Expected output shape {(BATCH_SIZE, SEQ_LEN, D_MODEL)}, but got {output.shape}"
