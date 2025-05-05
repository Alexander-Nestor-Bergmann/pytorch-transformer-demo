import torch
import pytest

from src.transformer.decoder import DecoderLayer, Decoder
from src.transformer.masks import create_padding_mask, create_subsequent_mask

# Test parameters
BATCH_SIZE = 4
SRC_SEQ_LEN = 12  # Source sequence length (from encoder)
TGT_SEQ_LEN = 10  # Target sequence length
D_MODEL = 512  # Model dimension
H = 8  # Number of heads
D_FF = 2048  # Feed-forward dimension
DROPOUT = 0.1
NUM_LAYERS = 6  # Number of decoder layers
PAD_IDX = 0  # Padding index


@pytest.fixture
def decoder_layer():
    """Fixture to create a DecoderLayer."""
    return DecoderLayer(d_model=D_MODEL, h=H, d_ff=D_FF, dropout=DROPOUT)


@pytest.fixture
def decoder(decoder_layer):
    """Fixture to create a Decoder stack."""
    return Decoder(layer=decoder_layer, num_layers=NUM_LAYERS)


@pytest.fixture
def tgt_tensor():
    """Fixture to create a sample target input tensor."""
    return torch.randn(BATCH_SIZE, TGT_SEQ_LEN, D_MODEL)


@pytest.fixture
def memory_tensor():
    """Fixture to create a sample encoder output tensor (memory)."""
    return torch.randn(BATCH_SIZE, SRC_SEQ_LEN, D_MODEL)


@pytest.fixture
def tgt_sequence_indices():
    """Fixture to create sample target sequence indices with padding."""
    seq = torch.randint(1, 100, (BATCH_SIZE, TGT_SEQ_LEN))
    for i in range(BATCH_SIZE):
        pad_len = torch.randint(0, TGT_SEQ_LEN // 2, (1,)).item()
        if pad_len > 0:
            seq[i, -pad_len:] = PAD_IDX
    return seq


@pytest.fixture
def src_sequence_indices():
    """Fixture to create sample source sequence indices with padding."""
    seq = torch.randint(1, 100, (BATCH_SIZE, SRC_SEQ_LEN))
    for i in range(BATCH_SIZE):
        pad_len = torch.randint(0, SRC_SEQ_LEN // 2, (1,)).item()
        if pad_len > 0:
            seq[i, -pad_len:] = PAD_IDX
    return seq


@pytest.fixture
def tgt_mask(tgt_sequence_indices):
    """Fixture to create a combined target mask (padding + subsequent)."""
    tgt_pad_mask = create_padding_mask(
        tgt_sequence_indices, PAD_IDX
    )  # (B, 1, 1, TGT_LEN)
    tgt_sub_mask = create_subsequent_mask(
        TGT_SEQ_LEN, device=tgt_sequence_indices.device
    )  # (1, 1, TGT_LEN, TGT_LEN)
    # Combine masks: True if padded OR subsequent.
    # Broadcasting handles combining the padding and subsequent masks.
    combined_mask = tgt_pad_mask | tgt_sub_mask
    return combined_mask


@pytest.fixture
def memory_mask(src_sequence_indices):
    """Fixture to create a source padding mask (for memory)."""
    # Shape: (B, 1, 1, SRC_LEN)
    return create_padding_mask(src_sequence_indices, PAD_IDX)


def test_decoder_layer_output_shape(
    decoder_layer, tgt_tensor, memory_tensor, tgt_mask, memory_mask
):
    """Test if the DecoderLayer returns the correct output shape."""
    decoder_layer.eval()  # Disable dropout
    with torch.no_grad():
        output = decoder_layer(tgt_tensor, memory_tensor, tgt_mask, memory_mask)

    assert output.shape == (
        BATCH_SIZE,
        TGT_SEQ_LEN,
        D_MODEL,
    ), f"Expected output shape {(BATCH_SIZE, TGT_SEQ_LEN, D_MODEL)}, but got {output.shape}"


def test_decoder_output_shape(
    decoder, tgt_tensor, memory_tensor, tgt_mask, memory_mask
):
    """Test if the full Decoder stack returns the correct output shape."""
    decoder.eval()  # Disable dropout
    with torch.no_grad():
        output = decoder(tgt_tensor, memory_tensor, tgt_mask, memory_mask)

    assert output.shape == (
        BATCH_SIZE,
        TGT_SEQ_LEN,
        D_MODEL,
    ), f"Expected output shape {(BATCH_SIZE, TGT_SEQ_LEN, D_MODEL)}, but got {output.shape}"
