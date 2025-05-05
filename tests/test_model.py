import torch
import pytest

from src.transformer.model import Transformer

# Test parameters
BATCH_SIZE = 4
SRC_SEQ_LEN = 12
TGT_SEQ_LEN = 10
SRC_VOCAB_SIZE = 1000
TGT_VOCAB_SIZE = 1200
D_MODEL = 512
NUM_ENC_LAYERS = 3
NUM_DEC_LAYERS = 3
H = 8
D_FF = 2048
DROPOUT = 0.1
MAX_LEN = 50
PAD_IDX = 0


@pytest.fixture
def transformer_model():
    """Fixture to create a Transformer model instance."""
    return Transformer(
        src_vocab_size=SRC_VOCAB_SIZE,
        tgt_vocab_size=TGT_VOCAB_SIZE,
        d_model=D_MODEL,
        num_encoder_layers=NUM_ENC_LAYERS,
        num_decoder_layers=NUM_DEC_LAYERS,
        h=H,
        d_ff=D_FF,
        dropout=DROPOUT,
        max_len=MAX_LEN,
    )


@pytest.fixture
def src_indices():
    """Fixture to create sample source sequence indices."""
    # Simple sequences without padding for shape testing
    seq = torch.randint(1, SRC_VOCAB_SIZE, (BATCH_SIZE, SRC_SEQ_LEN))
    return seq


@pytest.fixture
def tgt_indices():
    """Fixture to create sample target sequence indices."""
    # Simple sequences without padding for shape testing
    seq = torch.randint(1, TGT_VOCAB_SIZE, (BATCH_SIZE, TGT_SEQ_LEN))
    return seq


def test_transformer_forward_output_shape(transformer_model, src_indices, tgt_indices):
    """Test the output shape of the Transformer's forward pass."""
    transformer_model.eval()  # Disable dropout
    with torch.no_grad():
        output = transformer_model(
            src_indices, tgt_indices, src_pad_idx=PAD_IDX, tgt_pad_idx=PAD_IDX
        )

    expected_shape = (BATCH_SIZE, TGT_SEQ_LEN, TGT_VOCAB_SIZE)
    assert (
        output.shape == expected_shape
    ), f"Expected output shape {expected_shape}, but got {output.shape}"
