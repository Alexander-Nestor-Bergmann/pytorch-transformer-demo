import torch
import pytest

from src.transformer.layers import PositionwiseFeedForward

# Test parameters
BATCH_SIZE = 4
SEQ_LEN = 10
D_MODEL = 512
D_FF = 2048
DROPOUT = 0.1


@pytest.fixture
def ffn_layer():
    """Fixture to create a PositionwiseFeedForward layer."""
    return PositionwiseFeedForward(d_model=D_MODEL, d_ff=D_FF, dropout=DROPOUT)


@pytest.fixture
def input_tensor():
    """Fixture to create a sample input tensor."""
    return torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL)


def test_ffn_output_shape(ffn_layer, input_tensor):
    """Test if the PositionwiseFeedForward layer returns the correct output shape."""
    # Set the model to evaluation mode to disable dropout
    ffn_layer.eval()
    with torch.no_grad():
        output = ffn_layer(input_tensor)

    assert output.shape == (
        BATCH_SIZE,
        SEQ_LEN,
        D_MODEL,
    ), f"Expected output shape {(BATCH_SIZE, SEQ_LEN, D_MODEL)}, but got {output.shape}"


# Note: LayerNorm tests are implicitly covered by testing EncoderLayer/DecoderLayer
# as nn.LayerNorm is used directly within them.
