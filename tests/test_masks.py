import torch
import pytest

from src.transformer.masks import create_padding_mask, create_subsequent_mask

# Test parameters
PAD_IDX = 0
BATCH_SIZE = 2
SEQ_LEN = 5


@pytest.fixture
def sample_sequences():
    """Provides sample sequences with padding."""
    # Sequences: [[1, 2, 3, 0, 0], [4, 5, 0, 0, 0]]
    seq = torch.tensor(
        [
            [1, 2, 3, PAD_IDX, PAD_IDX],
            [4, 5, PAD_IDX, PAD_IDX, PAD_IDX],
        ],
        dtype=torch.long,
    )
    assert seq.shape == (BATCH_SIZE, SEQ_LEN)
    return seq


def test_create_padding_mask(sample_sequences):
    """Test the shape and content of the padding mask."""
    mask = create_padding_mask(sample_sequences, PAD_IDX)

    # Expected shape: (batch_size, 1, 1, seq_len)
    expected_shape = (BATCH_SIZE, 1, 1, SEQ_LEN)
    assert (
        mask.shape == expected_shape
    ), f"Expected shape {expected_shape}, got {mask.shape}"

    # Expected mask content (True where padded)
    expected_mask = torch.tensor(
        [
            [[[False, False, False, True, True]]],  # Mask for seq 1
            [[[False, False, True, True, True]]],  # Mask for seq 2
        ],
        dtype=torch.bool,
    )
    assert torch.equal(
        mask, expected_mask
    ), f"Expected mask {expected_mask}, got {mask}"


def test_create_subsequent_mask():
    """Test the shape and content of the subsequent mask."""
    size = 4
    device = torch.device("cpu")
    mask = create_subsequent_mask(size, device=device)

    # Expected shape: (1, 1, size, size)
    expected_shape = (1, 1, size, size)
    assert (
        mask.shape == expected_shape
    ), f"Expected shape {expected_shape}, got {mask.shape}"

    # Expected mask content (True for upper triangle, excluding diagonal)
    expected_mask = torch.tensor(
        [
            [
                [
                    [False, True, True, True],
                    [False, False, True, True],
                    [False, False, False, True],
                    [False, False, False, False],
                ]
            ]
        ],
        dtype=torch.bool,
    )
    assert torch.equal(
        mask, expected_mask
    ), f"Expected mask {expected_mask}, got {mask}"

    # Test on a different device if CUDA is available
    if torch.cuda.is_available():
        device_cuda = torch.device("cuda")
        mask_cuda = create_subsequent_mask(size, device=device_cuda)
        assert mask_cuda.device.type == "cuda"
        assert torch.equal(mask_cuda.cpu(), expected_mask)
