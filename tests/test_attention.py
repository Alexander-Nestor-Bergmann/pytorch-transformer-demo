import pytest
import torch
from src.transformer.attention import scaled_dot_product_attention, MultiHeadAttention

# Test parameters
BATCH_SIZE = 2
SEQ_LEN_Q = 5
SEQ_LEN_K = 7
SEQ_LEN_V = 7  # Must be same as SEQ_LEN_K for standard attention
D_MODEL = 16
N_HEADS = 4
D_K = D_MODEL // N_HEADS
D_V = D_MODEL // N_HEADS

# --- Tests for scaled_dot_product_attention ---


def test_scaled_dot_product_attention_output_shape():
    """Tests the output shape of scaled_dot_product_attention."""
    query = torch.randn(BATCH_SIZE, N_HEADS, SEQ_LEN_Q, D_K)
    key = torch.randn(BATCH_SIZE, N_HEADS, SEQ_LEN_K, D_K)
    value = torch.randn(BATCH_SIZE, N_HEADS, SEQ_LEN_V, D_V)

    output = scaled_dot_product_attention(query, key, value)

    assert output.shape == (
        BATCH_SIZE,
        N_HEADS,
        SEQ_LEN_Q,
        D_V,
    ), f"Expected shape {(BATCH_SIZE, N_HEADS, SEQ_LEN_Q, D_V)}, but got {output.shape}"


def test_scaled_dot_product_attention_masking():
    """Tests the masking functionality of scaled_dot_product_attention."""
    query = torch.randn(BATCH_SIZE, N_HEADS, SEQ_LEN_Q, D_K)
    key = torch.randn(BATCH_SIZE, N_HEADS, SEQ_LEN_K, D_K)
    value = torch.randn(BATCH_SIZE, N_HEADS, SEQ_LEN_V, D_V)

    # Mask shape needs to be broadcastable to (BATCH_SIZE, N_HEADS, SEQ_LEN_Q, SEQ_LEN_K)
    mask = torch.zeros(BATCH_SIZE, 1, 1, SEQ_LEN_K, dtype=torch.bool)
    mask[0, :, :, -3:] = True  # Mask last 3 key positions for the first batch item

    # Calculate attention scores manually for comparison
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(
        torch.tensor(d_k, dtype=torch.float32)
    )
    scores_before_masking = scores.clone()
    scores = scores.masked_fill(mask, float("-inf"))

    attn_weights = torch.softmax(scores, dim=-1)

    # Check that masked positions have zero attention weight
    assert torch.allclose(
        attn_weights[0, :, :, -3:], torch.zeros_like(attn_weights[0, :, :, -3:])
    ), "Attention weights for masked positions should be zero."

    # Check that non-masked positions in the first batch item sum to 1
    assert torch.allclose(
        attn_weights[0, :, :, :-3].sum(dim=-1),
        torch.ones_like(attn_weights[0, :, :, :-3].sum(dim=-1)),
    ), "Non-masked attention weights in the first batch item should sum to 1."

    # Check that the second batch item (unmasked) has weights summing to 1 across all key positions
    assert torch.allclose(
        attn_weights[1].sum(dim=-1), torch.ones_like(attn_weights[1].sum(dim=-1))
    ), "Attention weights in the second (unmasked) batch item should sum to 1."

    output = scaled_dot_product_attention(query, key, value, mask=mask)

    assert output.shape == (
        BATCH_SIZE,
        N_HEADS,
        SEQ_LEN_Q,
        D_V,
    ), f"Expected shape {(BATCH_SIZE, N_HEADS, SEQ_LEN_Q, D_V)} with mask, but got {output.shape}"


# --- Tests for MultiHeadAttention ---

# TODO: Add tests for MultiHeadAttention
