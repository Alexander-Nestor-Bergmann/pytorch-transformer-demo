import torch
import torch.nn as nn


def create_padding_mask(seq: torch.Tensor, pad_idx: int) -> torch.Tensor:
    """Creates a mask to hide padding tokens.

    Args:
        seq (torch.Tensor): Input sequence tensor; shape (batch_size, seq_len).
        pad_idx (int): The index representing padding in the vocabulary.

    Returns:
        torch.Tensor: Padding mask tensor; shape (batch_size, 1, 1, seq_len).
                      Returns `True` for positions that should be masked (padding),
                      and `False` otherwise.
    """
    # seq != pad_idx returns True for non-padding tokens, False for padding tokens.
    # We want the mask to be True for padding tokens, so we invert the condition.
    # Original shape: (batch_size, seq_len)
    mask = seq == pad_idx
    # Add dimensions for multi-head attention compatibility (batch_size, n_heads, seq_len_q, seq_len_k)
    # The mask should apply to the keys dimension (seq_len_k)
    return mask.unsqueeze(1).unsqueeze(2)  # shape: (batch_size, 1, 1, seq_len)


def create_subsequent_mask(
    size: int, device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    """Creates a mask to prevent attention to subsequent positions.

    Used in the decoder to prevent looking ahead at future tokens.

    Args:
        size (int): The length of the sequence (seq_len).
        device (torch.device): The device to create the tensor on.

    Returns:
        torch.Tensor: Subsequent mask tensor; shape (1, size, size).
                      Returns `True` for positions that should be masked (upper triangle),
                      and `False` otherwise (lower triangle including diagonal).
    """
    # Create an upper triangular matrix of ones
    mask = torch.triu(torch.ones(size, size, device=device), diagonal=1)
    # Convert to boolean: True where it's 1 (upper triangle), False where it's 0
    # Unsqueeze twice to get shape (1, 1, size, size) for multi-head compatibility
    return mask.bool().unsqueeze(0).unsqueeze(1)


# Example Usage (for understanding):
if __name__ == "__main__":
    # Example Padding Mask
    pad_token_index = 0
    # Batch of 2 sequences, max length 5
    sequences = torch.tensor(
        [
            [1, 2, 3, 0, 0],  # Sequence 1 with padding
            [4, 5, 0, 0, 0],  # Sequence 2 with padding
        ]
    )
    padding_mask = create_padding_mask(sequences, pad_token_index)
    print("Example Sequences:\n", sequences)
    print("Padding Mask (True means mask):\n", padding_mask)
    # Expected: tensor([[[[False, False, False,  True,  True]]], [[[False, False,  True,  True,  True]]]])

    # Example Subsequent Mask
    seq_length = 4
    subsequent_mask = create_subsequent_mask(seq_length)
    print(
        f"\nSubsequent Mask for size {seq_length} (True means mask):\n", subsequent_mask
    )
    # Expected shape (1, 1, 4, 4)
    # Expected:
    # tensor([[[[False,  True,  True,  True],
    #           [False, False,  True,  True],
    #           [False, False, False,  True],
    #           [False, False, False, False]]]])

    # Combining masks (conceptual, actual combination depends on context)
    # Typically masks are combined using logical OR (|)
    # The shapes need broadcasting compatibility. Padding mask is (B, 1, 1, Sk) and subsequent mask is (1, 1, T, T)
    # For decoder self-attention, T=Sk=target_seq_len
    target_seq_len = sequences.shape[1]
    subsequent_mask_for_target = create_subsequent_mask(target_seq_len)  # (1, 1, T, T)
    # Note: Padding mask needs to be compatible with target sequence length for decoder self-attention.
    # If using padding mask on the target sequence itself:
    target_padding_mask = create_padding_mask(
        sequences, pad_token_index
    )  # Use target sequence here
    combined_mask = (
        target_padding_mask | subsequent_mask_for_target
    )  # Broadcasting works: (B, 1, 1, T) | (1, 1, T, T) -> (B, 1, T, T)

    print(
        f"\nCombined Mask (Target Padding OR Subsequent) for target sequence length {target_seq_len}:\n",
        combined_mask,
    )
