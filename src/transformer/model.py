"""Main Transformer Model Implementation"""

import math
import copy
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from .embedding import PositionalEncoding, TokenEmbedding
from .encoder import Encoder, EncoderLayer
from .decoder import Decoder, DecoderLayer
from .attention import MultiHeadAttention
from .layers import PositionwiseFeedForward
from .masks import create_padding_mask, create_subsequent_mask


class Transformer(nn.Module):
    """A standard Transformer model.

    Implementation based on the paper "Attention Is All You Need".
    Connects the Encoder and Decoder stacks, integrates embeddings,
    positional encoding, and the final linear output layer.
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        h: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_len: int = 5000,
    ):
        """Initialises the Transformer model.

        Args:
            src_vocab_size (int): Size of the source vocabulary.
            tgt_vocab_size (int): Size of the target vocabulary.
            d_model (int): The dimensionality of the embeddings and model.
            num_encoder_layers (int): Number of Encoder layers.
            num_decoder_layers (int): Number of Decoder layers.
            h (int): Number of attention heads.
            d_ff (int): The dimension of the feed-forward network model.
            dropout (float): The dropout value.
            max_len (int): Maximum sequence length for positional encoding.
        """
        super().__init__()

        self.d_model = d_model

        # Embeddings and Positional Encoding
        self.src_tok_emb = TokenEmbedding(src_vocab_size, d_model)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)

        # Shared Layer Components (we create templates and copy them)
        attn = MultiHeadAttention(d_model, h, dropout)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        encoder_layer = EncoderLayer(
            d_model, copy.deepcopy(attn), copy.deepcopy(ff), dropout
        )
        decoder_layer = DecoderLayer(
            d_model,
            h,
            d_ff,
            dropout,
        )

        # Encoder and Decoder Stacks
        self.encoder = Encoder(encoder_layer, num_encoder_layers)
        self.decoder = Decoder(decoder_layer, num_decoder_layers)

        # Final Linear Layer
        self.generator = nn.Linear(d_model, tgt_vocab_size)

        # Initialise parameters
        self._reset_parameters()

    def _reset_parameters(self):
        """Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        src_pad_idx: int = 0,
        tgt_pad_idx: int = 0,
    ) -> Tensor:
        """Performs the forward pass of the Transformer model.

        Args:
            src (Tensor): The source sequence tensor. Shape: [batch_size, src_seq_len].
            tgt (Tensor): The target sequence tensor. Shape: [batch_size, tgt_seq_len].
            src_pad_idx (int): Index used for padding in the source sequence.
            tgt_pad_idx (int): Index used for padding in the target sequence.

        Returns:
            Tensor: The output logits from the final linear layer.
                    Shape: [batch_size, tgt_seq_len, tgt_vocab_size].
        """
        # Determine device from input tensors
        device = src.device

        # Generate masks
        src_mask = create_padding_mask(
            src, src_pad_idx
        )  # Shape: [batch_size, 1, 1, src_seq_len]

        # Create masks on the correct device
        tgt_subsequent_mask = create_subsequent_mask(tgt.size(-1)).to(device)
        tgt_padding_mask = create_padding_mask(
            tgt, tgt_pad_idx
        )  # Shape: [batch_size, 1, 1, tgt_seq_len]

        # Combine masks: True where either mask is True
        # Ensure tgt_padding_mask is broadcastable to tgt_subsequent_mask shape if needed
        # tgt_subsequent_mask shape: [1, 1, tgt_seq_len, tgt_seq_len] or similar
        # tgt_padding_mask shape: [batch_size, 1, 1, tgt_seq_len]
        # Broadcasting should handle this if dimensions match appropriately for the operation.
        # Need to be careful here if create_padding_mask doesn't return the expected shape for broadcasting.
        # Assuming create_padding_mask returns [batch_size, 1, 1, tgt_seq_len] and subsequent mask [1, 1, tgt_seq_len, tgt_seq_len]
        # Let's ensure padding mask is expanded for correct OR operation
        if tgt_padding_mask is not None:
            tgt_padding_mask_expanded = tgt_padding_mask.expand(
                -1, -1, tgt.size(-1), -1
            )
            tgt_mask = tgt_padding_mask_expanded | tgt_subsequent_mask
        else:
            tgt_mask = tgt_subsequent_mask

        memory_mask = src_mask  # Shape: [batch_size, 1, 1, src_seq_len]

        # Process source sequence
        src_emb = self.positional_encoding(self.src_tok_emb(src).transpose(0, 1))
        memory = self.encoder(src_emb, src_mask)

        # Process target sequence
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt).transpose(0, 1))
        decoder_output = self.decoder(tgt_emb, memory, tgt_mask, memory_mask)

        # Generate output logits
        logits = self.generator(decoder_output)
        return logits.transpose(0, 1)

    def encode(self, src: Tensor, src_mask: Optional[Tensor] = None) -> Tensor:
        """Encodes the source sequence.

        Used for inference where encoding happens once.

        Args:
            src (Tensor): The source sequence tensor. Shape: [batch_size, src_seq_len].
            src_mask (Optional[Tensor]): The source padding mask.
                                           Shape: [batch_size, 1, 1, src_seq_len].

        Returns:
            Tensor: The encoded memory. Shape: [src_seq_len, batch_size, d_model].
        """
        src_emb = self.positional_encoding(self.src_tok_emb(src).transpose(0, 1))
        return self.encoder(src_emb, src_mask)

    def decode(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Decodes the target sequence given the memory.

        Used for step-by-step inference.

        Args:
            tgt (Tensor): The target sequence tensor generated so far.
                          Shape: [batch_size, current_tgt_len].
            memory (Tensor): The encoded source sequence (output of encoder).
                             Shape: [src_seq_len, batch_size, d_model].
            tgt_mask (Optional[Tensor]): The target sequence mask (look-ahead + padding).
                                           Shape: [batch_size, 1, current_tgt_len, current_tgt_len].
            memory_mask (Optional[Tensor]): The source padding mask.
                                            Shape: [batch_size, 1, 1, src_seq_len].

        Returns:
            Tensor: The output logits from the final linear layer for the next token.
                    Shape: [batch_size, current_tgt_len, tgt_vocab_size].
        """
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt).transpose(0, 1))
        decoder_output = self.decoder(tgt_emb, memory, tgt_mask, memory_mask)
        logits = self.generator(decoder_output)
        return logits.transpose(0, 1)


# Example usage (conceptual)
# src_vocab = 10000
# tgt_vocab = 12000
# model = Transformer(src_vocab_size=src_vocab, tgt_vocab_size=tgt_vocab)
# src_data = torch.randint(1, src_vocab, (64, 32)) # [batch_size, seq_len]
# tgt_data = torch.randint(1, tgt_vocab, (64, 40)) # [batch_size, seq_len]
# output = model(src_data, tgt_data[:, :-1]) # Exclude last token for target input
# print(output.shape) # Should be [64, 39, tgt_vocab]
