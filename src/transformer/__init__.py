"""Transformer Package Initialisation"""

# Expose key components at the package level
from .model import Transformer
from .encoder import Encoder, EncoderLayer
from .decoder import Decoder, DecoderLayer
from .attention import MultiHeadAttention, scaled_dot_product_attention
from .layers import PositionwiseFeedForward
from .embedding import PositionalEncoding, TokenEmbedding
from .masks import create_padding_mask, create_subsequent_mask

__all__ = [
    "Transformer",
    "Encoder",
    "EncoderLayer",
    "Decoder",
    "DecoderLayer",
    "MultiHeadAttention",
    "scaled_dot_product_attention",
    "PositionwiseFeedForward",
    "PositionalEncoding",
    "TokenEmbedding",
    "create_padding_mask",
    "create_subsequent_mask",
]
