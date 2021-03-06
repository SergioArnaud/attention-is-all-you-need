import torch.nn as nn
from transformer.attention import MultiHeadAttention
from transformer.util_layers import Residual, PositionwiseFeedForward, PositionalEncoding
import math

class EncoderLayer(nn.Module):
    def __init__(self, h, d_model, attn_dropout=0.1, feed_forward_dropout=0.1):

        super(EncoderLayer, self).__init__()

        d_k = d_model // h
        d_v = d_model // h

        self.attention = Residual(
            MultiHeadAttention(h, d_model, d_k, d_v, dropout=attn_dropout), d_model
        )

        self.feed_forward = Residual(
            PositionwiseFeedForward(d_model=d_model, dropout=feed_forward_dropout),
            d_model,
        )

    def forward(self, x, mask=None):
        return self.feed_forward(self.attention(x, x, x, mask=mask))


class Encoder(nn.Module):
    def __init__(
        self,
        num_tokens,
        N=6,
        h=8,
        d_model=512,
        d_word_vec=512,
        attn_dropout=0.1,
        feed_forward_dropout=0.1,
    ):

        super(Encoder, self).__init__()

        self.d_model = d_model
        self.token_emb = nn.Embedding(num_tokens, d_word_vec) 
        self.positional_enc = PositionalEncoding(d_word_vec)

        self.layers = nn.ModuleList(
            [
                EncoderLayer(h, d_model, attn_dropout, feed_forward_dropout)
                for _ in range(N)
            ]
        )

    def forward(self, x, mask=None):
        x = self.token_emb(x)
        x = self.positional_enc(x).float()

        for layer in self.layers:
            x = layer(x, mask=mask)

        return x
