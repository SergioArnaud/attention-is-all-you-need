import torch.nn as nn
from transformer.attention import MultiHeadAttention
from transformer.util_layers import Residual, PositionwiseFeedForward, PositionalEncoding
import math


class DecoderLayer(nn.Module):
    def __init__(self, h, d_model, attn_dropout=0.1, feed_forward_dropout=0.1):

        super(DecoderLayer, self).__init__()

        d_k = d_model // h
        d_v = d_model // h

        self.self_attention = Residual(
            MultiHeadAttention(h, d_model, d_k, d_v, dropout=attn_dropout), d_model
        )

        self.encoder_decoder_attention = Residual(
            MultiHeadAttention(h, d_model, d_k, d_v, dropout=attn_dropout), d_model
        )

        self.feed_forward = Residual(
            PositionwiseFeedForward(d_model=d_model, dropout=feed_forward_dropout),
            d_model,
        )

    def forward(self, decoder_input, encoder_output, self_attn_mask=None):

        decoder_output = self.self_attention(
            decoder_input, decoder_input, decoder_input, mask=self_attn_mask
        )

        attn = self.encoder_decoder_attention(
            decoder_output, encoder_output, encoder_output
        )
        return self.feed_forward(attn)


class Decoder(nn.Module):
    def __init__(
        self,
        num_tokens,
        N=6,
        h=8,
        d_model=512,
        d_word_vec=512,
        attn_dropout=0.1,
        feed_forward_dropout=0.1
    ):

        super(Decoder, self).__init__()

        self.d_model = d_model
        self.token_emb = nn.Embedding(num_tokens, d_word_vec) 
        self.positional_enc = PositionalEncoding(d_word_vec)

        self.layers = nn.ModuleList(
            [
                DecoderLayer(h, d_model, attn_dropout, feed_forward_dropout)
                for _ in range(N)
            ]
        )

    def forward(self, target, encoder_output, self_attn_mask=None):

        target = self.token_emb(target)
        dec_output = self.positional_enc(target).float()

        for layer in self.layers:
            dec_output = layer(
                dec_output, encoder_output, self_attn_mask=self_attn_mask
            )

        return dec_output
