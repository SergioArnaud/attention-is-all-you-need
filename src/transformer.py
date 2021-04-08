import torch
import torch.nn as nn
from encoder import Encoder
from decoder import Decoder


def get_subsequent_mask(seq):
    sz_b, len_s = seq.size()
    subsequent_mask = (
        1 - torch.triu(torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)
    ).bool()
    return subsequent_mask


class Transformer(nn.Module):
    def __init__(
        self,
        num_tokens_src,
        num_tokens_tgt,
        d_word_vec=512,
        N=6,
        h=8,
        d_model=512,
        attn_dropout=0.1,
        feed_forward_dropout=0.1,
        n_positions=512,
    ):

        super(Transformer, self).__init__()

        self.encoder = Encoder(
            num_tokens_src,
            N,
            h,
            d_model,
            d_word_vec,
            attn_dropout,
            feed_forward_dropout,
            n_positions,
        )

        self.decoder = Decoder(
            num_tokens_tgt,
            N,
            h,
            d_model,
            d_word_vec,
            attn_dropout,
            feed_forward_dropout,
            n_positions,
        )

        self.linear = nn.Linear(d_model, num_tokens_tgt)

    def forward(self, source, target):

        mask = get_subsequent_mask(target)

        encoder_output = self.encoder(source)

        dec_output = self.decoder(target, encoder_output, mask)

        dec_output = self.linear(dec_output)

        return dec_output
