import torch
import torch.nn as nn
from transformer.encoder import Encoder
from transformer.decoder import Decoder
import torch.nn.functional as F


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
        d_word_vec=128,
        N=2,
        h=2,
        d_model=128,
        attn_dropout=0.1,
        feed_forward_dropout=0.1,
    ):

        super(Transformer, self).__init__()
        self.d_model = d_model
        self.encoder = Encoder(
            num_tokens_src,
            N,
            h,
            d_model,
            d_word_vec,
            attn_dropout,
            feed_forward_dropout,
        )

        self.decoder = Decoder(
            num_tokens_tgt,
            N,
            h,
            d_model,
            d_word_vec,
            attn_dropout,
            feed_forward_dropout,
        )

        self.linear = nn.Linear(d_model, num_tokens_tgt)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform(p)

    def forward(self, source, target, pad=0, subsequent_mask=True):
        
        src_mask = (source != pad).unsqueeze(-2)
        
        tgt_mask = (target != pad).unsqueeze(-2)

        if subsequent_mask:
            tgt_mask = tgt_mask & get_subsequent_mask(target)

        encoder_output = self.encoder(source, src_mask)
        dec_output = self.decoder(target, encoder_output, tgt_mask)

        return self.linear(dec_output)
