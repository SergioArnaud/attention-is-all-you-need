import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(Q, K, V, temperature, dropout, mask=None):
    attn = Q.bmm(K.transpose(1, 2))

    if mask is not None:
        attn = attn.masked_fill(mask == 0, -1e9)

    attn = F.softmax(attn / temperature, dim=-1)
    attn = dropout(attn)
    return attn.bmm(V)


class AttentionHead(nn.Module):
    def __init__(self, d_model, d_k, d_v, dropout=0.1):

        super(AttentionHead, self).__init__()

        self.W_Q = nn.Linear(d_model, d_k, bias=False)
        self.W_K = nn.Linear(d_model, d_k, bias=False)
        self.W_V = nn.Linear(d_model, d_v, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.temperature = d_k ** 2

    def forward(self, Q, K, V, mask=None):

        Q = self.W_Q(Q)
        K = self.W_K(K)
        V = self.W_V(V)

        head = scaled_dot_product_attention(
            Q, K, V, self.temperature, self.dropout, mask=mask
        )
        return head


class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, d_k, d_v, dropout=0.1):

        super(MultiHeadAttention, self).__init__()

        self.W_O = nn.Linear(h * d_v, d_model, bias=False)
        self.heads = nn.ModuleList(
            [AttentionHead(d_model, d_k, d_v, dropout) for _ in range(h)]
        )

    def forward(self, Q, K, V, mask=None):

        heads = [h(Q, K, V, mask) for h in self.heads]
        return self.W_O(torch.cat(heads, dim=-1))
