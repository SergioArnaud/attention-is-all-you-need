import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model

    def _get_positional_encoding(self, n_positions):

        # Get numpy arrays with positions and dimensions
        
        positions, dimensions = np.ix_(np.arange(n_positions), np.arange(self.d_model))

        # Compute frequency
        positional_encoding = positions / (
            np.power(10000, 2 * (dimensions // 2) / self.d_model)
        )

        # Map frequency to sinusoidal value
        positional_encoding[:, 0::2] = np.sin(positional_encoding[:, 0::2])  # dim 2i
        positional_encoding[:, 1::2] = np.cos(positional_encoding[:, 1::2])  # dim 2i+1

        return torch.tensor(positional_encoding).unsqueeze_(0).cuda()

    def forward(self, x):
        return x + self._get_positional_encoding(x.size(1))[:, :x.size(1)]


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model=512, d_hidden=2048, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.W1 = nn.Linear(d_model, d_hidden)
        self.W2 = nn.Linear(d_hidden, d_model)

    def forward(self, x):
        return self.W2(F.relu(self.W1(x)))


class Residual(nn.Module):
    def __init__(self, sublayer, d_model):
        super(Residual, self).__init__()
        self.layer_normalization = nn.LayerNorm(d_model)
        self.current_sublayer = sublayer

    def forward(self, *x, mask=None):
        if mask is None:
            return self.layer_normalization(x[-1] + self.current_sublayer(*x))
        else:
            return self.layer_normalization(
                x[-1] + self.current_sublayer(*x, mask=mask)
            )
