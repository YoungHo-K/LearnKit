import math
import torch
import torch.nn as nn


class _PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_len):
        super(_PositionalEncoder, self).__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        encoding = torch.zeros(max_len, d_model)
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("encoding", encoding)

    def forward(self, x):
        return x + self.encoding[: x.size(1)]


class _TransformerEncoder(nn.Module):
    def __init__(self, in_features=6, max_len=300, d_model=16, num_heads=8, d_hidden=64, num_layers=3, dropout=0.5, device=None):
        super(_TransformerEncoder, self).__init__()

        base_encoder = nn.TransformerEncoderLayer(d_model, num_heads, d_hidden, dropout, batch_first=True, device=device)
        self.tranformer_encoder = nn.TransformerEncoder(base_encoder, num_layers)
        self.positional_encoder = _PositionalEncoder(d_model, max_len)
        self.src_mask = torch.triu(torch.ones(300, 300) * float('-inf'), diagonal=1).to(device)
        self.linear = nn.Linear(in_features, d_model)

    def forward(self, x):
        x = self.linear(x)
        x = self.positional_encoder(x)
        output = self.tranformer_encoder(x, self.src_mask)

        return output


class TransformerEncoderAnalyzer(nn.Module):
    def __init__(self, in_features=6, max_len=300, d_model=16, num_heads=8, d_hidden=64, num_layers=3, dropout=0.5, device=None):
        super(TransformerEncoderAnalyzer, self).__init__()

        sequence_encoder = _TransformerEncoder(in_features, max_len, d_model, num_heads, d_hidden, num_layers, dropout, device)
        self.network = nn.Sequential(
            sequence_encoder,
            nn.AvgPool1d(kernel_size=d_model),
            nn.Flatten(),
            nn.Linear(in_features=300, out_features=16),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(),
            nn.Linear(in_features=16, out_features=1))

    def forward(self, x):
        return self.network(x)
