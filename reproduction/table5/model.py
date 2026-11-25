"""Paired PatchTST models using the TimeFT repository's shared forecasting path."""

import torch
from torch import nn
from torch.nn import functional as F

from Models.FreqTokenizer import Model as ForecastBackbone
from layers.Embed import PatchEmbedding, PositionalEmbedding


class TimeFTEmbedding(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.seq_len = args.seq_len
        self.stride = args.stride
        self.patch_len = args.patch_len
        self.band_width = self.seq_len // self.stride // 2
        self.frequency_bins = self.seq_len // 2 + 1
        self.bands = (self.frequency_bins + self.band_width - 1) // self.band_width
        patch_count = (self.seq_len + self.stride - self.patch_len) // self.stride + 1
        self.projection = nn.Linear(self.bands * patch_count, args.d_model)
        self.position_embedding = PositionalEmbedding(args.d_model)
        self.dropout = nn.Dropout(args.dropout)

    def tokenize(self, x):
        spectrum = torch.fft.rfft(x, dim=-1)
        spectrum = F.pad(spectrum, (0, self.bands * self.band_width - self.frequency_bins))
        bands = spectrum.unflatten(-1, (self.bands, self.band_width))
        shifted = torch.fft.irfft(bands, n=self.seq_len, dim=-1)
        padded = torch.cat([shifted, shifted[..., -1:].expand(*shifted.shape[:-1], self.stride)], dim=-1)
        return padded.unfold(-1, self.patch_len, self.stride).permute(0, 1, 4, 2, 3)

    def forward(self, x):
        channels = x.shape[1]
        tokens = self.tokenize(x).flatten(-2).flatten(0, 1)
        embedded = self.projection(tokens)
        return self.dropout(embedded + self.position_embedding(embedded)), channels


class PairedPatchTST(ForecastBackbone):
    def __init__(self, args, variant):
        super().__init__(args)
        patch_count = (args.seq_len + args.stride - args.patch_len) // args.stride + 1
        if variant == "patch":
            self.patch_embedding = PatchEmbedding(
                args.d_model, args.patch_len, args.stride, args.stride, args.dropout
            )
            self.token_count = patch_count
        else:
            self.patch_embedding = TimeFTEmbedding(args)
            self.token_count = args.patch_len

        # The inherited forecast path expects [B, C, d_model, tokens] at the head.
        # Install its Linear before optimizer creation and checkpoint loading.
        self.head_nf = args.d_model * self.token_count
        self.head.linear = nn.Linear(self.head_nf, args.pred_len)
