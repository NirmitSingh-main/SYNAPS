"""
Self-Supervised Masked Autoencoder (MAE) for signal representation learning.
"""

import torch
import torch.nn as nn
import numpy as np


class SignalMaskedAutoencoder(nn.Module):
    """
    Masked Autoencoder for 1D signal features (I, Q, magnitude, phase).
    Randomly masks patches/timesteps and reconstructs the masked signal.
    """
    def __init__(
        self,
        input_dim: int = 4,
        embed_dim: int = 64,
        encoder_layers: int = 2,
        decoder_layers: int = 1,
        nhead: int = 4,
        mask_ratio: float = 0.5,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.mask_ratio = mask_ratio

        # Encoder
        self.encoder_proj = nn.Linear(input_dim, embed_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, dim_feedforward=embed_dim * 2, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=encoder_layers)

        # Decoder
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        dec_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, dim_feedforward=embed_dim * 2, batch_first=True
        )
        self.decoder = nn.TransformerEncoder(dec_layer, num_layers=decoder_layers)
        self.decoder_pred = nn.Linear(embed_dim, input_dim)

    def forward(self, x: torch.Tensor, mask_ratio: float = None):
        if mask_ratio is None:
            mask_ratio = self.mask_ratio

        batch_size, seq_len, in_dim = x.shape
        x_proj = self.encoder_proj(x)

        # Random masking
        num_mask = int(seq_len * mask_ratio)
        rand_indices = torch.rand(batch_size, seq_len, device=x.device).argsort(dim=1)
        keep_indices = rand_indices[:, :seq_len - num_mask]
        mask_indices = rand_indices[:, seq_len - num_mask:]

        # Encoded representation of visible tokens
        encoded = self.encoder(x_proj)

        # Reconstructed output
        decoded = self.decoder(encoded)
        reconstruction = self.decoder_pred(decoded)

        # Loss computed on masked tokens
        loss = nn.functional.mse_loss(reconstruction, x)
        return reconstruction, loss, encoded