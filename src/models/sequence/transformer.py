"""Causal Transformer encoder for LOB sequences.

A vanilla pre-LN transformer with a causal mask, so step *t* only attends
to steps ``<= t``. This makes the model usable both for full-sequence
training and step-by-step inference via the default rolling buffer.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.base import SequenceModel
from models.sequence._common import (
    FeatureProjector,
    RegressionHead,
    SinusoidalPositionalEncoding,
)
from models import register_model


@register_model("transformer", kind="sequence")
class TransformerModel(SequenceModel):
    def __init__(self, config: dict):
        super().__init__(config)
        n_features = int(config["n_features"])
        n_targets = int(config["n_targets"])
        d_model = int(config.get("d_model", 128))
        nhead = int(config.get("nhead", 4))
        n_layers = int(config.get("num_layers", 3))
        dim_ff = int(config.get("dim_feedforward", 256))
        dropout = float(config.get("dropout", 0.1))
        max_len = int(config.get("max_len", 1024))

        self.proj = FeatureProjector(n_features, d_model, dropout=dropout)
        self.pos = SinusoidalPositionalEncoding(d_model, max_len=max_len)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = RegressionHead(d_model, n_targets, dropout=dropout)
        self.window_size = int(config.get("window_size", 0))

    @staticmethod
    def _causal_mask(T: int, device: torch.device) -> torch.Tensor:
        return torch.triu(
            torch.full((T, T), float("-inf"), device=device), diagonal=1
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.proj(features)
        x = self.pos(x)
        mask = self._causal_mask(x.size(1), x.device)
        h = self.encoder(x, mask=mask, is_causal=True)
        return self.head(h)
