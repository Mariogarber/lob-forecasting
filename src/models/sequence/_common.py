"""Shared building blocks for sequence models."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class FeatureProjector(nn.Module):
    """Input projection: ``Linear -> LayerNorm -> Dropout``."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.norm(self.proj(x)))


class RegressionHead(nn.Module):
    """Predict (..., K) targets from a hidden representation."""

    def __init__(self, hidden_dim: int, n_targets: int, dropout: float = 0.0):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, n_targets)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.drop(x))


class SinusoidalPositionalEncoding(nn.Module):
    """Sin/cos positional encoding bounded to ``max_len`` positions."""

    def __init__(self, dim: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float) * -(math.log(10000.0) / dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: dim - dim // 2 * 0])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        T = x.size(1)
        if T > self.pe.size(1):
            raise ValueError(
                f"sequence length {T} exceeds positional encoding length {self.pe.size(1)}"
            )
        return x + self.pe[:, :T]
