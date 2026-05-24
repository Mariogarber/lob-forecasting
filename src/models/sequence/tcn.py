"""Temporal Convolutional Network (TCN) for LOB per-step regression.

Architecture from Bai, Kolter & Koltun (2018), "An Empirical Evaluation of
Generic Convolutional and Recurrent Networks for Sequence Modeling"
(arXiv:1803.01271). The TCN is a stack of residual blocks where each block
applies two causal 1-D convolutions with exponentially increasing dilation.

Why it fits the LOB problem
---------------------------
* **Causal by construction** — left-only padding guarantees that the output
  at step ``t`` depends solely on inputs ``<= t``. Streaming and batched
  forward give identical predictions (verified by the registry test).
* **Long receptive field, cheaply** — dilation doubles at every level, so
  with ``L`` blocks of kernel ``k`` we cover
  ``RF = 1 + 2 * (k - 1) * (2^L - 1)`` time steps. Five blocks with ``k=3``
  already see 125 past steps with ~125 k parameters (vs >300 k for the
  default LSTM and ~700 k for the transformer).
* **Parallel training** — unlike an RNN, every step of a sequence is
  computed in one pass on the GPU, so a 1000-step sequence trains in the
  same number of FLOPs whether the receptive field is 30 or 300.
* **Stable gradients** — residual connections + weight-norm keep deep
  stacks (8-10 blocks) trainable, where a plain CNN would saturate.

Compared to the other sequence models in the repo:
  - vs GRU/LSTM:      no recurrent bottleneck; trains in parallel,
                       receptive field is exact (not "effectively long").
  - vs Transformer:   no O(T^2) attention; same causal property without
                       the quadratic cost.
  - vs DeepLOB:       same causal-conv idea but with dilation instead of
                       stacked kernel-4 layers — orders of magnitude more
                       receptive field per parameter.
  - vs Mamba-2:       fully PyTorch, no CUDA kernels needed, no
                       d_model channel-layout constraints.

Reference implementation: ``locuslab/TCN``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm

from models.base import SequenceModel
from models.sequence._common import RegressionHead
from models import register_model


class CausalConv1d(nn.Module):
    """Conv1d with **left-only** padding on the time axis.

    Input shape ``(B, C, T)``. The kernel at output position ``t`` reads
    positions ``[t - (k - 1) * d, ..., t]``, never the future.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
    ):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)


class TemporalBlock(nn.Module):
    """One residual block of the TCN.

    Two causal dilated convs, ReLU + dropout between them, plus a 1x1
    projection on the residual path when channel counts differ.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        # weight_norm() expects a Module with a ``weight`` parameter, so we
        # wrap plain Conv1d here and handle the left-only padding in forward.
        self.pad = (kernel_size - 1) * dilation
        self.conv1 = weight_norm(
            nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=0)
        )
        self.conv2 = weight_norm(
            nn.Conv1d(out_channels, out_channels, kernel_size, dilation=dilation, padding=0)
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.residual = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )
        self._init_weights()

    def _init_weights(self) -> None:
        # Following locuslab/TCN: small initial scale keeps deep stacks tame.
        nn.init.normal_(self.conv1.weight, 0.0, 0.01)
        nn.init.normal_(self.conv2.weight, 0.0, 0.01)
        if isinstance(self.residual, nn.Conv1d):
            nn.init.normal_(self.residual.weight, 0.0, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.pad(x, (self.pad, 0))
        h = self.conv1(h)
        h = self.dropout1(F.relu(h))
        h = F.pad(h, (self.pad, 0))
        h = self.conv2(h)
        h = self.dropout2(F.relu(h))
        return F.relu(h + self.residual(x))


@register_model("tcn", kind="sequence")
class TCN(SequenceModel):
    """Stack of residual ``TemporalBlock``s with exponential dilation.

    Default configuration (5 blocks of 64 channels, kernel 3) has a
    receptive field of 125 steps and ~125 k parameters. Bump ``num_layers``
    to grow the receptive field cheaply: each additional layer doubles it.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        n_features = int(config["n_features"])
        n_targets = int(config["n_targets"])
        channels = int(config.get("channels", 64))
        num_layers = int(config.get("num_layers", 5))
        kernel_size = int(config.get("kernel_size", 3))
        dropout = float(config.get("dropout", 0.1))

        layers: list[nn.Module] = []
        in_ch = n_features
        for i in range(num_layers):
            dilation = 2 ** i
            layers.append(
                TemporalBlock(
                    in_channels=in_ch,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
            in_ch = channels
        self.tcn = nn.Sequential(*layers)
        self.head = RegressionHead(channels, n_targets, dropout=dropout)

        # Receptive field for reference / matches the window_size default.
        self.receptive_field = 1 + 2 * (kernel_size - 1) * (2 ** num_layers - 1)
        # Default window_size to the receptive field so streaming step() does
        # not pay for buffering more history than the model can see anyway.
        self.window_size = int(config.get("window_size", self.receptive_field))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # (B, T, F) -> (B, F, T) for Conv1d -> stack -> (B, C, T) -> (B, T, C)
        x = features.transpose(1, 2)
        x = self.tcn(x)
        x = x.transpose(1, 2)
        return self.head(x)
