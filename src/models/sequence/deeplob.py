"""DeepLOB-style regressor (Zhang, Zohren, Roberts, 2019).

The original architecture (arXiv:1808.03668) consumes a fixed window of
100 LOB steps and emits a single 3-class softmax for the price-movement
at step ``T + k``. Inside that window the convolutions can mix
neighbouring time steps freely because the *whole window* is in the
past relative to the prediction.

We adapt it to the Wunder layout (32 features, per-step regression
targets at every step of a 1000-step sequence). The naive port — copy
the unpadded ``(4,1)`` convs and use ``F.interpolate(mode='nearest')``
to recover the original ``T`` — is **not** valid here: each output
position is also a prediction point, so any conv kernel that mixes
future inputs into a given step's representation leaks the answer.

This rewrite keeps the DeepLOB receptive-field philosophy but makes
every temporal operator strictly causal:

  * every Conv2d with ``kernel_t > 1`` uses left-only padding so the
    output at step ``t`` depends only on inputs at steps ``<= t``,
  * the inception MaxPool is left-padded the same way,
  * ``T`` is preserved end-to-end, so the temporal upsample disappears.

After this change, ``forward(X)[:, t]`` equals the streaming output at
step ``t`` for any ``t`` — see ``tests/models/test_registry.py``.

Reference implementation: ``zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import functional as F

from models.base import SequenceModel
from models.sequence._common import RegressionHead
from models import register_model


class CausalConv2d(nn.Module):
    """Conv2d with **left-only** padding along the temporal axis.

    Input shape ``(B, C, T, F)``. The temporal kernel reads positions
    ``[t - (k_t - 1), t]`` — never the future.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        stride: tuple[int, int] = (1, 1),
    ):
        super().__init__()
        self.pad_t = kernel_size[0] - 1
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(0, 0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # F.pad order on a 4-D tensor: (left_F, right_F, left_T, right_T).
        x = F.pad(x, (0, 0, self.pad_t, 0))
        return self.conv(x)


class CausalMaxPool2d(nn.Module):
    """MaxPool2d with left-only temporal padding.

    Kernel must be ``(k_t, 1)``; the feature axis is left untouched.
    """

    def __init__(self, kernel_size: tuple[int, int], stride: tuple[int, int] = (1, 1)):
        super().__init__()
        self.pad_t = kernel_size[0] - 1
        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=(0, 0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (0, 0, self.pad_t, 0))
        return self.pool(x)


@register_model("deeplob", kind="sequence")
class DeepLOBModel(SequenceModel):
    def __init__(self, config: dict):
        super().__init__(config)
        self.n_features = int(config["n_features"])
        self.n_targets = int(config["n_targets"])
        self.window_size = int(config.get("window_size", 100))
        lstm_hidden = int(config.get("lstm_hidden", 64))
        dropout = float(config.get("dropout", 0.1))

        # We treat the F features as a 1xF "image strip" per time step.
        # Each block: one feature-axis compression conv (time-invariant)
        # followed by two causal temporal convs (kernel 4 along T).
        c1 = int(config.get("c1", 16))
        c2 = int(config.get("c2", 16))
        c3 = int(config.get("c3", 32))
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(0.01),
            CausalConv2d(c1, c1, kernel_size=(4, 1)),
            nn.LeakyReLU(0.01),
            CausalConv2d(c1, c1, kernel_size=(4, 1)),
            nn.LeakyReLU(0.01),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(0.01),
            CausalConv2d(c2, c2, kernel_size=(4, 1)),
            nn.LeakyReLU(0.01),
            CausalConv2d(c2, c2, kernel_size=(4, 1)),
            nn.LeakyReLU(0.01),
        )
        # Final block collapses the feature axis to width 1.
        feat_after_2_blocks = self.n_features // 4
        self.conv3 = nn.Sequential(
            nn.Conv2d(c2, c3, kernel_size=(1, max(1, feat_after_2_blocks)), stride=(1, 1)),
            nn.LeakyReLU(0.01),
            CausalConv2d(c3, c3, kernel_size=(4, 1)),
            nn.LeakyReLU(0.01),
            CausalConv2d(c3, c3, kernel_size=(4, 1)),
            nn.LeakyReLU(0.01),
        )

        # Causal inception: kernels 3 / 5 / pool(3), all left-padded.
        inc = int(config.get("inception_channels", 32))
        self.inc_branch1 = nn.Sequential(
            nn.Conv2d(c3, inc, kernel_size=(1, 1)),
            nn.LeakyReLU(0.01),
            CausalConv2d(inc, inc, kernel_size=(3, 1)),
            nn.LeakyReLU(0.01),
        )
        self.inc_branch2 = nn.Sequential(
            nn.Conv2d(c3, inc, kernel_size=(1, 1)),
            nn.LeakyReLU(0.01),
            CausalConv2d(inc, inc, kernel_size=(5, 1)),
            nn.LeakyReLU(0.01),
        )
        self.inc_branch3 = nn.Sequential(
            CausalMaxPool2d(kernel_size=(3, 1)),
            nn.Conv2d(c3, inc, kernel_size=(1, 1)),
            nn.LeakyReLU(0.01),
        )
        inception_out = inc * 3

        self.lstm = nn.LSTM(
            input_size=inception_out,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
        )
        self.head = RegressionHead(lstm_hidden, self.n_targets, dropout=dropout)

    def _conv_forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F) -> (B, 1, T, F)
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)                   # (B, c3, T, 1)
        b1 = self.inc_branch1(x)
        b2 = self.inc_branch2(x)
        b3 = self.inc_branch3(x)
        x = torch.cat([b1, b2, b3], dim=1)  # (B, 3*inc, T, 1)
        x = x.squeeze(-1).transpose(1, 2)   # (B, T, 3*inc)
        return x

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features: (B, T, F). The causal conv stack preserves T, so the
        # per-step head can be applied directly with no upsampling.
        x = self._conv_forward(features)             # (B, T, H)
        x, _ = self.lstm(x)                          # (B, T, H)
        preds = self.head(x)                         # (B, T, K)
        return preds
