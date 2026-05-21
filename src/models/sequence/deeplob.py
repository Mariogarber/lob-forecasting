"""DeepLOB-style regressor (Zhang, Zohren, Roberts, 2019).

The original architecture targets a 100-step LOB window with 40 features
(prices + volumes interleaved at 10 levels). We adapt it to the Wunder
layout (32 features, sequence steps from 1 to N) by:

  * keeping the 1xN convolutional stack that learns price/volume pairs,
  * using ``window_size`` from the config (default 100) instead of a hard
    100-step window — the streaming wrapper buffers that many steps,
  * replacing the 3-way softmax classification head with a 2-output
    linear regression head.

Reference implementation: ``zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import functional as F

from models.base import SequenceModel
from models.sequence._common import RegressionHead
from models import register_model


@register_model("deeplob", kind="sequence")
class DeepLOBModel(SequenceModel):
    def __init__(self, config: dict):
        super().__init__(config)
        self.n_features = int(config["n_features"])
        self.n_targets = int(config["n_targets"])
        self.window_size = int(config.get("window_size", 100))
        lstm_hidden = int(config.get("lstm_hidden", 64))
        dropout = float(config.get("dropout", 0.1))

        # We treat the 32 features as a 1xF "image strip" per time step.
        # Three convolution blocks compress feature dimension while
        # extracting micro-structure across pairs of feature columns.
        c1 = int(config.get("c1", 16))
        c2 = int(config.get("c2", 16))
        c3 = int(config.get("c3", 32))
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(0.01),
            nn.Conv2d(c1, c1, kernel_size=(4, 1), padding=(0, 0)),
            nn.LeakyReLU(0.01),
            nn.Conv2d(c1, c1, kernel_size=(4, 1), padding=(0, 0)),
            nn.LeakyReLU(0.01),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(0.01),
            nn.Conv2d(c2, c2, kernel_size=(4, 1), padding=(0, 0)),
            nn.LeakyReLU(0.01),
            nn.Conv2d(c2, c2, kernel_size=(4, 1), padding=(0, 0)),
            nn.LeakyReLU(0.01),
        )
        # Last conv collapses the feature axis to width 1.
        feat_after_2_blocks = self.n_features // 4
        self.conv3 = nn.Sequential(
            nn.Conv2d(c2, c3, kernel_size=(1, max(1, feat_after_2_blocks)), stride=(1, 1)),
            nn.LeakyReLU(0.01),
            nn.Conv2d(c3, c3, kernel_size=(4, 1), padding=(0, 0)),
            nn.LeakyReLU(0.01),
            nn.Conv2d(c3, c3, kernel_size=(4, 1), padding=(0, 0)),
            nn.LeakyReLU(0.01),
        )

        # An inception module — 1x1 / 3x1 / max-pool branches.
        inc = int(config.get("inception_channels", 32))
        self.inc_branch1 = nn.Sequential(
            nn.Conv2d(c3, inc, kernel_size=(1, 1)),
            nn.LeakyReLU(0.01),
            nn.Conv2d(inc, inc, kernel_size=(3, 1), padding=(1, 0)),
            nn.LeakyReLU(0.01),
        )
        self.inc_branch2 = nn.Sequential(
            nn.Conv2d(c3, inc, kernel_size=(1, 1)),
            nn.LeakyReLU(0.01),
            nn.Conv2d(inc, inc, kernel_size=(5, 1), padding=(2, 0)),
            nn.LeakyReLU(0.01),
        )
        self.inc_branch3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 1), stride=1, padding=(1, 0)),
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
        x = self.conv3(x)              # (B, c3, T', 1)
        b1 = self.inc_branch1(x)
        b2 = self.inc_branch2(x)
        b3 = self.inc_branch3(x)
        x = torch.cat([b1, b2, b3], dim=1)  # (B, 3*inc, T', 1)
        x = x.squeeze(-1).transpose(1, 2)   # (B, T', 3*inc)
        return x

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features: (B, T, F). The conv stack reduces T to T'; we upsample
        # the per-step predictions back to T by nearest-neighbour repeat
        # so the loss can be applied element-wise.
        B, T, _ = features.shape
        x = self._conv_forward(features)             # (B, T', H)
        x, _ = self.lstm(x)                          # (B, T', H)
        preds = self.head(x)                         # (B, T', K)

        # Upsample to T via interpolation along the time axis. This is a
        # straight nearest-neighbour expansion so each "step" gets the
        # latest available conv-window prediction.
        if preds.size(1) != T:
            preds = preds.transpose(1, 2)                 # (B, K, T')
            preds = F.interpolate(preds, size=T, mode="nearest")
            preds = preds.transpose(1, 2)                 # (B, T, K)
        return preds
