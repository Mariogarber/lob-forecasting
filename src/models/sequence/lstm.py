"""LSTM baseline (same recipe as the GRU, just a different recurrent core)."""

from __future__ import annotations

import torch
import torch.nn as nn

from models.base import SequenceModel, StreamingState
from models.sequence._common import FeatureProjector, RegressionHead
from models import register_model


@register_model("lstm", kind="sequence")
class LSTMModel(SequenceModel):
    def __init__(self, config: dict):
        super().__init__(config)
        n_features = int(config["n_features"])
        n_targets = int(config["n_targets"])
        hidden = int(config.get("hidden_size", 128))
        n_layers = int(config.get("num_layers", 2))
        dropout = float(config.get("dropout", 0.1))
        bidirectional = bool(config.get("bidirectional", False))

        self.proj = FeatureProjector(n_features, hidden, dropout=dropout)
        self.rnn = nn.LSTM(
            input_size=hidden,
            hidden_size=hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        out_dim = hidden * (2 if bidirectional else 1)
        self.head = RegressionHead(out_dim, n_targets, dropout=dropout)
        self.bidirectional = bidirectional

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.proj(features)
        h, _ = self.rnn(x)
        return self.head(h)

    def init_state(self, batch_size: int, device: torch.device) -> StreamingState:
        if self.bidirectional:
            # Bidirectional LSTM cannot be streamed step-by-step honestly;
            # we fall back to the rolling-buffer default in that case.
            return super().init_state(batch_size, device)
        h0 = torch.zeros(
            self.rnn.num_layers, batch_size, self.rnn.hidden_size, device=device
        )
        c0 = torch.zeros_like(h0)
        return StreamingState(payload={"hidden": (h0, c0)})

    def step(self, feature_step: torch.Tensor, state: StreamingState):
        if self.bidirectional:
            return super().step(feature_step, state)
        h, c = state.payload["hidden"]
        x = self.proj(feature_step.unsqueeze(1))
        out, (h_new, c_new) = self.rnn(x, (h, c))
        pred = self.head(out[:, -1, :])
        state.payload["hidden"] = (h_new, c_new)
        return pred, state
