"""GRU baseline: vanilla recurrent regressor on the 32 LOB features."""

from __future__ import annotations

import torch
import torch.nn as nn

from models.base import SequenceModel, StreamingState
from models.sequence._common import FeatureProjector, RegressionHead
from models import register_model


@register_model("gru", kind="sequence")
class GRUModel(SequenceModel):
    def __init__(self, config: dict):
        super().__init__(config)
        n_features = int(config["n_features"])
        n_targets = int(config["n_targets"])
        hidden = int(config.get("hidden_size", 128))
        n_layers = int(config.get("num_layers", 2))
        dropout = float(config.get("dropout", 0.1))

        self.proj = FeatureProjector(n_features, hidden, dropout=dropout)
        self.rnn = nn.GRU(
            input_size=hidden,
            hidden_size=hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.head = RegressionHead(hidden, n_targets, dropout=dropout)

    # Training forward.
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.proj(features)         # (B, T, H)
        h, _ = self.rnn(x)              # (B, T, H)
        return self.head(h)             # (B, T, K)

    # Streaming.
    def init_state(self, batch_size: int, device: torch.device) -> StreamingState:
        h0 = torch.zeros(
            self.rnn.num_layers, batch_size, self.rnn.hidden_size, device=device
        )
        return StreamingState(payload={"hidden": h0})

    def step(self, feature_step: torch.Tensor, state: StreamingState):
        h = state.payload["hidden"]
        x = self.proj(feature_step.unsqueeze(1))   # (B, 1, H)
        out, h_new = self.rnn(x, h)
        pred = self.head(out[:, -1, :])
        state.payload["hidden"] = h_new
        return pred, state
