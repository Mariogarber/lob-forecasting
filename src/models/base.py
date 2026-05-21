"""Abstract base classes for the model registry.

Two flavours:

* ``ClassicalModel`` — wraps a fit/predict library (sklearn, LightGBM,
  CatBoost, ...). Internally one estimator per target column because
  most of the boosting libraries cannot do multi-output regression with
  per-sample weights in a single call. Optional but recommended.

* ``SequenceModel`` — a ``torch.nn.Module`` that consumes a 3-D
  ``(batch, time, features)`` tensor and emits a 3-D
  ``(batch, time, targets)`` tensor. The trainer handles the loss,
  optimiser and mask. The model is also responsible for exposing two
  helpers used at inference time:

      * ``init_state(batch_size, device)`` — fresh per-sequence state.
      * ``step(features, state)``          — single-step forward pass.

  Implementing both methods is what makes the streaming PredictionModel
  fast at submission time. Window-based models (DeepLOB) just keep a
  rolling buffer inside the state dict.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


class BaseModel(ABC):
    """Marker class so the registry can hold both classical and DL models."""

    _registered_name: str = ""
    _registered_kind: str = ""

    @property
    def name(self) -> str:
        return self._registered_name

    @property
    def kind(self) -> str:
        return self._registered_kind

    @abstractmethod
    def save(self, path: str | Path) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path, **kwargs) -> "BaseModel": ...


# ---------------------------------------------------------------------------
# Classical
# ---------------------------------------------------------------------------


class ClassicalModel(BaseModel):
    """sklearn-style models. One sub-estimator per target column."""

    def __init__(self, config: dict[str, Any]):
        self.config = dict(config or {})
        self.estimators_: list[Any] = []
        self.feature_names_: list[str] = []
        self.target_names_: list[str] = []

    # ------------------------------------------------------------------
    # The two abstract methods that concrete classical models implement.
    # ------------------------------------------------------------------
    @abstractmethod
    def _new_estimator(self):
        """Return a fresh sklearn-style estimator (fit / predict)."""

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
        *,
        eval_set: tuple[np.ndarray, np.ndarray] | None = None,
        feature_names: list[str] | None = None,
        target_names: list[str] | None = None,
    ) -> "ClassicalModel":
        self.estimators_ = []
        self.feature_names_ = list(feature_names or [])
        self.target_names_ = list(target_names or [f"y{i}" for i in range(y.shape[1])])
        for i in range(y.shape[1]):
            est = self._new_estimator()
            self._fit_one(est, X, y[:, i], sample_weight, eval_set=eval_set, target_idx=i)
            self.estimators_.append(est)
        return self

    def _fit_one(self, est, X, y_col, sample_weight, *, eval_set, target_idx):
        # Default: pass sample_weight through to fit, ignore eval_set.
        # LightGBM/XGBoost overrides this to wire in early stopping.
        kwargs = {}
        if sample_weight is not None:
            kwargs["sample_weight"] = sample_weight
        est.fit(X, y_col, **kwargs)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.estimators_:
            raise RuntimeError(f"{self.name}: model is not fitted.")
        preds = np.stack([est.predict(X) for est in self.estimators_], axis=1)
        return preds.astype(np.float32)

    # ------------------------------------------------------------------
    # Persistence (joblib).
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        import joblib
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "config": self.config,
                "estimators": self.estimators_,
                "feature_names": self.feature_names_,
                "target_names": self.target_names_,
                "registered_name": self._registered_name,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, **kwargs) -> "ClassicalModel":
        import joblib
        blob = joblib.load(path)
        obj = cls(blob["config"])  # type: ignore[arg-type]
        obj.estimators_ = blob["estimators"]
        obj.feature_names_ = blob["feature_names"]
        obj.target_names_ = blob["target_names"]
        return obj


# ---------------------------------------------------------------------------
# Sequence
# ---------------------------------------------------------------------------


@dataclass
class StreamingState:
    """Lightweight container for per-sequence state passed between steps."""

    payload: Any = None


class SequenceModel(BaseModel, nn.Module):
    """Base class for ``torch.nn.Module`` sequence regressors.

    Subclasses implement:
      * ``forward(features) -> predictions``                — for training.
      * ``init_state(batch_size, device)``                  — for streaming.
      * ``step(feature_step, state) -> (pred_step, state)`` — for streaming.

    The default implementations of ``init_state`` / ``step`` are *batched*
    re-uses of ``forward`` that buffer the input window — slow but correct.
    Subclasses with a real recurrent / SSM state override them for speed.
    """

    def __init__(self, config: dict[str, Any]):
        nn.Module.__init__(self)
        self.config = dict(config or {})
        self.window_size: int = int(self.config.get("window_size", 0))  # 0 = full seq

    # ------------------------------------------------------------------
    # Required forward (training).
    # ------------------------------------------------------------------
    @abstractmethod
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Map ``features`` (B, T, F) -> predictions (B, T, K)."""
        ...

    # ------------------------------------------------------------------
    # Streaming inference (default = rolling buffer).
    # ------------------------------------------------------------------
    def init_state(self, batch_size: int, device: torch.device) -> StreamingState:
        n_features = int(self.config["n_features"])
        buffer = torch.zeros(batch_size, 0, n_features, device=device)
        return StreamingState(payload={"buffer": buffer})

    def step(
        self,
        feature_step: torch.Tensor,    # (B, F)
        state: StreamingState,
    ) -> tuple[torch.Tensor, StreamingState]:
        buffer = state.payload["buffer"]              # (B, T_so_far, F)
        buffer = torch.cat([buffer, feature_step.unsqueeze(1)], dim=1)
        if self.window_size and buffer.shape[1] > self.window_size:
            buffer = buffer[:, -self.window_size:]
        preds = self.forward(buffer)                  # (B, T_so_far, K)
        state.payload["buffer"] = buffer
        return preds[:, -1, :], state

    # ------------------------------------------------------------------
    # Persistence.
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": self.config,
                "state_dict": self.state_dict(),
                "registered_name": self._registered_name,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, **kwargs) -> "SequenceModel":
        blob = torch.load(path, map_location=kwargs.get("map_location", "cpu"), weights_only=False)
        obj = cls(blob["config"])
        obj.load_state_dict(blob["state_dict"])
        return obj
