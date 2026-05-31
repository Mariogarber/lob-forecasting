"""Streaming inference wrappers used by the competition scorer.

The competition's ``ScorerStepByStep`` iterates a parquet row by row and
calls ``model.predict(data_point)``. Our wrappers implement that
interface on top of either a trained ``SequenceModel`` (DL) or a
``ClassicalModel`` (sklearn / LightGBM).

A few invariants:
  * state must reset on every new ``seq_ix``,
  * predictions must be ``None`` when ``need_prediction == False``,
  * predictions must be a numpy array of shape ``(2,)`` otherwise.

The classical wrapper maintains a small in-memory buffer of the most
recent rows of the current sequence so it can recompute engineered
features (rolling momentum, etc.) without re-reading anything.
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from data.constants import FEATURE_COLS, N_TARGETS, WARMUP_STEPS
from data.scaling import FeatureScaler


# ---------------------------------------------------------------------------
# DL wrapper
# ---------------------------------------------------------------------------


class SequencePredictionModel:
    """Wrap a trained ``SequenceModel`` for the row-by-row scorer."""

    def __init__(
        self,
        model: Any,
        *,
        scaler: FeatureScaler | None = None,
        device: str = "cpu",
    ):
        self.model = model
        self.scaler = scaler
        self.device = torch.device(device)
        self.model.to(self.device).eval()
        self._reset()

    def _reset(self) -> None:
        self.current_seq_ix: int | None = None
        self.state = None

    def _maybe_scale(self, state_vector: np.ndarray) -> np.ndarray:
        if self.scaler is None or self.scaler.center_ is None:
            return state_vector.astype(np.float32, copy=False)
        v = state_vector.astype(np.float32, copy=False)
        center = self.scaler.center_.astype(np.float32, copy=False)
        scale = self.scaler.scale_.astype(np.float32, copy=False)
        return (v - center) / scale

    @torch.no_grad()
    def predict(self, data_point: Any) -> np.ndarray | None:
        if self.current_seq_ix != data_point.seq_ix:
            self.current_seq_ix = data_point.seq_ix
            self.state = self.model.init_state(batch_size=1, device=self.device)

        feat = self._maybe_scale(np.asarray(data_point.state, dtype=np.float32))
        feat_t = torch.from_numpy(feat).unsqueeze(0).to(self.device)
        pred_t, self.state = self.model.step(feat_t, self.state)

        if not bool(data_point.need_prediction):
            return None

        pred = pred_t.detach().cpu().numpy().reshape(-1)
        return np.clip(pred, -6.0, 6.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Classical wrapper
# ---------------------------------------------------------------------------


class ClassicalPredictionModel:
    """Wrap a ``ClassicalModel`` + tabular feature pipeline for streaming.

    Maintains a per-sequence rolling buffer (one ``pd.DataFrame`` per
    sequence) so the engineered-feature functions in
    ``feature_engineering`` can be called incrementally. Memory is
    bounded by the longest rolling window we use (default 20 steps), so
    the buffer is trimmed once the warm-up is complete.
    """

    def __init__(
        self,
        model: Any,
        feature_columns: list[str],
        *,
        with_engineered: bool = True,
        rolling_windows: tuple[int, ...] = (5, 20),
        buffer_size: int | None = None,
    ):
        self.model = model
        self.feature_columns = list(feature_columns)
        self.with_engineered = with_engineered
        self.rolling_windows = tuple(rolling_windows)
        self.buffer_size = buffer_size or max(
            max(self.rolling_windows or (1,)), WARMUP_STEPS
        )
        self._reset()

    def _reset(self) -> None:
        self.current_seq_ix: int | None = None
        self.buffer: deque[dict] = deque(maxlen=self.buffer_size)

    def _row_to_dict(self, data_point: Any) -> dict:
        state = np.asarray(data_point.state, dtype=np.float32)
        row = {col: float(state[i]) for i, col in enumerate(FEATURE_COLS)}
        row["seq_ix"] = int(data_point.seq_ix)
        row["step_in_seq"] = int(data_point.step_in_seq)
        return row

    def _build_features_for_last_row(self) -> np.ndarray:
        buf_df = pd.DataFrame(list(self.buffer))
        # Lazy import to avoid pulling feature_engineering at import time.
        from feature_engineering import add_engineered_features

        if self.with_engineered:
            buf_df = add_engineered_features(buf_df)
        else:
            buf_df["mid_price"] = (buf_df["p0"] + buf_df["p6"]) / 2

        # Match the rolling features added at training time.
        total_vol_cols = [f"v{i}" for i in range(12)]
        buf_df["_total_volume"] = buf_df[total_vol_cols].sum(axis=1)
        for w in self.rolling_windows:
            if w <= 1:
                continue
            buf_df[f"mid_rmean_{w}"] = (
                buf_df["mid_price"].rolling(w, min_periods=1).mean()
            )
            buf_df[f"mid_rstd_{w}"] = (
                buf_df["mid_price"].rolling(w, min_periods=2).std()
            )
            buf_df[f"vol_rmean_{w}"] = (
                buf_df["_total_volume"].rolling(w, min_periods=1).mean()
            )
        buf_df.drop(columns=["_total_volume"], inplace=True)

        last = buf_df.iloc[[-1]][self.feature_columns]
        # Fill any NaNs (rolling stats during warm-up) with zeros — should
        # never matter because by step 99+ all features are warm.
        return last.fillna(0.0).to_numpy(dtype=np.float32, copy=False)

    def predict(self, data_point: Any) -> np.ndarray | None:
        if self.current_seq_ix != data_point.seq_ix:
            self._reset()
            self.current_seq_ix = data_point.seq_ix
        self.buffer.append(self._row_to_dict(data_point))

        if not bool(data_point.need_prediction):
            return None

        X = self._build_features_for_last_row()
        pred = self.model.predict(X).reshape(-1)
        return np.clip(pred, -6.0, 6.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Ensemble wrapper
# ---------------------------------------------------------------------------


class EnsemblePredictionModel:
    """Average the predictions of several member wrappers.

    Each member is itself a ``SequencePredictionModel`` or
    ``ClassicalPredictionModel`` (anything exposing ``predict(data_point) ->
    (2,) | None``). The crucial invariant: every member must be ``predict``-ed
    on *every* row — including warm-up rows where it returns ``None`` — so its
    internal streaming state (SSM/RNN state, rolling buffer) stays in sync with
    the scorer. We therefore always call every member, then only combine the
    outputs on scored rows.

    Weights are normalised to sum to 1; the weighted average is clipped to the
    scorer's ``[-6, 6]`` range (each member already clips, so this is just a
    safety net).
    """

    def __init__(self, members: list[Any], weights: list[float] | None = None):
        if not members:
            raise ValueError("EnsemblePredictionModel needs at least one member.")
        self.members = list(members)
        n = len(self.members)
        w = np.ones(n, dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
        if w.shape != (n,):
            raise ValueError(f"weights must have length {n}, got {w.shape}")
        s = w.sum()
        self.weights = (w / s) if s > 0 else np.full(n, 1.0 / n)

    def predict(self, data_point: Any) -> np.ndarray | None:
        # Advance EVERY member's state, always (warm-up included).
        outs = [m.predict(data_point) for m in self.members]
        if not bool(data_point.need_prediction):
            return None
        acc = np.zeros(N_TARGETS, dtype=np.float64)
        wsum = 0.0
        for w, o in zip(self.weights, outs):
            if o is None:  # defensive: a member declined a scored row
                continue
            acc += w * np.asarray(o, dtype=np.float64).reshape(-1)
            wsum += w
        if wsum <= 0:
            return None
        return np.clip(acc / wsum, -6.0, 6.0).astype(np.float32)


def load_member(member_dir: str | Path) -> Any:
    """Reconstruct a single member wrapper from a packaged ``members/<i>/`` dir.

    The dir mirrors a single-model package: ``meta.json`` (kind/model_name/...),
    the artefact (``model.pt`` or ``model.joblib``), and ``scaler.json`` for
    sequence members. Used by the auto-generated ensemble ``solution.py``.
    """
    member_dir = Path(member_dir)
    with (member_dir / "meta.json").open() as fh:
        meta = json.load(fh)
    kind = meta["kind"]

    from models import get_model_class  # local: ensures registry is populated

    if kind == "sequence":
        ModelCls, _ = get_model_class(meta["model_name"])
        ckpt = torch.load(member_dir / "model.pt", map_location="cpu", weights_only=False)
        model = ModelCls(ckpt["config"])
        model.load_state_dict(ckpt["state_dict"])
        scaler = None
        scaler_path = member_dir / "scaler.json"
        if scaler_path.exists():
            with scaler_path.open() as fh:
                scaler = FeatureScaler.from_state_dict(json.load(fh))
        return SequencePredictionModel(model, scaler=scaler, device="cpu")

    if kind == "classical":
        import joblib

        ModelCls, _ = get_model_class(meta["model_name"])
        blob = joblib.load(member_dir / "model.joblib")
        model = ModelCls(blob["config"])
        model.estimators_ = blob["estimators"]
        model.feature_names_ = blob["feature_names"]
        model.target_names_ = blob["target_names"]
        return ClassicalPredictionModel(
            model,
            feature_columns=meta["feature_columns"],
            with_engineered=meta.get("with_engineered", True),
            rolling_windows=tuple(meta.get("rolling_windows", (5, 20))),
        )

    raise ValueError(f"unknown member kind: {kind!r}")
