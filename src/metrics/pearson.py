"""Weighted Pearson + helper diagnostic metrics.

The competition uses the weighted-Pearson correlation per target with
weights = ``|y_true|`` and predictions clipped to ``[-6, 6]``. We mirror
that exactly so train-time validation numbers match the official scorer.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

PRED_CLIP = 6.0
EPS = 1e-8


def _weighted_pearson_scalar(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_pred = np.clip(y_pred, -PRED_CLIP, PRED_CLIP)
    w = np.maximum(np.abs(y_true), EPS)
    sum_w = w.sum()
    if sum_w <= 0:
        return 0.0
    mean_true = (y_true * w).sum() / sum_w
    mean_pred = (y_pred * w).sum() / sum_w
    dt = y_true - mean_true
    dp = y_pred - mean_pred
    cov = (w * dt * dp).sum() / sum_w
    var_t = (w * dt * dt).sum() / sum_w
    var_p = (w * dp * dp).sum() / sum_w
    if var_t <= 0 or var_p <= 0:
        return 0.0
    return float(cov / (np.sqrt(var_t) * np.sqrt(var_p)))


def weighted_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean weighted-Pearson across both target columns (matches the scorer)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim == 1:
        return _weighted_pearson_scalar(y_true, y_pred)
    cols = y_true.shape[1]
    return float(np.mean([
        _weighted_pearson_scalar(y_true[:, i], y_pred[:, i]) for i in range(cols)
    ]))


def weighted_pearson_per_target(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Sequence[str] = ("t0", "t1"),
) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        name: _weighted_pearson_scalar(y_true[:, i], y_pred[:, i])
        for i, name in enumerate(target_names)
    }


def per_sequence_weighted_pearson(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    seq_ids: np.ndarray,
) -> dict[int, float]:
    """Weighted-Pearson per ``seq_ix`` (across both targets, averaged)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    seq_ids = np.asarray(seq_ids)
    out: dict[int, float] = {}
    for sid in np.unique(seq_ids):
        m = seq_ids == sid
        out[int(sid)] = weighted_pearson(y_true[m], y_pred[m])
    return out


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    err = np.asarray(y_true) - np.asarray(y_pred)
    return float((err * err).mean())


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.abs(np.asarray(y_true) - np.asarray(y_pred)).mean())


def summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Sequence[str] = ("t0", "t1"),
) -> dict:
    """One-shot diagnostic bundle for the evaluator."""
    per_target = weighted_pearson_per_target(y_true, y_pred, target_names)
    return {
        "weighted_pearson": float(np.mean(list(per_target.values()))),
        "per_target": per_target,
        "mse": mse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "n_samples": int(np.asarray(y_true).shape[0]),
    }
