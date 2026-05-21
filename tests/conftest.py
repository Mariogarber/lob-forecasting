"""Shared fixtures for the pipeline tests."""

from __future__ import annotations

# Force matplotlib to its non-interactive Agg backend BEFORE any test (or
# any tested module) imports pyplot. Done here so the production modules
# (e.g. evaluation.plots) can stay backend-agnostic and not contaminate
# notebooks or downstream scripts.
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from data.constants import FEATURE_COLS, SEQ_LEN, TARGET_COLS


@pytest.fixture
def tiny_lob_df() -> pd.DataFrame:
    """A small but structurally-correct LOB dataframe (4 sequences x 1000 steps)."""
    rng = np.random.default_rng(0)
    n_seq = 4
    rows = []
    for seq_id in range(n_seq):
        mid = 100 + rng.normal(0, 0.05, size=SEQ_LEN).cumsum()
        spread = np.clip(0.05 + 0.02 * rng.standard_normal(SEQ_LEN), 0.01, 0.5)
        bids = mid - spread / 2
        asks = mid + spread / 2
        for step in range(SEQ_LEN):
            row = {"seq_ix": seq_id, "step_in_seq": step}
            # bid prices p0..p5 descending from best bid
            for i in range(6):
                row[f"p{i}"] = float(bids[step] - i * 0.01)
            # ask prices p6..p11 ascending from best ask
            for i in range(6):
                row[f"p{i+6}"] = float(asks[step] + i * 0.01)
            # volumes
            for i in range(12):
                row[f"v{i}"] = float(abs(rng.normal(100, 30)))
            # trade prices/volumes
            for i in range(4):
                row[f"dp{i}"] = float(mid[step] + rng.normal(0, 0.02))
                row[f"dv{i}"] = float(abs(rng.normal(5, 2)))
            row["t0"] = float(rng.normal(0, 1))
            row["t1"] = float(rng.normal(0, 1))
            row["need_prediction"] = step >= 99
            rows.append(row)
    df = pd.DataFrame(rows)
    cols = ["seq_ix", "step_in_seq", "need_prediction"] + FEATURE_COLS + TARGET_COLS
    return df[cols]


@pytest.fixture
def tiny_features_targets() -> tuple[np.ndarray, np.ndarray]:
    """Quick (N, K) arrays for testing losses / metrics in isolation."""
    rng = np.random.default_rng(0)
    y_true = rng.normal(0, 1, size=(1000, 2)).astype(np.float32)
    # Predictions are y_true plus mild noise — should give positive correlation.
    y_pred = y_true + rng.normal(0, 0.5, size=y_true.shape).astype(np.float32)
    return y_true, y_pred
