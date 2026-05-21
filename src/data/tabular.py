"""Tabular features for classical / gradient-boosting models.

Classical models do not understand sequences natively, so we flatten each
step into a feature vector and let the model see one row per step. Two
things make this work for LOB data:

  1. We add engineered features (microstructure + dynamics) from
     `feature_engineering.add_engineered_features`. Rolling features
     that depend on previous steps respect sequence boundaries through
     the same `_lagged_change` helper, so no leakage across `seq_ix`.

  2. We add multi-lag rolling stats (rolling mean / std / momentum of
     mid-price and total volume) to let trees model temporal context
     without rolling at inference time.

The output is a `TabularDataset` (numpy arrays). Sample weights for the
weighted-Pearson objective are stored alongside so the trainer does not
need to recompute `|y|` repeatedly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from data.constants import FEATURE_COLS, TARGET_COLS, WARMUP_STEPS

# Imported lazily inside the function to keep this module import cost low.
# from feature_engineering import add_engineered_features


def _add_rolling_features(df: pd.DataFrame, windows: Iterable[int]) -> list[str]:
    """Add multi-window rolling mean / std of mid-price and total volume.

    Boundaries are respected via ``groupby("seq_ix")``. New columns are
    appended to ``df`` in place; their names are returned so the caller
    can include them in the feature list.
    """
    new_cols: list[str] = []
    grouped_mid = df.groupby("seq_ix", sort=False)["mid_price"]
    grouped_vol = df.groupby("seq_ix", sort=False)
    total_vol_cols = [f"v{i}" for i in range(12)]
    df["_total_volume"] = df[total_vol_cols].sum(axis=1)
    grouped_total = df.groupby("seq_ix", sort=False)["_total_volume"]

    for w in windows:
        if w <= 1:
            continue
        col_mid_mean = f"mid_rmean_{w}"
        col_mid_std = f"mid_rstd_{w}"
        col_vol_mean = f"vol_rmean_{w}"

        df[col_mid_mean] = grouped_mid.transform(
            lambda s, w=w: s.rolling(w, min_periods=1).mean()
        )
        df[col_mid_std] = grouped_mid.transform(
            lambda s, w=w: s.rolling(w, min_periods=2).std()
        )
        df[col_vol_mean] = grouped_total.transform(
            lambda s, w=w: s.rolling(w, min_periods=1).mean()
        )
        new_cols.extend([col_mid_mean, col_mid_std, col_vol_mean])

    df.drop(columns=["_total_volume"], inplace=True)
    return new_cols


def build_tabular_features(
    df: pd.DataFrame,
    *,
    with_engineered: bool = True,
    rolling_windows: tuple[int, ...] = (5, 20),
) -> tuple[pd.DataFrame, list[str]]:
    """Return a copy of ``df`` enriched with engineered + rolling features.

    Returns
    -------
    (out_df, feature_columns)
        ``out_df`` is the enriched dataframe (rows unchanged); ``feature_columns``
        is the ordered list of columns that classical models should consume.
    """
    out = df.copy()
    feature_columns: list[str] = list(FEATURE_COLS)

    if with_engineered:
        # Lazy import to avoid circulars and to keep the data package import
        # cheap when feature_engineering is not needed (e.g. pure DL flows).
        from feature_engineering import add_engineered_features, ENGINEERED_FEATURES
        out = add_engineered_features(out)
        feature_columns.extend(ENGINEERED_FEATURES)
    else:
        # Still need mid_price for the rolling features.
        out["mid_price"] = (out["p0"] + out["p6"]) / 2

    if rolling_windows:
        rolling_cols = _add_rolling_features(out, rolling_windows)
        feature_columns.extend(rolling_cols)

    # Drop the helper mid_price column from feature list if it was added
    # for rolling features only.
    if not with_engineered and "mid_price" in feature_columns:
        feature_columns.remove("mid_price")

    return out, feature_columns


@dataclass
class TabularDataset:
    X: np.ndarray            # (n_rows, n_features) float32
    y: np.ndarray            # (n_rows, n_targets)  float32
    weights: np.ndarray      # (n_rows,)            float32  — |y| (weighted-Pearson)
    seq_ids: np.ndarray      # (n_rows,)            int      — for grouped CV
    step_ids: np.ndarray     # (n_rows,)            int      — step within sequence
    feature_names: list[str]
    target_names: list[str]


def make_tabular_dataset(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_columns: list[str] = TARGET_COLS,
    *,
    only_scored_rows: bool = True,
    drop_warmup: bool = True,
) -> TabularDataset:
    """Slice the dataframe down to rows that have valid features + targets.

    Parameters
    ----------
    only_scored_rows
        Keep only rows where ``need_prediction == True`` if the column is
        present. This matches the evaluation contract.
    drop_warmup
        Additionally drop the first ``WARMUP_STEPS`` of each sequence
        (default: 99). For most engineered features the rolling stats are
        still warming up there.
    """
    rows = df
    if only_scored_rows and "need_prediction" in rows.columns:
        rows = rows[rows["need_prediction"]]
    if drop_warmup:
        rows = rows[rows["step_in_seq"] >= WARMUP_STEPS]
    if rows.empty:
        raise ValueError("No rows left after filtering for tabular dataset.")

    # Drop rows with NaN features (typically the very first few steps of
    # each sequence when lagged features have not warmed up yet).
    X_df = rows[feature_columns]
    valid_mask = X_df.notna().all(axis=1)
    rows = rows[valid_mask]
    X = rows[feature_columns].to_numpy(dtype=np.float32, copy=False)
    y = rows[target_columns].to_numpy(dtype=np.float32, copy=False)
    seq_ids = rows["seq_ix"].to_numpy(dtype=np.int64, copy=False)
    step_ids = rows["step_in_seq"].to_numpy(dtype=np.int64, copy=False)
    weights = np.abs(y).max(axis=1).astype(np.float32, copy=False)

    return TabularDataset(
        X=X,
        y=y,
        weights=weights,
        seq_ids=seq_ids,
        step_ids=step_ids,
        feature_names=list(feature_columns),
        target_names=list(target_columns),
    )
