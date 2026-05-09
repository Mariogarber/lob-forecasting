# src/profiling/descriptive.py
from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from profiling.constants import TARGETS
from profiling.io import _validate_columns
from profiling.plots import grid_axes


def feature_stats(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Return descriptive statistics for each column as a DataFrame (columns as index)."""
    _validate_columns(df, list(columns))
    rows = []
    for col in columns:
        s = df[col]
        rows.append({
            "column": col,
            "mean":    s.mean(),
            "std":     s.std(),
            "min":     s.min(),
            "p1":      s.quantile(0.01),
            "p25":     s.quantile(0.25),
            "p50":     s.quantile(0.50),
            "p75":     s.quantile(0.75),
            "p99":     s.quantile(0.99),
            "max":     s.max(),
            "skew":    s.skew(),
            "kurt":    s.kurtosis(),
            "n_zeros": int((s == 0).sum()),
            "n_nan":   int(s.isna().sum()),
        })
    return pd.DataFrame(rows).set_index("column")


def plot_distributions(
    df: pd.DataFrame,
    columns: Sequence[str],
    sample: int = 200_000,
    bins: int = 100,
    seed: int = 42,
) -> plt.Figure:
    """Histogram grid — one subplot per column. Samples rows for speed."""
    _validate_columns(df, list(columns))
    n = len(columns)
    df_sample = df.sample(min(sample, len(df)), random_state=seed)
    fig, axes = grid_axes(n, ncols=4, figsize_per_panel=(5.0, 3.0))
    for ax, col in zip(axes, columns):
        ax.hist(df_sample[col].dropna(), bins=bins, edgecolor="none", alpha=0.8)
        ax.set_title(col)
        ax.set_xlabel("value")
        ax.set_ylabel("count")
    fig.tight_layout()
    return fig


def plot_target_distributions(
    df: pd.DataFrame,
    target_cols: Sequence[str] = TARGETS,
) -> plt.Figure:
    """Histogram for each target with percentile lines and metric clip range marked."""
    _validate_columns(df, list(target_cols))
    fig, axes = grid_axes(len(target_cols), ncols=len(target_cols),
                          figsize_per_panel=(7.0, 4.5))
    for ax, col in zip(axes, target_cols):
        s = df[col].dropna()
        ax.hist(s, bins=150, edgecolor="none", alpha=0.8, color="steelblue")
        for q, style in [(0.01, "--"), (0.50, "-"), (0.99, "--")]:
            val = s.quantile(q)
            ax.axvline(val, color="black", linestyle=style, linewidth=1.0,
                       label=f"p{int(q*100)}={val:.2f}")
        for clip_val in (-6, 6):
            ax.axvline(clip_val, color="crimson", linestyle=":", linewidth=1.5,
                       label=f"clip={clip_val}")
        ax.set_title(col)
        ax.set_xlabel("value")
        ax.set_ylabel("count")
        ax.legend(fontsize=8)
    fig.tight_layout()
    return fig
