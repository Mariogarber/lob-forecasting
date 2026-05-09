# src/profiling/temporal.py
from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from profiling.io import _validate_columns
from profiling.plots import grid_axes


_VALID_AGGS = {"mean", "std", "min", "max", "range", "skew", "kurt"}


def plot_feature_trajectories(
    df: pd.DataFrame,
    columns: Sequence[str],
    seq_ixs: Sequence[int],
) -> plt.Figure:
    """Plot time series of columns for each sequence. One subplot per sequence."""
    _validate_columns(df, list(columns) + ["seq_ix", "step_in_seq"])
    n = len(seq_ixs)
    fig, axes = grid_axes(n, ncols=2, figsize_per_panel=(9.0, 3.5))
    for ax, seq_ix in zip(axes, seq_ixs):
        seq_data = df[df["seq_ix"] == seq_ix].sort_values("step_in_seq")
        for col in columns:
            ax.plot(seq_data["step_in_seq"], seq_data[col], label=col, linewidth=0.8)
        ax.set_title(f"seq_ix={seq_ix}")
        ax.set_xlabel("step_in_seq")
        ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    return fig


def autocorrelation(
    df: pd.DataFrame,
    column: str,
    max_lag: int = 100,
) -> pd.DataFrame:
    """Compute autocorrelation averaged across all sequences.

    Returns DataFrame with columns: lag, mean_autocorr, ci_low, ci_high.
    ci_low/ci_high are +/-1.96 standard errors of the mean across sequences.
    """
    _validate_columns(df, [column, "seq_ix"])

    def _acf(vals: np.ndarray) -> np.ndarray:
        n = len(vals)
        x = vals - vals.mean()
        variance = (x ** 2).mean()
        if variance < 1e-10:
            return np.zeros(max_lag + 1)
        full = np.correlate(x, x, mode="full")
        acf = full[n - 1 : n + max_lag] / (n * variance)
        acf[0] = 1.0  # clamp lag-0 to exactly 1
        return acf

    acfs = np.stack(
        df.groupby("seq_ix")[column]
        .apply(lambda s: _acf(s.to_numpy()))
        .to_numpy()
    )  # shape: (n_seqs, max_lag + 1)

    mean_acf = acfs.mean(axis=0)
    sem = acfs.std(axis=0) / np.sqrt(acfs.shape[0])

    return pd.DataFrame({
        "lag": np.arange(max_lag + 1),
        "mean_autocorr": mean_acf,
        "ci_low":  mean_acf - 1.96 * sem,
        "ci_high": mean_acf + 1.96 * sem,
    })


def plot_autocorrelation(acf_df: pd.DataFrame, column_name: str) -> plt.Figure:
    """Line plot of mean autocorrelation with +/-1.96 SE band."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(acf_df["lag"], acf_df["mean_autocorr"], color="steelblue", linewidth=1.5)
    ax.fill_between(
        acf_df["lag"], acf_df["ci_low"], acf_df["ci_high"],
        alpha=0.25, color="steelblue", label="+-1.96 SE"
    )
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("lag")
    ax.set_ylabel("mean autocorrelation")
    ax.set_title(f"Autocorrelation - {column_name}")
    ax.legend()
    fig.tight_layout()
    return fig


def per_sequence_stat(
    df: pd.DataFrame,
    column: str,
    agg: str = "mean",
) -> pd.Series:
    """Compute one summary statistic per sequence. agg in {mean, std, min, max, range, skew, kurt}."""
    if agg not in _VALID_AGGS:
        raise ValueError(f"agg must be one of {_VALID_AGGS}, got '{agg}'")
    _validate_columns(df, [column, "seq_ix"])
    g = df.groupby("seq_ix")[column]
    if agg == "range":
        return g.apply(lambda s: s.max() - s.min()).rename(f"{column}_{agg}")
    if agg == "kurt":
        return g.apply(pd.Series.kurtosis).rename(f"{column}_{agg}")
    return g.agg(agg).rename(f"{column}_{agg}")


def plot_per_sequence_stat_distribution(
    stat_series: pd.Series,
    column_name: str,
    agg_name: str,
) -> plt.Figure:
    """Histogram of a per-sequence summary stat across all sequences."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(stat_series.dropna(), bins=60, edgecolor="none", alpha=0.85,
            color="darkorange")
    ax.set_xlabel(f"{agg_name}({column_name}) per sequence")
    ax.set_ylabel("number of sequences")
    ax.set_title(
        f"Distribution of per-sequence {agg_name}({column_name})  "
        f"[n={stat_series.notna().sum()} seqs]"
    )
    fig.tight_layout()
    return fig
