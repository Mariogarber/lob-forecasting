# src/profiling/relations.py
from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from profiling.constants import TARGETS
from profiling.io import _validate_columns


def correlation_matrix(
    df: pd.DataFrame,
    columns: Sequence[str],
    method: str = "pearson",
) -> pd.DataFrame:
    """Compute a pairwise correlation matrix for the given columns."""
    _validate_columns(df, list(columns))
    return df[list(columns)].corr(method=method)


def plot_correlation_heatmap(
    corr_df: pd.DataFrame,
    mask_diagonal: bool = True,
    annotate: bool = True,
) -> plt.Figure:
    """Seaborn heatmap of a correlation matrix."""
    mask = np.eye(len(corr_df), dtype=bool) if mask_diagonal else None
    fig, ax = plt.subplots(figsize=(max(6, len(corr_df) * 0.7),
                                    max(5, len(corr_df) * 0.65)))
    sns.heatmap(
        corr_df,
        ax=ax,
        mask=mask,
        annot=annotate,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.3,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Correlation matrix")
    fig.tight_layout()
    return fig


def feature_target_correlations(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_cols: Sequence[str] = TARGETS,
    method: str = "pearson",
) -> pd.DataFrame:
    """Return a DataFrame of shape (n_features, n_targets) with correlation values."""
    _validate_columns(df, list(feature_cols) + list(target_cols))
    results = {}
    for target in target_cols:
        results[target] = [
            df[feat].corr(df[target], method=method) for feat in feature_cols
        ]
    return pd.DataFrame(results, index=list(feature_cols))


def plot_feature_target_scatter(
    df: pd.DataFrame,
    feature: str,
    target: str,
    sample: int = 20_000,
    seed: int = 42,
) -> plt.Figure:
    """Scatter plot of one feature vs one target (sampled for speed)."""
    _validate_columns(df, [feature, target])
    df_s = df[[feature, target]].dropna().sample(
        min(sample, len(df)), random_state=seed
    )
    corr_val = df_s[feature].corr(df_s[target])
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(df_s[feature], df_s[target], alpha=0.15, s=4, rasterized=True)
    ax.set_xlabel(feature)
    ax.set_ylabel(target)
    ax.set_title(f"{feature} vs {target}  (r={corr_val:.3f})")
    fig.tight_layout()
    return fig
