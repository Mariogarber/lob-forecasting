"""Trainer for classical (sklearn / gradient-boosting) models.

Thin wrapper around ``ClassicalModel.fit`` that:

  * builds tabular features from the train and val dataframes,
  * passes per-sample weights = ``|y|`` so the loss is aligned with
    weighted Pearson,
  * scores the validation set with ``metrics.summary`` and returns the
    full diagnostic dict alongside the trained estimator.

The function intentionally keeps the inputs as pandas DataFrames (rather
than the ``TabularDataset`` dataclass) so the caller has flexibility in
choosing the feature set per model.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from rich.console import Console

from data import build_tabular_features, make_tabular_dataset
from metrics import summary
from models.base import ClassicalModel

CONSOLE = Console()


@dataclass
class ClassicalTrainerConfig:
    with_engineered: bool = True
    rolling_windows: tuple[int, ...] = (5, 20)
    only_scored_rows: bool = True
    drop_warmup: bool = True
    log: bool = True


def train_classical_model(
    model: ClassicalModel,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None,
    config: ClassicalTrainerConfig | None = None,
) -> dict[str, Any]:
    config = config or ClassicalTrainerConfig()
    if config.log:
        CONSOLE.print(f"[bold]Training classical model[/]: {model.name}")
        t0 = time.time()
        CONSOLE.print("  -> building tabular features (train)...")

    train_full, feature_cols = build_tabular_features(
        train_df,
        with_engineered=config.with_engineered,
        rolling_windows=config.rolling_windows,
    )
    train_ds = make_tabular_dataset(
        train_full,
        feature_columns=feature_cols,
        only_scored_rows=config.only_scored_rows,
        drop_warmup=config.drop_warmup,
    )

    eval_set = None
    val_ds = None
    if val_df is not None:
        if config.log:
            CONSOLE.print("  -> building tabular features (val)...")
        val_full, _ = build_tabular_features(
            val_df,
            with_engineered=config.with_engineered,
            rolling_windows=config.rolling_windows,
        )
        val_ds = make_tabular_dataset(
            val_full,
            feature_columns=feature_cols,
            only_scored_rows=config.only_scored_rows,
            drop_warmup=config.drop_warmup,
        )
        eval_set = (val_ds.X, val_ds.y)

    if config.log:
        CONSOLE.print(
            f"  -> shapes: train X={train_ds.X.shape} y={train_ds.y.shape}"
            + (f", val X={val_ds.X.shape}" if val_ds is not None else "")
        )
        CONSOLE.print(f"  -> fitting (n_features={len(feature_cols)})...")

    fit_t0 = time.time()
    model.fit(
        train_ds.X,
        train_ds.y,
        sample_weight=train_ds.weights,
        eval_set=eval_set,
        feature_names=train_ds.feature_names,
        target_names=train_ds.target_names,
    )
    fit_elapsed = time.time() - fit_t0

    train_preds = model.predict(train_ds.X)
    train_metrics = summary(train_ds.y, train_preds)
    val_metrics = None
    if val_ds is not None:
        val_preds = model.predict(val_ds.X)
        val_metrics = summary(val_ds.y, val_preds)

    if config.log:
        CONSOLE.print(
            f"  -> done in {fit_elapsed:.1f}s; "
            f"train_corr={train_metrics['weighted_pearson']:+.4f}"
            + (
                f"  val_corr={val_metrics['weighted_pearson']:+.4f}"
                if val_metrics
                else ""
            )
        )

    return {
        "model": model,
        "feature_columns": feature_cols,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "fit_seconds": fit_elapsed,
    }
