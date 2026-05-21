"""Standardised on-disk report for an experiment run.

Layout produced by ``write_report``::

    experiments/<model>/<run_id>/
        config.yaml
        train_history.csv
        train_history.png
        val/
            metrics.json
            predictions.parquet
            scatter.png
            per_seq_corr.png
        meta.json

Every model writes the same layout so a notebook can compare them
without special-casing.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from evaluation.evaluator import EvalResult
from evaluation.plots import (
    plot_loss_curve,
    plot_pred_vs_target_scatter,
    plot_per_sequence_corr_hist,
)

REPORT_VERSION = 1


def make_run_id(model_name: str) -> str:
    return f"{model_name}__{time.strftime('%Y%m%d-%H%M%S')}"


def write_report(
    *,
    run_dir: str | Path,
    config: dict,
    history: list[dict] | None,
    val_eval: EvalResult | None,
    extra_meta: dict[str, Any] | None = None,
) -> Path:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # --- config dump ---
    with (run_dir / "config.yaml").open("w") as fh:
        yaml.safe_dump(config, fh, sort_keys=False)

    # --- training history ---
    if history:
        hist_df = pd.DataFrame(history)
        hist_df.to_csv(run_dir / "train_history.csv", index=False)
        plot_loss_curve(history, run_dir / "train_history.png")

    # --- validation block ---
    if val_eval is not None:
        val_dir = run_dir / "val"
        val_dir.mkdir(exist_ok=True)
        with (val_dir / "metrics.json").open("w") as fh:
            json.dump(_to_jsonable(val_eval.metrics), fh, indent=2)
        # raw predictions as a parquet for downstream notebooks
        pred_df = pd.DataFrame({
            "seq_ix": val_eval.seq_ids,
            "step_in_seq": val_eval.step_ids,
        })
        for j, name in enumerate(val_eval.metrics.get("per_target", {}).keys() or [f"t{j}" for j in range(val_eval.targets.shape[1])]):
            pred_df[f"true_{name}"] = val_eval.targets[:, j]
            pred_df[f"pred_{name}"] = val_eval.predictions[:, j]
        pred_df.to_parquet(val_dir / "predictions.parquet", index=False)
        plot_pred_vs_target_scatter(
            val_eval.targets, val_eval.predictions, val_dir / "scatter.png",
        )
        plot_per_sequence_corr_hist(
            val_eval.per_sequence_corr, val_dir / "per_seq_corr.png",
        )

    # --- meta ---
    meta = {
        "report_version": REPORT_VERSION,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        **(extra_meta or {}),
    }
    with (run_dir / "meta.json").open("w") as fh:
        json.dump(_to_jsonable(meta), fh, indent=2)

    return run_dir


def _to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
