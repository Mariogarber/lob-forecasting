"""High-level run orchestrator.

A ``run`` is a single (config, model) pair: load data, build the model,
train, evaluate, dump a standardised report + a submission zip. This is
the function the CLI calls; it is also fully usable from a notebook.

Two dispatch paths share most plumbing:

  * sequence models -> ``Trainer`` + ``Evaluator.run_sequence_model``
  * classical models -> ``train_classical_model`` + ``run_classical_model``
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

# IMPORTANT: pandas/pyarrow must be imported BEFORE torch on Windows to
# avoid an MSVC runtime DLL conflict that segfaults pq.read_table().
# Keep `from data import (...)` ahead of any torch import.
from data import (
    FeatureScaler,
    load_train_valid,
    sequences_from_dataframe,
    split_by_seq_ix,
    SequenceDataset,
)
from data.constants import FEATURE_COLS, N_FEATURES, N_TARGETS

import torch
from rich.console import Console

from evaluation import Evaluator, write_report, make_run_id
from models import get_model_class
from pipeline.config import RunConfig
from submission import package_run
from training import TrainerConfig, train_classical_model, train_sequence_model
from training.classical_trainer import ClassicalTrainerConfig

CONSOLE = Console()

EXPERIMENTS_ROOT = Path("experiments")


def _materialise_data(cfg: RunConfig):
    """Load train + val parquets and apply the optional split / scaler."""
    train_path = cfg.data.get("train_path")
    valid_path = cfg.data.get("valid_path")
    train_df, valid_df = load_train_valid(train_path, valid_path)

    val_fraction = float(cfg.data.get("val_fraction", 0.0))
    if val_fraction > 0:
        # Re-derive a val split *from train* and ignore the official valid file.
        train_df, valid_df = split_by_seq_ix(
            train_df, val_fraction=val_fraction, seed=int(cfg.data.get("seed", 0))
        )

    scaler_mode = cfg.data.get("scaling", "standard")
    scaler = FeatureScaler(mode=scaler_mode).fit(train_df, FEATURE_COLS)
    train_df = scaler.transform(train_df)
    valid_df = scaler.transform(valid_df)
    return train_df, valid_df, scaler


# ---------------------------------------------------------------------------
# Sequence path
# ---------------------------------------------------------------------------


def _run_sequence(cfg: RunConfig, model_name: str) -> dict[str, Any]:
    CONSOLE.print(f"[bold green]>>> SEQUENCE RUN[/]: {model_name}")
    train_df, valid_df, scaler = _materialise_data(cfg)

    # Build datasets. Augmentation (random crop + feature jitter) is applied to
    # the TRAIN set only — the val set must mirror the scorer (full sequences,
    # no noise) so its weighted-Pearson stays comparable across runs.
    aug = dict(cfg.data.get("augment", {}) or {})
    train_arr = sequences_from_dataframe(train_df)
    valid_arr = sequences_from_dataframe(valid_df)
    train_ds = SequenceDataset(
        train_arr,
        crop_len=aug.get("crop_len"),
        jitter_std=float(aug.get("jitter_std", 0.0)),
        seed=int(cfg.data.get("seed", 0)),
    )
    val_ds = SequenceDataset(valid_arr)
    CONSOLE.print(
        f"  train: {len(train_ds)} sequences | val: {len(val_ds)} sequences"
        + (f" | augment: {aug}" if train_ds.augmented else " | augment: off")
    )

    # Build model.
    ModelCls, _kind = get_model_class(model_name)
    model_cfg = {
        **cfg.model.get("params", {}),
        "n_features": N_FEATURES,
        "n_targets": N_TARGETS,
    }
    model = ModelCls(model_cfg)  # type: ignore[call-arg]

    # Train.
    trainer_cfg = TrainerConfig(**cfg.training)
    train_info = train_sequence_model(model, train_ds, val_ds, trainer_cfg)

    # Evaluate (batched, fast).
    ev = Evaluator()
    val_eval = ev.run_sequence_model(model, valid_df, device=trainer_cfg.resolve_device())
    CONSOLE.print(
        f"  val weighted_pearson = {val_eval.metrics['weighted_pearson']:+.4f}"
    )

    # Write report + save model artefact.
    run_id = make_run_id(model_name)
    run_dir = EXPERIMENTS_ROOT / model_name / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    model_path = run_dir / "model.pt"
    model.save(model_path)

    write_report(
        run_dir=run_dir,
        config=cfg.raw,
        history=train_info["history"],
        val_eval=val_eval,
        extra_meta={
            "kind": "sequence",
            "model_name": model_name,
            "best_val_pearson": float(train_info["best_score"]),
        },
    )

    # Build submission zip.
    utils_py = Path("competition_package/utils.py")
    if utils_py.exists():
        zip_path = package_run(
            run_dir=run_dir,
            src_root=Path("src"),
            utils_py=utils_py,
            model_kind="sequence",
            model_name=model_name,
            model_artifact=model_path,
            scaler_state=scaler.state_dict(),
        )
        CONSOLE.print(f"  submission zip: {zip_path}")

    return {
        "run_dir": str(run_dir),
        "val_pearson": float(val_eval.metrics["weighted_pearson"]),
    }


# ---------------------------------------------------------------------------
# Classical path
# ---------------------------------------------------------------------------


def _run_classical(cfg: RunConfig, model_name: str) -> dict[str, Any]:
    CONSOLE.print(f"[bold green]>>> CLASSICAL RUN[/]: {model_name}")
    train_df, valid_df, _scaler = _materialise_data(cfg)

    ModelCls, _kind = get_model_class(model_name)
    model = ModelCls(cfg.model.get("params", {}))

    classical_cfg = ClassicalTrainerConfig(**cfg.classical)
    result = train_classical_model(model, train_df, valid_df, classical_cfg)

    # Evaluate via the Evaluator (re-runs predict, gives per-sequence stats).
    ev = Evaluator()
    val_eval = ev.run_classical_model(
        result["model"],
        valid_df,
        with_engineered=classical_cfg.with_engineered,
        rolling_windows=classical_cfg.rolling_windows,
    )
    CONSOLE.print(
        f"  val weighted_pearson = {val_eval.metrics['weighted_pearson']:+.4f}"
    )

    run_id = make_run_id(model_name)
    run_dir = EXPERIMENTS_ROOT / model_name / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    model_path = run_dir / "model.joblib"
    result["model"].save(model_path)

    write_report(
        run_dir=run_dir,
        config=cfg.raw,
        history=None,
        val_eval=val_eval,
        extra_meta={
            "kind": "classical",
            "model_name": model_name,
            "feature_columns": result["feature_columns"],
            "fit_seconds": result["fit_seconds"],
            "train_metrics": result["train_metrics"],
            "val_metrics": result["val_metrics"],
        },
    )

    utils_py = Path("competition_package/utils.py")
    if utils_py.exists():
        zip_path = package_run(
            run_dir=run_dir,
            src_root=Path("src"),
            utils_py=utils_py,
            model_kind="classical",
            model_name=model_name,
            model_artifact=model_path,
            feature_columns=result["feature_columns"],
            with_engineered=classical_cfg.with_engineered,
            rolling_windows=classical_cfg.rolling_windows,
        )
        CONSOLE.print(f"  submission zip: {zip_path}")

    return {
        "run_dir": str(run_dir),
        "val_pearson": float(val_eval.metrics["weighted_pearson"]),
    }


# ---------------------------------------------------------------------------
# Public entry point.
# ---------------------------------------------------------------------------


def run_from_config(config_path: str | Path) -> dict[str, Any]:
    from pipeline.config import load_config
    cfg = load_config(config_path)
    model_name = cfg.model.get("name")
    if not model_name:
        raise ValueError("config must specify model.name")
    _cls, kind = get_model_class(model_name)
    if kind == "sequence":
        return _run_sequence(cfg, model_name)
    return _run_classical(cfg, model_name)
