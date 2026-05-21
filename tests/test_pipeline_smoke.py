"""End-to-end pipeline smoke tests.

These exercise the same code path as the CLI ``train`` command, but on
the tiny synthetic dataframe so they finish in a few seconds. They check
that:

  * a classical model can be trained, evaluated and packaged,
  * a sequence model (GRU) can be trained for a single epoch, evaluated
    in batched mode AND in streaming mode, and the two agree.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from data import (
    FeatureScaler,
    SequenceDataset,
    sequences_from_dataframe,
    split_by_seq_ix,
)
from data.constants import FEATURE_COLS, N_FEATURES, N_TARGETS
from evaluation import Evaluator
from evaluation.evaluator import predict_streaming
from models import get_model_class
from training import TrainerConfig, train_classical_model, train_sequence_model
from training.classical_trainer import ClassicalTrainerConfig


@pytest.fixture
def split_dfs(tiny_lob_df):
    train_df, val_df = split_by_seq_ix(tiny_lob_df, val_fraction=0.25, seed=0)
    return train_df, val_df


def test_linear_end_to_end(split_dfs):
    train_df, val_df = split_dfs
    cls, _ = get_model_class("linear")
    model = cls({"fit_intercept": True, "n_jobs": 1})
    classical_cfg = ClassicalTrainerConfig(rolling_windows=(5,))
    result = train_classical_model(model, train_df, val_df, classical_cfg)
    assert result["val_metrics"] is not None
    assert isinstance(result["val_metrics"]["weighted_pearson"], float)
    # Evaluator runs end-to-end without crashing.
    ev = Evaluator()
    val_eval = ev.run_classical_model(
        result["model"], val_df, rolling_windows=classical_cfg.rolling_windows
    )
    assert val_eval.predictions.shape == val_eval.targets.shape
    assert val_eval.predictions.shape[1] == 2


def test_lightgbm_end_to_end(split_dfs):
    train_df, val_df = split_dfs
    cls, _ = get_model_class("lightgbm")
    model = cls({
        "params": {
            "num_leaves": 7, "min_data_in_leaf": 2, "learning_rate": 0.1,
            "verbose": -1,
        },
        "num_boost_round": 20,
        "early_stopping_rounds": 5,
        "log_period": 1000,  # silence
    })
    cfg = ClassicalTrainerConfig(rolling_windows=(5,))
    cfg.log = False  # quiet test
    result = train_classical_model(model, train_df, val_df, cfg)
    assert result["val_metrics"] is not None


def test_gru_end_to_end(split_dfs):
    train_df, val_df = split_dfs
    cls, _ = get_model_class("gru")
    model = cls({
        "n_features": N_FEATURES,
        "n_targets": N_TARGETS,
        "hidden_size": 16,
        "num_layers": 1,
        "dropout": 0.0,
    })
    train_arr = sequences_from_dataframe(train_df)
    val_arr = sequences_from_dataframe(val_df)
    train_ds = SequenceDataset(train_arr)
    val_ds = SequenceDataset(val_arr)
    trainer_cfg = TrainerConfig(
        epochs=1, batch_size=2, learning_rate=1e-3,
        device="cpu", amp=False, warmup_epochs=0,
        eval_every=1, early_stopping_patience=0, num_workers=0,
    )
    info = train_sequence_model(model, train_ds, val_ds, trainer_cfg)
    assert info["history"]
    # Streaming inference must match batched inference up to floating-point eps.
    ev = Evaluator()
    batched = ev.run_sequence_model(model, val_df, device="cpu")
    streamed = predict_streaming(model, val_df, device="cpu")
    np.testing.assert_allclose(
        streamed.predictions, batched.predictions, atol=1e-4, rtol=1e-3,
    )


def test_submission_packager_with_linear(split_dfs, tmp_path):
    """Verify that package_run produces a valid solution.zip that contains
    every expected artefact."""
    from submission.packager import package_run

    train_df, val_df = split_dfs
    cls, _ = get_model_class("linear")
    model = cls({"fit_intercept": True})
    cfg = ClassicalTrainerConfig(rolling_windows=(5,))
    cfg.log = False
    result = train_classical_model(model, train_df, val_df, cfg)

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    model_path = run_dir / "model.joblib"
    result["model"].save(model_path)

    # Use the project's competition_package utils.py if present, else fabricate.
    utils_py = Path("competition_package/utils.py")
    if not utils_py.exists():
        pytest.skip("competition_package/utils.py not present")

    zip_path = package_run(
        run_dir=run_dir,
        src_root=Path("src"),
        utils_py=utils_py,
        model_kind="classical",
        model_name="linear",
        model_artifact=model_path,
        feature_columns=result["feature_columns"],
        with_engineered=True,
        rolling_windows=(5,),
    )
    assert zip_path.exists()

    # Spot check: zip must contain solution.py + utils.py + model.joblib.
    import zipfile
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
    assert "solution.py" in names
    assert "utils.py" in names
    assert "model.joblib" in names
    assert "meta.json" in names
    assert any(n.startswith("lob_runtime/") for n in names)
