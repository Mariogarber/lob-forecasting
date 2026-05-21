"""Evaluation runner.

Two main entry points:

  * ``Evaluator.run_sequence_model`` — feeds a trained ``SequenceModel``
    one full sequence at a time, predicts on every step, and aggregates
    metrics + per-sequence correlations.

  * ``Evaluator.run_classical_model`` — flat tabular eval on the masked
    rows of the validation parquet.

Both return a dict with the metric summary plus the raw predictions /
targets for downstream plotting. The on-disk report (JSON + plots) is
produced by ``evaluation.report.write_report`` (separate file).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import torch

from data import (
    SequenceDataset,
    build_tabular_features,
    make_tabular_dataset,
    sequences_from_dataframe,
)
from metrics import per_sequence_weighted_pearson, summary
from models.base import ClassicalModel, SequenceModel


@dataclass
class EvalResult:
    metrics: dict
    predictions: np.ndarray
    targets: np.ndarray
    seq_ids: np.ndarray
    step_ids: np.ndarray
    per_sequence_corr: dict[int, float] = field(default_factory=dict)


class Evaluator:
    """Stateless object that runs evaluation for any model kind."""

    # ------------------------------------------------------------------
    # Sequence models.
    # ------------------------------------------------------------------
    def run_sequence_model(
        self,
        model: SequenceModel,
        df: pd.DataFrame,
        *,
        device: torch.device | str = "cpu",
        batch_size: int = 32,
    ) -> EvalResult:
        device = torch.device(device)
        model.eval().to(device)

        arrays = sequences_from_dataframe(df)
        ds = SequenceDataset(arrays)

        preds_all: list[np.ndarray] = []
        targets_all: list[np.ndarray] = []
        seq_all: list[np.ndarray] = []
        step_all: list[np.ndarray] = []

        with torch.no_grad():
            for start in range(0, len(ds), batch_size):
                end = min(start + batch_size, len(ds))
                batch_feats = torch.stack(
                    [torch.from_numpy(arrays.features[i]) for i in range(start, end)]
                ).to(device)
                batch_targets = arrays.targets[start:end]   # (b, T, K)
                batch_mask = arrays.mask[start:end]         # (b, T)
                batch_seq_ids = arrays.seq_ids[start:end]   # (b,)

                preds = model(batch_feats).cpu().numpy()    # (b, T, K)

                for b in range(end - start):
                    m = batch_mask[b]
                    if m.any():
                        preds_all.append(preds[b][m])
                        targets_all.append(batch_targets[b][m])
                        seq_all.append(
                            np.full(int(m.sum()), batch_seq_ids[b], dtype=np.int64)
                        )
                        step_all.append(np.flatnonzero(m).astype(np.int64))

        y_pred = np.concatenate(preds_all, axis=0)
        y_true = np.concatenate(targets_all, axis=0)
        seq_ids = np.concatenate(seq_all, axis=0)
        step_ids = np.concatenate(step_all, axis=0)
        per_seq = per_sequence_weighted_pearson(y_true, y_pred, seq_ids)

        return EvalResult(
            metrics=summary(y_true, y_pred),
            predictions=y_pred,
            targets=y_true,
            seq_ids=seq_ids,
            step_ids=step_ids,
            per_sequence_corr=per_seq,
        )

    # ------------------------------------------------------------------
    # Classical models.
    # ------------------------------------------------------------------
    def run_classical_model(
        self,
        model: ClassicalModel,
        df: pd.DataFrame,
        *,
        with_engineered: bool = True,
        rolling_windows: tuple[int, ...] = (5, 20),
    ) -> EvalResult:
        full, feature_cols = build_tabular_features(
            df,
            with_engineered=with_engineered,
            rolling_windows=rolling_windows,
        )
        if model.feature_names_:
            # Reorder / restrict to the feature set the model was trained on.
            feature_cols = model.feature_names_
        ds = make_tabular_dataset(
            full,
            feature_columns=feature_cols,
            only_scored_rows=True,
            drop_warmup=True,
        )
        preds = model.predict(ds.X)
        per_seq = per_sequence_weighted_pearson(ds.y, preds, ds.seq_ids)
        return EvalResult(
            metrics=summary(ds.y, preds),
            predictions=preds,
            targets=ds.y,
            seq_ids=ds.seq_ids,
            step_ids=ds.step_ids,
            per_sequence_corr=per_seq,
        )


def evaluate_run(
    model,
    df: pd.DataFrame,
    *,
    kind: str,
    device: torch.device | str = "cpu",
    **kwargs,
) -> EvalResult:
    ev = Evaluator()
    if kind == "sequence":
        return ev.run_sequence_model(model, df, device=device, **kwargs)
    if kind == "classical":
        return ev.run_classical_model(model, df, **kwargs)
    raise ValueError(f"unknown model kind: {kind!r}")


# ---------------------------------------------------------------------------
# Streaming-mode evaluation (mirrors the official scorer; slower).
# ---------------------------------------------------------------------------


def predict_streaming(
    model: SequenceModel,
    df: pd.DataFrame,
    *,
    device: torch.device | str = "cpu",
) -> EvalResult:
    """Step-by-step evaluation using ``model.step``.

    This is what the competition scorer does internally and is the
    correct sanity check for streaming models. Roughly an order of
    magnitude slower than the batched ``run_sequence_model`` path; use
    it for final reporting, not for hyperparameter sweeps.
    """
    from data.constants import FEATURE_COLS  # local import to avoid cycle

    device = torch.device(device)
    model.eval().to(device)

    preds_all: list[np.ndarray] = []
    targets_all: list[np.ndarray] = []
    seq_all: list[int] = []
    step_all: list[int] = []

    state = None
    current_seq = None
    for row in df.itertuples(index=False):
        # Reset state on sequence boundary.
        if row.seq_ix != current_seq:
            state = model.init_state(batch_size=1, device=device)
            current_seq = row.seq_ix

        feat_step = torch.tensor(
            [getattr(row, c) for c in FEATURE_COLS],
            dtype=torch.float32, device=device,
        ).unsqueeze(0)                                  # (1, F)
        pred, state = model.step(feat_step, state)
        pred_np = pred.detach().cpu().numpy()[0]        # (K,)

        if bool(row.need_prediction):
            preds_all.append(pred_np)
            targets_all.append(np.array([row.t0, row.t1], dtype=np.float32))
            seq_all.append(int(row.seq_ix))
            step_all.append(int(row.step_in_seq))

    y_pred = np.stack(preds_all)
    y_true = np.stack(targets_all)
    seq_ids = np.asarray(seq_all, dtype=np.int64)
    step_ids = np.asarray(step_all, dtype=np.int64)
    return EvalResult(
        metrics=summary(y_true, y_pred),
        predictions=y_pred,
        targets=y_true,
        seq_ids=seq_ids,
        step_ids=step_ids,
        per_sequence_corr=per_sequence_weighted_pearson(y_true, y_pred, seq_ids),
    )
