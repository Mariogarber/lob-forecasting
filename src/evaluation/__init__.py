"""Evaluation reporting: metrics + plots + predictions dumping."""

from evaluation.evaluator import Evaluator, evaluate_run, predict_streaming
from evaluation.plots import (
    plot_loss_curve,
    plot_pred_vs_target_scatter,
    plot_per_sequence_corr_hist,
)
from evaluation.report import write_report, make_run_id

__all__ = [
    "Evaluator",
    "evaluate_run",
    "predict_streaming",
    "plot_loss_curve",
    "plot_pred_vs_target_scatter",
    "plot_per_sequence_corr_hist",
    "write_report",
    "make_run_id",
]
