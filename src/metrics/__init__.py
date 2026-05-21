"""Numpy-side metrics (used for evaluation reporting).

The official scorer in ``utils.py`` ships its own implementation of the
weighted Pearson correlation; we re-export an identical function here
plus a handful of complementary diagnostics (MSE, MAE, per-target /
per-sequence breakdowns) so the evaluator can produce a richer report
without forking the scorer.
"""

from metrics.pearson import (
    weighted_pearson,
    weighted_pearson_per_target,
    per_sequence_weighted_pearson,
    mse,
    mae,
    summary,
)

__all__ = [
    "weighted_pearson",
    "weighted_pearson_per_target",
    "per_sequence_weighted_pearson",
    "mse",
    "mae",
    "summary",
]
