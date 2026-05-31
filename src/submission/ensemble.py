"""Build-time helpers for assembling a model ensemble.

This module is **not** copied into ``solution.zip`` — it only runs while you
are constructing the ensemble (tuning weights from validation predictions).
Inference uses ``submission.prediction_model.EnsemblePredictionModel``.
"""

from __future__ import annotations

from itertools import product

import numpy as np

from metrics import summary


def _simplex_grid(n: int, step: float) -> list[np.ndarray]:
    """All weight vectors of length ``n`` on the probability simplex, on a grid.

    For ``n`` members and grid ``step`` (e.g. 0.05) this enumerates every convex
    combination whose entries are multiples of ``step`` and sum to 1.
    """
    k = int(round(1.0 / step))
    grids = []
    for combo in product(range(k + 1), repeat=n - 1):
        if sum(combo) <= k:
            last = k - sum(combo)
            grids.append(np.array([*combo, last], dtype=np.float64) / k)
    return grids


def tune_ensemble_weights(
    member_preds: list[np.ndarray],
    y_true: np.ndarray,
    *,
    step: float = 0.05,
) -> dict:
    """Grid-search convex weights that maximise the scorer's weighted-Pearson.

    Parameters
    ----------
    member_preds : list of (N, K) arrays
        Each member's predictions on the **same** masked validation rows.
    y_true : (N, K) array
        The matching ground-truth targets.
    step : float
        Simplex grid resolution (0.05 -> 5% increments). Keep n_members small
        (<= 4) or the grid blows up.

    Returns
    -------
    dict with ``weights`` (list), ``score`` (best weighted-Pearson),
    and ``per_member`` (each member's solo weighted-Pearson, for reference).
    """
    n = len(member_preds)
    if n == 0:
        raise ValueError("need at least one member")
    preds = [np.asarray(p, dtype=np.float64) for p in member_preds]
    y = np.asarray(y_true, dtype=np.float64)

    per_member = [float(summary(y, p)["weighted_pearson"]) for p in preds]
    if n == 1:
        return {"weights": [1.0], "score": per_member[0], "per_member": per_member}

    best_w, best_s = None, -np.inf
    for w in _simplex_grid(n, step):
        blend = sum(wi * pi for wi, pi in zip(w, preds))
        s = float(summary(y, blend)["weighted_pearson"])
        if s > best_s:
            best_s, best_w = s, w
    return {
        "weights": [float(x) for x in best_w],
        "score": best_s,
        "per_member": per_member,
    }
