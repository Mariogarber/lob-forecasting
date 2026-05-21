"""Differentiable weighted Pearson loss.

Matches the numpy reference (``metrics.weighted_pearson``) numerically:

    pred_clip  = clip(pred, -6, 6)
    w          = max(|y_true|, eps)
    means      = weighted means of y_true and pred_clip
    cov, var_t, var_p = weighted (co)variances
    corr       = cov / sqrt(var_t * var_p)
    loss       = 1 - corr      (so we minimise)

The loss aggregates **across all unmasked elements of a batch**, per
target column. The two columns are averaged (same convention as the
official scorer).

Important subtlety: Pearson is a *batch-level* statistic. With very
small batches the estimate is noisy and gradients are unstable, so use
batch sizes in the hundreds (i.e. one full sequence per sample is
already plenty, because the loss sees seq_len * batch_size elements).
"""

from __future__ import annotations

import torch
import torch.nn as nn

PRED_CLIP = 6.0
EPS = 1e-8


def _weighted_pearson_per_column(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Compute weighted Pearson per target column.

    Shapes
    ------
    y_true, y_pred : (N, K)
    weights        : (N,) or (N, K)
    Returns        : (K,)
    """
    if weights.dim() == 1:
        weights = weights.unsqueeze(-1).expand_as(y_true)

    sum_w = weights.sum(dim=0).clamp_min(EPS)
    mean_t = (y_true * weights).sum(dim=0) / sum_w
    mean_p = (y_pred * weights).sum(dim=0) / sum_w
    dt = y_true - mean_t
    dp = y_pred - mean_p
    cov = (weights * dt * dp).sum(dim=0) / sum_w
    var_t = (weights * dt * dt).sum(dim=0) / sum_w + EPS
    var_p = (weights * dp * dp).sum(dim=0) / sum_w + EPS
    return cov / torch.sqrt(var_t * var_p)


def weighted_pearson_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    clip_predictions: bool = True,
    eps: float = EPS,
) -> torch.Tensor:
    """Functional form (see ``WeightedPearsonLoss`` for the nn.Module wrapper)."""
    if clip_predictions:
        # Use clamp (preserves gradients inside the range).
        y_pred = torch.clamp(y_pred, -PRED_CLIP, PRED_CLIP)

    # Flatten to (N, K)
    if y_true.dim() == 3:
        # (B, T, K) -> (B*T, K)
        y_true = y_true.reshape(-1, y_true.shape[-1])
        y_pred = y_pred.reshape(-1, y_pred.shape[-1])
        if mask is not None:
            mask = mask.reshape(-1)

    if mask is not None:
        mask = mask.bool()
        if mask.sum() == 0:
            # Nothing scored — return zero loss with grad-safe path.
            return y_pred.sum() * 0.0
        y_true = y_true[mask]
        y_pred = y_pred[mask]

    # Scorer-aligned: each target column has its own weights w_k = |y_true[:, k]|.
    weights = y_true.abs().clamp_min(eps)
    corr = _weighted_pearson_per_column(y_true, y_pred, weights)
    # Mean across target columns matches the scorer.
    return 1.0 - corr.mean()


class WeightedPearsonLoss(nn.Module):
    """``1 - weighted_pearson(y_true, y_pred)``, mask-aware."""

    def __init__(self, clip_predictions: bool = True, eps: float = EPS):
        super().__init__()
        self.clip_predictions = clip_predictions
        self.eps = eps

    def forward(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return weighted_pearson_loss(
            y_true=y_true,
            y_pred=y_pred,
            mask=mask,
            clip_predictions=self.clip_predictions,
            eps=self.eps,
        )
