"""Weighted MSE loss.

Each row is weighted by ``|y_true|`` (matching the competition weighting
scheme). Useful as a stable warm-up loss before switching to the
non-convex Pearson objective.
"""

from __future__ import annotations

import torch
import torch.nn as nn

EPS = 1e-8


def weighted_mse_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    eps: float = EPS,
) -> torch.Tensor:
    if y_true.dim() == 3:
        y_true = y_true.reshape(-1, y_true.shape[-1])
        y_pred = y_pred.reshape(-1, y_pred.shape[-1])
        if mask is not None:
            mask = mask.reshape(-1)

    if mask is not None:
        mask = mask.bool()
        if mask.sum() == 0:
            return y_pred.sum() * 0.0
        y_true = y_true[mask]
        y_pred = y_pred[mask]

    weights = y_true.abs().amax(dim=-1).clamp_min(eps)  # (N,)
    sq_err = ((y_pred - y_true) ** 2).mean(dim=-1)      # (N,)
    return (weights * sq_err).sum() / weights.sum().clamp_min(eps)


class WeightedMSELoss(nn.Module):
    """Weighted MSE with ``w = |y_true|``."""

    def __init__(self, eps: float = EPS):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return weighted_mse_loss(y_true=y_true, y_pred=y_pred, mask=mask, eps=self.eps)
