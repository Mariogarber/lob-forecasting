"""Composite loss: alpha * weighted_MSE + (1 - alpha) * (1 - weighted_corr).

Useful when the pure Pearson loss is unstable early in training. Start
``alpha`` near 1.0 (mostly MSE) and decay it toward 0.0 (mostly Pearson)
over a few epochs.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from losses.pearson import weighted_pearson_loss
from losses.mse import weighted_mse_loss


class CompositeLoss(nn.Module):
    def __init__(self, alpha: float = 0.5, clip_predictions: bool = True):
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.alpha = alpha
        self.clip_predictions = clip_predictions

    def forward(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mse_term = weighted_mse_loss(y_true, y_pred, mask)
        pearson_term = weighted_pearson_loss(
            y_true, y_pred, mask, clip_predictions=self.clip_predictions
        )
        return self.alpha * mse_term + (1.0 - self.alpha) * pearson_term

    def set_alpha(self, alpha: float) -> None:
        self.alpha = float(alpha)
