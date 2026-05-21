"""Differentiable weighted-Pearson loss == numpy reference."""

import numpy as np
import pytest
import torch

from losses.pearson import weighted_pearson_loss
from metrics.pearson import _weighted_pearson_scalar


def test_matches_numpy_reference(tiny_features_targets):
    y_true, y_pred = tiny_features_targets
    # Numpy reference: average per-target weighted Pearson.
    refs = [
        _weighted_pearson_scalar(y_true[:, i], y_pred[:, i]) for i in range(2)
    ]
    ref = float(np.mean(refs))

    yt = torch.from_numpy(y_true)
    yp = torch.from_numpy(y_pred)
    torch_loss = weighted_pearson_loss(yt, yp)
    torch_corr = 1.0 - float(torch_loss.item())
    assert torch_corr == pytest.approx(ref, abs=1e-5)


def test_perfect_predictions_zero_loss():
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, size=(500, 2)).astype(np.float32)
    yt = torch.from_numpy(y)
    loss = weighted_pearson_loss(yt, yt)
    assert float(loss.item()) == pytest.approx(0.0, abs=1e-5)


def test_negative_predictions_loss_two():
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, size=(500, 2)).astype(np.float32)
    yt = torch.from_numpy(y)
    yp = -yt
    loss = weighted_pearson_loss(yt, yp, clip_predictions=False)
    # corr = -1 -> loss = 1 - (-1) = 2
    assert float(loss.item()) == pytest.approx(2.0, abs=1e-4)


def test_mask_excludes_rows():
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, size=(8, 100, 2)).astype(np.float32)
    yp = y.copy()
    yp[:, :50, :] = rng.normal(0, 1, size=(8, 50, 2))  # warm-up garbage
    yt = torch.from_numpy(y)
    ypt = torch.from_numpy(yp)
    mask = torch.zeros(8, 100, dtype=torch.bool)
    mask[:, 50:] = True
    loss = weighted_pearson_loss(yt, ypt, mask=mask)
    # Within mask=True region, y_pred == y_true -> corr = 1 -> loss = 0.
    assert float(loss.item()) == pytest.approx(0.0, abs=1e-4)
