"""Smoke tests for the model registry and per-model forward passes."""

import pytest
import torch

from data.constants import N_FEATURES, N_TARGETS
from models import MODEL_REGISTRY, get_model_class, list_models


EXPECTED_MODELS = {
    "linear", "ridge", "random_forest", "lightgbm",
    "gru", "lstm", "transformer", "deeplob", "mamba2",
}


def test_all_models_registered():
    registered = set(MODEL_REGISTRY.keys())
    missing = EXPECTED_MODELS - registered
    assert not missing, f"Missing registrations: {missing}"


def test_get_model_class_unknown():
    with pytest.raises(KeyError):
        get_model_class("does-not-exist")


def test_list_models_filter():
    classical = set(list_models("classical"))
    sequence = set(list_models("sequence"))
    assert "linear" in classical
    assert "gru" in sequence
    assert classical.isdisjoint(sequence)


@pytest.mark.parametrize("name", ["gru", "lstm", "transformer", "deeplob"])
def test_sequence_forward_shape(name):
    cls, _ = get_model_class(name)
    model = cls({"n_features": N_FEATURES, "n_targets": N_TARGETS}).eval()
    B, T = 2, 128
    x = torch.randn(B, T, N_FEATURES)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (B, T, N_TARGETS)


def test_mamba2_forward_shape_cpu():
    """Mamba-2 CPU forward — forces the pure-PyTorch fallback."""
    cls, _ = get_model_class("mamba2")
    try:
        model = cls({
            "n_features": N_FEATURES,
            "n_targets": N_TARGETS,
            "backend": "mambapy",  # CPU-friendly backend
        }).eval()
    except ImportError:
        pytest.skip("mambapy is not available.")
    B, T = 2, 64
    x = torch.randn(B, T, N_FEATURES)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (B, T, N_TARGETS)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_mamba2_native_cuda_forward_shape():
    """Native Mamba-2 forward on CUDA — verifies the mamba-ssm CUDA path."""
    cls, _ = get_model_class("mamba2")
    try:
        model = cls({
            "n_features": N_FEATURES,
            "n_targets": N_TARGETS,
            "d_model": 256,
            "backend": "mamba_ssm",
        }).eval().cuda()
    except ImportError:
        pytest.skip("mamba-ssm is not available.")
    B, T = 2, 256
    x = torch.randn(B, T, N_FEATURES, device="cuda")
    with torch.no_grad():
        y = model(x)
    assert y.shape == (B, T, N_TARGETS)
    assert y.device.type == "cuda"


def test_gru_streaming_matches_forward():
    cls, _ = get_model_class("gru")
    model = cls({"n_features": N_FEATURES, "n_targets": N_TARGETS}).eval()
    B, T = 1, 32
    x = torch.randn(B, T, N_FEATURES)
    with torch.no_grad():
        y_full = model(x)
        state = model.init_state(batch_size=B, device=x.device)
        preds = []
        for t in range(T):
            p, state = model.step(x[:, t], state)
            preds.append(p)
        y_stream = torch.stack(preds, dim=1)
    torch.testing.assert_close(y_full, y_stream, atol=1e-5, rtol=1e-4)
