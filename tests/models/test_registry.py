"""Smoke tests for the model registry and per-model forward passes."""

import pytest
import torch

from data.constants import N_FEATURES, N_TARGETS
from models import MODEL_REGISTRY, get_model_class, list_models


EXPECTED_MODELS = {
    "linear", "ridge", "random_forest", "lightgbm",
    "gru", "lstm", "transformer", "deeplob", "mamba2", "tcn", "tlob",
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


@pytest.mark.parametrize("name", ["gru", "lstm", "transformer", "deeplob", "tcn", "tlob"])
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


def test_tcn_streaming_matches_forward():
    # TCN is fully causal by construction (dilated left-padded convs), so the
    # batched forward at step t must equal the streaming output at step t
    # whenever the buffer is shorter than ``window_size``.
    cls, _ = get_model_class("tcn")
    model = cls({
        "n_features": N_FEATURES,
        "n_targets": N_TARGETS,
        "channels": 16,
        "num_layers": 4,        # RF = 1 + 2*(3-1)*(2^4-1) = 61
        "kernel_size": 3,
        "dropout": 0.0,
    }).eval()
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


def test_tcn_causality():
    # Independent leakage test: changing inputs at step t+1..T must NOT
    # change the prediction at step t. We compare forward(x) against
    # forward(x_shuffled_future) at every step and assert equality up to t.
    cls, _ = get_model_class("tcn")
    model = cls({
        "n_features": N_FEATURES,
        "n_targets": N_TARGETS,
        "channels": 16,
        "num_layers": 3,
        "kernel_size": 3,
        "dropout": 0.0,
    }).eval()
    B, T = 1, 24
    x = torch.randn(B, T, N_FEATURES)
    with torch.no_grad():
        y_clean = model(x)
        for t_cut in [5, 10, 15]:
            x_perturbed = x.clone()
            # Replace every step > t_cut with totally different random data.
            x_perturbed[:, t_cut + 1 :, :] = torch.randn_like(x_perturbed[:, t_cut + 1 :, :])
            y_perturbed = model(x_perturbed)
            torch.testing.assert_close(
                y_clean[:, : t_cut + 1, :],
                y_perturbed[:, : t_cut + 1, :],
                atol=1e-5, rtol=1e-4,
                msg=f"TCN leaks future info into step <= {t_cut}",
            )


def test_tlob_causality():
    # TLOB's time-attention uses is_causal=True. Perturbing future inputs
    # at step t+1..T must NOT change the prediction at step t. This is
    # the load-bearing invariant — any leak here invalidates the model.
    cls, _ = get_model_class("tlob")
    model = cls({
        "n_features": N_FEATURES,
        "n_targets": N_TARGETS,
        "d_model": 32,
        "num_layers": 2,
        "n_heads": 4,
        "dropout": 0.0,
        "drop_path": 0.0,
    }).eval()
    B, T = 1, 24
    x = torch.randn(B, T, N_FEATURES)
    with torch.no_grad():
        y_clean = model(x)
        for t_cut in [5, 10, 15]:
            x_perturbed = x.clone()
            x_perturbed[:, t_cut + 1 :, :] = torch.randn_like(x_perturbed[:, t_cut + 1 :, :])
            y_perturbed = model(x_perturbed)
            torch.testing.assert_close(
                y_clean[:, : t_cut + 1, :],
                y_perturbed[:, : t_cut + 1, :],
                atol=1e-5, rtol=1e-4,
                msg=f"TLOB leaks future info into step <= {t_cut}",
            )


def test_tlob_per_target_heads_decouple():
    # Per-target heads must produce independent predictions: perturbing
    # head[0] weights must NOT change head[1]'s output. Catches accidental
    # parameter sharing across heads.
    cls, _ = get_model_class("tlob")
    model = cls({
        "n_features": N_FEATURES,
        "n_targets": N_TARGETS,
        "d_model": 32,
        "num_layers": 2,
        "n_heads": 4,
        "dropout": 0.0,
        "drop_path": 0.0,
    }).eval()
    x = torch.randn(1, 16, N_FEATURES)
    with torch.no_grad():
        y_clean = model(x)
        # Perturb only head[0].
        for p in model.heads[0].parameters():
            p.data.add_(torch.randn_like(p) * 0.1)
        y_perturbed = model(x)
    # head[0] (column 0) should change; head[1] (column 1) should NOT.
    assert not torch.allclose(y_clean[..., 0], y_perturbed[..., 0])
    torch.testing.assert_close(y_clean[..., 1], y_perturbed[..., 1])


def test_deeplob_streaming_matches_forward():
    # DeepLOB uses causal convs + a rolling-buffer step. Within a buffer
    # shorter than ``window_size`` (no trim path triggered), the batched
    # forward at position t must equal the streamed prediction at step t.
    cls, _ = get_model_class("deeplob")
    model = cls({
        "n_features": N_FEATURES,
        "n_targets": N_TARGETS,
        "window_size": 100,
        "c1": 8, "c2": 8, "c3": 16, "inception_channels": 16,
        "lstm_hidden": 16, "dropout": 0.0,
    }).eval()
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
