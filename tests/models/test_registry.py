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


def _tiny_mamba2(**overrides):
    """Small, deterministic Mamba-2 for CPU tests (pure-PyTorch SSD path)."""
    cls, _ = get_model_class("mamba2")
    cfg = {
        "n_features": N_FEATURES,
        "n_targets": N_TARGETS,
        "d_model": 64,
        "num_layers": 3,
        "headdim": 32,
        "d_state": 32,
        "chunk_size": 16,
        "dropout": 0.0,
        "drop_path": 0.0,
    }
    cfg.update(overrides)
    return cls(cfg).eval()


def test_mamba2_forward_shape_cpu():
    """Mamba-2 pure-PyTorch SSD forward. T=100 is not a multiple of chunk_size,
    so this also exercises the right-padding path in the scan."""
    model = _tiny_mamba2()
    assert model.backend_used == "pytorch_ssd"
    B, T = 2, 100
    x = torch.randn(B, T, N_FEATURES)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (B, T, N_TARGETS)
    assert torch.isfinite(y).all()


def test_mamba2_streaming_matches_forward():
    # The recurrent step() (carrying the SSM state + conv ring buffer) must
    # reproduce the chunked-scan forward() at every timestep. This is what
    # makes O(1)-per-step submission inference correct.
    model = _tiny_mamba2()
    B, T = 1, 48
    x = torch.randn(B, T, N_FEATURES)
    with torch.no_grad():
        y_full = model(x)
        state = model.init_state(batch_size=B, device=x.device)
        preds = []
        for t in range(T):
            p, state = model.step(x[:, t], state)
            preds.append(p)
        y_stream = torch.stack(preds, dim=1)
    torch.testing.assert_close(y_full, y_stream, atol=1e-4, rtol=1e-3)


def test_mamba2_causality():
    # The selective scan is strictly causal: perturbing inputs at t+1..T must
    # not change predictions at <= t. A leak here would invalidate streaming.
    model = _tiny_mamba2()
    B, T = 1, 40
    x = torch.randn(B, T, N_FEATURES)
    with torch.no_grad():
        y_clean = model(x)
        for t_cut in [5, 15, 25]:
            x_perturbed = x.clone()
            x_perturbed[:, t_cut + 1 :, :] = torch.randn_like(x_perturbed[:, t_cut + 1 :, :])
            y_perturbed = model(x_perturbed)
            torch.testing.assert_close(
                y_clean[:, : t_cut + 1, :],
                y_perturbed[:, : t_cut + 1, :],
                atol=1e-5, rtol=1e-4,
                msg=f"Mamba-2 leaks future info into step <= {t_cut}",
            )


def test_mamba2_per_target_heads_decouple():
    # Independent per-target heads: perturbing head[0] must not move head[1].
    model = _tiny_mamba2()
    x = torch.randn(1, 16, N_FEATURES)
    with torch.no_grad():
        y_clean = model(x)
        for p in model.heads[0].parameters():
            p.data.add_(torch.randn_like(p) * 0.1)
        y_perturbed = model(x)
    assert not torch.allclose(y_clean[..., 0], y_perturbed[..., 0])
    torch.testing.assert_close(y_clean[..., 1], y_perturbed[..., 1])


def test_mamba2_dual_trunk_forward_and_streaming():
    # split_targets shares early blocks then runs one Mamba stack per target.
    # Forward shape + streaming parity must hold exactly as in single-trunk.
    model = _tiny_mamba2(split_targets=True, n_shared_layers=2, n_branch_layers=2)
    assert model.backend_used.startswith("pytorch_ssd_dualtrunk")
    B, T = 1, 48
    x = torch.randn(B, T, N_FEATURES)
    with torch.no_grad():
        y_full = model(x)
        assert y_full.shape == (B, T, N_TARGETS)
        state = model.init_state(batch_size=B, device=x.device)
        preds = []
        for t in range(T):
            p, state = model.step(x[:, t], state)
            preds.append(p)
        y_stream = torch.stack(preds, dim=1)
    torch.testing.assert_close(y_full, y_stream, atol=1e-4, rtol=1e-3)


def test_mamba2_dual_trunk_targets_are_independent():
    # The load-bearing property: with separate branches, perturbing the t1
    # branch (blocks + norm + head) must leave t0's output *bit-for-bit*
    # unchanged. This is what cures the t1-collapse interference.
    model = _tiny_mamba2(split_targets=True, n_shared_layers=2, n_branch_layers=2)
    x = torch.randn(1, 24, N_FEATURES)
    with torch.no_grad():
        y_clean = model(x)
        for module in (model.branches[1], model.branch_norms[1], model.heads[1]):
            for p in module.parameters():
                p.add_(torch.randn_like(p) * 0.1)
        y_perturbed = model(x)
    torch.testing.assert_close(y_clean[..., 0], y_perturbed[..., 0])      # t0 frozen
    assert not torch.allclose(y_clean[..., 1], y_perturbed[..., 1])       # t1 moved


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_mamba2_cuda_forward_finite_under_amp():
    """Powerful config on CUDA under fp16 AMP: the fp32 scan core must keep
    the output finite (no exp/cumsum overflow -> no metric collapse)."""
    cls, _ = get_model_class("mamba2")
    model = cls({
        "n_features": N_FEATURES,
        "n_targets": N_TARGETS,
        "d_model": 256,
        "num_layers": 2,
        "d_state": 128,
        "headdim": 64,
        "chunk_size": 125,
    }).eval().cuda()
    B, T = 2, 256
    x = torch.randn(B, T, N_FEATURES, device="cuda")
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
        y = model(x)
    assert y.shape == (B, T, N_TARGETS)
    assert y.device.type == "cuda"
    assert torch.isfinite(y).all()


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


@pytest.mark.parametrize("opts", [
    {"activation": "gelu"},
    {"norm": "layer"},
    {"per_target_heads": True},
    {"norm": "layer", "activation": "gelu", "per_target_heads": True},
])
def test_tcn_options_preserve_causality_and_streaming(opts):
    # The new config-gated options (gelu activation, channel LayerNorm,
    # per-target heads) must keep the TCN strictly causal and keep step()
    # numerically equal to forward(). Channel LayerNorm normalises per timestep
    # across channels, so it must not leak across time.
    cls, _ = get_model_class("tcn")
    model = cls({
        "n_features": N_FEATURES, "n_targets": N_TARGETS,
        "channels": 16, "num_layers": 4, "kernel_size": 3, "dropout": 0.0,
        **opts,
    }).eval()
    B, T = 1, 32
    x = torch.randn(B, T, N_FEATURES)
    with torch.no_grad():
        y_full = model(x)
        assert y_full.shape == (B, T, N_TARGETS)
        # streaming parity
        state = model.init_state(batch_size=B, device=x.device)
        preds = []
        for t in range(T):
            p, state = model.step(x[:, t], state)
            preds.append(p)
        y_stream = torch.stack(preds, dim=1)
        torch.testing.assert_close(y_full, y_stream, atol=1e-5, rtol=1e-4)
        # causality
        for t_cut in (8, 20):
            xp = x.clone()
            xp[:, t_cut + 1 :, :] = torch.randn_like(xp[:, t_cut + 1 :, :])
            torch.testing.assert_close(
                y_full[:, : t_cut + 1, :], model(xp)[:, : t_cut + 1, :],
                atol=1e-5, rtol=1e-4,
                msg=f"TCN with {opts} leaks future into step <= {t_cut}",
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


def test_tlob_grad_checkpoint_matches_and_flows():
    # grad_checkpoint must be numerically exact vs the non-checkpointed forward
    # (same weights, dropout off) and must still produce finite gradients.
    cls, _ = get_model_class("tlob")
    cfg = dict(n_features=N_FEATURES, n_targets=N_TARGETS, d_model=32,
               num_layers=2, n_heads=4, dropout=0.0, drop_path=0.0)
    m_plain = cls({**cfg, "grad_checkpoint": False}).train()
    m_ckpt = cls({**cfg, "grad_checkpoint": True}).train()
    m_ckpt.load_state_dict(m_plain.state_dict())
    x = torch.randn(2, 40, N_FEATURES)
    y_plain = m_plain(x)
    y_ckpt = m_ckpt(x)
    torch.testing.assert_close(y_plain, y_ckpt, atol=1e-5, rtol=1e-4)
    y_ckpt.sum().backward()
    gnorm = sum((p.grad ** 2).sum() for p in m_ckpt.parameters() if p.grad is not None)
    assert torch.isfinite(gnorm) and gnorm > 0


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
