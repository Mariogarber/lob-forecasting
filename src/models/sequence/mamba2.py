"""Mamba-2 backbone (selective SSM with linear complexity).

Uses the native CUDA kernels from ``mamba-ssm`` when available, otherwise
falls back to the pure-PyTorch implementation from ``mambapy``. The
fallback is roughly 3-4x slower but does not require building Triton or
NVCC, which matters under WSL2 / consumer GPUs.

Anti-overfit design (rev 2):

* Per-target heads — t0 and t1 stop fighting over the same final Linear.
  In the previous design we observed t1 collapse to ~0.04 while t0 held
  at 0.37. Decoupling the heads kills that pathology.
* Dropout between Mamba blocks (not only at input + head). The residual
  stream was memorising training sequences otherwise.
* Stochastic depth (DropPath) on the native path with a linear schedule
  from 0 in layer 0 to ``drop_path`` in the last layer. Standard recipe
  for deep SSM / Transformer stacks.
"""

from __future__ import annotations

import warnings

import torch
import torch.nn as nn

from models.base import SequenceModel
from models.sequence._common import FeatureProjector, RegressionHead
from models import register_model


# d_inner+B+C channel layouts the native causal_conv1d_cuda kernel accepts.
# Determined empirically on the official 2.2.2 build; other d_model values
# trigger ``RuntimeError: causal_conv1d with channel last layout requires
# strides (x.stride(0) and x.stride(2)) to be multiples of 8``. The pure-
# PyTorch fallback has no such constraint.
_NATIVE_SUPPORTED_D_MODEL: set[int] = {256, 512, 1024, 2048}


def _try_import_mamba_ssm():
    try:
        # Import the layer directly. The package ``__init__`` pulls in
        # ``transformers`` for the LM head, which we don't need; import the
        # submodule to avoid that side effect when transformers is missing.
        from mamba_ssm.modules.mamba2 import Mamba2 as _Mamba2  # type: ignore
        return _Mamba2
    except Exception:
        try:
            from mamba_ssm import Mamba2 as _Mamba2  # type: ignore
            return _Mamba2
        except Exception:
            return None


def _try_import_mambapy_mamba1():
    """Pure-PyTorch Mamba-1 stack from mambapy. Always works, no nvcc."""
    try:
        from mambapy.mamba import Mamba as _Mamba1  # type: ignore
        from mambapy.mamba import MambaConfig as _MambaConfig  # type: ignore
        return _Mamba1, _MambaConfig
    except Exception:
        return None


def _drop_path(x: torch.Tensor, drop_prob: float, training: bool) -> torch.Tensor:
    """Per-sample stochastic depth. Drops the entire residual branch for a
    fraction ``drop_prob`` of the batch, then rescales by ``1/(1-drop_prob)``
    so the expected output magnitude is unchanged at inference."""
    if drop_prob <= 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = torch.empty(shape, dtype=x.dtype, device=x.device).bernoulli_(keep_prob)
    return x.div(keep_prob) * mask


@register_model("mamba2", kind="sequence")
class Mamba2Model(SequenceModel):
    """Mamba-2 selective SSM backbone with anti-overfit regularisers.

    When the native CUDA kernels are available (``mamba-ssm`` installed
    and nvcc usable), we get true Mamba-2 with linear attention via SSD.
    Otherwise we fall back to the pure-PyTorch Mamba-1 stack from
    ``mambapy`` — slower and using the older selective-scan formulation,
    but identical in interface and good enough as a baseline. The
    ``backend_used`` attribute records which path was taken.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        n_features = int(config["n_features"])
        n_targets = int(config["n_targets"])
        d_model = int(config.get("d_model", 128))
        n_layers = int(config.get("num_layers", 3))
        headdim = int(config.get("headdim", 64))
        dropout = float(config.get("dropout", 0.3))
        drop_path = float(config.get("drop_path", 0.1))
        backend_request = config.get("backend", "auto")  # auto | mamba_ssm | mambapy

        self.n_targets = n_targets
        self.drop_path_max = drop_path

        self.proj = FeatureProjector(n_features, d_model, dropout=dropout)

        # Per-target heads. Each one emits 1 scalar; we concat at the end.
        # This is the single biggest change against the previous design,
        # because t0 and t1 had very different scales and a shared Linear
        # was letting the loss converge to t0-only.
        self.heads = nn.ModuleList(
            [RegressionHead(d_model, 1, dropout=dropout) for _ in range(n_targets)]
        )

        self.backend_used: str = ""
        self._uses_internal_stack: bool = False

        backend = self._resolve_backend(backend_request, d_model=d_model)
        if backend == "mamba_ssm":
            mamba2_cls = _try_import_mamba_ssm()
            assert mamba2_cls is not None
            self.layers = nn.ModuleList(
                [mamba2_cls(d_model=d_model, headdim=headdim) for _ in range(n_layers)]
            )
            self.norms = nn.ModuleList(
                [nn.LayerNorm(d_model) for _ in range(n_layers)]
            )
            self.block_dropout = nn.Dropout(dropout)
            # Linear schedule: layer 0 keeps everything, last layer drops at
            # ``drop_path`` rate. Standard ViT / Mamba recipe.
            self.drop_path_rates = [
                drop_path * i / max(1, n_layers - 1) for i in range(n_layers)
            ]
            self.backend_used = "mamba_ssm"
        else:
            loaded = _try_import_mambapy_mamba1()
            assert loaded is not None
            Mamba1, MambaConfig = loaded
            mamba_cfg = MambaConfig(
                d_model=d_model,
                n_layers=n_layers,
                d_state=int(config.get("d_state", 16)),
                expand_factor=int(config.get("expand_factor", 2)),
                use_cuda=False,
            )
            # mambapy's Mamba already stacks n_layers internally + norms,
            # so we cannot inject per-block dropout / drop_path the same
            # way. Apply input-level dropout (already in FeatureProjector)
            # and final-head dropout, then a single trunk-output dropout.
            self.stack = Mamba1(mamba_cfg)
            self.trunk_dropout = nn.Dropout(dropout)
            self._uses_internal_stack = True
            self.backend_used = "mambapy_mamba1"
            warnings.warn(
                "mamba-ssm CUDA kernels unavailable; using mambapy Mamba-1 fallback.",
                stacklevel=2,
            )

    def _resolve_backend(self, request: str, d_model: int | None = None) -> str:
        if request not in {"auto", "mamba_ssm", "mambapy"}:
            raise ValueError(
                f"backend must be auto|mamba_ssm|mambapy, got {request!r}"
            )
        if request == "mamba_ssm":
            if _try_import_mamba_ssm() is None:
                raise ImportError("mamba-ssm requested but not installed.")
            return "mamba_ssm"
        if request == "mambapy":
            if _try_import_mambapy_mamba1() is None:
                raise ImportError("mambapy requested but not installed.")
            return "mambapy"
        # auto: prefer mamba_ssm if installed AND the d_model is one of the
        # sizes its CUDA kernel accepts; otherwise fall back to mambapy.
        native_ok = (
            _try_import_mamba_ssm() is not None
            and (d_model is None or d_model in _NATIVE_SUPPORTED_D_MODEL)
        )
        if native_ok:
            return "mamba_ssm"
        if _try_import_mambapy_mamba1() is not None:
            return "mambapy"
        raise ImportError(
            "Neither mamba-ssm nor mambapy is available. "
            "Run `pip install mambapy` (and optionally mamba-ssm)."
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.proj(features)                  # (B, T, D)
        if self._uses_internal_stack:
            x = self.stack(x)
            x = self.trunk_dropout(x)
        else:
            for i, (block, norm) in enumerate(zip(self.layers, self.norms)):
                residual = block(norm(x))
                residual = self.block_dropout(residual)
                residual = _drop_path(residual, self.drop_path_rates[i], self.training)
                x = x + residual
        # Per-target heads, then concat along the feature axis -> (B, T, K).
        return torch.cat([h(x) for h in self.heads], dim=-1)
