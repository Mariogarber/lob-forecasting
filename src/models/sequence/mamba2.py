"""Mamba-2 backbone (selective SSM with linear complexity).

Uses the native CUDA kernels from ``mamba-ssm`` when available, otherwise
falls back to the pure-PyTorch implementation from ``mambapy``. The
fallback is roughly 3-4x slower but does not require building Triton or
NVCC, which matters under WSL2 / consumer GPUs.

We expose Mamba-2 specifically (not Mamba-1) because:
  * Mamba-2 maps cleanly to matmul kernels (SSD form),
  * the published TSF literature (S-Mamba, Bi-Mamba+) uses Mamba-2,
  * Mamba-1 is still available as ``backbone: 'mamba1'`` if needed.

The block stack is interleaved with RMSNorm and a residual projection
layer to keep the gradients well-behaved at depth 4-8.
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


@register_model("mamba2", kind="sequence")
class Mamba2Model(SequenceModel):
    """Mamba-2 selective SSM backbone.

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
        n_layers = int(config.get("num_layers", 4))
        headdim = int(config.get("headdim", 64))
        dropout = float(config.get("dropout", 0.1))
        backend_request = config.get("backend", "auto")  # auto | mamba_ssm | mambapy

        self.proj = FeatureProjector(n_features, d_model, dropout=dropout)
        self.head = RegressionHead(d_model, n_targets, dropout=dropout)
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
            # mambapy's Mamba already stacks n_layers internally + norms.
            self.stack = Mamba1(mamba_cfg)
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
            x = self.stack(x)                    # mambapy stacks + norms internally
        else:
            for block, norm in zip(self.layers, self.norms):
                x = x + block(norm(x))           # pre-LN residual
        return self.head(x)
