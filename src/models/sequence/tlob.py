"""TLOB — Dual-Attention Transformer for Limit Order Book forecasting.

Reference: Garcia et al., "TLOB: A Novel Transformer Model with Dual
Attention for Limit Order Book Forecasting" (arXiv:2403.09989, 2024).

Key insight against generic Transformers/Mamba on LOB data:

  * **Per-feature embedding**: each of the 32 input columns gets its own
    learnable projection to ``d_model`` dims. This decouples the model
    from any specific column ordering — unlike DeepLOB whose ``(1, 2)``
    convs assume the bid/ask/price/vol interleave of FI-2010.

  * **Dual-axis attention**: stacked alternation of
        feature-attention (each timestep attends across the 32 features)
        time-attention (each feature attends across past timesteps, causal)
    This captures both LOB shape (which levels matter) and temporal
    dynamics (how the book evolves) without mixing them prematurely.

  * Strong regularisation built in: dropout inside every sublayer,
    stochastic depth (DropPath) with a linear schedule across blocks,
    per-target heads (so t0 and t1 cannot collapse into each other).

  * Time attention is **strictly causal**; we set ``is_causal=True`` so
    PyTorch's scaled_dot_product_attention auto-picks the FlashAttention
    backend on CUDA, keeping memory linear in the sequence length.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base import SequenceModel
from models import register_model


def _drop_path(x: torch.Tensor, drop_prob: float, training: bool) -> torch.Tensor:
    """Per-sample stochastic depth on the leading batch axis."""
    if drop_prob <= 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = torch.empty(shape, dtype=x.dtype, device=x.device).bernoulli_(keep_prob)
    return x.div(keep_prob) * mask


class _MHA(nn.Module):
    """Multi-head self-attention with optional causal masking.

    We hand-roll Q/K/V projections (rather than use
    ``nn.MultiheadAttention``) so the call site stays explicit and
    PyTorch's ``scaled_dot_product_attention`` can pick FlashAttention
    when running on CUDA fp16.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads}).")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, *, is_causal: bool) -> torch.Tensor:
        # x: (B', L, D)
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)                    # each (B, L, H, Hd)
        q = q.transpose(1, 2)                          # (B, H, L, Hd)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=is_causal, dropout_p=0.0
        )
        out = out.transpose(1, 2).reshape(B, L, D)     # (B, L, D)
        return self.drop(self.out(out))


class _TransformerBlock(nn.Module):
    """Pre-norm transformer block: LN -> MHA -> residual -> LN -> FFN -> residual."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ffn_mult: int,
        dropout: float,
        drop_path: float,
        causal: bool,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = _MHA(d_model, n_heads, dropout=dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_mult * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.drop_path = float(drop_path)
        self.causal = bool(causal)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.attn(self.ln1(x), is_causal=self.causal)
        x = x + _drop_path(h, self.drop_path, self.training)
        h = self.ffn(self.ln2(x))
        x = x + _drop_path(h, self.drop_path, self.training)
        return x


@register_model("tlob", kind="sequence")
class TLOBModel(SequenceModel):
    """Dual-axis Transformer for LOB forecasting.

    Parameters (via ``config`` dict)
    --------------------------------
    n_features, n_targets : int
        Wired by the pipeline.
    d_model : int, default 64
        Per-feature embedding dimension. Each input feature scalar maps to
        a vector of this size.
    num_layers : int, default 3
        Number of *dual* blocks (1 feature-attn + 1 time-attn = 1 dual).
        Effective depth is ``2 * num_layers``.
    n_heads : int, default 4
        Attention heads (both axes share the head count).
    ffn_mult : int, default 4
        Hidden multiplier inside the FFN.
    dropout : float, default 0.2
        Applied inside attention projections, FFN and pre-head.
    drop_path : float, default 0.1
        Maximum stochastic-depth rate; scheduled linearly from 0 in the
        first block to ``drop_path`` in the last.
    max_len : int, default 1024
        Upper bound for the learnable positional encoding (must cover the
        full sequence length used at training / inference time).
    """

    def __init__(self, config: dict):
        super().__init__(config)
        n_features = int(config["n_features"])
        n_targets = int(config["n_targets"])
        d_model = int(config.get("d_model", 64))
        n_layers = int(config.get("num_layers", 3))
        n_heads = int(config.get("n_heads", 4))
        ffn_mult = int(config.get("ffn_mult", 4))
        dropout = float(config.get("dropout", 0.2))
        drop_path = float(config.get("drop_path", 0.1))
        max_len = int(config.get("max_len", 1024))

        self.n_features = n_features
        self.n_targets = n_targets
        self.d_model = d_model

        # Per-feature affine embedding: scalar value -> d-dim vector.
        # Equivalent to ``n_features`` independent ``nn.Linear(1, d_model)``
        # heads, just vectorised. Shape: (F, d).
        self.feat_scale = nn.Parameter(torch.empty(n_features, d_model))
        self.feat_bias = nn.Parameter(torch.empty(n_features, d_model))
        nn.init.normal_(self.feat_scale, std=0.02)
        nn.init.normal_(self.feat_bias, std=0.02)

        # Sinusoidal positional encoding for the time axis. We register as
        # buffer (non-trainable) — TLOB paper uses learned positions, but
        # sinusoidal generalises beyond ``max_len`` automatically.
        self.register_buffer(
            "time_pos", _sinusoidal_positions(max_len, d_model), persistent=False
        )

        # Alternating feature / time blocks. drop_path scheduled across the
        # 2 * num_layers total sub-blocks.
        total = 2 * n_layers
        rates = [drop_path * i / max(1, total - 1) for i in range(total)]
        blocks: list[nn.Module] = []
        for i in range(n_layers):
            blocks.append(_TransformerBlock(
                d_model, n_heads, ffn_mult, dropout, rates[2 * i], causal=False
            ))  # feature axis: unordered, no mask
            blocks.append(_TransformerBlock(
                d_model, n_heads, ffn_mult, dropout, rates[2 * i + 1], causal=True
            ))  # time axis: causal
        self.blocks = nn.ModuleList(blocks)

        # Pool features at the output: mean across F -> (B, T, d).
        self.pool_norm = nn.LayerNorm(d_model)

        # Per-target heads. Independent heads so t0 and t1 don't compete
        # for the same final Linear's weights.
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Dropout(dropout),
                nn.Linear(d_model, 1),
            )
            for _ in range(n_targets)
        ])

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features: (B, T, F)
        B, T, F_ = features.shape
        if F_ != self.n_features:
            raise ValueError(f"expected {self.n_features} features, got {F_}")
        if T > self.time_pos.size(0):
            raise ValueError(f"sequence length {T} exceeds max_len {self.time_pos.size(0)}")

        d = self.d_model

        # Per-feature embedding: (B, T, F) -> (B, T, F, d).
        # x_emb[b, t, f, :] = features[b, t, f] * scale[f, :] + bias[f, :]
        x = features.unsqueeze(-1) * self.feat_scale + self.feat_bias

        # Add time positional encoding to every feature row.
        pos = self.time_pos[:T].view(1, T, 1, d).to(x.dtype)
        x = x + pos

        for i, block in enumerate(self.blocks):
            if i % 2 == 0:
                # Feature attention: (B, T, F, d) -> (B*T, F, d).
                x = x.reshape(B * T, F_, d)
                x = block(x)
                x = x.reshape(B, T, F_, d)
            else:
                # Time attention (causal): (B, T, F, d) -> (B*F, T, d).
                x = x.permute(0, 2, 1, 3).reshape(B * F_, T, d)
                x = block(x)
                x = x.reshape(B, F_, T, d).permute(0, 2, 1, 3).contiguous()

        # Pool across feature axis -> (B, T, d).
        x = self.pool_norm(x.mean(dim=2))

        # Per-target heads.
        return torch.cat([h(x) for h in self.heads], dim=-1)


def _sinusoidal_positions(max_len: int, d_model: int) -> torch.Tensor:
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float)
        * -(math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe
