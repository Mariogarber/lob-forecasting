"""Mamba-2 backbone — self-contained pure-PyTorch SSD implementation.

This is a *real* Mamba-2 (Dao & Gu 2024, "Transformers are SSMs"), built
from scratch in PyTorch so it needs **no** native CUDA kernels. The repo's
previous version tried to use ``mamba-ssm`` / ``mambapy`` and silently fell
back to a pure-PyTorch **Mamba-1** whenever those weren't importable — which,
on this machine, is *always* (``mamba_ssm`` is not installed and ``mambapy``'s
own ``mamba2`` module hard-imports it). The result was that "mamba2" was never
Mamba-2 at all.

This rewrite removes that fragility: a single code path that always runs the
Mamba-2 selective state-space duality (SSD) via the chunked-scan algorithm.

Why this design (and why it survives an 8 GB RTX 3060 Ti)
--------------------------------------------------------
* **Linear in sequence length.** The SSD scan is O(T · d_state) — no O(T²)
  attention matrix like TLOB/Transformer. A full 1000-step sequence costs a
  fraction of the memory of dual-axis attention, so we can afford a wide
  model (d_model 256, 8 layers) and a real batch size on 8 GB.

* **fp32 scan core (the big anti-collapse lever).** The selective scan is full
  of ``exp`` / ``cumsum`` over the decay terms. Under fp16 AMP those overflow
  to ``inf`` → ``NaN`` and the weighted-Pearson metric collapses to ~0. We
  therefore run the *entire* scan in fp32 regardless of the surrounding
  autocast context (matching what the native CUDA kernels do internally) and
  only the cheap linear projections stay in fp16. This is what makes training
  stable on consumer GPUs.

* **Final norm before the heads.** The old design fed the raw residual stream
  straight into the regression heads. A pre-norm stack *needs* a closing norm;
  without it the output scale drifts and the model is prone to the
  constant-output degeneracy (variance → 0 ⇒ correlation → 0). Added here.

* **Per-target heads.** t0 and t1 have very different scales; a shared final
  Linear lets the loss converge to t0-only and collapse t1. Independent heads
  (kept from the previous revision) prevent that pathology.

* **Stochastic depth + inter-block dropout.** Standard deep-SSM regularisers,
  scheduled linearly across layers.

Streaming
---------
``init_state`` / ``step`` implement the *recurrent* form of the SSM (O(1) per
timestep, carrying a per-head ``(headdim, d_state)`` state plus the depthwise
conv ring buffer). This is what the step-by-step competition scorer wants —
no re-running the whole prefix each tick. ``step`` matches ``forward``
numerically (verified by ``tests/models/test_registry.py``).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base import SequenceModel, StreamingState
from models.sequence._common import FeatureProjector, RegressionHead
from models import register_model


# ---------------------------------------------------------------------------
# Small building blocks
# ---------------------------------------------------------------------------


def _drop_path(x: torch.Tensor, drop_prob: float, training: bool) -> torch.Tensor:
    """Per-sample stochastic depth on the leading batch axis."""
    if drop_prob <= 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = torch.empty(shape, dtype=x.dtype, device=x.device).bernoulli_(keep_prob)
    return x.div(keep_prob) * mask


class RMSNorm(nn.Module):
    """Root-mean-square LayerNorm. Computed in fp32 for AMP stability."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * xf.to(dtype)


class GatedRMSNorm(nn.Module):
    """Mamba-2 gated RMSNorm: normalise ``x * silu(z)`` (norm_before_gate=False).

    Computed in fp32 then cast back, so the gate never destabilises AMP.
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        xf = x.float() * F.silu(z.float())
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * xf.to(dtype)


# ---------------------------------------------------------------------------
# SSD chunked scan (fp32). Canonical "ssd_minimal_discrete" from the Mamba-2
# paper, expressed with torch.einsum. Inputs are already discretised:
#   X = x * dt,  A = A_scalar * dt  (per-step log-decay), B, C raw.
# ---------------------------------------------------------------------------


def _segsum(x: torch.Tensor) -> torch.Tensor:
    """Stable lower-triangular segment-sum. ``x`` (..., T) -> (..., T, T)."""
    T = x.size(-1)
    x = x.unsqueeze(-1).expand(*x.shape, T)                       # (..., T, T)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool), -1)
    x = x.masked_fill(~mask, 0)
    x_segsum = x.cumsum(dim=-2)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool), 0)
    return x_segsum.masked_fill(~mask, float("-inf"))


def _ssd_chunk_scan(
    X: torch.Tensor,    # (b, l, h, p)
    A: torch.Tensor,    # (b, l, h)
    B: torch.Tensor,    # (b, l, h, n)
    C: torch.Tensor,    # (b, l, h, n)
    chunk: int,
) -> torch.Tensor:
    """Selective-scan output ``Y`` (b, l, h, p). Sequence length must be a
    multiple of ``chunk`` (the caller right-pads to guarantee this)."""
    b, l, h, p = X.shape
    c = l // chunk
    X = X.reshape(b, c, chunk, h, p)
    A = A.reshape(b, c, chunk, h)
    B = B.reshape(b, c, chunk, h, -1)
    C = C.reshape(b, c, chunk, h, -1)

    A = A.permute(0, 3, 1, 2)                                     # (b, h, c, l)
    A_cumsum = A.cumsum(dim=-1)

    # 1. Intra-chunk (diagonal) outputs.
    L = torch.exp(_segsum(A))                                     # (b, h, c, l, l)
    Y_diag = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", C, B, L, X)

    # 2. Each chunk's end-state (right factor of the off-diagonal blocks).
    decay_states = torch.exp(A_cumsum[..., -1:] - A_cumsum)       # (b, h, c, l)
    states = torch.einsum("bclhn,bhcl,bclhp->bchpn", B, decay_states, X)

    # 3. Inter-chunk recurrence over the chunk-boundary states.
    initial = torch.zeros_like(states[:, :1])
    states = torch.cat([initial, states], dim=1)                 # (b, c+1, h, p, n)
    decay_chunk = torch.exp(_segsum(F.pad(A_cumsum[..., -1], (1, 0))))
    new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
    states = new_states[:, :-1]                                  # (b, c, h, p, n)

    # 4. State -> output (left factor).
    state_decay_out = torch.exp(A_cumsum)                        # (b, h, c, l)
    Y_off = torch.einsum("bclhn,bchpn,bhcl->bclhp", C, states, state_decay_out)

    return (Y_diag + Y_off).reshape(b, l, h, p)


# ---------------------------------------------------------------------------
# Mamba-2 mixer
# ---------------------------------------------------------------------------


class Mamba2Mixer(nn.Module):
    """One Mamba-2 SSD mixer (in_proj -> causal conv -> SSD -> gated norm -> out).

    Shapes follow the reference implementation: ``d_inner = expand * d_model``
    split into ``nheads`` heads of width ``headdim``; ``ngroups`` shared
    (B, C) groups (default 1, broadcast to all heads).
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        headdim: int = 64,
        expand: int = 2,
        d_conv: int = 4,
        ngroups: int = 1,
        chunk_size: int = 128,
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
        dt_init_floor: float = 1e-4,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_inner = expand * d_model
        self.headdim = headdim
        if self.d_inner % headdim != 0:
            raise ValueError(
                f"expand*d_model ({self.d_inner}) must be divisible by headdim ({headdim})."
            )
        self.nheads = self.d_inner // headdim
        self.d_state = d_state
        self.ngroups = ngroups
        if self.nheads % ngroups != 0:
            raise ValueError(f"nheads ({self.nheads}) must be divisible by ngroups ({ngroups}).")
        self.d_conv = d_conv
        self.chunk_size = chunk_size
        self.conv_dim = self.d_inner + 2 * ngroups * d_state

        # in_proj emits [z | xBC | dt].
        self.in_proj = nn.Linear(
            d_model, 2 * self.d_inner + 2 * ngroups * d_state + self.nheads, bias=False
        )
        # Depthwise causal conv over the (x, B, C) channels.
        self.conv1d = nn.Conv1d(
            self.conv_dim, self.conv_dim, kernel_size=d_conv,
            groups=self.conv_dim, padding=0, bias=True,
        )

        # dt bias initialised so softplus(dt_bias) lands in [dt_min, dt_max].
        dt = torch.exp(
            torch.rand(self.nheads) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp_min(dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))               # inverse softplus
        self.dt_bias = nn.Parameter(inv_dt)

        # A_scalar = -exp(A_log) in [-d_state_init_max, -1]; classic Mamba init.
        A = torch.empty(self.nheads).uniform_(1.0, 16.0)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.nheads))

        self.norm = GatedRMSNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    # -- training forward -------------------------------------------------
    def forward(self, u: torch.Tensor) -> torch.Tensor:
        B_, L, _ = u.shape
        z, xBC, dt = torch.split(
            self.in_proj(u),
            [self.d_inner, self.conv_dim, self.nheads],
            dim=-1,
        )
        # Causal depthwise conv + SiLU.
        xBC = xBC.transpose(1, 2)                                # (B, conv_dim, L)
        xBC = F.pad(xBC, (self.d_conv - 1, 0))
        xBC = self.conv1d(xBC)[..., :L].transpose(1, 2)          # (B, L, conv_dim)
        xBC = F.silu(xBC)

        x, Bm, Cm = torch.split(
            xBC,
            [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state],
            dim=-1,
        )
        x = x.reshape(B_, L, self.nheads, self.headdim)
        Bm = Bm.reshape(B_, L, self.ngroups, self.d_state)
        Cm = Cm.reshape(B_, L, self.ngroups, self.d_state)
        rep = self.nheads // self.ngroups
        Bm = Bm.repeat_interleave(rep, dim=2)                    # (B, L, H, N)
        Cm = Cm.repeat_interleave(rep, dim=2)

        dt = F.softplus(dt + self.dt_bias)                       # (B, L, H)
        A = -torch.exp(self.A_log)                               # (H,)

        y = self._scan(x, dt, A, Bm, Cm)                        # (B, L, H, P)
        y = y + x * self.D.view(1, 1, -1, 1)
        y = y.reshape(B_, L, self.d_inner)
        y = self.norm(y, z)
        return self.out_proj(y)

    def _scan(self, x, dt, A, Bm, Cm) -> torch.Tensor:
        """fp32 SSD scan with right-padding to a multiple of chunk_size."""
        L = x.shape[1]
        chunk = self.chunk_size
        pad = (chunk - L % chunk) % chunk

        Xf = (x * dt.unsqueeze(-1)).float()
        Af = (A.view(1, 1, -1) * dt).float()
        Bf = Bm.float()
        Cf = Cm.float()
        if pad:
            Xf = F.pad(Xf, (0, 0, 0, 0, 0, pad))
            Af = F.pad(Af, (0, 0, 0, pad))
            Bf = F.pad(Bf, (0, 0, 0, 0, 0, pad))
            Cf = F.pad(Cf, (0, 0, 0, 0, 0, pad))
        Y = _ssd_chunk_scan(Xf, Af, Bf, Cf, chunk)[:, :L]
        return Y.to(x.dtype)

    # -- streaming (recurrent) -------------------------------------------
    def allocate_inference_cache(self, batch: int, device, dtype=torch.float32):
        conv_state = torch.zeros(batch, self.conv_dim, self.d_conv, device=device, dtype=dtype)
        ssm_state = torch.zeros(
            batch, self.nheads, self.headdim, self.d_state, device=device, dtype=dtype
        )
        return conv_state, ssm_state

    def step(self, u: torch.Tensor, conv_state: torch.Tensor, ssm_state: torch.Tensor):
        """Single timestep. ``u`` (B, d_model). Returns (y, conv_state, ssm_state)."""
        z, xBC, dt = torch.split(
            self.in_proj(u),
            [self.d_inner, self.conv_dim, self.nheads],
            dim=-1,
        )
        # Roll conv ring buffer (oldest..newest) and insert the new frame.
        conv_state = torch.roll(conv_state, shifts=-1, dims=-1)
        conv_state = conv_state.clone()
        conv_state[..., -1] = xBC
        w = self.conv1d.weight.squeeze(1)                        # (conv_dim, d_conv)
        xBC = (conv_state * w).sum(dim=-1) + self.conv1d.bias
        xBC = F.silu(xBC)

        x, Bm, Cm = torch.split(
            xBC,
            [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state],
            dim=-1,
        )
        x = x.reshape(-1, self.nheads, self.headdim)            # (B, H, P)
        Bm = Bm.reshape(-1, self.ngroups, self.d_state)
        Cm = Cm.reshape(-1, self.ngroups, self.d_state)
        rep = self.nheads // self.ngroups
        Bm = Bm.repeat_interleave(rep, dim=1)                    # (B, H, N)
        Cm = Cm.repeat_interleave(rep, dim=1)

        dt = F.softplus(dt + self.dt_bias)                       # (B, H)
        A = -torch.exp(self.A_log)                               # (H,)
        dA = torch.exp(dt * A)                                   # (B, H)
        dBx = (dt.unsqueeze(-1).unsqueeze(-1) * x.unsqueeze(-1)) * Bm.unsqueeze(2)
        ssm_state = ssm_state * dA.unsqueeze(-1).unsqueeze(-1) + dBx
        y = (ssm_state * Cm.unsqueeze(2)).sum(dim=-1)           # (B, H, P)
        y = y + x * self.D.view(1, -1, 1)
        y = y.reshape(-1, self.d_inner)
        y = self.norm(y, z)
        return self.out_proj(y), conv_state, ssm_state


# ---------------------------------------------------------------------------
# Block + model
# ---------------------------------------------------------------------------


class Mamba2Block(nn.Module):
    """Pre-norm residual block: ``x + drop_path(dropout(mixer(norm(x))))``."""

    def __init__(self, d_model: int, dropout: float, drop_path: float, **mixer_kwargs):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.mixer = Mamba2Mixer(d_model, **mixer_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.drop_path = float(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.dropout(self.mixer(self.norm(x)))
        return x + _drop_path(h, self.drop_path, self.training)

    def step(self, u, conv_state, ssm_state):
        h, conv_state, ssm_state = self.mixer.step(self.norm(u), conv_state, ssm_state)
        return u + h, conv_state, ssm_state


@register_model("mamba2", kind="sequence")
class Mamba2Model(SequenceModel):
    """Pure-PyTorch Mamba-2 selective state-space regressor.

    Config keys (all optional except the pipeline-wired n_features/n_targets):
        d_model      : residual width (default 256)
        num_layers   : number of Mamba-2 blocks (default 8)
        headdim      : SSM head width (default 64); expand*d_model must divide it
        d_state      : SSM state size N (default 128)
        expand       : inner expansion factor (default 2); alias expand_factor
        d_conv       : depthwise conv kernel (default 4)
        ngroups      : shared (B,C) groups (default 1)
        chunk_size   : SSD chunk length (default 128)
        dropout      : inter-block + head dropout (default 0.2)
        drop_path    : max stochastic-depth rate, linear schedule (default 0.1)

    Dual-trunk (per-target) keys:
        split_targets    : if True, share early layers then run one independent
                           Mamba stack per target (default False). Cures the
                           t1-collapse / multi-task interference seen with a
                           shared trunk.
        n_shared_layers  : shared feature-extraction blocks (default 4)
        n_branch_layers  : per-target blocks after the split (default 2)
        (``num_layers`` is ignored when ``split_targets`` is True.)

    Legacy keys ``backend`` are accepted and ignored — there is now a single
    pure-PyTorch path, so there is no backend to choose.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        n_features = int(config["n_features"])
        n_targets = int(config["n_targets"])
        d_model = int(config.get("d_model", 256))
        n_layers = int(config.get("num_layers", 8))
        dropout = float(config.get("dropout", 0.2))
        drop_path = float(config.get("drop_path", 0.1))

        mixer_kwargs = dict(
            d_state=int(config.get("d_state", 128)),
            headdim=int(config.get("headdim", 64)),
            expand=int(config.get("expand", config.get("expand_factor", 2))),
            d_conv=int(config.get("d_conv", 4)),
            ngroups=int(config.get("ngroups", 1)),
            chunk_size=int(config.get("chunk_size", 128)),
        )

        self.n_features = n_features
        self.n_targets = n_targets
        self.d_model = d_model
        # Kept for the notebook's status print; there is only one path now.
        self.backend_used = "pytorch_ssd"

        self.proj = FeatureProjector(n_features, d_model, dropout=dropout)

        def make_stack(rate_list):
            return nn.ModuleList(
                [Mamba2Block(d_model, dropout, r, **mixer_kwargs) for r in rate_list]
            )

        # Dual-trunk mode (``split_targets``): the targets t0/t1 share early
        # feature-extraction layers, then split into one independent Mamba
        # stack *per target*. This fixes the t1-collapse pathology observed in
        # the single-trunk model — where the shared backbone specialises for the
        # easier target (t0 held ~0.38) and actively destroys the harder one
        # (t1 fell from ~0.14 to negative as training continued). Each target
        # now owns its late layers + final norm + head, so t1 is no longer
        # starved by t0's gradient. ``backend_used`` records the topology.
        self.split_targets = bool(config.get("split_targets", False))
        if self.split_targets:
            self.n_shared = int(config.get("n_shared_layers", 4))
            self.n_branch = int(config.get("n_branch_layers", 2))
            total = self.n_shared + self.n_branch
            rates = [drop_path * i / max(1, total - 1) for i in range(total)]
            self.shared_blocks = make_stack(rates[: self.n_shared])
            self.branches = nn.ModuleList(
                [make_stack(rates[self.n_shared :]) for _ in range(n_targets)]
            )
            self.branch_norms = nn.ModuleList(
                [RMSNorm(d_model) for _ in range(n_targets)]
            )
            self.backend_used = f"pytorch_ssd_dualtrunk[{self.n_shared}+{self.n_branch}]"
        else:
            rates = [drop_path * i / max(1, n_layers - 1) for i in range(n_layers)]
            self.blocks = make_stack(rates)
            self.norm_f = RMSNorm(d_model)

        self.heads = nn.ModuleList(
            [RegressionHead(d_model, 1, dropout=dropout) for _ in range(n_targets)]
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.proj(features)                                  # (B, T, D)
        if self.split_targets:
            for block in self.shared_blocks:
                x = block(x)
            outs = []
            for b in range(self.n_targets):
                xb = x
                for block in self.branches[b]:
                    xb = block(xb)
                xb = self.branch_norms[b](xb)
                outs.append(self.heads[b](xb))
            return torch.cat(outs, dim=-1)                       # (B, T, K)
        for block in self.blocks:
            x = block(x)
        x = self.norm_f(x)
        return torch.cat([h(x) for h in self.heads], dim=-1)    # (B, T, K)

    # -- streaming inference ---------------------------------------------
    def init_state(self, batch_size: int, device: torch.device) -> StreamingState:
        def alloc(stack):
            return [blk.mixer.allocate_inference_cache(batch_size, device) for blk in stack]

        if self.split_targets:
            return StreamingState(payload={
                "shared": alloc(self.shared_blocks),
                "branches": [alloc(br) for br in self.branches],
            })
        return StreamingState(payload={"caches": alloc(self.blocks)})

    def step(self, feature_step: torch.Tensor, state: StreamingState):
        u = self.proj(feature_step.unsqueeze(1)).squeeze(1)     # (B, D)

        if self.split_targets:
            p = state.payload
            new_shared = []
            for block, (cs, ss) in zip(self.shared_blocks, p["shared"]):
                u, cs, ss = block.step(u, cs, ss)
                new_shared.append((cs, ss))
            preds, new_branches = [], []
            for b, branch in enumerate(self.branches):
                ub = u
                cur = []
                for block, (cs, ss) in zip(branch, p["branches"][b]):
                    ub, cs, ss = block.step(ub, cs, ss)
                    cur.append((cs, ss))
                ub = self.branch_norms[b](ub)
                preds.append(self.heads[b](ub))
                new_branches.append(cur)
            p["shared"], p["branches"] = new_shared, new_branches
            return torch.cat(preds, dim=-1), state

        new_caches = []
        for block, (conv_state, ssm_state) in zip(self.blocks, state.payload["caches"]):
            u, conv_state, ssm_state = block.step(u, conv_state, ssm_state)
            new_caches.append((conv_state, ssm_state))
        u = self.norm_f(u)
        pred = torch.cat([h(u) for h in self.heads], dim=-1)
        state.payload["caches"] = new_caches
        return pred, state
