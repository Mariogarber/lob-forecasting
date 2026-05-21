"""Canonical column groups and sequence constants for the Wunder dataset.

Mirrors `src/profiling/constants.py` but lives in the `data` package so the
training pipeline does not need to import the profiling library. The names
are kept in sync; do not redefine them ad-hoc elsewhere.
"""

from __future__ import annotations

BID_PRICES = [f"p{i}" for i in range(6)]
ASK_PRICES = [f"p{i}" for i in range(6, 12)]
BID_VOLUMES = [f"v{i}" for i in range(6)]
ASK_VOLUMES = [f"v{i}" for i in range(6, 12)]
TRADE_PRICES = [f"dp{i}" for i in range(4)]
TRADE_VOLUMES = [f"dv{i}" for i in range(4)]

FEATURE_COLS: list[str] = (
    BID_PRICES + ASK_PRICES + BID_VOLUMES + ASK_VOLUMES
    + TRADE_PRICES + TRADE_VOLUMES
)  # 32 columns

TARGET_COLS: list[str] = ["t0", "t1"]
META_COLS: list[str] = ["seq_ix", "step_in_seq", "need_prediction"]

N_FEATURES = len(FEATURE_COLS)
N_TARGETS = len(TARGET_COLS)

# Competition contract: every sequence is exactly 1000 steps,
# steps 0..98 are warm-up (not scored), 99..999 are evaluated.
SEQ_LEN: int = 1000
WARMUP_STEPS: int = 99
