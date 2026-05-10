"""Fixtures for feature engineering tests.

Provides synthetic LOB data with realistic structure for testing
microstructure and dynamics features.
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_df():
    """Generate synthetic LOB DataFrame with realistic structure.

    Creates:
    - 50 sequences × 100 steps (5,000 rows total)
    - Realistic bid/ask prices (p0..p11)
    - Realistic volumes (v0..v11)
    - Trade data (dp0..dp3, dv0..dv3)
    - Meta columns (seq_ix, step_in_seq, need_prediction)
    - Targets (t0, t1)

    Column structure matches real dataset:
    - p0-p5: bid prices (levels 0-5, 0=best bid)
    - p6-p11: ask prices (levels 0-5, 6=best ask, 11=worst ask)
    - v0-v5: bid volumes
    - v6-v11: ask volumes
    - dp0-dp3: trade prices
    - dv0-dv3: trade volumes
    - seq_ix: sequence index [0, 50)
    - step_in_seq: step within sequence [0, 100)
    - need_prediction: boolean (True for step >= 10)
    - t0, t1: targets

    Returns:
        pd.DataFrame with all columns initialized with realistic values.
        Structure is deterministic (seed=42) for reproducibility.
    """
    rng = np.random.default_rng(42)
    n_seqs, n_steps = 50, 100
    n_rows = n_seqs * n_steps

    seq_ixs = np.repeat(np.arange(n_seqs), n_steps)
    steps = np.tile(np.arange(n_steps), n_seqs)
    need_pred = steps >= 10

    # Base prices and volumes with realistic ranges
    # Bid prices: start at ~100, bid levels are decreasing
    base_bid = 100.0
    data = {
        "seq_ix": seq_ixs,
        "step_in_seq": steps,
        "need_prediction": need_pred,
    }

    # Bid prices (p0-p5): p0 is best bid (highest), decreasing
    for i in range(6):
        data[f"p{i}"] = base_bid - i * 0.02 + rng.normal(0, 0.002, n_rows)

    # Ask prices (p6-p11): p6 is best ask (lowest), increasing
    # Best ask (p6) is typically 0.01-0.02 above best bid (p0)
    for i in range(6):
        data[f"p{i + 6}"] = base_bid + 0.02 + i * 0.02 + rng.normal(0, 0.002, n_rows)

    # Bid volumes (v0-v5): decreasing with depth
    for i in range(6):
        data[f"v{i}"] = np.maximum(
            rng.exponential(scale=1000, size=n_rows) / (1 + i * 0.3),
            1,  # Minimum 1 contract
        )

    # Ask volumes (v6-v11): decreasing with depth
    for i in range(6):
        data[f"v{i + 6}"] = np.maximum(
            rng.exponential(scale=1000, size=n_rows) / (1 + i * 0.3),
            1,
        )

    # Trade prices (dp0-dp3): small random offsets
    for i in range(4):
        data[f"dp{i}"] = base_bid + rng.normal(0, 0.02, n_rows)

    # Trade volumes (dv0-dv3): smaller than LOB volumes
    for i in range(4):
        data[f"dv{i}"] = np.maximum(rng.exponential(scale=100, size=n_rows), 1)

    # Targets (t0, t1): binary or small continuous
    data["t0"] = rng.choice([0, 1], size=n_rows)
    data["t1"] = rng.choice([0, 1], size=n_rows)

    df = pd.DataFrame(data)

    # Introduce NaNs and extreme values as per spec Section 7
    # Specific indices for deterministic introduction of edge cases
    # Proportionally scaled for 5000-row dataset (50 seqs × 100 steps)

    # NaNs in price columns (missing bid/ask) - spread across dataset
    nan_price_rows = [50, 150, 250, 500, 1000, 1500, 2000, 2500, 3000, 3500]
    nan_prices = ["p0", "p6", "p2", "p8", "p1", "p7", "p3", "p9", "p4", "p10"]
    for row, col in zip(nan_price_rows, nan_prices):
        df.loc[row, col] = pd.NA

    # Extreme spreads: very wide (ask >> bid)
    wide_spread_rows = [100, 600, 1200, 2000, 3000, 4000]
    for row in wide_spread_rows:
        df.loc[row, "p6"] = df.loc[row, "p0"] + 2.0  # Extremely wide spread

    # Extreme spreads: very narrow (ask ≈ bid)
    narrow_spread_rows = [200, 700, 1500, 2500, 3500, 4500]
    for row in narrow_spread_rows:
        df.loc[row, "p6"] = df.loc[row, "p0"] + 0.0001  # Nearly no spread

    # Zero or NaN volumes (representing dry liquidity) - spread across dataset
    zero_vol_rows = [300, 800, 1800, 2200, 3200, 3800]
    zero_vol_cols = ["v0", "v6", "v3", "v9", "v1", "v7"]
    for row, col in zip(zero_vol_rows, zero_vol_cols):
        df.loc[row, col] = 0.0  # Zero volume at specific level

    # Missing volumes - spread across dataset
    nan_vol_rows = [400, 900, 1900, 2300, 3300, 3900]
    nan_vol_cols = ["v2", "v8", "v4", "v10", "v5", "v11"]
    for row, col in zip(nan_vol_rows, nan_vol_cols):
        df.loc[row, col] = pd.NA

    # Missing trade volumes for variety
    missing_trade_vol_rows = [450, 950, 1950, 2450, 3450, 4450]
    missing_trade_vol_cols = ["dv0", "dv1", "dv2", "dv3", "dv0", "dv1"]
    for row, col in zip(missing_trade_vol_rows, missing_trade_vol_cols):
        df.loc[row, col] = pd.NA

    return df
