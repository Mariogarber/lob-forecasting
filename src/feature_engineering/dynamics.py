"""Dynamics features for LOB analysis.

Extracts temporal change features from LOB sequences:
- Price momentum: change in mid-price over lag steps
- Volume momentum: change in total volume over lag steps
- Bid-ask momentum difference: asymmetry in bid vs ask volume changes
"""

import pandas as pd
from src.feature_engineering.microstructure import mid_price
from src.feature_engineering.utils import _lagged_change


def price_momentum(df: pd.DataFrame, lag: int = 1) -> pd.Series:
    """Calculate price momentum.

    Formula: mid_price(t) - mid_price(t-lag)
    where mid_price = (p0 + p6) / 2

    Interpretation:
    - Positive: price increasing (uptrend)
    - Negative: price decreasing (downtrend)
    - Calculated per sequence (does not cross sequence boundaries)

    Args:
        df: DataFrame with columns 'p0', 'p6' and 'seq_ix' (sequence index)
        lag: Number of steps to look back (default: 1)

    Returns:
        Series with momentum values. First 'lag' steps of each sequence are NaN.

    Raises:
        ValueError: If required columns are missing.
    """
    mid = mid_price(df)
    return _lagged_change(mid, lag=lag, seq_ix=df["seq_ix"])


def volume_momentum(df: pd.DataFrame, lag: int = 1) -> pd.Series:
    """Calculate total volume momentum.

    Formula: total_volume(t) - total_volume(t-lag)
    where total_volume = sum(v0:v11)

    Interpretation:
    - Positive: liquidity increasing
    - Negative: liquidity decreasing
    - Calculated per sequence (does not cross sequence boundaries)

    Args:
        df: DataFrame with columns 'v0' through 'v11' and 'seq_ix' (sequence index)
        lag: Number of steps to look back (default: 1)

    Returns:
        Series with momentum values. First 'lag' steps of each sequence are NaN.

    Raises:
        ValueError: If required columns are missing.
    """
    vol_cols = [f"v{i}" for i in range(12)]
    total_vol = df[vol_cols].sum(axis=1)
    return _lagged_change(total_vol, lag=lag, seq_ix=df["seq_ix"])


def bid_ask_momentum_diff(df: pd.DataFrame, lag: int = 1) -> pd.Series:
    """Calculate asymmetry in bid vs ask volume momentum.

    Formula:
    - bid_momentum = sum(v0:v5)(t) - sum(v0:v5)(t-lag)
    - ask_momentum = sum(v6:v11)(t) - sum(v6:v11)(t-lag)
    - Result: bid_momentum - ask_momentum

    Interpretation:
    - Positive: bid side liquidity increasing faster (buy pressure)
    - Negative: ask side liquidity increasing faster (sell pressure)
    - Calculated per sequence (does not cross sequence boundaries)

    Args:
        df: DataFrame with columns 'v0' through 'v11' and 'seq_ix' (sequence index)
        lag: Number of steps to look back (default: 1)

    Returns:
        Series with momentum difference values. First 'lag' steps of each sequence are NaN.

    Raises:
        ValueError: If required columns are missing.
    """
    bid_vol_cols = [f"v{i}" for i in range(6)]
    ask_vol_cols = [f"v{i}" for i in range(6, 12)]

    bid_vol = df[bid_vol_cols].sum(axis=1)
    ask_vol = df[ask_vol_cols].sum(axis=1)

    bid_momentum = _lagged_change(bid_vol, lag=lag, seq_ix=df["seq_ix"])
    ask_momentum = _lagged_change(ask_vol, lag=lag, seq_ix=df["seq_ix"])

    return bid_momentum - ask_momentum
