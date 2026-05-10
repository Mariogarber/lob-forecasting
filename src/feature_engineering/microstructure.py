"""Microstructure features for LOB analysis.

Extracts structural features from the limit order book:
- Spread: best-ask price - best-bid price (liquidity cost)
- Mid-price: (best-bid + best-ask) / 2 (reference price)
- Bid-ask volume imbalance: directional pressure indicator
- Depth imbalance by level: level-specific volume imbalance
"""

import pandas as pd


def spread(df: pd.DataFrame) -> pd.Series:
    """Calculate the bid-ask spread.

    Formula: p6 - p0 (best-ask price - best-bid price)

    Args:
        df: DataFrame with columns 'p0' (best-bid) and 'p6' (best-ask)

    Returns:
        Series with spread values. Never NaN if p0/p6 are present.

    Raises:
        ValueError: If required columns are missing.
    """
    return df["p6"] - df["p0"]


def mid_price(df: pd.DataFrame) -> pd.Series:
    """Calculate the mid-price (center of the spread).

    Formula: (p0 + p6) / 2

    Args:
        df: DataFrame with columns 'p0' (best-bid) and 'p6' (best-ask)

    Returns:
        Series with mid-price values. Never NaN if p0/p6 are present.

    Raises:
        ValueError: If required columns are missing.
    """
    return (df["p0"] + df["p6"]) / 2


def bid_ask_volume_imbalance(df: pd.DataFrame) -> pd.Series:
    """Calculate bid-ask volume imbalance.

    Formula: (sum(v0:v5) - sum(v6:v11)) / (sum(v0:v5) + sum(v6:v11))
    where v0:v5 are bid-side volumes and v6:v11 are ask-side volumes.

    Interpretation:
    - Values close to -1: more volume on bid side (buyers waiting)
    - Values close to +1: more volume on ask side (sellers waiting)
    - Values close to 0: balanced liquidity

    Args:
        df: DataFrame with columns 'v0' through 'v11' (volumes at 12 levels)

    Returns:
        Series with imbalance values in range [-1, 1]. Returns 0 if total volume is 0.

    Raises:
        ValueError: If required columns are missing.
    """
    bid_vol = df[[f"v{i}" for i in range(6)]].sum(axis=1)
    ask_vol = df[[f"v{i}" for i in range(6, 12)]].sum(axis=1)
    total = bid_vol + ask_vol
    numerator = bid_vol - ask_vol
    result = pd.Series(0.0, index=df.index)
    mask = total.abs() > 0.1
    result[mask] = numerator[mask] / total[mask]
    return result.clip(-1, 1)


def depth_imbalance_by_level(df: pd.DataFrame, level: int) -> pd.Series:
    """Calculate volume imbalance at a specific LOB level.

    Formula: (v[level] - v[level+6]) / (v[level] + v[level+6])
    where level=0 is best-bid/best-ask, level=5 is deepest level.

    Args:
        df: DataFrame with columns 'v0' through 'v11'
        level: Integer in range [0, 5] indicating the LOB level

    Returns:
        Series with imbalance values in range [-1, 1]. Returns 0 if sum of volumes is 0.

    Raises:
        ValueError: If level is outside [0, 5] or required columns are missing.
    """
    if not (0 <= level < 6):
        raise ValueError(f"level must be in 0..5, got {level}")

    bid_col = f"v{level}"
    ask_col = f"v{level + 6}"
    bid_vol = df[bid_col]
    ask_vol = df[ask_col]
    total = bid_vol + ask_vol
    numerator = bid_vol - ask_vol
    result = pd.Series(0.0, index=df.index)
    mask = total.abs() > 0.1
    result[mask] = numerator[mask] / total[mask]
    return result.clip(-1, 1)
