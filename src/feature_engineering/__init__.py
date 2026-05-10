"""Feature engineering for LOB (Limit Order Book) prediction.

This module provides functions to extract engineered features from raw LOB data,
including microstructure features (spread, imbalance, mid-price) and dynamics
features (momentum, changes in volume/price).

Main entry point: `add_engineered_features(df)`
"""

import pandas as pd
from src.feature_engineering.microstructure import (
    spread, mid_price, bid_ask_volume_imbalance, depth_imbalance_by_level
)
from src.feature_engineering.dynamics import (
    price_momentum, volume_momentum, bid_ask_momentum_diff
)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all engineered features to a LOB DataFrame.

    Adds microstructure features (spread, imbalance, mid-price) and
    dynamics features (momentum). Returns original df plus new columns.

    Parameters
    ----------
    df : pd.DataFrame
        Raw LOB data with columns p0..p11, v0..v11, seq_ix.

    Returns
    -------
    pd.DataFrame
        Same df with new columns appended:
        - spread: p6 - p0
        - mid_price: (p0 + p6) / 2
        - bid_ask_volume_imbalance: (bid_vol - ask_vol) / (bid_vol + ask_vol)
        - depth_imbalance_level_0 to level_5: imbalance at each level
        - price_momentum: change in mid_price
        - volume_momentum: change in total volume
        - bid_ask_momentum_diff: momentum_bid - momentum_ask

    Notes
    -----
    - New columns are added to df in-place (actually returns a new df with added cols)
    - Momentum features have NaN in first step of each sequence
    - All functions are deterministic
    """
    result = df.copy()

    # Microstructure
    result["spread"] = spread(df)
    result["mid_price"] = mid_price(df)
    result["bid_ask_volume_imbalance"] = bid_ask_volume_imbalance(df)

    # Depth imbalance by level
    for level in range(6):
        result[f"depth_imbalance_level_{level}"] = depth_imbalance_by_level(df, level=level)

    # Dynamics
    result["price_momentum"] = price_momentum(df)
    result["volume_momentum"] = volume_momentum(df)
    result["bid_ask_momentum_diff"] = bid_ask_momentum_diff(df)

    return result


__all__ = [
    "add_engineered_features",
    "spread", "mid_price", "bid_ask_volume_imbalance", "depth_imbalance_by_level",
    "price_momentum", "volume_momentum", "bid_ask_momentum_diff",
]
