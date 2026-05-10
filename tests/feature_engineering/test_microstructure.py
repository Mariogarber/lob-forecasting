"""Tests for microstructure features.

Tests for:
- spread()
- mid_price()
- bid_ask_volume_imbalance()
- depth_imbalance_by_level()
"""

import pytest
import pandas as pd
import numpy as np
from src.feature_engineering.microstructure import spread, mid_price, bid_ask_volume_imbalance, depth_imbalance_by_level


def test_spread_basic(synthetic_df):
    """Spread is ask (p6) minus bid (p0)."""
    result = spread(synthetic_df)
    assert isinstance(result, pd.Series)
    assert len(result) == len(synthetic_df)
    # Manual check on first row (avoiding NaN/edge cases)
    expected_first = synthetic_df.loc[0, "p6"] - synthetic_df.loc[0, "p0"]
    assert abs(result.iloc[0] - expected_first) < 1e-6


def test_spread_positive(synthetic_df):
    """Spread should be positive (ask > bid in normal markets)."""
    result = spread(synthetic_df)
    # Filter out NaN rows
    valid = result.dropna()
    assert (valid >= 0).all()


def test_mid_price_basic(synthetic_df):
    """Mid-price is (p0 + p6) / 2."""
    result = mid_price(synthetic_df)
    assert isinstance(result, pd.Series)
    assert len(result) == len(synthetic_df)
    # Manual check on first row
    expected_first = (synthetic_df.loc[0, "p0"] + synthetic_df.loc[0, "p6"]) / 2
    assert abs(result.iloc[0] - expected_first) < 1e-6


def test_mid_price_bounds(synthetic_df):
    """Mid-price should be between best bid and best ask."""
    mid = mid_price(synthetic_df)
    bid = synthetic_df["p0"]
    ask = synthetic_df["p6"]
    # Filter out rows with NaN prices
    valid_idx = ~(bid.isna() | ask.isna() | mid.isna())
    assert (mid[valid_idx] >= bid[valid_idx]).all()
    assert (mid[valid_idx] <= ask[valid_idx]).all()


class TestSpread:
    """Tests for spread() function."""

    pass


class TestMidPrice:
    """Tests for mid_price() function."""

    pass


class TestBidAskVolumeImbalance:
    """Tests for bid_ask_volume_imbalance() function."""

    pass


class TestDepthImbalanceByLevel:
    """Tests for depth_imbalance_by_level() function."""

    pass


def test_bid_ask_volume_imbalance_range(synthetic_df):
    """Imbalance is in [-1, 1]."""
    result = bid_ask_volume_imbalance(synthetic_df)
    # Filter out NaN values before checking bounds
    valid = result.dropna()
    assert (valid >= -1).all()
    assert (valid <= 1).all()


def test_bid_ask_volume_imbalance_balanced(synthetic_df):
    """When bid volume == ask volume, imbalance == 0."""
    df = pd.DataFrame([{f"v{i}": 10.0 for i in range(12)}])
    result = bid_ask_volume_imbalance(df)
    assert abs(result.iloc[0]) < 1e-6


def test_depth_imbalance_by_level_basic(synthetic_df):
    """Depth imbalance at level 0 is (v0 - v6) / (v0 + v6)."""
    result = depth_imbalance_by_level(synthetic_df, level=0)
    assert isinstance(result, pd.Series)
    assert len(result) == len(synthetic_df)
    expected = (synthetic_df.loc[0, "v0"] - synthetic_df.loc[0, "v6"]) / (synthetic_df.loc[0, "v0"] + synthetic_df.loc[0, "v6"])
    assert abs(result.iloc[0] - expected) < 1e-6


def test_depth_imbalance_by_level_range(synthetic_df):
    """Depth imbalance is in [-1, 1]."""
    for level in range(6):
        result = depth_imbalance_by_level(synthetic_df, level=level)
        # Filter out NaN values before checking bounds
        valid = result.dropna()
        assert (valid >= -1).all()
        assert (valid <= 1).all()


def test_add_engineered_features_basic(synthetic_df):
    """add_engineered_features returns df with all new columns."""
    from src.feature_engineering import add_engineered_features
    result = add_engineered_features(synthetic_df)

    # Check that all original columns are preserved
    for col in synthetic_df.columns:
        assert col in result.columns

    # Check that new columns were added
    expected_new_cols = [
        "spread", "mid_price", "bid_ask_volume_imbalance",
        "price_momentum", "volume_momentum", "bid_ask_momentum_diff"
    ]
    for col in expected_new_cols:
        assert col in result.columns
