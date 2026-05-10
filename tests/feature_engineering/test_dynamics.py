"""Tests for dynamics features.

Tests for:
- price_momentum()
- volume_momentum()
- bid_ask_momentum_diff()
"""

import pytest
import pandas as pd
import numpy as np
from src.feature_engineering.dynamics import (
    price_momentum,
    volume_momentum,
    bid_ask_momentum_diff,
)
from src.feature_engineering.microstructure import mid_price


class TestPriceMomentum:
    """Tests for price_momentum() function."""

    def test_price_momentum_basic(self, synthetic_df):
        """Price momentum is mid_price(t) - mid_price(t-1)."""
        result = price_momentum(synthetic_df)
        assert isinstance(result, pd.Series)
        assert len(result) == len(synthetic_df)
        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[100])  # First step of seq 1

    def test_price_momentum_calculation(self, synthetic_df):
        """Price momentum is correctly calculated."""
        result = price_momentum(synthetic_df)
        mid = mid_price(synthetic_df)
        expected = mid.iloc[1] - mid.iloc[0]
        assert abs(result.iloc[1] - expected) < 1e-6

    def test_price_momentum_lag_parameter(self, synthetic_df):
        """Price momentum respects lag parameter."""
        result_lag1 = price_momentum(synthetic_df, lag=1)
        result_lag2 = price_momentum(synthetic_df, lag=2)
        # Both should have NaNs at the start, but at different positions
        assert pd.isna(result_lag1.iloc[0])
        assert pd.isna(result_lag1.iloc[1]) is False or pd.isna(result_lag1.iloc[1])
        assert pd.isna(result_lag2.iloc[0])
        assert pd.isna(result_lag2.iloc[1])

    def test_price_momentum_respects_sequence_boundaries(self, synthetic_df):
        """Price momentum should be NaN at sequence boundaries."""
        result = price_momentum(synthetic_df)
        # seq 1 starts at index 100, so result.iloc[100] should be NaN
        assert pd.isna(result.iloc[100])
        # seq 2 starts at index 200, so result.iloc[200] should be NaN
        assert pd.isna(result.iloc[200])


class TestVolumeMomentum:
    """Tests for volume_momentum() function."""

    def test_volume_momentum_basic(self, synthetic_df):
        """Volume momentum is total_volume(t) - total_volume(t-1)."""
        result = volume_momentum(synthetic_df)
        assert isinstance(result, pd.Series)
        assert len(result) == len(synthetic_df)
        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[100])

    def test_volume_momentum_calculation(self, synthetic_df):
        """Volume momentum is correctly calculated."""
        result = volume_momentum(synthetic_df)
        vol_cols = [f"v{i}" for i in range(12)]
        total_vol = synthetic_df[vol_cols].sum(axis=1)
        expected = total_vol.iloc[1] - total_vol.iloc[0]
        assert abs(result.iloc[1] - expected) < 1e-6

    def test_volume_momentum_lag_parameter(self, synthetic_df):
        """Volume momentum respects lag parameter."""
        result_lag1 = volume_momentum(synthetic_df, lag=1)
        result_lag2 = volume_momentum(synthetic_df, lag=2)
        assert pd.isna(result_lag1.iloc[0])
        assert pd.isna(result_lag2.iloc[0])
        assert pd.isna(result_lag2.iloc[1])

    def test_volume_momentum_respects_sequence_boundaries(self, synthetic_df):
        """Volume momentum should be NaN at sequence boundaries."""
        result = volume_momentum(synthetic_df)
        assert pd.isna(result.iloc[100])
        assert pd.isna(result.iloc[200])


class TestBidAskMomentumDiff:
    """Tests for bid_ask_momentum_diff() function."""

    def test_bid_ask_momentum_diff_basic(self, synthetic_df):
        """Bid-ask momentum difference returns a Series."""
        result = bid_ask_momentum_diff(synthetic_df)
        assert isinstance(result, pd.Series)
        assert len(result) == len(synthetic_df)
        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[100])

    def test_bid_ask_momentum_diff_calculation(self, synthetic_df):
        """Bid-ask momentum difference is correctly calculated."""
        result = bid_ask_momentum_diff(synthetic_df)

        bid_vol_cols = [f"v{i}" for i in range(6)]
        ask_vol_cols = [f"v{i}" for i in range(6, 12)]

        bid_vol = synthetic_df[bid_vol_cols].sum(axis=1)
        ask_vol = synthetic_df[ask_vol_cols].sum(axis=1)

        bid_momentum_expected = bid_vol.iloc[1] - bid_vol.iloc[0]
        ask_momentum_expected = ask_vol.iloc[1] - ask_vol.iloc[0]
        expected = bid_momentum_expected - ask_momentum_expected

        assert abs(result.iloc[1] - expected) < 1e-6

    def test_bid_ask_momentum_diff_lag_parameter(self, synthetic_df):
        """Bid-ask momentum diff respects lag parameter."""
        result_lag1 = bid_ask_momentum_diff(synthetic_df, lag=1)
        result_lag2 = bid_ask_momentum_diff(synthetic_df, lag=2)
        assert pd.isna(result_lag1.iloc[0])
        assert pd.isna(result_lag2.iloc[0])
        assert pd.isna(result_lag2.iloc[1])

    def test_bid_ask_momentum_diff_respects_sequence_boundaries(self, synthetic_df):
        """Bid-ask momentum diff should be NaN at sequence boundaries."""
        result = bid_ask_momentum_diff(synthetic_df)
        assert pd.isna(result.iloc[100])
        assert pd.isna(result.iloc[200])
