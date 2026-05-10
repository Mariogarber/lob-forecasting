"""Tests for feature engineering utility functions.

Tests _lagged_change helper function for computing lagged differences
while respecting sequence boundaries.
"""

import pandas as pd
import pytest
import numpy as np
from src.feature_engineering.utils import _lagged_change


def test_lagged_change_basic(synthetic_df):
    """Lagged change: s(t) - s(t-1)."""
    result = _lagged_change(synthetic_df["p0"], lag=1, seq_ix=synthetic_df["seq_ix"])
    assert isinstance(result, pd.Series)
    assert len(result) == len(synthetic_df)
    assert pd.isna(result.iloc[0])  # First step of seq 0
    assert pd.isna(result.iloc[100])  # First step of seq 1 (should be row 100 in 50×100 fixture)


def test_lagged_change_calculation(synthetic_df):
    """Lagged change is correctly calculated within sequences."""
    result = _lagged_change(synthetic_df["p0"], lag=1, seq_ix=synthetic_df["seq_ix"])
    # Check second step of first sequence (index 1)
    expected = synthetic_df.loc[1, "p0"] - synthetic_df.loc[0, "p0"]
    assert abs(result.iloc[1] - expected) < 1e-6


def test_lagged_change_respects_sequences(synthetic_df):
    """Lagged change does not cross sequence boundaries."""
    result = _lagged_change(synthetic_df["p0"], lag=1, seq_ix=synthetic_df["seq_ix"])
    # Step 100 is first step of seq 1, should be NaN
    assert pd.isna(result.iloc[100])
    # But step 101 should have a valid value
    assert not pd.isna(result.iloc[101])


def test_lagged_change_lag_2(synthetic_df):
    """Lagged change with lag=2."""
    result = _lagged_change(synthetic_df["p0"], lag=2, seq_ix=synthetic_df["seq_ix"])
    assert pd.isna(result.iloc[0])
    assert pd.isna(result.iloc[1])
    expected = synthetic_df.loc[2, "p0"] - synthetic_df.loc[0, "p0"]
    assert abs(result.iloc[2] - expected) < 1e-6
