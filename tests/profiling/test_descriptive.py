# tests/profiling/test_descriptive.py
import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt


def test_feature_stats_known_values():
    from profiling.descriptive import feature_stats
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    stats = feature_stats(df, ["x"])
    assert stats.loc["x", "mean"] == pytest.approx(3.0)
    assert stats.loc["x", "min"] == 1.0
    assert stats.loc["x", "max"] == 5.0
    assert stats.loc["x", "n_nan"] == 0
    assert stats.loc["x", "n_zeros"] == 0


def test_feature_stats_nan_count():
    from profiling.descriptive import feature_stats
    df = pd.DataFrame({"x": [1.0, np.nan, 3.0, np.nan, 5.0]})
    stats = feature_stats(df, ["x"])
    assert stats.loc["x", "n_nan"] == 2


def test_feature_stats_zero_count():
    from profiling.descriptive import feature_stats
    df = pd.DataFrame({"x": [0.0, 0.0, 1.0, 2.0, 3.0]})
    stats = feature_stats(df, ["x"])
    assert stats.loc["x", "n_zeros"] == 2


def test_feature_stats_multiple_columns(synthetic_df):
    from profiling.descriptive import feature_stats
    from profiling.constants import BID_PRICES
    stats = feature_stats(synthetic_df, BID_PRICES)
    assert set(stats.index) == set(BID_PRICES)
    expected_cols = {"mean", "std", "min", "p1", "p25", "p50", "p75", "p99",
                     "max", "skew", "kurt", "n_zeros", "n_nan"}
    assert expected_cols.issubset(set(stats.columns))


def test_feature_stats_raises_on_missing_column(synthetic_df):
    from profiling.descriptive import feature_stats
    with pytest.raises(KeyError):
        feature_stats(synthetic_df, ["p0", "nonexistent"])


def test_plot_distributions_returns_figure(synthetic_df):
    from profiling.descriptive import plot_distributions
    from profiling.constants import BID_PRICES
    fig = plot_distributions(synthetic_df, BID_PRICES, sample=500)
    assert isinstance(fig, plt.Figure)
    plt.close("all")


def test_plot_target_distributions_returns_figure(synthetic_df):
    from profiling.descriptive import plot_target_distributions
    fig = plot_target_distributions(synthetic_df)
    assert isinstance(fig, plt.Figure)
    plt.close("all")
