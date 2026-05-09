# tests/profiling/test_temporal.py
import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt


def test_autocorrelation_lag0_is_one(synthetic_df):
    from profiling.temporal import autocorrelation
    result = autocorrelation(synthetic_df, "p1", max_lag=5)
    lag0 = result.loc[result["lag"] == 0, "mean_autocorr"].values[0]
    assert lag0 == pytest.approx(1.0, abs=1e-6)


def test_autocorrelation_white_noise_near_zero(synthetic_df):
    from profiling.temporal import autocorrelation
    result = autocorrelation(synthetic_df, "p1", max_lag=10)
    lagged_abs = result.loc[result["lag"] > 0, "mean_autocorr"].abs()
    assert lagged_abs.mean() < 0.1


def test_autocorrelation_columns(synthetic_df):
    from profiling.temporal import autocorrelation
    result = autocorrelation(synthetic_df, "p0", max_lag=5)
    assert set(result.columns) == {"lag", "mean_autocorr", "ci_low", "ci_high"}
    assert len(result) == 6  # lags 0..5


def test_per_sequence_stat_count(synthetic_df):
    from profiling.temporal import per_sequence_stat
    result = per_sequence_stat(synthetic_df, "v0", agg="mean")
    assert len(result) == 50  # one value per sequence


def test_per_sequence_stat_mean_value():
    from profiling.temporal import per_sequence_stat
    df = pd.DataFrame({
        "seq_ix": [0, 0, 1, 1],
        "x": [1.0, 3.0, 2.0, 4.0],
    })
    result = per_sequence_stat(df, "x", agg="mean")
    assert result[0] == pytest.approx(2.0)
    assert result[1] == pytest.approx(3.0)


def test_per_sequence_stat_range():
    from profiling.temporal import per_sequence_stat
    df = pd.DataFrame({
        "seq_ix": [0, 0, 1, 1],
        "x": [1.0, 5.0, 2.0, 8.0],
    })
    result = per_sequence_stat(df, "x", agg="range")
    assert result[0] == pytest.approx(4.0)
    assert result[1] == pytest.approx(6.0)


def test_plot_feature_trajectories_returns_figure(synthetic_df):
    from profiling.temporal import plot_feature_trajectories
    from profiling.io import sample_sequences
    seq_ixs = sample_sequences(synthetic_df, n=4)
    fig = plot_feature_trajectories(synthetic_df, ["p0", "p1"], seq_ixs)
    assert isinstance(fig, plt.Figure)
    plt.close("all")


def test_plot_autocorrelation_returns_figure(synthetic_df):
    from profiling.temporal import autocorrelation, plot_autocorrelation
    acf_df = autocorrelation(synthetic_df, "p0", max_lag=20)
    fig = plot_autocorrelation(acf_df, "p0")
    assert isinstance(fig, plt.Figure)
    plt.close("all")


def test_plot_per_sequence_stat_distribution_returns_figure(synthetic_df):
    from profiling.temporal import per_sequence_stat, plot_per_sequence_stat_distribution
    stat = per_sequence_stat(synthetic_df, "v0", agg="mean")
    fig = plot_per_sequence_stat_distribution(stat, "v0", "mean")
    assert isinstance(fig, plt.Figure)
    plt.close("all")
