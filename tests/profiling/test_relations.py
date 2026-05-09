# tests/profiling/test_relations.py
import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt


def test_correlation_matrix_self_correlation():
    from profiling.relations import correlation_matrix
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    corr = correlation_matrix(df, ["a", "b"])
    assert corr.loc["a", "a"] == pytest.approx(1.0)
    assert corr.loc["b", "b"] == pytest.approx(1.0)


def test_correlation_matrix_independent_columns():
    from profiling.relations import correlation_matrix
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "x": rng.standard_normal(10_000),
        "y": rng.standard_normal(10_000),
    })
    corr = correlation_matrix(df, ["x", "y"])
    assert abs(corr.loc["x", "y"]) < 0.05


def test_correlation_matrix_shape():
    from profiling.relations import correlation_matrix
    from profiling.constants import BID_PRICES
    df = pd.DataFrame(np.random.randn(100, 6), columns=BID_PRICES)
    corr = correlation_matrix(df, BID_PRICES)
    assert corr.shape == (6, 6)


def test_feature_target_correlations_shape(synthetic_df):
    from profiling.relations import feature_target_correlations
    from profiling.constants import ALL_FEATURES, TARGETS
    result = feature_target_correlations(synthetic_df, ALL_FEATURES, TARGETS)
    assert result.shape == (len(ALL_FEATURES), len(TARGETS))
    assert list(result.columns) == TARGETS
    assert list(result.index) == ALL_FEATURES


def test_feature_target_correlations_range(synthetic_df):
    from profiling.relations import feature_target_correlations
    from profiling.constants import ALL_FEATURES, TARGETS
    result = feature_target_correlations(synthetic_df, ALL_FEATURES, TARGETS)
    assert (result.abs() <= 1.0).all().all()


def test_plot_correlation_heatmap_returns_figure(synthetic_df):
    from profiling.relations import correlation_matrix, plot_correlation_heatmap
    from profiling.constants import BID_PRICES
    corr = correlation_matrix(synthetic_df, BID_PRICES)
    fig = plot_correlation_heatmap(corr)
    assert isinstance(fig, plt.Figure)
    plt.close("all")


def test_plot_feature_target_scatter_returns_figure(synthetic_df):
    from profiling.relations import plot_feature_target_scatter
    fig = plot_feature_target_scatter(synthetic_df, "p0", "t0", sample=200)
    assert isinstance(fig, plt.Figure)
    plt.close("all")
