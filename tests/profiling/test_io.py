# tests/profiling/test_io.py  (full file)
import pytest
import numpy as np


def test_constants_import():
    from profiling.constants import (
        BID_PRICES, ASK_PRICES, BID_VOLUMES, ASK_VOLUMES,
        TRADE_PRICES, TRADE_VOLUMES, TARGETS, META, ALL_FEATURES,
    )
    assert len(BID_PRICES) == 6
    assert len(ASK_PRICES) == 6
    assert len(BID_VOLUMES) == 6
    assert len(ASK_VOLUMES) == 6
    assert len(TRADE_PRICES) == 4
    assert len(TRADE_VOLUMES) == 4
    assert TARGETS == ["t0", "t1"]
    assert len(ALL_FEATURES) == 32


def test_dataset_overview_counts(synthetic_df):
    from profiling.io import dataset_overview
    ov = dataset_overview(synthetic_df)
    assert ov["n_rows"] == 5000
    assert ov["n_sequences"] == 50
    assert ov["steps_per_sequence"] == 100


def test_sample_sequences_reproducible(synthetic_df):
    from profiling.io import sample_sequences
    a = sample_sequences(synthetic_df, n=6, seed=42)
    b = sample_sequences(synthetic_df, n=6, seed=42)
    assert a == b


def test_sample_sequences_returns_n(synthetic_df):
    from profiling.io import sample_sequences
    result = sample_sequences(synthetic_df, n=8, seed=0)
    assert len(result) == 8
    assert len(set(result)) == 8  # no duplicates


def test_filter_sequences(synthetic_df):
    from profiling.io import filter_sequences
    seq_ixs = [0, 1, 2]
    result = filter_sequences(synthetic_df, seq_ixs)
    assert set(result["seq_ix"].unique()) == {0, 1, 2}
    assert len(result) == 300  # 3 seqs x 100 steps


def test_filter_sequences_empty_raises(synthetic_df):
    from profiling.io import filter_sequences
    with pytest.raises(ValueError, match="Empty DataFrame"):
        filter_sequences(synthetic_df, [9999])


def test_validate_columns_raises_on_missing(synthetic_df):
    from profiling.io import _validate_columns
    with pytest.raises(KeyError, match="nonexistent_col"):
        _validate_columns(synthetic_df, ["p0", "nonexistent_col"])


def test_dataset_overview_nan_report(synthetic_df):
    from profiling.io import dataset_overview
    ov = dataset_overview(synthetic_df)
    assert ov["nan_counts"]["p0"] == 10  # fixture injects 10 NaNs in p0
