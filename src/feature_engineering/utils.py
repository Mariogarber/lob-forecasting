"""Utility functions for feature engineering.

Internal helpers for common operations used across microstructure and dynamics modules.
"""

import pandas as pd


def _lagged_change(series: pd.Series, lag: int = 1, seq_ix: pd.Series = None) -> pd.Series:
    """Calculate lagged change in a series by sequence.

    Formula: series(t) - series(t-lag)
    Respects sequence boundaries via groupby('seq_ix').

    Args:
        series: Series to compute lagged change on
        lag: Number of steps to look back (default: 1)
        seq_ix: Series with sequence indices. If None, assumes no sequence boundaries.

    Returns:
        Series with lagged changes. First 'lag' steps of each sequence are NaN.
        If seq_ix is provided, changes are computed within each sequence separately.

    Raises:
        ValueError: If lag <= 0.
    """
    if lag <= 0:
        raise ValueError(f"lag must be positive, got {lag}")

    if seq_ix is None:
        # Simple case: no sequence boundaries, just shift and subtract
        return series - series.shift(lag)

    # Complex case: respect sequence boundaries
    # Shift both series and seq_ix by lag
    shifted_series = series.shift(lag)
    shifted_seq_ix = seq_ix.shift(lag)

    # Compute difference
    result = series - shifted_series

    # Invalidate where seq_ix changed (crossed boundary)
    seq_boundary = seq_ix != shifted_seq_ix
    result[seq_boundary] = pd.NA

    return result
