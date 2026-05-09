# tests/profiling/conftest.py  (full file)
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from profiling.constants import ALL_FEATURES, TARGETS


@pytest.fixture
def synthetic_df():
    """50 sequences x 100 steps, same column structure as real dataset."""
    rng = np.random.default_rng(0)
    n_seqs, n_steps = 50, 100
    n_rows = n_seqs * n_steps

    seq_ixs = np.repeat(np.arange(n_seqs), n_steps)
    steps   = np.tile(np.arange(n_steps), n_seqs)
    need_pred = steps >= 10

    data: dict = {
        "seq_ix": seq_ixs,
        "step_in_seq": steps,
        "need_prediction": need_pred,
    }
    for col in ALL_FEATURES + TARGETS:
        data[col] = rng.standard_normal(n_rows)

    df = pd.DataFrame(data)
    df.loc[:9, "p0"] = np.nan   # inject NaNs in p0 for first 10 rows
    return df
