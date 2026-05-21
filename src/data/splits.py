"""Sequence-aware splits. Always split by `seq_ix`, never inside a sequence.

The competition guarantees sequences are independent and shuffled, so a
random hold-out by `seq_ix` is the right default. Two helpers are exposed:

  * `split_by_seq_ix` — single hold-out fraction (e.g. 90% / 10%).
  * `kfold_seq_ix`   — K-fold generator yielding (train_df, val_df) pairs.

Both helpers respect a `seed` for reproducibility and never leak rows from
a sequence between train and val.
"""

from __future__ import annotations

from typing import Iterator, Tuple

import numpy as np
import pandas as pd


def _seq_ids(df: pd.DataFrame) -> np.ndarray:
    return df["seq_ix"].unique()


def split_by_seq_ix(
    df: pd.DataFrame,
    val_fraction: float = 0.1,
    seed: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < val_fraction < 1:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")

    rng = np.random.default_rng(seed)
    seqs = _seq_ids(df)
    perm = rng.permutation(seqs)
    n_val = max(1, int(round(len(perm) * val_fraction)))
    val_seqs = set(perm[:n_val].tolist())

    val_mask = df["seq_ix"].isin(val_seqs)
    return df[~val_mask].reset_index(drop=True), df[val_mask].reset_index(drop=True)


def kfold_seq_ix(
    df: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 0,
) -> Iterator[Tuple[pd.DataFrame, pd.DataFrame]]:
    if n_splits < 2:
        raise ValueError(f"n_splits must be >= 2, got {n_splits}")

    rng = np.random.default_rng(seed)
    seqs = _seq_ids(df)
    perm = rng.permutation(seqs)
    folds = np.array_split(perm, n_splits)
    for fold_idx in range(n_splits):
        val_seqs = set(folds[fold_idx].tolist())
        val_mask = df["seq_ix"].isin(val_seqs)
        yield (
            df[~val_mask].reset_index(drop=True),
            df[val_mask].reset_index(drop=True),
        )
