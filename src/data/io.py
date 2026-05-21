"""Parquet loaders for the Wunder Predictorium dataset.

Thin wrappers over `pd.read_parquet` that always:
  * use the pyarrow engine,
  * downcast float64 -> float32 (halves memory; predictions live in float32
    anyway and the targets are normalised in [-6, 6]),
  * verify the schema matches what the pipeline expects.

The actual splits live in `data.splits`; this module just gets bytes to
DataFrames.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from data.constants import FEATURE_COLS, META_COLS, TARGET_COLS

# Default location relative to the repo root.
DEFAULT_TRAIN = Path("competition_package/datasets/train.parquet")
DEFAULT_VALID = Path("competition_package/datasets/valid.parquet")


def _ensure_schema(df: pd.DataFrame, *, require_targets: bool) -> None:
    missing_features = [c for c in FEATURE_COLS if c not in df.columns]
    if missing_features:
        raise KeyError(
            f"Dataset is missing required feature columns: {missing_features[:5]}..."
        )
    missing_meta = [c for c in META_COLS if c not in df.columns]
    if missing_meta:
        raise KeyError(
            f"Dataset is missing required metadata columns: {missing_meta}"
        )
    if require_targets:
        missing_targets = [c for c in TARGET_COLS if c not in df.columns]
        if missing_targets:
            raise KeyError(
                f"Dataset is missing required target columns: {missing_targets}"
            )


def load_dataset(
    path: str | Path,
    *,
    downcast_floats: bool = True,
    require_targets: bool = True,
) -> pd.DataFrame:
    """Read a parquet file, downcast floats and validate the schema."""
    df = pd.read_parquet(path, engine="pyarrow")
    if downcast_floats:
        float_cols = df.select_dtypes("float64").columns
        df[float_cols] = df[float_cols].astype("float32")
    # pyarrow 24 + pandas 3.0 surface ``need_prediction`` as int8 instead of
    # bool. Coerce here so downstream code (``df[df['need_prediction']]``,
    # masks, etc.) sees a real boolean dtype.
    if "need_prediction" in df.columns and df["need_prediction"].dtype != bool:
        df["need_prediction"] = df["need_prediction"].astype(bool)
    _ensure_schema(df, require_targets=require_targets)
    # Sequences are independent; sorting puts them in canonical order
    # (the README guarantees within-sequence ordering by time, so we just
    # need to ensure step_in_seq is monotone inside each seq_ix).
    df = df.sort_values(["seq_ix", "step_in_seq"], kind="stable").reset_index(drop=True)
    return df


def load_train_valid(
    train_path: str | Path | None = None,
    valid_path: str | Path | None = None,
    **kwargs,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load the official train / valid parquets in one call."""
    train_path = Path(train_path or DEFAULT_TRAIN)
    valid_path = Path(valid_path or DEFAULT_VALID)
    return load_dataset(train_path, **kwargs), load_dataset(valid_path, **kwargs)


def load_subset(
    path: str | Path,
    n_seqs: int | None = None,
    *,
    seed: int = 0,
    downcast_floats: bool = True,
    require_targets: bool = True,
) -> pd.DataFrame:
    """Load only ``n_seqs`` random sequences from a parquet, with filter pushdown.

    Memory saver for low-RAM machines: pyarrow only reads the row groups whose
    ``seq_ix`` matches the chosen subset, so we never materialise the full
    dataframe in memory. Saves ~92% of RAM when reading 800 of 10 721 sequences.

    Falls back to a regular ``load_dataset`` when ``n_seqs`` is None or larger
    than the total number of sequences in the file.
    """
    path = Path(path)
    if n_seqs is None:
        return load_dataset(path, downcast_floats=downcast_floats, require_targets=require_targets)

    # Step 1: cheap scan to discover all available seq_ix without loading features.
    pf = pq.ParquetFile(path)
    seq_table = pf.read(columns=["seq_ix"])
    unique_seqs = pd.unique(seq_table.column("seq_ix").to_numpy())
    del seq_table

    if n_seqs >= len(unique_seqs):
        return load_dataset(path, downcast_floats=downcast_floats, require_targets=require_targets)

    rng = np.random.default_rng(seed)
    sampled = rng.choice(unique_seqs, size=n_seqs, replace=False)

    # Step 2: read with a pushed-down `seq_ix IN (...)` filter.
    table = pq.read_table(
        path,
        filters=[("seq_ix", "in", list(map(int, sampled)))],
    )
    df = table.to_pandas()
    del table

    if downcast_floats:
        float_cols = df.select_dtypes("float64").columns
        df[float_cols] = df[float_cols].astype("float32")
    if "need_prediction" in df.columns and df["need_prediction"].dtype != bool:
        df["need_prediction"] = df["need_prediction"].astype(bool)
    _ensure_schema(df, require_targets=require_targets)
    return df.sort_values(["seq_ix", "step_in_seq"], kind="stable").reset_index(drop=True)
