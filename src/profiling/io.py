# src/profiling/io.py
from __future__ import annotations

import difflib
from typing import Sequence

import numpy as np
import pandas as pd


def _validate_columns(df: pd.DataFrame, columns: Sequence[str]) -> None:
    """Raise KeyError with a helpful suggestion if any column is missing."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        all_cols = list(df.columns)
        suggestions = {
            c: difflib.get_close_matches(c, all_cols, n=1, cutoff=0.6)
            for c in missing
        }
        msg_parts = []
        for c, sugg in suggestions.items():
            hint = f" (did you mean '{sugg[0]}'?)" if sugg else ""
            msg_parts.append(f"'{c}'{hint}")
        raise KeyError("Column(s) not found: " + ", ".join(msg_parts))


def load_dataset(path: str, downcast_floats: bool = True) -> pd.DataFrame:
    """Read a parquet dataset. Optionally downcast float64 to float32 (~50% memory)."""
    df = pd.read_parquet(path, engine="pyarrow")
    if downcast_floats:
        float_cols = df.select_dtypes("float64").columns
        df[float_cols] = df[float_cols].astype("float32")
    return df


def dataset_overview(df: pd.DataFrame) -> dict:
    """Return a summary dict: row/sequence counts, NaN report, memory usage."""
    n_rows = len(df)
    n_sequences = df["seq_ix"].nunique() if "seq_ix" in df.columns else None
    steps_per_seq = df.groupby("seq_ix")["step_in_seq"].count()
    nan_counts = df.isna().sum().to_dict()

    return {
        "n_rows": n_rows,
        "n_sequences": n_sequences,
        "steps_per_sequence": int(steps_per_seq.median()),
        "pct_need_prediction": (
            df["need_prediction"].mean() * 100 if "need_prediction" in df.columns else None
        ),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1_048_576, 1),
        "nan_counts": nan_counts,
    }


def sample_sequences(df: pd.DataFrame, n: int = 6, seed: int = 42) -> list[int]:
    """Return a reproducible sample of n unique seq_ix values."""
    all_seqs = list(df["seq_ix"].unique())
    rng = np.random.default_rng(seed)
    chosen = rng.choice(all_seqs, size=min(n, len(all_seqs)), replace=False)
    return list(map(int, chosen))


def filter_sequences(df: pd.DataFrame, seq_ixs: Sequence[int]) -> pd.DataFrame:
    """Filter to rows belonging to the given seq_ix values."""
    result = df[df["seq_ix"].isin(seq_ixs)].copy()
    if len(result) == 0:
        raise ValueError(
            f"Empty DataFrame after filtering to seq_ixs={list(seq_ixs)}. "
            "Check that these IDs exist in the dataset."
        )
    return result
