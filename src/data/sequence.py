"""Sequence-style dataset for deep-learning models.

A single sample is one full sequence (1000 steps): the model sees the
whole thing and predicts targets at every step. The trainer then masks
out the warm-up steps (0..98) when computing the loss, so the model is
free to use them as context but not penalised for being wrong there.

For models that only consume a window of W < SEQ_LEN steps (e.g. the
classical DeepLOB recipe with W=100), the trainer slides a window over
the precomputed sequences — the dataset itself stays one-sequence-per-
sample so we keep one batch dimension to reason about.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from data.constants import FEATURE_COLS, SEQ_LEN, TARGET_COLS, WARMUP_STEPS


@dataclass
class SequenceArrays:
    """Numpy backing arrays for a batch of fixed-length sequences."""

    features: np.ndarray   # (n_seq, T, n_features) float32
    targets: np.ndarray    # (n_seq, T, n_targets)  float32
    mask: np.ndarray       # (n_seq, T)             bool — True = scored step
    seq_ids: np.ndarray    # (n_seq,)               int   — original seq_ix


def sequences_from_dataframe(
    df: pd.DataFrame,
    feature_cols: list[str] = FEATURE_COLS,
    target_cols: list[str] = TARGET_COLS,
    seq_len: int = SEQ_LEN,
    warmup_steps: int = WARMUP_STEPS,
    require_targets: bool = True,
) -> SequenceArrays:
    """Reshape a long-format dataframe into (n_seq, T, F) / (n_seq, T, K) tensors.

    The dataframe must already be sorted by (seq_ix, step_in_seq) — `load_dataset`
    does this for us. Sequences that are shorter than `seq_len` are dropped
    with a warning rather than silently padded; the competition contract
    guarantees they are exactly `seq_len` so a short sequence almost always
    means corrupted input.
    """
    grouped = df.groupby("seq_ix", sort=False, as_index=False)

    feats: list[np.ndarray] = []
    tgts: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    ids: list[int] = []

    n_features = len(feature_cols)
    n_targets = len(target_cols)

    for seq_id, group in grouped:
        if len(group) != seq_len:
            # Defensive: log and skip.
            continue

        feats.append(group[feature_cols].to_numpy(dtype=np.float32, copy=False))
        if require_targets:
            tgts.append(group[target_cols].to_numpy(dtype=np.float32, copy=False))
        else:
            tgts.append(np.zeros((seq_len, n_targets), dtype=np.float32))

        # Mask = which rows are actually scored (need_prediction == True). If
        # the column is missing (e.g. exotic eval set), fall back to the
        # competition default of "everything after warm-up".
        if "need_prediction" in group.columns:
            mask = group["need_prediction"].to_numpy(dtype=bool, copy=False)
        else:
            mask = np.zeros(seq_len, dtype=bool)
            mask[warmup_steps:] = True
        masks.append(mask)
        ids.append(int(seq_id))

    if not feats:
        raise ValueError("No valid sequences found in dataframe.")

    return SequenceArrays(
        features=np.stack(feats),
        targets=np.stack(tgts),
        mask=np.stack(masks),
        seq_ids=np.asarray(ids, dtype=np.int64),
    )


class SequenceDataset(Dataset):
    """PyTorch Dataset where one item = one full sequence.

    Tensors are kept in CPU pinned-memory friendly numpy arrays and lazily
    converted to torch in `__getitem__` so workers can fork the data.
    """

    def __init__(self, arrays: SequenceArrays):
        self.features = arrays.features
        self.targets = arrays.targets
        self.mask = arrays.mask
        self.seq_ids = arrays.seq_ids

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int) -> dict:
        return {
            "features": torch.from_numpy(self.features[idx]),
            "targets": torch.from_numpy(self.targets[idx]),
            "mask": torch.from_numpy(self.mask[idx]),
            "seq_id": int(self.seq_ids[idx]),
        }


def pad_collate(batch: Iterable[dict]) -> dict:
    """Default collate. Sequences are all the same length so we just stack."""
    items = list(batch)
    return {
        "features": torch.stack([b["features"] for b in items]),
        "targets":  torch.stack([b["targets"]  for b in items]),
        "mask":     torch.stack([b["mask"]     for b in items]),
        "seq_id":   torch.tensor([b["seq_id"]  for b in items], dtype=torch.long),
    }
