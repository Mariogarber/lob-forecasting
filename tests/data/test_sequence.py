"""Tests for the sequence-data pipeline."""

import numpy as np
import torch

from data import (
    FEATURE_COLS,
    SequenceDataset,
    pad_collate,
    sequences_from_dataframe,
    split_by_seq_ix,
)
from data.constants import SEQ_LEN, WARMUP_STEPS


def test_sequences_have_expected_shape(tiny_lob_df):
    arrays = sequences_from_dataframe(tiny_lob_df)
    n_seq = tiny_lob_df["seq_ix"].nunique()
    assert arrays.features.shape == (n_seq, SEQ_LEN, len(FEATURE_COLS))
    assert arrays.targets.shape == (n_seq, SEQ_LEN, 2)
    assert arrays.mask.shape == (n_seq, SEQ_LEN)


def test_mask_excludes_warmup(tiny_lob_df):
    arrays = sequences_from_dataframe(tiny_lob_df)
    # warm-up rows (step < 99) should be False, the rest True
    assert (arrays.mask[:, :WARMUP_STEPS] == False).all()
    assert (arrays.mask[:, WARMUP_STEPS:] == True).all()


def test_split_by_seq_ix_disjoint(tiny_lob_df):
    train_df, val_df = split_by_seq_ix(tiny_lob_df, val_fraction=0.25, seed=0)
    train_ids = set(train_df["seq_ix"].unique())
    val_ids = set(val_df["seq_ix"].unique())
    assert train_ids.isdisjoint(val_ids)
    assert train_ids | val_ids == set(tiny_lob_df["seq_ix"].unique())


def test_sequence_dataset_collate(tiny_lob_df):
    arrays = sequences_from_dataframe(tiny_lob_df)
    ds = SequenceDataset(arrays)
    assert not ds.augmented
    batch = pad_collate([ds[i] for i in range(min(3, len(ds)))])
    assert batch["features"].shape == (3, SEQ_LEN, len(FEATURE_COLS))
    assert batch["targets"].shape == (3, SEQ_LEN, 2)
    assert batch["mask"].dtype == torch.bool
    assert batch["seq_id"].shape == (3,)


def test_crop_augmentation_shape_and_warmup(tiny_lob_df):
    # Cropping returns fixed-length windows; the first `warmup` steps of each
    # crop are always unscored (cold-start context for the SSM/RNN state).
    arrays = sequences_from_dataframe(tiny_lob_df)
    crop = 300
    ds = SequenceDataset(arrays, crop_len=crop, warmup=WARMUP_STEPS, seed=0)
    assert ds.augmented
    item = ds[0]
    assert item["features"].shape == (crop, len(FEATURE_COLS))
    assert item["targets"].shape == (crop, 2)
    assert item["mask"].shape == (crop,)
    assert not item["mask"][:WARMUP_STEPS].any()      # warmup remasked off
    # Collate still stacks because every crop is the same length.
    batch = pad_collate([ds[i] for i in range(min(3, len(ds)))])
    assert batch["features"].shape == (3, crop, len(FEATURE_COLS))


def test_crop_is_random_across_draws(tiny_lob_df):
    # Different draws should pick different windows (stochastic across epochs).
    arrays = sequences_from_dataframe(tiny_lob_df)
    ds = SequenceDataset(arrays, crop_len=200, seed=0)
    draws = {ds[0]["features"][0, 0].item() for _ in range(20)}
    assert len(draws) > 1, "crop start is not varying across draws"


def test_jitter_perturbs_features_only(tiny_lob_df):
    arrays = sequences_from_dataframe(tiny_lob_df)
    ds = SequenceDataset(arrays, jitter_std=0.1, seed=0)
    assert ds.augmented
    clean = torch.from_numpy(arrays.features[0])
    noisy = ds[0]["features"]
    assert noisy.shape == clean.shape
    assert not torch.allclose(noisy, clean)            # features perturbed
    # targets/mask untouched by jitter
    torch.testing.assert_close(ds[0]["targets"], torch.from_numpy(arrays.targets[0]))
