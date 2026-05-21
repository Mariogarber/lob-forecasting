"""Data loading, splitting and windowing for the LOB pipeline.

The data layer is the single source of truth for:

  * which columns are features / targets / metadata
  * how the train / val parquet files become tensors
  * how sequences are split for cross-validation (always by `seq_ix`)
  * how features are scaled (fit on train, applied on val)
  * how a `DataPoint` stream is fed to a streaming model at inference time

All downstream code (models, trainers, evaluators) consumes these objects;
nothing else should call `pd.read_parquet` directly so that NaN-handling,
dtype downcasting and sequence-boundary semantics stay consistent.
"""

from data.constants import (
    FEATURE_COLS,
    TARGET_COLS,
    META_COLS,
    WARMUP_STEPS,
    SEQ_LEN,
    N_FEATURES,
    N_TARGETS,
)
from data.io import load_dataset, load_subset, load_train_valid
from data.splits import split_by_seq_ix, kfold_seq_ix
from data.scaling import FeatureScaler
from data.sequence import (
    SequenceDataset,
    sequences_from_dataframe,
    pad_collate,
)
from data.tabular import (
    build_tabular_features,
    make_tabular_dataset,
    TabularDataset,
)

__all__ = [
    "FEATURE_COLS",
    "TARGET_COLS",
    "META_COLS",
    "WARMUP_STEPS",
    "SEQ_LEN",
    "N_FEATURES",
    "N_TARGETS",
    "load_dataset",
    "load_subset",
    "load_train_valid",
    "split_by_seq_ix",
    "kfold_seq_ix",
    "FeatureScaler",
    "SequenceDataset",
    "sequences_from_dataframe",
    "pad_collate",
    "build_tabular_features",
    "make_tabular_dataset",
    "TabularDataset",
]
