"""YAML config loader for the pipeline.

Schema (one file per model, see ``configs/models/<name>.yaml``)::

    model:
      name: gru                 # registry name
      params:                   # constructor kwargs (n_features/n_targets injected)
        hidden_size: 128
        num_layers: 2
        dropout: 0.1

    data:
      train_path: competition_package/datasets/train.parquet
      valid_path: competition_package/datasets/valid.parquet
      val_fraction: 0.0         # 0 = use the official valid split
      seed: 0
      scaling: standard         # standard | robust | none

    training:                   # sequence-model only
      epochs: 20
      batch_size: 32
      learning_rate: 1e-3
      weight_decay: 1e-4
      warmup_epochs: 1
      grad_clip: 1.0
      eval_every: 1
      device: auto
      amp: true
      seed: 0
      loss_name: weighted_pearson
      loss_kwargs: {}
      early_stopping_patience: 5

    classical:                  # classical-model only
      with_engineered: true
      rolling_windows: [5, 20]
      only_scored_rows: true
      drop_warmup: true
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class RunConfig:
    raw: dict[str, Any]
    config_path: Path | None = None

    @property
    def model(self) -> dict[str, Any]:
        return self.raw.get("model", {})

    @property
    def data(self) -> dict[str, Any]:
        return self.raw.get("data", {})

    @property
    def training(self) -> dict[str, Any]:
        return self.raw.get("training", {})

    @property
    def classical(self) -> dict[str, Any]:
        return self.raw.get("classical", {})


def load_config(path: str | Path) -> RunConfig:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open() as fh:
        raw = yaml.safe_load(fh) or {}
    return RunConfig(raw=raw, config_path=path)


def deep_merge(base: dict, override: dict) -> dict:
    """Merge ``override`` into ``base`` (returns a new dict)."""
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out
