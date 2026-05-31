# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

Solution to the **Wunder Predictorium** challenge (`PROJECT_README.md`): predict two future
price-movement indicators (`t0`, `t1`) from sequences of Limit Order Book + trade states.
Repo now contains the **full training/evaluation/submission pipeline** plus 9 baseline models
(see below). Competition rules, domain background and the metric are in
`PROJECT_README.md`, `docs/dominio-lob.md`, `docs/metrica.md`.

## Environment & common commands

Python 3.12 in a dedicated conda env named `lob`. Create / refresh with:

```bash
bash scripts/install_env.sh    # creates env, installs torch+cu121, lightgbm, mambapy, ...
conda activate lob
```

The env path is `/home/eder/miniconda3/envs/lob`. For non-interactive runs use the absolute
interpreter `/home/eder/miniconda3/envs/lob/bin/python`. PyTorch is pinned to the CUDA 12.1
wheel index. The **`Mamba2` model is a self-contained pure-PyTorch Mamba-2 (SSD)** — it does
**not** require `mamba-ssm` / `causal-conv1d` (those are not installed and are awkward to
build under WSL2). The selective scan runs in fp32 even under AMP, so it is numerically
stable on consumer GPUs without any native kernels. `d_model` is unconstrained (the old
`{256, 512, 1024, ...}` kernel-strides limit no longer applies); only
`expand*d_model % headdim == 0` must hold.

Pinned versions: torch==2.5.1+cu121. (`mamba-ssm` / `causal-conv1d` / the `transformers`
trio are no longer needed by any model — the previous Mamba path that imported them has been
removed.)

```bash
pytest                                                   # full test suite (81 tests)
pytest tests/test_pipeline_smoke.py                      # end-to-end smoke tests
python -m pipeline.cli list                              # list registered models
python -m pipeline.cli train --config configs/models/lightgbm.yaml
python -m pipeline.cli score --run experiments/lightgbm/<run_id>
# The deliverable notebook (the project's main entry point for grading):
python scripts/make_notebook.py     # regenerate main.ipynb from its Python source
jupyter lab main.ipynb              # open it
```

## `main.ipynb` — the deliverable notebook

The notebook lives at the repo root and is the single artefact handed in for grading.
It is **regenerated** by `scripts/make_notebook.py` (nbformat builder); never hand-edit
`main.ipynb` directly — apply the change in `scripts/make_notebook.py` and re-run it.

Structure: 10 sections (intro · EDA · feature engineering · 3 iteration rounds · final
comparison · submission packaging · conclusions). The notebook is a thin orchestration
layer — every non-trivial operation is delegated to `src/` (data loaders, Trainer,
Evaluator, packager) so the codebase is the single source of truth.

A top-level `QUICK_MODE` flag (default `True`) trains a subset of sequences for a few
epochs (~10-20 min total); set it to `False` for the full submission run.

Data lives under `competition_package/datasets/` (gitignored). Tests use synthetic
fixtures and never touch the real parquet.

## Pipeline architecture

`src/` is on `sys.path` via `conftest.py`. **All imports are flat** (no `src.` prefix).
The pipeline lives entirely under `src/` in named subpackages; ad-hoc imports must
follow the same convention.

```
src/
├── data/                # parquet IO, splits by seq_ix, scaling, sequence windowing
├── feature_engineering/ # microstructure + dynamics features (pre-existing)
├── losses/              # weighted Pearson / MSE / composite
├── metrics/             # numpy-side metrics for reporting (mirrors utils.weighted_pearson_correlation)
├── models/
│   ├── base.py          # ClassicalModel / SequenceModel ABCs
│   ├── __init__.py      # MODEL_REGISTRY + @register_model decorator
│   ├── classical/       # linear, ridge, random_forest, lightgbm
│   └── sequence/        # gru, lstm, transformer, deeplob, mamba2
├── training/            # Trainer (DL) + train_classical_model
├── evaluation/          # Evaluator + plots + report writer
├── submission/          # solution.zip packager + streaming PredictionModel
├── pipeline/            # CLI + run orchestrator + YAML config loader
├── profiling/           # exploratory analysis (pre-existing)
└── utils.py             # competition stub (DataPoint, ScorerStepByStep)
```

## Model registry pattern — non-obvious

Adding a new model is **three lines + a YAML**:

```python
@register_model("my_model", kind="sequence")   # name + kind classical|sequence
class MyModel(SequenceModel):
    def __init__(self, config): super().__init__(config); ...
    def forward(self, x): ...                  # (B, T, F) -> (B, T, K)
```

Then import the module from `src/models/<kind>/__init__.py`'s `__init__.py` or directly
in `src/models/__init__.py` so the decorator fires at import time. The CLI dispatches on
the registry; no other plumbing is needed.

`SequenceModel` provides default `init_state` / `step` methods that buffer the
input window — override them for true streaming (RNN hidden state, Mamba state)
when latency matters.

## Loss / metric alignment — non-obvious

`losses.pearson.weighted_pearson_loss` uses **per-target weights**
(`w[n, k] = max(|y_true[n, k]|, eps)`), not a row-level max. This matches the
official `utils.weighted_pearson_correlation` byte-for-byte
(verified by `tests/losses/test_pearson.py::test_matches_numpy_reference`). Do not
"simplify" to a single weight vector — it silently breaks the metric alignment.

The scorer clips predictions to `[-6, 6]`. The training loss does the same via
`torch.clamp` inside the loss (gradients flow inside the range, are zero outside).

## Data conventions

`data.constants` re-exports the canonical column groups (mirror of
`profiling.constants` — kept duplicated so the training pipeline does not depend
on the profiling package). The 32 LOB features are
`BID_PRICES + ASK_PRICES + BID_VOLUMES + ASK_VOLUMES + TRADE_PRICES + TRADE_VOLUMES`
in that order. Sequences are exactly 1000 steps with the first 99 as warm-up; the
mask returned by `sequences_from_dataframe` excludes warm-up automatically.

Splits **must** go by `seq_ix`; use `data.splits.split_by_seq_ix` or `kfold_seq_ix`.
Never split rows within a sequence — leakage warning.

For classical models, `data.tabular.build_tabular_features` adds engineered
features (`feature_engineering.add_engineered_features`) and multi-window rolling
stats of mid-price / total volume. Sequence boundaries are respected through
`groupby('seq_ix').transform(rolling)`.

## Streaming inference

The competition scorer iterates row-by-row. `submission.prediction_model`
provides two wrappers:

- `SequencePredictionModel(model, scaler)` — uses `model.init_state` / `model.step`
  to advance state per step. Resets on new `seq_ix`. Applies the trained `FeatureScaler`.
- `ClassicalPredictionModel(model, feature_columns, with_engineered, rolling_windows)` —
  maintains a per-sequence deque of rows, re-runs `feature_engineering`, recomputes
  the rolling stats incrementally, and predicts on the last row.

Both clip predictions to `[-6, 6]` (matches the scorer) and return `None` when
`need_prediction == False`.

## Submission packaging

`submission.packager.package_run` generates `solution.zip` with this layout:

```
solution.zip
├── solution.py                # auto-generated, instantiates the right PredictionModel
├── utils.py                   # copied verbatim from competition_package/
├── model.pt OR model.joblib   # the trained artefact
├── scaler.json                # FeatureScaler state (sequence models only)
├── meta.json                  # kind, model_name, feature_columns, rolling_windows
└── lob_runtime/               # minimal src/ subset needed at inference time
    ├── data/{constants,scaling}.py
    ├── models/{base.py, __init__.py, classical/*, sequence/*}
    ├── feature_engineering/*
    ├── submission/prediction_model.py
    ├── metrics/* losses/*
```

`solution.py` is auto-generated. **Do not edit `_SOLUTION_TEMPLATE` casually** — the
official scorer imports the `PredictionModel` class from it, and the file must be at
the zip root. The list of files copied into `lob_runtime/` is `_RUNTIME_FILES`; when
adding a new model module, also add it to that list or `solution.zip` will fail at
import time.

## Tests

`pytest>=8.0`. 80 tests in total:

- `tests/feature_engineering/` — 25 tests, pre-existing.
- `tests/profiling/` — 34 tests, pre-existing.
- `tests/data/test_sequence.py` — 4 tests (sequence windowing, splits, collate).
- `tests/losses/test_pearson.py` — 4 tests (numpy/torch parity, perfect/inverse predictions).
- `tests/models/test_registry.py` — 9 tests (registry populated, forward shapes, streaming parity).
- `tests/test_pipeline_smoke.py` — 4 end-to-end tests (linear, lightgbm, gru, packager).

`tests/conftest.py` provides `tiny_lob_df` (4 sequences × 1000 steps of synthetic but
structurally-correct LOB data) and `tiny_features_targets` for fast iteration.

## LOB column conventions (must-know)

`p0..p5` = bid prices (best bid at `p0`, decreasing). `p6..p11` = ask prices
(best ask at `p6`, increasing). `v0..v5` and `v6..v11` are paired volumes.
`dp0..dp3`, `dv0..dv3` = trades. Depth-imbalance pairing is `level k -> (v[k], v[k+6])`.

`seq_ix` is independent between sequences after sorting. Each sequence has
exactly 1000 steps; steps 0–98 are warm-up (not scored), 99–999 are evaluated.
`need_prediction` marks scored rows.
