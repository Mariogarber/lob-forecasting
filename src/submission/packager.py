"""Build a ``solution.zip`` from a trained-model run directory.

The competition expects the zip to contain ``solution.py`` at root level
plus any auxiliary files (weights, scalers). We bundle:

  * ``solution.py``  — generated entry point that instantiates the
    right ``PredictionModel`` wrapper depending on model kind,
  * ``utils.py``     — copied from the competition_package (the official
    DataPoint dataclass that the scorer imports),
  * ``model.<ext>``  — model artefact (torch state_dict or joblib pickle),
  * ``scaler.json``  — feature scaler state, if one was used,
  * ``meta.json``    — model class name, config, feature columns,
  * ``lob_runtime/`` — minimal subset of the ``src/`` tree needed at
    inference time (data.constants, models.base, etc.).

Building this manually keeps the zip small (~few MB instead of dragging
the entire repo) and avoids accidentally shipping training-only code.
"""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path
from textwrap import dedent
from typing import Any

# Files copied into lob_runtime/ — keep this list short!
_RUNTIME_FILES = [
    ("data/constants.py", "data/constants.py"),
    ("data/scaling.py", "data/scaling.py"),
    ("models/base.py", "models/base.py"),
    ("models/__init__.py", "models/__init__.py"),
    ("models/sequence/__init__.py", "models/sequence/__init__.py"),
    ("models/sequence/_common.py", "models/sequence/_common.py"),
    ("models/sequence/gru.py", "models/sequence/gru.py"),
    ("models/sequence/lstm.py", "models/sequence/lstm.py"),
    ("models/sequence/transformer.py", "models/sequence/transformer.py"),
    ("models/sequence/deeplob.py", "models/sequence/deeplob.py"),
    ("models/sequence/mamba2.py", "models/sequence/mamba2.py"),
    ("models/sequence/tcn.py", "models/sequence/tcn.py"),
    ("models/sequence/tlob.py", "models/sequence/tlob.py"),
    ("models/classical/__init__.py", "models/classical/__init__.py"),
    ("models/classical/linear.py", "models/classical/linear.py"),
    ("models/classical/ridge.py", "models/classical/ridge.py"),
    ("models/classical/random_forest.py", "models/classical/random_forest.py"),
    ("models/classical/lightgbm_model.py", "models/classical/lightgbm_model.py"),
    ("feature_engineering/__init__.py", "feature_engineering/__init__.py"),
    ("feature_engineering/microstructure.py", "feature_engineering/microstructure.py"),
    ("feature_engineering/dynamics.py", "feature_engineering/dynamics.py"),
    ("feature_engineering/utils.py", "feature_engineering/utils.py"),
    ("submission/prediction_model.py", "submission/prediction_model.py"),
    ("submission/__init__.py", "submission/__init__.py"),
    ("metrics/__init__.py", "metrics/__init__.py"),
    ("metrics/pearson.py", "metrics/pearson.py"),
    ("losses/__init__.py", "losses/__init__.py"),
    ("losses/pearson.py", "losses/pearson.py"),
    ("losses/mse.py", "losses/mse.py"),
    ("losses/composite.py", "losses/composite.py"),
]

_SOLUTION_TEMPLATE = dedent(
    '''\
    """Auto-generated submission entry point. Do not edit by hand."""

    import json
    import os
    import sys
    from pathlib import Path

    HERE = Path(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, str(HERE / "lob_runtime"))

    import numpy as np
    from utils import DataPoint  # competition-provided

    with (HERE / "meta.json").open() as fh:
        META = json.load(fh)

    MODEL_KIND = META["kind"]
    MODEL_NAME = META["model_name"]
    SCALER_PATH = HERE / "scaler.json" if (HERE / "scaler.json").exists() else None

    # Force eager import of all model modules so the registry is populated.
    from models import get_model_class  # noqa: E402

    if MODEL_KIND == "sequence":
        from submission.prediction_model import SequencePredictionModel
        from data.scaling import FeatureScaler
        import torch

        ModelCls, _ = get_model_class(MODEL_NAME)
        ckpt = torch.load(HERE / "model.pt", map_location="cpu", weights_only=False)
        model = ModelCls(ckpt["config"])
        model.load_state_dict(ckpt["state_dict"])

        scaler = None
        if SCALER_PATH is not None:
            with SCALER_PATH.open() as fh:
                scaler = FeatureScaler.from_state_dict(json.load(fh))

        _wrapper = SequencePredictionModel(model, scaler=scaler, device="cpu")

    elif MODEL_KIND == "classical":
        from submission.prediction_model import ClassicalPredictionModel
        import joblib

        ModelCls, _ = get_model_class(MODEL_NAME)
        blob = joblib.load(HERE / "model.joblib")
        model = ModelCls(blob["config"])
        model.estimators_ = blob["estimators"]
        model.feature_names_ = blob["feature_names"]
        model.target_names_ = blob["target_names"]

        _wrapper = ClassicalPredictionModel(
            model,
            feature_columns=META["feature_columns"],
            with_engineered=META.get("with_engineered", True),
            rolling_windows=tuple(META.get("rolling_windows", (5, 20))),
        )
    else:
        raise RuntimeError(f"unknown model kind: {MODEL_KIND!r}")


    class PredictionModel:
        def __init__(self):
            pass

        def predict(self, data_point: DataPoint):
            return _wrapper.predict(data_point)
    '''
)


_ENSEMBLE_SOLUTION_TEMPLATE = dedent(
    '''\
    """Auto-generated ENSEMBLE submission entry point. Do not edit by hand."""

    import json
    import os
    import sys
    from pathlib import Path

    HERE = Path(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, str(HERE / "lob_runtime"))

    from utils import DataPoint  # competition-provided

    # Force eager import of all model modules so the registry is populated.
    from models import get_model_class  # noqa: F401
    from submission.prediction_model import EnsemblePredictionModel, load_member

    with (HERE / "meta.json").open() as fh:
        META = json.load(fh)

    _members = [load_member(HERE / "members" / str(i)) for i in range(len(META["members"]))]
    _wrapper = EnsemblePredictionModel(_members, weights=META["weights"])


    class PredictionModel:
        def __init__(self):
            pass

        def predict(self, data_point: DataPoint):
            return _wrapper.predict(data_point)
    '''
)


def _write_member_dir(member_dir: Path, member: dict) -> dict:
    """Write one member's artefact + scaler + meta into ``member_dir``.

    ``member`` keys: kind, model_name, model_artifact, [scaler_state],
    [feature_columns], [with_engineered], [rolling_windows]. Returns the
    member's meta dict (also written to ``member_dir/meta.json``).
    """
    member_dir.mkdir(parents=True, exist_ok=True)
    kind = member["kind"]
    artifact_src = Path(member["model_artifact"])
    if kind == "sequence":
        shutil.copy2(artifact_src, member_dir / "model.pt")
        if member.get("scaler_state") is not None:
            with (member_dir / "scaler.json").open("w") as fh:
                json.dump(member["scaler_state"], fh)
    elif kind == "classical":
        shutil.copy2(artifact_src, member_dir / "model.joblib")
    else:
        raise ValueError(f"unknown member kind: {kind!r}")

    meta = {
        "kind": kind,
        "model_name": member["model_name"],
        "feature_columns": list(member.get("feature_columns") or []),
        "with_engineered": bool(member.get("with_engineered", True)),
        "rolling_windows": list(member.get("rolling_windows", (5, 20))),
    }
    with (member_dir / "meta.json").open("w") as fh:
        json.dump(meta, fh, indent=2)
    return meta


def package_ensemble(
    *,
    run_dir: str | Path,
    src_root: str | Path,
    utils_py: str | Path,
    members: list[dict],
    weights: list[float],
    output_zip: str | Path | None = None,
) -> Path:
    """Build a ``solution.zip`` that averages several trained members.

    ``members`` is a list of descriptors (see ``_write_member_dir``); ``weights``
    are the per-member blend weights (need not be normalised — the runtime
    ``EnsemblePredictionModel`` normalises them). The zip layout mirrors a
    single-model package but with a ``members/<i>/`` dir per member and a
    top-level ``meta.json`` recording the members + weights.
    """
    if len(members) != len(weights):
        raise ValueError(f"{len(members)} members but {len(weights)} weights")
    run_dir = Path(run_dir)
    pkg_dir = run_dir / "submission_pkg"
    if pkg_dir.exists():
        shutil.rmtree(pkg_dir)
    pkg_dir.mkdir(parents=True)

    _copy_runtime_files(Path(src_root), pkg_dir / "lob_runtime")
    (pkg_dir / "solution.py").write_text(_ENSEMBLE_SOLUTION_TEMPLATE)
    shutil.copy2(utils_py, pkg_dir / "utils.py")

    member_metas = []
    for i, member in enumerate(members):
        member_metas.append(_write_member_dir(pkg_dir / "members" / str(i), member))

    meta = {
        "kind": "ensemble",
        "members": [
            {"kind": m["kind"], "model_name": m["model_name"]} for m in member_metas
        ],
        "weights": [float(w) for w in weights],
    }
    with (pkg_dir / "meta.json").open("w") as fh:
        json.dump(meta, fh, indent=2)

    output_zip = Path(output_zip or (run_dir / "solution.zip"))
    if output_zip.exists():
        output_zip.unlink()
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in pkg_dir.rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(pkg_dir))

    return output_zip


def _copy_runtime_files(src_root: Path, runtime_dir: Path) -> None:
    for src_rel, dst_rel in _RUNTIME_FILES:
        src = src_root / src_rel
        if not src.exists():
            continue
        dst = runtime_dir / dst_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def package_run(
    *,
    run_dir: str | Path,
    src_root: str | Path,
    utils_py: str | Path,
    model_kind: str,
    model_name: str,
    model_artifact: str | Path,
    scaler_state: dict | None = None,
    feature_columns: list[str] | None = None,
    with_engineered: bool = True,
    rolling_windows: tuple[int, ...] = (5, 20),
    output_zip: str | Path | None = None,
) -> Path:
    run_dir = Path(run_dir)
    pkg_dir = run_dir / "submission_pkg"
    if pkg_dir.exists():
        shutil.rmtree(pkg_dir)
    pkg_dir.mkdir(parents=True)

    # --- runtime tree ---
    _copy_runtime_files(Path(src_root), pkg_dir / "lob_runtime")

    # --- solution.py + utils.py ---
    (pkg_dir / "solution.py").write_text(_SOLUTION_TEMPLATE)
    shutil.copy2(utils_py, pkg_dir / "utils.py")

    # --- model artifact ---
    artifact_src = Path(model_artifact)
    if model_kind == "sequence":
        shutil.copy2(artifact_src, pkg_dir / "model.pt")
    elif model_kind == "classical":
        shutil.copy2(artifact_src, pkg_dir / "model.joblib")
    else:
        raise ValueError(f"unknown model_kind: {model_kind!r}")

    # --- scaler ---
    if scaler_state is not None:
        with (pkg_dir / "scaler.json").open("w") as fh:
            json.dump(scaler_state, fh)

    # --- meta ---
    meta = {
        "kind": model_kind,
        "model_name": model_name,
        "feature_columns": list(feature_columns or []),
        "with_engineered": bool(with_engineered),
        "rolling_windows": list(rolling_windows),
    }
    with (pkg_dir / "meta.json").open("w") as fh:
        json.dump(meta, fh, indent=2)

    # --- zip it up ---
    output_zip = Path(output_zip or (run_dir / "solution.zip"))
    if output_zip.exists():
        output_zip.unlink()
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in pkg_dir.rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(pkg_dir))

    return output_zip
