"""Submission packaging: a `solution.zip` ready to upload."""

from submission.prediction_model import (
    SequencePredictionModel,
    ClassicalPredictionModel,
    EnsemblePredictionModel,
    load_member,
)

# ``packager`` is a build-time-only module: it pulls in the training stack and
# is deliberately NOT copied into the ``lob_runtime/`` subset shipped inside
# ``solution.zip``. Importing it lazily keeps the runtime ``submission`` package
# importable at inference time (where only the PredictionModel wrappers exist).
try:
    from submission.packager import package_run, package_ensemble
except ModuleNotFoundError:  # pragma: no cover - runtime (zip) has no packager.py
    package_run = None
    package_ensemble = None

__all__ = [
    "SequencePredictionModel",
    "ClassicalPredictionModel",
    "EnsemblePredictionModel",
    "load_member",
    "package_run",
    "package_ensemble",
]
