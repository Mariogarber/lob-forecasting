"""Submission packaging: a `solution.zip` ready to upload."""

from submission.prediction_model import (
    SequencePredictionModel,
    ClassicalPredictionModel,
)
from submission.packager import package_run

__all__ = [
    "SequencePredictionModel",
    "ClassicalPredictionModel",
    "package_run",
]
