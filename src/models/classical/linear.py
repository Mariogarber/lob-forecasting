"""Linear regression baseline with sample weights.

A weighted least-squares fit; one estimator per target. With weighted
Pearson as the eval metric this is a surprisingly strong floor when
combined with the engineered features in ``feature_engineering`` —
classic finance result: a well-conditioned linear model on the right
features beats most NNs on short-horizon, low-SNR forecasting.
"""

from __future__ import annotations

from typing import Any

from sklearn.linear_model import LinearRegression

from models.base import ClassicalModel
from models import register_model


@register_model("linear", kind="classical")
class LinearModel(ClassicalModel):
    def _new_estimator(self):
        return LinearRegression(
            fit_intercept=self.config.get("fit_intercept", True),
            n_jobs=self.config.get("n_jobs", -1),
        )
