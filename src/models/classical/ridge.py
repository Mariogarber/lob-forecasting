"""Ridge regression baseline (L2-regularised linear model)."""

from __future__ import annotations

from sklearn.linear_model import Ridge

from models.base import ClassicalModel
from models import register_model


@register_model("ridge", kind="classical")
class RidgeModel(ClassicalModel):
    def _new_estimator(self):
        return Ridge(
            alpha=self.config.get("alpha", 1.0),
            fit_intercept=self.config.get("fit_intercept", True),
            random_state=self.config.get("random_state", 0),
        )
