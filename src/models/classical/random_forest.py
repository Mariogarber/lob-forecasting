"""Random Forest regressor baseline.

One forest per target (sklearn does multi-output natively, but training
two separate forests with per-sample weights is more flexible and
matches the rest of the classical-model contract).
"""

from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor

from models.base import ClassicalModel
from models import register_model


@register_model("random_forest", kind="classical")
class RandomForestModel(ClassicalModel):
    def _new_estimator(self):
        return RandomForestRegressor(
            n_estimators=self.config.get("n_estimators", 300),
            max_depth=self.config.get("max_depth", None),
            min_samples_leaf=self.config.get("min_samples_leaf", 32),
            n_jobs=self.config.get("n_jobs", -1),
            random_state=self.config.get("random_state", 0),
        )
