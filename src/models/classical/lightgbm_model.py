"""LightGBM regressor with early stopping on weighted MSE.

The eval metric used during training is the (negative) weighted Pearson
on the validation set — same as the competition metric — so early
stopping picks the iteration that maximises the actual score.

The output of one estimator per target is the same convention as the
rest of the classical wrappers; LightGBM does not support multi-output
regression with per-sample weights in one call.
"""

from __future__ import annotations

from typing import Any

import numpy as np

import lightgbm as lgb

from metrics.pearson import _weighted_pearson_scalar  # type: ignore
from models.base import ClassicalModel
from models import register_model


def _lgb_weighted_pearson(y_pred: np.ndarray, dataset: "lgb.Dataset"):
    """Custom LightGBM eval function: higher is better."""
    y_true = dataset.get_label()
    corr = _weighted_pearson_scalar(y_true, y_pred)
    return "weighted_pearson", corr, True  # name, value, is_higher_better


@register_model("lightgbm", kind="classical")
class LightGBMModel(ClassicalModel):
    DEFAULT_PARAMS = {
        "objective": "regression",
        "metric": "None",
        "learning_rate": 0.03,
        "num_leaves": 127,
        "max_depth": -1,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "lambda_l2": 1.0,
        "verbose": -1,
    }

    def _params(self) -> dict[str, Any]:
        return {**self.DEFAULT_PARAMS, **(self.config.get("params") or {})}

    def _new_estimator(self):
        # We don't construct the booster eagerly because LightGBM training
        # is done through `lgb.train` in `_fit_one`. Return a placeholder
        # that the base class will overwrite.
        return None

    def _fit_one(self, est, X, y_col, sample_weight, *, eval_set, target_idx):
        params = self._params()
        train_data = lgb.Dataset(X, label=y_col, weight=sample_weight)
        eval_data = None
        valid_sets = [train_data]
        valid_names = ["train"]
        if eval_set is not None:
            X_val, y_val = eval_set
            y_val_col = y_val[:, target_idx]
            w_val = np.abs(y_val_col).clip(min=1e-8).astype(np.float32)
            eval_data = lgb.Dataset(X_val, label=y_val_col, weight=w_val, reference=train_data)
            valid_sets.append(eval_data)
            valid_names.append("valid")

        booster = lgb.train(
            params=params,
            train_set=train_data,
            num_boost_round=self.config.get("num_boost_round", 3000),
            valid_sets=valid_sets,
            valid_names=valid_names,
            feval=_lgb_weighted_pearson,
            callbacks=[
                lgb.log_evaluation(period=self.config.get("log_period", 100)),
                lgb.early_stopping(
                    stopping_rounds=self.config.get("early_stopping_rounds", 100),
                    first_metric_only=True,
                ),
            ],
        )
        # Replace placeholder with the trained booster.
        self.estimators_.append(booster)  # base class will overwrite this slot

    def fit(
        self,
        X,
        y,
        sample_weight=None,
        *,
        eval_set=None,
        feature_names=None,
        target_names=None,
    ):
        # Override to control the per-target loop ourselves so estimators_
        # gets the actual booster instances appended by `_fit_one`.
        self.estimators_ = []
        self.feature_names_ = list(feature_names or [])
        self.target_names_ = list(target_names or [f"y{i}" for i in range(y.shape[1])])
        for i in range(y.shape[1]):
            self._fit_one(
                None, X, y[:, i], sample_weight,
                eval_set=eval_set, target_idx=i,
            )
        return self

    def predict(self, X):
        if not self.estimators_:
            raise RuntimeError(f"{self.name}: model is not fitted.")
        preds = np.stack([est.predict(X) for est in self.estimators_], axis=1)
        return preds.astype(np.float32)
