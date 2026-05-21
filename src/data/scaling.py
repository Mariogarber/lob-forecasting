"""Feature scaling for the LOB pipeline.

Scalers are stateful (fit on train, applied on val/test); they are
intentionally simple — robust to NaNs that show up in engineered features
during the first lag steps of a sequence.

Two modes are supported:

  * ``standard`` — z-score using the *training* mean/std.
  * ``robust``   — median / IQR; less sensitive to LOB outliers.
  * ``none``     — passthrough, only fills NaNs with the column median.

Anything more sophisticated (rolling-window normalisation, DAIN, etc.)
belongs in a model-specific preprocessing layer, not here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

ScalingMode = Literal["standard", "robust", "none"]


@dataclass
class FeatureScaler:
    """Fit-on-train / apply-anywhere scaler for the 32 raw LOB features."""

    mode: ScalingMode = "standard"
    eps: float = 1e-6
    center_: np.ndarray | None = field(default=None, init=False, repr=False)
    scale_: np.ndarray | None = field(default=None, init=False, repr=False)
    fill_: np.ndarray | None = field(default=None, init=False, repr=False)
    columns_: list[str] = field(default_factory=list, init=False, repr=False)

    def fit(self, df: pd.DataFrame, columns: list[str]) -> "FeatureScaler":
        self.columns_ = list(columns)
        values = df[columns].to_numpy(dtype=np.float64, copy=False)
        if self.mode == "standard":
            self.center_ = np.nanmean(values, axis=0)
            self.scale_ = np.nanstd(values, axis=0) + self.eps
        elif self.mode == "robust":
            self.center_ = np.nanmedian(values, axis=0)
            q75 = np.nanquantile(values, 0.75, axis=0)
            q25 = np.nanquantile(values, 0.25, axis=0)
            self.scale_ = (q75 - q25) + self.eps
        elif self.mode == "none":
            self.center_ = np.zeros(len(columns))
            self.scale_ = np.ones(len(columns))
        else:
            raise ValueError(f"unknown scaling mode: {self.mode!r}")
        # NaN fill value = column median (works for any mode, after scaling).
        self.fill_ = np.nanmedian(values, axis=0)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.center_ is None or self.scale_ is None:
            raise RuntimeError("FeatureScaler.transform() called before fit().")
        out = df.copy()
        arr = out[self.columns_].to_numpy(dtype=np.float32, copy=True)
        # Fill NaNs with training median first, then scale.
        nan_mask = np.isnan(arr)
        if nan_mask.any():
            for j in range(arr.shape[1]):
                col_nan = nan_mask[:, j]
                if col_nan.any():
                    fill_val = self.fill_[j] if self.fill_ is not None else 0.0
                    arr[col_nan, j] = (
                        fill_val if np.isfinite(fill_val) else 0.0
                    )
        arr = (arr - self.center_.astype(np.float32)) / self.scale_.astype(np.float32)
        out[self.columns_] = arr
        return out

    def fit_transform(self, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        return self.fit(df, columns).transform(df)

    def state_dict(self) -> dict:
        return {
            "mode": self.mode,
            "eps": self.eps,
            "center_": None if self.center_ is None else self.center_.tolist(),
            "scale_": None if self.scale_ is None else self.scale_.tolist(),
            "fill_": None if self.fill_ is None else self.fill_.tolist(),
            "columns_": list(self.columns_),
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> "FeatureScaler":
        s = cls(mode=state["mode"], eps=state["eps"])
        s.center_ = None if state["center_"] is None else np.asarray(state["center_"])
        s.scale_ = None if state["scale_"] is None else np.asarray(state["scale_"])
        s.fill_ = None if state["fill_"] is None else np.asarray(state["fill_"])
        s.columns_ = list(state["columns_"])
        return s
