"""Unit tests for the ensemble wrapper + weight tuner."""

import numpy as np
import pytest

from submission import EnsemblePredictionModel
from submission.ensemble import tune_ensemble_weights


class _FakeDP:
    def __init__(self, seq_ix, step, need):
        self.seq_ix = seq_ix
        self.step_in_seq = step
        self.need_prediction = need
        self.state = np.zeros(32, dtype=np.float32)


class _FakeMember:
    """Returns a fixed vector on scored rows, None on warm-up. Counts calls
    so we can assert the ensemble advances every member's state each step."""

    def __init__(self, value):
        self.value = np.asarray(value, dtype=np.float32)
        self.calls = 0

    def predict(self, dp):
        self.calls += 1
        return self.value.copy() if dp.need_prediction else None


def test_ensemble_warmup_returns_none_but_advances_members():
    m0, m1 = _FakeMember([1.0, 2.0]), _FakeMember([3.0, 4.0])
    ens = EnsemblePredictionModel([m0, m1], weights=[0.25, 0.75])
    out = ens.predict(_FakeDP(0, 0, need=False))
    assert out is None
    # Both members must have been stepped even though nothing was scored.
    assert m0.calls == 1 and m1.calls == 1


def test_ensemble_weighted_average_on_scored_rows():
    m0, m1 = _FakeMember([1.0, 1.0]), _FakeMember([3.0, 3.0])
    ens = EnsemblePredictionModel([m0, m1], weights=[0.25, 0.75])
    out = ens.predict(_FakeDP(0, 99, need=True))
    # 0.25*1 + 0.75*3 = 2.5 on both targets.
    np.testing.assert_allclose(out, [2.5, 2.5], rtol=1e-6)
    assert out.shape == (2,)


def test_ensemble_clips_to_scorer_range():
    m0, m1 = _FakeMember([100.0, -100.0]), _FakeMember([100.0, -100.0])
    ens = EnsemblePredictionModel([m0, m1])  # default equal weights
    out = ens.predict(_FakeDP(0, 99, need=True))
    np.testing.assert_allclose(out, [6.0, -6.0])


def test_ensemble_weights_normalised():
    ens = EnsemblePredictionModel([_FakeMember([2.0, 2.0]), _FakeMember([4.0, 4.0])],
                                  weights=[1.0, 3.0])  # -> 0.25 / 0.75
    np.testing.assert_allclose(ens.weights, [0.25, 0.75])


def test_tune_weights_beats_or_matches_best_single():
    # Build two predictors whose errors are anti-correlated, so a blend should
    # score at least as well as the better single one.
    rng = np.random.default_rng(0)
    n = 2000
    y = rng.standard_normal((n, 2)).astype(np.float64)
    p0 = y + rng.standard_normal((n, 2)) * 1.0    # noisy estimate A
    p1 = y + rng.standard_normal((n, 2)) * 1.0    # independent noisy estimate B
    res = tune_ensemble_weights([p0, p1], y, step=0.05)
    assert len(res["weights"]) == 2
    assert abs(sum(res["weights"]) - 1.0) < 1e-9
    assert res["score"] >= max(res["per_member"]) - 1e-9   # blend never worse
    # With two independent unbiased estimators the blend should strictly help.
    assert res["score"] > max(res["per_member"])


def test_tune_weights_single_member():
    y = np.random.default_rng(1).standard_normal((500, 2))
    p = y + 0.5
    res = tune_ensemble_weights([p], y)
    assert res["weights"] == [1.0]
