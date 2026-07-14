"""Tests for foley_cw.class_probes — Arc-3 Tier-B B1 probes (NON-GATING)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from foley_cw.class_probes import (class_readability_curve, eval_majority_prior,
                                    mlp_probe_accuracy, s_read_internal_class)


def _separable(n, k, d, seed, sep=1.0):
    rng = np.random.default_rng(seed)
    y = np.array([chr(ord("a") + i) for i in range(k)])[rng.integers(0, k, n)]
    X = rng.normal(0, 0.1, (n, d))
    for i in range(k):
        X[y == chr(ord("a") + i), i % d] += sep
    return X, list(y)


class TestEvalMajorityPrior:
    def test_balanced_binary(self):
        assert eval_majority_prior(["a", "a", "b", "b"]) == pytest.approx(0.5)

    def test_skewed(self):
        assert abs(eval_majority_prior(["a"] * 7 + ["b"] * 3) - 0.7) < 1e-9

    def test_empty_is_nan(self):
        assert math.isnan(eval_majority_prior([]))


class TestMLPProbeAccuracy:
    def test_separable_binary_high(self):
        X, y = _separable(160, 2, 6, 0, sep=2.0)
        acc = mlp_probe_accuracy(X[:110], y[:110], X[110:], y[110:], seed=0)
        assert acc > 0.85

    def test_random_labels_near_chance(self):
        rng = np.random.default_rng(1)
        X = rng.normal(0, 1, (160, 6))
        y = list(rng.integers(0, 2, 160).astype(str))
        acc = mlp_probe_accuracy(X[:110], y[:110], X[110:], y[110:], seed=0)
        assert 0.3 < acc < 0.7

    def test_empty_eval_is_nan(self):
        X = np.zeros((8, 4))
        assert math.isnan(
            mlp_probe_accuracy(X, ["a", "b"] * 4, np.zeros((0, 4)), [], seed=0))

    def test_single_train_class_predicts_majority(self):
        X = np.random.default_rng(3).normal(0, 1, (20, 4))
        acc = mlp_probe_accuracy(X[:10], ["a"] * 10, X[10:], ["a"] * 7 + ["b"] * 3, seed=0)
        assert acc == pytest.approx(0.7)  # constant-'a' prediction -> 7/10 on eval

    def test_three_class_separable(self):
        X, y = _separable(210, 3, 9, 2, sep=2.0)
        acc = mlp_probe_accuracy(X[:150], y[:150], X[150:], y[150:], seed=0)
        assert acc > 0.8


class TestSReadInternalClass:
    def test_requires_both_theta_and_margin(self):
        # acc 0.72 >= theta but chance 0.65 -> margin 0.15 fails at s=0.15; 0.85 passes at 0.45
        best = {0.05: 0.40, 0.15: 0.72, 0.45: 0.85}
        chance = {0.05: 0.30, 0.15: 0.65, 0.45: 0.30}
        assert s_read_internal_class(best, chance, 0.70, 0.15) == pytest.approx(0.45)

    def test_first_crossing_when_both_hold(self):
        best = {0.05: 0.95, 0.45: 0.99}
        chance = {0.05: 0.40, 0.45: 0.40}
        assert s_read_internal_class(best, chance, 0.70, 0.15) == pytest.approx(0.05)

    def test_never_is_nan(self):
        best = {0.05: 0.40, 0.9: 0.55}
        chance = {0.05: 0.30, 0.9: 0.30}
        assert math.isnan(s_read_internal_class(best, chance, 0.70, 0.15))

    def test_high_chance_blocks_despite_high_acc(self):
        # acc clears theta everywhere but never beats chance by the margin -> never
        best = {0.05: 0.80, 0.45: 0.82}
        chance = {0.05: 0.78, 0.45: 0.79}
        assert math.isnan(s_read_internal_class(best, chance, 0.70, 0.15))


class TestClassReadabilityCurve:
    def _feats(self, n, d, seed):
        # informative layer 0 (separable), noise layer 1; same gids/labels per s.
        X0, y = _separable(n, 3, d, seed, sep=2.0)
        noise = np.random.default_rng(seed + 7).normal(0, 1, (n, d))
        tr = slice(0, int(0.7 * n)); te = slice(int(0.7 * n), n)
        feats = {}
        for s in (0.25, 0.75):
            feats[(0, s)] = {"train": (X0[tr], y[tr]), "eval": (X0[te], y[te])}
            feats[(1, s)] = {"train": (noise[tr], y[tr]), "eval": (noise[te], y[te])}
        return feats, y[te]

    def test_picks_informative_layer_and_records_spec(self):
        feats, yte = self._feats(210, 9, 4)
        cur = class_readability_curve(feats, [0, 1], [0.25, 0.75], mlp_seed=0)
        assert cur["best_spec"][0.25]["layer"] == 0
        assert cur["best_acc"][0.25] > 0.8
        assert cur["best_spec"][0.25]["probe"] in ("ridge", "mlp")
        # chance = eval majority prior, same for both s
        assert abs(cur["chance"][0.25] - eval_majority_prior(list(yte))) < 1e-9

    def test_chance_computed_per_s(self):
        feats, _ = self._feats(150, 6, 5)
        cur = class_readability_curve(feats, [0, 1], [0.25, 0.75], mlp_seed=0)
        assert math.isclose(cur["chance"][0.25], cur["chance"][0.75])
