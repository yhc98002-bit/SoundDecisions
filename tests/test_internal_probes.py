"""Tests for foley_cw.internal_probes — Track P linear probes (non-gating)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from foley_cw.internal_probes import (best_layer_curve, probe_accuracy,
                                      s_read_internal)


def _separable(n, k, d, seed):
    rng = np.random.default_rng(seed)
    y = np.array([chr(ord("a") + i) for i in range(k)])[rng.integers(0, k, n)]
    X = rng.normal(0, 0.1, (n, d))
    for i in range(k):
        X[y == chr(ord("a") + i), i % d] += 1.0
    return X, list(y)


class TestProbeAccuracy:
    def test_separable_binary_near_one(self):
        X, y = _separable(120, 2, 5, 0)
        acc = probe_accuracy(X[:80], y[:80], X[80:], y[80:])
        assert acc > 0.9

    def test_random_labels_near_chance(self):
        rng = np.random.default_rng(1)
        X = rng.normal(0, 1, (120, 5)); y = list(rng.integers(0, 2, 120).astype(str))
        acc = probe_accuracy(X[:80], y[:80], X[80:], y[80:])
        assert 0.3 < acc < 0.7

    def test_three_class_separable(self):
        X, y = _separable(150, 3, 6, 2)
        acc = probe_accuracy(X[:100], y[:100], X[100:], y[100:])
        assert acc > 0.9

    def test_single_class_train_predicts_majority(self):
        X = np.random.default_rng(3).normal(0, 1, (20, 4))
        acc = probe_accuracy(X[:10], ["a"] * 10, X[10:], ["a"] * 7 + ["b"] * 3)
        assert acc == pytest.approx(0.7)  # majority 'a' on eval = 7/10

    def test_empty_eval_is_nan(self):
        X = np.zeros((4, 3))
        assert math.isnan(probe_accuracy(X, ["a", "b", "a", "b"], np.zeros((0, 3)), []))


class TestSReadInternal:
    def test_first_crossing(self):
        assert s_read_internal({0.05: 0.4, 0.45: 0.6, 0.9: 0.8}, 0.7) == pytest.approx(0.9)
        assert s_read_internal({0.05: 0.95, 0.9: 0.99}, 0.7) == pytest.approx(0.05)

    def test_never_is_nan(self):
        assert math.isnan(s_read_internal({0.05: 0.3, 0.9: 0.5}, 0.7))


class TestBestLayerCurve:
    def test_picks_informative_layer(self):
        # layer 0 separable (2-d), layer 1 pure noise; best layer should be 0.
        X0, y = _separable(120, 2, 2, 4)
        noise = np.random.default_rng(5).normal(0, 1, (120, 2))
        feats = {
            (0, 0.5): {"train": (X0[:80], y[:80]), "eval": (X0[80:], y[80:])},
            (1, 0.5): {"train": (noise[:80], y[:80]), "eval": (noise[80:], y[80:])},
        }
        cur = best_layer_curve(feats, [0, 1], [0.5])
        assert cur["best_layer"][0.5] == 0
        assert cur["best_layer_acc"][0.5] > 0.9
