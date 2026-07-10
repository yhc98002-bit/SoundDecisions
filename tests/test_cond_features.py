"""Tests for foley_cw.cond_features — Arc-3 Tier-B §B2 conditioning-channel audit.

CPU-only synthetic stubs exercise the probe + aggregate math (pooling, per-clip label
derivation, ridge/MLP eval accuracy vs chance, leakage-clean frozen split, bootstrap-by-
video CI, and the COND_BOTTLENECK decision rule). No GPU, no MMAudio, no real features.
"""
from __future__ import annotations

import math

import numpy as np

from foley_cw.cond_features import (bootstrap_acc_ci, build_cond_feature,
                                    clip_class_label, decide_cond_bottleneck,
                                    majority_class_accuracy, mlp_accuracy,
                                    pool_cond_tensor, run_cond_probe)


class TestPooling:
    def test_mean_max_pool_shape_and_values(self):
        a = np.array([[1.0, 2.0], [3.0, 4.0]])           # (T=2, D=2)
        out = pool_cond_tensor(a)
        assert out.shape == (4,)                          # [mean(2), max(2)]
        assert np.allclose(out, [2.0, 3.0, 3.0, 4.0])

    def test_squeezes_leading_batch_axis(self):
        a = np.zeros((1, 5, 7))
        assert pool_cond_tensor(a).shape == (14,)

    def test_1d_passthrough(self):
        v = np.arange(8.0)
        assert np.allclose(pool_cond_tensor(v), v)

    def test_build_concatenates_in_key_order(self):
        parts = {"clip_f": np.ones((3, 2)), "sync_f": np.zeros((4, 2)),
                 "clip_f_c": np.full(5, 9.0)}
        out = build_cond_feature(parts, ["clip_f", "sync_f", "clip_f_c"])
        # 2*2 + 2*2 + 5 = 13
        assert out.shape == (13,)
        assert out[0] == 1.0 and out[4] == 0.0 and out[-1] == 9.0


class TestLabelDerivation:
    def test_majority_drops_abstain_when_confident_exists(self):
        assert clip_class_label(["abstain", "dog", "dog", "abstain", "cat"]) == "dog"

    def test_all_abstain_is_none(self):
        assert clip_class_label(["abstain", "abstain"]) is None

    def test_deterministic_tie_break_lexicographic(self):
        # tie between 'a' and 'b' -> smallest label wins, deterministically
        assert clip_class_label(["b", "a"]) == "a"

    def test_majority_class_accuracy_uses_eval_prior(self):
        acc = majority_class_accuracy(["x"] * 3, ["x"] * 7 + ["y"] * 3)
        assert acc == 0.7


class TestBootstrap:
    def test_ci_brackets_point_estimate(self):
        rng = np.random.default_rng(0)
        clips = [f"c{i}" for i in range(40) for _ in range(2)]   # 2 rows / clip
        correct = (rng.random(80) < 0.75).astype(float)
        lo, hi = bootstrap_acc_ci(correct, clips, n_boot=400, seed=1)
        assert 0.0 <= lo <= correct.mean() <= hi <= 1.0

    def test_perfect_accuracy_ci_is_one(self):
        clips = [f"c{i}" for i in range(20)]
        lo, hi = bootstrap_acc_ci(np.ones(20), clips, n_boot=200, seed=2)
        assert lo == 1.0 and hi == 1.0


def _two_class_clips(n_per_class, d, seed, signal=1.5):
    """Synthetic per-clip features with a real class signal; clip ids unique."""
    rng = np.random.default_rng(seed)
    X, y, clips = [], [], []
    for ci in range(2 * n_per_class):
        cls = "A" if ci < n_per_class else "B"
        v = rng.normal(0, 0.3, d)
        v[0] += signal if cls == "A" else -signal
        X.append(v); y.append(cls); clips.append(f"clip{ci}")
    return np.array(X), y, clips


class TestRunCondProbe:
    def test_ridge_separable_above_chance_no_leakage(self):
        X, y, clips = _two_class_clips(40, 6, 7)
        train = {c for i, c in enumerate(clips) if i % 2 == 0}
        ev = {c for i, c in enumerate(clips) if i % 2 == 1}
        assert not (train & ev)                      # leakage-clean split
        r = run_cond_probe(X, y, clips, train, ev, family="ridge", n_boot=200)
        assert r["accuracy"] > 0.8
        assert r["accuracy"] > r["chance"]
        assert r["n_eval"] == len(ev)
        lo, hi = r["ci95"]
        assert lo <= r["accuracy"] <= hi

    def test_mlp_separable_above_chance(self):
        X, y, clips = _two_class_clips(40, 6, 11)
        train = {c for i, c in enumerate(clips) if i % 2 == 0}
        ev = {c for i, c in enumerate(clips) if i % 2 == 1}
        r = run_cond_probe(X, y, clips, train, ev, family="mlp", n_boot=100)
        assert r["accuracy"] > 0.8

    def test_random_features_near_chance(self):
        rng = np.random.default_rng(3)
        n = 80
        X = rng.normal(0, 1, (n, 6))
        y = ["A" if i < n // 2 else "B" for i in range(n)]
        clips = [f"k{i}" for i in range(n)]
        train = {c for i, c in enumerate(clips) if i % 2 == 0}
        ev = {c for i, c in enumerate(clips) if i % 2 == 1}
        r = run_cond_probe(X, y, clips, train, ev, family="ridge", n_boot=100)
        # no signal -> close to chance (eval prior here is 0.5); allow slack
        assert r["delta_over_chance"] < 0.25

    def test_none_labels_are_dropped_from_both_sides(self):
        X, y, clips = _two_class_clips(20, 5, 5)
        y = list(y); y[0] = None; y[-1] = None       # drop one train-eligible, one eval
        train = {c for i, c in enumerate(clips) if i % 2 == 0}
        ev = {c for i, c in enumerate(clips) if i % 2 == 1}
        r = run_cond_probe(X, y, clips, train, ev, family="ridge", n_boot=50)
        assert r["n_train"] + r["n_eval"] == sum(v is not None for v in y)


class TestMlpFallback:
    def test_numpy_mlp_fallback_runs_and_separates(self):
        # force the numpy fallback path directly (no sklearn) and check it separates
        from foley_cw import cond_features as cf
        X, y, _ = _two_class_clips(40, 6, 9)
        idx = np.arange(len(y))
        acc = cf._mlp_accuracy_numpy(X[idx[::2]], [y[i] for i in idx[::2]],
                                     X[idx[1::2]], [y[i] for i in idx[1::2]])
        assert acc > 0.7

    def test_mlp_degenerate_single_class_train(self):
        X = np.zeros((6, 3))
        acc = mlp_accuracy(X[:3], ["A"] * 3, X[3:], ["A", "A", "B"])
        assert acc == 2 / 3


class TestDecision:
    def test_cond_bottleneck_when_near_chance_and_below_b1(self):
        d = decide_cond_bottleneck(cond_best_acc=0.40, chance=0.30, b1_best_acc=0.70)
        assert d["token"] == "COND_BOTTLENECK"
        assert d["near_chance"] and d["below_b1"] and d["continue"] is True

    def test_not_bottleneck_when_cond_readable(self):
        d = decide_cond_bottleneck(cond_best_acc=0.65, chance=0.30, b1_best_acc=0.70)
        assert d["token"] == "COND_NOT_BOTTLENECK"
        assert not d["near_chance"]

    def test_not_bottleneck_when_not_below_b1(self):
        # near chance but B1 is ALSO low (not substantially above cond) -> not a
        # conditioning-specific bottleneck
        d = decide_cond_bottleneck(cond_best_acc=0.42, chance=0.30, b1_best_acc=0.45)
        assert d["below_b1"] is False
        assert d["token"] == "COND_NOT_BOTTLENECK"

    def test_b1_nan_blocks_below_b1(self):
        d = decide_cond_bottleneck(cond_best_acc=0.40, chance=0.30, b1_best_acc=float("nan"))
        assert d["below_b1"] is False
        assert not math.isnan(d["cond_best_acc"])
