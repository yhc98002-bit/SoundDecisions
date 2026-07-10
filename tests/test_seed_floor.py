"""Tests for foley_cw.seed_floor — Arc-3 Tier-B §B3 seed-floor direct test (non-gating)."""
from __future__ import annotations

import numpy as np

from foley_cw.seed_floor import (CFG1, RP_DIM, bootstrap_slope_ci, chance_accuracy,
                                 decide, gaussian_projection, mlp_probe_accuracy,
                                 ols_slope, probe_cfg, project)


# ---------------------------------------------------------------------------
# Fixed Gaussian random projection (JL)
# ---------------------------------------------------------------------------
class TestProjection:
    def test_deterministic_same_seed(self):
        R1 = gaussian_projection(5000, RP_DIM, seed=0)
        R2 = gaussian_projection(5000, RP_DIM, seed=0)
        assert np.array_equal(R1, R2)

    def test_shape_and_seed_changes_matrix(self):
        R = gaussian_projection(5000, 256, seed=0)
        assert R.shape == (5000, 256)
        assert not np.array_equal(R, gaussian_projection(5000, 256, seed=1))

    def test_norm_preserved_in_expectation(self):
        rng = np.random.default_rng(3)
        X = rng.normal(0, 1, (200, 1000))
        Xp = project(X, gaussian_projection(1000, 256, seed=0))
        assert Xp.shape == (200, 256)
        ratio = (Xp ** 2).sum(axis=1).mean() / (X ** 2).sum(axis=1).mean()
        assert 0.85 < ratio < 1.15            # JL ≈ norm-preserving

    def test_project_reuses_passed_matrix(self):
        R = gaussian_projection(50, 8, seed=0)
        X = np.random.default_rng(0).normal(0, 1, (5, 50))
        assert np.allclose(project(X, R), X @ R)


# ---------------------------------------------------------------------------
# Chance baseline + abstain handling
# ---------------------------------------------------------------------------
class TestChance:
    def test_majority_prior(self):
        assert chance_accuracy(["a", "a", "a", "b"]) == 0.75

    def test_empty_is_nan(self):
        assert np.isnan(chance_accuracy([]))


# ---------------------------------------------------------------------------
# MLP probe sanity
# ---------------------------------------------------------------------------
def _separable(n, k, d, seed, scale=1.0):
    rng = np.random.default_rng(seed)
    y = np.array([chr(ord("a") + i) for i in range(k)])[rng.integers(0, k, n)]
    X = rng.normal(0, 0.1, (n, d))
    for i in range(k):
        X[y == chr(ord("a") + i), i % d] += scale
    return X, list(y)


class TestMLP:
    def test_separable_high(self):
        X, y = _separable(160, 3, 8, 0, scale=3.0)
        acc = mlp_probe_accuracy(X[:110], y[:110], X[110:], y[110:], seed=0)
        assert acc > 0.85

    def test_random_near_chance(self):
        rng = np.random.default_rng(1)
        X = rng.normal(0, 1, (160, 8)); y = list(rng.integers(0, 2, 160).astype(str))
        acc = mlp_probe_accuracy(X[:110], y[:110], X[110:], y[110:], seed=0)
        assert 0.3 < acc < 0.7

    def test_single_class_train_predicts_majority(self):
        X = np.random.default_rng(3).normal(0, 1, (20, 4))
        acc = mlp_probe_accuracy(X[:10], ["a"] * 10, X[10:], ["a"] * 7 + ["b"] * 3, seed=0)
        assert acc == 0.7


# ---------------------------------------------------------------------------
# OLS slope
# ---------------------------------------------------------------------------
class TestSlope:
    def test_positive(self):
        assert ols_slope([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4]) > 0.09

    def test_flat_zero(self):
        assert abs(ols_slope([1, 2, 3], [0.5, 0.5, 0.5])) < 1e-9

    def test_negative(self):
        assert ols_slope([1, 2, 3], [0.6, 0.4, 0.2]) < 0

    def test_degenerate_nan(self):
        assert np.isnan(ols_slope([2, 2, 2], [0.1, 0.2, 0.3]))


# ---------------------------------------------------------------------------
# probe_cfg on the frozen split + data-leakage discipline
# ---------------------------------------------------------------------------
def _synthetic_pool(n_clips, n_indep, d_in, k, seed, signal):
    """Per clip, each independent's noise carries a class signal of strength `signal`."""
    rng = np.random.default_rng(seed)
    noise_by_clip, labels_by_clip = {}, {}
    classes = [chr(ord("a") + i) for i in range(k)]
    for c in range(n_clips):
        labs = [classes[i % k] for i in range(n_indep)]
        N = rng.normal(0, 1, (n_indep, d_in))
        for i, lab in enumerate(labs):
            N[i, classes.index(lab) % d_in] += signal      # class-from-noise signal
        noise_by_clip[str(c)] = N
        labels_by_clip[str(c)] = labs
    return noise_by_clip, labels_by_clip


class TestProbeCfg:
    def test_signal_above_chance(self):
        nb, lb = _synthetic_pool(20, 12, 60, 3, 0, signal=4.0)
        train = [str(c) for c in range(12)]; ev = [str(c) for c in range(12, 20)]
        r = probe_cfg(nb, lb, train, ev, lam=1.0, cfg=1.0)
        assert r.best_acc > r.chance + 0.10
        assert r.n_eval > 0 and len(r.eval_clip_ids) == r.n_eval

    def test_no_signal_near_chance(self):
        nb, lb = _synthetic_pool(20, 12, 60, 3, 1, signal=0.0)
        train = [str(c) for c in range(12)]; ev = [str(c) for c in range(12, 20)]
        r = probe_cfg(nb, lb, train, ev, lam=1.0, cfg=1.0)
        assert abs(r.best_acc - r.chance) < 0.20

    def test_abstain_dropped(self):
        nb, lb = _synthetic_pool(6, 8, 30, 2, 2, signal=3.0)
        lb["0"] = ["abstain"] * 8                # whole clip abstains
        r = probe_cfg(nb, lb, ["0", "1", "2"], ["3", "4", "5"], lam=1.0, cfg=1.0)
        # clip "0" contributes no train rows
        assert r.n_train == 16                   # clips 1,2 × 8 indep

    def test_eval_clip_disjoint_from_train(self):
        nb, lb = _synthetic_pool(10, 6, 20, 2, 4, signal=2.0)
        train = [str(c) for c in range(6)]; ev = [str(c) for c in range(6, 10)]
        r = probe_cfg(nb, lb, train, ev, lam=1.0, cfg=1.0)
        assert set(r.eval_clip_ids).issubset(set(ev))
        assert set(r.eval_clip_ids).isdisjoint(set(train))


# ---------------------------------------------------------------------------
# Bootstrap CI + decision rule
# ---------------------------------------------------------------------------
def _results_with_trend(signal_by_cfg, seed=0):
    cfgs = sorted(signal_by_cfg)
    results = {}
    for cfg in cfgs:
        nb, lb = _synthetic_pool(20, 12, 60, 3, seed + int(cfg * 10),
                                 signal=signal_by_cfg[cfg])
        train = [str(c) for c in range(12)]; ev = [str(c) for c in range(12, 20)]
        results[cfg] = probe_cfg(nb, lb, train, ev, lam=1.0, cfg=cfg)
    return results


class TestBootstrapAndDecision:
    def test_bootstrap_ci_structure(self):
        results = _results_with_trend({1.0: 4.0, 2.0: 4.0, 3.0: 4.0})
        ci = bootstrap_slope_ci(results, n_boot=300, seed=0)
        assert ci["lo"] <= ci["point"] <= ci["hi"]
        assert ci["eval_clips"] == 8

    def test_seed_floor_confirmed_at_cfg1(self):
        results = _results_with_trend({1.0: 4.0, 1.5: 4.0, 2.0: 4.0})
        ci = bootstrap_slope_ci(results, n_boot=200, seed=0)
        d = decide(results, ci)
        assert d["seed_floor_confirmed"] is True
        assert "SEED_FLOOR_CONFIRMED" in d["suggested_token"]

    def test_flat_slope_keeps_f1_refuted(self):
        # moderate, FLAT seed grip (accuracy well below ceiling, equal across cfg) → the
        # slope CI must include 0, so no amplification and F-1 stays refuted.
        results = _results_with_trend({1.0: 1.2, 2.0: 1.2, 3.0: 1.2, 4.5: 1.2})
        ci = bootstrap_slope_ci(results, n_boot=400, seed=0)
        d = decide(results, ci)
        assert d["slope_positive"] is False
        assert "F-1 stays refuted" in d["f1_status"]

    def test_no_seed_floor_token(self):
        results = _results_with_trend({1.0: 0.0, 2.0: 0.0, 3.0: 0.0})
        ci = bootstrap_slope_ci(results, n_boot=200, seed=0)
        d = decide(results, ci)
        assert d["seed_floor_confirmed"] is False
        assert d["suggested_token"] in ("NO_SEED_FLOOR", "SEED_AMPLIFICATION")
        # with zero signal it must not be amplification
        assert d["suggested_token"] == "NO_SEED_FLOOR"

    def test_rising_signal_can_flag_amplification(self):
        # monotone rising margin with cfg → CI should sit above 0
        results = _results_with_trend({1.0: 0.4, 2.0: 1.2, 3.0: 2.4, 4.5: 4.0})
        ci = bootstrap_slope_ci(results, n_boot=600, seed=0)
        d = decide(results, ci)
        assert d["slope_positive"] is True
        assert "SEED_AMPLIFICATION" in d["suggested_token"]
        assert CFG1 == 1.0
