"""Tests for foley_cw.cfg_dial — F-1 cfg-dial evidence (manual §8.3). CPU-only, numpy.

Exercises the three pure functions and the three suggested-token outcomes
(F1_SUPPORTED / F1_REFUTED / F1_INCONCLUSIVE) with synthetic per-cfg arrays.
"""
from __future__ import annotations

import math

import numpy as np

from foley_cw.cfg_dial import (F1_INCONCLUSIVE, F1_REFUTED, F1_SUPPORTED,
                               alpha_star, f1_verdict, seed_predictability)

CFGS = (1.0, 1.5, 2.0, 2.5, 3.0, 4.5)


# ---------------------------------------------------------------------------
# (1) seed_predictability
# ---------------------------------------------------------------------------
class TestSeedPredictability:
    def test_rising_accuracy_grows(self):
        acc = {1.0: 0.30, 1.5: 0.40, 2.0: 0.52, 2.5: 0.61, 3.0: 0.72, 4.5: 0.88}
        out = seed_predictability(acc, chance=0.33)
        assert out["grows"] is True
        assert out["trend"]["slope"] > 0
        assert out["above_chance_at_top"] is True

    def test_flat_does_not_grow(self):
        acc = {c: 0.5 for c in CFGS}
        out = seed_predictability(acc)
        assert out["grows"] is False
        assert out["above_chance_at_top"] is None      # no chance given

    def test_shrinking_does_not_grow(self):
        acc = {1.0: 0.9, 1.5: 0.8, 2.0: 0.7, 2.5: 0.6, 3.0: 0.5, 4.5: 0.4}
        out = seed_predictability(acc, chance=0.33)
        assert out["grows"] is False
        assert out["trend"]["slope"] < 0

    def test_below_chance_at_top(self):
        acc = {1.0: 0.30, 4.5: 0.31}
        out = seed_predictability(acc, chance=0.5)
        assert out["above_chance_at_top"] is False

    def test_too_few_points_inconclusive(self):
        out = seed_predictability({2.0: 0.7})
        assert out["grows"] is False
        assert math.isnan(out["trend"]["slope"])


# ---------------------------------------------------------------------------
# (2) alpha_star
# ---------------------------------------------------------------------------
class TestAlphaStar:
    def test_minimum_unlocking_alpha(self):
        # diversity crosses 0.02 first at alpha=0.2 here.
        div = {0.05: 0.0, 0.1: 0.005, 0.2: 0.03, 0.4: 0.10, 0.8: 0.25}
        out = alpha_star({2.0: div}, diversity_min=0.02)
        assert out["by_cfg"][2.0] == 0.2

    def test_never_unlocks_is_nan(self):
        div = {0.05: 0.0, 0.1: 0.001, 0.2: 0.005}     # never reaches 0.02
        out = alpha_star({3.0: div}, diversity_min=0.02)
        assert math.isnan(out["by_cfg"][3.0])

    def test_alpha_star_rises_with_cfg(self):
        # higher cfg needs a larger alpha to unlock (more guidance = more mode-locked).
        by = {
            1.0: {0.05: 0.05, 0.1: 0.2, 0.2: 0.3},          # α* = 0.05
            2.0: {0.05: 0.0, 0.1: 0.04, 0.2: 0.2},          # α* = 0.1
            3.0: {0.05: 0.0, 0.1: 0.01, 0.2: 0.03},         # α* = 0.2
            4.5: {0.05: 0.0, 0.1: 0.0, 0.2: 0.01, 0.4: 0.05},  # α* = 0.4
        }
        out = alpha_star(by, diversity_min=0.02)
        assert out["increasing"] is True
        assert out["by_cfg"][1.0] == 0.05 and out["by_cfg"][4.5] == 0.4


# ---------------------------------------------------------------------------
# (3) f1_verdict — the three tokens
# ---------------------------------------------------------------------------
class TestF1Verdict:
    def test_supported_all_predictions_met(self):
        seed_pred = {1.0: 0.30, 1.5: 0.42, 2.0: 0.55, 2.5: 0.64, 3.0: 0.75, 4.5: 0.90}
        astar = {1.0: 0.05, 1.5: 0.1, 2.0: 0.2, 2.5: 0.2, 3.0: 0.4, 4.5: 0.8}
        share = {1.0: 0.10, 1.5: 0.16, 2.0: 0.22, 2.5: 0.29, 3.0: 0.35, 4.5: 0.48}
        v = f1_verdict(seed_pred, astar, share, diversity_min=0.02, chance=0.33)
        assert v["suggested_token"] == F1_SUPPORTED
        assert v["predictions_met"] == {"seed_predictability_grows": True,
                                        "alpha_star_increases": True,
                                        "seed_share_grows": True}

    def test_refuted_seed_share_shrinks(self):
        # the manual's cfg=4.5 tension: seed share SHRINKS with cfg (reversed) → REFUTED.
        seed_pred = {1.0: 0.30, 1.5: 0.42, 2.0: 0.55, 2.5: 0.64, 3.0: 0.75, 4.5: 0.90}
        astar = {1.0: 0.05, 1.5: 0.1, 2.0: 0.2, 2.5: 0.2, 3.0: 0.4, 4.5: 0.8}
        share = {1.0: 0.40, 1.5: 0.34, 2.0: 0.28, 2.5: 0.20, 3.0: 0.12, 4.5: 0.05}
        v = f1_verdict(seed_pred, astar, share, diversity_min=0.02)
        assert v["suggested_token"] == F1_REFUTED
        assert v["predictions_met"]["seed_share_grows"] is False
        assert "seed_share" in v["rationale"]

    def test_refuted_seed_pred_shrinks(self):
        seed_pred = {1.0: 0.90, 1.5: 0.80, 2.0: 0.70, 2.5: 0.60, 3.0: 0.50, 4.5: 0.40}
        astar = {1.0: 0.05, 1.5: 0.1, 2.0: 0.2, 2.5: 0.2, 3.0: 0.4, 4.5: 0.8}
        share = {1.0: 0.10, 1.5: 0.16, 2.0: 0.22, 2.5: 0.29, 3.0: 0.35, 4.5: 0.48}
        v = f1_verdict(seed_pred, astar, share)
        assert v["suggested_token"] == F1_REFUTED
        assert v["predictions_met"]["seed_predictability_grows"] is False

    def test_inconclusive_flat_mixed(self):
        # flat seed-pred + flat share (no growth, no reversal) → INCONCLUSIVE.
        seed_pred = {c: 0.50 for c in CFGS}
        astar = {1.0: 0.2, 1.5: 0.2, 2.0: 0.2, 2.5: 0.2, 3.0: 0.2, 4.5: 0.2}
        share = {c: 0.30 for c in CFGS}
        v = f1_verdict(seed_pred, astar, share)
        assert v["suggested_token"] == F1_INCONCLUSIVE
        assert not any(v["predictions_met"].values())

    def test_inconclusive_partial_no_reversal(self):
        # seed-pred + α* rise, but seed share flat (met 2/3, no reversal) → INCONCLUSIVE.
        seed_pred = {1.0: 0.30, 1.5: 0.42, 2.0: 0.55, 2.5: 0.64, 3.0: 0.75, 4.5: 0.90}
        astar = {1.0: 0.05, 1.5: 0.1, 2.0: 0.2, 2.5: 0.2, 3.0: 0.4, 4.5: 0.8}
        share = {c: 0.30 for c in CFGS}
        v = f1_verdict(seed_pred, astar, share)
        assert v["suggested_token"] == F1_INCONCLUSIVE
        assert sum(v["predictions_met"].values()) == 2

    def test_missing_seed_share_inconclusive(self):
        # seed share unavailable (e.g. §8.3 part-a budgets not yet dumped) → cannot meet P3.
        seed_pred = {1.0: 0.30, 1.5: 0.42, 2.0: 0.55, 2.5: 0.64, 3.0: 0.75, 4.5: 0.90}
        astar = {1.0: 0.05, 1.5: 0.1, 2.0: 0.2, 2.5: 0.2, 3.0: 0.4, 4.5: 0.8}
        v = f1_verdict(seed_pred, astar, {})
        assert v["suggested_token"] == F1_INCONCLUSIVE
        assert v["predictions_met"]["seed_share_grows"] is False
        assert v["provenance"]["n_cfg"]["seed_share"] == 0


# ---------------------------------------------------------------------------
# integration: the (noise, video) -> class probe path used by the script aggregate
# ---------------------------------------------------------------------------
class TestProbePathIntegration:
    def test_seedlocked_noise_predicts_class(self):
        # synthetic seed-predictable regime: class is a deterministic function of the noise
        # latent's leading coordinate → the reused probe_accuracy should read it high, the
        # exact (noise,video)->class signal seed_predictability summarizes across cfg.
        from foley_cw.internal_probes import probe_accuracy
        rng = np.random.default_rng(0)
        X = rng.normal(0, 1, (120, 8))
        y = ["c0" if x0 > 0 else "c1" for x0 in X[:, 0]]
        acc = probe_accuracy(X[:80], y[:80], X[80:], y[80:])
        assert acc > 0.9
        out = seed_predictability({4.5: acc}, chance=0.5)
        assert out["above_chance_at_top"] is True
