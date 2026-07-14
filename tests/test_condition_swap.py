"""Tests for foley_cw.condition_swap — Stage R condition-swap math (manual §8.1, Fig. 5).

CPU-only: the follow/retention/s_cond/sanity math runs on a synthetic measurer stub
(plain labels / embedding vectors, no GPU, no MMAudio). One end-to-end swap-completion
test uses the pure-numpy SyntheticGaussianFlow oracle to confirm the causal signature
the manual pre-registers (swap at s~=0 -> follow donor; swap near s~=1 -> retain source).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from foley_cw import condition_swap as CS
from foley_cw import score_sde as K
from foley_cw.synthetic_backend import SyntheticGaussianFlow, SyntheticVideoCond
from foley_cw.types import AxisKind, ScheduleSpec


# ======================================================================
# matches() — the per-axis "matches donor/source" relation
# ======================================================================
class TestMatches:
    def test_categorical_exact(self):
        assert CS.matches("dog", "dog", AxisKind.CATEGORICAL)
        assert not CS.matches("dog", "cat", AxisKind.CATEGORICAL)

    def test_embedding_nearer_anchor_wins(self):
        donor = np.array([1.0, 0.0])
        source = np.array([0.0, 1.0])
        swapped = np.array([0.9, 0.1])  # clearly closer to donor
        assert CS.matches(swapped, donor, AxisKind.EMBEDDING, other=source)
        assert not CS.matches(swapped, source, AxisKind.EMBEDDING, other=donor)

    def test_embedding_cos_floor_blocks_match(self):
        # Orthogonal to the anchor -> cos 0; with a positive floor it cannot match.
        swapped = np.array([0.0, 1.0])
        anchor = np.array([1.0, 0.0])
        assert not CS.matches(swapped, anchor, AxisKind.EMBEDDING, embed_cos_min=0.1)

    def test_embedding_cosine_tie_matches_neither(self):
        # Equidistant (equal cosine) to both anchors -> matches NEITHER (strict >),
        # so a tie is never counted as both follow and retention.
        donor = np.array([1.0, 0.0])
        source = np.array([0.0, 1.0])
        swapped = np.array([1.0, 1.0])  # cos to donor == cos to source
        assert not CS.matches(swapped, donor, AxisKind.EMBEDDING, other=source,
                              embed_cos_min=-1.0)
        assert not CS.matches(swapped, source, AxisKind.EMBEDDING, other=donor,
                              embed_cos_min=-1.0)
        # and the pair-level rates put it in 'neither'
        r = CS.follow_retention_rates([swapped], [donor], [source],
                                      AxisKind.EMBEDDING, embed_cos_min=-1.0)
        assert (r["follow"], r["retention"], r["neither"]) == pytest.approx(
            (0.0, 0.0, 1.0)
        )

    def test_unknown_kind_raises(self):
        with pytest.raises(ValueError):
            CS.matches("x", "x", "not_a_kind")  # type: ignore[arg-type]


# ======================================================================
# follow_retention_rates() — categorical
# ======================================================================
class TestFollowRetentionCategorical:
    def test_all_follow(self):
        # every swap lands on the donor's label
        r = CS.follow_retention_rates(["dog"] * 5, ["dog"] * 5, ["cat"] * 5,
                                      AxisKind.CATEGORICAL)
        assert (r["follow"], r["retention"]) == pytest.approx((1.0, 0.0))
        assert r["n"] == 5

    def test_all_retain(self):
        r = CS.follow_retention_rates(["cat"] * 4, ["dog"] * 4, ["cat"] * 4,
                                      AxisKind.CATEGORICAL)
        assert (r["retention"], r["follow"]) == pytest.approx((1.0, 0.0))

    def test_mixed_and_neither(self):
        # 2 follow (dog), 1 retain (cat), 1 third label (bird) -> neither
        sw = ["dog", "dog", "cat", "bird"]
        dv = ["dog"] * 4
        sv = ["cat"] * 4
        r = CS.follow_retention_rates(sw, dv, sv, AxisKind.CATEGORICAL)
        assert r["follow"] == pytest.approx(0.5)
        assert r["retention"] == pytest.approx(0.25)
        assert r["neither"] == pytest.approx(0.25)
        assert math.isclose(r["follow"] + r["retention"] + r["neither"], 1.0)

    def test_shared_donor_source_label_counts_both(self):
        # When donor and source share the label, a matching swap is both follow
        # and retention (an uninformative pair) — neither is then 0.
        r = CS.follow_retention_rates(["dog", "dog"], ["dog", "dog"], ["dog", "dog"],
                                      AxisKind.CATEGORICAL)
        assert (r["follow"], r["retention"], r["neither"]) == pytest.approx(
            (1.0, 1.0, 0.0)
        )

    def test_empty_is_nan(self):
        r = CS.follow_retention_rates([], [], [], AxisKind.CATEGORICAL)
        assert math.isnan(r["follow"]) and r["n"] == 0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            CS.follow_retention_rates(["a"], ["a", "b"], ["a"], AxisKind.CATEGORICAL)


# ======================================================================
# follow_retention_rates() — embedding
# ======================================================================
class TestFollowRetentionEmbedding:
    def test_embedding_follow_vs_retention(self):
        donor = [np.array([1.0, 0.0]) for _ in range(3)]
        source = [np.array([0.0, 1.0]) for _ in range(3)]
        # two swaps near donor, one near source
        swapped = [np.array([0.95, 0.05]), np.array([0.9, 0.1]), np.array([0.1, 0.9])]
        r = CS.follow_retention_rates(swapped, donor, source, AxisKind.EMBEDDING)
        assert r["follow"] == pytest.approx(2 / 3)
        assert r["retention"] == pytest.approx(1 / 3)


# ======================================================================
# s_cond() — earliest s with follow < 0.5
# ======================================================================
class TestSCond:
    def test_basic_crossing(self):
        rates = {0.1: {"follow": 1.0}, 0.3: {"follow": 0.7},
                 0.5: {"follow": 0.4}, 0.7: {"follow": 0.1}}
        assert CS.s_cond(rates) == pytest.approx(0.5)

    def test_never_crosses_is_nan(self):
        rates = {0.1: {"follow": 0.9}, 0.5: {"follow": 0.8}, 0.9: {"follow": 0.6}}
        assert math.isnan(CS.s_cond(rates))

    def test_skips_nan_follow(self):
        rates = {0.1: {"follow": float("nan")}, 0.3: {"follow": 0.9},
                 0.5: {"follow": 0.2}}
        assert CS.s_cond(rates) == pytest.approx(0.5)

    def test_custom_majority_threshold(self):
        rates = {0.2: {"follow": 0.85}, 0.4: {"follow": 0.75}, 0.6: {"follow": 0.65}}
        assert CS.s_cond(rates, follow_majority=0.8) == pytest.approx(0.4)


# ======================================================================
# sanity_check() — swap at s~=0 => follow; s~=1 => retain
# ======================================================================
class TestSanity:
    def test_pass(self):
        rates = {0.05: {"follow": 0.95, "retention": 0.05},
                 0.90: {"follow": 0.10, "retention": 0.90}}
        out = CS.sanity_check(rates)
        assert out["passed"] is True
        assert out["follow_ok"] is True and out["retention_ok"] is True
        assert (out["low_s"], out["high_s"]) == pytest.approx((0.05, 0.90))

    def test_fail_low_does_not_follow(self):
        rates = {0.05: {"follow": 0.2, "retention": 0.8},
                 0.90: {"follow": 0.1, "retention": 0.9}}
        out = CS.sanity_check(rates)
        assert out["passed"] is False and out["follow_ok"] is False

    def test_nan_rate_yields_none(self):
        rates = {0.05: {"follow": float("nan"), "retention": 0.0},
                 0.90: {"follow": 0.1, "retention": 0.9}}
        out = CS.sanity_check(rates)
        assert out["follow_ok"] is None and out["passed"] is None

    def test_empty(self):
        out = CS.sanity_check({})
        assert out["passed"] is None and out["low_s"] is None


# ======================================================================
# summarize_axis() — end-to-end from raw measured values keyed by s
# ======================================================================
class TestSummarizeAxis:
    def _synthetic_swap_values(self, s_points, follow_curve):
        """Stub measurer output: at each s, a fraction follow_curve[s] of swaps carry
        the donor label and the rest carry the source label (n=10)."""
        donor_lbl, source_lbl = "DONOR", "SOURCE"
        swapped, donor_by, source_by = {}, {}, {}
        n = 10
        for s in s_points:
            k = int(round(follow_curve[s] * n))
            swapped[s] = [donor_lbl] * k + [source_lbl] * (n - k)
            donor_by[s] = [donor_lbl] * n
            source_by[s] = [source_lbl] * n
        return swapped, donor_by, source_by

    def test_monotone_decay_gives_expected_s_cond_and_sanity(self):
        s_points = [0.05, 0.30, 0.50, 0.90]
        follow_curve = {0.05: 1.0, 0.30: 0.8, 0.50: 0.4, 0.90: 0.0}
        sw, dn, sr = self._synthetic_swap_values(s_points, follow_curve)
        res = CS.summarize_axis(sw, dn, sr, AxisKind.CATEGORICAL)
        # s_cond is the first s with follow < 0.5
        assert res["s_cond"] == pytest.approx(0.50)
        # sanity: full follow at 0.05, full retention at 0.90
        assert res["sanity"]["passed"] is True
        # curves are sorted ascending in s and aligned
        assert res["curves"]["s"] == pytest.approx([0.05, 0.30, 0.50, 0.90])
        assert res["curves"]["follow"][0] == pytest.approx(1.0)
        assert res["curves"]["retention"][-1] == pytest.approx(1.0)


# ======================================================================
# Swap-completion helper on the pure-numpy oracle (no GPU / no MMAudio)
# ======================================================================
class TestSwapCompletionSynthetic:
    """The CAUSAL signature: replacing the cond mid-ODE on the SyntheticGaussianFlow
    backend (whose 'audio' is the final latent) follows the donor when swapped early
    and retains the source when swapped late — the manual §8.1 sanity controls, but
    on a backend where the math is exact and CPU-only."""

    def _setup(self):
        be = SyntheticGaussianFlow(dim=3, sigma2=0.01)
        sched = ScheduleSpec(n_steps=64)
        src = SyntheticVideoCond(mu=np.array([5.0, 0.0, 0.0]), video_id="src")
        don = SyntheticVideoCond(mu=np.array([0.0, 0.0, -5.0]), video_id="don")
        return be, sched, src, don

    def test_swap_early_follows_late_retains(self):
        be, sched, src, don = self._setup()
        s_points = (0.05, 0.95)
        tr = K.generate_trajectory(be, src, sched, np.random.default_rng(0),
                                   alpha=0.0, record_points=s_points)
        src_final = tr["audio"]
        don_final = K.generate_trajectory(be, don, sched, np.random.default_rng(1),
                                          alpha=0.0)["audio"]

        sw_early = CS.cond_swap_complete(be, tr["states"][0.05], 0.05, don, sched)
        sw_late = CS.cond_swap_complete(be, tr["states"][0.95], 0.95, don, sched)

        # early swap is much closer to the donor's own final; late swap to source's
        assert np.linalg.norm(sw_early - don_final) < np.linalg.norm(sw_early - src_final)
        assert np.linalg.norm(sw_late - src_final) < np.linalg.norm(sw_late - don_final)

    def test_completion_feeds_follow_retention_math(self):
        # Drive the embedding follow/retention math with REAL swap completions.
        be, sched, src, don = self._setup()
        tr = K.generate_trajectory(be, src, sched, np.random.default_rng(2),
                                   alpha=0.0, record_points=(0.05, 0.95))
        src_final = tr["audio"]
        don_final = K.generate_trajectory(be, don, sched, np.random.default_rng(3),
                                          alpha=0.0)["audio"]
        early = CS.cond_swap_complete(be, tr["states"][0.05], 0.05, don, sched)
        late = CS.cond_swap_complete(be, tr["states"][0.95], 0.95, don, sched)
        # Use raw L2-nearest via the embedding match (cos floor off): treat the 3-vecs
        # as 'embeddings'. Early -> follow, late -> retain.
        r_early = CS.follow_retention_rates([early], [don_final], [src_final],
                                            AxisKind.EMBEDDING, embed_cos_min=-1.0)
        r_late = CS.follow_retention_rates([late], [don_final], [src_final],
                                           AxisKind.EMBEDDING, embed_cos_min=-1.0)
        assert r_early["follow"] == pytest.approx(1.0)
        assert r_late["retention"] == pytest.approx(1.0)


# ======================================================================
# mix_cond() fallback plumbing (interp mode)
# ======================================================================
class TestMixCond:
    def test_raises_without_backend_support(self):
        class _NoMix:
            pass
        with pytest.raises(NotImplementedError):
            CS.mix_cond(_NoMix(), object(), object(), 0.5)

    def test_delegates_to_backend(self):
        class _Mix:
            def mix_cond(self, a, b, w):
                return ("mixed", a, b, w)
        out = CS.mix_cond(_Mix(), "S", "D", 0.3)
        assert out == ("mixed", "S", "D", 0.3)

    def test_interp_complete_matches_swap_with_full_donor(self):
        # A backend whose mix_cond at w=1 returns the donor cond should make
        # cond_interp_complete identical to cond_swap_complete with the donor.
        be = SyntheticGaussianFlow(dim=3, sigma2=0.01)
        sched = ScheduleSpec(n_steps=64)
        src = SyntheticVideoCond(mu=np.array([5.0, 0.0, 0.0]))
        don = SyntheticVideoCond(mu=np.array([0.0, 0.0, -5.0]))
        tr = K.generate_trajectory(be, src, sched, np.random.default_rng(4),
                                   alpha=0.0, record_points=(0.3,))
        x_s = tr["states"][0.3]
        swapped = CS.cond_swap_complete(be, x_s, 0.3, don, sched)
        interp = CS.cond_interp_complete(be, x_s, 0.3, don, sched)  # mixed==donor
        assert np.allclose(swapped, interp)


def test_neither_rate_correct_for_mixed_batch():
    """Regression (workflow review): neither must be counted per-pair, not by
    1-follow-retention — shared-value pairs otherwise clip a true neither to 0."""
    from foley_cw.condition_swap import follow_retention_rates
    from foley_cw.types import AxisKind
    # pair0 ("a","a","a"): shared -> both follow & retention; pair1 ("b","x","y"): neither
    r = follow_retention_rates(["a", "b"], ["a", "x"], ["a", "y"], AxisKind.CATEGORICAL)
    assert (r["follow"], r["retention"]) == pytest.approx((0.5, 0.5))
    assert r["neither"] == pytest.approx(0.5)
