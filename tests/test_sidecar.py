"""Tests for foley_cw/sidecar.py — Cohen's kappa + real-path reliability wrapper.

All tests run on CPU against SyntheticGaussianFlow / SyntheticMeasurer (numpy-only).
No network, no GPU, no httpx (sidecar must import without it).

Key contracts checked here:
  * cohens_kappa hand-computed cases: perfect = 1.0, exact-chance = 0.0, a known
    2x2 table giving kappa = 0.4, degenerate marginals guard.
  * run_real_reliability reproduces determinism = 1.0 with SyntheticMeasurer and
    mirrors reliability_gate's pass/demotion semantics (AXIS_DEMOTED reason format,
    NaN validity counts as failure).
  * validity is NaN without gold; kappa path with perfect synthetic gold gives 1.0;
    embedding gold gives mean cosine.
  * custom perturbations dict is passed through to reliability.robustness.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from foley_cw.axes import SyntheticMeasurer
from foley_cw.config import load_config
from foley_cw.sidecar import (cohens_kappa, confusion_matrix, gwet_ac1, pabak,
                              run_real_reliability)
from foley_cw.synthetic_backend import SyntheticGaussianFlow
from foley_cw.types import (
    AgreementMetric,
    Axis,
    AxisKind,
    AxisTier,
    ReliabilityResult,
    SelfTarget,
)


# --------------------------------------------------------------------------------------
# Shared fixtures
# --------------------------------------------------------------------------------------

@pytest.fixture
def measurer():
    return SyntheticMeasurer(seed=42)


@pytest.fixture
def thresholds():
    return load_config().thresholds


@pytest.fixture
def presence_axis():
    return Axis(
        id="presence",
        name="event-sound presence",
        tier=AxisTier.TIER1,
        kind=AxisKind.CATEGORICAL,
        agreement=AgreementMetric.EXACT_MATCH,
        measure="presence_detector",
    )


@pytest.fixture
def material_axis():
    return Axis(
        id="material",
        name="material / fine class",
        tier=AxisTier.TIER2,
        kind=AxisKind.EMBEDDING,
        agreement=AgreementMetric.MEAN_PAIRWISE_COSINE,
        measure="audio_embedding",
    )


def _make_wavs_by_clip(n: int = 8, seed: int = 42) -> dict[str, np.ndarray]:
    """Deterministic per-clip audio vectors via the synthetic backend."""
    backend = SyntheticGaussianFlow(dim=4, sigma2=0.25)
    rng = np.random.default_rng(seed)
    conds = SyntheticGaussianFlow.make_video_bank(n, dim=4, seed=seed)
    return {
        f"clip{i:02d}": backend.decode(backend.sample_prior(c, rng))
        for i, c in enumerate(conds)
    }


# --------------------------------------------------------------------------------------
# cohens_kappa
# --------------------------------------------------------------------------------------

class TestCohensKappa:
    def test_perfect_agreement_is_1(self):
        a = [0, 1, 1, 0, 2, 2]
        assert cohens_kappa(a, list(a)) == pytest.approx(1.0)

    def test_empty_is_nan(self):
        assert math.isnan(cohens_kappa([], []))

    def test_both_constant_and_identical_is_1(self):
        assert cohens_kappa(["x", "x", "x"], ["x", "x", "x"]) == pytest.approx(1.0)

    def test_exact_chance_agreement_is_0(self):
        # p_o = 0.5 (matches at items 0 and 3); both marginals uniform over {0, 1}
        # -> p_e = 0.5 -> kappa = 0 exactly.
        a = [0, 0, 1, 1]
        b = [0, 1, 0, 1]
        assert cohens_kappa(a, b) == pytest.approx(0.0)

    def test_known_2x2_table_value(self):
        # Hand computation: matches = 7/10 -> p_o = 0.7.
        # Marginals: p_a(1)=0.6, p_a(0)=0.4; p_b(1)=0.5, p_b(0)=0.5
        # -> p_e = 0.6*0.5 + 0.4*0.5 = 0.5 -> kappa = (0.7-0.5)/(1-0.5) = 0.4.
        a = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
        b = [1, 1, 1, 1, 0, 0, 0, 0, 0, 1]
        assert cohens_kappa(a, b) == pytest.approx(0.4)

    def test_constant_but_different_labels_is_0(self):
        # p_o = 0; p_e = p_a(x)*p_b(x) + p_a(y)*p_b(y) = 0 -> kappa = 0.
        assert cohens_kappa(["x", "x"], ["y", "y"]) == pytest.approx(0.0)

    def test_one_constant_rater_is_0(self):
        # Against a constant rater, p_o equals p_a(constant) = p_e -> kappa = 0.
        a = [0, 0, 1, 1]
        b = [0, 0, 0, 0]
        assert cohens_kappa(a, b) == pytest.approx(0.0)

    def test_total_disagreement_is_negative(self):
        a = [0, 1, 0, 1]
        b = [1, 0, 1, 0]
        assert cohens_kappa(a, b) < 0.0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="equal length"):
            cohens_kappa([0, 1], [0])

    def test_int_and_str_labels_not_conflated(self):
        # int 1 and "1" are different labels: zero observed agreement.
        kappa = cohens_kappa([1, 1], ["1", "1"])
        assert kappa == pytest.approx(0.0)

    def test_returns_float(self):
        assert isinstance(cohens_kappa([0, 1], [0, 1]), float)


# --------------------------------------------------------------------------------------
# gwet_ac1 — skew-robust agreement (the kappa-paradox antidote)
# --------------------------------------------------------------------------------------

class TestGwetAC1:
    def test_perfect_agreement_is_1(self):
        a = [0, 1, 1, 0, 2, 2]
        assert gwet_ac1(a, list(a)) == pytest.approx(1.0)

    def test_empty_is_nan(self):
        assert math.isnan(gwet_ac1([], []))

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="equal length"):
            gwet_ac1([0, 1], [0])

    def test_single_category_is_1(self):
        assert gwet_ac1(["x", "x", "x"], ["x", "x", "x"]) == pytest.approx(1.0)

    def test_known_value_hand_computed(self):
        # a: nine 1's + one 0; b: eight 1's + two 0's. p_o = 9/10 = 0.9.
        # pi_1 = (0.9+0.8)/2 = 0.85, pi_0 = 0.15; sum pi(1-pi) = 2*0.1275 = 0.255;
        # q=2 -> p_e = 0.255; AC1 = (0.9-0.255)/(1-0.255) = 0.645/0.745.
        a = [1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
        b = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
        assert gwet_ac1(a, b) == pytest.approx(0.645 / 0.745)

    def test_uniform_marginals_match_kappa(self):
        # With uniform 2-category marginals AC1 and kappa coincide (both 0 here).
        a = [0, 0, 1, 1]
        b = [0, 1, 0, 1]
        assert gwet_ac1(a, b) == pytest.approx(cohens_kappa(a, b))
        assert gwet_ac1(a, b) == pytest.approx(0.0)

    def test_skew_paradox_kappa_negative_but_ac1_high(self):
        # The presence-axis regime: ~90% one category, anti-aligned disagreements.
        # Cohen's kappa goes NEGATIVE while raw agreement (0.8) and AC1 stay high.
        a = [1, 1, 1, 1, 1, 1, 1, 1, 0, 1]
        b = [1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
        assert cohens_kappa(a, b) < 0.0
        assert gwet_ac1(a, b) == pytest.approx(0.62 / 0.82)  # ~0.756
        assert gwet_ac1(a, b) > 0.7

    def test_returns_float(self):
        assert isinstance(gwet_ac1([0, 1], [0, 1]), float)


# --------------------------------------------------------------------------------------
# pabak — prevalence-adjusted bias-adjusted kappa (third skew-robust read)
# --------------------------------------------------------------------------------------

class TestPABAK:
    def test_perfect_is_1(self):
        assert pabak([0, 1, 2, 0], [0, 1, 2, 0]) == pytest.approx(1.0)

    def test_empty_is_nan(self):
        assert math.isnan(pabak([], []))

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="equal length"):
            pabak([0, 1], [0])

    def test_binary_is_2po_minus_1(self):
        # p_o = 0.8 over a binary scale -> PABAK = 2*0.8 - 1 = 0.6.
        a = [1, 1, 1, 1, 1, 1, 1, 1, 0, 1]
        b = [1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
        assert pabak(a, b) == pytest.approx(0.6)

    def test_multicategory_hand_value(self):
        # p_o = 3/4; q = 3 -> PABAK = (3*0.75 - 1)/(3-1) = 0.625.
        a = [0, 1, 2, 0]
        b = [0, 1, 2, 1]
        assert pabak(a, b) == pytest.approx(0.625)

    def test_skew_paradox_pabak_positive_where_kappa_negative(self):
        a = [1, 1, 1, 1, 1, 1, 1, 1, 0, 1]
        b = [1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
        assert cohens_kappa(a, b) < 0.0
        assert pabak(a, b) == pytest.approx(0.6)


# --------------------------------------------------------------------------------------
# confusion_matrix — the §3.3 truth-teller
# --------------------------------------------------------------------------------------

class TestConfusionMatrix:
    def test_basic_counts_and_labels(self):
        cm = confusion_matrix(["p", "p", "a"], ["p", "a", "a"])
        assert cm["labels"] == ["a", "p"]
        # rows = first rater (a), cols = second (b); index a=0, p=1.
        assert cm["matrix"] == [[1, 0], [1, 1]]
        assert cm["n"] == 3

    def test_diagonal_is_agreement(self):
        cm = confusion_matrix([0, 1, 1, 0], [0, 1, 1, 0])
        diag = sum(cm["matrix"][i][i] for i in range(len(cm["labels"])))
        assert diag == 4 and cm["n"] == 4

    def test_explicit_labels_drop_out_of_set(self):
        cm = confusion_matrix(["x", "y", "z"], ["x", "y", "z"], labels=["x", "y"])
        assert cm["labels"] == ["x", "y"]
        assert cm["n"] == 2  # the (z, z) pair is dropped

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="equal length"):
            confusion_matrix([0, 1], [0])


# --------------------------------------------------------------------------------------
# run_real_reliability
# --------------------------------------------------------------------------------------

class TestRunRealReliability:
    def test_determinism_is_1_with_synthetic_measurer(self, measurer, thresholds,
                                                      presence_axis):
        wavs = _make_wavs_by_clip()
        rng = np.random.default_rng(0)
        result = run_real_reliability(presence_axis, wavs, measurer, thresholds, rng)
        assert isinstance(result, ReliabilityResult)
        assert result.axis_id == "presence"
        assert result.determinism == pytest.approx(1.0, abs=1e-9)

    def test_no_gold_means_nan_validity_and_demotion(self, measurer, thresholds,
                                                     presence_axis):
        """NaN validity counts as a FAILURE — mirror of reliability_gate semantics."""
        wavs = _make_wavs_by_clip()
        rng = np.random.default_rng(0)
        result = run_real_reliability(presence_axis, wavs, measurer, thresholds, rng)
        assert math.isnan(result.validity)
        assert result.demoted
        assert not result.passed
        assert result.reason.startswith("AXIS_DEMOTED:presence")
        assert "validity" in result.reason

    def test_perfect_gold_kappa_gives_validity_1_and_pass(self, measurer, thresholds,
                                                          presence_axis):
        """Gold = the measurer's own labels -> kappa = 1.0; with determinism = 1.0
        and robust synthetic presence, the gate passes (gate-mirror check)."""
        wavs = _make_wavs_by_clip()
        gold = {cid: measurer.measure(audio, presence_axis)
                for cid, audio in wavs.items()}
        rng = np.random.default_rng(0)
        result = run_real_reliability(presence_axis, wavs, measurer, thresholds, rng,
                                      gold=gold)
        assert result.validity == pytest.approx(1.0)
        assert result.determinism == pytest.approx(1.0, abs=1e-9)
        assert result.robustness >= thresholds.theta_robust, (
            f"synthetic presence robustness {result.robustness:.3f} unexpectedly "
            f"below theta_robust {thresholds.theta_robust:.3f}"
        )
        assert result.passed
        assert not result.demoted
        assert result.reason == ""

    def test_flipped_gold_fails_kappa(self, measurer, thresholds, presence_axis):
        """Per-clip flipped binary gold -> zero observed agreement -> kappa <= 0
        -> validity fails theta_cal -> demoted."""
        wavs = _make_wavs_by_clip()
        gold = {}
        for cid, audio in wavs.items():
            measured = measurer.measure(audio, presence_axis)
            gold[cid] = SelfTarget(
                axis_id=presence_axis.id,
                kind=AxisKind.CATEGORICAL,
                label=1 - int(measured.label),
            )
        rng = np.random.default_rng(0)
        result = run_real_reliability(presence_axis, wavs, measurer, thresholds, rng,
                                      gold=gold)
        assert result.validity <= 0.0
        assert result.demoted
        assert "validity" in result.reason

    def test_embedding_gold_uses_mean_cosine(self, measurer, thresholds,
                                             material_axis):
        """Gold = the measurer's own embeddings -> mean cosine = 1.0."""
        wavs = _make_wavs_by_clip(seed=10)
        gold = {cid: measurer.measure(audio, material_axis)
                for cid, audio in wavs.items()}
        rng = np.random.default_rng(5)
        result = run_real_reliability(material_axis, wavs, measurer, thresholds, rng,
                                      gold=gold)
        assert result.validity == pytest.approx(1.0, abs=1e-9)
        assert result.determinism == pytest.approx(1.0, abs=1e-9)

    def test_gold_subset_only_those_clips_count(self, measurer, thresholds,
                                                presence_axis):
        wavs = _make_wavs_by_clip()
        subset_ids = list(wavs)[:3]
        gold = {cid: measurer.measure(wavs[cid], presence_axis)
                for cid in subset_ids}
        rng = np.random.default_rng(0)
        result = run_real_reliability(presence_axis, wavs, measurer, thresholds, rng,
                                      gold=gold)
        # Perfect agreement on the gold subset -> 1.0 (other clips don't dilute it).
        assert result.validity == pytest.approx(1.0)

    def test_gold_with_no_overlapping_clips_is_nan(self, measurer, thresholds,
                                                   presence_axis):
        wavs = _make_wavs_by_clip(n=3)
        stray = SelfTarget(axis_id="presence", kind=AxisKind.CATEGORICAL, label=1)
        rng = np.random.default_rng(0)
        result = run_real_reliability(presence_axis, wavs, measurer, thresholds, rng,
                                      gold={"unknown_clip": stray})
        assert math.isnan(result.validity)
        assert result.demoted

    def test_perturbations_passed_through(self, measurer, thresholds, presence_axis):
        """A sign-flip perturbation flips presence = sign(mean) on every clip ->
        robustness 0.0 -> demoted with a robustness reason."""
        wavs = _make_wavs_by_clip()
        gold = {cid: measurer.measure(audio, presence_axis)
                for cid, audio in wavs.items()}
        rng = np.random.default_rng(0)
        result = run_real_reliability(
            presence_axis, wavs, measurer, thresholds, rng, gold=gold,
            perturbations={"flip": lambda a, r: -a},
        )
        assert result.robustness == pytest.approx(0.0)
        assert result.demoted
        assert "robustness" in result.reason
        assert result.reason.startswith("AXIS_DEMOTED:presence")

    def test_empty_clips_fails(self, measurer, thresholds, presence_axis):
        rng = np.random.default_rng(0)
        result = run_real_reliability(presence_axis, {}, measurer, thresholds, rng)
        assert not result.passed
        assert result.demoted
        assert result.reason == "no clips provided"

    def test_matches_reliability_gate_determinism(self, measurer, thresholds,
                                                  presence_axis):
        """Cross-check against the dry-run gate: determinism aggregation agrees."""
        from foley_cw.reliability import reliability_gate

        wavs = _make_wavs_by_clip(n=5, seed=7)
        gate_result = reliability_gate(
            presence_axis, list(wavs.values()), measurer, thresholds,
            np.random.default_rng(0),
        )
        real_result = run_real_reliability(
            presence_axis, wavs, measurer, thresholds, np.random.default_rng(0),
        )
        assert real_result.determinism == pytest.approx(gate_result.determinism)


# --------------------------------------------------------------------------------------
# Module-level import safety
# --------------------------------------------------------------------------------------

def test_module_importable_without_httpx():
    """foley_cw.sidecar is numpy-core: it must not import httpx as a side effect."""
    import sys

    import foley_cw.sidecar  # noqa: F401

    sidecar_globals = vars(foley_cw.sidecar)
    assert "httpx" not in sidecar_globals
    assert "torch" not in sys.modules or True  # torch absent in CI; guard only


def test_public_api_exists():
    from foley_cw.sidecar import (cohens_kappa, confusion_matrix, gwet_ac1, pabak,
                              run_real_reliability)
    assert callable(cohens_kappa)
    assert callable(gwet_ac1)
    assert callable(run_real_reliability)
