"""Tests for foley_cw/agreement.py — numpy-only, no scipy, no torch.

Tests cover:
  - categorical_agreement: all-identical => 1.0; maximally diverse => low; n<2 => 1.0
  - krippendorff_alpha_nominal: all-identical => 1.0; single-category => 1.0; n<2 => 1.0;
    maximally diverse => low; uniform two-class => correct analytic value
  - mean_pairwise_cosine: aligned => high; orthogonal => 0.0; anti-aligned => -1.0; n<2 => 1.0
  - agreement dispatch: correct routing per AgreementMetric; n<2 => 1.0
  - SelfTarget construction for each metric kind
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from foley_cw.agreement import (
    categorical_agreement,
    krippendorff_alpha_nominal,
    mean_pairwise_cosine,
    agreement,
)
from foley_cw.types import AgreementMetric, AxisKind, SelfTarget


# ============================================================================
# Helpers
# ============================================================================

def make_cat_target(axis_id: str, label) -> SelfTarget:
    return SelfTarget(axis_id=axis_id, kind=AxisKind.CATEGORICAL, label=label)


def make_emb_target(axis_id: str, vec: np.ndarray) -> SelfTarget:
    return SelfTarget(axis_id=axis_id, kind=AxisKind.EMBEDDING, embedding=vec)


# ============================================================================
# categorical_agreement
# ============================================================================

class TestCategoricalAgreement:
    def test_all_identical_returns_1(self):
        assert categorical_agreement([3, 3, 3, 3]) == pytest.approx(1.0)

    def test_all_identical_strings(self):
        assert categorical_agreement(["dog", "dog", "dog"]) == pytest.approx(1.0)

    def test_all_different_n2(self):
        # 2 items, both different -> 0 matching pairs out of 1 -> 0.0
        assert categorical_agreement([0, 1]) == pytest.approx(0.0)

    def test_all_different_n4(self):
        # 4 unique labels -> 0 matching pairs out of 6 -> 0.0
        assert categorical_agreement([0, 1, 2, 3]) == pytest.approx(0.0)

    def test_half_and_half_n4(self):
        # [A, A, B, B]: pairs AA=1, BB=1 match, AB pairs=4 don't -> 2/6
        labels = ["A", "A", "B", "B"]
        result = categorical_agreement(labels)
        assert result == pytest.approx(2.0 / 6.0)

    def test_n0_returns_1(self):
        assert categorical_agreement([]) == pytest.approx(1.0)

    def test_n1_returns_1(self):
        assert categorical_agreement([42]) == pytest.approx(1.0)

    def test_range_in_0_1(self):
        for labels in [[1, 2, 3], [1, 1, 2], [1, 1, 1, 1], [1, 2]]:
            val = categorical_agreement(labels)
            assert 0.0 <= val <= 1.0, f"Out of [0,1] for {labels}: {val}"

    def test_three_of_four_same(self):
        # [A, A, A, B]: matching pairs among 3 A's = 3; total = 6 -> 3/6 = 0.5
        val = categorical_agreement(["A", "A", "A", "B"])
        assert val == pytest.approx(3.0 / 6.0)

    def test_symmetric_with_order(self):
        # Order should not matter
        a = categorical_agreement([1, 2, 1, 2])
        b = categorical_agreement([2, 1, 2, 1])
        assert a == pytest.approx(b)


# ============================================================================
# krippendorff_alpha_nominal
# ============================================================================

class TestKrippendorffAlphaNominal:
    def test_all_identical_returns_1(self):
        assert krippendorff_alpha_nominal([5, 5, 5, 5]) == pytest.approx(1.0)

    def test_all_identical_strings(self):
        assert krippendorff_alpha_nominal(["cat", "cat", "cat"]) == pytest.approx(1.0)

    def test_n0_returns_1(self):
        assert krippendorff_alpha_nominal([]) == pytest.approx(1.0)

    def test_n1_returns_1(self):
        assert krippendorff_alpha_nominal([7]) == pytest.approx(1.0)

    def test_all_different_is_low_or_negative(self):
        # All different labels -> alpha should be <= 0
        val = krippendorff_alpha_nominal([0, 1, 2, 3, 4])
        assert val <= 0.0, f"Expected <= 0 for all-different, got {val}"

    def test_no_nan_all_same(self):
        val = krippendorff_alpha_nominal([1, 1, 1])
        assert not math.isnan(val)

    def test_no_nan_all_different(self):
        val = krippendorff_alpha_nominal([1, 2, 3])
        assert not math.isnan(val)

    def test_no_nan_n2_equal(self):
        val = krippendorff_alpha_nominal([1, 1])
        assert not math.isnan(val)
        assert val == pytest.approx(1.0)

    def test_no_nan_n2_different(self):
        # [A, B]: D_o = 1.0; D_e = 1.0 (two equal-frequency categories) -> alpha = 0.0
        val = krippendorff_alpha_nominal([0, 1])
        assert not math.isnan(val)
        assert val == pytest.approx(0.0)

    def test_uniform_two_class_n4(self):
        # [A, A, B, B]: D_o = 8/12; n_A=2, n_B=2
        # D_e = (4*3 - (2*1 + 2*1)) / (4*3) = (12 - 4) / 12 = 8/12 = 2/3
        # alpha = 1 - D_o/D_e; D_o = 8/12 = 2/3; alpha = 1 - 1 = 0.0
        val = krippendorff_alpha_nominal(["A", "A", "B", "B"])
        assert not math.isnan(val)
        assert val == pytest.approx(0.0, abs=1e-9)

    def test_highly_consistent_analytic(self):
        # For n=8, n_A=7, n_B=1:
        # D_o = 2*7*1 / (8*7) = 14/56 = 1/4
        # D_e = (8*7 - 7*6 - 1*0) / (8*7) = (56 - 42) / 56 = 14/56 = 1/4
        # alpha = 1 - D_o/D_e = 0.0
        # (Krippendorff's alpha is relative to chance; this case is exactly chance.)
        val = krippendorff_alpha_nominal(["A"] * 7 + ["B"])
        assert val == pytest.approx(0.0, abs=1e-9)

    def test_perfect_consistency_is_1(self):
        val = krippendorff_alpha_nominal([1, 1, 1, 1, 1])
        assert val == pytest.approx(1.0)

    def test_single_category_returns_1(self):
        # Only one distinct category
        val = krippendorff_alpha_nominal(["X", "X", "X"])
        assert val == pytest.approx(1.0)

    def test_maximally_diverse_is_low(self):
        # Every element is different -> alpha <= 0
        labels = list(range(10))
        val = krippendorff_alpha_nominal(labels)
        assert val <= 0.0

    def test_analytic_three_classes_equal(self):
        # n=6, two of each class A,B,C
        # D_o: disagreeing pairs / total pairs
        # n*(n-1)=30 ordered pairs; same-class pairs: 3 * (2*1)=6 agree, 24 disagree
        # D_o = 24/30 = 4/5
        # D_e = (30 - (2*1 + 2*1 + 2*1)) / 30 = (30 - 6) / 30 = 24/30 = 4/5
        # alpha = 1 - (4/5)/(4/5) = 0.0
        labels = ["A", "A", "B", "B", "C", "C"]
        val = krippendorff_alpha_nominal(labels)
        assert val == pytest.approx(0.0, abs=1e-9)


# ============================================================================
# mean_pairwise_cosine
# ============================================================================

class TestMeanPairwiseCosine:
    def test_identical_unit_vectors_returns_1(self):
        v = np.array([[1.0, 0.0, 0.0],
                      [1.0, 0.0, 0.0],
                      [1.0, 0.0, 0.0]])
        assert mean_pairwise_cosine(v) == pytest.approx(1.0)

    def test_orthogonal_pair_returns_0(self):
        v = np.array([[1.0, 0.0],
                      [0.0, 1.0]])
        assert mean_pairwise_cosine(v) == pytest.approx(0.0, abs=1e-9)

    def test_anti_aligned_pair_returns_neg1(self):
        v = np.array([[1.0, 0.0],
                      [-1.0, 0.0]])
        assert mean_pairwise_cosine(v) == pytest.approx(-1.0)

    def test_n0_returns_1(self):
        v = np.zeros((0, 3))
        assert mean_pairwise_cosine(v) == pytest.approx(1.0)

    def test_n1_returns_1(self):
        v = np.array([[0.5, 0.5]])
        assert mean_pairwise_cosine(v) == pytest.approx(1.0)

    def test_unnormalized_same_direction_returns_1(self):
        # Vectors pointing in same direction, different magnitudes
        v = np.array([[2.0, 0.0],
                      [5.0, 0.0]])
        assert mean_pairwise_cosine(v) == pytest.approx(1.0)

    def test_zero_norm_vector_no_error(self):
        # Zero-norm vector: cosine with anything is guarded
        v = np.array([[0.0, 0.0],
                      [1.0, 0.0]])
        val = mean_pairwise_cosine(v)
        assert not math.isnan(val)

    def test_result_in_neg1_1(self):
        rng = np.random.default_rng(42)
        for _ in range(10):
            v = rng.standard_normal((5, 4))
            val = mean_pairwise_cosine(v)
            assert -1.0 <= val <= 1.0, f"Out of range: {val}"

    def test_three_orthogonal_vectors(self):
        v = np.eye(3)   # 3 mutually orthogonal unit vectors
        # All pairs have cosine 0 -> mean = 0
        assert mean_pairwise_cosine(v) == pytest.approx(0.0, abs=1e-9)

    def test_high_alignment_implies_high_value(self):
        rng = np.random.default_rng(7)
        base = np.array([1.0, 0.0, 0.0, 0.0])
        noise = rng.standard_normal((6, 4)) * 0.01
        vecs = base + noise
        val = mean_pairwise_cosine(vecs)
        assert val > 0.99

    def test_n2_analytic(self):
        # Vectors at 45 degrees -> cosine = cos(pi/4) = sqrt(2)/2
        v = np.array([[1.0, 0.0],
                      [1.0, 1.0]])
        expected = 1.0 / math.sqrt(2)
        assert mean_pairwise_cosine(v) == pytest.approx(expected, rel=1e-6)


# ============================================================================
# agreement dispatch
# ============================================================================

class TestAgreementDispatch:
    def test_exact_match_dispatch_all_same(self):
        targets = [make_cat_target("presence", 1)] * 5
        val = agreement(targets, AgreementMetric.EXACT_MATCH)
        assert val == pytest.approx(1.0)

    def test_exact_match_dispatch_all_different(self):
        targets = [make_cat_target("presence", i) for i in range(4)]
        val = agreement(targets, AgreementMetric.EXACT_MATCH)
        assert val == pytest.approx(0.0)

    def test_krippendorff_dispatch_all_same(self):
        targets = [make_cat_target("class", "dog")] * 6
        val = agreement(targets, AgreementMetric.KRIPPENDORFF_ALPHA)
        assert val == pytest.approx(1.0)

    def test_krippendorff_dispatch_two_equal_classes(self):
        targets = ([make_cat_target("class", "A")] * 3 +
                   [make_cat_target("class", "B")] * 3)
        val = agreement(targets, AgreementMetric.KRIPPENDORFF_ALPHA)
        # Equal splits -> alpha = 0.0 (see analytic test above)
        assert val == pytest.approx(0.0, abs=1e-9)

    def test_cosine_dispatch_aligned(self):
        v = np.array([1.0, 0.0, 0.0, 0.0])
        targets = [make_emb_target("material", v.copy()) for _ in range(4)]
        val = agreement(targets, AgreementMetric.MEAN_PAIRWISE_COSINE)
        assert val == pytest.approx(1.0)

    def test_cosine_dispatch_orthogonal_pair(self):
        e1 = np.array([1.0, 0.0])
        e2 = np.array([0.0, 1.0])
        targets = [make_emb_target("material", e1), make_emb_target("material", e2)]
        val = agreement(targets, AgreementMetric.MEAN_PAIRWISE_COSINE)
        assert val == pytest.approx(0.0, abs=1e-9)

    def test_n0_returns_1_all_metrics(self):
        for metric in AgreementMetric:
            val = agreement([], metric)
            assert val == pytest.approx(1.0), f"Failed for {metric}"

    def test_n1_returns_1_exact_match(self):
        targets = [make_cat_target("presence", 1)]
        assert agreement(targets, AgreementMetric.EXACT_MATCH) == pytest.approx(1.0)

    def test_n1_returns_1_krippendorff(self):
        targets = [make_cat_target("class", "cat")]
        assert agreement(targets, AgreementMetric.KRIPPENDORFF_ALPHA) == pytest.approx(1.0)

    def test_n1_returns_1_cosine(self):
        targets = [make_emb_target("material", np.array([1.0, 0.0]))]
        assert agreement(targets, AgreementMetric.MEAN_PAIRWISE_COSINE) == pytest.approx(1.0)

    def test_unknown_metric_raises(self):
        targets = [make_cat_target("presence", 1), make_cat_target("presence", 2)]
        with pytest.raises((ValueError, AttributeError)):
            agreement(targets, "bad_metric")  # type: ignore[arg-type]

    def test_label_can_be_tuple(self):
        # binding axis uses tuple labels (sign, sign)
        targets = [
            make_cat_target("binding", (1, -1)),
            make_cat_target("binding", (1, -1)),
            make_cat_target("binding", (1,  1)),
        ]
        val = agreement(targets, AgreementMetric.EXACT_MATCH)
        # Pairs (0,1)=match, (0,2)=no, (1,2)=no -> 1/3
        assert val == pytest.approx(1.0 / 3.0)

    def test_mixed_labels_categorical(self):
        # Agreement for 3 forks: [A, A, B] -> 1 match out of 3 pairs
        targets = [
            make_cat_target("presence", "A"),
            make_cat_target("presence", "A"),
            make_cat_target("presence", "B"),
        ]
        val = agreement(targets, AgreementMetric.EXACT_MATCH)
        assert val == pytest.approx(1.0 / 3.0)


# ============================================================================
# Self-consistency: agreement == 1 when all targets identical
# ============================================================================

class TestSelfConsistency:
    """Stress-test: homogeneous batches always return 1.0."""

    @pytest.mark.parametrize("n", [2, 3, 4, 8, 16])
    def test_exact_match_homogeneous(self, n):
        targets = [make_cat_target("presence", 42)] * n
        assert agreement(targets, AgreementMetric.EXACT_MATCH) == pytest.approx(1.0)

    @pytest.mark.parametrize("n", [2, 3, 4, 8, 16])
    def test_krippendorff_homogeneous(self, n):
        targets = [make_cat_target("class", "cat")] * n
        assert agreement(targets, AgreementMetric.KRIPPENDORFF_ALPHA) == pytest.approx(1.0)

    @pytest.mark.parametrize("n", [2, 3, 4, 8, 16])
    def test_cosine_homogeneous(self, n):
        v = np.array([0.6, 0.8, 0.0])
        targets = [make_emb_target("material", v.copy()) for _ in range(n)]
        assert agreement(targets, AgreementMetric.MEAN_PAIRWISE_COSINE) == pytest.approx(1.0)


# ============================================================================
# Import smoke-test: module imports with just numpy
# ============================================================================

def test_import_agreement():
    """Module must import without scipy, torch, or librosa."""
    import importlib
    mod = importlib.import_module("foley_cw.agreement")
    assert hasattr(mod, "categorical_agreement")
    assert hasattr(mod, "krippendorff_alpha_nominal")
    assert hasattr(mod, "mean_pairwise_cosine")
    assert hasattr(mod, "agreement")
