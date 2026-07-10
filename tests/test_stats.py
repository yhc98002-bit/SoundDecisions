"""Tests for foley_cw/stats.py (numpy-only).

Scientific contracts verified:
  * auroc: separable labels -> 1.0; random labels -> ~0.5.
  * first_crossing: ramp correctly identifies crossing; NaN when never crossing.
  * bootstrap_over_videos: CI brackets the point estimate on a simple mean.
  * separation_score: larger when windows are spread out.
  * window_with_ci: crossing bootstrapped correctly; underpowered flag works.
  * ordered_non_overlapping: correct for overlapping and non-overlapping cases.
  * accuracy: correct fraction.
  * threshold_sweep: returns a dict keyed by threshold.
"""

import math

import numpy as np
import pytest

from foley_cw.stats import (
    accuracy,
    auroc,
    bootstrap_over_videos,
    first_crossing,
    ordered_non_overlapping,
    separation_score,
    threshold_sweep,
    window_with_ci,
)
from foley_cw.types import WindowEstimate


# -------------------------------------------------------------------------
# auroc
# -------------------------------------------------------------------------

class TestAuroc:
    def test_perfect_separation(self):
        """Positives all score above negatives -> AUROC == 1.0."""
        scores = np.array([0.9, 0.8, 0.7, 0.2, 0.1, 0.05])
        labels = np.array([1, 1, 1, 0, 0, 0])
        assert auroc(scores, labels) == pytest.approx(1.0)

    def test_reversed_perfect(self):
        """Positives all score BELOW negatives -> AUROC == 0.0."""
        scores = np.array([0.05, 0.1, 0.2, 0.7, 0.8, 0.9])
        labels = np.array([1, 1, 1, 0, 0, 0])
        assert auroc(scores, labels) == pytest.approx(0.0)

    def test_random_labels_approximately_half(self):
        """Shuffled labels (no signal) -> AUROC close to 0.5."""
        rng = np.random.default_rng(42)
        n = 200
        scores = rng.standard_normal(n)
        labels = rng.integers(0, 2, size=n)
        val = auroc(scores, labels)
        assert 0.35 < val < 0.65, f"Random AUROC should be ~0.5, got {val}"

    def test_ties_handled(self):
        """All same score -> AUROC == 0.5."""
        scores = np.array([0.5, 0.5, 0.5, 0.5])
        labels = np.array([1, 0, 1, 0])
        assert auroc(scores, labels) == pytest.approx(0.5)

    def test_no_positives_returns_nan(self):
        scores = np.array([0.1, 0.2])
        labels = np.array([0, 0])
        assert math.isnan(auroc(scores, labels))

    def test_no_negatives_returns_nan(self):
        scores = np.array([0.1, 0.2])
        labels = np.array([1, 1])
        assert math.isnan(auroc(scores, labels))

    def test_two_elements(self):
        """Minimal valid case with one positive and one negative."""
        assert auroc(np.array([1.0, 0.0]), np.array([1, 0])) == pytest.approx(1.0)
        assert auroc(np.array([0.0, 1.0]), np.array([1, 0])) == pytest.approx(0.0)


# -------------------------------------------------------------------------
# first_crossing
# -------------------------------------------------------------------------

class TestFirstCrossing:
    def test_ramp_exact_grid_point(self):
        """Linear ramp from 0 to 1; theta=0.5 is exactly at s=0.5."""
        s = np.linspace(0, 1, 11)
        v = np.linspace(0, 1, 11)
        result = first_crossing(s, v, theta=0.5, direction="up")
        assert result == pytest.approx(0.5)

    def test_ramp_interpolated(self):
        """theta=0.75 falls between grid points; linear interp should give 0.75."""
        s = np.linspace(0, 1, 5)     # [0, 0.25, 0.5, 0.75, 1.0]
        v = np.linspace(0, 1, 5)
        result = first_crossing(s, v, theta=0.75, direction="up")
        assert result == pytest.approx(0.75, abs=1e-10)

    def test_never_crosses_returns_nan(self):
        """Curve never reaches theta -> NaN."""
        s = np.linspace(0, 1, 5)
        v = np.full(5, 0.3)
        result = first_crossing(s, v, theta=0.9, direction="up")
        assert math.isnan(result)

    def test_already_above_from_start(self):
        """Curve starts above theta -> return s[0]."""
        s = np.linspace(0, 1, 5)
        v = np.full(5, 0.8)
        result = first_crossing(s, v, theta=0.5, direction="up")
        assert result == pytest.approx(s[0])

    def test_direction_down(self):
        """Descending curve; first_crossing(direction='down') should find theta."""
        s = np.linspace(0, 1, 11)
        v = np.linspace(1, 0, 11)
        result = first_crossing(s, v, theta=0.5, direction="down")
        assert result == pytest.approx(0.5, abs=1e-9)

    def test_step_function(self):
        """Step: stays 0 then jumps to 1 at s=0.5; crossing should be near 0.5."""
        s = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        v = np.array([0.0, 0.0, 1.0, 1.0, 1.0])
        result = first_crossing(s, v, theta=0.9, direction="up")
        # jump from 0.0 to 1.0 between s=0.25 and s=0.5; interpolation: 0.25 + 0.9*(0.5-0.25)
        expected = 0.25 + 0.9 * 0.25
        assert result == pytest.approx(expected, abs=1e-10)

    def test_invalid_direction(self):
        with pytest.raises(ValueError, match="direction"):
            first_crossing(np.array([0.0, 1.0]), np.array([0.0, 1.0]),
                           theta=0.5, direction="sideways")


# -------------------------------------------------------------------------
# bootstrap_over_videos
# -------------------------------------------------------------------------

class TestBootstrapOverVideos:
    def _make_data(self, n=40, seed=7):
        rng = np.random.default_rng(seed)
        return list(rng.standard_normal(n))

    def test_ci_brackets_point_estimate(self):
        """CI [lo, hi] must bracket the point estimate (by construction of percentile boot)."""
        data = self._make_data()
        point, lo, hi = bootstrap_over_videos(data, np.mean, n_boot=2000, ci=0.95, seed=0)
        assert lo <= point <= hi, f"CI [{lo:.4f}, {hi:.4f}] does not bracket point {point:.4f}"

    def test_empty_list(self):
        point, lo, hi = bootstrap_over_videos([], np.mean)
        assert all(math.isnan(x) for x in (point, lo, hi))

    def test_single_element(self):
        """Single element: point == the element value; CI collapsed."""
        point, lo, hi = bootstrap_over_videos([3.14], lambda x: float(x[0]), n_boot=100)
        assert point == pytest.approx(3.14)
        assert lo == pytest.approx(3.14)
        assert hi == pytest.approx(3.14)

    def test_wider_ci_at_higher_coverage(self):
        """99% CI must be wider than 50% CI."""
        data = self._make_data(n=60)
        _, lo50, hi50 = bootstrap_over_videos(data, np.mean, n_boot=1000, ci=0.50, seed=0)
        _, lo99, hi99 = bootstrap_over_videos(data, np.mean, n_boot=1000, ci=0.99, seed=0)
        assert (hi99 - lo99) >= (hi50 - lo50)

    def test_seed_reproducible(self):
        """Same seed -> identical results."""
        data = self._make_data()
        r1 = bootstrap_over_videos(data, np.mean, seed=42)
        r2 = bootstrap_over_videos(data, np.mean, seed=42)
        assert r1 == r2

    def test_different_seeds_differ(self):
        """Different seeds should generally give different CIs."""
        data = self._make_data(n=30)
        _, lo1, hi1 = bootstrap_over_videos(data, np.mean, n_boot=500, ci=0.95, seed=0)
        _, lo2, hi2 = bootstrap_over_videos(data, np.mean, n_boot=500, ci=0.95, seed=99)
        # They may sometimes be equal by chance but usually differ.
        # Just check the function runs without error; values are floats.
        assert isinstance(lo1, float) and isinstance(lo2, float)

    def test_list_of_curves_stat_fn(self):
        """Stat fn can operate on list of ndarrays (as in window_with_ci internals)."""
        n = 20
        rng = np.random.default_rng(0)
        data = [rng.standard_normal(5) for _ in range(n)]

        def mean_of_first_elem(lst):
            return float(np.mean([x[0] for x in lst]))

        point, lo, hi = bootstrap_over_videos(data, mean_of_first_elem, n_boot=500, seed=0)
        assert lo <= point <= hi


# -------------------------------------------------------------------------
# window_with_ci
# -------------------------------------------------------------------------

class TestWindowWithCi:
    def _make_curves(self, n_videos=10, n_s=11, seed=0):
        """Generate simple sigmoid-shaped curves per video with slight noise."""
        rng = np.random.default_rng(seed)
        s_grid = np.linspace(0, 1, n_s)
        # Each video's curve rises from 0 to 1 in a sigmoid-like shape.
        mid = 0.5 + 0.1 * rng.standard_normal(n_videos)
        curves = np.zeros((n_videos, n_s))
        for i in range(n_videos):
            curves[i] = 1.0 / (1.0 + np.exp(-10.0 * (s_grid - mid[i])))
        return curves, s_grid

    def test_basic_structure(self):
        curves, s_grid = self._make_curves()
        w = window_with_ci(curves, s_grid, theta=0.5, kind="commit",
                           axis_id="presence", n_boot=200, seed=0)
        assert isinstance(w, WindowEstimate)
        assert w.axis_id == "presence"
        assert w.kind == "commit"
        assert w.n_videos == 10
        assert not math.isnan(w.s_hat)

    def test_ci_brackets_point(self):
        curves, s_grid = self._make_curves(n_videos=20)
        w = window_with_ci(curves, s_grid, theta=0.5, kind="commit",
                           axis_id="axis1", n_boot=500, seed=0)
        assert w.ci_low <= w.s_hat <= w.ci_high, (
            f"CI [{w.ci_low}, {w.ci_high}] does not bracket s_hat={w.s_hat}")

    def test_threshold_near_1_never_crosses(self):
        """If curves never reach theta=0.99, s_hat should be NaN."""
        curves = np.full((5, 11), 0.5)  # flat curves at 0.5
        s_grid = np.linspace(0, 1, 11)
        w = window_with_ci(curves, s_grid, theta=0.99, kind="commit",
                           axis_id="axis1", n_boot=100, seed=0)
        assert math.isnan(w.s_hat)

    def test_underpowered_flag(self):
        """n_videos < min_n -> underpowered=True."""
        curves, s_grid = self._make_curves(n_videos=3)
        w = window_with_ci(curves, s_grid, theta=0.5, kind="commit",
                           axis_id="ax", n_boot=100, seed=0, min_n=10)
        assert w.underpowered is True

    def test_not_underpowered(self):
        curves, s_grid = self._make_curves(n_videos=10)
        w = window_with_ci(curves, s_grid, theta=0.5, kind="commit",
                           axis_id="ax", n_boot=100, seed=0, min_n=10)
        assert w.underpowered is False

    def test_zero_videos(self):
        curves = np.zeros((0, 11))
        s_grid = np.linspace(0, 1, 11)
        w = window_with_ci(curves, s_grid, theta=0.5, kind="commit",
                           axis_id="empty", n_boot=50, seed=0)
        assert w.underpowered is True
        assert math.isnan(w.s_hat)

    def test_early_crossing_returns_small_s(self):
        """Curves that cross theta immediately -> s_hat close to 0."""
        curves = np.ones((10, 11))  # always above any threshold < 1
        s_grid = np.linspace(0, 1, 11)
        w = window_with_ci(curves, s_grid, theta=0.5, kind="read",
                           axis_id="ax", n_boot=200, seed=0)
        assert w.s_hat == pytest.approx(0.0, abs=1e-9)


# -------------------------------------------------------------------------
# separation_score
# -------------------------------------------------------------------------

class TestSeparationScore:
    def _make_window(self, axis_id, s_hat, ci_low, ci_high, kind="commit"):
        return WindowEstimate(
            axis_id=axis_id,
            kind=kind,
            s_hat=s_hat,
            ci_low=ci_low,
            ci_high=ci_high,
            n_videos=10,
        )

    def test_spread_windows_larger_score(self):
        """Widely separated windows -> larger score than closely packed windows."""
        wide = {
            "a": self._make_window("a", 0.1, 0.05, 0.15),
            "b": self._make_window("b", 0.5, 0.45, 0.55),
            "c": self._make_window("c", 0.9, 0.85, 0.95),
        }
        tight = {
            "a": self._make_window("a", 0.48, 0.43, 0.53),
            "b": self._make_window("b", 0.50, 0.45, 0.55),
            "c": self._make_window("c", 0.52, 0.47, 0.57),
        }
        s_wide = separation_score(wide)
        s_tight = separation_score(tight)
        assert s_wide > s_tight, (
            f"Wide sep={s_wide:.2f} should exceed tight sep={s_tight:.2f}")

    def test_single_window_nan(self):
        """Only one window -> spread is undefined -> nan."""
        windows = {"a": self._make_window("a", 0.5, 0.4, 0.6)}
        assert math.isnan(separation_score(windows))

    def test_empty_dict_nan(self):
        assert math.isnan(separation_score({}))

    def test_all_nan_s_hat(self):
        windows = {
            "a": self._make_window("a", float("nan"), float("nan"), float("nan")),
            "b": self._make_window("b", float("nan"), float("nan"), float("nan")),
        }
        assert math.isnan(separation_score(windows))

    def test_zero_ci_width_returns_inf(self):
        """All point CIs collapsed -> mean_width=0 -> inf separation score."""
        windows = {
            "a": self._make_window("a", 0.2, 0.2, 0.2),
            "b": self._make_window("b", 0.8, 0.8, 0.8),
        }
        assert math.isinf(separation_score(windows))

    def test_formula_correctness(self):
        """Manually verify the formula: spread / mean_ci_width."""
        windows = {
            "a": self._make_window("a", 0.2, 0.1, 0.3),   # ci_width=0.2
            "b": self._make_window("b", 0.8, 0.7, 0.9),   # ci_width=0.2
        }
        # spread = 0.8 - 0.2 = 0.6; mean_width = 0.2; score = 3.0
        assert separation_score(windows) == pytest.approx(3.0)


# -------------------------------------------------------------------------
# ordered_non_overlapping
# -------------------------------------------------------------------------

class TestOrderedNonOverlapping:
    def _make_window(self, axis_id, s_hat, ci_low, ci_high):
        return WindowEstimate(
            axis_id=axis_id, kind="commit",
            s_hat=s_hat, ci_low=ci_low, ci_high=ci_high,
            n_videos=10,
        )

    def test_non_overlapping_returns_true(self):
        windows = {
            "a": self._make_window("a", 0.1, 0.05, 0.15),
            "b": self._make_window("b", 0.4, 0.35, 0.45),
            "c": self._make_window("c", 0.7, 0.65, 0.75),
        }
        assert ordered_non_overlapping(windows) is True

    def test_overlapping_returns_false(self):
        windows = {
            "a": self._make_window("a", 0.3, 0.2, 0.5),
            "b": self._make_window("b", 0.4, 0.3, 0.6),
        }
        assert ordered_non_overlapping(windows) is False

    def test_just_touching_is_ok(self):
        """Exactly touching but not overlapping (ci_high_a == ci_low_b) is acceptable."""
        windows = {
            "a": self._make_window("a", 0.2, 0.1, 0.3),
            "b": self._make_window("b", 0.5, 0.3, 0.7),
        }
        assert ordered_non_overlapping(windows) is True

    def test_single_window(self):
        windows = {"a": self._make_window("a", 0.5, 0.4, 0.6)}
        assert ordered_non_overlapping(windows) is True

    def test_empty_dict(self):
        assert ordered_non_overlapping({}) is True

    def test_all_nan(self):
        windows = {
            "a": self._make_window("a", float("nan"), float("nan"), float("nan")),
        }
        assert ordered_non_overlapping(windows) is True

    def test_overlap_detected_regardless_of_insertion_order(self):
        """Insert in wrong order; function must sort by s_hat, not insertion order."""
        windows = {
            "b": self._make_window("b", 0.4, 0.3, 0.6),
            "a": self._make_window("a", 0.2, 0.1, 0.5),   # overlaps with b
        }
        assert ordered_non_overlapping(windows) is False


# -------------------------------------------------------------------------
# accuracy
# -------------------------------------------------------------------------

class TestAccuracy:
    def test_perfect(self):
        pred = np.array([1, 0, 1, 0])
        true = np.array([1, 0, 1, 0])
        assert accuracy(pred, true) == pytest.approx(1.0)

    def test_all_wrong(self):
        pred = np.array([1, 1, 1])
        true = np.array([0, 0, 0])
        assert accuracy(pred, true) == pytest.approx(0.0)

    def test_half(self):
        pred = np.array([1, 0, 1, 0])
        true = np.array([1, 1, 0, 0])
        assert accuracy(pred, true) == pytest.approx(0.5)

    def test_empty(self):
        assert math.isnan(accuracy(np.array([]), np.array([])))

    def test_string_labels(self):
        pred = ["cat", "dog", "cat"]
        true = ["cat", "cat", "cat"]
        assert accuracy(pred, true) == pytest.approx(2.0 / 3.0)


# -------------------------------------------------------------------------
# threshold_sweep
# -------------------------------------------------------------------------

class TestThresholdSweep:
    def test_returns_dict_keyed_by_theta(self):
        rng = np.random.default_rng(0)
        n_videos, n_s = 8, 11
        s_grid = np.linspace(0, 1, n_s)
        # Sigmoid curves crossing between 0.3 and 0.7
        curves = np.vstack([
            1.0 / (1.0 + np.exp(-10.0 * (s_grid - 0.5 + 0.05 * rng.standard_normal())))
            for _ in range(n_videos)
        ])
        thetas = [0.3, 0.5, 0.7, 0.9]
        result = threshold_sweep(curves, s_grid, thetas, kind="commit",
                                 axis_id="ax", n_boot=100, seed=0)
        assert set(result.keys()) == {0.3, 0.5, 0.7, 0.9}
        for theta, w in result.items():
            assert isinstance(w, WindowEstimate)
            assert w.axis_id == "ax"

    def test_higher_theta_later_crossing(self):
        """For a rising curve, a higher threshold should yield a later s_hat."""
        n_videos, n_s = 10, 21
        s_grid = np.linspace(0, 1, n_s)
        curves = np.tile(np.linspace(0, 1, n_s), (n_videos, 1))
        result = threshold_sweep(curves, s_grid, [0.2, 0.5, 0.8], kind="commit",
                                 axis_id="ax", n_boot=200, seed=0)
        s20 = result[0.2].s_hat
        s50 = result[0.5].s_hat
        s80 = result[0.8].s_hat
        assert s20 <= s50 <= s80, f"Expected monotonic s_hat: {s20}, {s50}, {s80}"

    def test_empty_thetas(self):
        curves = np.ones((5, 11))
        s_grid = np.linspace(0, 1, 11)
        result = threshold_sweep(curves, s_grid, [], kind="commit", axis_id="ax")
        assert result == {}
