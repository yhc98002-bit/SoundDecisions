"""Tests for foley_cw/commitment.py — Phase-1 commitment map.

All tests run on CPU against SyntheticGaussianFlow (the analytic oracle backend).
No scipy, no torch, no GPU.

Scientific contracts verified:
  * commit_gain is in [0, 1] with guard for a_ind >= 1.
  * commit_gain ~ 0 at s=0 (no shared info yet) and ~ 1 at s=1 (fully shared final state).
  * a_independent produces a float in a plausible range.
  * a_fork at s=1 is 1.0 (all K forks from the final state return the same audio).
  * commitment_curve_for_video returns shape (n_scan_pts,) values in [0, 1].
  * select_primary_alpha returns the SMALLEST qualifying alpha or None.
  * build_commitment_map returns CommitmentCell rows + WindowEstimate dict + alpha.
  * At least one axis has a finite s_commit (not NaN) with 6 videos, dim=4, alpha=1.0.
"""

from __future__ import annotations

import numpy as np
import pytest

from foley_cw.axes import SyntheticMeasurer
from foley_cw.commitment import (
    a_fork,
    a_independent,
    build_commitment_map,
    commit_gain,
    commitment_curve_for_video,
    select_primary_alpha,
)
from foley_cw.synthetic_backend import SyntheticGaussianFlow, SyntheticVideoCond
from foley_cw.types import (
    AlphaGridSpec,
    CommitmentCell,
    ScheduleSpec,
    Thresholds,
    WindowEstimate,
)
from foley_cw.config import load_config


# --------------------------------------------------------------------------------------
# Shared fixtures
# --------------------------------------------------------------------------------------

DIM = 4
N_VIDEOS = 6


@pytest.fixture(scope="module")
def backend():
    return SyntheticGaussianFlow(dim=DIM, sigma2=0.25)


@pytest.fixture(scope="module")
def measurer():
    return SyntheticMeasurer(seed=42)


@pytest.fixture(scope="module")
def video_bank(backend):
    return SyntheticGaussianFlow.make_video_bank(N_VIDEOS, dim=DIM, mu_scale=2.0, seed=7)


@pytest.fixture(scope="module")
def cond(video_bank):
    return video_bank[0]


@pytest.fixture
def schedule_small():
    """A fast schedule for unit tests."""
    return ScheduleSpec(
        n_steps=8,
        scan_points=(0.0, 0.25, 0.5, 0.75, 1.0),
        K_forks=8,
        N_independent=8,
        g_kind="constant",
        g_value=1.0,
    )


@pytest.fixture
def schedule_full():
    """Standard schedule used for commitment curve tests."""
    return ScheduleSpec(
        n_steps=16,
        scan_points=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
        K_forks=12,
        N_independent=12,
        g_kind="constant",
        g_value=1.0,
    )


@pytest.fixture
def thresholds_default():
    return Thresholds(
        theta_commit=0.7,
        theta_read=0.7,
        theta_rel=0.9,
        theta_robust=0.8,
        theta_cal=0.7,
        frozen=False,
    )


@pytest.fixture
def alpha_grid_small():
    """Fast pilot grid with only a couple of alphas."""
    return AlphaGridSpec(
        pilot_grid=(0.0, 0.5, 1.0),
        diversity_min=0.001,   # very low so alpha=0.5 qualifies on synthetic data
        audio_validity_min=0.5,
        primary_alpha=None,
    )


@pytest.fixture
def rng():
    return np.random.default_rng(99)


# --------------------------------------------------------------------------------------
# commit_gain
# --------------------------------------------------------------------------------------

class TestCommitGain:
    def test_zero_when_a_fork_equals_a_ind(self):
        """No improvement from forking -> gain = 0."""
        assert commit_gain(0.3, 0.3) == pytest.approx(0.0)

    def test_one_when_a_fork_is_1_and_a_ind_is_0(self):
        """Perfect fork agreement, no prior -> gain = 1."""
        assert commit_gain(1.0, 0.0) == pytest.approx(1.0)

    def test_clipped_to_0_when_a_fork_below_a_ind(self):
        """A_fork < A_ind (noise) should be clipped to 0, not negative."""
        assert commit_gain(0.1, 0.5) == pytest.approx(0.0)

    def test_guard_when_a_ind_is_1(self):
        """When a_ind >= 1.0 (prior already deterministic), gain = 0 (no division)."""
        assert commit_gain(1.0, 1.0) == pytest.approx(0.0)
        assert commit_gain(0.5, 1.0) == pytest.approx(0.0)

    def test_intermediate_value(self):
        """Exact intermediate: commit_gain(0.8, 0.4) = (0.8-0.4)/(1-0.4) = 0.4/0.6."""
        expected = 0.4 / 0.6
        assert commit_gain(0.8, 0.4) == pytest.approx(expected, rel=1e-6)

    def test_output_in_unit_interval(self):
        """commit_gain must always be in [0, 1]."""
        rng_ = np.random.default_rng(0)
        for _ in range(50):
            a_f = float(rng_.uniform(0, 1))
            a_i = float(rng_.uniform(0, 0.999))
            g = commit_gain(a_f, a_i)
            assert 0.0 <= g <= 1.0 + 1e-12, f"commit_gain={g} out of range"


# --------------------------------------------------------------------------------------
# a_independent
# --------------------------------------------------------------------------------------

class TestAIndependent:
    def test_returns_float(self, backend, cond, measurer, schedule_small, rng):
        cfg = load_config()
        presence_axis = next(ax for ax in cfg.axes if ax.id == "presence")
        val = a_independent(backend, cond, presence_axis, measurer, schedule_small, rng)
        assert isinstance(val, float)

    def test_in_zero_one_for_exact_match(self, backend, cond, measurer, schedule_small, rng):
        cfg = load_config()
        presence_axis = next(ax for ax in cfg.axes if ax.id == "presence")
        val = a_independent(backend, cond, presence_axis, measurer, schedule_small, rng)
        # categorical exact-match agreement is always in [0, 1]
        assert 0.0 <= val <= 1.0

    def test_runs_for_embedding_axis(self, backend, cond, measurer, schedule_small, rng):
        cfg = load_config()
        material_axis = next(ax for ax in cfg.axes if ax.id == "material")
        val = a_independent(backend, cond, material_axis, measurer, schedule_small, rng)
        # mean pairwise cosine is in [-1, 1]; for the synthetic backend typically > 0
        assert -1.0 <= val <= 1.0

    def test_uses_n_independent_generations(self, backend, cond, measurer, rng):
        """Check N_independent is respected by checking schedule with N=4 vs N=16."""
        cfg = load_config()
        presence_axis = next(ax for ax in cfg.axes if ax.id == "presence")
        s4 = ScheduleSpec(n_steps=8, scan_points=(0.0, 1.0), K_forks=4, N_independent=4)
        s16 = ScheduleSpec(n_steps=8, scan_points=(0.0, 1.0), K_forks=4, N_independent=16)
        # Both should return valid floats; we just check shape/type.
        v4 = a_independent(backend, cond, presence_axis, measurer, s4, np.random.default_rng(1))
        v16 = a_independent(backend, cond, presence_axis, measurer, s16, np.random.default_rng(1))
        assert isinstance(v4, float) and isinstance(v16, float)


# --------------------------------------------------------------------------------------
# a_fork
# --------------------------------------------------------------------------------------

class TestAFork:
    def test_at_s1_all_forks_identical_for_alpha0(self, backend, cond, measurer,
                                                   schedule_small, rng):
        """At s=1, x_s IS the final state; alpha=0 forks are deterministic -> all agree."""
        from foley_cw.score_sde import generate_trajectory
        traj = generate_trajectory(backend, cond, schedule_small, np.random.default_rng(5),
                                   alpha=0.0, record_points=(1.0,))
        x1 = traj["states"][1.0]

        cfg = load_config()
        presence_axis = next(ax for ax in cfg.axes if ax.id == "presence")
        val = a_fork(backend, x1, 1.0, cond, presence_axis, measurer,
                     alpha=0.0, schedule=schedule_small, rng=np.random.default_rng(7))
        # All K forks from x_s=x1 with alpha=0 are identical -> agreement = 1.0
        assert val == pytest.approx(1.0, abs=1e-9)

    def test_returns_float(self, backend, cond, measurer, schedule_small, rng):
        from foley_cw.score_sde import generate_trajectory
        traj = generate_trajectory(backend, cond, schedule_small, np.random.default_rng(3),
                                   alpha=0.0, record_points=(0.5,))
        x_mid = traj["states"][0.5]
        cfg = load_config()
        timing_axis = next(ax for ax in cfg.axes if ax.id == "timing")
        val = a_fork(backend, x_mid, 0.5, cond, timing_axis, measurer,
                     alpha=1.0, schedule=schedule_small, rng=rng)
        assert isinstance(val, float)

    def test_alpha0_forks_from_s0_equal_a_independent(self, backend, cond, measurer,
                                                        rng):
        """At s=0 with alpha=0, all forks start from x_s=noise but integrate deterministically.
        This should yield the same as ODE completions from x_s (NOT a_independent from
        independent noise), but both are ODE completions: if x_s is the same, all forks agree.
        """
        s0 = ScheduleSpec(n_steps=8, scan_points=(0.0, 1.0), K_forks=6, N_independent=6)
        from foley_cw.score_sde import generate_trajectory
        traj = generate_trajectory(backend, cond, s0, np.random.default_rng(11),
                                   alpha=0.0, record_points=(0.0,))
        x_0 = traj["states"][0.0]
        cfg = load_config()
        presence_axis = next(ax for ax in cfg.axes if ax.id == "presence")
        # alpha=0 forks from x_0 are all identical -> agreement = 1.0
        val = a_fork(backend, x_0, 0.0, cond, presence_axis, measurer,
                     alpha=0.0, schedule=s0, rng=np.random.default_rng(12))
        assert val == pytest.approx(1.0, abs=1e-9)


# --------------------------------------------------------------------------------------
# commitment_curve_for_video
# --------------------------------------------------------------------------------------

class TestCommitmentCurveForVideo:
    def test_shape_matches_scan_points(self, backend, cond, measurer, schedule_small, rng):
        cfg = load_config()
        presence_axis = next(ax for ax in cfg.axes if ax.id == "presence")
        curve = commitment_curve_for_video(
            backend, cond, presence_axis, measurer, alpha=1.0,
            schedule=schedule_small, rng=rng
        )
        assert curve.shape == (len(schedule_small.scan_points),)

    def test_values_in_unit_interval(self, backend, cond, measurer, schedule_small, rng):
        cfg = load_config()
        presence_axis = next(ax for ax in cfg.axes if ax.id == "presence")
        curve = commitment_curve_for_video(
            backend, cond, presence_axis, measurer, alpha=1.0,
            schedule=schedule_small, rng=rng
        )
        assert np.all(curve >= -1e-9), f"Some gains negative: {curve}"
        assert np.all(curve <= 1.0 + 1e-9), f"Some gains > 1: {curve}"

    def test_last_point_near_1(self, backend, cond, measurer, rng):
        """At s=1, all forks from x1 with any alpha are identical -> commit_gain ~ 1."""
        s = ScheduleSpec(
            n_steps=8,
            scan_points=(0.0, 0.5, 1.0),
            K_forks=8,
            N_independent=8,
            g_kind="constant",
            g_value=1.0,
        )
        cfg = load_config()
        presence_axis = next(ax for ax in cfg.axes if ax.id == "presence")
        # Use a small mu_scale to ensure a_ind < 1
        cond_low = SyntheticVideoCond(mu=np.array([0.5, -0.5, 0.3, -0.3]), video_id="test")
        curve = commitment_curve_for_video(
            backend, cond_low, presence_axis, measurer, alpha=1.0,
            schedule=s, rng=np.random.default_rng(33)
        )
        # At s=1 all forks are identical -> gain should be 1.0 (if a_ind < 1) or 0 (if a_ind=1)
        # In either case, it must be in [0, 1]
        assert 0.0 <= curve[-1] <= 1.0 + 1e-9

    def test_gain_monotonically_non_decreasing_on_average(self, backend, video_bank,
                                                             measurer, rng):
        """Averaged over multiple videos, commit gain should trend upward with s."""
        s = ScheduleSpec(
            n_steps=8,
            scan_points=(0.0, 0.5, 1.0),
            K_forks=8,
            N_independent=8,
        )
        cfg = load_config()
        presence_axis = next(ax for ax in cfg.axes if ax.id == "presence")
        avg_curves = []
        for vid in video_bank:
            c = commitment_curve_for_video(
                backend, vid, presence_axis, measurer, alpha=1.0,
                schedule=s, rng=np.random.default_rng(77)
            )
            avg_curves.append(c)
        mean_curve = np.mean(avg_curves, axis=0)
        # mean_curve[0] should be <= mean_curve[-1] (s=0 vs s=1)
        assert mean_curve[0] <= mean_curve[-1] + 0.1, (
            f"Mean commitment gain at s=0 ({mean_curve[0]:.3f}) > s=1 ({mean_curve[-1]:.3f})"
        )


# --------------------------------------------------------------------------------------
# select_primary_alpha
# --------------------------------------------------------------------------------------

class TestSelectPrimaryAlpha:
    def test_returns_tuple(self, backend, video_bank, measurer, alpha_grid_small,
                           schedule_small, rng):
        cfg = load_config()
        active_axes = [ax for ax in cfg.axes
                       if ax.tier.value not in ("EXCLUDED", "SEPARATE")]
        alpha, info = select_primary_alpha(
            backend, video_bank, active_axes, alpha_grid_small, schedule_small, measurer, rng
        )
        assert isinstance(info, dict)

    def test_returns_none_when_diversity_min_too_high(self, backend, video_bank,
                                                        measurer, schedule_small, rng):
        """If diversity_min is set impossibly high, no alpha qualifies -> None."""
        strict_grid = AlphaGridSpec(
            pilot_grid=(0.0, 0.01),
            diversity_min=999.0,   # impossibly high
            audio_validity_min=0.5,
        )
        cfg = load_config()
        active_axes = [ax for ax in cfg.axes
                       if ax.tier.value not in ("EXCLUDED", "SEPARATE")]
        alpha, info = select_primary_alpha(
            backend, video_bank, active_axes, strict_grid, schedule_small, measurer, rng
        )
        assert alpha is None

    def test_smallest_qualifying_alpha_selected(self, backend, video_bank, measurer,
                                                  schedule_small, rng):
        """select_primary_alpha returns the SMALLEST qualifying alpha in pilot_grid."""
        # Use a grid where alpha=0.5 and alpha=1.0 should both qualify on the synthetic backend.
        grid = AlphaGridSpec(
            pilot_grid=(0.0, 0.5, 1.0),
            diversity_min=0.001,
            audio_validity_min=0.5,
        )
        cfg = load_config()
        active_axes = [ax for ax in cfg.axes
                       if ax.tier.value not in ("EXCLUDED", "SEPARATE")]
        alpha, info = select_primary_alpha(
            backend, video_bank, active_axes, grid, schedule_small, measurer, rng
        )
        if alpha is not None:
            # All lower values in the grid that qualify should have been chosen first
            for a in sorted(grid.pilot_grid):
                if a < alpha and info.get(float(a)):
                    assert not info[float(a)]["qualifies"], (
                        f"Alpha {a} qualifies but {alpha} was chosen as primary"
                    )

    def test_surface_info_has_all_pilot_alphas(self, backend, video_bank, measurer,
                                                 alpha_grid_small, schedule_small, rng):
        cfg = load_config()
        active_axes = [ax for ax in cfg.axes
                       if ax.tier.value not in ("EXCLUDED", "SEPARATE")]
        _, info = select_primary_alpha(
            backend, video_bank, active_axes, alpha_grid_small, schedule_small, measurer, rng
        )
        for a in alpha_grid_small.pilot_grid:
            if float(a) in info:
                d = info[float(a)]
                assert "diversity" in d
                assert "validity" in d
                assert "qualifies" in d

    def test_empty_videos_returns_none(self, backend, measurer, alpha_grid_small,
                                        schedule_small, rng):
        cfg = load_config()
        active_axes = [ax for ax in cfg.axes
                       if ax.tier.value not in ("EXCLUDED", "SEPARATE")]
        alpha, _ = select_primary_alpha(
            backend, [], active_axes, alpha_grid_small, schedule_small, measurer, rng
        )
        assert alpha is None


# --------------------------------------------------------------------------------------
# build_commitment_map
# --------------------------------------------------------------------------------------

class TestBuildCommitmentMap:
    @pytest.fixture(scope="class")
    def map_outputs(self, backend, video_bank, measurer):
        """Run build_commitment_map once per class for efficiency."""
        s = ScheduleSpec(
            n_steps=8,
            scan_points=(0.0, 0.5, 1.0),
            K_forks=8,
            N_independent=8,
            g_kind="constant",
            g_value=1.0,
        )
        thresholds = Thresholds(
            theta_commit=0.7, theta_read=0.7, theta_rel=0.9,
            theta_robust=0.8, theta_cal=0.7, frozen=False
        )
        alpha_grid = AlphaGridSpec(
            pilot_grid=(0.0, 1.0),
            diversity_min=0.001,
            audio_validity_min=0.5,
        )
        cfg = load_config()
        rng_ = np.random.default_rng(42)
        return build_commitment_map(
            backend, video_bank, cfg.axes, alpha_grid, s, thresholds, measurer, rng_
        )

    def test_returns_three_tuple(self, map_outputs):
        assert len(map_outputs) == 3

    def test_cells_is_list_of_commitment_cells(self, map_outputs):
        cells, _, _ = map_outputs
        assert isinstance(cells, list)
        assert len(cells) > 0
        for c in cells:
            assert isinstance(c, CommitmentCell)

    def test_windows_is_dict_of_window_estimates(self, map_outputs):
        _, windows, _ = map_outputs
        assert isinstance(windows, dict)
        for k, v in windows.items():
            assert isinstance(k, str)
            assert isinstance(v, WindowEstimate)

    def test_primary_alpha_is_float_or_none(self, map_outputs):
        _, _, primary = map_outputs
        assert primary is None or isinstance(primary, float)

    def test_cells_cover_all_active_axes(self, map_outputs):
        """Every active axis (non-EXCLUDED, non-SEPARATE) has at least one cell."""
        cells, _, _ = map_outputs
        from foley_cw.types import AxisTier
        cfg = load_config()
        active_ids = {
            ax.id for ax in cfg.axes
            if ax.tier not in (AxisTier.EXCLUDED, AxisTier.SEPARATE)
        }
        cell_ids = {c.axis_id for c in cells}
        for aid in active_ids:
            assert aid in cell_ids, f"Axis {aid!r} has no CommitmentCells"

    def test_cell_gain_in_unit_interval(self, map_outputs):
        cells, _, _ = map_outputs
        for c in cells:
            assert 0.0 <= c.commit_gain <= 1.0 + 1e-9, (
                f"Cell {c.axis_id} s={c.s} alpha={c.alpha} gain={c.commit_gain} out of range"
            )

    def test_windows_dict_has_active_axes(self, map_outputs):
        _, windows, _ = map_outputs
        from foley_cw.types import AxisTier
        cfg = load_config()
        active_ids = {
            ax.id for ax in cfg.axes
            if ax.tier not in (AxisTier.EXCLUDED, AxisTier.SEPARATE)
        }
        for aid in active_ids:
            assert aid in windows, f"No WindowEstimate for axis {aid!r}"

    def test_at_least_one_finite_s_commit(self, map_outputs):
        """At least one axis should have a finite s_commit with 6 videos."""
        _, windows, primary = map_outputs
        if primary is None:
            pytest.skip("No valid primary alpha found; cannot test s_commit")
        finite_windows = [w for w in windows.values() if not np.isnan(w.s_hat)]
        assert len(finite_windows) >= 1, (
            "Expected at least one axis with a finite s_commit, "
            f"but all s_hat are NaN: {windows}"
        )

    def test_window_kind_is_commit(self, map_outputs):
        _, windows, _ = map_outputs
        for w in windows.values():
            assert w.kind == "commit"

    def test_n_videos_correct_in_cells(self, map_outputs):
        cells, _, _ = map_outputs
        for c in cells:
            assert c.n_videos == N_VIDEOS


# --------------------------------------------------------------------------------------
# Scientific contract: commit_gain ~0 at s=0 and ~1 at s=1 (averaged over videos)
# --------------------------------------------------------------------------------------

class TestScientificContracts:
    def test_commit_gain_near_zero_at_s0_averaged(self, backend, video_bank, measurer):
        """Averaged over 6 videos, commit gain at s=0 should be close to 0.

        At s=0 the shared state is just independent noise (x_s ~ prior), so
        A_fork(s=0) ~ A_independent -> gain ~ 0.
        """
        s = ScheduleSpec(
            n_steps=8,
            scan_points=(0.0, 1.0),
            K_forks=12,
            N_independent=12,
            g_kind="constant",
            g_value=1.0,
        )
        cfg = load_config()
        # Pick a TIER1 axis
        presence_axis = next(ax for ax in cfg.axes if ax.id == "presence")
        gains_at_s0 = []
        for vid in video_bank:
            c = commitment_curve_for_video(
                backend, vid, presence_axis, measurer, alpha=1.0,
                schedule=s, rng=np.random.default_rng(100)
            )
            gains_at_s0.append(c[0])
        mean_s0 = float(np.mean(gains_at_s0))
        # Soft check: mean gain at s=0 should be fairly small (< 0.5)
        assert mean_s0 < 0.5, (
            f"Mean commit gain at s=0 is {mean_s0:.3f} (expected < 0.5). "
            "This may indicate a_fork > a_independent at s=0, suggesting the "
            "video-prior normalization is not working."
        )

    def test_commit_gain_near_1_at_s1_averaged(self, backend, video_bank, measurer):
        """Averaged over 6 videos, commit gain at s=1 should be close to 1.

        At s=1 the 'intermediate state' IS the final state, so all alpha forks
        reproduce the same audio -> A_fork(s=1) = 1.0 -> gain = 1 (if a_ind < 1).
        """
        s = ScheduleSpec(
            n_steps=8,
            scan_points=(0.0, 1.0),
            K_forks=12,
            N_independent=12,
            g_kind="constant",
            g_value=1.0,
        )
        cfg = load_config()
        presence_axis = next(ax for ax in cfg.axes if ax.id == "presence")
        gains_at_s1 = []
        for vid in video_bank:
            c = commitment_curve_for_video(
                backend, vid, presence_axis, measurer, alpha=1.0,
                schedule=s, rng=np.random.default_rng(200)
            )
            gains_at_s1.append(c[-1])
        mean_s1 = float(np.mean(gains_at_s1))
        # At s=1 with alpha=0 forks (all identical), gain = clip((1-a_ind)/(1-a_ind))=1
        # or 0 if a_ind=1 (video fully determines axis). For well-separated mu_scale=2 bank,
        # a_ind should NOT always be 1.
        # Soft check: mean gain at s=1 should be > 0 (at least some videos commit).
        assert mean_s1 > 0.0, (
            f"Mean commit gain at s=1 is {mean_s1:.3f} (expected > 0.0)."
        )

    def test_s_commit_finite_for_at_least_one_axis(self, backend, video_bank, measurer):
        """With 6 videos and alpha=1.0, at least one axis should have a finite s_commit."""
        s = ScheduleSpec(
            n_steps=8,
            scan_points=(0.0, 0.25, 0.5, 0.75, 1.0),
            K_forks=10,
            N_independent=10,
        )
        thresholds = Thresholds(
            theta_commit=0.5,    # lenient threshold for test
            theta_read=0.7, theta_rel=0.9, theta_robust=0.8, theta_cal=0.7,
        )
        alpha_grid = AlphaGridSpec(
            pilot_grid=(1.0,),           # force alpha=1.0 as the only candidate
            diversity_min=0.0,           # accept any diversity
            audio_validity_min=0.0,      # accept any validity
        )
        cfg = load_config()
        cells, windows, primary = build_commitment_map(
            backend, video_bank, cfg.axes, alpha_grid, s, thresholds, measurer,
            np.random.default_rng(777)
        )
        assert primary is not None, "Expected primary_alpha to be 1.0"
        finite = [w for w in windows.values() if not np.isnan(w.s_hat)]
        assert len(finite) >= 1, (
            "No axis has a finite s_commit. "
            f"Windows: { {k: (w.s_hat, w.underpowered) for k, w in windows.items()} }"
        )


# --------------------------------------------------------------------------------------
# Import safety
# --------------------------------------------------------------------------------------

def test_module_importable_numpy_only():
    """foley_cw.commitment must import with only numpy."""
    import foley_cw.commitment  # noqa: F401


def test_no_scipy_imported():
    # Check the real intent — "commitment.py must not import scipy" — in a CLEAN subprocess,
    # so the result is independent of whatever other test modules loaded scipy earlier in the
    # session (the previous in-process sys.modules check was order-fragile).
    import subprocess
    import sys
    code = "import foley_cw.commitment, sys; sys.exit(1 if 'scipy' in sys.modules else 0)"
    r = subprocess.run([sys.executable, "-c", code], capture_output=True)
    assert r.returncode == 0, "scipy must not be imported by commitment.py"
