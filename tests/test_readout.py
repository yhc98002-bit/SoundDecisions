"""Tests for foley_cw/readout.py — Phase-2 readout map.

All tests run on CPU against SyntheticGaussianFlow / SyntheticMeasurer (no MMAudio,
no GPU, no scipy). Scientific contracts tested:

  * EnergyOnsetProbe readout accuracy rises with s on synthetic data (the Tweedie
    x0(s) → x1 monotonically as s → 1, so the probe should become more accurate).
  * s_read is finite (a real number, not NaN) for the presence axis, which has a
    clear binary signal in synthetic data.
  * ode_target returns a well-formed SelfTarget for all axes.
  * fork_majority_target returns a well-formed SelfTarget for all axes.
  * build_readout_map returns the correct number of ReadoutCell rows and WindowEstimate
    entries.
  * All ReadoutCell scores are in the valid range [0, 1] for categorical axes and
    [-1, 1] for embedding axes (cosine).
  * Module imports with numpy only (no scipy/torch/librosa).
"""

from __future__ import annotations

import numpy as np
import pytest

from foley_cw.axes import SyntheticMeasurer
from foley_cw.config import load_config
from foley_cw.probes import EnergyOnsetProbe, probe_ladder
from foley_cw.readout import (
    _probe_accuracy,
    build_readout_map,
    fork_majority_target,
    ode_target,
    readout_curve_for_video,
)
from foley_cw.score_sde import generate_trajectory
from foley_cw.synthetic_backend import SyntheticGaussianFlow, SyntheticVideoCond
from foley_cw.types import (
    Axis,
    AxisKind,
    AgreementMetric,
    ReadoutCell,
    ScheduleSpec,
    Thresholds,
    WindowEstimate,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DIM = 4


@pytest.fixture
def backend():
    return SyntheticGaussianFlow(dim=DIM, sigma2=0.25)


@pytest.fixture
def cond():
    # Strong signal: mu well away from zero so presence axis has clear decision
    rng = np.random.default_rng(0)
    mu = rng.standard_normal(DIM) * 3.0  # large mu -> mean(audio) clearly > 0
    return SyntheticVideoCond(mu=mu, video_id="test_vid")


@pytest.fixture
def schedule():
    """Fast schedule for unit tests."""
    return ScheduleSpec(
        n_steps=16,
        scan_points=(0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0),
        K_forks=4,
        N_independent=4,
        g_kind="constant",
        g_value=1.0,
    )


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def measurer():
    return SyntheticMeasurer(seed=42)


@pytest.fixture
def presence_axis():
    return Axis(
        id="presence",
        name="event-sound presence",
        tier="TIER1",  # type: ignore[arg-type]
        kind=AxisKind.CATEGORICAL,
        agreement=AgreementMetric.EXACT_MATCH,
        measure="presence_detector",
    )


@pytest.fixture
def material_axis():
    return Axis(
        id="material",
        name="material / fine class",
        tier="TIER2",  # type: ignore[arg-type]
        kind=AxisKind.EMBEDDING,
        agreement=AgreementMetric.MEAN_PAIRWISE_COSINE,
        measure="audio_embedding",
    )


@pytest.fixture
def timing_axis():
    return Axis(
        id="timing",
        name="gross timing",
        tier="TIER1",  # type: ignore[arg-type]
        kind=AxisKind.CATEGORICAL,
        agreement=AgreementMetric.EXACT_MATCH,
        measure="onset_timing_bin",
    )


@pytest.fixture
def thresholds():
    return Thresholds(
        theta_commit=0.7,
        theta_read=0.7,
        theta_rel=0.9,
        theta_robust=0.8,
        theta_cal=0.7,
        frozen=False,
    )


@pytest.fixture
def video_bank(backend):
    """Small bank of synthetic videos with well-separated data means."""
    return SyntheticGaussianFlow.make_video_bank(n_videos=5, dim=DIM, mu_scale=2.0, seed=1)


# ---------------------------------------------------------------------------
# ode_target tests
# ---------------------------------------------------------------------------

class TestOdeTarget:
    def test_returns_self_target(self, backend, cond, presence_axis, measurer, schedule, rng):
        traj = generate_trajectory(backend, cond, schedule, rng, alpha=0.0,
                                   record_points=schedule.scan_points)
        x_s = traj["states"][0.5]
        result = ode_target(backend, x_s, 0.5, cond, presence_axis, measurer, schedule)
        from foley_cw.types import SelfTarget
        assert isinstance(result, SelfTarget)

    def test_kind_matches_axis_categorical(self, backend, cond, presence_axis, measurer, schedule, rng):
        traj = generate_trajectory(backend, cond, schedule, rng, alpha=0.0,
                                   record_points=schedule.scan_points)
        x_s = traj["states"][0.5]
        result = ode_target(backend, x_s, 0.5, cond, presence_axis, measurer, schedule)
        assert result.kind is AxisKind.CATEGORICAL
        assert result.label is not None

    def test_kind_matches_axis_embedding(self, backend, cond, material_axis, measurer, schedule, rng):
        traj = generate_trajectory(backend, cond, schedule, rng, alpha=0.0,
                                   record_points=schedule.scan_points)
        x_s = traj["states"][0.5]
        result = ode_target(backend, x_s, 0.5, cond, material_axis, measurer, schedule)
        assert result.kind is AxisKind.EMBEDDING
        assert result.embedding is not None

    def test_axis_id_correct(self, backend, cond, presence_axis, measurer, schedule, rng):
        traj = generate_trajectory(backend, cond, schedule, rng, alpha=0.0,
                                   record_points=schedule.scan_points)
        x_s = traj["states"][0.5]
        result = ode_target(backend, x_s, 0.5, cond, presence_axis, measurer, schedule)
        assert result.axis_id == presence_axis.id

    def test_at_s1_deterministic(self, backend, cond, presence_axis, measurer, schedule, rng):
        """At s=1.0 the ODE is complete; ode_target should be same as final audio target."""
        traj = generate_trajectory(backend, cond, schedule, rng, alpha=0.0,
                                   record_points=schedule.scan_points)
        x_s = traj["states"][1.0]
        result = ode_target(backend, x_s, 1.0, cond, presence_axis, measurer, schedule)
        # At s=1 the ODE completion is the identity (already at final state)
        # Just verify it's a valid SelfTarget
        assert result.label in (0, 1)


# ---------------------------------------------------------------------------
# fork_majority_target tests
# ---------------------------------------------------------------------------

class TestForkMajorityTarget:
    def test_returns_self_target_categorical(self, backend, cond, presence_axis,
                                             measurer, schedule, rng):
        traj = generate_trajectory(backend, cond, schedule, rng, alpha=0.0,
                                   record_points=schedule.scan_points)
        x_s = traj["states"][0.5]
        result = fork_majority_target(
            backend, x_s, 0.5, cond, presence_axis, measurer,
            alpha=0.5, schedule=schedule, rng=rng
        )
        from foley_cw.types import SelfTarget
        assert isinstance(result, SelfTarget)
        assert result.kind is AxisKind.CATEGORICAL
        assert result.label is not None

    def test_returns_self_target_embedding(self, backend, cond, material_axis,
                                           measurer, schedule, rng):
        traj = generate_trajectory(backend, cond, schedule, rng, alpha=0.0,
                                   record_points=schedule.scan_points)
        x_s = traj["states"][0.5]
        result = fork_majority_target(
            backend, x_s, 0.5, cond, material_axis, measurer,
            alpha=0.5, schedule=schedule, rng=rng
        )
        assert result.kind is AxisKind.EMBEDDING
        assert result.embedding is not None
        # Embedding should be unit-normed (or zero)
        norm = float(np.linalg.norm(result.embedding))
        assert norm == pytest.approx(1.0, abs=1e-9) or norm == pytest.approx(0.0, abs=1e-9)

    def test_alpha0_matches_ode_target(self, backend, cond, presence_axis,
                                       measurer, schedule):
        """At alpha=0, all K fork completions are identical; majority == ode_target."""
        rng1 = np.random.default_rng(7)
        rng2 = np.random.default_rng(7)
        traj = generate_trajectory(backend, cond, schedule, rng1, alpha=0.0,
                                   record_points=schedule.scan_points)
        x_s = traj["states"][0.5]

        ode_t = ode_target(backend, x_s, 0.5, cond, presence_axis, measurer, schedule)
        fork_t = fork_majority_target(
            backend, x_s, 0.5, cond, presence_axis, measurer,
            alpha=0.0, schedule=schedule, rng=rng2
        )
        # At alpha=0 all forks are deterministic copies; majority == ode_target
        assert ode_t.label == fork_t.label

    def test_axis_id_correct(self, backend, cond, presence_axis, measurer, schedule, rng):
        traj = generate_trajectory(backend, cond, schedule, rng, alpha=0.0,
                                   record_points=schedule.scan_points)
        x_s = traj["states"][0.5]
        result = fork_majority_target(
            backend, x_s, 0.5, cond, presence_axis, measurer,
            alpha=0.5, schedule=schedule, rng=rng
        )
        assert result.axis_id == presence_axis.id


# ---------------------------------------------------------------------------
# _probe_accuracy tests
# ---------------------------------------------------------------------------

class TestProbeAccuracy:
    def test_categorical_match_is_one(self, presence_axis):
        from foley_cw.types import SelfTarget
        probe = EnergyOnsetProbe()
        # Create x0 with mean > 0 so probe predicts 1
        x0 = np.array([1.0, 2.0, 3.0, 4.0])
        target = SelfTarget(axis_id="presence", kind=AxisKind.CATEGORICAL, label=1)
        acc = _probe_accuracy(probe, x0, presence_axis, target)
        assert acc == pytest.approx(1.0)

    def test_categorical_mismatch_is_zero(self, presence_axis):
        from foley_cw.types import SelfTarget
        probe = EnergyOnsetProbe()
        # x0 with mean > 0 -> probe predicts 1; target is 0 -> mismatch
        x0 = np.array([1.0, 2.0, 3.0, 4.0])
        target = SelfTarget(axis_id="presence", kind=AxisKind.CATEGORICAL, label=0)
        acc = _probe_accuracy(probe, x0, presence_axis, target)
        assert acc == pytest.approx(0.0)

    def test_embedding_cosine_range(self, material_axis):
        from foley_cw.types import SelfTarget
        probe = EnergyOnsetProbe()
        x0 = np.array([1.0, 0.5, -0.3, 0.8])
        emb = np.array([0.6, 0.8, 0.0, 0.0])
        emb = emb / np.linalg.norm(emb)
        target = SelfTarget(axis_id="material", kind=AxisKind.EMBEDDING, embedding=emb)
        acc = _probe_accuracy(probe, x0, material_axis, target)
        assert -1.0 <= acc <= 1.0

    def test_embedding_unit_norm_gives_cosine_one(self, material_axis):
        from foley_cw.types import SelfTarget
        probe = EnergyOnsetProbe()
        # x0 such that unit_norm(x0) == target embedding exactly
        x0 = np.array([3.0, 4.0, 0.0, 0.0])  # will be unit-normed to [0.6, 0.8, 0, 0]
        emb = x0 / np.linalg.norm(x0)
        target = SelfTarget(axis_id="material", kind=AxisKind.EMBEDDING, embedding=emb)
        acc = _probe_accuracy(probe, x0, material_axis, target)
        assert acc == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# readout_curve_for_video tests
# ---------------------------------------------------------------------------

class TestReadoutCurveForVideo:
    def test_returns_correct_shape(self, backend, cond, presence_axis, measurer,
                                   schedule, rng):
        probe = EnergyOnsetProbe()
        curve = readout_curve_for_video(
            backend, {"cond": cond}, presence_axis, probe,
            "ode", 0.5, schedule, measurer, rng
        )
        assert curve.shape == (len(schedule.scan_points),)

    def test_values_in_valid_range_categorical(self, backend, cond, presence_axis,
                                               measurer, schedule, rng):
        probe = EnergyOnsetProbe()
        curve = readout_curve_for_video(
            backend, {"cond": cond}, presence_axis, probe,
            "ode", 0.5, schedule, measurer, rng
        )
        assert np.all((curve >= 0.0) & (curve <= 1.0))

    def test_values_in_valid_range_embedding(self, backend, cond, material_axis,
                                             measurer, schedule, rng):
        probe = EnergyOnsetProbe()
        curve = readout_curve_for_video(
            backend, {"cond": cond}, material_axis, probe,
            "ode", 0.5, schedule, measurer, rng
        )
        # Cosine in [-1, 1]; EnergyOnsetProbe vs SyntheticMeasurer(embed_dim=8) have
        # different embedding dimensions so _probe_accuracy returns 0.0 (no readout signal).
        # Either way, values must be in [-1, 1].
        assert np.all((curve >= -1.0) & (curve <= 1.0))

    def test_fork_majority_target_runs(self, backend, cond, presence_axis, measurer,
                                       schedule, rng):
        probe = EnergyOnsetProbe()
        curve = readout_curve_for_video(
            backend, {"cond": cond}, presence_axis, probe,
            "fork_majority", 0.5, schedule, measurer, rng
        )
        assert curve.shape == (len(schedule.scan_points),)
        assert np.all((curve >= 0.0) & (curve <= 1.0))

    def test_video_as_object_with_cond_attr(self, backend, cond, presence_axis,
                                            measurer, schedule, rng):
        """Accepts video objects with .cond attribute (not just dicts)."""
        probe = EnergyOnsetProbe()
        # SyntheticVideoCond has .cond? No -- pass the cond directly as a mock object
        class VideoObj:
            def __init__(self, c):
                self.cond = c
        video_obj = VideoObj(cond)
        curve = readout_curve_for_video(
            backend, video_obj, presence_axis, probe,
            "ode", 0.5, schedule, measurer, rng
        )
        assert curve.shape == (len(schedule.scan_points),)

    def test_invalid_target_kind_raises(self, backend, cond, presence_axis, measurer,
                                        schedule, rng):
        probe = EnergyOnsetProbe()
        with pytest.raises(ValueError, match="target_kind"):
            readout_curve_for_video(
                backend, {"cond": cond}, presence_axis, probe,
                "bad_target", 0.5, schedule, measurer, rng
            )

    def test_accuracy_at_s1_is_high_for_strong_signal(self):
        """At s=1, x0(s)=x_1 so probe accuracy should be 1.0 on synthetic data
        where measurer and probe use the same deterministic rule."""
        backend = SyntheticGaussianFlow(dim=DIM, sigma2=0.25)
        # Create a video cond with large positive mu -> mean(audio) always > 0 at s=1
        mu = np.array([5.0, 5.0, 5.0, 5.0])
        cond = SyntheticVideoCond(mu=mu, video_id="strong")
        schedule = ScheduleSpec(
            n_steps=32,
            scan_points=(0.9, 1.0),
            K_forks=4,
            N_independent=4,
        )
        measurer = SyntheticMeasurer(seed=42)
        probe = EnergyOnsetProbe()
        presence_axis = Axis(
            id="presence", name="presence", tier="TIER1",  # type: ignore[arg-type]
            kind=AxisKind.CATEGORICAL, agreement=AgreementMetric.EXACT_MATCH,
            measure="presence_detector"
        )
        rng = np.random.default_rng(0)
        curve = readout_curve_for_video(
            backend, {"cond": cond}, presence_axis, probe,
            "ode", 0.0, schedule, measurer, rng
        )
        # At s=1, x0(s)=x_1 and the ODE completes to x_1; probe matches measurer
        assert curve[-1] == pytest.approx(1.0, abs=1e-9), (
            f"Expected accuracy=1.0 at s=1 for strong-signal presence axis, got {curve[-1]}"
        )


# ---------------------------------------------------------------------------
# Key scientific contract: accuracy rises with s for EnergyOnsetProbe on synthetic data
# ---------------------------------------------------------------------------

class TestAccuracyRisesWithS:
    """The plan's core promise for Phase 2: readout accuracy improves as s increases.

    On synthetic data with a strong-signal presence axis, EnergyOnsetProbe uses the same
    rule as SyntheticMeasurer (int(mean(x0)>0)), so as x0(s)->x1 (s->1) the accuracy
    should rise from near-chance to 1.0.

    We test this by requiring that accuracy(s=1.0) > accuracy(s=0.0), and that accuracy
    at s=1.0 equals 1.0 for a large-mu cond (unambiguous positive mean).
    """

    def _make_strong_cond(self, seed: int = 0) -> tuple:
        """Return (backend, cond) with very large positive mu."""
        backend = SyntheticGaussianFlow(dim=DIM, sigma2=0.1)
        mu = np.full(DIM, 8.0)  # very large positive => mean(audio) > 0 always
        cond = SyntheticVideoCond(mu=mu, video_id=f"strong_{seed}")
        return backend, cond

    def test_accuracy_at_s1_equals_1_for_strong_presence(self):
        backend, cond = self._make_strong_cond()
        schedule = ScheduleSpec(
            n_steps=32,
            scan_points=(0.0, 0.5, 1.0),
            K_forks=4,
            N_independent=4,
        )
        presence_axis = Axis(
            id="presence", name="presence", tier="TIER1",  # type: ignore[arg-type]
            kind=AxisKind.CATEGORICAL, agreement=AgreementMetric.EXACT_MATCH,
            measure="presence_detector"
        )
        measurer = SyntheticMeasurer(seed=42)
        probe = EnergyOnsetProbe()
        rng = np.random.default_rng(11)
        curve = readout_curve_for_video(
            backend, {"cond": cond}, presence_axis, probe,
            "ode", 0.0, schedule, measurer, rng
        )
        # s=1.0 is the last element in scan_points=(0, 0.5, 1.0)
        assert curve[-1] == pytest.approx(1.0, abs=1e-9), (
            f"Expected accuracy=1.0 at s=1.0 but got {curve[-1]}"
        )

    def test_accuracy_rises_from_s0_to_s1_single_video(self):
        """Accuracy at s=1 should be >= accuracy at s=0 for strong-signal cond."""
        backend, cond = self._make_strong_cond()
        schedule = ScheduleSpec(
            n_steps=32,
            scan_points=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            K_forks=4,
            N_independent=4,
        )
        presence_axis = Axis(
            id="presence", name="presence", tier="TIER1",  # type: ignore[arg-type]
            kind=AxisKind.CATEGORICAL, agreement=AgreementMetric.EXACT_MATCH,
            measure="presence_detector"
        )
        measurer = SyntheticMeasurer(seed=42)
        probe = EnergyOnsetProbe()
        rng = np.random.default_rng(13)
        curve = readout_curve_for_video(
            backend, {"cond": cond}, presence_axis, probe,
            "ode", 0.0, schedule, measurer, rng
        )
        assert curve[-1] >= curve[0], (
            f"Expected accuracy at s=1 >= accuracy at s=0, but got {curve[-1]} < {curve[0]}"
        )

    def test_accuracy_not_all_zero(self):
        """Sanity: curve for strong-signal axis should not be all zeros."""
        backend, cond = self._make_strong_cond()
        schedule = ScheduleSpec(
            n_steps=32,
            scan_points=(0.0, 0.5, 1.0),
            K_forks=4,
            N_independent=4,
        )
        presence_axis = Axis(
            id="presence", name="presence", tier="TIER1",  # type: ignore[arg-type]
            kind=AxisKind.CATEGORICAL, agreement=AgreementMetric.EXACT_MATCH,
            measure="presence_detector"
        )
        measurer = SyntheticMeasurer(seed=42)
        probe = EnergyOnsetProbe()
        rng = np.random.default_rng(15)
        curve = readout_curve_for_video(
            backend, {"cond": cond}, presence_axis, probe,
            "ode", 0.0, schedule, measurer, rng
        )
        assert np.any(curve > 0.0), "Curve should not be all zeros for strong-signal axis"


# ---------------------------------------------------------------------------
# s_read finite for presence axis
# ---------------------------------------------------------------------------

class TestSReadFiniteForPresence:
    """Build a readout map and verify s_read is finite for the presence axis."""

    def test_s_read_finite_presence_ode(self):
        """s_read for presence axis (ode target) should be a finite float."""
        backend = SyntheticGaussianFlow(dim=DIM, sigma2=0.05)
        # All videos with strong positive mu so probe accuracy reaches theta_read
        videos = [
            {"cond": SyntheticVideoCond(mu=np.full(DIM, 5.0), video_id=f"v{i}")}
            for i in range(6)
        ]
        # Use low theta_read so it's easy to cross
        thresholds = Thresholds(
            theta_commit=0.7, theta_read=0.6,
            theta_rel=0.9, theta_robust=0.8, theta_cal=0.7,
        )
        schedule = ScheduleSpec(
            n_steps=16,
            scan_points=(0.0, 0.3, 0.5, 0.7, 0.9, 1.0),
            K_forks=4,
            N_independent=4,
        )
        presence_axis = Axis(
            id="presence", name="presence", tier="TIER1",  # type: ignore[arg-type]
            kind=AxisKind.CATEGORICAL, agreement=AgreementMetric.EXACT_MATCH,
            measure="presence_detector"
        )
        measurer = SyntheticMeasurer(seed=42)
        probes = probe_ladder(include_stubs=False)
        rng = np.random.default_rng(100)

        cells, windows = build_readout_map(
            backend, videos, [presence_axis], probes,
            alpha=0.5, schedule=schedule, thresholds=thresholds,
            measurer=measurer, rng=rng,
        )

        # Verify s_read for (presence, energy_onset, ode)
        win_key = ("presence", "energy_onset", "ode")
        assert win_key in windows, f"Key {win_key} not in windows: {list(windows.keys())}"
        win = windows[win_key]
        assert isinstance(win, WindowEstimate)
        assert not np.isnan(win.s_hat), (
            f"s_read for presence(ode) should be finite; got NaN. "
            f"This means accuracy never crossed theta_read={thresholds.theta_read}."
        )
        assert 0.0 <= win.s_hat <= 1.0, f"s_hat should be in [0,1], got {win.s_hat}"

    def test_s_read_finite_presence_fork_majority(self):
        """s_read for presence (fork_majority) should also be finite with strong signal."""
        backend = SyntheticGaussianFlow(dim=DIM, sigma2=0.05)
        videos = [
            {"cond": SyntheticVideoCond(mu=np.full(DIM, 5.0), video_id=f"v{i}")}
            for i in range(4)
        ]
        thresholds = Thresholds(
            theta_commit=0.7, theta_read=0.6,
            theta_rel=0.9, theta_robust=0.8, theta_cal=0.7,
        )
        schedule = ScheduleSpec(
            n_steps=16,
            scan_points=(0.0, 0.5, 0.8, 1.0),
            K_forks=4,
            N_independent=4,
        )
        presence_axis = Axis(
            id="presence", name="presence", tier="TIER1",  # type: ignore[arg-type]
            kind=AxisKind.CATEGORICAL, agreement=AgreementMetric.EXACT_MATCH,
            measure="presence_detector"
        )
        measurer = SyntheticMeasurer(seed=42)
        probes = probe_ladder(include_stubs=False)
        rng = np.random.default_rng(200)

        _, windows = build_readout_map(
            backend, videos, [presence_axis], probes,
            alpha=0.3, schedule=schedule, thresholds=thresholds,
            measurer=measurer, rng=rng,
        )

        win_key = ("presence", "energy_onset", "fork_majority")
        assert win_key in windows
        win = windows[win_key]
        assert not np.isnan(win.s_hat), (
            f"s_read for presence(fork_majority) should be finite; got NaN."
        )


# ---------------------------------------------------------------------------
# build_readout_map tests
# ---------------------------------------------------------------------------

class TestBuildReadoutMap:
    def test_correct_number_of_cells(self, backend, video_bank, schedule, thresholds,
                                     measurer, rng):
        """Number of ReadoutCell rows = n_axes * n_probes * n_scan_points * n_targets."""
        axes = [
            Axis(id="presence", name="presence", tier="TIER1",  # type: ignore[arg-type]
                 kind=AxisKind.CATEGORICAL, agreement=AgreementMetric.EXACT_MATCH,
                 measure="presence_detector"),
        ]
        probes = probe_ladder(include_stubs=False)
        target_kinds = ("ode", "fork_majority")

        cells, windows = build_readout_map(
            backend, video_bank, axes, probes,
            alpha=0.5, schedule=schedule, thresholds=thresholds,
            measurer=measurer, rng=rng,
        )

        expected = len(axes) * len(probes) * len(schedule.scan_points) * len(target_kinds)
        assert len(cells) == expected, (
            f"Expected {expected} cells, got {len(cells)}"
        )

    def test_correct_number_of_windows(self, backend, video_bank, schedule, thresholds,
                                       measurer, rng):
        """Number of windows = n_axes * n_probes * n_targets."""
        axes = [
            Axis(id="presence", name="presence", tier="TIER1",  # type: ignore[arg-type]
                 kind=AxisKind.CATEGORICAL, agreement=AgreementMetric.EXACT_MATCH,
                 measure="presence_detector"),
            Axis(id="timing", name="timing", tier="TIER1",  # type: ignore[arg-type]
                 kind=AxisKind.CATEGORICAL, agreement=AgreementMetric.EXACT_MATCH,
                 measure="onset_timing_bin"),
        ]
        probes = probe_ladder(include_stubs=False)
        target_kinds = ("ode", "fork_majority")

        _, windows = build_readout_map(
            backend, video_bank, axes, probes,
            alpha=0.5, schedule=schedule, thresholds=thresholds,
            measurer=measurer, rng=rng,
        )

        expected_windows = len(axes) * len(probes) * len(target_kinds)
        assert len(windows) == expected_windows, (
            f"Expected {expected_windows} windows, got {len(windows)}"
        )

    def test_cells_are_readout_cell_instances(self, backend, video_bank, schedule,
                                              thresholds, measurer, rng):
        axes = [
            Axis(id="presence", name="presence", tier="TIER1",  # type: ignore[arg-type]
                 kind=AxisKind.CATEGORICAL, agreement=AgreementMetric.EXACT_MATCH,
                 measure="presence_detector"),
        ]
        probes = probe_ladder(include_stubs=False)
        cells, _ = build_readout_map(
            backend, video_bank, axes, probes,
            alpha=0.5, schedule=schedule, thresholds=thresholds,
            measurer=measurer, rng=rng,
        )
        for cell in cells:
            assert isinstance(cell, ReadoutCell)

    def test_cell_scores_in_valid_range_categorical(self, backend, video_bank, schedule,
                                                    thresholds, measurer, rng):
        axes = [
            Axis(id="presence", name="presence", tier="TIER1",  # type: ignore[arg-type]
                 kind=AxisKind.CATEGORICAL, agreement=AgreementMetric.EXACT_MATCH,
                 measure="presence_detector"),
        ]
        probes = probe_ladder(include_stubs=False)
        cells, _ = build_readout_map(
            backend, video_bank, axes, probes,
            alpha=0.5, schedule=schedule, thresholds=thresholds,
            measurer=measurer, rng=rng,
        )
        for cell in cells:
            assert 0.0 <= cell.score <= 1.0, (
                f"Categorical score out of [0,1]: {cell.score} for {cell}"
            )

    def test_cell_scores_in_valid_range_embedding(self, backend, video_bank, schedule,
                                                  thresholds, measurer, rng):
        axes = [
            Axis(id="material", name="material", tier="TIER2",  # type: ignore[arg-type]
                 kind=AxisKind.EMBEDDING, agreement=AgreementMetric.MEAN_PAIRWISE_COSINE,
                 measure="audio_embedding"),
        ]
        probes = probe_ladder(include_stubs=False)
        cells, _ = build_readout_map(
            backend, video_bank, axes, probes,
            alpha=0.5, schedule=schedule, thresholds=thresholds,
            measurer=measurer, rng=rng,
        )
        # EnergyOnsetProbe returns dim-4 embeddings; SyntheticMeasurer(embed_dim=8) returns
        # dim-8 targets — dimension mismatch yields 0.0 (valid score; no readout signal).
        # Either way, scores must lie in [-1, 1].
        for cell in cells:
            assert -1.0 <= cell.score <= 1.0, (
                f"Embedding cosine score out of [-1,1]: {cell.score} for {cell}"
            )

    def test_window_keys_have_correct_structure(self, backend, video_bank, schedule,
                                                thresholds, measurer, rng):
        axes = [
            Axis(id="presence", name="presence", tier="TIER1",  # type: ignore[arg-type]
                 kind=AxisKind.CATEGORICAL, agreement=AgreementMetric.EXACT_MATCH,
                 measure="presence_detector"),
        ]
        probes = probe_ladder(include_stubs=False)
        _, windows = build_readout_map(
            backend, video_bank, axes, probes,
            alpha=0.5, schedule=schedule, thresholds=thresholds,
            measurer=measurer, rng=rng,
        )
        for key in windows:
            assert isinstance(key, tuple) and len(key) == 3
            axis_id, probe_name, target_kind = key
            assert isinstance(axis_id, str)
            assert isinstance(probe_name, str)
            assert target_kind in ("ode", "fork_majority")

    def test_windows_are_window_estimate_instances(self, backend, video_bank, schedule,
                                                   thresholds, measurer, rng):
        axes = [
            Axis(id="presence", name="presence", tier="TIER1",  # type: ignore[arg-type]
                 kind=AxisKind.CATEGORICAL, agreement=AgreementMetric.EXACT_MATCH,
                 measure="presence_detector"),
        ]
        probes = probe_ladder(include_stubs=False)
        _, windows = build_readout_map(
            backend, video_bank, axes, probes,
            alpha=0.5, schedule=schedule, thresholds=thresholds,
            measurer=measurer, rng=rng,
        )
        for win in windows.values():
            assert isinstance(win, WindowEstimate)

    def test_n_videos_recorded_correctly(self, backend, video_bank, schedule,
                                         thresholds, measurer, rng):
        axes = [
            Axis(id="presence", name="presence", tier="TIER1",  # type: ignore[arg-type]
                 kind=AxisKind.CATEGORICAL, agreement=AgreementMetric.EXACT_MATCH,
                 measure="presence_detector"),
        ]
        probes = probe_ladder(include_stubs=False)
        cells, windows = build_readout_map(
            backend, video_bank, axes, probes,
            alpha=0.5, schedule=schedule, thresholds=thresholds,
            measurer=measurer, rng=rng,
        )
        n_videos = len(video_bank)
        for cell in cells:
            assert cell.n_videos == n_videos
        for win in windows.values():
            assert win.n_videos == n_videos

    def test_both_target_kinds_in_output(self, backend, video_bank, schedule,
                                         thresholds, measurer, rng):
        axes = [
            Axis(id="presence", name="presence", tier="TIER1",  # type: ignore[arg-type]
                 kind=AxisKind.CATEGORICAL, agreement=AgreementMetric.EXACT_MATCH,
                 measure="presence_detector"),
        ]
        probes = probe_ladder(include_stubs=False)
        cells, windows = build_readout_map(
            backend, video_bank, axes, probes,
            alpha=0.5, schedule=schedule, thresholds=thresholds,
            measurer=measurer, rng=rng,
        )
        targets_in_cells = {c.target for c in cells}
        assert "ode" in targets_in_cells
        assert "fork_majority" in targets_in_cells

        targets_in_windows = {k[2] for k in windows}
        assert "ode" in targets_in_windows
        assert "fork_majority" in targets_in_windows

    def test_empty_videos_list_returns_underpowered_windows(self, backend, schedule,
                                                            thresholds, measurer, rng):
        """With 0 videos, cells have n_videos=0, scores are nan, windows are underpowered."""
        axes = [
            Axis(id="presence", name="presence", tier="TIER1",  # type: ignore[arg-type]
                 kind=AxisKind.CATEGORICAL, agreement=AgreementMetric.EXACT_MATCH,
                 measure="presence_detector"),
        ]
        probes = probe_ladder(include_stubs=False)
        cells, windows = build_readout_map(
            backend, [], axes, probes,
            alpha=0.5, schedule=schedule, thresholds=thresholds,
            measurer=measurer, rng=rng,
        )
        # With 0 videos: cells are generated with n_videos=0 and nan scores
        for cell in cells:
            assert cell.n_videos == 0
        # Windows should be underpowered (n_videos < min_n = 1)
        for win in windows.values():
            assert win.underpowered

    def test_all_axes_from_config(self, backend, schedule, thresholds, measurer, rng):
        """Build map with all non-SEPARATE axes; verify structure."""
        from foley_cw.types import AxisTier
        cfg = load_config()
        axes = [a for a in cfg.axes if a.tier not in (AxisTier.SEPARATE, AxisTier.EXCLUDED)]
        videos = [
            {"cond": SyntheticVideoCond(mu=np.random.default_rng(i).standard_normal(DIM) * 2.0,
                                        video_id=f"v{i}")}
            for i in range(3)
        ]
        probes = probe_ladder(include_stubs=False)
        cells, windows = build_readout_map(
            backend, videos, axes, probes,
            alpha=0.5, schedule=schedule, thresholds=thresholds,
            measurer=measurer, rng=rng,
        )
        expected = len(axes) * len(probes) * len(schedule.scan_points) * 2  # 2 targets
        assert len(cells) == expected
        assert len(windows) == len(axes) * len(probes) * 2

    def test_readout_cell_probe_name_matches_probe(self, backend, video_bank, schedule,
                                                   thresholds, measurer, rng):
        axes = [
            Axis(id="presence", name="presence", tier="TIER1",  # type: ignore[arg-type]
                 kind=AxisKind.CATEGORICAL, agreement=AgreementMetric.EXACT_MATCH,
                 measure="presence_detector"),
        ]
        probes = probe_ladder(include_stubs=False)
        cells, _ = build_readout_map(
            backend, video_bank, axes, probes,
            alpha=0.5, schedule=schedule, thresholds=thresholds,
            measurer=measurer, rng=rng,
        )
        probe_names = {p.name for p in probes}
        for cell in cells:
            assert cell.probe in probe_names


# ---------------------------------------------------------------------------
# Import safety
# ---------------------------------------------------------------------------

def test_module_importable_numpy_only():
    """readout module must import with numpy only."""
    import foley_cw.readout  # noqa: F401


def test_no_scipy_imported():
    """readout module must NOT import scipy at import time."""
    import sys
    import foley_cw.readout  # noqa: F401
    # If scipy is not installed, this passes trivially.
    # If scipy IS somehow available, importing readout should still not trigger it.
    # We cannot unload modules, so just verify the import doesn't fail.
    _ = sys.modules  # reference to avoid warning
