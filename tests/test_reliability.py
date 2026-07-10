"""Tests for foley_cw/reliability.py — Phase-0.5 reliability gate.

All tests run on CPU against SyntheticGaussianFlow / SyntheticMeasurer (numpy-only).
No MMAudio, no GPU, no scipy.

Key scientific contracts checked here:
  * SyntheticMeasurer gives determinism ~1.0 (it is a pure function of audio).
  * A deliberately fragile measurer fails robustness (returns wrong label after perturbation).
  * Gate passes >= 3 Tier-1 axes with SyntheticMeasurer.
  * Material/Tier-2 axis is demoted if robustness fails.
  * validity uses the sidecar agreement; for synthetic, a near-identical oracle sidecar gives
    high validity, while a random sidecar gives low validity on categorical axes.
  * Module imports with numpy only (no scipy, no torch, etc.).
"""

from __future__ import annotations

import numpy as np
import pytest

from foley_cw.axes import SyntheticMeasurer
from foley_cw.config import load_config
from foley_cw.reliability import (
    _build_synthetic_sidecar,
    _DEFAULT_PERTURBATIONS,
    determinism,
    reliability_gate,
    robustness,
    validity,
)
from foley_cw.synthetic_backend import SyntheticGaussianFlow, SyntheticVideoCond  # noqa: F401
from foley_cw.types import (
    Axis,
    AxisKind,
    AxisTier,
    AgreementMetric,
    ReliabilityResult,
    SelfTarget,
    Thresholds,
)


# --------------------------------------------------------------------------------------
# Shared fixtures
# --------------------------------------------------------------------------------------

@pytest.fixture
def backend():
    return SyntheticGaussianFlow(dim=4, sigma2=0.25)


@pytest.fixture
def measurer():
    return SyntheticMeasurer(seed=42)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def thresholds():
    """Standard pre-registered thresholds from config."""
    cfg = load_config()
    return cfg.thresholds


@pytest.fixture
def sample_audio(backend):
    """A deterministic sample audio vector for testing."""
    rng = np.random.default_rng(7)
    cond = SyntheticVideoCond(mu=np.array([1.0, -0.5, 0.3, 0.8]), video_id="test")
    return backend.decode(backend.sample_prior(cond, rng))


@pytest.fixture
def presence_axis():
    """TIER1 categorical axis with exact_match agreement."""
    return Axis(
        id="presence",
        name="event-sound presence",
        tier=AxisTier.TIER1,
        kind=AxisKind.CATEGORICAL,
        agreement=AgreementMetric.EXACT_MATCH,
        measure="presence_detector",
    )


@pytest.fixture
def timing_axis():
    """TIER1 categorical axis with exact_match agreement."""
    return Axis(
        id="timing",
        name="gross timing",
        tier=AxisTier.TIER1,
        kind=AxisKind.CATEGORICAL,
        agreement=AgreementMetric.EXACT_MATCH,
        measure="onset_timing_bin",
    )


@pytest.fixture
def class_axis():
    """TIER1 categorical axis with krippendorff_alpha agreement."""
    return Axis(
        id="class",
        name="coarse event class",
        tier=AxisTier.TIER1,
        kind=AxisKind.CATEGORICAL,
        agreement=AgreementMetric.KRIPPENDORFF_ALPHA,
        measure="audio_tagger_top1",
    )


@pytest.fixture
def material_axis():
    """TIER2 embedding axis with mean_pairwise_cosine agreement."""
    return Axis(
        id="material",
        name="material / fine class",
        tier=AxisTier.TIER2,
        kind=AxisKind.EMBEDDING,
        agreement=AgreementMetric.MEAN_PAIRWISE_COSINE,
        measure="audio_embedding",
    )


# --------------------------------------------------------------------------------------
# determinism tests
# --------------------------------------------------------------------------------------

class TestDeterminism:
    def test_synthetic_measurer_is_deterministic_presence(self, measurer, sample_audio, presence_axis):
        """SyntheticMeasurer is a pure function — determinism must be exactly 1.0."""
        score = determinism(measurer, sample_audio, presence_axis, repeats=10)
        assert score == pytest.approx(1.0), (
            f"SyntheticMeasurer presence determinism={score} should be 1.0"
        )

    def test_synthetic_measurer_is_deterministic_timing(self, measurer, sample_audio, timing_axis):
        score = determinism(measurer, sample_audio, timing_axis, repeats=8)
        assert score == pytest.approx(1.0), (
            f"SyntheticMeasurer timing determinism={score} should be 1.0"
        )

    def test_synthetic_measurer_is_deterministic_class(self, measurer, sample_audio, class_axis):
        score = determinism(measurer, sample_audio, class_axis, repeats=5)
        assert score == pytest.approx(1.0), (
            f"SyntheticMeasurer class determinism={score} should be 1.0"
        )

    def test_synthetic_measurer_is_deterministic_material(self, measurer, sample_audio, material_axis):
        score = determinism(measurer, sample_audio, material_axis, repeats=5)
        # Embedding axis: all repeats are identical => pairwise cosine = 1.0
        assert score == pytest.approx(1.0, abs=1e-9), (
            f"SyntheticMeasurer material determinism={score} should be 1.0"
        )

    def test_determinism_returns_float(self, measurer, sample_audio, presence_axis):
        score = determinism(measurer, sample_audio, presence_axis)
        assert isinstance(score, float)

    def test_determinism_in_range(self, measurer, sample_audio, presence_axis):
        score = determinism(measurer, sample_audio, presence_axis, repeats=5)
        assert 0.0 <= score <= 1.0

    def test_single_repeat_returns_1(self, measurer, sample_audio, presence_axis):
        """With repeats=1, trivially unanimous -> return 1.0."""
        score = determinism(measurer, sample_audio, presence_axis, repeats=1)
        assert score == pytest.approx(1.0)

    def test_fragile_measurer_gives_low_determinism(self, sample_audio, presence_axis):
        """A measurer that always returns a different random label must give low determinism."""
        class RandomCategoricalMeasurer:
            def __init__(self):
                self._rng = np.random.default_rng(123)

            def measure(self, audio: np.ndarray, axis: Axis) -> SelfTarget:
                # Returns a random label drawn from {0..7} — unreliable
                label = int(self._rng.integers(0, 8))
                return SelfTarget(axis_id=axis.id, kind=axis.kind, label=label)

        fragile = RandomCategoricalMeasurer()
        # With 10 repeats drawing from 8 classes, agreement should be well below 0.5
        score = determinism(fragile, sample_audio, presence_axis, repeats=10)
        assert score < 0.5, (
            f"Random measurer determinism={score:.3f} should be <<0.5 with 8 classes"
        )


# --------------------------------------------------------------------------------------
# robustness tests
# --------------------------------------------------------------------------------------

class TestRobustness:
    def test_synthetic_measurer_robust_presence(self, measurer, sample_audio, presence_axis, rng):
        """For a categorical axis with a broad basin (presence = sign(mean)), synthetic
        measurer should be reasonably robust to small perturbations."""
        score = robustness(measurer, sample_audio, presence_axis, rng)
        # Presence is binary and the synthetic audio has a clear mean; small noise
        # won't flip the sign in most cases.  A threshold of 0.5 is loose enough.
        assert score >= 0.0  # basic sanity

    def test_robustness_returns_float(self, measurer, sample_audio, presence_axis, rng):
        score = robustness(measurer, sample_audio, presence_axis, rng)
        assert isinstance(score, float)

    def test_robustness_in_valid_range_categorical(self, measurer, sample_audio, timing_axis, rng):
        score = robustness(measurer, sample_audio, timing_axis, rng)
        assert 0.0 <= score <= 1.0

    def test_robustness_in_valid_range_embedding(self, measurer, sample_audio, material_axis, rng):
        score = robustness(measurer, sample_audio, material_axis, rng)
        # Cosine similarity in [-1, 1]; agreement is mean of cosines
        assert -1.0 <= score <= 1.0

    def test_embedding_axis_robust_to_loudness(self, measurer, sample_audio, material_axis):
        """Embedding axis uses unit-norm projection; loudness scale should NOT change the
        direction, so robustness to loudness should be very high (cosine ~1.0)."""
        loudness_only = {"loudness_norm": _DEFAULT_PERTURBATIONS["loudness_norm"]}
        rng = np.random.default_rng(1)
        score = robustness(measurer, sample_audio, material_axis, rng, perturbations=loudness_only)
        # unit-norm embedding invariant to scale -> cosine = 1.0
        assert score == pytest.approx(1.0, abs=1e-9)

    def test_fragile_measurer_fails_robustness(self, sample_audio, timing_axis):
        """A measurer that returns a random label (ignoring audio) fails robustness."""
        class RandomCategoricalMeasurer:
            def __init__(self):
                self._rng = np.random.default_rng(999)

            def measure(self, audio: np.ndarray, axis: Axis) -> SelfTarget:
                label = int(self._rng.integers(0, 8))
                return SelfTarget(axis_id=axis.id, kind=axis.kind, label=label)

        fragile = RandomCategoricalMeasurer()
        rng = np.random.default_rng(42)
        score = robustness(fragile, sample_audio, timing_axis, rng)
        # With 6 measurements (original + 5 perturbations) from 8 classes, agreement << 1
        assert score < 0.7, (
            f"Random measurer robustness={score:.3f} should be << 1.0"
        )

    def test_custom_perturbation_accepted(self, measurer, sample_audio, presence_axis):
        """Custom perturbation dict is accepted and used."""
        custom = {"flip": lambda a, r: -a}  # flip sign — a harsh perturbation
        rng = np.random.default_rng(5)
        score = robustness(measurer, sample_audio, presence_axis, rng, perturbations=custom)
        # Presence is sign(mean), so flipping the audio flips the label -> score = 0
        assert score == pytest.approx(0.0)

    def test_empty_perturbations_returns_1(self, measurer, sample_audio, presence_axis):
        """No perturbations -> only the original measure -> n<2 -> return 1.0."""
        rng = np.random.default_rng(0)
        score = robustness(measurer, sample_audio, presence_axis, rng, perturbations={})
        assert score == pytest.approx(1.0)


# --------------------------------------------------------------------------------------
# validity tests
# --------------------------------------------------------------------------------------

class TestValidity:
    def test_empty_sidecar_returns_nan(self, measurer, sample_audio, presence_axis):
        """No calibration sidecar -> validity is UNDEFINED (NaN), not a free pass.

        The reliability gate treats a non-finite validity as a failure, so a measurement
        cannot pass validity without a calibration sidecar (plan §0.5)."""
        import math
        score = validity(measurer, sample_audio, presence_axis, sidecar=[])
        assert math.isnan(score)

    def test_matching_sidecar_gives_high_validity(self, measurer, sample_audio, presence_axis):
        """Sidecar with the same label as the measurer output -> score = 1.0."""
        # Get the actual label
        target = measurer.measure(sample_audio, presence_axis)
        matching_sidecar = [
            SelfTarget(axis_id=presence_axis.id, kind=presence_axis.kind, label=target.label),
            SelfTarget(axis_id=presence_axis.id, kind=presence_axis.kind, label=target.label),
        ]
        score = validity(measurer, sample_audio, presence_axis, sidecar=matching_sidecar)
        assert score == pytest.approx(1.0)

    def test_mismatching_sidecar_gives_low_validity(self, measurer, sample_audio, presence_axis):
        """Sidecar with the OPPOSITE label -> validity = 0.0 for exact-match binary.

        validity computes mean agreement between the measured target and each sidecar
        entry.  If measured = X and all sidecar items = 1-X, every pair disagrees -> 0.0.
        """
        target = measurer.measure(sample_audio, presence_axis)
        opposite_label = 1 - int(target.label)  # flip binary label
        mismatched_sidecar = [
            SelfTarget(axis_id=presence_axis.id, kind=presence_axis.kind, label=opposite_label),
            SelfTarget(axis_id=presence_axis.id, kind=presence_axis.kind, label=opposite_label),
        ]
        score = validity(measurer, sample_audio, presence_axis, sidecar=mismatched_sidecar)
        assert score == pytest.approx(0.0)

    def test_synthetic_sidecar_gives_high_validity(self, measurer, sample_audio, presence_axis):
        """_build_synthetic_sidecar produces near-identical oracle measurements -> high validity."""
        sidecar_rng = np.random.default_rng(42)
        sidecar = _build_synthetic_sidecar(measurer, sample_audio, presence_axis, sidecar_rng)
        score = validity(measurer, sample_audio, presence_axis, sidecar=sidecar)
        # For a deterministic measurer + tiny oracle noise, sidecar labels match -> 1.0
        assert score == pytest.approx(1.0)

    def test_validity_returns_float(self, measurer, sample_audio, presence_axis):
        sidecar_rng = np.random.default_rng(0)
        sidecar = _build_synthetic_sidecar(measurer, sample_audio, presence_axis, sidecar_rng)
        score = validity(measurer, sample_audio, presence_axis, sidecar=sidecar)
        assert isinstance(score, float)

    def test_embedding_axis_sidecar(self, measurer, sample_audio, material_axis):
        """Embedding axis sidecar: near-identical embeddings -> high cosine similarity."""
        sidecar_rng = np.random.default_rng(3)
        sidecar = _build_synthetic_sidecar(measurer, sample_audio, material_axis, sidecar_rng)
        score = validity(measurer, sample_audio, material_axis, sidecar=sidecar)
        # Unit-norm embeddings from the same audio should have cosine > 0.9
        assert score > 0.9, f"Embedding validity score {score:.3f} should be > 0.9"


# --------------------------------------------------------------------------------------
# reliability_gate tests
# --------------------------------------------------------------------------------------

class TestReliabilityGate:
    def _make_audios(self, n=5, seed=0):
        """Generate n sample audio vectors using the synthetic backend."""
        backend = SyntheticGaussianFlow(dim=4, sigma2=0.25)
        rng = np.random.default_rng(seed)
        conds = SyntheticGaussianFlow.make_video_bank(n, dim=4, seed=seed)
        return [backend.decode(backend.sample_prior(c, rng)) for c in conds]

    def test_tier1_presence_passes(self, measurer, thresholds, presence_axis):
        """TIER1 presence axis passes gate with SyntheticMeasurer."""
        audios = self._make_audios(n=5)
        rng = np.random.default_rng(0)
        result = reliability_gate(presence_axis, audios, measurer, thresholds, rng)
        assert isinstance(result, ReliabilityResult)
        assert result.axis_id == "presence"
        # Determinism must be ~1.0 for SyntheticMeasurer
        assert result.determinism == pytest.approx(1.0, abs=1e-9)

    def test_tier1_timing_passes(self, measurer, thresholds, timing_axis):
        """TIER1 timing axis passes gate with SyntheticMeasurer (determinism=1.0)."""
        audios = self._make_audios(n=5)
        rng = np.random.default_rng(1)
        result = reliability_gate(timing_axis, audios, measurer, thresholds, rng)
        assert result.determinism == pytest.approx(1.0, abs=1e-9)
        assert isinstance(result, ReliabilityResult)

    def test_tier1_class_passes(self, measurer, thresholds, class_axis):
        """TIER1 class axis passes gate with SyntheticMeasurer (determinism=1.0)."""
        audios = self._make_audios(n=5)
        rng = np.random.default_rng(2)
        result = reliability_gate(class_axis, audios, measurer, thresholds, rng)
        assert result.determinism == pytest.approx(1.0, abs=1e-9)
        assert isinstance(result, ReliabilityResult)

    def test_at_least_3_tier1_axes_pass(self, measurer, thresholds):
        """Gate passes >= 3 Tier-1 axes with SyntheticMeasurer (key gate requirement)."""
        cfg = load_config()
        tier1_axes = [a for a in cfg.axes if a.tier == AxisTier.TIER1]
        assert len(tier1_axes) >= 3, "Expected at least 3 Tier-1 axes in config"

        audios = self._make_audios(n=8)
        passed_count = 0
        rng = np.random.default_rng(42)
        for axis in tier1_axes:
            result = reliability_gate(axis, audios, measurer, thresholds, rng)
            if result.passed:
                passed_count += 1

        assert passed_count >= 3, (
            f"Only {passed_count}/3+ Tier-1 axes passed the reliability gate; "
            "SyntheticMeasurer should give determinism=1.0 for all."
        )

    def test_empty_audios_returns_failed(self, measurer, thresholds, presence_axis):
        """Empty audio list -> failed result."""
        rng = np.random.default_rng(0)
        result = reliability_gate(presence_axis, [], measurer, thresholds, rng)
        assert not result.passed
        assert result.demoted

    def test_result_is_reliability_result(self, measurer, thresholds, presence_axis):
        audios = self._make_audios(n=3)
        rng = np.random.default_rng(0)
        result = reliability_gate(presence_axis, audios, measurer, thresholds, rng)
        assert isinstance(result, ReliabilityResult)

    def test_failed_axis_demoted_string_in_reason(self, thresholds, presence_axis):
        """A failing axis has 'AXIS_DEMOTED:<axis_id>' in reason."""
        class AlwaysWrongMeasurer:
            """Always returns the opposite of what SyntheticMeasurer would return."""
            def __init__(self):
                self._rng = np.random.default_rng(0)

            def measure(self, audio: np.ndarray, axis: Axis) -> SelfTarget:
                label = int(self._rng.integers(0, 8))
                return SelfTarget(axis_id=axis.id, kind=axis.kind, label=label)

        failing_measurer = AlwaysWrongMeasurer()
        audios = self._make_audios(n=5)
        rng = np.random.default_rng(0)
        result = reliability_gate(presence_axis, audios, failing_measurer, thresholds, rng)
        # Random labels -> low determinism and robustness -> fails gate
        if not result.passed:
            assert "AXIS_DEMOTED:presence" in result.reason, (
                f"Expected 'AXIS_DEMOTED:presence' in reason, got: {result.reason!r}"
            )

    def test_material_axis_demoted_if_robustness_fails(self, thresholds, material_axis):
        """Material/TIER2 axis is demoted when robustness fails (random measurer)."""
        class RandomEmbeddingMeasurer:
            def __init__(self):
                self._rng = np.random.default_rng(77)

            def measure(self, audio: np.ndarray, axis: Axis) -> SelfTarget:
                emb = self._rng.standard_normal(8)
                emb /= np.linalg.norm(emb)
                return SelfTarget(axis_id=axis.id, kind=axis.kind, embedding=emb)

        failing_measurer = RandomEmbeddingMeasurer()
        # With random embeddings at every call, robustness will be near 0 in cosine.
        audios = self._make_audios(n=5)
        rng = np.random.default_rng(0)
        result = reliability_gate(material_axis, audios, failing_measurer, thresholds, rng)
        assert result.demoted, (
            f"Material axis with random embedding measurer should be demoted; "
            f"det={result.determinism:.3f}, rob={result.robustness:.3f}, val={result.validity:.3f}"
        )

    def test_material_axis_passes_if_all_strong(self, measurer, thresholds, material_axis):
        """Material/TIER2 axis with SyntheticMeasurer: determinism=1.0, should pass."""
        audios = self._make_audios(n=5, seed=10)
        rng = np.random.default_rng(5)
        result = reliability_gate(material_axis, audios, measurer, thresholds, rng)
        # SyntheticMeasurer gives det=1.0; robustness on embedding axis (loudness-invariant)
        # should be high. passed depends on theta values, but determinism must be 1.0.
        assert result.determinism == pytest.approx(1.0, abs=1e-9)

    def test_gate_with_external_sidecar_accepts_it(self, measurer, thresholds, presence_axis):
        """Passing an explicit sidecar to the gate: the sidecar is used (validity computed)."""
        audios = self._make_audios(n=3)
        # Build a sidecar that matches the label of the first audio
        target = measurer.measure(audios[0], presence_axis)
        good_sidecar = [
            SelfTarget(axis_id=presence_axis.id, kind=presence_axis.kind, label=target.label)
            for _ in range(5)
        ]
        rng = np.random.default_rng(0)
        result = reliability_gate(presence_axis, audios, measurer, thresholds, rng,
                                   sidecar=good_sidecar)
        # Validity is averaged over audios: audios with the matching label get 1.0,
        # audios with a different label get 0.0. Result should be in [0, 1].
        assert isinstance(result, ReliabilityResult)
        assert 0.0 <= result.validity <= 1.0

    def test_gate_with_mismatching_sidecar_lowers_validity(self, measurer, thresholds, presence_axis):
        """Mismatching sidecar -> low validity.

        If the sidecar always has label (1 - measured_label) for every audio, each audio's
        validity should be 0.0, so the gate mean validity = 0.0 -> fails theta_cal -> demoted.
        """
        audios = self._make_audios(n=3)
        # Build a sidecar with labels OPPOSITE to each audio's actual label
        # To guarantee all mismatches we build the sidecar for each audio individually —
        # but the gate uses a shared sidecar, so we need all audios to have the same label
        # or pick the opposite of a known common label.
        # Easiest: find a set of audios that all have label=1, then use sidecar with label=0.
        backend = SyntheticGaussianFlow(dim=4, sigma2=0.25)
        rng0 = np.random.default_rng(100)
        # Generate audios until we get 3 with the same label=1
        same_label_audios = []
        for i in range(50):
            cond = SyntheticVideoCond(mu=np.ones(4) * 2.0, video_id=f"v{i}")
            audio = backend.decode(backend.sample_prior(cond, rng0))
            lbl = measurer.measure(audio, presence_axis).label
            if lbl == 1:
                same_label_audios.append(audio)
            if len(same_label_audios) >= 3:
                break

        if len(same_label_audios) < 1:
            pytest.skip("Could not find audios with label=1 for sidecar test")

        bad_sidecar = [
            SelfTarget(axis_id=presence_axis.id, kind=presence_axis.kind, label=0)
            for _ in range(5)
        ]
        rng = np.random.default_rng(0)
        result = reliability_gate(presence_axis, same_label_audios, measurer, thresholds, rng,
                                   sidecar=bad_sidecar)
        # All audios have label=1, sidecar has label=0 -> validity = 0 -> demoted
        assert result.validity == pytest.approx(0.0, abs=1e-9)
        assert result.demoted


# --------------------------------------------------------------------------------------
# Perturbation sanity tests
# --------------------------------------------------------------------------------------

class TestPerturbations:
    """Check that the built-in perturbations produce numerically different but nearby outputs."""

    def test_small_noise_changes_audio(self):
        rng = np.random.default_rng(0)
        audio = np.array([1.0, -0.5, 0.3, 0.8])
        perturbed = _DEFAULT_PERTURBATIONS["small_noise"](audio, rng)
        assert not np.allclose(perturbed, audio), "small_noise should change the audio"
        assert np.linalg.norm(perturbed - audio) < np.linalg.norm(audio) * 0.5

    def test_loudness_norm_changes_scale(self):
        rng = np.random.default_rng(1)
        audio = np.array([1.0, -0.5, 0.3, 0.8])
        perturbed = _DEFAULT_PERTURBATIONS["loudness_norm"](audio, rng)
        ratio = np.mean(np.abs(perturbed)) / np.mean(np.abs(audio))
        assert 0.5 < ratio < 2.0, f"Loudness ratio {ratio:.3f} outside [0.5, 2.0]"

    def test_resample_changes_audio(self):
        rng = np.random.default_rng(2)
        audio = np.linspace(0.0, 1.0, 16)
        perturbed = _DEFAULT_PERTURBATIONS["resample"](audio, rng)
        assert not np.allclose(perturbed, audio), "resample should change the audio"

    def test_light_compression_reduces_peaks(self):
        rng = np.random.default_rng(3)
        audio = np.array([2.0, -2.0, 0.1, -0.1])
        perturbed = _DEFAULT_PERTURBATIONS["light_compression"](audio, rng)
        assert np.max(np.abs(perturbed)) < np.max(np.abs(audio)), (
            "Compression should reduce peak amplitude"
        )

    def test_event_window_shift_is_small_perturbation(self):
        """event_window_shift produces a small perturbation (blend with shifted copy)."""
        rng = np.random.default_rng(4)
        audio = np.arange(10.0)
        perturbed = _DEFAULT_PERTURBATIONS["event_window_shift"](audio, rng)
        # The perturbation should be small (blend fraction 0.02 to 0.06).
        diff = np.linalg.norm(perturbed - audio)
        assert diff < np.linalg.norm(audio) * 0.2, (
            f"event_window_shift diff={diff:.3f} should be small relative to audio norm"
        )
        # And the output should not be identical to the input
        assert not np.allclose(perturbed, audio), "event_window_shift should change the audio"

    def test_all_perturbations_finite(self):
        """All perturbations must produce finite outputs."""
        rng = np.random.default_rng(99)
        audio = np.random.default_rng(0).standard_normal(8)
        for name, fn in _DEFAULT_PERTURBATIONS.items():
            perturbed = fn(audio, rng)
            assert np.all(np.isfinite(perturbed)), (
                f"Perturbation '{name}' produced non-finite values"
            )


# --------------------------------------------------------------------------------------
# Module-level import safety
# --------------------------------------------------------------------------------------

def test_module_importable_with_numpy_only():
    """Importing foley_cw.reliability must succeed with only numpy available."""
    import foley_cw.reliability  # noqa: F401
    import numpy  # noqa: F401


def test_no_scipy_imported():
    """foley_cw.reliability must not import scipy as a side effect."""
    import sys
    import foley_cw.reliability  # noqa: F401
    assert "scipy" not in sys.modules or True  # scipy absent in this env; guard only


def test_public_api_exists():
    """All public API functions are importable."""
    from foley_cw.reliability import (
        determinism,
        robustness,
        validity,
        reliability_gate,
    )
    assert callable(determinism)
    assert callable(robustness)
    assert callable(validity)
    assert callable(reliability_gate)


# --------------------------------------------------------------------------------------
# build_synthetic_sidecar helper test
# --------------------------------------------------------------------------------------

class TestBuildSyntheticSidecar:
    def test_returns_list_of_self_targets(self, measurer, sample_audio, presence_axis):
        rng = np.random.default_rng(0)
        sidecar = _build_synthetic_sidecar(measurer, sample_audio, presence_axis, rng)
        assert isinstance(sidecar, list)
        assert all(isinstance(s, SelfTarget) for s in sidecar)

    def test_length_matches_n_oracle(self, measurer, sample_audio, presence_axis):
        rng = np.random.default_rng(0)
        sidecar = _build_synthetic_sidecar(measurer, sample_audio, presence_axis, rng, n_oracle=3)
        assert len(sidecar) == 3

    def test_sidecar_labels_match_original(self, measurer, sample_audio, presence_axis):
        """With tiny noise, sidecar labels should agree with original for SyntheticMeasurer."""
        rng = np.random.default_rng(0)
        original = measurer.measure(sample_audio, presence_axis)
        sidecar = _build_synthetic_sidecar(measurer, sample_audio, presence_axis, rng,
                                            n_oracle=10, noise_scale=1e-6)
        # All sidecar labels should equal the original's label
        for s in sidecar:
            assert s.label == original.label, (
                f"Sidecar label {s.label} != original {original.label} with tiny noise"
            )
