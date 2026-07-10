"""Tests for foley_cw/probes.py.

All tests run on CPU using numpy only. The synthetic backend
(SyntheticGaussianFlow / SyntheticVideoCond) is used as the flow backend.
No GPU, no heavy deps.

Test plan:
  A. Module imports without heavy deps.
  B. EnergyOnsetProbe returns a SelfTarget of the right kind for every axis.
  C. EnergyOnsetProbe.legacy is False; CLAPProbe/SyncNetProbe/ImageBindProbe legacy=True.
  D. MLLMPreviewProbe.legacy is False (headline probe, non-legacy).
  E. Stub probes raise NotImplementedError.
  F. probe_ladder(include_stubs=False) returns only EnergyOnsetProbe.
  G. probe_ladder(include_stubs=True) returns all probes in ladder order.
  H. Probe protocol structural compliance.
  I. EnergyOnsetProbe accuracy rises (weakly) as x0(s) -> x1 on synthetic data.
  J. EnergyOnsetProbe is deterministic: same audio -> same SelfTarget.
  K. EnergyOnsetProbe handles edge-case vectors (all zeros, large values).
  L. All returned probes are Probe protocol instances.
"""

import numpy as np
import pytest

from foley_cw.config import load_config
from foley_cw.probes import (
    AudioTaggerProbe,
    CLAPProbe,
    EnergyOnsetProbe,
    ImageBindProbe,
    MLLMPreviewProbe,
    Probe,
    SyncNetProbe,
    probe_ladder,
)
from foley_cw.score_sde import generate_trajectory, x0_at
from foley_cw.synthetic_backend import SyntheticGaussianFlow, SyntheticVideoCond
from foley_cw.types import Axis, AxisKind, AxisTier, AgreementMetric, SelfTarget


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def all_axes():
    """Load axes from configs/axes.json."""
    cfg = load_config()
    return cfg.axes


@pytest.fixture(scope="module")
def schedule():
    cfg = load_config()
    return cfg.schedule


@pytest.fixture(scope="module")
def backend():
    return SyntheticGaussianFlow(dim=4, sigma2=0.25)


@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(42)


@pytest.fixture(scope="module")
def cond():
    mu = np.array([1.0, -0.5, 0.3, -0.8])
    return SyntheticVideoCond(mu=mu, video_id="test_video")


def _make_axis(ax_id, kind, measure, agreement):
    """Helper to create a test axis."""
    return Axis(
        id=ax_id,
        name=ax_id,
        tier=AxisTier.TIER1,
        kind=kind,
        agreement=agreement,
        measure=measure,
    )


CATEGORICAL_AXES = [
    _make_axis("presence", AxisKind.CATEGORICAL, "presence_detector", AgreementMetric.EXACT_MATCH),
    _make_axis("timing", AxisKind.CATEGORICAL, "onset_timing_bin", AgreementMetric.EXACT_MATCH),
    _make_axis("class", AxisKind.CATEGORICAL, "audio_tagger_top1", AgreementMetric.KRIPPENDORFF_ALPHA),
    _make_axis("binding", AxisKind.CATEGORICAL, "binding_label", AgreementMetric.EXACT_MATCH),
    _make_axis("offscreen", AxisKind.CATEGORICAL, "seed_predictability", AgreementMetric.EXACT_MATCH),
]

EMBEDDING_AXES = [
    _make_axis("material", AxisKind.EMBEDDING, "audio_embedding", AgreementMetric.MEAN_PAIRWISE_COSINE),
]


# ---------------------------------------------------------------------------
# A. Import test
# ---------------------------------------------------------------------------

def test_module_imports_without_heavy_deps():
    """Module must import without torch, librosa, CLAP, etc."""
    import foley_cw.probes  # noqa: F401  — just check no import error
    assert True


# ---------------------------------------------------------------------------
# B. EnergyOnsetProbe returns right kind for every axis
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("axis", CATEGORICAL_AXES)
def test_energy_onset_categorical_returns_categorical(axis):
    probe = EnergyOnsetProbe()
    x0 = np.array([0.5, -0.2, 0.1, 0.8], dtype=float)
    st = probe.predict(x0, axis)
    assert isinstance(st, SelfTarget)
    assert st.kind is AxisKind.CATEGORICAL
    assert st.label is not None
    assert st.axis_id == axis.id


@pytest.mark.parametrize("axis", EMBEDDING_AXES)
def test_energy_onset_embedding_returns_embedding(axis):
    probe = EnergyOnsetProbe()
    x0 = np.array([0.5, -0.2, 0.1, 0.8], dtype=float)
    st = probe.predict(x0, axis)
    assert isinstance(st, SelfTarget)
    assert st.kind is AxisKind.EMBEDDING
    assert st.embedding is not None
    assert isinstance(st.embedding, np.ndarray)
    assert st.axis_id == axis.id


def test_energy_onset_embedding_is_unit_norm():
    probe = EnergyOnsetProbe()
    axis = EMBEDDING_AXES[0]
    x0 = np.array([0.5, -0.2, 0.1, 0.8], dtype=float)
    st = probe.predict(x0, axis)
    norm = float(np.linalg.norm(st.embedding))
    assert abs(norm - 1.0) < 1e-9, f"embedding not unit norm: {norm}"


def test_energy_onset_embedding_zero_vector():
    """Zero-vector input should produce a zero-norm embedding without crashing."""
    probe = EnergyOnsetProbe()
    axis = EMBEDDING_AXES[0]
    x0 = np.zeros(4, dtype=float)
    st = probe.predict(x0, axis)
    assert st.embedding is not None
    # Should be all zeros (safe fallback)
    assert np.allclose(st.embedding, 0.0)


# ---------------------------------------------------------------------------
# C. Legacy flags
# ---------------------------------------------------------------------------

def test_energy_onset_not_legacy():
    assert EnergyOnsetProbe().legacy is False


def test_clap_is_legacy():
    assert CLAPProbe().legacy is True


def test_syncnet_is_legacy():
    assert SyncNetProbe().legacy is True


def test_imagebind_is_legacy():
    assert ImageBindProbe().legacy is True


# ---------------------------------------------------------------------------
# D. MLLM headline probe is non-legacy
# ---------------------------------------------------------------------------

def test_mllm_preview_not_legacy():
    assert MLLMPreviewProbe().legacy is False


def test_mllm_preview_name():
    assert MLLMPreviewProbe().name == "mllm_preview"


# ---------------------------------------------------------------------------
# E. Stub probes raise NotImplementedError
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ProbeClass", [CLAPProbe, SyncNetProbe, ImageBindProbe,
                                         AudioTaggerProbe, MLLMPreviewProbe])
def test_stub_raises_not_implemented(ProbeClass):
    probe = ProbeClass()
    x0 = np.array([0.5, -0.2, 0.1, 0.8], dtype=float)
    axis = CATEGORICAL_AXES[0]
    with pytest.raises(NotImplementedError) as exc_info:
        probe.predict(x0, axis)
    # Error message should name the dependency
    assert len(str(exc_info.value)) > 10


# ---------------------------------------------------------------------------
# F. probe_ladder(include_stubs=False) returns only CPU-runnable probes
# ---------------------------------------------------------------------------

def test_probe_ladder_no_stubs_contains_only_energy_onset():
    ladder = probe_ladder(include_stubs=False)
    assert len(ladder) == 1
    assert isinstance(ladder[0], EnergyOnsetProbe)


def test_probe_ladder_no_stubs_is_default():
    ladder = probe_ladder()
    assert len(ladder) == 1
    assert isinstance(ladder[0], EnergyOnsetProbe)


# ---------------------------------------------------------------------------
# G. probe_ladder(include_stubs=True) returns all probes in order
# ---------------------------------------------------------------------------

def test_probe_ladder_with_stubs_length():
    ladder = probe_ladder(include_stubs=True)
    # rung1 + 3 legacy (rung2) + rung3 + rung5 = 6 probes
    assert len(ladder) == 6


def test_probe_ladder_with_stubs_order():
    ladder = probe_ladder(include_stubs=True)
    # First is energy onset (rung 1)
    assert isinstance(ladder[0], EnergyOnsetProbe)
    # Rung 2: legacy probes
    assert isinstance(ladder[1], CLAPProbe)
    assert isinstance(ladder[2], SyncNetProbe)
    assert isinstance(ladder[3], ImageBindProbe)
    # Rung 3: audio tagger
    assert isinstance(ladder[4], AudioTaggerProbe)
    # Rung 5: MLLM headline
    assert isinstance(ladder[5], MLLMPreviewProbe)


def test_probe_ladder_with_stubs_names():
    ladder = probe_ladder(include_stubs=True)
    names = [p.name for p in ladder]
    assert names[0] == "energy_onset"
    assert "clap" in names
    assert "syncnet" in names
    assert "imagebind" in names
    assert "audio_tagger" in names
    assert "mllm_preview" in names


# ---------------------------------------------------------------------------
# H. Probe Protocol structural compliance
# ---------------------------------------------------------------------------

def test_energy_onset_satisfies_probe_protocol():
    probe = EnergyOnsetProbe()
    assert isinstance(probe, Probe)


@pytest.mark.parametrize("ProbeClass", [CLAPProbe, SyncNetProbe, ImageBindProbe,
                                         AudioTaggerProbe, MLLMPreviewProbe])
def test_stubs_satisfy_probe_protocol(ProbeClass):
    probe = ProbeClass()
    assert isinstance(probe, Probe)


# ---------------------------------------------------------------------------
# I. EnergyOnsetProbe accuracy rises as x0(s) -> x1 on synthetic data
# ---------------------------------------------------------------------------

def test_energy_onset_accuracy_rises_with_s(backend, cond, schedule):
    """At s=1, x0(1)=x1 exactly, so the probe should perfectly predict the final label.
    At s=0, x0(0) is noisy, so accuracy should be lower on average.
    We test that accuracy at s=0.9 >= accuracy at s=0.1 over a small video bank.
    """
    rng_local = np.random.default_rng(777)
    probe = EnergyOnsetProbe()
    presence_axis = CATEGORICAL_AXES[0]  # presence_detector

    # Run multiple videos; collect predictions at s=0.1 and s=0.9
    n_videos = 8
    video_bank = SyntheticGaussianFlow.make_video_bank(n_videos, dim=4, mu_scale=3.0, seed=1)

    correct_early = 0
    correct_late = 0
    total = 0

    for vc in video_bank:
        traj = generate_trajectory(
            backend, vc, schedule, rng_local, alpha=0.0,
            record_points=(0.1, 0.9, 1.0)
        )
        states = traj["states"]
        audio_final = traj["audio"]

        # True label from final audio (self-target)
        true_label = int(np.mean(audio_final) > 0)

        x0_early = x0_at(backend, states[0.1], 0.1, vc)
        x0_late = x0_at(backend, states[0.9], 0.9, vc)

        st_early = probe.predict(x0_early, presence_axis)
        st_late = probe.predict(x0_late, presence_axis)

        correct_early += int(st_early.label == true_label)
        correct_late += int(st_late.label == true_label)
        total += 1

    acc_early = correct_early / total
    acc_late = correct_late / total

    # Late should not be worse than early (monotone expectation on synthetic data)
    # We allow some slack (5pp) since both can be high on well-separated synthetic mus
    assert acc_late >= acc_early - 0.05, (
        f"Late accuracy ({acc_late:.2f}) should be >= early accuracy ({acc_early:.2f}) - 0.05"
    )

    # At s=1 the probe should be perfect (x0(1) = x1 under the analytic backend)
    rng_local2 = np.random.default_rng(888)
    correct_final = 0
    for vc in video_bank:
        traj = generate_trajectory(
            backend, vc, schedule, rng_local2, alpha=0.0,
            record_points=(1.0,)
        )
        x1 = traj["final_state"]
        audio_final = traj["audio"]
        true_label = int(np.mean(audio_final) > 0)
        # At s=1, x0(1) = x1 = audio
        st = probe.predict(backend.decode(x1), presence_axis)
        correct_final += int(st.label == true_label)
    assert correct_final == n_videos, (
        f"EnergyOnsetProbe should be perfect at s=1 on presence, got {correct_final}/{n_videos}"
    )


# ---------------------------------------------------------------------------
# J. EnergyOnsetProbe is deterministic
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("axis", CATEGORICAL_AXES + EMBEDDING_AXES)
def test_energy_onset_deterministic(axis):
    probe = EnergyOnsetProbe()
    x0 = np.array([0.3, -0.7, 1.1, -0.4], dtype=float)
    st1 = probe.predict(x0, axis)
    st2 = probe.predict(x0, axis)
    if axis.kind is AxisKind.CATEGORICAL:
        assert st1.label == st2.label
    else:
        np.testing.assert_array_equal(st1.embedding, st2.embedding)


# ---------------------------------------------------------------------------
# K. Edge-case vectors
# ---------------------------------------------------------------------------

def test_energy_onset_large_values():
    probe = EnergyOnsetProbe()
    x0 = np.array([1e8, -1e8, 1e8, -1e8], dtype=float)
    for axis in CATEGORICAL_AXES + EMBEDDING_AXES:
        st = probe.predict(x0, axis)
        assert isinstance(st, SelfTarget)


def test_energy_onset_small_dim():
    """Works even on length-1 vectors."""
    probe = EnergyOnsetProbe()
    x0 = np.array([0.5])
    axis = CATEGORICAL_AXES[0]
    st = probe.predict(x0, axis)
    assert isinstance(st, SelfTarget)
    assert st.kind is AxisKind.CATEGORICAL


def test_energy_onset_binding_label_shape():
    probe = EnergyOnsetProbe()
    axis = _make_axis("binding", AxisKind.CATEGORICAL, "binding_label", AgreementMetric.EXACT_MATCH)
    x0 = np.array([0.5, -0.2, 0.1, 0.8], dtype=float)
    st = probe.predict(x0, axis)
    assert isinstance(st.label, tuple)
    assert len(st.label) == 2


# ---------------------------------------------------------------------------
# L. All probes from ladder are Probe protocol instances
# ---------------------------------------------------------------------------

def test_all_ladder_probes_are_probe_protocol():
    for probe in probe_ladder(include_stubs=True):
        assert isinstance(probe, Probe), f"{probe.name} does not satisfy Probe protocol"


# ---------------------------------------------------------------------------
# Additional: NotImplementedError messages mention the dependency
# ---------------------------------------------------------------------------

def test_clap_error_message_mentions_dep():
    probe = CLAPProbe()
    x0 = np.array([0.5, -0.2, 0.1, 0.8], dtype=float)
    axis = CATEGORICAL_AXES[0]
    with pytest.raises(NotImplementedError) as exc_info:
        probe.predict(x0, axis)
    msg = str(exc_info.value).lower()
    assert "clap" in msg or "laion" in msg


def test_mllm_error_message_mentions_dep():
    probe = MLLMPreviewProbe()
    x0 = np.array([0.5, -0.2, 0.1, 0.8], dtype=float)
    axis = CATEGORICAL_AXES[0]
    with pytest.raises(NotImplementedError) as exc_info:
        probe.predict(x0, axis)
    msg = str(exc_info.value).lower()
    assert "mllm" in msg or "client" in msg or "openai" in msg or "anthropic" in msg


def test_audio_tagger_error_message_mentions_dep():
    probe = AudioTaggerProbe()
    x0 = np.array([0.5, -0.2, 0.1, 0.8], dtype=float)
    axis = CATEGORICAL_AXES[0]
    with pytest.raises(NotImplementedError) as exc_info:
        probe.predict(x0, axis)
    msg = str(exc_info.value).lower()
    assert "tagger" in msg or "torch" in msg


# ---------------------------------------------------------------------------
# Presence probe: correct sign matches final x0 = x1 under analytic flow
# ---------------------------------------------------------------------------

def test_energy_onset_presence_exact_at_final(backend, cond, schedule):
    """x0_at(s=1) = decode(x1) = x1 for the SyntheticGaussianFlow.
    The presence label must match directly.
    """
    rng_local = np.random.default_rng(99)
    traj = generate_trajectory(backend, cond, schedule, rng_local, alpha=0.0,
                                record_points=(1.0,))
    audio_final = traj["audio"]
    probe = EnergyOnsetProbe()
    axis = CATEGORICAL_AXES[0]  # presence
    st = probe.predict(audio_final, axis)
    expected = int(np.mean(audio_final) > 0)
    assert st.label == expected
