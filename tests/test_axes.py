"""Tests for foley_cw/axes.py.

Test objectives:
 1. Determinism: measuring the same audio twice returns equal SelfTargets.
 2. Kind correctness: each measure returns a SelfTarget whose .kind matches the
    axis.kind from configs/axes.json (loaded via load_config()).
 3. Independence from SyntheticGaussianFlow: the measurer works on any numpy array.
 4. RealMeasurer: every measure raises NotImplementedError (with a meaningful message).
 5. measure_self_target convenience wrapper: delegates to measurer correctly.
 6. Protocol check: SyntheticMeasurer satisfies the Measurer protocol.
 7. Embedding output: unit-norm.
 8. Unknown measure name: raises ValueError.
 9. SyntheticMeasurer with SyntheticGaussianFlow decoded audio (integration test).
10. Different audio vectors -> (usually) different labels for categorical measures
    (probabilistic; skipped if dim is too small; structural test only).

All tests are numpy-only; no scipy/torch/librosa dependencies.
"""

from __future__ import annotations

import numpy as np
import pytest

from foley_cw.axes import (
    Measurer,
    RealMeasurer,
    SyntheticMeasurer,
    measure_self_target,
)
from foley_cw.config import load_config
from foley_cw.synthetic_backend import SyntheticGaussianFlow, SyntheticVideoCond
from foley_cw.types import Axis, AxisKind, AxisTier, AgreementMetric


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_axes():
    cfg = load_config()
    return cfg.axes


def _make_audio(dim: int = 4, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(dim).astype(float)


def _measurer(seed: int = 42) -> SyntheticMeasurer:
    return SyntheticMeasurer(seed=seed)


# ---------------------------------------------------------------------------
# 1. Determinism: same audio -> equal SelfTarget for every axis
# ---------------------------------------------------------------------------


class TestDeterminism:
    """Each call to measure on identical audio returns an equal SelfTarget."""

    def test_all_axes_deterministic(self):
        axes = _load_axes()
        measurer = _measurer()
        audio = _make_audio(dim=8, seed=7)

        for axis in axes:
            # Skip axes that are explicitly EXCLUDED from measurement (edge case)
            if axis.measure == "seed_predictability":
                # SEPARATE axis — still has a deterministic synthetic rule
                pass
            t1 = measurer.measure(audio, axis)
            t2 = measurer.measure(audio, axis)
            assert t1.axis_id == t2.axis_id
            assert t1.kind == t2.kind
            if axis.kind is AxisKind.CATEGORICAL:
                assert t1.label == t2.label, (
                    f"axis {axis.id!r}: label not deterministic "
                    f"({t1.label} vs {t2.label})"
                )
            else:
                assert t1.embedding is not None
                assert t2.embedding is not None
                np.testing.assert_array_equal(
                    t1.embedding, t2.embedding,
                    err_msg=f"axis {axis.id!r}: embedding not deterministic"
                )

    def test_repeated_calls_on_large_audio(self):
        """Determinism holds for dim > 256 (beyond base matrix width)."""
        measurer = _measurer()
        audio = _make_audio(dim=300, seed=13)
        axes = _load_axes()
        for axis in axes:
            t1 = measurer.measure(audio, axis)
            t2 = measurer.measure(audio, axis)
            if axis.kind is AxisKind.CATEGORICAL:
                assert t1.label == t2.label
            else:
                np.testing.assert_array_equal(t1.embedding, t2.embedding)

    def test_different_seeds_give_different_tagger(self):
        """Two measurers with different seeds may give different results (probabilistic)."""
        audio = _make_audio(dim=32, seed=3)
        axes = {a.id: a for a in _load_axes()}
        class_axis = axes["class"]

        m1 = SyntheticMeasurer(seed=1)
        m2 = SyntheticMeasurer(seed=999)
        t1 = m1.measure(audio, class_axis)
        t2 = m2.measure(audio, class_axis)
        # They may differ (not a hard guarantee, but extremely likely for dim=32)
        # We just verify each returned a valid CATEGORICAL SelfTarget.
        assert t1.kind is AxisKind.CATEGORICAL
        assert t2.kind is AxisKind.CATEGORICAL
        assert isinstance(t1.label, int)


# ---------------------------------------------------------------------------
# 2. Kind correctness: returned SelfTarget.kind matches axis.kind
# ---------------------------------------------------------------------------


class TestKindCorrectness:
    """Every measure returns the AxisKind declared in axes.json."""

    @pytest.fixture(scope="class")
    def axes_and_measurer(self):
        return _load_axes(), _measurer()

    def test_presence_is_categorical(self, axes_and_measurer):
        axes, m = axes_and_measurer
        axis = next(a for a in axes if a.id == "presence")
        assert axis.kind is AxisKind.CATEGORICAL
        t = m.measure(_make_audio(), axis)
        assert t.kind is AxisKind.CATEGORICAL

    def test_timing_is_categorical(self, axes_and_measurer):
        axes, m = axes_and_measurer
        axis = next(a for a in axes if a.id == "timing")
        assert axis.kind is AxisKind.CATEGORICAL
        t = m.measure(_make_audio(), axis)
        assert t.kind is AxisKind.CATEGORICAL

    def test_class_is_categorical(self, axes_and_measurer):
        axes, m = axes_and_measurer
        axis = next(a for a in axes if a.id == "class")
        assert axis.kind is AxisKind.CATEGORICAL
        t = m.measure(_make_audio(), axis)
        assert t.kind is AxisKind.CATEGORICAL

    def test_material_is_embedding(self, axes_and_measurer):
        axes, m = axes_and_measurer
        axis = next(a for a in axes if a.id == "material")
        assert axis.kind is AxisKind.EMBEDDING
        t = m.measure(_make_audio(), axis)
        assert t.kind is AxisKind.EMBEDDING

    def test_binding_is_categorical(self, axes_and_measurer):
        axes, m = axes_and_measurer
        axis = next(a for a in axes if a.id == "binding")
        assert axis.kind is AxisKind.CATEGORICAL
        t = m.measure(_make_audio(dim=4), axis)
        assert t.kind is AxisKind.CATEGORICAL

    def test_offscreen_hallucination_is_categorical(self, axes_and_measurer):
        axes, m = axes_and_measurer
        axis = next(a for a in axes if a.id == "offscreen_hallucination")
        assert axis.kind is AxisKind.CATEGORICAL
        t = m.measure(_make_audio(), axis)
        assert t.kind is AxisKind.CATEGORICAL

    def test_all_axes_have_matching_kind(self, axes_and_measurer):
        axes, m = axes_and_measurer
        audio = _make_audio(dim=8)
        for axis in axes:
            t = m.measure(audio, axis)
            assert t.kind is axis.kind, (
                f"axis {axis.id!r}: returned kind {t.kind!r}, expected {axis.kind!r}"
            )

    def test_axis_id_is_preserved(self, axes_and_measurer):
        axes, m = axes_and_measurer
        audio = _make_audio(dim=4)
        for axis in axes:
            t = m.measure(audio, axis)
            assert t.axis_id == axis.id


# ---------------------------------------------------------------------------
# 3. SelfTarget fields are well-formed
# ---------------------------------------------------------------------------


class TestSelfTargetFields:

    def test_categorical_has_label_no_embedding(self):
        axes = _load_axes()
        m = _measurer()
        audio = _make_audio(dim=4)
        for axis in [a for a in axes if a.kind is AxisKind.CATEGORICAL]:
            t = m.measure(audio, axis)
            assert t.label is not None
            # embedding may or may not be None — SelfTarget post_init only requires label
            # for categorical; we don't check embedding field

    def test_embedding_has_embedding_no_label_requirement(self):
        axes = _load_axes()
        m = _measurer()
        audio = _make_audio(dim=4)
        for axis in [a for a in axes if a.kind is AxisKind.EMBEDDING]:
            t = m.measure(audio, axis)
            assert t.embedding is not None
            assert isinstance(t.embedding, np.ndarray)


# ---------------------------------------------------------------------------
# 4. Embedding is unit-norm
# ---------------------------------------------------------------------------


class TestEmbeddingNorm:

    def test_unit_norm_dim4(self):
        axes = _load_axes()
        m = _measurer()
        material = next(a for a in axes if a.id == "material")
        for seed in range(10):
            audio = _make_audio(dim=4, seed=seed)
            t = m.measure(audio, material)
            norm = float(np.linalg.norm(t.embedding))
            assert abs(norm - 1.0) < 1e-9 or norm == pytest.approx(1.0, abs=1e-9), (
                f"embedding norm {norm} != 1.0 for seed={seed}"
            )

    def test_unit_norm_zero_audio(self):
        """Zero audio produces a zero embedding (edge case guard)."""
        axes = _load_axes()
        m = _measurer()
        material = next(a for a in axes if a.id == "material")
        audio = np.zeros(4)
        t = m.measure(audio, material)
        norm = float(np.linalg.norm(t.embedding))
        # For zero audio, the projection yields the zero vector; norm==0.
        assert norm == pytest.approx(0.0, abs=1e-12)

    def test_unit_norm_large_dim(self):
        axes = _load_axes()
        m = _measurer()
        material = next(a for a in axes if a.id == "material")
        audio = _make_audio(dim=64, seed=0)
        t = m.measure(audio, material)
        norm = float(np.linalg.norm(t.embedding))
        assert norm == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# 5. Presence detector: label is 0 or 1
# ---------------------------------------------------------------------------


class TestPresenceDetector:

    def test_positive_mean_gives_1(self):
        axes = _load_axes()
        m = _measurer()
        presence = next(a for a in axes if a.id == "presence")
        audio = np.ones(4)
        t = m.measure(audio, presence)
        assert t.label == 1

    def test_negative_mean_gives_0(self):
        axes = _load_axes()
        m = _measurer()
        presence = next(a for a in axes if a.id == "presence")
        audio = -np.ones(4)
        t = m.measure(audio, presence)
        assert t.label == 0


# ---------------------------------------------------------------------------
# 6. onset_timing_bin: label in [0, N_TIMING_BINS)
# ---------------------------------------------------------------------------


class TestOnsetTimingBin:

    def test_label_in_range(self):
        from foley_cw.axes import _N_TIMING_BINS
        axes = _load_axes()
        m = _measurer()
        timing = next(a for a in axes if a.id == "timing")
        rng = np.random.default_rng(0)
        for _ in range(50):
            audio = rng.standard_normal(4)
            t = m.measure(audio, timing)
            assert 0 <= t.label < _N_TIMING_BINS, f"bin out of range: {t.label}"

    def test_extreme_values_clamped(self):
        from foley_cw.axes import _N_TIMING_BINS
        axes = _load_axes()
        m = _measurer()
        timing = next(a for a in axes if a.id == "timing")
        # Very large positive first element
        audio_hi = np.array([1e9, 0.0, 0.0, 0.0])
        t_hi = m.measure(audio_hi, timing)
        assert t_hi.label == _N_TIMING_BINS - 1
        # Very large negative first element
        audio_lo = np.array([-1e9, 0.0, 0.0, 0.0])
        t_lo = m.measure(audio_lo, timing)
        assert t_lo.label == 0


# ---------------------------------------------------------------------------
# 7. binding_label: label is a string of the form "(s0,s1)"
# ---------------------------------------------------------------------------


class TestBindingLabel:

    def test_label_format(self):
        axes = _load_axes()
        m = _measurer()
        binding = next(a for a in axes if a.id == "binding")
        rng = np.random.default_rng(5)
        for _ in range(20):
            audio = rng.standard_normal(4)
            t = m.measure(audio, binding)
            assert isinstance(t.label, str)
            assert t.label.startswith("(") and t.label.endswith(")")

    def test_dim1_audio_uses_zero_for_second_element(self):
        axes = _load_axes()
        m = _measurer()
        binding = next(a for a in axes if a.id == "binding")
        audio = np.array([1.5])
        t = m.measure(audio, binding)
        # sign(1.5)=1, second element missing -> 0 -> sign(0)=0
        assert t.label == "(1,0)"

    def test_positive_positive(self):
        axes = _load_axes()
        m = _measurer()
        binding = next(a for a in axes if a.id == "binding")
        audio = np.array([2.0, 3.0])
        t = m.measure(audio, binding)
        assert t.label == "(1,1)"

    def test_negative_positive(self):
        axes = _load_axes()
        m = _measurer()
        binding = next(a for a in axes if a.id == "binding")
        audio = np.array([-1.0, 0.5])
        t = m.measure(audio, binding)
        assert t.label == "(-1,1)"


# ---------------------------------------------------------------------------
# 8. RealMeasurer: every measure raises NotImplementedError
# ---------------------------------------------------------------------------


class TestRealMeasurer:

    def test_all_measures_raise_not_implemented(self):
        axes = _load_axes()
        rm = RealMeasurer()
        audio = _make_audio(dim=4)
        for axis in axes:
            with pytest.raises(NotImplementedError) as exc_info:
                rm.measure(audio, axis)
            msg = str(exc_info.value)
            # Message must name the dependency
            assert "requires" in msg.lower() or "Phase 0" in msg, (
                f"axis {axis.id!r}: NotImplementedError message doesn't name dep: {msg!r}"
            )

    def test_real_measurer_error_names_dep_for_presence(self):
        axes = _load_axes()
        rm = RealMeasurer()
        presence = next(a for a in axes if a.id == "presence")
        with pytest.raises(NotImplementedError) as exc_info:
            rm.measure(np.zeros(4), presence)
        assert "librosa" in str(exc_info.value) or "torchaudio" in str(exc_info.value)

    def test_real_measurer_error_names_dep_for_class(self):
        axes = _load_axes()
        rm = RealMeasurer()
        class_axis = next(a for a in axes if a.id == "class")
        with pytest.raises(NotImplementedError) as exc_info:
            rm.measure(np.zeros(4), class_axis)
        assert "torch" in str(exc_info.value) or "tagger" in str(exc_info.value).lower()

    def test_real_measurer_error_names_dep_for_material(self):
        axes = _load_axes()
        rm = RealMeasurer()
        material = next(a for a in axes if a.id == "material")
        with pytest.raises(NotImplementedError) as exc_info:
            rm.measure(np.zeros(4), material)
        assert "clap" in str(exc_info.value).lower() or "torch" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 9. measure_self_target convenience wrapper
# ---------------------------------------------------------------------------


class TestMeasureSelfTarget:

    def test_delegates_to_measurer(self):
        axes = _load_axes()
        m = _measurer()
        audio = _make_audio(dim=4, seed=2)
        for axis in axes:
            t_direct = m.measure(audio, axis)
            t_via_wrapper = measure_self_target(audio, axis, m)
            if axis.kind is AxisKind.CATEGORICAL:
                assert t_direct.label == t_via_wrapper.label
            else:
                np.testing.assert_array_equal(t_direct.embedding, t_via_wrapper.embedding)

    def test_converts_to_float_array(self):
        """measure_self_target should work even if audio is passed as a list."""
        axes = _load_axes()
        m = _measurer()
        presence = next(a for a in axes if a.id == "presence")
        t = measure_self_target([1.0, 2.0, 3.0, 4.0], presence, m)
        assert t.kind is AxisKind.CATEGORICAL


# ---------------------------------------------------------------------------
# 10. Protocol check: SyntheticMeasurer implements Measurer
# ---------------------------------------------------------------------------


class TestProtocol:

    def test_synthetic_measurer_is_measurer(self):
        assert isinstance(SyntheticMeasurer(), Measurer)

    def test_real_measurer_is_measurer(self):
        assert isinstance(RealMeasurer(), Measurer)


# ---------------------------------------------------------------------------
# 11. Unknown measure raises ValueError
# ---------------------------------------------------------------------------


class TestUnknownMeasure:

    def test_unknown_measure_raises_value_error(self):
        m = _measurer()
        # Construct a dummy axis with an unregistered measure name.
        bad_axis = Axis(
            id="dummy",
            name="dummy",
            tier=AxisTier.TIER1,
            kind=AxisKind.CATEGORICAL,
            agreement=AgreementMetric.EXACT_MATCH,
            measure="nonexistent_measure_xyz",
        )
        with pytest.raises(ValueError, match="unknown measure name"):
            m.measure(_make_audio(), bad_axis)

    def test_real_unknown_measure_raises_value_error(self):
        rm = RealMeasurer()
        bad_axis = Axis(
            id="dummy",
            name="dummy",
            tier=AxisTier.TIER1,
            kind=AxisKind.CATEGORICAL,
            agreement=AgreementMetric.EXACT_MATCH,
            measure="nonexistent_measure_xyz",
        )
        with pytest.raises(ValueError):
            rm.measure(_make_audio(), bad_axis)


# ---------------------------------------------------------------------------
# 12. Integration test: measure on SyntheticGaussianFlow decoded audio
# ---------------------------------------------------------------------------


class TestIntegrationWithSyntheticBackend:

    def test_measure_on_final_state(self):
        """Measure on the decoded audio from a SyntheticGaussianFlow generation."""
        from foley_cw.score_sde import generate_trajectory
        from foley_cw.config import load_config

        cfg = load_config()
        backend = SyntheticGaussianFlow(dim=4)
        rng = np.random.default_rng(42)
        cond = SyntheticVideoCond(mu=np.array([1.0, -0.5, 0.3, -0.2]), video_id="test")

        traj = generate_trajectory(backend, cond, cfg.schedule, rng)
        audio = traj["audio"]

        m = _measurer()
        axes = cfg.axes
        for axis in axes:
            t = m.measure(audio, axis)
            assert t.kind is axis.kind
            assert t.axis_id == axis.id
            if axis.kind is AxisKind.CATEGORICAL:
                assert t.label is not None
            else:
                assert t.embedding is not None
                assert t.embedding.shape == (m._embed_dim,)

    def test_different_mus_can_give_different_presence(self):
        """Different video conditions (mu) -> potentially different self-targets."""
        backend = SyntheticGaussianFlow(dim=4)
        m = _measurer()
        axes_dict = {a.id: a for a in _load_axes()}
        presence = axes_dict["presence"]

        # Use ODE (alpha=0) so final state is deterministic given mu.
        from foley_cw.score_sde import generate_trajectory
        from foley_cw.config import load_config
        cfg = load_config()

        results = []
        for mu_val in [np.array([2.0, 2.0, 2.0, 2.0]), np.array([-2.0, -2.0, -2.0, -2.0])]:
            cond = SyntheticVideoCond(mu=mu_val, video_id="v")
            rng = np.random.default_rng(0)
            traj = generate_trajectory(backend, cond, cfg.schedule, rng)
            t = m.measure(traj["audio"], presence)
            results.append(t.label)
        # For large positive mu, mean(audio) > 0 => label=1; large negative => label=0.
        assert results[0] == 1
        assert results[1] == 0


# ---------------------------------------------------------------------------
# 13. SyntheticMeasurer preserves axis_id across construction-time projections
# ---------------------------------------------------------------------------


class TestAxisIdPreserved:

    def test_axis_id_in_result(self):
        axes = _load_axes()
        m = _measurer()
        audio = _make_audio()
        for axis in axes:
            t = m.measure(audio, axis)
            assert t.axis_id == axis.id


# ---------------------------------------------------------------------------
# 14. Embedding shape consistency
# ---------------------------------------------------------------------------


class TestEmbeddingShape:

    def test_embed_dim_default(self):
        from foley_cw.axes import _EMBED_DIM
        axes = _load_axes()
        m = SyntheticMeasurer()
        material = next(a for a in axes if a.id == "material")
        audio = _make_audio(dim=4)
        t = m.measure(audio, material)
        assert t.embedding.shape == (_EMBED_DIM,)

    def test_embed_dim_custom(self):
        axes = _load_axes()
        m = SyntheticMeasurer(embed_dim=4)
        material = next(a for a in axes if a.id == "material")
        audio = _make_audio(dim=4)
        t = m.measure(audio, material)
        assert t.embedding.shape == (4,)


# ---------------------------------------------------------------------------
# 15. Tagger label is an integer in [0, n_classes)
# ---------------------------------------------------------------------------


class TestTaggerLabel:

    def test_tagger_label_in_range(self):
        from foley_cw.axes import _N_TAGGER_CLASSES
        axes = _load_axes()
        m = _measurer()
        class_axis = next(a for a in axes if a.id == "class")
        rng = np.random.default_rng(99)
        for _ in range(30):
            audio = rng.standard_normal(4)
            t = m.measure(audio, class_axis)
            assert 0 <= t.label < _N_TAGGER_CLASSES, (
                f"tagger label {t.label} out of [0, {_N_TAGGER_CLASSES})"
            )

    def test_tagger_custom_classes(self):
        axes = _load_axes()
        m = SyntheticMeasurer(n_tagger_classes=5)
        class_axis = next(a for a in axes if a.id == "class")
        rng = np.random.default_rng(7)
        for _ in range(20):
            audio = rng.standard_normal(8)
            t = m.measure(audio, class_axis)
            assert 0 <= t.label < 5
