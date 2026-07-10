"""Tests for foley_cw/real_perturbations.py — real-waveform reliability perturbations.

CPU-only, no network, no GPU.  scipy is optional (resample_roundtrip tests skip
without it); everything else is numpy-only.

Key contracts checked here:
  * Each of the five perturbations is shape-preserving, float32, all-finite, and
    deterministic given the passed rng (manual §3.3 "small nuisance perturbations").
  * event_window_shift actually moves the onset by ~100 ms with ZERO padding
    (not cyclic).
  * small_noise sits within ±3 dB of the pre-registered 40 dB SNR; silent input is
    returned unchanged.
  * resample_roundtrip correlates > 0.9 with the original.
  * light_compression renormalizes to 0.95x the original peak and preserves signs.
  * REAL_PERTURBATIONS matches the reliability.robustness perturbation-dict contract
    ({name: fn(audio, rng) -> audio}); we call each fn directly rather than running
    robustness() because SyntheticMeasurer operates on tiny vectors, not waveforms.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from foley_cw.real_perturbations import (
    REAL_PERTURBATIONS,
    SAMPLE_RATE_HZ,
    event_window_shift,
    light_compression,
    loudness_norm,
    resample_roundtrip,
    small_noise,
)

SR = SAMPLE_RATE_HZ
SHIFT_SAMPLES = int(round(0.1 * SR))  # ±100 ms


def _needs_scipy(name: str) -> None:
    if name == "resample_roundtrip":
        pytest.importorskip("scipy")


@pytest.fixture
def sine_burst() -> np.ndarray:
    """1.2 s clip at 16 kHz: 440 Hz sine burst from 0.4 s to 0.7 s, float32."""
    n = int(1.2 * SR)
    t = np.arange(n) / SR
    x = np.zeros(n, dtype=np.float32)
    lo, hi = int(0.4 * SR), int(0.7 * SR)
    x[lo:hi] = (0.5 * np.sin(2.0 * np.pi * 440.0 * t[lo:hi])).astype(np.float32)
    return x


def _onset_index(x: np.ndarray, threshold: float = 0.05) -> int:
    """First sample index whose absolute amplitude exceeds the threshold."""
    above = np.abs(x) > threshold
    assert np.any(above), "no onset found above threshold"
    return int(np.argmax(above))


# --------------------------------------------------------------------------------------
# Shared contract: shape, dtype, finiteness, determinism
# --------------------------------------------------------------------------------------

class TestSharedContract:
    @pytest.mark.parametrize("name", sorted(REAL_PERTURBATIONS))
    def test_shape_dtype_finite(self, name, sine_burst):
        _needs_scipy(name)
        fn = REAL_PERTURBATIONS[name]
        out = fn(sine_burst, np.random.default_rng(0))
        assert isinstance(out, np.ndarray)
        assert out.shape == sine_burst.shape
        assert out.dtype == np.float32
        assert np.all(np.isfinite(out))

    @pytest.mark.parametrize("name", sorted(REAL_PERTURBATIONS))
    def test_deterministic_given_rng(self, name, sine_burst):
        _needs_scipy(name)
        fn = REAL_PERTURBATIONS[name]
        out1 = fn(sine_burst, np.random.default_rng(7))
        out2 = fn(sine_burst, np.random.default_rng(7))
        np.testing.assert_array_equal(out1, out2)

    @pytest.mark.parametrize("name", sorted(REAL_PERTURBATIONS))
    def test_input_not_mutated(self, name, sine_burst):
        _needs_scipy(name)
        fn = REAL_PERTURBATIONS[name]
        before = sine_burst.copy()
        fn(sine_burst, np.random.default_rng(0))
        np.testing.assert_array_equal(sine_burst, before)

    @pytest.mark.parametrize("name", sorted(REAL_PERTURBATIONS))
    def test_rejects_non_1d(self, name):
        fn = REAL_PERTURBATIONS[name]
        with pytest.raises(ValueError):
            fn(np.zeros((2, 100), dtype=np.float32), np.random.default_rng(0))


# --------------------------------------------------------------------------------------
# event_window_shift
# --------------------------------------------------------------------------------------

class TestEventWindowShift:
    def test_moves_onset_by_100ms(self, sine_burst):
        onset_orig = _onset_index(sine_burst)
        for seed in range(6):
            out = event_window_shift(sine_burst, np.random.default_rng(seed))
            onset_shifted = _onset_index(out)
            diff = abs(onset_shifted - onset_orig)
            assert abs(diff - SHIFT_SAMPLES) <= 2, (
                f"seed={seed}: onset moved by {diff} samples, expected ~{SHIFT_SAMPLES}"
            )

    def test_both_signs_occur(self, sine_burst):
        onset_orig = _onset_index(sine_burst)
        signs = set()
        for seed in range(16):
            out = event_window_shift(sine_burst, np.random.default_rng(seed))
            signs.add(int(np.sign(_onset_index(out) - onset_orig)))
        assert signs == {-1, 1}, f"expected both shift signs over 16 seeds, got {signs}"

    def test_zero_pads_not_cyclic(self):
        x = np.ones(SR, dtype=np.float32)
        out = event_window_shift(x, np.random.default_rng(0))
        # Exactly SHIFT_SAMPLES zeros, contiguous at one end (cyclic roll would
        # produce no zeros at all on an all-ones input).
        assert int(np.count_nonzero(out == 0.0)) == SHIFT_SAMPLES
        assert np.all(out[:SHIFT_SAMPLES] == 0.0) or np.all(out[-SHIFT_SAMPLES:] == 0.0)

    def test_shifted_content_preserved(self, sine_burst):
        out = event_window_shift(sine_burst, np.random.default_rng(1))
        # Whatever survived the shift must be an exact copy of original samples;
        # the shift direction is recovered from the onset (leading silence makes
        # the zero-padded region ambiguous on its own).
        if _onset_index(out) > _onset_index(sine_burst):
            np.testing.assert_array_equal(out[SHIFT_SAMPLES:], sine_burst[:-SHIFT_SAMPLES])
            assert np.all(out[:SHIFT_SAMPLES] == 0.0)
        else:
            np.testing.assert_array_equal(out[:-SHIFT_SAMPLES], sine_burst[SHIFT_SAMPLES:])
            assert np.all(out[-SHIFT_SAMPLES:] == 0.0)


# --------------------------------------------------------------------------------------
# loudness_norm
# --------------------------------------------------------------------------------------

class TestLoudnessNorm:
    def test_factor_within_pm_2db(self, sine_burst):
        lo, hi = 10.0 ** (-2.0 / 20.0), 10.0 ** (2.0 / 20.0)
        rms_in = float(np.sqrt(np.mean(sine_burst.astype(np.float64) ** 2)))
        for seed in range(8):
            out = loudness_norm(sine_burst, np.random.default_rng(seed))
            rms_out = float(np.sqrt(np.mean(out.astype(np.float64) ** 2)))
            ratio = rms_out / rms_in
            assert lo - 1e-4 <= ratio <= hi + 1e-4, (
                f"seed={seed}: loudness factor {ratio:.4f} outside [{lo:.4f}, {hi:.4f}]"
            )

    def test_pure_scaling(self, sine_burst):
        out = loudness_norm(sine_burst, np.random.default_rng(3))
        nz = np.abs(sine_burst) > 1e-4
        ratios = out[nz].astype(np.float64) / sine_burst[nz].astype(np.float64)
        assert np.allclose(ratios, ratios[0], rtol=1e-4), "loudness_norm must be a uniform scale"


# --------------------------------------------------------------------------------------
# resample_roundtrip
# --------------------------------------------------------------------------------------

class TestResampleRoundtrip:
    def test_correlation_above_0_9(self, sine_burst):
        pytest.importorskip("scipy")
        out = resample_roundtrip(sine_burst, np.random.default_rng(0))
        corr = float(np.corrcoef(sine_burst.astype(np.float64), out.astype(np.float64))[0, 1])
        assert corr > 0.9, f"roundtrip correlation {corr:.4f} should exceed 0.9"

    def test_changes_audio(self, sine_burst):
        pytest.importorskip("scipy")
        out = resample_roundtrip(sine_burst, np.random.default_rng(0))
        assert not np.array_equal(out, sine_burst), "roundtrip should perturb the waveform"

    def test_length_preserved_for_awkward_n(self):
        pytest.importorskip("scipy")
        # n not divisible by 320: trim/pad path must still return the input length.
        rng_audio = np.random.default_rng(5)
        x = rng_audio.standard_normal(SR + 123).astype(np.float32) * 0.1
        out = resample_roundtrip(x, np.random.default_rng(0))
        assert out.shape == x.shape


# --------------------------------------------------------------------------------------
# light_compression
# --------------------------------------------------------------------------------------

class TestLightCompression:
    def test_peak_renormalized_to_0_95(self, sine_burst):
        out = light_compression(sine_burst, np.random.default_rng(0))
        peak_in = float(np.max(np.abs(sine_burst)))
        peak_out = float(np.max(np.abs(out)))
        assert peak_out == pytest.approx(0.95 * peak_in, rel=1e-3)

    def test_signs_preserved(self, sine_burst):
        out = light_compression(sine_burst, np.random.default_rng(0))
        nz = np.abs(sine_burst) > 1e-6
        assert np.all(np.sign(out[nz]) == np.sign(sine_burst[nz]))

    def test_compresses_dynamics(self, sine_burst):
        """Quiet samples get more relative gain than peaks (soft-knee behaviour)."""
        out = light_compression(sine_burst, np.random.default_rng(0))
        quiet = (np.abs(sine_burst) > 0.01) & (np.abs(sine_burst) < 0.1)
        loud = np.abs(sine_burst) > 0.45
        assert np.any(quiet) and np.any(loud)
        gain_quiet = float(np.mean(np.abs(out[quiet]) / np.abs(sine_burst[quiet])))
        gain_loud = float(np.mean(np.abs(out[loud]) / np.abs(sine_burst[loud])))
        assert gain_quiet > gain_loud

    def test_silent_input_unchanged(self):
        x = np.zeros(1000, dtype=np.float32)
        out = light_compression(x, np.random.default_rng(0))
        np.testing.assert_array_equal(out, x)


# --------------------------------------------------------------------------------------
# small_noise
# --------------------------------------------------------------------------------------

class TestSmallNoise:
    def test_snr_within_3db_of_40(self, sine_burst):
        for seed in range(4):
            out = small_noise(sine_burst, np.random.default_rng(seed))
            noise = out.astype(np.float64) - sine_burst.astype(np.float64)
            rms_sig = float(np.sqrt(np.mean(sine_burst.astype(np.float64) ** 2)))
            rms_noise = float(np.sqrt(np.mean(noise**2)))
            snr_db = 20.0 * np.log10(rms_sig / rms_noise)
            assert abs(snr_db - 40.0) <= 3.0, f"seed={seed}: SNR {snr_db:.2f} dB not within 40±3"

    def test_changes_audio(self, sine_burst):
        out = small_noise(sine_burst, np.random.default_rng(0))
        assert not np.array_equal(out, sine_burst)

    def test_silent_input_guard(self):
        x = np.zeros(1000, dtype=np.float32)
        out = small_noise(x, np.random.default_rng(0))
        np.testing.assert_array_equal(out, x)
        assert out.dtype == np.float32


# --------------------------------------------------------------------------------------
# Perturbation-dict contract (drop-in for reliability.robustness)
# --------------------------------------------------------------------------------------

class TestDictContract:
    def test_exactly_the_five_manual_perturbations(self):
        assert set(REAL_PERTURBATIONS) == {
            "event_window_shift",
            "loudness_norm",
            "resample_roundtrip",
            "light_compression",
            "small_noise",
        }

    def test_all_values_callable_with_audio_rng(self):
        for name, fn in REAL_PERTURBATIONS.items():
            assert callable(fn), f"{name} must be callable"
            params = list(inspect.signature(fn).parameters)
            assert len(params) == 2, (
                f"{name} must accept (audio, rng) like reliability's perturbation fns"
            )

    def test_robustness_iteration_pattern(self, sine_burst):
        """Emulate the exact loop reliability.robustness runs over the dict."""
        pytest.importorskip("scipy")
        rng = np.random.default_rng(0)
        for _name, perturb_fn in REAL_PERTURBATIONS.items():
            perturbed = perturb_fn(sine_burst, rng)
            assert isinstance(perturbed, np.ndarray)
            assert perturbed.shape == sine_burst.shape
            assert np.all(np.isfinite(perturbed))

    def test_module_importable_with_numpy_only(self):
        """Module import must not require scipy (lazy import inside the function)."""
        import foley_cw.real_perturbations  # noqa: F401
