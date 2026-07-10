"""Real-waveform reliability perturbations (manual §3.3 robustness; numpy core).

Serves manual §3.3 (experiment/LONG_RANGE_EXPERIMENT_PLAN.md, Phase 0.5): robustness
under "event-window shift, loudness normalization, resampling, light compression,
small noise" measured on REAL 16 kHz mono waveforms — the real-path counterpart of
the tiny-vector emulations in foley_cw.reliability._DEFAULT_PERTURBATIONS.

Drop-in contract (identical to reliability.robustness(..., perturbations=...)):
  REAL_PERTURBATIONS maps name -> fn(audio, rng) -> perturbed_audio, where rng is a
  numpy Generator passed by the caller.  Every fn is:
    * deterministic given the passed rng,
    * shape-preserving on a 1-D mono float waveform at SAMPLE_RATE_HZ = 16000,
    * float32-out with all-finite values.

scipy (resample_roundtrip only) is imported lazily so this module stays importable
on a numpy-only environment.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

# ---------------------------------------------------------------------------
# Constants (perturbation magnitudes are pre-registered "small nuisance" sizes)
# ---------------------------------------------------------------------------

#: All real-path waveforms are 16 kHz mono (MMAudio small_16k / Cnn14_16k native).
SAMPLE_RATE_HZ: int = 16000
#: Event-window shift magnitude in seconds (±100 ms).
_SHIFT_SECONDS: float = 0.1
#: Loudness-normalization gain range in dB (factor in [10**(-2/20), 10**(+2/20)]).
_LOUDNESS_MAX_DB: float = 2.0
#: Small-noise SNR in dB relative to signal RMS (noise 40 dB below signal).
_SNR_DB: float = 40.0
#: tanh soft-knee drive for light compression (near-linear at low level).
_COMP_DRIVE: float = 1.5
#: Post-compression peak as a fraction of the original peak.
_COMP_PEAK_FRAC: float = 0.95
#: RMS below this counts as silence (no noise added; avoids 0*inf and dividing by ~0).
_SILENCE_RMS_FLOOR: float = 1e-8

#: Signature shared with reliability._DEFAULT_PERTURBATIONS entries.
PerturbationFn = Callable[[np.ndarray, np.random.Generator], np.ndarray]


def _as_wave(audio: np.ndarray) -> np.ndarray:
    """Validate and convert to a 1-D float32 mono waveform."""
    a = np.asarray(audio, dtype=np.float32)
    if a.ndim != 1:
        raise ValueError(
            f"real perturbations expect a 1-D mono waveform; got shape {a.shape}"
        )
    return a


# ---------------------------------------------------------------------------
# The five perturbations
# ---------------------------------------------------------------------------

def event_window_shift(audio: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Shift the waveform by ±100 ms (sign from rng) with ZERO padding (not cyclic).

    Cyclic roll would wrap clip-end content to the start and create a phantom onset;
    zero padding keeps the perturbation a pure window shift.
    """
    a = _as_wave(audio)
    n = a.shape[0]
    shift = int(round(_SHIFT_SECONDS * SAMPLE_RATE_HZ))
    if n == 0 or shift == 0:
        return a.copy()
    shift = min(shift, n)
    sign = 1 if int(rng.integers(0, 2)) == 1 else -1
    out = np.zeros_like(a)
    if sign > 0:
        out[shift:] = a[: n - shift]
    else:
        out[: n - shift] = a[shift:]
    return out


def loudness_norm(audio: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Scale by a gain factor drawn in [10**(-2/20), 10**(+2/20)] (uniform in dB)."""
    a = _as_wave(audio)
    gain_db = float(rng.uniform(-_LOUDNESS_MAX_DB, _LOUDNESS_MAX_DB))
    factor = 10.0 ** (gain_db / 20.0)
    return (a * factor).astype(np.float32)


def resample_roundtrip(audio: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Polyphase resample 16000 -> 22050 -> 16000 Hz; trim/pad to the input length.

    16000 * 441/320 = 22050 exactly, so the roundtrip uses the rational factors
    441/320 then 320/441.  rng is unused (the roundtrip is deterministic) but kept
    for the perturbation-dict contract.
    """
    del rng  # contract-only parameter
    a = _as_wave(audio)
    n = a.shape[0]
    if n == 0:
        return a.copy()
    try:
        from scipy.signal import resample_poly
    except ImportError as exc:
        raise ImportError(
            "resample_roundtrip requires scipy: pip install scipy"
        ) from exc
    up = resample_poly(a.astype(np.float64), 441, 320)
    down = resample_poly(up, 320, 441)
    out = np.zeros(n, dtype=np.float32)
    m = min(n, down.shape[0])
    out[:m] = down[:m].astype(np.float32)
    return out


def light_compression(audio: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Soft-knee tanh compression, renormalized to 0.95x the original peak.

    y = tanh(drive * a / peak) is near-linear for small samples and saturates peaks;
    renormalizing to peak*0.95 makes the net effect "quiet parts up relative to
    peaks", i.e. gentle dynamic-range compression.  rng is unused (deterministic)
    but kept for the perturbation-dict contract.
    """
    del rng  # contract-only parameter
    a = _as_wave(audio)
    peak = float(np.max(np.abs(a))) if a.size else 0.0
    if peak < 1e-12:
        return a.copy()
    y = np.tanh(_COMP_DRIVE * (a.astype(np.float64) / peak))
    y_peak = float(np.max(np.abs(y)))
    out = y * (_COMP_PEAK_FRAC * peak / y_peak)
    return out.astype(np.float32)


def small_noise(audio: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Add white Gaussian noise 40 dB below the signal RMS (i.e. SNR = 40 dB).

    Silent input (RMS below the floor) is returned unchanged: scaling noise by a
    near-zero RMS would make the perturbation vacuous, and SNR is undefined.
    """
    a = _as_wave(audio)
    rms = float(np.sqrt(np.mean(np.square(a, dtype=np.float64)))) if a.size else 0.0
    if rms < _SILENCE_RMS_FLOOR:
        return a.copy()
    noise_rms = rms * 10.0 ** (-_SNR_DB / 20.0)
    noise = rng.standard_normal(a.shape) * noise_rms
    return (a.astype(np.float64) + noise).astype(np.float32)


# ---------------------------------------------------------------------------
# Drop-in dict for reliability.robustness(..., perturbations=REAL_PERTURBATIONS)
# ---------------------------------------------------------------------------

REAL_PERTURBATIONS: dict[str, PerturbationFn] = {
    "event_window_shift": event_window_shift,
    "loudness_norm": loudness_norm,
    "resample_roundtrip": resample_roundtrip,
    "light_compression": light_compression,
    "small_noise": small_noise,
}
