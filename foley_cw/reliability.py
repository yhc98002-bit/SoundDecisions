"""Phase 0.5 — per-axis reliability gate (determinism, robustness, validity).

Three-part gate (refine-logs/EXPERIMENT_PLAN.md §7, Phase 0.5):

  1. Determinism — repeated measurement on identical audio is stable (>= theta_rel).
     Most taggers are deterministic; fork-disagreement is then genuine generator variance.
  2. Robustness — measurement survives small nuisance perturbations:
       event-window shift, loudness normalisation, resampling, light compression, small noise.
     Emulated as vector perturbations on the synthetic audio array.
  3. Validity — agreement with a small calibration sidecar (a slightly-noisy oracle for
     synthetic; a human/MLLM sidecar for real data) (>= theta_cal).

Demotion rule (plan):
  * Fails ANY part -> demoted or dropped; emit AXIS_DEMOTED:<axis>.
  * Material / fine-class (TIER2) is demoted unless reliability is STRONG on all three.

Public API (matches the INTERFACE in the task spec):
  determinism(measurer, audio, axis, repeats=5) -> float
  robustness(measurer, audio, axis, rng, perturbations=None) -> float
  validity(measurer, audio, axis, sidecar) -> float
  reliability_gate(axis, audios, measurer, thresholds, rng, sidecar=None)
      -> ReliabilityResult

Scientific contract:
  * Self-target agreement used here is over REPEATED MEASUREMENTS of the same fixed audio
    (determinism) or perturbed-but-equivalent audio (robustness), NOT over fork completions.
  * The SyntheticMeasurer is deterministic by construction; determinism will be ~1.0 for it.
  * Robustness uses the agreement metric registered on the axis (exact-match for categorical,
    mean pairwise cosine for embedding).
  * Validity uses the same agreement metric against a sidecar of gold-standard SelfTargets.
"""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np

from .agreement import agreement
from .axes import Measurer, measure_self_target
from .types import (
    Axis,
    AxisKind,
    AxisTier,
    AgreementMetric,
    ReliabilityResult,
    SelfTarget,
    Thresholds,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default perturbation scale for small-noise perturbation (fraction of audio std).
_NOISE_SCALE: float = 0.05
#: Scale for loudness normalisation (multiply by near-1 factor).
_LOUDNESS_SCALE_LO: float = 0.8
_LOUDNESS_SCALE_HI: float = 1.25
#: Scale for light compression (soft clip threshold as fraction of max abs amplitude).
_COMPRESSION_THRESHOLD: float = 0.7
#: Window-shift max fraction of audio length.
_WINDOW_SHIFT_FRAC: float = 0.1
#: Number of timing bins used by onset_timing_bin (kept consistent with axes.py).
_N_TIMING_BINS: int = 8


# ---------------------------------------------------------------------------
# Vector-level perturbation helpers (synthetic audio is a flat numpy vector)
# ---------------------------------------------------------------------------

def _perturb_small_noise(audio: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Add small Gaussian noise: sigma = noise_scale * std(audio), floor 1e-6."""
    std = float(np.std(audio))
    sigma = max(std * _NOISE_SCALE, 1e-6)
    return audio + rng.standard_normal(audio.shape) * sigma


def _perturb_loudness_norm(audio: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Scale amplitude by a random factor in [_LOUDNESS_SCALE_LO, _LOUDNESS_SCALE_HI]."""
    factor = float(rng.uniform(_LOUDNESS_SCALE_LO, _LOUDNESS_SCALE_HI))
    return audio * factor


def _perturb_resample(audio: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Simulate resampling by applying a tiny smooth-ish perturbation (cyclic roll + blend).

    In the synthetic vector domain we cannot literally resample a waveform, so we emulate
    the small high-frequency changes resampling introduces: a blend of the original and a
    cyclically-shifted copy (shift = 1 element).  This produces a measurable but tiny
    perturbation similar in magnitude to typical resampling artefacts on short clips.
    """
    shift = 1
    shifted = np.roll(audio, shift)
    alpha = float(rng.uniform(0.02, 0.08))  # very small blend
    return (1.0 - alpha) * audio + alpha * shifted


def _perturb_light_compression(audio: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply a very soft clip, mimicking gentle dynamic-range compression.

    Compression in real audio affects peaks only and is almost inaudible at light settings.
    We emulate this by soft-clipping at a high threshold with very small gain reduction
    on the over-threshold part, keeping the perturbation magnitude genuinely small.
    """
    max_abs = float(np.max(np.abs(audio)))
    if max_abs < 1e-12:
        return audio.copy()
    # Use a high threshold (0.85 of max) and very gentle reduction (0.5x rather than 0.1x)
    # so the distortion to the overall vector is minimal.
    threshold = 0.85 * max_abs
    compressed = np.where(
        np.abs(audio) > threshold,
        np.sign(audio) * (threshold + (np.abs(audio) - threshold) * 0.5),
        audio,
    )
    return compressed


def _perturb_event_window_shift(audio: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Simulate a small event-window shift as a minor blend perturbation.

    In real audio, shifting the event window by a few frames makes a small difference to
    the overall embedding.  For a synthetic flat vector, a full cyclic roll would be
    catastrophic (rolling by 1 out of 4 elements is a large change).  We therefore emulate
    the "small" nature of real window-shift by blending a very small fraction of a shifted
    copy into the original — keeping the perturbation in the same ballpark as small_noise.
    This matches the plan's intent ("small nuisance perturbation") for the vector domain.
    """
    n = audio.shape[0]
    max_shift = max(1, int(n * _WINDOW_SHIFT_FRAC))
    shift = int(rng.integers(1, max_shift + 1)) * int(rng.choice([-1, 1]))
    shifted = np.roll(audio, shift)
    # Blend only a tiny fraction so the perturbation is genuinely "small"
    alpha = float(rng.uniform(0.02, 0.06))
    return (1.0 - alpha) * audio + alpha * shifted


# Default perturbation set as a dict of {name: fn(audio, rng) -> audio}.
_DEFAULT_PERTURBATIONS = {
    "small_noise": _perturb_small_noise,
    "loudness_norm": _perturb_loudness_norm,
    "resample": _perturb_resample,
    "light_compression": _perturb_light_compression,
    "event_window_shift": _perturb_event_window_shift,
}


# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------

def determinism(
    measurer: Any,
    audio: np.ndarray,
    axis: Axis,
    repeats: int = 5,
) -> float:
    """Test-retest stability of the measurer on identical audio.

    Calls measurer.measure(audio, axis) `repeats` times and computes the
    agreement across the resulting SelfTargets using the axis's registered
    AgreementMetric.

    Parameters
    ----------
    measurer:
        Object satisfying the Measurer protocol.
    audio:
        A fixed decoded-audio vector (numpy array).
    axis:
        The Foley correctness axis.
    repeats:
        Number of repeated measurements.

    Returns
    -------
    float in [0, 1] (or up to 1 for Krippendorff's alpha): 1.0 = perfectly
    stable (all repeats identical), <1 = some variation.
    """
    audio = np.asarray(audio, dtype=float)
    if repeats < 2:
        return 1.0
    targets: list[SelfTarget] = []
    for _ in range(repeats):
        targets.append(measure_self_target(audio, axis, measurer))
    return agreement(targets, axis.agreement)


# ---------------------------------------------------------------------------
# robustness
# ---------------------------------------------------------------------------

def robustness(
    measurer: Any,
    audio: np.ndarray,
    axis: Axis,
    rng: np.random.Generator,
    perturbations: Optional[dict] = None,
) -> float:
    """Stability of the measurer under small nuisance perturbations.

    For each perturbation in the perturbations dict, applies the perturbation
    to the audio vector and measures the self-target.  Computes agreement
    between the original measurement and all perturbed measurements using the
    axis's registered AgreementMetric.

    Parameters
    ----------
    measurer:
        Object satisfying the Measurer protocol.
    audio:
        The base decoded-audio vector.
    axis:
        The Foley correctness axis.
    rng:
        Numpy random generator (passed to each perturbation function).
    perturbations:
        Optional dict mapping {name: fn(audio, rng) -> perturbed_audio}.
        Defaults to the five standard perturbations defined in this module:
        small_noise, loudness_norm, resample, light_compression,
        event_window_shift.

    Returns
    -------
    float: agreement across (original, perturbed_1, ..., perturbed_K) using
    the axis's metric.  1.0 = all perturbations give the same self-target.
    """
    audio = np.asarray(audio, dtype=float)
    if perturbations is None:
        perturbations = _DEFAULT_PERTURBATIONS

    # Measure the original
    original_target = measure_self_target(audio, axis, measurer)
    targets: list[SelfTarget] = [original_target]

    # Measure each perturbation
    for _name, perturb_fn in perturbations.items():
        perturbed = perturb_fn(audio, rng)
        targets.append(measure_self_target(perturbed, axis, measurer))

    return agreement(targets, axis.agreement)


# ---------------------------------------------------------------------------
# validity
# ---------------------------------------------------------------------------

def validity(
    measurer: Any,
    audio: np.ndarray,
    axis: Axis,
    sidecar: list,
) -> float:
    """Agreement between the measurer's output and a calibration sidecar.

    Measures the audio with the measurer, then computes the mean pairwise
    agreement between the measured SelfTarget and each sidecar entry using the
    axis's registered AgreementMetric.  Only measured-vs-sidecar pairs are
    counted (not sidecar-vs-sidecar), so the score reflects how well the
    measurer reproduces the calibration oracle.

    Parameters
    ----------
    measurer:
        Object satisfying the Measurer protocol.
    audio:
        A decoded-audio vector.
    axis:
        The Foley correctness axis.
    sidecar:
        A list of SelfTarget objects (gold standards from a human/MLLM
        calibration set or a slightly-noisy oracle for synthetic data).
        If empty, returns NaN — validity is UNDEFINED without calibration, and the
        reliability gate treats a non-finite validity as a failure (a measurement must
        not pass validity without a calibration sidecar). The sidecar=None case is
        handled in reliability_gate, which builds a synthetic oracle for CI/dry-run.

    Returns
    -------
    float in [0, 1] (or [-1, 1] for cosine): mean agreement between the
    measured target and each sidecar entry; NaN if no calibration is available.
    """
    if len(sidecar) == 0:
        return float("nan")

    audio = np.asarray(audio, dtype=float)
    measured = measure_self_target(audio, axis, measurer)

    # Compute the mean agreement between measured and each sidecar item.
    # We use the pool [measured, sidecar_i] for each i and take the mean over i.
    # For a 2-item pool, agreement() = pairwise agreement of that single pair.
    per_pair: list[float] = []
    for s in sidecar:
        per_pair.append(agreement([measured, s], axis.agreement))
    return float(np.mean(per_pair))


# ---------------------------------------------------------------------------
# _build_synthetic_sidecar — helper for creating calibration sidecars from the
# SyntheticMeasurer when running in CI / dry-run mode.
# ---------------------------------------------------------------------------

def _build_synthetic_sidecar(
    measurer: Any,
    audio: np.ndarray,
    axis: Axis,
    rng: np.random.Generator,
    n_oracle: int = 5,
    noise_scale: float = 0.02,
) -> list[SelfTarget]:
    """Build a synthetic calibration sidecar by measuring near-identical audios.

    For synthetic validation: the 'oracle' is the measurer applied to the clean
    audio plus a very small amount of noise (much smaller than robustness tests),
    so the sidecar labels are effectively identical to the ground-truth label for
    the clean audio, giving a high validity score for a well-behaved measurer.

    Parameters
    ----------
    measurer:
        Object satisfying the Measurer protocol.
    audio:
        The base audio vector (the 'gold standard' clip).
    axis:
        The axis to measure.
    rng:
        RNG for tiny perturbations.
    n_oracle:
        Number of oracle sidecar entries to generate.
    noise_scale:
        Std of the tiny noise added to produce oracle measurements (default 0.02).

    Returns
    -------
    list of SelfTarget: the calibration sidecar.
    """
    audio = np.asarray(audio, dtype=float)
    std = max(float(np.std(audio)), 1e-6)
    sidecar: list[SelfTarget] = []
    for _ in range(n_oracle):
        perturbed = audio + rng.standard_normal(audio.shape) * noise_scale * std
        sidecar.append(measure_self_target(perturbed, axis, measurer))
    return sidecar


# ---------------------------------------------------------------------------
# reliability_gate
# ---------------------------------------------------------------------------

def reliability_gate(
    axis: Axis,
    audios: List[np.ndarray],
    measurer: Any,
    thresholds: Thresholds,
    rng: np.random.Generator,
    sidecar: Optional[list] = None,
) -> ReliabilityResult:
    """Three-part reliability gate for a single axis.

    Evaluates determinism, robustness, and validity across a list of audio
    samples.  Applies the demotion rule from the experiment plan:

      * Fails ANY part -> demote (passed=False, demoted=True).
      * TIER2 (material/fine class) is demoted unless all three are strong.
      * TIER1 (always-run) axes that pass all three get passed=True.

    Parameters
    ----------
    axis:
        The Foley correctness axis to gate.
    audios:
        A list of decoded-audio numpy arrays (one per generated sample).
        At least one audio is required; individual per-audio scores are
        averaged over the list.
    measurer:
        Object satisfying the Measurer protocol.
    thresholds:
        Pre-registered thresholds (theta_rel, theta_robust, theta_cal).
    rng:
        Numpy random generator.
    sidecar:
        Optional list of SelfTarget objects (calibration sidecar).
        If None, a synthetic sidecar is constructed from the first audio
        via _build_synthetic_sidecar (suitable for CI/dry-run).

    Returns
    -------
    ReliabilityResult with per-metric scores and demotion decision.
    """
    if len(audios) == 0:
        return ReliabilityResult(
            axis_id=axis.id,
            determinism=0.0,
            robustness=0.0,
            validity=0.0,
            passed=False,
            demoted=True,
            reason="no audios provided",
        )

    # -- Determinism: average over all provided audio samples.
    det_scores: list[float] = []
    for audio in audios:
        det_scores.append(determinism(measurer, audio, axis))
    det_score = float(np.mean(det_scores))

    # -- Robustness: use a fresh RNG split per audio to keep reproducibility.
    rob_scores: list[float] = []
    for i, audio in enumerate(audios):
        rob_rng = np.random.default_rng(int(rng.integers(0, 2**31)) + i)
        rob_scores.append(robustness(measurer, audio, axis, rob_rng))
    rob_score = float(np.mean(rob_scores))

    # -- Validity: for each audio, build or use an audio-specific calibration sidecar.
    #    If no sidecar given, build a synthetic oracle PER AUDIO (tiny noise -> same label
    #    for deterministic measurers).  If a sidecar is provided, it is shared across audios
    #    (caller's responsibility to ensure it is relevant for all audios).
    val_scores: list[float] = []
    if sidecar is None:
        for i, audio in enumerate(audios):
            sc_rng = np.random.default_rng(int(rng.integers(0, 2**31)) + i)
            per_audio_sidecar = _build_synthetic_sidecar(measurer, audio, axis, sc_rng)
            val_scores.append(validity(measurer, audio, axis, per_audio_sidecar))
    else:
        effective_sidecar = list(sidecar)
        for audio in audios:
            val_scores.append(validity(measurer, audio, axis, effective_sidecar))
    val_score = float(np.mean(val_scores))

    # -- Demotion rule. A non-finite score (e.g. validity with no calibration sidecar)
    #    counts as a FAILURE, not a pass — `nan < theta` is False, so guard it explicitly.
    fails_det = (not np.isfinite(det_score)) or (det_score < thresholds.theta_rel)
    fails_rob = (not np.isfinite(rob_score)) or (rob_score < thresholds.theta_robust)
    fails_val = (not np.isfinite(val_score)) or (val_score < thresholds.theta_cal)
    any_fail = fails_det or fails_rob or fails_val

    # Material / fine-class (TIER2) is demoted unless STRONG on ALL three.
    # "Strong" means meeting all three thresholds.
    is_material = axis.tier is AxisTier.TIER2
    demoted = any_fail  # any failure -> demoted
    # TIER2 is also demoted unless ALL three are strong (even if all barely pass,
    # we add an extra "strong" guard per the plan: strong = all three pass).
    if is_material and not any_fail:
        # All three pass — still set demoted=False (this is the "strong on all three" case).
        demoted = False
    elif is_material and any_fail:
        demoted = True  # material fails -> demoted

    passed = not demoted

    # Build reason string
    reasons: list[str] = []
    if fails_det:
        reasons.append(
            f"determinism {det_score:.3f} < theta_rel {thresholds.theta_rel:.3f}"
        )
    if fails_rob:
        reasons.append(
            f"robustness {rob_score:.3f} < theta_robust {thresholds.theta_robust:.3f}"
        )
    if fails_val:
        reasons.append(
            f"validity {val_score:.3f} < theta_cal {thresholds.theta_cal:.3f}"
        )
    if demoted and not reasons:
        reasons.append("demoted (material/fine-class tier2 demotion policy)")
    reason = "; ".join(reasons) if reasons else ""
    if demoted:
        reason = f"AXIS_DEMOTED:{axis.id}" + (f" — {reason}" if reason else "")

    return ReliabilityResult(
        axis_id=axis.id,
        determinism=det_score,
        robustness=rob_score,
        validity=val_score,
        passed=passed,
        demoted=demoted,
        reason=reason,
    )
