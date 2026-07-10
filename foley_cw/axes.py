"""Per-axis self-target measurement registry (foley_cw/axes.py).

This module provides:
  * The ``Measurer`` protocol — measure(audio, axis) -> SelfTarget.
  * ``SyntheticMeasurer`` — CPU-only, deterministic functions of the audio VECTOR.
    Used in dry-runs and CI against SyntheticGaussianFlow.  Every rule is a pure
    function of the numpy array; no tagger/librosa/torch dependency.
  * ``RealMeasurer`` — same interface; every method raises NotImplementedError
    naming the dependency.  Only lazy-imported inside the method bodies.
  * ``measure_self_target(audio, axis, measurer)`` — thin convenience wrapper.

MEASURE → AxisKind mapping (must be consistent with configs/axes.json):
  presence_detector   → CATEGORICAL  int(mean(audio) > 0)
  onset_timing_bin    → CATEGORICAL  small int bin of audio[0]
  audio_tagger_top1   → CATEGORICAL  argmax of seeded fixed projection W @ audio
  audio_embedding     → EMBEDDING    unit-normed audio (via seeded fixed projection)
  binding_label       → CATEGORICAL  tuple (sign(audio[0]), sign(audio[1])) as str
  seed_predictability → handled elsewhere (SEPARATE axis; not a window)

The ``audio_tagger_top1`` projection matrix W (shape n_classes × dim) and the
``audio_embedding`` projection matrix P (shape embed_dim × dim) are stored on the
SyntheticMeasurer instance; they are initialised from a seeded RNG at construction
time and NEVER change, guaranteeing pure-function behaviour on the same instance.

Per the plan (refine-logs/EXPERIMENT_PLAN.md §0.5), each measure must be:
  - Deterministic (test-retest on identical audio is stable).
  - Robust (survives small nuisance perturbations on the REAL path).
  - Valid (agreement with a calibration sidecar — wired in Phase 0 for RealMeasurer).
"""

from __future__ import annotations

from typing import Any, Optional, Protocol, runtime_checkable

import numpy as np

from .types import Axis, AxisKind, SelfTarget

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

#: Default number of tagger output classes for SyntheticMeasurer.
_N_TAGGER_CLASSES: int = 16
#: Default output dimension for the synthetic embedding projection.
_EMBED_DIM: int = 8
#: Number of onset-timing bins for onset_timing_bin.
_N_TIMING_BINS: int = 8
#: Default seed used when constructing SyntheticMeasurer.
_DEFAULT_SEED: int = 42

# ---------------------------------------------------------------------------
# Measurer Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Measurer(Protocol):
    """Protocol: given decoded audio and an Axis, return the SelfTarget."""

    def measure(self, audio: np.ndarray, axis: Axis) -> SelfTarget:
        ...


# ---------------------------------------------------------------------------
# SyntheticMeasurer — deterministic CPU rules operating on the audio vector
# ---------------------------------------------------------------------------


class SyntheticMeasurer:
    """Deterministic self-target measurer for use with SyntheticGaussianFlow.

    All rules are pure functions of the audio numpy vector.  The projection
    matrices used for audio_tagger_top1 and audio_embedding are seeded at
    construction time and stored as instance attributes so that two calls to
    ``measure`` with the *same* audio vector always return equal SelfTargets.

    Parameters
    ----------
    seed:
        RNG seed for the fixed projection matrices.
    n_tagger_classes:
        Number of discrete classes produced by audio_tagger_top1.
    embed_dim:
        Output dimension for audio_embedding.
    """

    def __init__(
        self,
        seed: int = _DEFAULT_SEED,
        n_tagger_classes: int = _N_TAGGER_CLASSES,
        embed_dim: int = _EMBED_DIM,
    ) -> None:
        self._seed = int(seed)
        self._n_classes = int(n_tagger_classes)
        self._embed_dim = int(embed_dim)

        rng = np.random.default_rng(self._seed)
        # W: (n_classes, dim_placeholder) — extended lazily in _tagger_projection
        # to match the actual audio dimension.  We store a large seed-derived base
        # matrix and slice / extend as needed.  This guarantees the same label for
        # the same audio regardless of when we first see the dimension.
        self._tagger_base: np.ndarray = rng.standard_normal((self._n_classes, 256))
        self._embed_base: np.ndarray = rng.standard_normal((self._embed_dim, 256))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _tagger_W(self, dim: int) -> np.ndarray:
        """Return a (n_classes, dim) projection, reproducibly derived from the seed."""
        if dim <= self._tagger_base.shape[1]:
            return self._tagger_base[:, :dim]
        # dim > 256: regenerate a wider base (rare; keep same seed path)
        rng = np.random.default_rng(self._seed)
        return rng.standard_normal((self._n_classes, dim))

    def _embed_P(self, dim: int) -> np.ndarray:
        """Return an (embed_dim, dim) projection, reproducibly derived from the seed."""
        if dim <= self._embed_base.shape[1]:
            return self._embed_base[:, :dim]
        rng = np.random.default_rng(self._seed + 1)
        return rng.standard_normal((self._embed_dim, dim))

    # ------------------------------------------------------------------
    # Per-measure rules (pure functions of the audio vector)
    # ------------------------------------------------------------------

    def _presence_detector(self, audio: np.ndarray, axis: Axis) -> SelfTarget:
        """Binary presence: 1 if mean(audio) > 0 else 0."""
        label = int(np.mean(audio) > 0.0)
        return SelfTarget(axis_id=axis.id, kind=AxisKind.CATEGORICAL, label=label)

    def _onset_timing_bin(self, audio: np.ndarray, axis: Axis) -> SelfTarget:
        """Coarse timing bin derived from the first element of the audio vector.

        We apply a uniform quantisation of audio[0] into _N_TIMING_BINS bins
        over the range [-3, 3] (covers ~3 sigma for standard normal).  Values
        outside this range are clamped to the nearest bin.
        """
        lo, hi = -3.0, 3.0
        val = float(np.asarray(audio).flat[0])
        val_clipped = np.clip(val, lo, hi)
        bin_float = (val_clipped - lo) / (hi - lo) * _N_TIMING_BINS
        label = int(np.clip(int(bin_float), 0, _N_TIMING_BINS - 1))
        return SelfTarget(axis_id=axis.id, kind=AxisKind.CATEGORICAL, label=label)

    def _audio_tagger_top1(self, audio: np.ndarray, axis: Axis) -> SelfTarget:
        """Coarse event class: argmax of W @ audio (W is the stored projection)."""
        a = np.asarray(audio, dtype=float).ravel()
        W = self._tagger_W(a.shape[0])
        logits = W @ a
        label = int(np.argmax(logits))
        return SelfTarget(axis_id=axis.id, kind=AxisKind.CATEGORICAL, label=label)

    def _audio_embedding(self, audio: np.ndarray, axis: Axis) -> SelfTarget:
        """Material embedding: unit-normed projection of the audio vector."""
        a = np.asarray(audio, dtype=float).ravel()
        P = self._embed_P(a.shape[0])
        emb = P @ a
        norm = np.linalg.norm(emb)
        if norm > 0.0:
            emb = emb / norm
        else:
            emb = np.zeros_like(emb)
        return SelfTarget(axis_id=axis.id, kind=AxisKind.EMBEDDING, embedding=emb)

    def _binding_label(self, audio: np.ndarray, axis: Axis) -> SelfTarget:
        """Multi-event binding label: (sign(audio[0]), sign(audio[1])) as a string.

        Requires at least 2 elements in the audio vector.  If the audio has only
        one element, the second dimension is treated as 0.0 (sign → 0).
        """
        a = np.asarray(audio, dtype=float).ravel()
        s0 = int(np.sign(a[0]))
        s1 = int(np.sign(a[1])) if a.shape[0] > 1 else 0
        label = f"({s0},{s1})"
        return SelfTarget(axis_id=axis.id, kind=AxisKind.CATEGORICAL, label=label)

    def _seed_predictability(self, audio: np.ndarray, axis: Axis) -> SelfTarget:
        """Placeholder for the offscreen-hallucination / seed-predictability axis.

        This axis is SEPARATE (not a commitment window).  We return a binary label
        derived from audio sign for reproducibility in testing; the real pipeline
        analyses this separately from the window maps.
        """
        a = np.asarray(audio, dtype=float).ravel()
        label = int(np.mean(a) > 0.0)
        return SelfTarget(axis_id=axis.id, kind=AxisKind.CATEGORICAL, label=label)

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    _DISPATCH = {
        "presence_detector": "_presence_detector",
        "onset_timing_bin": "_onset_timing_bin",
        "audio_tagger_top1": "_audio_tagger_top1",
        "audio_embedding": "_audio_embedding",
        "binding_label": "_binding_label",
        "seed_predictability": "_seed_predictability",
    }

    def measure(self, audio: np.ndarray, axis: Axis) -> SelfTarget:
        """Return the SelfTarget for *audio* on *axis* using the synthetic rules.

        The returned SelfTarget.kind always matches axis.kind.
        """
        method_name = self._DISPATCH.get(axis.measure)
        if method_name is None:
            raise ValueError(
                f"SyntheticMeasurer: unknown measure name {axis.measure!r} "
                f"for axis {axis.id!r}.  Register it in the _DISPATCH table."
            )
        target = getattr(self, method_name)(np.asarray(audio, dtype=float), axis)
        # Consistency guard: the returned kind must match the axis kind.
        if target.kind is not axis.kind:
            raise RuntimeError(
                f"SyntheticMeasurer: measure {axis.measure!r} returned kind "
                f"{target.kind!r} but axis {axis.id!r} expects {axis.kind!r}."
            )
        return target


# ---------------------------------------------------------------------------
# RealMeasurer — lazy-import stubs; raises NotImplementedError with dep name
# ---------------------------------------------------------------------------


class RealMeasurer:
    """Self-target measurer for real (non-synthetic) audio.

    Each measure method raises ``NotImplementedError`` with the name of the
    dependency it requires until Phase 0 wires the real tagger/onset/embedder.
    Heavy imports (torch, librosa, laion-clap, …) are LAZY: they appear only
    inside the method bodies so the package stays importable on a numpy-only env.
    """

    # ------------------------------------------------------------------
    # Individual measures
    # ------------------------------------------------------------------

    def _presence_detector(self, audio: np.ndarray, axis: Axis) -> SelfTarget:  # noqa: ARG002
        raise NotImplementedError(
            "Phase 0: wire presence_detector on waveforms; "
            "requires librosa or torchaudio (energy thresholding on real waveform)."
        )

    def _onset_timing_bin(self, audio: np.ndarray, axis: Axis) -> SelfTarget:  # noqa: ARG002
        raise NotImplementedError(
            "Phase 0: wire onset_timing_bin on waveforms; "
            "requires librosa (onset detection + timing quantisation)."
        )

    def _audio_tagger_top1(self, audio: np.ndarray, axis: Axis) -> SelfTarget:  # noqa: ARG002
        raise NotImplementedError(
            "Phase 0: wire audio_tagger_top1 on waveforms; "
            "requires an audio-event tagger (e.g. PANNs / AudioSet classifier, torch)."
        )

    def _audio_embedding(self, audio: np.ndarray, axis: Axis) -> SelfTarget:  # noqa: ARG002
        raise NotImplementedError(
            "Phase 0: wire audio_embedding on waveforms; "
            "requires laion-clap or equivalent audio encoder (torch)."
        )

    def _binding_label(self, audio: np.ndarray, axis: Axis) -> SelfTarget:  # noqa: ARG002
        raise NotImplementedError(
            "Phase 0: wire binding_label on waveforms; "
            "requires an onset detector + event-class tagger for two-event clips "
            "(librosa + audio tagger, torch)."
        )

    def _seed_predictability(self, audio: np.ndarray, axis: Axis) -> SelfTarget:  # noqa: ARG002
        raise NotImplementedError(
            "Phase 0: wire seed_predictability; "
            "requires comparing initial-noise fingerprints — separate from window maps."
        )

    # ------------------------------------------------------------------
    # Dispatch (identical table as SyntheticMeasurer)
    # ------------------------------------------------------------------

    _DISPATCH = {
        "presence_detector": "_presence_detector",
        "onset_timing_bin": "_onset_timing_bin",
        "audio_tagger_top1": "_audio_tagger_top1",
        "audio_embedding": "_audio_embedding",
        "binding_label": "_binding_label",
        "seed_predictability": "_seed_predictability",
    }

    def measure(self, audio: np.ndarray, axis: Axis) -> SelfTarget:
        """Dispatch to the real measure for *axis*.  Raises NotImplementedError."""
        method_name = self._DISPATCH.get(axis.measure)
        if method_name is None:
            raise ValueError(
                f"RealMeasurer: unknown measure name {axis.measure!r} "
                f"for axis {axis.id!r}."
            )
        return getattr(self, method_name)(np.asarray(audio, dtype=float), axis)


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------


def measure_self_target(
    audio: np.ndarray,
    axis: Axis,
    measurer: Any,
) -> SelfTarget:
    """Thin convenience: call measurer.measure(audio, axis) and return the result.

    The *measurer* argument may be any object satisfying the Measurer protocol
    (SyntheticMeasurer, RealMeasurer, or a custom implementation).
    """
    return measurer.measure(np.asarray(audio, dtype=float), axis)
