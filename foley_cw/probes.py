"""Readout probes over x0(s) — the Phase-2 probe ladder.

Every probe receives the DECODED x0(s) vector (the Tweedie best-guess of the final audio
at progress s, output of score_sde.x0_at) and an Axis descriptor, and returns a
SelfTarget in the correct kind for that axis.

Probe ladder order (refine-logs/EXPERIMENT_PLAN.md §9 / Phase 2):
  1. energy / onset heuristics  (EnergyOnsetProbe, CPU, deterministic)   — rung 1
  2. CLAP / SyncNet / ImageBind — legacy baselines                         — rung 2
  3. audio tagger on x0(s)      (AudioTaggerProbe stub)                    — rung 3
  4. frontier MLLM-on-x0(s)    (MLLMPreviewProbe stub, headline, §9 rung 5) — rung 5
  6. internal-feature probes    (non-blocking, Phase 7; not in this ladder)

Design rules (matching the plan):
  * CPU-runnable probes return a real SelfTarget.  Heavy probes raise
    NotImplementedError("<ProbeName>: <dep> not wired; requires <package>").
  * Lazy imports only for heavy deps; the module must import with numpy alone.
  * `legacy=True`  for CLAP / SyncNet / ImageBind (deprecated baselines, plan §9 rung 2).
  * `legacy=False` for all other probes (EnergyOnset, AudioTagger, MLLMPreview).
  * probe_ladder(include_stubs=False) returns only CPU-runnable probes (EnergyOnsetProbe)
    so the synthetic dry-run works without any heavy dependency.
"""

from __future__ import annotations

import abc
from typing import Protocol, runtime_checkable

import numpy as np

from .types import Axis, AxisKind, SelfTarget

# ---------------------------------------------------------------------------
# Protocol (structural; no import from sibling axes module needed here)
# ---------------------------------------------------------------------------

@runtime_checkable
class Probe(Protocol):
    """A probe that predicts the axis self-target from a decoded x0 VECTOR."""

    name: str
    legacy: bool

    def predict(self, x0_audio: np.ndarray, axis: Axis) -> SelfTarget:
        ...


# ---------------------------------------------------------------------------
# Helper shared between EnergyOnsetProbe and the SyntheticMeasurer-style logic
# ---------------------------------------------------------------------------

def _n_classes_for_timing_bin(n: int = 10) -> int:
    """Number of timing bins — kept as a constant so probes agree with axes.SyntheticMeasurer."""
    return n


def _energy_presence(x0_audio: np.ndarray) -> int:
    """Predict presence = 1 if mean(x0) > 0, else 0 — mirrors SyntheticMeasurer."""
    return int(np.mean(x0_audio) > 0)


def _onset_bin(x0_audio: np.ndarray, n_bins: int = 10) -> int:
    """Bin the first element of x0 into n_bins equal segments over [-3, 3] — mirrors SyntheticMeasurer."""
    val = float(x0_audio[0]) if x0_audio.size > 0 else 0.0
    # Clamp to [-3, 3] and bin — same deterministic rule used in SyntheticMeasurer
    val_clamped = max(-3.0, min(3.0, val))
    idx = int((val_clamped + 3.0) / 6.0 * n_bins)
    return min(idx, n_bins - 1)


def _unit_norm(v: np.ndarray) -> np.ndarray:
    """Unit-normalize a vector; return zero vector if norm is zero."""
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return np.zeros_like(v, dtype=float)
    return v.astype(float) / n


def _binding_label(x0_audio: np.ndarray) -> tuple:
    """Predict binding label = (sign(x0[0]), sign(x0[1])) — mirrors SyntheticMeasurer."""
    s0 = int(np.sign(x0_audio[0])) if x0_audio.size > 0 else 0
    s1 = int(np.sign(x0_audio[1])) if x0_audio.size > 1 else 0
    return (s0, s1)


# ---------------------------------------------------------------------------
# Concrete CPU probe — rung 1
# ---------------------------------------------------------------------------

class EnergyOnsetProbe:
    """Rung-1 energy/onset heuristic probe — deterministic, CPU-only, no heavy deps.

    Applied to the decoded x0(s) VECTOR (the Tweedie best-guess of final audio).
    Uses the same deterministic rules as axes.SyntheticMeasurer so that on synthetic
    data the probe accuracy rises monotonically as x0(s) → x1 (s → 1):

      * presence  →  int(mean(x0) > 0)               [categorical]
      * timing    →  bin(x0[0], n_bins=10)            [categorical]
      * class     →  int(mean(x0) > 0)  (proxy)       [categorical]
      * material  →  unit_norm(x0)                    [embedding]
      * binding   →  (sign(x0[0]), sign(x0[1]))       [categorical]
      * offscreen_hallucination  →  int(mean(x0) > 0) [categorical]

    For any unknown measure the probe falls back to presence-style rule so the
    categorical/embedding kind contract is always satisfied.
    """

    name: str = "energy_onset"
    legacy: bool = False

    def predict(self, x0_audio: np.ndarray, axis: Axis) -> SelfTarget:
        """Predict the axis self-target from the decoded x0 vector."""
        x0 = np.asarray(x0_audio, dtype=float)
        measure = axis.measure

        if axis.kind is AxisKind.EMBEDDING:
            # material / audio_embedding — unit-norm of x0
            emb = _unit_norm(x0)
            return SelfTarget(axis_id=axis.id, kind=axis.kind, embedding=emb)

        # All categorical axes
        if measure == "presence_detector":
            label = _energy_presence(x0)
        elif measure == "onset_timing_bin":
            label = _onset_bin(x0)
        elif measure == "audio_tagger_top1":
            # Proxy: class = presence for synthetic (same deterministic signal)
            label = _energy_presence(x0)
        elif measure == "binding_label":
            label = _binding_label(x0)
        elif measure == "seed_predictability":
            # offscreen_hallucination axis — use presence proxy
            label = _energy_presence(x0)
        else:
            # Unknown measure — safe fallback for categorical
            label = _energy_presence(x0)

        return SelfTarget(axis_id=axis.id, kind=axis.kind, label=label)


# ---------------------------------------------------------------------------
# Legacy heavy stubs — rung 2 (CLAP, SyncNet, ImageBind)
# ---------------------------------------------------------------------------

class CLAPProbe:
    """Rung-2 CLAP legacy baseline — NOT the headline probe.

    Raises NotImplementedError until laion-clap is installed and wired.
    """

    name: str = "clap"
    legacy: bool = True

    def predict(self, x0_audio: np.ndarray, axis: Axis) -> SelfTarget:
        # Lazy import guard — do not load at module import time
        try:
            import laion_clap  # noqa: F401  # type: ignore[import]
        except ImportError:
            pass
        raise NotImplementedError(
            "CLAPProbe: laion-clap not wired; requires laion-clap (see requirements.txt Phase 0+)"
        )


class SyncNetProbe:
    """Rung-2 SyncNet legacy baseline — NOT the headline probe.

    Raises NotImplementedError until SyncNet is installed and wired.
    """

    name: str = "syncnet"
    legacy: bool = True

    def predict(self, x0_audio: np.ndarray, axis: Axis) -> SelfTarget:
        try:
            import syncnet  # noqa: F401  # type: ignore[import]
        except ImportError:
            pass
        raise NotImplementedError(
            "SyncNetProbe: syncnet not wired; requires syncnet (Phase 0+ AV-sync dependency)"
        )


class ImageBindProbe:
    """Rung-2 ImageBind legacy baseline — NOT the headline probe.

    Raises NotImplementedError until ImageBind is installed and wired.
    """

    name: str = "imagebind"
    legacy: bool = True

    def predict(self, x0_audio: np.ndarray, axis: Axis) -> SelfTarget:
        try:
            import imagebind  # noqa: F401  # type: ignore[import]
        except ImportError:
            pass
        raise NotImplementedError(
            "ImageBindProbe: imagebind not wired; requires imagebind (Phase 0+ multimodal embedding dependency)"
        )


# ---------------------------------------------------------------------------
# Non-legacy heavy stubs — rung 3 and rung 5
# ---------------------------------------------------------------------------

class AudioTaggerProbe:
    """Rung-3 audio tagger on x0(s).

    Raises NotImplementedError until the audio tagger is wired.
    """

    name: str = "audio_tagger"
    legacy: bool = False

    def predict(self, x0_audio: np.ndarray, axis: Axis) -> SelfTarget:
        try:
            import torch  # noqa: F401  # type: ignore[import]
        except ImportError:
            pass
        raise NotImplementedError(
            "AudioTaggerProbe: audio tagger not wired; requires torch and torchaudio "
            "(Phase 0+ audio tagging dependency; wire tagger checkpoint in Phase 0.5)"
        )


class MLLMPreviewProbe:
    """Rung-5 frontier MLLM-on-x0(s) — the HEADLINE probe (non-legacy, non-blocking).

    Raises NotImplementedError until the MLLM client is wired.
    The MLLM operates on the decoded x0(s) audio preview and returns a self-target
    for the axis via a structured prompt. This is the Phase-2 headline probe.
    """

    name: str = "mllm_preview"
    legacy: bool = False

    def predict(self, x0_audio: np.ndarray, axis: Axis) -> SelfTarget:
        # Lazy import: the MLLM client (e.g. openai, anthropic SDK) is not imported at
        # module level — it is only needed when the probe is actually called.
        raise NotImplementedError(
            "MLLMPreviewProbe: frontier MLLM client not wired; requires an MLLM client "
            "(e.g. openai or anthropic SDK; Phase 2 headline probe — wire in Phase 0.5 "
            "after reliability gate passes)"
        )


# ---------------------------------------------------------------------------
# Probe ladder factory
# ---------------------------------------------------------------------------

def probe_ladder(include_stubs: bool = False) -> list:
    """Return probes in plan-specified order.

    With include_stubs=False (the default), only CPU-runnable probes are returned:
    [EnergyOnsetProbe()] — so the synthetic dry-run works on any machine.

    With include_stubs=True, all probes are returned in ladder order:
      rung 1: EnergyOnsetProbe
      rung 2: CLAPProbe, SyncNetProbe, ImageBindProbe  (legacy=True)
      rung 3: AudioTaggerProbe
      rung 5: MLLMPreviewProbe  (headline, legacy=False)
    """
    cpu_probes: list = [EnergyOnsetProbe()]
    if not include_stubs:
        return cpu_probes

    # Full ladder with stubs (ordered as per plan §9)
    return [
        EnergyOnsetProbe(),       # rung 1: energy/onset heuristics
        CLAPProbe(),              # rung 2: legacy CLAP
        SyncNetProbe(),           # rung 2: legacy SyncNet
        ImageBindProbe(),         # rung 2: legacy ImageBind
        AudioTaggerProbe(),       # rung 3: audio tagger on x0(s)
        MLLMPreviewProbe(),       # rung 5: frontier MLLM (headline)
    ]
