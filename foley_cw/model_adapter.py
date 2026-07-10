"""Model backend seam: the only place that touches a concrete flow generator.

`FlowModelBackend` is the abstract interface the entire pipeline integrates against. Two
implementations exist:

  * `synthetic_backend.SyntheticGaussianFlow` — a closed-form analytic flow used to
    unit-test the score/SDE math on CPU (the validated path).
  * `MMAudioBackend` — the real video-conditioned audio flow model. It is a STUB here:
    MMAudio source/weights are not vendored in this repo, so every method raises
    `MMAudioNotWired` with the exact Phase-0 audit point it depends on. Wiring it is the
    job of the Phase-0 feasibility diagnostic (owner=human, GPU), NOT of this audit-only
    bridge.

Everything model-specific that the plan flags as a silent-bug risk lives behind this
seam: the velocity field v_theta(x, t, cond), the s<->t direction (via time_map), the
latent decode to audio, and the noise prior.
"""

from __future__ import annotations

import abc
from typing import Any, Optional

import numpy as np

from .time_map import SToT


class MMAudioNotWired(NotImplementedError):
    """Raised by MMAudioBackend until Phase 0.1 wires real MMAudio trajectory access."""


class FlowModelBackend(abc.ABC):
    """Abstract video-conditioned audio flow model.

    State `x` is a numpy array of shape `state_shape` (a latent for MMAudio, a plain
    vector for the synthetic backend). `cond` is an opaque conditioning object (the video
    features for MMAudio; ignored by the synthetic backend). Model time `t` is the NATIVE
    integration time; callers obtain it only via a `time_map.SToT`.
    """

    #: s<->t convention this backend integrates under. Pipelines must read this, never
    #: assume a direction.
    s_to_t: SToT

    @property
    @abc.abstractmethod
    def state_shape(self) -> tuple[int, ...]:
        ...

    @abc.abstractmethod
    def sample_prior(self, cond: Any, rng: np.random.Generator) -> np.ndarray:
        """Draw x at progress s=0 (the noise prior x_{s=0})."""

    @abc.abstractmethod
    def velocity(self, x: np.ndarray, t: float, cond: Any) -> np.ndarray:
        """v_theta(x, t, cond): the PF-ODE velocity dx/dt at native time t."""

    @abc.abstractmethod
    def decode(self, x: np.ndarray) -> np.ndarray:
        """Decode a final state x (at s=1) to an audio waveform/features.

        For the synthetic backend this is identity. For MMAudio this is the VAE/vocoder
        decode and is part of the trajectory-access audit (Phase 0.1).
        """


class MMAudioBackend(FlowModelBackend):
    """STUB for the real MMAudio v1 backend — intentionally not runnable in audit-only.

    Each method documents exactly which Phase-0 audit point must wire it. Constructing the
    object is allowed (so configs/imports resolve); calling a trajectory method raises.
    """

    def __init__(self, checkpoint: Optional[str] = None, s_to_t: Optional[SToT] = None) -> None:
        from .time_map import MMAUDIO_S_TO_T

        self.checkpoint = checkpoint
        # Phase 0.1: replace with the AUDITED mapping and set verified=True.
        self.s_to_t = s_to_t or MMAUDIO_S_TO_T

    @property
    def state_shape(self) -> tuple[int, ...]:
        raise MMAudioNotWired(
            "Phase 0.1: read MMAudio's latent shape (seq_len x latent_dim) from its config."
        )

    def sample_prior(self, cond: Any, rng: np.random.Generator) -> np.ndarray:
        raise MMAudioNotWired(
            "Phase 0.1: draw MMAudio's initial noise latent (match its prior scaling)."
        )

    def velocity(self, x: np.ndarray, t: float, cond: Any) -> np.ndarray:
        raise MMAudioNotWired(
            "Phase 0.1/0.2: call MMAudio's transformer to get v_theta(x, t, video_cond). "
            "Audit the time argument convention (see time_map.MMAUDIO_S_TO_T) and confirm "
            "the network predicts velocity (not score/epsilon) before using score_sde."
        )

    def decode(self, x: np.ndarray) -> np.ndarray:
        raise MMAudioNotWired(
            "Phase 0.1: decode the final latent with MMAudio's VAE/vocoder to a waveform."
        )
