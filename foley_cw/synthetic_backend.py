"""Closed-form analytic flow backend — the CPU test oracle.

This is NOT a stand-in for MMAudio's science; it is a numerically exact fixture whose
score, velocity, marginals, and marginal-preservation property are known in closed form.
It exists so the highest-silent-bug-risk math (velocity->score conversion and the
marginal-preserving SDE fork kernel) can be validated with no GPU and no MMAudio, exactly
the Phase-0.2 checks the plan demands (alpha=0 reproduces the ODE; small-alpha continuity;
nontrivial diversity; and — stronger than the plan asks — exact score recovery and exact
marginal preservation against the analytic truth).

Convention (rectified-flow / linear interpolant), matched to score_sde.RECTIFIED_LINEAR:

    x_t = t * x1 + (1 - t) * eps,   x1 ~ N(mu, diag(sigma2)),  eps ~ N(0, I),  t in [0, 1]

so t=0 is pure noise (progress s=0) and t=1 is data/audio (progress s=1). Under the
IdentitySToT map (t = s) used here, every formula below is exact for the whole pipeline.

Closed forms (diagonal Sigma = diag(sigma2)), with d = x - t*mu and Sigma_t = t^2 sigma2 + (1-t)^2:

    marginal    x_t ~ N(t*mu, Sigma_t)
    score       s(x,t) = -(x - t*mu) / Sigma_t                       [= grad log p_t]
    velocity    v(x,t) = mu + (t*sigma2 - (1-t)) / Sigma_t * d        [PF-ODE drift dx/dt]
    Tweedie x0  x0(x,t) = x + (1-t) * v(x,t) = E[x1 | x_t]

One can verify  score_from_velocity(v, x, t) = (t*v - x)/(1-t) = s(x,t)  identically for
t in [0,1) (algebra in score_sde / tests), which is the conversion this fixture validates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model_adapter import FlowModelBackend
from .time_map import IdentitySToT


@dataclass(frozen=True)
class SyntheticVideoCond:
    """Per-"video" conditioning: shifts the data mean so different videos imply different
    self-targets. Independent generations of the SAME cond give the video-prior agreement
    A_independent; stochastic tail-forks from a shared x_s give A_fork."""

    mu: np.ndarray
    video_id: str = "synthetic"


class SyntheticGaussianFlow(FlowModelBackend):
    """Analytic Gaussian rectified-flow with diagonal data covariance."""

    def __init__(self, dim: int = 4, sigma2: float | np.ndarray = 0.25) -> None:
        self.dim = int(dim)
        sig = np.broadcast_to(np.asarray(sigma2, dtype=float), (self.dim,)).astype(float)
        if np.any(sig <= 0):
            raise ValueError("sigma2 must be positive (finite score at t=1)")
        self.sigma2 = sig.copy()
        self.s_to_t = IdentitySToT

    # -- FlowModelBackend interface ----------------------------------------------------
    @property
    def state_shape(self) -> tuple[int, ...]:
        return (self.dim,)

    def sample_prior(self, cond: SyntheticVideoCond, rng: np.random.Generator) -> np.ndarray:
        # x at progress s=0 is the standard-normal noise eps.
        return rng.standard_normal(self.dim)

    def velocity(self, x: np.ndarray, t: float, cond: SyntheticVideoCond) -> np.ndarray:
        mu = np.asarray(cond.mu, dtype=float)
        d = x - t * mu
        sigma_t = t * t * self.sigma2 + (1.0 - t) ** 2
        return mu + (t * self.sigma2 - (1.0 - t)) / sigma_t * d

    def decode(self, x: np.ndarray) -> np.ndarray:
        # Synthetic "audio" is the final state itself.
        return np.asarray(x, dtype=float)

    # -- analytic truth (for validation only; a real backend cannot provide these) -----
    def analytic_score(self, x: np.ndarray, t: float, cond: SyntheticVideoCond) -> np.ndarray:
        mu = np.asarray(cond.mu, dtype=float)
        sigma_t = t * t * self.sigma2 + (1.0 - t) ** 2
        return -(x - t * mu) / sigma_t

    def analytic_marginal(self, t: float, cond: SyntheticVideoCond) -> tuple[np.ndarray, np.ndarray]:
        """(mean, diagonal covariance) of x_t under this cond."""
        mu = np.asarray(cond.mu, dtype=float)
        cov_diag = t * t * self.sigma2 + (1.0 - t) ** 2
        return t * mu, cov_diag

    # -- convenience -------------------------------------------------------------------
    @staticmethod
    def make_video_bank(n_videos: int, dim: int = 4, mu_scale: float = 2.0,
                        seed: int = 0) -> list[SyntheticVideoCond]:
        """A reproducible bank of synthetic videos with well-separated data means."""
        rng = np.random.default_rng(seed)
        return [
            SyntheticVideoCond(mu=mu_scale * rng.standard_normal(dim), video_id=f"vid{i:03d}")
            for i in range(n_videos)
        ]
