"""The commitment fork kernel and the velocity<->score conversion (the crux).

This module is model-agnostic: it operates only through a `FlowModelBackend`. It is the
single home of the two silent-bug risks the plan names:

  1. velocity -> score conversion (refine-logs/EXPERIMENT_PLAN.md Phase 0.2, watch-list #3)
  2. the marginal-preserving SDE step (drift v + 1/2 sigma^2 score; alpha=0 => pure ODE)

CONVENTION. All formulas assume the rectified-flow / linear interpolant
`x_t = t*x1 + (1-t)*eps`, under which (derivation in tests / synthetic_backend docstring):

    score_from_velocity(v, x, t) = (t*v - x) / (1 - t)          # = grad log p_t, exact for t<1
    tweedie_x0(v, x, t)          = x + (1 - t)*v                 # = E[x1 | x_t] (readout input)

These are validated to machine precision against `SyntheticGaussianFlow.analytic_score`.
The MMAudio convention (sign/direction, whether the net predicts v vs epsilon/score, and
the s<->t direction) is UNVERIFIED and MUST be audited in Phase 0.2 before this kernel is
trusted on MMAudio. alpha=0 reproducing the ODE is necessary but does NOT exercise the
score term (it is multiplied by 0); the real test is nonzero-alpha (see foley_cw.validation).

Restart re-noising is deliberately NOT implemented here — it is reserved for Phase 6
rollback. Commitment uses ONLY this marginal-preserving SDE.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np

from .model_adapter import FlowModelBackend
from .types import ScheduleSpec

# Score conversion convention id. A second branch would be added (and audited) for
# MMAudio if its parameterization differs.
RECTIFIED_LINEAR = "rectified_linear"

# Floor on (1 - t) so the score conversion stays finite at the final step. The integrator
# evaluates the score only at left endpoints s_i < 1, so this floor is a guard, not a
# routine code path.
_ONE_MINUS_T_FLOOR = 1e-6


# --------------------------------------------------------------------------------------
# velocity <-> score and Tweedie x0
# --------------------------------------------------------------------------------------
def score_from_velocity(v: np.ndarray, x: np.ndarray, t: float,
                        convention: str = RECTIFIED_LINEAR) -> np.ndarray:
    """grad_x log p_t(x) from the velocity, under the stated convention.

    rectified_linear:  score = (t*v - x) / (1 - t).
    """
    if convention != RECTIFIED_LINEAR:
        raise ValueError(
            f"unknown/unaudited score convention {convention!r}; MMAudio's parameterization "
            "must be derived and validated in Phase 0.2 before adding a branch here"
        )
    denom = max(1.0 - t, _ONE_MINUS_T_FLOOR)
    return (t * v - x) / denom


def velocity_from_score(score: np.ndarray, x: np.ndarray, t: float,
                        convention: str = RECTIFIED_LINEAR) -> np.ndarray:
    """Inverse of `score_from_velocity` (used only for testing the round trip)."""
    if convention != RECTIFIED_LINEAR:
        raise ValueError(f"unknown score convention {convention!r}")
    # v = (x + (1 - t)*score) / t
    t_floor = max(t, _ONE_MINUS_T_FLOOR)
    return (x + (1.0 - t) * score) / t_floor


def tweedie_x0(v: np.ndarray, x: np.ndarray, t: float,
               convention: str = RECTIFIED_LINEAR) -> np.ndarray:
    """Tweedie best-guess of the final audio E[x1 | x_t] = x + (1-t)*v (the readout input x0(s))."""
    if convention != RECTIFIED_LINEAR:
        raise ValueError(f"unknown score convention {convention!r}")
    return x + (1.0 - t) * v


def x0_at(backend: FlowModelBackend, x_s: np.ndarray, s: float, cond: Any) -> np.ndarray:
    """Decoded Tweedie x0(s) from a state x_s at progress s (the blurry readout preview)."""
    t = backend.s_to_t.s_to_t(s)
    v = backend.velocity(x_s, t, cond)
    return backend.decode(tweedie_x0(v, x_s, t))


# --------------------------------------------------------------------------------------
# diffusion schedule g(s)
# --------------------------------------------------------------------------------------
def make_g(kind: str = "constant", value: float = 1.0) -> Callable[[float], float]:
    """Diffusion-scale g(s); sigma = alpha * g(s). alpha is the only tuned knob."""
    if kind == "constant":
        return lambda s: float(value)
    if kind == "linear_down":           # more noise early, anneal toward s=1
        return lambda s: float(value) * (1.0 - s)
    if kind == "sqrt_down":
        return lambda s: float(value) * float(np.sqrt(max(1.0 - s, 0.0)))
    raise ValueError(f"unknown g kind {kind!r}")


# --------------------------------------------------------------------------------------
# Euler-Maruyama step (forward integration in native time t over signed dt)
# --------------------------------------------------------------------------------------
def euler_maruyama_step(x: np.ndarray, v: np.ndarray, score: Optional[np.ndarray],
                        sigma: float, dt: float, rng: np.random.Generator) -> np.ndarray:
    """One marginal-preserving SDE step.

        x <- x + (v + 1/2 * sigma^2 * score) * dt + sigma * sqrt(|dt|) * xi

    The 1/2 sigma^2 score drift correction is the UNIQUE coefficient that preserves the
    PF-ODE marginals p_t (Fokker-Planck; see tests). sigma=0 collapses to the ODE Euler
    step x <- x + v*dt regardless of `score`.
    """
    x_next = x + v * dt
    if sigma != 0.0:
        if score is None:
            raise ValueError("score required when sigma != 0")
        x_next = x_next + 0.5 * sigma * sigma * score * dt
        x_next = x_next + sigma * np.sqrt(abs(dt)) * rng.standard_normal(x.shape)
    return x_next


# --------------------------------------------------------------------------------------
# Segment integrator: walk progress s over a grid, return final state
# --------------------------------------------------------------------------------------
def _segment_grid(schedule: ScheduleSpec, s_start: float, s_end: float) -> np.ndarray:
    """Sub-step grid in progress from s_start to s_end, aligned to the integration grid."""
    base = schedule.integration_s_grid()
    inner = base[(base > s_start + 1e-12) & (base < s_end - 1e-12)]
    grid = np.concatenate([[s_start], inner, [s_end]])
    grid = np.unique(np.round(grid, 12))
    if grid.size < 2:
        grid = np.array([s_start, s_end])
    return grid


def integrate_segment(backend: FlowModelBackend, x: np.ndarray, cond: Any,
                      s_start: float, s_end: float, schedule: ScheduleSpec,
                      alpha: float, g: Callable[[float], float],
                      rng: Optional[np.random.Generator],
                      convention: str = RECTIFIED_LINEAR) -> np.ndarray:
    """Integrate state x from progress s_start to s_end. alpha=0 => deterministic ODE."""
    s_to_t = backend.s_to_t
    grid = _segment_grid(schedule, s_start, s_end)
    x = np.array(x, dtype=float, copy=True)
    for i in range(grid.size - 1):
        s_i = float(grid[i])
        t_i = s_to_t.s_to_t(s_i)
        t_next = s_to_t.s_to_t(float(grid[i + 1]))
        dt = t_next - t_i
        v = backend.velocity(x, t_i, cond)
        if alpha == 0.0:
            x = euler_maruyama_step(x, v, None, 0.0, dt, rng)  # type: ignore[arg-type]
        else:
            score = score_from_velocity(v, x, t_i, convention)
            sigma = alpha * g(s_i)
            if rng is None:
                raise ValueError("rng required for stochastic (alpha>0) integration")
            x = euler_maruyama_step(x, v, score, sigma, dt, rng)
    return x


# --------------------------------------------------------------------------------------
# Trajectory access (Phase 0.1) and the commitment fork kernel (Phase 1)
# --------------------------------------------------------------------------------------
def generate_trajectory(backend: FlowModelBackend, cond: Any, schedule: ScheduleSpec,
                        rng: np.random.Generator, alpha: float = 0.0,
                        record_points: Optional[tuple[float, ...]] = None,
                        x_init: Optional[np.ndarray] = None,
                        g: Optional[Callable[[float], float]] = None) -> dict[str, Any]:
    """One full generation s=0 -> s=1, recording intermediate states x_s at `record_points`.

    alpha=0 is the deterministic ODE generation (the normal sampler). Returns:
      {"states": {s: x_s}, "final_state": x_1, "audio": decode(x_1)}.
    This is the trajectory-access primitive audited in Phase 0.1 (extract x_s, resume from
    x_s, compute x0(s)).
    """
    if g is None:
        g = make_g(schedule.g_kind, schedule.g_value)
    if record_points is None:
        record_points = schedule.scan_points
    record_points = tuple(sorted(set(float(s) for s in record_points) | {0.0, 1.0}))

    x = backend.sample_prior(cond, rng) if x_init is None else np.array(x_init, dtype=float, copy=True)
    states: dict[float, np.ndarray] = {}
    s_cursor = 0.0
    if 0.0 in record_points:
        states[0.0] = np.array(x, copy=True)
    for s_next in record_points[1:]:
        x = integrate_segment(backend, x, cond, s_cursor, s_next, schedule, alpha, g, rng)
        states[float(s_next)] = np.array(x, copy=True)
        s_cursor = float(s_next)
    return {"states": states, "final_state": x, "audio": backend.decode(x)}


def ode_complete(backend: FlowModelBackend, x_s: np.ndarray, s: float, cond: Any,
                 schedule: ScheduleSpec) -> np.ndarray:
    """Deterministic alpha=0 completion of x_s to s=1: the ODE-target path's final audio.

    This is the 'original path this candidate would realize' used as the ODE-target in the
    readout map (Phase 2).
    """
    x1 = integrate_segment(backend, x_s, cond, s, 1.0, schedule, alpha=0.0, g=lambda _s: 0.0,
                           rng=None)
    return backend.decode(x1)


def fork_tail(backend: FlowModelBackend, x_s: np.ndarray, s: float, cond: Any,
              alpha: float, K: int, schedule: ScheduleSpec, rng: np.random.Generator,
              g: Optional[Callable[[float], float]] = None,
              convention: str = RECTIFIED_LINEAR) -> list[np.ndarray]:
    """K marginal-preserving stochastic tail-forks from x_s to s=1 (Phase 1 kernel).

    Returns K decoded final audios. alpha=0 yields K identical ODE completions (the
    necessary-not-sufficient alpha=0 check). Restart re-noising is NOT used here.
    """
    if g is None:
        g = make_g(schedule.g_kind, schedule.g_value)
    comps: list[np.ndarray] = []
    for _k in range(int(K)):
        x1 = integrate_segment(backend, x_s, cond, s, 1.0, schedule, alpha, g, rng, convention)
        comps.append(backend.decode(x1))
    return comps
