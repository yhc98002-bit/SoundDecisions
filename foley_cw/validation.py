"""Phase-0.1 trajectory access + Phase-0.2 SDE validation checks.

These checks are the load-bearing gate before any commitment/readout map is run.
They validate:

  Phase 0.1 — Trajectory access:
    * extract x_s at a scan point, resume integration from it, compute x0(s);
      shapes must be consistent and values finite.

  Phase 0.2 — SDE / score conversion validation:
    * alpha=0 reproduces the deterministic ODE completion (necessary, not sufficient;
      note: does NOT exercise the score term since it is multiplied by 0).
    * Exact score recovery vs analytic_score (only for SyntheticGaussianFlow or any
      backend that provides analytic_score); validated to machine precision.
    * Small-alpha continuity: as alpha -> 0+, fork outputs converge continuously to
      the ODE output (this exercises the score term).
    * Fork validity: forks at the test alpha produce finite, non-trivially large output.
    * Nontrivial diversity: at the test alpha, K forks show measurable spread.
    * Marginal preservation (synthetic backend only): empirical marginal matches the
      analytic marginal after many fork samples.

  run_sde_validation collects all Phase-0.2 checks and emits a token:
    "OK"                   — all checks pass on the synthetic backend.
    "FIX_SCORE_CONVERSION" — alpha=0 fails to reproduce ODE, or small-alpha continuity
                             is violated; indicates a broken score conversion.

CONVENTIONS (from EXPERIMENT_PLAN.md §2 and score_sde.py):
  * Progress s in [0, 1]: s=0 pure noise, s=1 final audio.
  * x0(s) = Tweedie best-guess E[x1 | x_s].
  * alpha = stochasticity knob; alpha=0 is the deterministic ODE.
  * Restart re-noising is NOT used here (reserved for Phase 6).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .model_adapter import FlowModelBackend
from .score_sde import (
    fork_tail,
    generate_trajectory,
    make_g,
    ode_complete,
    score_from_velocity,
    x0_at,
)
from .types import ScheduleSpec, ValidationResult

# --------------------------------------------------------------------------------------
# Tolerances (pre-registered; sensitivity to these values is low for the analytic backend)
# --------------------------------------------------------------------------------------
_ALPHA0_ODE_TOL = 1e-6       # max L2 distance between alpha=0 fork and ODE completion
_SCORE_EXACT_TOL = 1e-8      # max mean |score_from_velocity - analytic_score| per dim
_CONTINUITY_TOL = 2.0        # max ratio continuity_norm / ode_norm (< 2 means fork not far from ODE)
_MARGINAL_TOL = 3.0          # max absolute z-score for marginal mean/var (per dimension)
_DIVERSITY_MIN = 1e-6        # minimum std-dev of fork outputs to call diversity "nontrivial"


# --------------------------------------------------------------------------------------
# Phase 0.1 — trajectory access
# --------------------------------------------------------------------------------------
def check_trajectory_access(
    backend: FlowModelBackend,
    cond: Any,
    schedule: ScheduleSpec,
    rng: np.random.Generator,
) -> ValidationResult:
    """Phase 0.1: extract x_s, resume from x_s, compute x0(s); shapes must match.

    Specifically:
      1. Run generate_trajectory and extract a recorded intermediate state x_s.
      2. Re-complete the trajectory from x_s using ode_complete.
      3. Compute x0(s) at the same scan point using x0_at.
      4. Verify shapes are consistent and values are finite.
    Returns a ValidationResult with value = 0.0 on success (all checks pass).
    """
    name = "trajectory_access"
    threshold = 0.0  # all-pass threshold: value must equal 0.0 (no failures counted)

    try:
        # Pick a mid-trajectory scan point
        scan_s = 0.5
        record_pts = (0.0, scan_s, 1.0)
        traj = generate_trajectory(
            backend, cond, schedule, rng, alpha=0.0, record_points=record_pts
        )

        states = traj["states"]
        final_state = traj["final_state"]
        audio = traj["audio"]

        # 1. Check that the expected keys exist
        if scan_s not in states:
            return ValidationResult(
                name=name, passed=False, value=1.0, threshold=threshold,
                detail=f"scan point s={scan_s} not recorded in trajectory states",
            )

        x_s = states[scan_s]

        # 2. Check shapes
        state_shape = backend.state_shape
        if x_s.shape != state_shape:
            return ValidationResult(
                name=name, passed=False, value=1.0, threshold=threshold,
                detail=f"x_s.shape {x_s.shape} != state_shape {state_shape}",
            )
        if final_state.shape != state_shape:
            return ValidationResult(
                name=name, passed=False, value=1.0, threshold=threshold,
                detail=f"final_state.shape {final_state.shape} != state_shape {state_shape}",
            )

        # 3. Check finiteness of intermediate state
        if not np.all(np.isfinite(x_s)):
            return ValidationResult(
                name=name, passed=False, value=1.0, threshold=threshold,
                detail="x_s contains non-finite values",
            )

        # 4. Resume from x_s with ode_complete, check shape and finiteness
        resumed_audio = ode_complete(backend, x_s, scan_s, cond, schedule)
        if resumed_audio.shape != audio.shape:
            return ValidationResult(
                name=name, passed=False, value=1.0, threshold=threshold,
                detail=(
                    f"resumed audio shape {resumed_audio.shape} != "
                    f"original audio shape {audio.shape}"
                ),
            )
        if not np.all(np.isfinite(resumed_audio)):
            return ValidationResult(
                name=name, passed=False, value=1.0, threshold=threshold,
                detail="resumed audio from ode_complete contains non-finite values",
            )

        # 5. Compute x0(s) and check shape and finiteness
        x0 = x0_at(backend, x_s, scan_s, cond)
        if x0.shape != audio.shape:
            return ValidationResult(
                name=name, passed=False, value=1.0, threshold=threshold,
                detail=f"x0.shape {x0.shape} != audio.shape {audio.shape}",
            )
        if not np.all(np.isfinite(x0)):
            return ValidationResult(
                name=name, passed=False, value=1.0, threshold=threshold,
                detail="x0(s) from x0_at contains non-finite values",
            )

        return ValidationResult(
            name=name, passed=True, value=0.0, threshold=threshold,
            detail=(
                f"trajectory access OK: x_s shape {x_s.shape}, "
                f"ode_complete shape {resumed_audio.shape}, "
                f"x0(s) shape {x0.shape} — all finite"
            ),
        )

    except Exception as exc:  # noqa: BLE001
        return ValidationResult(
            name=name, passed=False, value=1.0, threshold=threshold,
            detail=f"exception during trajectory access: {exc}",
        )


# --------------------------------------------------------------------------------------
# Phase 0.2 — SDE / score conversion checks
# --------------------------------------------------------------------------------------
def check_alpha0_reproduces_ode(
    backend: FlowModelBackend,
    cond: Any,
    schedule: ScheduleSpec,
    rng: np.random.Generator,
) -> ValidationResult:
    """Phase 0.2: fork_tail(alpha=0) must reproduce ode_complete to numerical tolerance.

    alpha=0 collapses the SDE to the deterministic ODE; K forks of alpha=0 are all
    identical and equal to ode_complete. This is a NECESSARY check (not sufficient —
    it does not exercise the score term since sigma=alpha*g=0 multiplies it out).

    value = max L2 distance between fork output and ODE output (should be <=_ALPHA0_ODE_TOL).
    """
    name = "alpha0_reproduces_ode"
    threshold = _ALPHA0_ODE_TOL

    try:
        scan_s = 0.4
        # Generate a trajectory to get a specific x_s
        traj = generate_trajectory(
            backend, cond, schedule, rng, alpha=0.0, record_points=(0.0, scan_s, 1.0)
        )
        x_s = traj["states"][scan_s]

        # ODE completion
        ode_audio = ode_complete(backend, x_s, scan_s, cond, schedule)

        # Fork tails with alpha=0 (K=2 suffices; all should be identical to ODE)
        g = make_g(schedule.g_kind, schedule.g_value)
        fork_audios = fork_tail(
            backend, x_s, scan_s, cond, alpha=0.0, K=2, schedule=schedule, rng=rng, g=g
        )

        max_dist = 0.0
        for fork_audio in fork_audios:
            dist = float(np.linalg.norm(fork_audio - ode_audio))
            max_dist = max(max_dist, dist)

        passed = max_dist <= threshold
        return ValidationResult(
            name=name,
            passed=passed,
            value=max_dist,
            threshold=threshold,
            detail=(
                f"max L2(fork_alpha0, ode_complete) = {max_dist:.3e} "
                f"(threshold {threshold:.3e}); "
                f"{'PASS' if passed else 'FAIL — score convention may be broken'}"
            ),
        )

    except Exception as exc:  # noqa: BLE001
        return ValidationResult(
            name=name, passed=False, value=float("inf"), threshold=threshold,
            detail=f"exception: {exc}",
        )


def check_score_conversion_exact(
    synth_backend: Any,
    cond: Any,
    schedule: ScheduleSpec,
    rng: np.random.Generator,
) -> ValidationResult:
    """Phase 0.2: score_from_velocity vs analytic_score — ONLY for SyntheticGaussianFlow.

    Computes the mean absolute error |score_from_velocity(v,x,t) - analytic_score(x,t)|
    per dimension, averaged over several scan points. This directly validates the
    velocity->score conversion formula (the highest silent-bug risk).

    Only called when hasattr(backend, 'analytic_score').

    value = mean |computed_score - analytic_score| (should be < _SCORE_EXACT_TOL ~ 1e-8).
    """
    name = "score_conversion_exact"
    threshold = _SCORE_EXACT_TOL

    if not hasattr(synth_backend, "analytic_score"):
        return ValidationResult(
            name=name, passed=True, value=0.0, threshold=threshold,
            detail="skipped: backend has no analytic_score (not SyntheticGaussianFlow)",
        )

    try:
        # Sample a random state x at a mid-progress point and compare scores
        errors: list[float] = []
        test_s_values = [0.1, 0.3, 0.5, 0.7, 0.9]
        for s in test_s_values:
            t = synth_backend.s_to_t.s_to_t(s)
            x = rng.standard_normal(synth_backend.state_shape)
            v = synth_backend.velocity(x, t, cond)
            computed_score = score_from_velocity(v, x, t)
            true_score = synth_backend.analytic_score(x, t, cond)
            err = float(np.mean(np.abs(computed_score - true_score)))
            errors.append(err)

        mean_err = float(np.mean(errors))
        passed = mean_err < threshold
        return ValidationResult(
            name=name,
            passed=passed,
            value=mean_err,
            threshold=threshold,
            detail=(
                f"mean |score_from_velocity - analytic_score| = {mean_err:.3e} "
                f"over {len(test_s_values)} scan points "
                f"(threshold {threshold:.3e}); "
                f"{'PASS' if passed else 'FAIL'}"
            ),
        )

    except Exception as exc:  # noqa: BLE001
        return ValidationResult(
            name=name, passed=False, value=float("inf"), threshold=threshold,
            detail=f"exception: {exc}",
        )


def check_small_alpha_continuity(
    backend: FlowModelBackend,
    cond: Any,
    schedule: ScheduleSpec,
    rng: np.random.Generator,
    alphas: tuple[float, ...] = (0.0, 0.02, 0.05),
) -> ValidationResult:
    """Phase 0.2: as alpha -> 0+, fork outputs converge continuously to the ODE output.

    This check tests the score term (unlike the alpha=0 check). We verify that:
      dist(fork(alpha), ode_complete) is monotonically non-decreasing as alpha increases,
      and that fork(alpha ~ 0) stays close to ode_complete.

    value = max_ratio of |fork(alpha) - ode| / max(|ode|, 1e-8) for the smallest nonzero alpha.
    A low value (< _CONTINUITY_TOL) means the SDE is continuous at alpha=0.
    """
    name = "small_alpha_continuity"
    threshold = _CONTINUITY_TOL

    try:
        scan_s = 0.4
        traj = generate_trajectory(
            backend, cond, schedule, rng, alpha=0.0, record_points=(0.0, scan_s, 1.0)
        )
        x_s = traj["states"][scan_s]
        ode_audio = ode_complete(backend, x_s, scan_s, cond, schedule)
        ode_norm = float(np.linalg.norm(ode_audio))

        g = make_g(schedule.g_kind, schedule.g_value)

        dists: list[tuple[float, float]] = []  # (alpha, mean_dist)
        for alpha in alphas:
            # Use K=4 forks per alpha and take the mean L2 distance to ODE
            fork_audios = fork_tail(
                backend, x_s, scan_s, cond,
                alpha=alpha, K=4, schedule=schedule, rng=rng, g=g,
            )
            mean_dist = float(np.mean([np.linalg.norm(f - ode_audio) for f in fork_audios]))
            dists.append((alpha, mean_dist))

        # The critical metric: smallest nonzero alpha dist relative to ODE norm
        nonzero = [(a, d) for a, d in dists if a > 0.0]
        if not nonzero:
            # Only alpha=0 in the list — trivially continuous
            return ValidationResult(
                name=name, passed=True, value=0.0, threshold=threshold,
                detail="only alpha=0 provided; continuity trivially satisfied",
            )

        smallest_nonzero_alpha, smallest_dist = nonzero[0]
        # ratio: how far the smallest-alpha fork is from ODE, relative to ODE scale
        scale = max(ode_norm, 1e-8)
        ratio = smallest_dist / scale

        # Also check monotonicity of mean distances (approximate; bootstrapped mean may
        # not be strictly monotone but gross violations indicate score problems)
        dists_sorted = [d for _, d in dists]
        monotone = all(d2 >= d1 - 0.1 * scale for d1, d2 in zip(dists_sorted, dists_sorted[1:]))

        passed = ratio < threshold and monotone
        return ValidationResult(
            name=name,
            passed=passed,
            value=ratio,
            threshold=threshold,
            detail=(
                f"smallest nonzero alpha={smallest_nonzero_alpha}: "
                f"mean dist={smallest_dist:.4f}, ode_norm={ode_norm:.4f}, "
                f"ratio={ratio:.4f} (threshold={threshold}); "
                f"monotone={monotone}; "
                f"{'PASS' if passed else 'FAIL — score conversion may be inconsistent'}"
            ),
        )

    except Exception as exc:  # noqa: BLE001
        return ValidationResult(
            name=name, passed=False, value=float("inf"), threshold=threshold,
            detail=f"exception: {exc}",
        )


def check_fork_validity(
    backend: FlowModelBackend,
    cond: Any,
    schedule: ScheduleSpec,
    rng: np.random.Generator,
    alpha: float,
) -> ValidationResult:
    """Phase 0.2: forks at the test alpha produce finite, non-trivially large output.

    value = fraction of forks that are finite AND have L2 norm > some floor (1e-6).
    Passes if all K forks are finite (value == 1.0).
    """
    name = "fork_validity"
    threshold = 1.0  # all forks must be valid

    try:
        scan_s = 0.5
        K = schedule.K_forks
        traj = generate_trajectory(
            backend, cond, schedule, rng, alpha=0.0, record_points=(0.0, scan_s, 1.0)
        )
        x_s = traj["states"][scan_s]

        g = make_g(schedule.g_kind, schedule.g_value)
        fork_audios = fork_tail(
            backend, x_s, scan_s, cond, alpha=alpha, K=K, schedule=schedule, rng=rng, g=g
        )

        valid_count = 0
        for audio in fork_audios:
            if np.all(np.isfinite(audio)) and float(np.linalg.norm(audio)) > 1e-9:
                valid_count += 1

        fraction_valid = valid_count / max(len(fork_audios), 1)
        passed = fraction_valid >= threshold
        return ValidationResult(
            name=name,
            passed=passed,
            value=fraction_valid,
            threshold=threshold,
            detail=(
                f"alpha={alpha}: {valid_count}/{len(fork_audios)} forks are finite "
                f"and non-trivially large (fraction={fraction_valid:.3f}, "
                f"threshold={threshold})"
            ),
        )

    except Exception as exc:  # noqa: BLE001
        return ValidationResult(
            name=name, passed=False, value=0.0, threshold=threshold,
            detail=f"exception: {exc}",
        )


def check_nontrivial_diversity(
    backend: FlowModelBackend,
    cond: Any,
    schedule: ScheduleSpec,
    rng: np.random.Generator,
    alpha: float,
) -> ValidationResult:
    """Phase 0.2: at the test alpha, K forks show measurable diversity (std > floor).

    value = mean per-dimension std of fork outputs over all dimensions.
    Passes if value > _DIVERSITY_MIN (1e-6).
    """
    name = "nontrivial_diversity"
    threshold = _DIVERSITY_MIN

    try:
        scan_s = 0.5
        K = schedule.K_forks
        traj = generate_trajectory(
            backend, cond, schedule, rng, alpha=0.0, record_points=(0.0, scan_s, 1.0)
        )
        x_s = traj["states"][scan_s]

        g = make_g(schedule.g_kind, schedule.g_value)
        fork_audios = fork_tail(
            backend, x_s, scan_s, cond, alpha=alpha, K=K, schedule=schedule, rng=rng, g=g
        )

        # Stack to (K, dim) and compute per-dim std
        stacked = np.stack(fork_audios, axis=0)  # (K, ...)
        # Compute std across the fork dimension (axis 0)
        per_dim_std = np.std(stacked, axis=0)
        mean_std = float(np.mean(per_dim_std))

        passed = mean_std > threshold
        return ValidationResult(
            name=name,
            passed=passed,
            value=mean_std,
            threshold=threshold,
            detail=(
                f"alpha={alpha}: mean per-dim std of {K} forks = {mean_std:.4e} "
                f"(threshold={threshold:.2e}); "
                f"{'PASS' if passed else 'FAIL — alpha too small to probe diversity'}"
            ),
        )

    except Exception as exc:  # noqa: BLE001
        return ValidationResult(
            name=name, passed=False, value=0.0, threshold=threshold,
            detail=f"exception: {exc}",
        )


def check_marginal_preservation(
    synth_backend: Any,
    cond: Any,
    schedule: ScheduleSpec,
    rng: np.random.Generator,
    alpha: float,
    n: int = 2000,
) -> ValidationResult:
    """Phase 0.2 (synthetic only): empirical marginal matches analytic marginal.

    Draws n independent fork completions starting from s_start=0.3, measures the
    empirical mean and variance of final outputs, and compares to the analytic
    marginal at s=1 (which equals the data distribution mean and variance).

    Only valid for SyntheticGaussianFlow (requires analytic_marginal).

    value = max z-score of |empirical_mean - analytic_mean| / (analytic_std / sqrt(n))
    across dimensions. Passes if max_z < _MARGINAL_TOL (3.0).
    """
    name = "marginal_preservation"
    threshold = _MARGINAL_TOL

    if not hasattr(synth_backend, "analytic_marginal"):
        return ValidationResult(
            name=name, passed=True, value=0.0, threshold=threshold,
            detail="skipped: backend has no analytic_marginal",
        )

    try:
        scan_s = 0.3
        # Analytic marginal at s=1 (t=1) is the data distribution
        analytic_mean, analytic_cov_diag = synth_backend.analytic_marginal(1.0, cond)

        # Generate n trajectories; start each from a fresh prior sample so they are i.i.d.
        # We fork from the prior (s_start=0.0) so no shared x_s — this tests marginal preservation.
        # Each sample: draw prior, integrate 0 -> 1 with the SDE at given alpha.
        g = make_g(schedule.g_kind, schedule.g_value)

        samples: list[np.ndarray] = []
        for _ in range(n):
            # Generate an independent full trajectory at the given alpha
            x_init = synth_backend.sample_prior(cond, rng)
            audio_list = fork_tail(
                synth_backend, x_init, 0.0, cond,
                alpha=alpha, K=1, schedule=schedule, rng=rng, g=g,
            )
            samples.append(audio_list[0])

        empirical = np.stack(samples, axis=0)  # (n, dim)
        emp_mean = np.mean(empirical, axis=0)
        emp_var = np.var(empirical, axis=0)

        # z-score for mean: |emp_mean - analytic_mean| / (sqrt(analytic_cov / n))
        std_of_mean = np.sqrt(np.maximum(analytic_cov_diag, 1e-30) / n)
        z_mean = np.abs(emp_mean - analytic_mean) / std_of_mean
        max_z_mean = float(np.max(z_mean))

        # Also check variance: relative error per dimension
        rel_var_err = np.abs(emp_var - analytic_cov_diag) / np.maximum(analytic_cov_diag, 1e-30)
        max_rel_var = float(np.max(rel_var_err))

        passed = max_z_mean < threshold
        return ValidationResult(
            name=name,
            passed=passed,
            value=max_z_mean,
            threshold=threshold,
            detail=(
                f"n={n} samples, alpha={alpha}: "
                f"max z-score of mean = {max_z_mean:.3f} (threshold={threshold}), "
                f"max relative variance error = {max_rel_var:.3f}"
            ),
        )

    except Exception as exc:  # noqa: BLE001
        return ValidationResult(
            name=name, passed=False, value=float("inf"), threshold=threshold,
            detail=f"exception: {exc}",
        )


# --------------------------------------------------------------------------------------
# Aggregate runner
# --------------------------------------------------------------------------------------
def run_sde_validation(
    backend: FlowModelBackend,
    cond: Any,
    schedule: ScheduleSpec,
    rng: np.random.Generator,
    alpha: float,
) -> tuple[list[ValidationResult], str]:
    """Run all Phase-0.2 SDE validation checks and emit a token.

    Returns (results, token) where token is:
      "OK"                   — alpha=0 reproduces the ODE AND every applicable nonzero-alpha
                               check passes: score conversion exact, small-alpha continuity,
                               fork validity, nontrivial diversity, and marginal preservation
                               (the last two synthetic-only checks return passed=True when
                               the backend cannot provide analytic truth).
      "FIX_SCORE_CONVERSION" — the velocity->score conversion is broken: alpha=0 fails to
                               reproduce the ODE, small-alpha continuity is violated, the
                               score does not match the analytic score, or the SDE does not
                               preserve the marginal.
      "FORK_ALPHA_NO_VALID_OPERATING_POINT" — the conversion is sound but the operating alpha
                               yields invalid forks or no nontrivial diversity.

    Per EXPERIMENT_PLAN.md §0.2: GO_MAPS_PHASE requires the SDE to be validated at alpha=0
    AND nonzero-alpha. The nonzero-alpha validation is small-alpha continuity + fork audio
    validity + nontrivial diversity, so a failure in any of those must NOT return "OK".
    """
    results: list[ValidationResult] = []

    # 1. Phase 0.1: trajectory access
    r_access = check_trajectory_access(backend, cond, schedule, rng)
    results.append(r_access)

    # 2. alpha=0 reproduces ODE (necessary, not sufficient)
    r_alpha0 = check_alpha0_reproduces_ode(backend, cond, schedule, rng)
    results.append(r_alpha0)

    # 3. Score conversion exact (synthetic only)
    r_score = check_score_conversion_exact(backend, cond, schedule, rng)
    results.append(r_score)

    # 4. Small-alpha continuity (tests the score term for nonzero alpha)
    r_cont = check_small_alpha_continuity(backend, cond, schedule, rng)
    results.append(r_cont)

    # 5. Fork validity at the operating alpha
    r_fvalid = check_fork_validity(backend, cond, schedule, rng, alpha=alpha)
    results.append(r_fvalid)

    # 6. Nontrivial diversity at the operating alpha
    r_div = check_nontrivial_diversity(backend, cond, schedule, rng, alpha=alpha)
    results.append(r_div)

    # 7. Marginal preservation (synthetic only)
    r_marg = check_marginal_preservation(backend, cond, schedule, rng, alpha=alpha)
    results.append(r_marg)

    # Emit token per plan §0.2. The token must reflect ALL applicable validation checks,
    # not just alpha=0 + continuity, or a broken nonzero-alpha SDE could still be approved
    # for maps via GO_MAPS_PHASE.
    #
    # score_broken: the velocity->score conversion / SDE drift is wrong. alpha=0 reproduction
    #   and small-alpha continuity test the conversion; the synthetic-only exact-score and
    #   marginal-preservation checks (passed=True when skipped on a real backend) test it
    #   directly. Any failure => FIX_SCORE_CONVERSION.
    score_broken = (
        (not r_alpha0.passed)
        or (not r_cont.passed)
        or (not r_score.passed)
        or (not r_marg.passed)
    )
    # alpha_unusable: conversion is sound but the operating alpha gives invalid forks or no
    # nontrivial diversity => no valid operating point (not a score bug).
    alpha_unusable = (not r_fvalid.passed) or (not r_div.passed)

    if score_broken:
        token = "FIX_SCORE_CONVERSION"
    elif alpha_unusable:
        token = "FORK_ALPHA_NO_VALID_OPERATING_POINT"
    else:
        token = "OK"

    return results, token
