"""Phase-1 commitment map — normalized stochastic fork agreement over the video prior.

Scientific contract (refine-logs/EXPERIMENT_PLAN.md §7 Phase 1):
  * A_independent(video, axis) = agreement across N_independent full alpha=0 generations
    from INDEPENDENT initial noise for the same video cond.  This is the video-conditioned
    prior: how tightly the video already determines the axis value WITHOUT any shared
    intermediate state.
  * A_fork(x_s, s, axis, alpha) = agreement across K stochastic tail-forks from the shared
    intermediate state x_s.  Higher than A_independent means x_s has partially committed.
  * Normalized commitment gain:
        commit_gain(s) = clip((A_fork - A_ind) / (1 - A_ind), 0, 1)
    with a guard for A_ind == 1 (already fully committed in prior -> return 0 gain).
  * s_commit(axis) = first s where commit_gain >= theta_commit, bootstrapped over videos.
  * Primary alpha = smallest alpha in pilot_grid with:
        tail diversity >= diversity_min  AND  audio validity >= audio_validity_min
    None if no alpha qualifies -> FORK_ALPHA_NO_VALID_OPERATING_POINT.

Restart re-noising is NOT used (reserved for Phase 6 rollback).
Commitment is NEVER reported raw (always normalized vs A_independent).
Bootstrap unit = video (not individual measurements).
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .agreement import agreement
from .axes import measure_self_target
from .score_sde import fork_tail, generate_trajectory, make_g, x0_at
from .stats import window_with_ci
from .types import (
    AlphaGridSpec,
    Axis,
    AxisKind,
    AxisTier,
    CommitmentCell,
    ScheduleSpec,
    Thresholds,
    WindowEstimate,
)


# ---------------------------------------------------------------------------
# video -> cond normalization
# ---------------------------------------------------------------------------

def _as_cond(video: Any) -> Any:
    """Normalize a 'video' item to its conditioning object.

    Accepts three shapes so callers can pass whatever the dataset layer produces:
      * a dataset dict ``{"video_id", "cond", "anchors"}`` -> returns ``video["cond"]``;
      * an object exposing a ``.cond`` attribute            -> returns ``video.cond``;
      * a bare conditioning object (e.g. ``SyntheticVideoCond``) -> returned unchanged.

    (Replaces an earlier inverted ``hasattr``/subscript expression that passed the
    whole dict through as the cond for dict-shaped videos.)
    """
    if isinstance(video, dict):
        return video.get("cond", video)
    if hasattr(video, "cond"):
        return video.cond
    return video


# ---------------------------------------------------------------------------
# commit_gain
# ---------------------------------------------------------------------------

def commit_gain(a_fork: float, a_ind: float) -> float:
    """Normalized commitment gain: clip((A_fork - A_ind) / (1 - A_ind), 0, 1).

    Guard: if A_ind >= 1.0 the video prior already completely determines the
    axis (A_independent = 1.0 means all independent generations agree), so there
    is no 'reducible' diversity left to commit — return 0.0 rather than divide
    by zero.

    This is the ONLY place commitment gain is computed; all callers use this.
    """
    if a_ind >= 1.0 - 1e-9:
        # Video prior fully determines the axis; gain is undefined / set to 0
        return 0.0
    gain = (a_fork - a_ind) / (1.0 - a_ind)
    return float(np.clip(gain, 0.0, 1.0))


# ---------------------------------------------------------------------------
# a_independent
# ---------------------------------------------------------------------------

def a_independent(
    backend: Any,
    cond: Any,
    axis: Axis,
    measurer: Any,
    schedule: ScheduleSpec,
    rng: np.random.Generator,
) -> float:
    """Agreement of the self-target across N_independent full alpha=0 generations.

    Each generation uses an INDEPENDENT initial noise sample (independent call to
    sample_prior).  Alpha=0 so the fork kernel is the deterministic ODE — diversity
    here comes only from the different starting noise, reflecting the video-conditioned
    prior P(axis | video).

    Returns a float in [0, 1] (or possibly slightly negative for Krippendorff's alpha,
    but commit_gain clamps; see commit_gain docstring).
    """
    N = schedule.N_independent
    targets = []
    for _ in range(N):
        result = generate_trajectory(backend, cond, schedule, rng, alpha=0.0,
                                     record_points=(1.0,))
        audio = result["audio"]
        tgt = measure_self_target(audio, axis, measurer)
        targets.append(tgt)
    return agreement(targets, axis.agreement)


# ---------------------------------------------------------------------------
# a_fork
# ---------------------------------------------------------------------------

def a_fork(
    backend: Any,
    x_s: np.ndarray,
    s: float,
    cond: Any,
    axis: Axis,
    measurer: Any,
    alpha: float,
    schedule: ScheduleSpec,
    rng: np.random.Generator,
) -> float:
    """Agreement of the self-target across K stochastic tail-forks from x_s.

    Forks from the SAME intermediate state x_s; diversity here reflects what is
    NOT yet committed in x_s (stochastic re-completion).  Uses schedule.K_forks forks.
    """
    K = schedule.K_forks
    g = make_g(schedule.g_kind, schedule.g_value)
    audios = fork_tail(backend, x_s, s, cond, alpha, K, schedule, rng, g=g)
    targets = [measure_self_target(audio, axis, measurer) for audio in audios]
    return agreement(targets, axis.agreement)


# ---------------------------------------------------------------------------
# commitment_curve_for_video
# ---------------------------------------------------------------------------

def commitment_curve_for_video(
    backend: Any,
    video: Any,
    axis: Axis,
    measurer: Any,
    alpha: float,
    schedule: ScheduleSpec,
    rng: np.random.Generator,
    return_raw: bool = False,
):
    """Commitment gain curve over schedule.scan_points for one video.

    Algorithm:
      1. Generate the video's trajectory at alpha=0 (deterministic ODE) to get x_s
         at each scan point (the trajectory of this specific video realization).
      2. Compute A_independent for this video (N_independent full independent
         generations; measures the video-conditioned prior).
      3. At each scan point s, compute A_fork from x_s and compute commit_gain.

    Returns shape (len(scan_points),) with commit_gain in [0, 1] when
    ``return_raw`` is False; otherwise returns ``(gains, a_ind, a_fork_per_s)`` so
    callers can record the raw A_fork / A_independent surface the plan asks for in
    ``commitment_map.csv``.

    Scientific note: the trajectory is generated once (alpha=0, deterministic) to
    give x_s as the 'base state' from which we fork.  The commit_gain at each s
    measures how much x_s has narrowed the distribution beyond the video prior.
    """
    cond = _as_cond(video)

    scan_pts = tuple(schedule.scan_points)

    # Step 1: obtain x_s at every scan point via a single alpha=0 trajectory.
    traj = generate_trajectory(backend, cond, schedule, rng, alpha=0.0,
                               record_points=scan_pts)
    states = traj["states"]  # dict: float(s) -> x_s ndarray

    # Step 2: A_independent (video prior) — independent of the trajectory.
    a_ind = a_independent(backend, cond, axis, measurer, schedule, rng)

    # Step 3: commitment gain at each scan point.
    gains = np.empty(len(scan_pts), dtype=float)
    a_fork_per_s = np.empty(len(scan_pts), dtype=float)
    for i, s_val in enumerate(scan_pts):
        s_key = float(s_val)
        x_s = states[s_key]
        a_f = a_fork(backend, x_s, s_key, cond, axis, measurer, alpha, schedule, rng)
        a_fork_per_s[i] = a_f
        gains[i] = commit_gain(a_f, a_ind)

    if return_raw:
        return gains, float(a_ind), a_fork_per_s
    return gains


# ---------------------------------------------------------------------------
# select_primary_alpha
# ---------------------------------------------------------------------------

def _tail_diversity(backend: Any, cond: Any, alpha: float, schedule: ScheduleSpec,
                    rng: np.random.Generator) -> float:
    """Mean pairwise std of fork outputs from s=0.5 as a quick diversity estimate.

    Forks from the midpoint (s=0.5) of a single alpha=0 trajectory; returns the
    mean per-dimension std across K forks as a scalar diversity measure.
    """
    K = schedule.K_forks
    g = make_g(schedule.g_kind, schedule.g_value)

    # Get x at s=0.5 via a short alpha=0 trajectory
    traj = generate_trajectory(backend, cond, schedule, rng, alpha=0.0,
                               record_points=(0.5,))
    x_mid = traj["states"][0.5]

    audios = fork_tail(backend, x_mid, 0.5, cond, alpha, K, schedule, rng, g=g)
    if len(audios) < 2:
        return 0.0
    arr = np.stack(audios, axis=0)  # (K, dim)
    return float(np.mean(np.std(arr, axis=0)))


def _audio_validity(backend: Any, cond: Any, alpha: float, schedule: ScheduleSpec,
                    rng: np.random.Generator) -> float:
    """Fraction of fork completions with non-trivial audio (mean abs > small threshold).

    A fork output is 'valid' if it has nonzero energy (not a zero/NaN output).
    This is the audio-validity guard in the smallest-valid-alpha rule.
    """
    K = schedule.K_forks
    g = make_g(schedule.g_kind, schedule.g_value)

    traj = generate_trajectory(backend, cond, schedule, rng, alpha=0.0,
                               record_points=(0.5,))
    x_mid = traj["states"][0.5]

    audios = fork_tail(backend, x_mid, 0.5, cond, alpha, K, schedule, rng, g=g)
    valid_count = sum(
        1 for a in audios
        if np.isfinite(a).all() and float(np.mean(np.abs(a))) > 1e-6
    )
    return valid_count / max(len(audios), 1)


def select_primary_alpha(
    backend: Any,
    videos: list,
    axes: list[Axis],
    alpha_grid: AlphaGridSpec,
    schedule: ScheduleSpec,
    measurer: Any,
    rng: np.random.Generator,
) -> tuple[Optional[float], dict]:
    """Select the smallest alpha satisfying diversity_min AND audio_validity_min.

    For each alpha in pilot_grid (ascending), estimate tail diversity and audio
    validity across the video bank (using the first video for speed in the pilot).
    Returns (primary_alpha, surface_info) where primary_alpha is None if no alpha
    qualifies -> FORK_ALPHA_NO_VALID_OPERATING_POINT.

    surface_info is a dict: alpha -> {"diversity": float, "validity": float, "qualifies": bool}

    Scientific note: diversity is measured at s=0.5 so the forks have something to
    diverge from; validity is measured on the fork final audios.
    """
    surface_info: dict[float, dict] = {}

    # Use a single representative video for pilot alpha selection (plan §1)
    pilot_videos = videos[:1] if videos else []
    if not pilot_videos:
        return None, surface_info

    pilot_cond = _as_cond(pilot_videos[0])

    for alpha in sorted(alpha_grid.pilot_grid):
        div = _tail_diversity(backend, pilot_cond, alpha, schedule, rng)
        val = _audio_validity(backend, pilot_cond, alpha, schedule, rng)
        qualifies = (div >= alpha_grid.diversity_min) and (val >= alpha_grid.audio_validity_min)
        surface_info[float(alpha)] = {
            "diversity": div,
            "validity": val,
            "qualifies": qualifies,
        }
        if qualifies:
            return float(alpha), surface_info

    # No alpha qualifies
    return None, surface_info


# ---------------------------------------------------------------------------
# build_commitment_map
# ---------------------------------------------------------------------------

def build_commitment_map(
    backend: Any,
    videos: list,
    axes: list[Axis],
    alpha_grid: AlphaGridSpec,
    schedule: ScheduleSpec,
    thresholds: Thresholds,
    measurer: Any,
    rng: np.random.Generator,
    min_n_per_axis: Optional[dict[str, int]] = None,
) -> tuple[list[CommitmentCell], dict[str, WindowEstimate], Optional[float]]:
    """Build the full commitment map A(axis, s, alpha) surface.

    For each axis (excluding EXCLUDED and SEPARATE tiers) and each alpha in
    pilot_grid:
      - compute commitment_curve_for_video for every video;
      - aggregate cells (CommitmentCell rows for commitment_map.csv);
      - compute per-axis WindowEstimate via stats.window_with_ci at the primary alpha.

    The primary alpha is selected first via select_primary_alpha; if None ->
    no window estimates are produced (all NaN) and FORK_ALPHA_NO_VALID_OPERATING_POINT
    should be emitted by the caller.

    Returns:
      cells       -- list of CommitmentCell (all alphas × all axes × all scan points)
      windows     -- dict axis_id -> WindowEstimate at the primary alpha
                     (NaN s_hat if primary_alpha is None or axis never commits)
      primary_alpha -- the selected primary alpha (None if none qualifies)
    """
    # Filter axes: skip EXCLUDED and SEPARATE tiers (not window axes).
    active_axes = [
        ax for ax in axes
        if ax.tier not in (AxisTier.EXCLUDED, AxisTier.SEPARATE)
    ]

    # Select primary alpha first.
    primary_alpha, surface_info = select_primary_alpha(
        backend, videos, active_axes, alpha_grid, schedule, measurer, rng
    )

    scan_pts = list(schedule.scan_points)
    s_grid = np.array(scan_pts, dtype=float)
    n_videos = len(videos)

    cells: list[CommitmentCell] = []
    # windows keyed by axis_id -> WindowEstimate (at primary_alpha)
    windows: dict[str, WindowEstimate] = {}

    for axis in active_axes:
        # Precompute, ONCE per (video, axis): the base alpha=0 trajectory states (the
        # video's realization we fork from) and A_independent (the video-conditioned prior).
        # Both are alpha-INDEPENDENT, so they must be shared across the whole A(axis,s,alpha)
        # surface; recomputing them inside each alpha pass (with an advancing RNG) would let
        # the baseline drift by alpha and contaminate the surface.
        base_states: list[dict] = []
        base_aind = np.empty(n_videos, dtype=float)
        for vi, video in enumerate(videos):
            cnd = _as_cond(video)
            traj = generate_trajectory(backend, cnd, schedule, rng, alpha=0.0,
                                       record_points=tuple(scan_pts))
            base_states.append(traj["states"])
            base_aind[vi] = a_independent(backend, cnd, axis, measurer, schedule, rng)
        mean_a_ind = float(np.mean(base_aind))

        # per_alpha_curves: alpha -> ndarray (n_videos, n_s)
        per_alpha_curves: dict[float, np.ndarray] = {}

        for alpha in sorted(alpha_grid.pilot_grid):
            alpha_f = float(alpha)
            curves = np.empty((n_videos, len(scan_pts)), dtype=float)
            a_fork_curves = np.empty((n_videos, len(scan_pts)), dtype=float)
            for vi, video in enumerate(videos):
                cnd = _as_cond(video)
                states = base_states[vi]
                a_ind_v = float(base_aind[vi])
                for si, s_val in enumerate(scan_pts):
                    s_key = float(s_val)
                    a_f = a_fork(backend, states[s_key], s_key, cnd, axis, measurer,
                                 alpha_f, schedule, rng)
                    a_fork_curves[vi, si] = a_f
                    curves[vi, si] = commit_gain(a_f, a_ind_v)

            per_alpha_curves[alpha_f] = curves

            # Build CommitmentCell rows with the REAL A_fork / A_independent surface
            # (mean over videos), as the plan's commitment_map.csv columns require.
            # A_independent does not depend on s or alpha (it is the alpha=0 video prior),
            # so the same per-axis mean (mean_a_ind, computed once above) is recorded across
            # the row; A_fork is per (s, alpha).
            for si, s_val in enumerate(scan_pts):
                cell = CommitmentCell(
                    axis_id=axis.id,
                    s=float(s_val),
                    alpha=alpha_f,
                    a_fork=float(np.mean(a_fork_curves[:, si])),
                    a_independent=mean_a_ind,
                    commit_gain=float(np.mean(curves[:, si])),
                    n_videos=n_videos,
                )
                cells.append(cell)

        # Compute per-axis WindowEstimate at the primary alpha.
        if primary_alpha is None:
            # No valid alpha -> NaN window
            windows[axis.id] = WindowEstimate(
                axis_id=axis.id,
                kind="commit",
                s_hat=float("nan"),
                ci_low=float("nan"),
                ci_high=float("nan"),
                n_videos=n_videos,
                underpowered=True,
                extra={"reason": "FORK_ALPHA_NO_VALID_OPERATING_POINT"},
            )
        else:
            curves_primary = per_alpha_curves[primary_alpha]
            win = window_with_ci(
                curves_primary,
                s_grid,
                theta=thresholds.theta_commit,
                kind="commit",
                axis_id=axis.id,
                n_boot=200,   # reduced for speed; production uses 1000
                ci=0.95,
                seed=0,
                min_n=int((min_n_per_axis or {}).get(axis.id, 1)),
            )
            # Expose the per-video primary-alpha curves so downstream code can bootstrap
            # gap CIs and threshold-sensitivity separation over the SAME videos (plan §3).
            win.extra["per_video_curves"] = curves_primary
            win.extra["s_grid"] = s_grid
            windows[axis.id] = win

    return cells, windows, primary_alpha
