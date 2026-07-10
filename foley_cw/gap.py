"""Phase 3 gap/R1-R2 / decision gate logic (foley_cw/gap.py).

This module computes:

  Phase 3:
    * gap(s_read, s_commit) — readout lag behind commitment.
    * r1_r2_crosstab — per (axis, s) classification into R1 (uncommitted), R2
      (committed but not cheaply readable), or early_action (committed & readable).
    * decide_phase3 — emits ALL applicable tokens from {GO_MAP, GO_READOUT,
      GO_RESTRICTED, GO_DIAGNOSTIC, STOP_ADSR, STOP_PROJECT,
      FORK_ALPHA_NO_VALID_OPERATING_POINT}.

  Phase 0 gate:
    * decide_phase0 — emits GO_MAPS_PHASE only when all four conditions hold;
      otherwise emits the appropriate failure token(s).

Decision token semantics (from EXPERIMENT_PLAN.md §7 Phase 3):
  GO_MAP         — axes show separated commitment windows beyond CIs
                   (ordered_non_overlapping AND separation_score > 1.0).
  GO_READOUT     — at least one feasible probe reads the early axes well before
                   the end (s_read << 1 for at least one axis).
  GO_RESTRICTED  — only presence / gross timing show early actionable windows;
                   restricted policy suggested.
  GO_DIAGNOSTIC  — commitment exists but cheap readout lags far behind (R2-dominated).
  STOP_ADSR      — all s_commit coincide or only near s=1; degenerates to scalar
                   DiffRS; publishable NEGATIVE result.
  STOP_PROJECT   — reliability/feasibility failure or fewer than min_reliable_axes.
  FORK_ALPHA_NO_VALID_OPERATING_POINT — alpha_ok is False (no usable operating alpha).

Phase 0 tokens:
  GO_MAPS_PHASE  — trajectory OK, SDE validated, manifest OK, >=min_reliable_axes pass.
  FIX_SCORE_CONVERSION — validation_token == "FIX_SCORE_CONVERSION".
  NO_TRAJECTORY_ACCESS — trajectory_ok is False.
  STOP_PROJECT   — <min_reliable_axes pass, or manifest not OK.

Conventions (plan §2, §3):
  * All s values are in [0, 1] (generation progress).
  * Bootstrap unit = video; CIs on WindowEstimate are over videos.
  * Thresholds are pre-registered; never inspect headline curves before freezing them.
  * ALL applicable tokens are emitted (a good map is not killed by a failed first policy).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .stats import ordered_non_overlapping, separation_score
from .types import GoNoGoDecision, ReliabilityResult, Thresholds, WindowEstimate

# ---------------------------------------------------------------------------
# Constants — separation thresholds for GO_MAP and STOP_ADSR
# ---------------------------------------------------------------------------

# Minimum separation_score to declare GO_MAP (axes spread > 1× mean CI width).
_SEP_SCORE_MIN_GO_MAP: float = 1.0

# s_commit values are "near s=1" if they are >= this threshold (STOP_ADSR test).
_NEAR_S1_THRESHOLD: float = 0.85

# Gap between s_read and s_commit beyond which readout is judged to "lag far behind."
_READOUT_LAG_THRESHOLD: float = 0.15

# s_read ceiling: if s_read is below this for at least one axis/probe pair, a probe
# reads early axes "well before the end."
_EARLY_READ_CEILING: float = 0.8

# Axes whose IDs (or names) suggest "restricted" early-window axes.
_RESTRICTED_AXIS_KEYWORDS: tuple[str, ...] = (
    "presence",
    "timing",
    "gross",
    "onset",
)


# ---------------------------------------------------------------------------
# gap
# ---------------------------------------------------------------------------

def gap(s_read: WindowEstimate, s_commit: WindowEstimate) -> float:
    """gap(axis, probe) = s_read.s_hat − s_commit.s_hat.

    Returns NaN when either estimate is NaN.  A positive gap means the probe
    can only read the committed self-target AFTER the commitment window closes.
    A negative gap (readout before commitment) is valid and means the probe
    reads the axis before it stabilizes — interpret with care.
    """
    if np.isnan(s_read.s_hat) or np.isnan(s_commit.s_hat):
        return float("nan")
    return float(s_read.s_hat - s_commit.s_hat)


# ---------------------------------------------------------------------------
# r1_r2_crosstab
# ---------------------------------------------------------------------------

def r1_r2_crosstab(
    commit_curves: dict[str, np.ndarray],
    readout_curves: dict[str, np.ndarray],
    s_grid: np.ndarray,
    thresholds: Thresholds,
) -> dict[tuple[str, float], str]:
    """Per (axis, s) classification into R1 / R2 / early_action.

    Each entry of the returned dict maps (axis_id, s) -> label where label is:
      "R1"           — uncommitted at s (commit_gain(s) < theta_commit)
      "R2"           — committed but not cheaply readable at s
      "early_action" — committed AND readable at s (the high-value cell)

    Parameters
    ----------
    commit_curves : dict axis_id -> ndarray shape (n_s,)
        Mean commitment gain curve (over videos) at each s in s_grid.
    readout_curves : dict axis_id -> ndarray shape (n_s,)
        Mean readout accuracy/AUROC curve (over videos) at each s in s_grid.
        Keys must be the same as commit_curves (or a subset).
    s_grid : ndarray shape (n_s,)
        Progress values.
    thresholds : Thresholds
        Pre-registered theta_commit and theta_read.

    Returns
    -------
    dict[(axis_id, s_float), str]
    """
    s_grid = np.asarray(s_grid, dtype=float)
    result: dict[tuple[str, float], str] = {}

    for axis_id, commit_curve in commit_curves.items():
        commit_arr = np.asarray(commit_curve, dtype=float)
        read_arr = np.asarray(readout_curves.get(axis_id, np.zeros_like(commit_arr)), dtype=float)

        for idx, s_val in enumerate(s_grid):
            c = float(commit_arr[idx]) if idx < len(commit_arr) else float("nan")
            r = float(read_arr[idx]) if idx < len(read_arr) else float("nan")
            s_key = float(s_val)

            if np.isnan(c) or c < thresholds.theta_commit:
                label = "R1"
            elif np.isnan(r) or r < thresholds.theta_read:
                label = "R2"
            else:
                label = "early_action"

            result[(axis_id, s_key)] = label

    return result


# ---------------------------------------------------------------------------
# decide_phase0
# ---------------------------------------------------------------------------

def decide_phase0(
    validation_token: str,
    reliability: list[ReliabilityResult],
    trajectory_ok: bool,
    manifest_ok: bool,
    min_reliable_axes: int = 3,
) -> GoNoGoDecision:
    """Phase 0 gate: emit GO_MAPS_PHASE or a failure token.

    Conditions for GO_MAPS_PHASE (ALL must hold):
      1. trajectory_ok is True
      2. validation_token == "OK"  (SDE validated alpha=0 AND nonzero-alpha)
      3. manifest_ok is True
      4. len([r for r in reliability if r.passed]) >= min_reliable_axes

    If any condition fails, the appropriate token(s) are emitted.  Multiple
    failure tokens can be present simultaneously; they are listed in priority
    order (most critical first).

    Parameters
    ----------
    validation_token : str
        Token emitted by run_sde_validation: "OK" or "FIX_SCORE_CONVERSION".
    reliability : list[ReliabilityResult]
        Per-axis reliability gate results (Phase 0.5).
    trajectory_ok : bool
        True if Phase 0.1 trajectory-access check passed.
    manifest_ok : bool
        True if the dataset/anchor manifest is ready.
    min_reliable_axes : int
        Minimum number of axes that must pass all three reliability parts.

    Returns
    -------
    GoNoGoDecision
    """
    tokens: list[str] = []
    reasons: list[str] = []

    # 1. Trajectory access
    if not trajectory_ok:
        tokens.append("NO_TRAJECTORY_ACCESS")
        reasons.append("trajectory access failed (extract/resume x_s or compute x0(s) failed)")

    # 2. Score/SDE validation — ANY non-OK validation token blocks GO_MAPS_PHASE.
    # (FIX_SCORE_CONVERSION = conversion broken; FORK_ALPHA_NO_VALID_OPERATING_POINT =
    # nonzero-alpha forks invalid / no diversity. Both mean the SDE is not validated.)
    if validation_token != "OK":
        tokens.append(validation_token)
        reasons.append(
            f"SDE not validated (token={validation_token}): alpha=0 reproduction, "
            "small-alpha continuity, exact score, marginal preservation, fork validity, "
            "or nontrivial diversity did not pass"
        )

    # 3. Manifest/anchor
    if not manifest_ok:
        tokens.append("STOP_PROJECT")
        reasons.append("dataset/anchor manifest not ready or invalid")

    # 4. Reliability gate
    n_passed = sum(1 for r in reliability if r.passed)
    if n_passed < min_reliable_axes:
        demoted = [r.axis_id for r in reliability if r.demoted]
        tokens.append("STOP_PROJECT")
        reasons.append(
            f"only {n_passed}/{len(reliability)} axes passed reliability gate "
            f"(need >= {min_reliable_axes}); demoted: {demoted}"
        )

    # De-duplicate STOP_PROJECT (may have been added twice)
    seen: set[str] = set()
    unique_tokens: list[str] = []
    for t in tokens:
        if t not in seen:
            unique_tokens.append(t)
            seen.add(t)
    tokens = unique_tokens

    # All conditions met -> GO_MAPS_PHASE
    if not tokens:
        tokens = ["GO_MAPS_PHASE"]
        justification = (
            f"trajectory access OK; SDE validation token='{validation_token}'; "
            f"manifest OK; {n_passed}/{len(reliability)} axes passed reliability gate "
            f"(threshold={min_reliable_axes})."
        )
    else:
        justification = " | ".join(reasons)

    return GoNoGoDecision(
        tokens=tokens,
        justification=justification,
        extra={
            "n_reliable_axes": n_passed,
            "min_reliable_axes": min_reliable_axes,
            "validation_token": validation_token,
            "trajectory_ok": trajectory_ok,
            "manifest_ok": manifest_ok,
        },
    )


# ---------------------------------------------------------------------------
# decide_phase3
# ---------------------------------------------------------------------------

def decide_phase3(
    commit_windows: dict[str, WindowEstimate],
    read_windows: dict[tuple[str, str, str], WindowEstimate],
    separation: float,
    thresholds: Thresholds,
    alpha_ok: bool,
) -> GoNoGoDecision:
    """Phase 3 decision gate: emit ALL applicable tokens.

    Examines commitment windows (per axis) and readout windows (per
    (axis_id, probe, target) key) against the pre-registered thresholds, then
    emits every applicable token.  A successful map is NOT killed by a failed
    policy condition.

    Parameters
    ----------
    commit_windows : dict axis_id -> WindowEstimate
        Per-axis commitment windows (s_commit with CI).
    read_windows : dict (axis_id, probe, target) -> WindowEstimate
        Per-(axis, probe, target) readout windows (s_read with CI).
    separation : float
        A separation_score value from the caller. NOTE: it is superseded inside this
        function by a recomputation over result windows only (non-NaN AND not underpowered),
        so an underpowered outlier cannot drive GO_MAP. Kept in the signature for
        backward compatibility / logging.
    thresholds : Thresholds
        Pre-registered theta_commit, theta_read thresholds.
    alpha_ok : bool
        False if commitment.select_primary_alpha found no valid operating point.

    Emitted tokens (all that apply):
      FORK_ALPHA_NO_VALID_OPERATING_POINT  — alpha_ok is False.
      GO_MAP                               — axes show separated commitment windows
                                             beyond CIs (ordered + separation > 1).
      GO_READOUT                           — at least one probe reads early before s=1.
      GO_RESTRICTED                        — only presence/timing axes show early windows.
      GO_DIAGNOSTIC                        — committed but readout lags (R2-dominated).
      STOP_ADSR                            — all s_commit coincide or all >= _NEAR_S1.
      STOP_PROJECT                         — no valid windows at all.

    Returns
    -------
    GoNoGoDecision with .tokens containing all applicable tokens.
    """
    tokens: list[str] = []
    reasons: list[str] = []
    extra: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # 0. alpha validity
    # ------------------------------------------------------------------
    if not alpha_ok:
        tokens.append("FORK_ALPHA_NO_VALID_OPERATING_POINT")
        reasons.append(
            "no valid operating alpha: smallest alpha with diversity >= diversity_min "
            "AND audio-validity >= audio_validity_min not found"
        )
        extra["alpha_ok"] = False

    # ------------------------------------------------------------------
    # Filter to valid commitment windows. A window counts as a RESULT only if it
    # crossed the threshold (non-NaN s_hat) AND is not underpowered. Per plan §3, an
    # axis below its minimum usable n is reported as underpowered, NOT as a result, so
    # underpowered windows must not drive GO_MAP / GO_READOUT / etc.
    # ------------------------------------------------------------------
    valid_commit = {
        k: w for k, w in commit_windows.items()
        if not np.isnan(w.s_hat) and not w.underpowered
    }
    n_commit = len(valid_commit)
    all_s_commit = [w.s_hat for w in valid_commit.values()]

    # Recompute separation over RESULT windows ONLY (valid_commit already excludes NaN and
    # underpowered). This supersedes the `separation` argument so an underpowered outlier in
    # the caller's window set cannot inflate the separation used for GO_MAP (plan §3:
    # underpowered axes are not results). ordered_non_overlapping below is likewise computed
    # over valid_commit.
    separation = separation_score(valid_commit)

    # Window-spread descriptors, computed once and shared by the STOP_ADSR and GO_MAP
    # branches so the two can never contradict each other. "coincide" = spread negligible
    # (<= 0.05); "all_near_s1" = every window at/after _NEAR_S1_THRESHOLD.
    if n_commit >= 1:
        spread = float(np.max(all_s_commit) - np.min(all_s_commit))
        coincide = spread <= 0.05
        all_near_s1 = all(s >= _NEAR_S1_THRESHOLD for s in all_s_commit)
    else:
        spread = float("nan")
        coincide = False
        all_near_s1 = False

    # ------------------------------------------------------------------
    # 1. STOP_ADSR — all s_commit coincide or only near s=1
    # ------------------------------------------------------------------
    if n_commit == 0:
        # No usable windows at all (none crossed theta, or all underpowered).
        n_underpowered = sum(1 for w in commit_windows.values() if w.underpowered)
        n_total = len(commit_windows)
        tokens.append("STOP_PROJECT")
        reasons.append(
            f"no usable commitment windows: 0/{n_total} axes produced a valid result "
            f"({n_underpowered} underpowered below min usable n; "
            "the rest never crossed theta_commit)"
        )
    elif coincide or all_near_s1:
        tokens.append("STOP_ADSR")
        if coincide:
            reasons.append(
                f"all s_commit are coincident (spread={spread:.3f}; degenerates to scalar DiffRS)"
            )
        if all_near_s1:
            reasons.append(
                f"all s_commit are near s=1 (all >= {_NEAR_S1_THRESHOLD}; degenerates to scalar DiffRS)"
            )

    # ------------------------------------------------------------------
    # 2. GO_MAP — axes show separated commitment windows beyond CIs.
    # Mutually exclusive with STOP_ADSR by construction: separated windows cannot
    # also be coincident / all-near-s=1.
    # ------------------------------------------------------------------
    if n_commit >= 2 and not coincide and not all_near_s1:
        non_overlapping = ordered_non_overlapping(valid_commit)
        well_separated = (
            not np.isnan(separation) and separation > _SEP_SCORE_MIN_GO_MAP
        )
        if non_overlapping and well_separated:
            tokens.append("GO_MAP")
            reasons.append(
                f"commitment windows are ordered and non-overlapping; "
                f"separation_score={separation:.3f} > {_SEP_SCORE_MIN_GO_MAP}"
            )
            extra["separation_score"] = separation
            extra["ordered_non_overlapping"] = True

    # ------------------------------------------------------------------
    # 3. GO_READOUT — at least one feasible probe reads early axis well before end
    # ------------------------------------------------------------------
    valid_read = {
        k: w for k, w in read_windows.items()
        if not np.isnan(w.s_hat) and not w.underpowered
    }
    early_read_pairs: list[tuple[str, str, str]] = []
    for (axis_id, probe, target), w in valid_read.items():
        if axis_id not in valid_commit:
            continue
        c_w = valid_commit[axis_id]
        # GO_READOUT means a probe reads an EARLY axis well before the end. Per plan §2/§3,
        # reading an uncommitted or late-committing axis is NOT a decided-axis readout:
        #   (a) the axis must commit early (s_commit < _NEAR_S1_THRESHOLD) — else there is no
        #       "early axis" to read (this also keeps GO_READOUT exclusive of an all-near-s=1
        #       STOP_ADSR);
        #   (b) the probe must read well before the end (s_read < _EARLY_READ_CEILING);
        #   (c) it must read a COMMITTED axis, not an uncommitted path, and not lag far
        #       behind: -0.05 <= (s_read - s_commit) <= _READOUT_LAG_THRESHOLD. A large
        #       positive gap is GO_DIAGNOSTIC, not GO_READOUT.
        if c_w.s_hat >= _NEAR_S1_THRESHOLD:
            continue
        if w.s_hat >= _EARLY_READ_CEILING:
            continue
        read_gap = w.s_hat - c_w.s_hat
        if read_gap < -0.05 or read_gap > _READOUT_LAG_THRESHOLD:
            continue
        early_read_pairs.append((axis_id, probe, target))

    if early_read_pairs:
        tokens.append("GO_READOUT")
        reasons.append(
            f"{len(early_read_pairs)} (axis, probe, target) pair(s) achieve "
            f"readout before s={_EARLY_READ_CEILING}: {early_read_pairs[:3]}"
        )
        extra["early_read_pairs"] = early_read_pairs

    # ------------------------------------------------------------------
    # 4. GO_RESTRICTED — only presence/timing axes show early windows
    # ------------------------------------------------------------------
    # Check whether early axes are exclusively presence/timing.
    if valid_commit:
        early_axes = [
            axis_id for axis_id, w in valid_commit.items()
            if w.s_hat < _NEAR_S1_THRESHOLD
        ]
        restricted_early = [
            ax for ax in early_axes
            if any(kw in ax.lower() for kw in _RESTRICTED_AXIS_KEYWORDS)
        ]
        non_restricted_early = [ax for ax in early_axes if ax not in restricted_early]
        # "Actionable" requires the axis to be READABLE early, not merely committed early —
        # reading a committed axis is what licenses a restricted policy (plan §3). Reuse the
        # GO_READOUT pairs (computed above) as the set of early-readable axes.
        early_read_axis_ids = {p[0] for p in early_read_pairs}
        restricted_actionable = [ax for ax in restricted_early if ax in early_read_axis_ids]

        if restricted_actionable and not non_restricted_early:
            tokens.append("GO_RESTRICTED")
            reasons.append(
                f"only presence/timing axes show early ACTIONABLE (readable) windows: "
                f"{restricted_actionable}; no higher-order axis is early — restricted policy"
            )
            extra["restricted_early_axes"] = restricted_actionable

    # ------------------------------------------------------------------
    # 5. GO_DIAGNOSTIC — committed but readout lags far behind (R2-dominated)
    # ------------------------------------------------------------------
    lagging_pairs: list[tuple[str, str, str]] = []
    for (axis_id, probe, target), r_w in valid_read.items():
        if axis_id not in valid_commit:
            continue
        c_w = valid_commit[axis_id]
        g = gap(r_w, c_w)
        if not np.isnan(g) and g > _READOUT_LAG_THRESHOLD:
            lagging_pairs.append((axis_id, probe, target))

    # Also flag cases where commitment exists but NO readout window appears before s=1
    committed_no_read: list[str] = []
    committed_axes = set(valid_commit.keys())
    axes_with_read = {k[0] for k in valid_read.keys()}
    for ax in committed_axes:
        c_w = valid_commit[ax]
        if not np.isnan(c_w.s_hat) and c_w.s_hat < _NEAR_S1_THRESHOLD:
            ax_reads = [
                (k, w) for k, w in valid_read.items()
                if k[0] == ax and not np.isnan(w.s_hat)
            ]
            if not ax_reads:
                committed_no_read.append(ax)

    if lagging_pairs or committed_no_read:
        tokens.append("GO_DIAGNOSTIC")
        reasons.append(
            f"readout lags behind commitment (gap > {_READOUT_LAG_THRESHOLD}) "
            f"for {len(lagging_pairs)} pairs; {len(committed_no_read)} committed "
            "axes have no readable probe before end — R2-dominated; publish gap + "
            "probe-limitation, motivate internal probes (Phase 7)"
        )
        extra["lagging_pairs"] = lagging_pairs
        extra["committed_no_read_axes"] = committed_no_read

    # ------------------------------------------------------------------
    # Deduplicate tokens (preserve order)
    # ------------------------------------------------------------------
    seen_t: set[str] = set()
    dedup_tokens: list[str] = []
    for t in tokens:
        if t not in seen_t:
            dedup_tokens.append(t)
            seen_t.add(t)
    tokens = dedup_tokens

    # ------------------------------------------------------------------
    # Fallback: STOP_PROJECT if nothing positive fires and no STOP_ADSR
    # ------------------------------------------------------------------
    positive_tokens = {
        "GO_MAP", "GO_READOUT", "GO_RESTRICTED", "GO_DIAGNOSTIC", "STOP_ADSR"
    }
    if not any(t in positive_tokens for t in tokens) and "STOP_PROJECT" not in tokens:
        if not alpha_ok:
            # Already have FORK_ALPHA token; don't double-emit STOP_PROJECT yet
            pass
        else:
            tokens.append("STOP_PROJECT")
            reasons.append(
                "no commitment or readout windows found and no specific failure token applies"
            )

    justification = " | ".join(reasons) if reasons else "no issues detected"

    return GoNoGoDecision(
        tokens=tokens,
        justification=justification,
        thresholds=thresholds,
        extra=extra,
    )
