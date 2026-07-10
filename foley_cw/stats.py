"""Statistics utilities for foley_cw (numpy-only; no scipy).

All bootstrap CIs resample at the VIDEO level (bootstrap unit = video), faithfully
implementing the statistical protocol from refine-logs/EXPERIMENT_PLAN.md §3:
  "Bootstrap unit = video (resample videos, not individual measurements)."

Public API
----------
bootstrap_over_videos   -- percentile bootstrap CI over a list of per-video values
first_crossing          -- min s on an (s,values) grid that crosses theta (linear interp)
window_with_ci          -- bootstrap s_commit / s_read crossing with CI
separation_score        -- spread(s_hat across axes) / mean(within-axis CI width)
ordered_non_overlapping -- True if all axis CIs are ordered and non-overlapping
auroc                   -- rank-based AUROC (numpy)
accuracy                -- fraction correctly predicted
threshold_sweep         -- WindowEstimate dict keyed by theta, one per threshold
"""

from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np

from .types import WindowEstimate


# ---------------------------------------------------------------------------
# bootstrap_over_videos
# ---------------------------------------------------------------------------

def bootstrap_over_videos(
    per_video_values: list,
    stat_fn: Callable[[list], float],
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI resampling VIDEOS (list elements) with replacement.

    Parameters
    ----------
    per_video_values : list
        One element per video; each element can be any type accepted by stat_fn.
    stat_fn : callable
        Accepts a list of the same structure as per_video_values, returns float.
    n_boot : int
        Number of bootstrap replicates.
    ci : float
        Coverage level, e.g. 0.95 for 95 % CI.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    (point, lo, hi) -- point estimate on the original data; CI bounds.
    """
    n = len(per_video_values)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))

    point = stat_fn(per_video_values)

    rng = np.random.default_rng(seed)
    boot_stats = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        resample = [per_video_values[i] for i in idx]
        boot_stats[b] = stat_fn(resample)

    alpha = 1.0 - ci
    lo = float(np.nanpercentile(boot_stats, 100.0 * alpha / 2.0))
    hi = float(np.nanpercentile(boot_stats, 100.0 * (1.0 - alpha / 2.0)))
    return (float(point), lo, hi)


# ---------------------------------------------------------------------------
# first_crossing
# ---------------------------------------------------------------------------

def first_crossing(
    s_grid: np.ndarray,
    values: np.ndarray,
    theta: float,
    direction: str = "up",
) -> float:
    """Find the minimum s at which values cross theta (linear interpolation between grid
    points).

    Parameters
    ----------
    s_grid : ndarray, shape (n_s,)
        Progress values, assumed monotonically increasing.
    values : ndarray, shape (n_s,)
        Metric values at each s.
    theta : float
        Threshold to cross.
    direction : "up" | "down"
        "up"  -- first s where values >= theta (threshold crossed from below).
        "down" -- first s where values <= theta (threshold crossed from above).

    Returns
    -------
    float -- interpolated s at first crossing, or np.nan if never crosses.
    """
    s_grid = np.asarray(s_grid, dtype=float)
    values = np.asarray(values, dtype=float)

    if direction == "up":
        crossed = values >= theta
    elif direction == "down":
        crossed = values <= theta
    else:
        raise ValueError(f"direction must be 'up' or 'down', got {direction!r}")

    # If the very first point already satisfies the criterion return it.
    if crossed[0]:
        return float(s_grid[0])

    # Look for the first index where we've crossed; interpolate between i-1 and i.
    for i in range(1, len(values)):
        if np.isnan(values[i]):
            continue
        if crossed[i]:
            # Linear interpolation: find s where values(s) == theta
            v0, v1 = float(values[i - 1]), float(values[i])
            s0, s1 = float(s_grid[i - 1]), float(s_grid[i])
            if v1 == v0:
                # flat segment exactly at threshold — return left endpoint
                return s0
            s_cross = s0 + (theta - v0) / (v1 - v0) * (s1 - s0)
            # Clamp to [s0, s1] for safety.
            return float(np.clip(s_cross, s0, s1))

    return float("nan")


# ---------------------------------------------------------------------------
# window_with_ci
# ---------------------------------------------------------------------------

def window_with_ci(
    per_video_curves: np.ndarray,
    s_grid: np.ndarray,
    theta: float,
    kind: str,
    axis_id: str,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
    min_n: int = 1,
) -> WindowEstimate:
    """Bootstrap the s_commit / s_read crossing over videos (plan §3 protocol).

    The crossing is computed PER VIDEO (on each video's curve), then the point estimate
    and CI are obtained from bootstrap resampling of the resulting per-video s values.

    Parameters
    ----------
    per_video_curves : ndarray, shape (n_videos, n_s)
        Each row is one video's metric curve over s_grid.
    s_grid : ndarray, shape (n_s,)
        Progress grid (monotonically increasing).
    theta : float
        Crossing threshold.
    kind : str
        "commit" or "read" (stored in WindowEstimate.kind).
    axis_id : str
    n_boot : int
    ci : float
    seed : int
    min_n : int
        Minimum number of videos; WindowEstimate.underpowered=True if n_videos < min_n.

    Returns
    -------
    WindowEstimate
    """
    per_video_curves = np.asarray(per_video_curves, dtype=float)
    s_grid = np.asarray(s_grid, dtype=float)

    n_videos = per_video_curves.shape[0]
    underpowered = n_videos < min_n

    if n_videos == 0:
        return WindowEstimate(
            axis_id=axis_id,
            kind=kind,
            s_hat=float("nan"),
            ci_low=float("nan"),
            ci_high=float("nan"),
            n_videos=0,
            underpowered=True,
        )

    # Compute per-video crossing values.
    per_video_s = [
        first_crossing(s_grid, per_video_curves[v], theta, direction="up")
        for v in range(n_videos)
    ]

    def _mean_crossing(video_list: list) -> float:
        vals = [v for v in video_list if not np.isnan(v)]
        if len(vals) == 0:
            return float("nan")
        return float(np.mean(vals))

    point, lo, hi = bootstrap_over_videos(per_video_s, _mean_crossing, n_boot=n_boot,
                                          ci=ci, seed=seed)

    return WindowEstimate(
        axis_id=axis_id,
        kind=kind,
        s_hat=point,
        ci_low=lo,
        ci_high=hi,
        n_videos=n_videos,
        underpowered=underpowered,
    )


# ---------------------------------------------------------------------------
# separation_score
# ---------------------------------------------------------------------------

def separation_score(windows: Dict[str, WindowEstimate]) -> float:
    """Separation of s_hat values across axes relative to CI widths.

    separation_score = spread(s_hat across axes) / mean(within-axis CI width)

    'spread' = max(s_hat) - min(s_hat) across all axes (excluding NaN).
    'within-axis CI width' = ci_high - ci_low for each axis (excluding NaN).

    Returns nan if fewer than 2 non-NaN windows or all CI widths are zero.

    This implements the formula from refine-logs/EXPERIMENT_PLAN.md §3 Phase 3:
        separation_score = spread(s_commit across axes) / mean(within-axis CI width)
    """
    s_hats = []
    ci_widths = []

    for w in windows.values():
        if not np.isnan(w.s_hat):
            s_hats.append(w.s_hat)
        w_ci = w.ci_high - w.ci_low
        if not np.isnan(w_ci):
            ci_widths.append(w_ci)

    if len(s_hats) < 2:
        return float("nan")

    spread = float(np.max(s_hats) - np.min(s_hats))

    # Coincident windows are NOT separated, regardless of CI width. This guard must
    # come BEFORE the zero-width check: otherwise coincident point-windows (spread~0,
    # width 0) would return inf and spuriously satisfy a GO_MAP separation test while
    # STOP_ADSR (windows coincide) also fires — a logical contradiction.
    if spread <= 1e-9:
        return 0.0

    if len(ci_widths) == 0:
        return float("nan")

    mean_width = float(np.mean(ci_widths))
    if mean_width == 0.0:
        # Genuinely distinct point-windows (spread > 0) with zero-width CIs: separation
        # is unbounded. decide_phase3 additionally requires the windows not to be
        # coincident/near-s=1, so this can never co-fire with STOP_ADSR.
        return float("inf")

    return spread / mean_width


# ---------------------------------------------------------------------------
# ordered_non_overlapping
# ---------------------------------------------------------------------------

def ordered_non_overlapping(windows: Dict[str, WindowEstimate]) -> bool:
    """True iff all axis CI intervals are mutually non-overlapping when sorted by s_hat.

    An interval [ci_low_i, ci_high_i] does NOT overlap with [ci_low_j, ci_high_j] iff
    ci_high_i <= ci_low_j (or vice versa). We check the sorted list.

    Windows with NaN s_hat are excluded.
    """
    valid = [w for w in windows.values() if not np.isnan(w.s_hat)]
    if len(valid) < 2:
        return True  # trivially non-overlapping

    sorted_wins = sorted(valid, key=lambda w: w.s_hat)

    for i in range(len(sorted_wins) - 1):
        # [sorted_wins[i].ci_low, sorted_wins[i].ci_high] must be strictly below next
        if sorted_wins[i].ci_high > sorted_wins[i + 1].ci_low:
            return False
    return True


# ---------------------------------------------------------------------------
# bootstrap_gap_ci
# ---------------------------------------------------------------------------

def bootstrap_gap_ci(
    commit_curves: np.ndarray,
    read_curves: np.ndarray,
    s_grid: np.ndarray,
    theta_commit: float,
    theta_read: float,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float, int]:
    """Bootstrap-over-videos CI for gap = s_read - s_commit (plan §3: CIs on gaps too).

    commit_curves, read_curves : ndarray (n_videos, n_s), aligned to s_grid and to the SAME
    video order (so a resample draws each video's commit and readout curve jointly). The gap
    uses the SAME crossing definition as window_with_ci — the per-video crossing is computed
    first, then averaged over videos — so gap_point == s_read.s_hat - s_commit.s_hat. On each
    resample the gap is mean(per-video s_read crossings) - mean(per-video s_commit crossings);
    resamples where either side has no defined crossing are dropped.

    Returns (gap_point, ci_low, ci_high, n_valid_boot). gap_point is NaN when either side has
    no defined crossing over the full sample.
    """
    commit_curves = np.asarray(commit_curves, dtype=float)
    read_curves = np.asarray(read_curves, dtype=float)
    s_grid = np.asarray(s_grid, dtype=float)
    n_videos = commit_curves.shape[0]
    if n_videos == 0 or read_curves.shape[0] != n_videos:
        return (float("nan"), float("nan"), float("nan"), 0)

    # Per-video crossings (NaN where a video never crosses), matching window_with_ci's
    # definition so gap_point == s_read.s_hat - s_commit.s_hat (mean-of-per-video-crossings),
    # NOT crossing-of-the-mean-curve.
    s_commit_v = np.array(
        [first_crossing(s_grid, commit_curves[v], theta_commit, "up") for v in range(n_videos)]
    )
    s_read_v = np.array(
        [first_crossing(s_grid, read_curves[v], theta_read, "up") for v in range(n_videos)]
    )

    def _gap(idx: np.ndarray) -> float:
        cs = s_commit_v[idx]
        rs = s_read_v[idx]
        cs = cs[~np.isnan(cs)]
        rs = rs[~np.isnan(rs)]
        if cs.size == 0 or rs.size == 0:
            return float("nan")
        return float(np.mean(rs) - np.mean(cs))

    point = _gap(np.arange(n_videos))
    rng = np.random.default_rng(seed)
    gaps: list[float] = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n_videos, size=n_videos)
        g = _gap(idx)
        if not np.isnan(g):
            gaps.append(g)
    if len(gaps) < 2:
        return (point, float("nan"), float("nan"), len(gaps))
    lo_q = (1.0 - ci) / 2.0 * 100.0
    hi_q = (1.0 + ci) / 2.0 * 100.0
    lo, hi = np.percentile(gaps, [lo_q, hi_q])
    return (point, float(lo), float(hi), len(gaps))


# ---------------------------------------------------------------------------
# separation_under_thresholds
# ---------------------------------------------------------------------------

def separation_under_thresholds(
    per_axis_commit_curves: Dict[str, np.ndarray],
    s_grid: np.ndarray,
    thetas: list,
    min_n_per_axis: "Dict[str, int] | None" = None,
    n_boot: int = 200,
    ci: float = 0.95,
    seed: int = 0,
) -> Dict[float, dict]:
    """Re-report axis SEPARATION under a theta_commit sweep (plan §3 threshold sensitivity).

    per_axis_commit_curves : dict axis_id -> ndarray (n_videos, n_s) of commitment-gain curves
    at the primary alpha. For each theta the per-axis s_commit windows are recomputed (bootstrap
    over videos) and separation_score / ordered_non_overlapping are reported, so the headline
    separation result can be checked for threshold sensitivity (not just a single axis's s_hat).

    Returns dict theta -> {"separation": float, "ordered_non_overlapping": bool,
                           "windows": dict axis_id -> WindowEstimate}.
    """
    s_grid = np.asarray(s_grid, dtype=float)
    mn = min_n_per_axis or {}
    out: Dict[float, dict] = {}
    for theta in thetas:
        windows: Dict[str, WindowEstimate] = {}
        for axis_id, curves in per_axis_commit_curves.items():
            windows[axis_id] = window_with_ci(
                np.asarray(curves, dtype=float), s_grid, theta=float(theta),
                kind="commit", axis_id=axis_id, n_boot=n_boot, ci=ci, seed=seed,
                min_n=int(mn.get(axis_id, 1)),
            )
        # Separation is reported over RESULT windows only (non-NaN AND not underpowered), so
        # the sensitivity is consistent with the main decision (underpowered axes are not
        # results — plan §3).
        result_windows = {
            a: w for a, w in windows.items()
            if not w.underpowered and not np.isnan(w.s_hat)
        }
        out[float(theta)] = {
            "separation": separation_score(result_windows),
            "ordered_non_overlapping": ordered_non_overlapping(result_windows),
            "n_result_axes": len(result_windows),
            "windows": windows,
        }
    return out


# ---------------------------------------------------------------------------
# auroc
# ---------------------------------------------------------------------------

def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUROC (numpy only; equivalent to Mann-Whitney U / Wilcoxon statistic).

    Parameters
    ----------
    scores : ndarray, shape (n,)
        Predicted scores; higher means more likely to be class 1.
    labels : ndarray, shape (n,)
        Binary labels: 1 for the positive class, 0 for negative.

    Returns
    -------
    float in [0, 1]. 0.5 = random; 1.0 = perfect.

    Implementation: sorts by score descending, then computes the fraction of
    (positive, negative) pairs where the positive has a higher score (with ties
    broken by averaging, i.e. standard AUROC).
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)

    pos_mask = labels == 1
    neg_mask = labels == 0
    n_pos = int(np.sum(pos_mask))
    n_neg = int(np.sum(neg_mask))

    if n_pos == 0 or n_neg == 0:
        return float("nan")

    pos_scores = scores[pos_mask]
    neg_scores = scores[neg_mask]

    # For each positive, count #negatives with strictly lower score
    # plus 0.5 * #negatives with equal score.
    # This is the U statistic / (n_pos * n_neg).
    u_stat = 0.0
    for ps in pos_scores:
        u_stat += float(np.sum(neg_scores < ps))
        u_stat += 0.5 * float(np.sum(neg_scores == ps))

    return u_stat / (n_pos * n_neg)


# ---------------------------------------------------------------------------
# accuracy
# ---------------------------------------------------------------------------

def accuracy(pred: np.ndarray, true: np.ndarray) -> float:
    """Fraction of elements where pred == true (exact match).

    Parameters
    ----------
    pred, true : array-like
        Predicted and ground-truth labels; compared element-wise.

    Returns
    -------
    float in [0, 1], or nan if empty.
    """
    pred = np.asarray(pred)
    true = np.asarray(true)
    if pred.size == 0:
        return float("nan")
    return float(np.mean(pred == true))


# ---------------------------------------------------------------------------
# threshold_sweep
# ---------------------------------------------------------------------------

def threshold_sweep(
    per_video_curves: np.ndarray,
    s_grid: np.ndarray,
    thetas: list,
    kind: str,
    axis_id: str,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
    min_n: int = 1,
) -> Dict[float, WindowEstimate]:
    """Compute window_with_ci for each theta in the sweep grid.

    Parameters
    ----------
    per_video_curves : ndarray, shape (n_videos, n_s)
    s_grid : ndarray, shape (n_s,)
    thetas : list of float
        Threshold values to sweep.
    kind, axis_id, n_boot, ci, seed, min_n :
        Forwarded to window_with_ci.

    Returns
    -------
    dict mapping float(theta) -> WindowEstimate
    """
    results: Dict[float, WindowEstimate] = {}
    for theta in thetas:
        results[float(theta)] = window_with_ci(
            per_video_curves,
            s_grid,
            theta=float(theta),
            kind=kind,
            axis_id=axis_id,
            n_boot=n_boot,
            ci=ci,
            seed=seed,
            min_n=min_n,
        )
    return results
