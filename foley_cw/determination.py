"""Three-share determination budget + Fig-1 taxonomy (manual §4, the paper's lead object).

From the Phase-1 commitment map (per clip, per axis, per s: A_independent, A_fork(s)) this
module computes the determination budget — how each axis's decision is apportioned among
the conditioning, seed, and trajectory sources — and the Fig-1 taxonomy, both bootstrapped
by video (§1.5: the bootstrap unit is the video).

Per clip × axis (shares clipped at 0; s_min = smallest scan point = the seed floor):
    conditioning share = A_independent
    seed share         = A_fork(s_min) − A_independent          (the g₀ seed floor at scale)
    trajectory share(s)= A_fork(s)      − A_fork(s_min)
    residual(s)        = 1 − A_fork(s)
Label axes use commit_gain = clip((A_fork − A_independent)/(1 − A_independent), 0, 1);
embedding axes (material) use the trajectory-share normalization
    commit_traj(s) = clip((A_fork(s) − A_fork(s_min))/(1 − A_fork(s_min)), 0, 1)
with A_independent reported alongside, NEVER as the denominator (§4).

Taxonomy (Fig 1): video-determined (A_independent ≥ 0.9); seed-determined
(commit(s_min) ≥ θ_commit, a dominant seed share); trajectory-early (s_commit ≤ 0.4) /
-late (s_commit ≥ 0.7), descriptive labels on the continuum. numpy-only.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

VIDEO_DETERMINED_MIN = 0.9
TRAJ_EARLY_MAX = 0.4
TRAJ_LATE_MIN = 0.7


def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def s_commit(commit_by_s: dict[float, float], theta_commit: float) -> float:
    """Earliest s with commit_gain ≥ θ_commit; NaN if it never crosses."""
    for s in sorted(commit_by_s):
        v = commit_by_s[s]
        if np.isfinite(v) and v >= theta_commit:
            return float(s)
    return float("nan")


def clip_shares(a_independent: float, a_fork_by_s: dict[float, float],
                is_embedding: bool) -> dict:
    """Per-clip three-share decomposition for one axis (shares clipped at 0)."""
    s_sorted = sorted(a_fork_by_s)
    s_min = s_sorted[0]
    s_max = s_sorted[-1]
    af_min = a_fork_by_s[s_min]
    af_max = a_fork_by_s[s_max]
    seed = _clip01((af_min - a_independent)) if np.isfinite(af_min) and np.isfinite(a_independent) else float("nan")
    traj = _clip01((af_max - af_min)) if np.isfinite(af_max) and np.isfinite(af_min) else float("nan")
    residual = _clip01(1.0 - af_max) if np.isfinite(af_max) else float("nan")
    cond = _clip01(a_independent) if np.isfinite(a_independent) else float("nan")
    # commit curve
    commit = {}
    for s in s_sorted:
        af = a_fork_by_s[s]
        if not np.isfinite(af):
            commit[s] = float("nan"); continue
        if is_embedding:
            denom = 1.0 - af_min
            commit[s] = _clip01((af - af_min) / denom) if (np.isfinite(af_min) and denom > 1e-9) else float("nan")
        else:
            denom = 1.0 - a_independent
            commit[s] = _clip01((af - a_independent) / denom) if (np.isfinite(a_independent) and denom > 1e-9) else float("nan")
    return {"conditioning_share": cond, "seed_share": seed, "trajectory_share": traj,
            "residual": residual, "commit": commit, "s_min": s_min, "s_max": s_max,
            "a_fork_s_min": af_min, "a_fork_s_max": af_max}


def _boot_mean_ci(vals: list[float], n_boot: int, rng: np.random.Generator) -> dict:
    """Mean + 95% CI over clips (bootstrap by video). NaNs dropped."""
    v = np.array([x for x in vals if np.isfinite(x)], dtype=float)
    if v.size == 0:
        return {"mean": float("nan"), "ci95": [float("nan"), float("nan")], "n": 0}
    if v.size == 1:
        return {"mean": float(v[0]), "ci95": [float(v[0]), float(v[0])], "n": 1}
    boots = [float(np.mean(v[rng.integers(0, v.size, v.size)])) for _ in range(n_boot)]
    return {"mean": float(np.mean(v)), "ci95": [float(np.percentile(boots, 2.5)),
            float(np.percentile(boots, 97.5))], "n": int(v.size)}


def build_determination_budget(
    per_clip: dict[str, dict[str, dict[float, float]]],
    a_independent: dict[str, dict[str, float]],
    embedding_axes: set[str],
    theta_commit: float,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict:
    """Aggregate the determination budget + taxonomy over clips for every axis.

    per_clip[axis][clip] = {s: A_fork(s)}; a_independent[axis][clip] = A_independent.
    Returns per-axis {budget shares (mean+CI), s_commit (mean+CI), taxonomy counts}.
    """
    rng = np.random.default_rng(seed)
    out: dict[str, dict] = {}
    for axis, clip_forks in per_clip.items():
        is_emb = axis in embedding_axes
        per = {}
        for clip, af_by_s in clip_forks.items():
            aind = a_independent.get(axis, {}).get(clip, float("nan"))
            per[clip] = clip_shares(aind, af_by_s, is_emb)
        clips = sorted(per)
        budget = {k: _boot_mean_ci([per[c][k] for c in clips], n_boot, rng)
                  for k in ("conditioning_share", "seed_share", "trajectory_share", "residual")}
        scs = {c: s_commit(per[c]["commit"], theta_commit) for c in clips}
        budget_scommit = _boot_mean_ci([scs[c] for c in clips if np.isfinite(scs[c])], n_boot, rng)
        # taxonomy
        tax = {"video_determined": 0, "seed_determined": 0, "trajectory_early": 0,
               "trajectory_late": 0, "trajectory_mid": 0, "never_commits": 0}
        for c in clips:
            aind = a_independent.get(axis, {}).get(c, float("nan"))
            if np.isfinite(aind) and aind >= VIDEO_DETERMINED_MIN:
                tax["video_determined"] += 1
            s_min = per[c]["s_min"]
            commit_smin = per[c]["commit"].get(s_min, float("nan"))
            if np.isfinite(commit_smin) and commit_smin >= theta_commit:
                tax["seed_determined"] += 1
            sc = scs[c]
            if not np.isfinite(sc):
                tax["never_commits"] += 1
            elif sc <= TRAJ_EARLY_MAX:
                tax["trajectory_early"] += 1
            elif sc >= TRAJ_LATE_MIN:
                tax["trajectory_late"] += 1
            else:
                tax["trajectory_mid"] += 1
        out[axis] = {"n_clips": len(clips), "is_embedding": is_emb, "budget": budget,
                     "s_commit": budget_scommit, "taxonomy": tax}
    return out
