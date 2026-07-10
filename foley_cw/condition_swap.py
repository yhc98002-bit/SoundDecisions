"""Stage R condition-swap intervention (manual §8.1, Fig. 5).

The CLEAN CAUSAL test that disambiguates the three sources of a Foley decision
(seed vs. conditioning vs. entropy-reduction — the PI's Decision 1). At progress
``s`` along a SOURCE clip's deterministic ODE trajectory we replace its
conditioning (CLIP + synchformer + text, i.e. the whole ``MMAudioCond``) with a
DONOR's and complete the deterministic ODE (alpha=0) at the deployed cfg, then
measure the final axes. Two rates per axis per s:

  * follow-rate    = fraction of swaps whose final axis value matches the DONOR's
                     own (unswapped) final value.
  * retention-rate = fraction whose final value matches the SOURCE's own value.

``s_cond(axis)`` = earliest s at which follow-rate < 0.5 (the conditioning has
lost majority control of the axis by that point). Sanity controls: a swap at
s ~= 0 should fully FOLLOW the donor (no source dynamics have run yet); a swap
at s ~= 1 should fully RETAIN the source (the trajectory is essentially done).

This module is the numpy-only / pure-ish layer: (a) the swap-completion helpers
``cond_swap_complete`` / ``cond_interp_complete`` that drive score_sde with a
donor (or interpolated) cond, and (b) the follow/retention/s_cond math and
sanity checks that operate on ALREADY-MEASURED axis values (labels or
embeddings) — no GPU, no MMAudio. The GPU orchestration (generation, measuring,
journaling) lives in scripts/stage_r_cond_swap.py. Mid-trajectory switching is
off-distribution; interpret as steerability, never as a gate (§8.1).
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .score_sde import integrate_segment
from .types import AxisKind, ScheduleSpec

#: Follow-rate threshold defining s_cond(axis): the earliest s at which the
#: conditioning no longer controls a MAJORITY of swaps (manual §8.1).
FOLLOW_MAJORITY = 0.5

#: Default cosine threshold for the EMBEDDING-axis "matches" relation: an
#: embedding "matches" whichever anchor (donor/source) it is closer to AND that
#: cosine exceeds this floor. Tie / both-below -> matches neither.
EMBED_COS_MIN = 0.0


# ======================================================================
# Swap-completion helpers (drive score_sde with a swapped/mixed cond)
# ======================================================================
def cond_swap_complete(backend: Any, x_s_source: np.ndarray, s: float,
                       donor_cond: Any, schedule: ScheduleSpec,
                       convention: Optional[str] = None) -> np.ndarray:
    """Complete the deterministic ODE (alpha=0) from the SOURCE state ``x_s_source``
    at progress ``s`` to s=1 using the DONOR's conditioning, and decode to audio.

    The swap is purely a matter of which ``cond`` object is handed to
    ``integrate_segment`` (the backend passes it straight to ``velocity``), so a
    full conditioning swap = integrate s->1 with ``donor_cond`` instead of the
    source cond. alpha=0 and g=0 make this the deterministic ODE completion (the
    deployed-cfg sampler is whatever ``backend.cfg_strength`` is set to). Returns
    the decoded final audio.
    """
    kw = {} if convention is None else {"convention": convention}
    x1 = integrate_segment(backend, x_s_source, donor_cond, float(s), 1.0,
                           schedule, alpha=0.0, g=lambda _s: 0.0, rng=None, **kw)
    return backend.decode(x1)


def cond_interp_complete(backend: Any, x_s_source: np.ndarray, s: float,
                         mixed_cond: Any, schedule: ScheduleSpec,
                         convention: Optional[str] = None) -> np.ndarray:
    """Identical to ``cond_swap_complete`` but driven by a pre-built INTERPOLATED
    cond (the pre-registered fallback for when full swaps degenerate; §8.1).

    The interpolation itself is backend-specific (it lives in the conditioning
    tensors), so the script builds ``mixed_cond`` via ``mix_cond(...)`` and hands
    it here; this helper only runs the deterministic completion. Kept separate
    from ``cond_swap_complete`` so call sites read as swap vs. interp explicitly.
    """
    return cond_swap_complete(backend, x_s_source, s, mixed_cond, schedule,
                              convention=convention)


def mix_cond(backend: Any, source_cond: Any, donor_cond: Any, w: float) -> Any:
    """Build an interpolated cond ``(1-w)*source + w*donor`` for --mode interp.

    Delegates to ``backend.mix_cond(source_cond, donor_cond, w)`` when the GPU
    backend provides it (the conditioning tensors are model-private), so this
    numpy module never needs to know MMAudio's PreprocessedConditions layout.
    w=0 -> source, w=1 -> donor; the script sweeps w to trace a steerability
    curve. Raises if the backend cannot interpolate conditions.
    """
    fn = getattr(backend, "mix_cond", None)
    if fn is None:
        raise NotImplementedError(
            "interp mode needs backend.mix_cond(source_cond, donor_cond, w); the "
            "active backend cannot interpolate conditioning tensors")
    return fn(source_cond, donor_cond, float(np.clip(w, 0.0, 1.0)))


# ======================================================================
# Pure stats: match relation, follow/retention rates, s_cond, sanity
# ======================================================================
def _cos(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def matches(swapped: Any, anchor: Any, kind: AxisKind, *, other: Any = None,
            embed_cos_min: float = EMBED_COS_MIN) -> bool:
    """Does a swapped final axis value match the ``anchor`` (donor or source)?

    CATEGORICAL: exact label match.
    EMBEDDING:   the swapped embedding "matches" ``anchor`` iff it is STRICTLY
                 closer (by cosine) to ``anchor`` than to ``other`` AND
                 cos(swapped, anchor) exceeds ``embed_cos_min``. ``other`` is the
                 competing anchor (the source when scoring follow, the donor when
                 scoring retention); a cosine TIE matches neither (so a swap is
                 never counted as both follow and retention). If ``other`` is None
                 the nearer-than test is skipped and only the floor applies.
    """
    if kind is AxisKind.CATEGORICAL:
        return swapped == anchor
    if kind is AxisKind.EMBEDDING:
        c_anchor = _cos(swapped, anchor)
        if c_anchor < embed_cos_min:
            return False
        if other is None:
            return True
        return c_anchor > _cos(swapped, other)
    raise ValueError(f"unknown axis kind {kind!r}")


def follow_retention_rates(swapped: list, donor_vals: list, source_vals: list,
                           kind: AxisKind,
                           embed_cos_min: float = EMBED_COS_MIN) -> dict:
    """Follow-rate and retention-rate over a set of (source,donor) swap pairs at ONE s.

    ``swapped[i]`` is the measured final axis value of the i-th swap; ``donor_vals[i]``
    / ``source_vals[i]`` are that pair's donor / source own (unswapped) final
    values (labels for categorical, embedding vectors for embedding axes). All
    three lists are aligned and the same length.

    Returns {"follow", "retention", "neither", "n"}; rates are NaN when n==0.
    follow + retention + neither == 1 (categorical: neither>0 when the swap landed
    on a third label; embedding: neither>0 when it failed the cosine floor / tie).
    """
    n = len(swapped)
    if not (n == len(donor_vals) == len(source_vals)):
        raise ValueError("swapped/donor_vals/source_vals must be the same length")
    if n == 0:
        return {"follow": float("nan"), "retention": float("nan"),
                "neither": float("nan"), "n": 0}
    nf = nr = nn = 0
    for sw, dv, sv in zip(swapped, donor_vals, source_vals):
        f = matches(sw, dv, kind, other=sv, embed_cos_min=embed_cos_min)
        r = matches(sw, sv, kind, other=dv, embed_cos_min=embed_cos_min)
        # donor==source (categorical) is an uninformative pair: BOTH a follow and a
        # retention (the axis value is shared) — correct to count in both.
        nf += int(f)
        nr += int(r)
        # neither counted DIRECTLY per pair (not by subtraction): a pair that matched
        # neither anchor. Computing it as 1-follow-retention is WRONG for a mixed batch —
        # shared-value pairs inflate follow+retention and clip a true neither to 0.
        if not f and not r:
            nn += 1
    return {"follow": float(nf / n), "retention": float(nr / n),
            "neither": float(nn / n), "n": int(n)}


def s_cond(rates_by_s: dict, follow_majority: float = FOLLOW_MAJORITY) -> float:
    """Earliest s with follow-rate < ``follow_majority`` (manual §8.1).

    ``rates_by_s`` maps s -> a dict carrying a "follow" key (e.g. the output of
    ``follow_retention_rates``). s-points whose follow-rate is NaN (no data) are
    skipped. Returns NaN if the conditioning keeps majority control at every s
    (follow never drops below the threshold) — i.e. the axis is conditioning-bound
    across the scanned window.
    """
    for s in sorted(rates_by_s, key=float):
        fr = rates_by_s[s].get("follow", float("nan"))
        if np.isfinite(fr) and fr < follow_majority:
            return float(s)
    return float("nan")


def follow_retention_curves(rates_by_s: dict) -> dict:
    """Reshape {s: rate_dict} into sorted parallel arrays for plotting Fig. 5.

    Returns {"s": [...], "follow": [...], "retention": [...], "neither": [...],
    "n": [...]} sorted ascending in s.
    """
    s_sorted = sorted(rates_by_s, key=float)
    out: dict[str, list] = {"s": [float(s) for s in s_sorted],
                            "follow": [], "retention": [], "neither": [], "n": []}
    for s in s_sorted:
        d = rates_by_s[s]
        out["follow"].append(float(d.get("follow", float("nan"))))
        out["retention"].append(float(d.get("retention", float("nan"))))
        out["neither"].append(float(d.get("neither", float("nan"))))
        out["n"].append(int(d.get("n", 0)))
    return out


def sanity_check(rates_by_s: dict, *, s_low: float = 0.0, s_high: float = 1.0,
                 follow_min_at_low: float = 0.5,
                 retention_min_at_high: float = 0.5,
                 tol: float = 1e-9) -> dict:
    """Pre-registered sanity controls (manual §8.1): swap at s~=0 => full follow,
    swap near s~=1 => full retention.

    Picks the scanned s-point nearest ``s_low`` and nearest ``s_high`` and checks
    follow-rate(low) >= ``follow_min_at_low`` and retention-rate(high) >=
    ``retention_min_at_high``. Returns a dict with the chosen s-points, the rates,
    and per-control + overall pass booleans. ``passed`` is None for a control with
    no usable (NaN) rate rather than a silent False.
    """
    if not rates_by_s:
        return {"low_s": None, "high_s": None, "follow_at_low": float("nan"),
                "retention_at_high": float("nan"), "follow_ok": None,
                "retention_ok": None, "passed": None}
    s_keys = [float(s) for s in rates_by_s]
    low_s = min(s_keys, key=lambda s: abs(s - s_low))
    high_s = min(s_keys, key=lambda s: abs(s - s_high))
    f_low = float(rates_by_s[low_s].get("follow", float("nan")))
    r_high = float(rates_by_s[high_s].get("retention", float("nan")))
    follow_ok = None if not np.isfinite(f_low) else bool(f_low + tol >= follow_min_at_low)
    retention_ok = (None if not np.isfinite(r_high)
                    else bool(r_high + tol >= retention_min_at_high))
    if follow_ok is None or retention_ok is None:
        passed: Optional[bool] = None
    else:
        passed = bool(follow_ok and retention_ok)
    return {"low_s": float(low_s), "high_s": float(high_s),
            "follow_at_low": f_low, "retention_at_high": r_high,
            "follow_ok": follow_ok, "retention_ok": retention_ok,
            "passed": passed}


def summarize_axis(swapped_by_s: dict, donor_by_s: dict, source_by_s: dict,
                   kind: AxisKind, embed_cos_min: float = EMBED_COS_MIN,
                   follow_majority: float = FOLLOW_MAJORITY) -> dict:
    """End-to-end per-axis Fig.-5 summary from raw measured values keyed by s.

    Each of ``swapped_by_s`` / ``donor_by_s`` / ``source_by_s`` maps s -> aligned
    list of measured values (labels or embeddings) over the swap pairs at that s.
    Returns {"rates": {s: rate_dict}, "curves": {...}, "s_cond": float,
    "sanity": {...}}.
    """
    rates_by_s = {
        float(s): follow_retention_rates(
            swapped_by_s[s], donor_by_s[s], source_by_s[s], kind,
            embed_cos_min=embed_cos_min)
        for s in swapped_by_s
    }
    return {
        "rates": rates_by_s,
        "curves": follow_retention_curves(rates_by_s),
        "s_cond": s_cond(rates_by_s, follow_majority=follow_majority),
        "sanity": sanity_check(rates_by_s),
    }
