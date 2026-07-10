"""F-1 cfg-dial — Decision-1 disambiguator (manual §8.3 part b; Fig 2 α*(cfg)).

Pure numpy-only evidence functions for the F-1 dial: "does guidance move the decision
into the seed?". The dial measures, per cfg in {1.0,1.5,2.0,2.5,3.0,4.5}, three quantities
whose pre-registered F-1 predictions (experiment/preregistered/f1_protocol_predictions.md)
are:

  (1) SEED-PREDICTABILITY of class grows with cfg — a (noise, video) -> class linear probe
      (reuse foley_cw.internal_probes.probe_accuracy) trained on the INITIAL noise latent
      x(s=0) predicts the final class better at high cfg (the seed predetermines class =
      mode-locking); summarized over cfg by `seed_predictability`.
  (2) α*(cfg) — the minimum UNLOCKING alpha per cfg (smallest alpha giving measurable tail
      diversity >= diversity_min) — increases with cfg; computed by `alpha_star`.
  (3) The seed SHARE (A_fork(s_min) − A_independent, from determination.build_determination_
      budget) grows monotonically with cfg.

`f1_verdict` combines all three pre-registered predictions into a SUGGESTED token
(F1_SUPPORTED / F1_REFUTED / F1_INCONCLUSIVE) + rationale + predictions_met. This module
ONLY computes the evidence and a suggestion; the PI holds the final F-1 narrative call and
emits the binding token (§1.6/§15.4 — diagnostics never self-emit). numpy-only, no GPU.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

# Suggested-token strings (the PI emits the binding one; these are advisory only).
F1_SUPPORTED = "F1_SUPPORTED"
F1_REFUTED = "F1_REFUTED"
F1_INCONCLUSIVE = "F1_INCONCLUSIVE"

#: Default minimum tail diversity for an alpha to count as "unlocking" (mirrors
#: AlphaGridSpec.diversity_min); a cfg whose largest piloted alpha never reaches it has
#: α* = NaN ("never unlocks within the piloted grid").
DEFAULT_DIVERSITY_MIN = 0.02
#: Min slope (per unit cfg, via least-squares over cfg) to call a trend "increasing"; a
#: small positive floor so flat-with-noise series do not read as monotone.
_TREND_EPS = 1e-3


def _finite_pairs(by_cfg: dict[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """Sorted (cfg, value) arrays over the finite, non-None entries of a cfg-keyed dict."""
    items = [(float(c), float(v)) for c, v in by_cfg.items()
             if v is not None and np.isfinite(v)]
    items.sort()
    if not items:
        return np.array([]), np.array([])
    cfgs = np.array([c for c, _ in items], dtype=float)
    vals = np.array([v for _, v in items], dtype=float)
    return cfgs, vals


def _trend(by_cfg: dict[float, float], eps: float = _TREND_EPS) -> dict:
    """Least-squares slope of value vs cfg + a monotone-nondecreasing flag.

    Returns {slope, increasing, n, lo_cfg, hi_cfg, lo_val, hi_val, delta}. `increasing`
    requires BOTH a positive least-squares slope (> eps) AND that the value at the highest
    cfg exceeds the value at the lowest (endpoint check guards against a single outlier
    tilting the fit). With < 2 finite points slope is NaN and `increasing` is False.
    """
    cfgs, vals = _finite_pairs(by_cfg)
    n = int(cfgs.size)
    if n < 2:
        return {"slope": float("nan"), "increasing": False, "n": n,
                "lo_cfg": float("nan"), "hi_cfg": float("nan"),
                "lo_val": float("nan"), "hi_val": float("nan"), "delta": float("nan")}
    slope = float(np.polyfit(cfgs, vals, 1)[0])
    delta = float(vals[-1] - vals[0])
    increasing = bool(slope > eps and delta > 0.0)
    return {"slope": slope, "increasing": increasing, "n": n,
            "lo_cfg": float(cfgs[0]), "hi_cfg": float(cfgs[-1]),
            "lo_val": float(vals[0]), "hi_val": float(vals[-1]), "delta": delta}


# ---------------------------------------------------------------------------
# (1) seed-predictability of class across cfg
# ---------------------------------------------------------------------------
def seed_predictability(probe_acc_by_cfg: dict[float, float],
                        chance: Optional[float] = None,
                        eps: float = _TREND_EPS) -> dict:
    """Summarize (noise, video) -> class probe accuracy across cfg.

    `probe_acc_by_cfg[cfg]` is the held-out eval accuracy of a linear probe predicting the
    final class from the INITIAL noise latent x(s=0) at that cfg (computed upstream via
    foley_cw.internal_probes.probe_accuracy on the dial's collected (noise, label) pairs).
    High, rising accuracy => the seed increasingly predetermines class (mode-locking).

    Returns {trend (from _trend), grows (alias of trend.increasing), above_chance_at_top}.
    `above_chance_at_top` is True iff the highest-cfg accuracy clears `chance` (when given);
    None-chance leaves it None (caller supplies the probe's class-prior baseline).
    """
    tr = _trend(probe_acc_by_cfg, eps=eps)
    above = None
    if chance is not None:
        # Use the value at the highest finite cfg directly (well-defined even with a single
        # cfg, where _trend's endpoint is NaN because the slope needs >= 2 points).
        cfgs, vals = _finite_pairs(probe_acc_by_cfg)
        if vals.size:
            above = bool(vals[-1] > float(chance))
    return {"trend": tr, "grows": tr["increasing"], "above_chance_at_top": above}


# ---------------------------------------------------------------------------
# (2) α*(cfg) — minimum unlocking alpha per cfg
# ---------------------------------------------------------------------------
def _alpha_star_one(diversity_by_alpha: dict[float, float],
                    diversity_min: float) -> float:
    """Smallest alpha whose tail diversity >= diversity_min; NaN if none qualifies.

    alpha=0 (the deterministic ODE) yields zero diversity by construction, so it can never
    be the unlocking point; the dial reports the first piloted alpha that crosses the floor.
    """
    qualifying = [float(a) for a, d in diversity_by_alpha.items()
                  if d is not None and np.isfinite(d) and float(d) >= diversity_min]
    return min(qualifying) if qualifying else float("nan")


def alpha_star(diversity_by_alpha: dict[float, dict[float, float]],
               diversity_min: float = DEFAULT_DIVERSITY_MIN,
               eps: float = _TREND_EPS) -> dict:
    """Per-cfg minimum unlocking alpha α*(cfg) and its trend over cfg.

    `diversity_by_alpha[cfg][alpha]` = measured tail diversity of K forks at that (cfg,
    alpha) (e.g. fork-final label entropy or 1 − fork agreement; any monotone-in-stochastic
    -spread scalar). For each cfg, α* = smallest alpha clearing `diversity_min`. The
    pre-registered F-1 prediction is α*(cfg) INCREASING (more guidance => more stochastic
    push needed to unlock the seed-locked class).

    Returns {by_cfg: {cfg: α*}, trend: _trend(by_cfg over finite α*), increasing}.
    Cfgs that never unlock (α* = NaN) are dropped from the trend fit but kept in by_cfg.
    """
    by_cfg = {float(c): _alpha_star_one(div, diversity_min)
              for c, div in diversity_by_alpha.items()}
    tr = _trend(by_cfg, eps=eps)
    return {"by_cfg": by_cfg, "trend": tr, "increasing": tr["increasing"],
            "diversity_min": float(diversity_min)}


# ---------------------------------------------------------------------------
# (3) verdict: combine the three pre-registered F-1 predictions
# ---------------------------------------------------------------------------
def f1_verdict(seed_pred_by_cfg: dict[float, float],
               alpha_star_by_cfg: dict[float, float],
               seed_share_by_cfg: dict[float, float],
               diversity_min: Optional[float] = None,
               chance: Optional[float] = None,
               eps: float = _TREND_EPS) -> dict:
    """Suggest an F-1 token from the three pre-registered predictions (PI emits the binding one).

    Inputs are already per-cfg SCALARS (the upstream dial aggregate reduces each curve):
      * seed_pred_by_cfg[cfg]   = (noise, video) -> class probe accuracy at that cfg;
      * alpha_star_by_cfg[cfg]  = α*(cfg), the minimum unlocking alpha (NaN if never unlocks);
      * seed_share_by_cfg[cfg]  = seed share (A_fork(s_min) − A_independent) at that cfg.

    Pre-registered predictions (each must be MET to support F-1):
      P1 seed-predictability grows with cfg;
      P2 α*(cfg) increases with cfg;
      P3 seed share grows monotonically with cfg.

    Suggestion rule (advisory; the PI holds the final call):
      * all three met                      -> F1_SUPPORTED;
      * a prediction's trend is REVERSED
        (slope < −eps, e.g. seed share/seed-pred SHRINKS with cfg — the cfg=4.5
        commitment tension in the manual status line) -> F1_REFUTED;
      * otherwise (flat / mixed / too few cfg points) -> F1_INCONCLUSIVE.

    `predictions_met` carries the per-prediction booleans + trend dicts so the PI sees the
    evidence, not just the suggestion. `chance` (probe class-prior baseline) and
    `diversity_min` are recorded for provenance only.
    """
    seed_pred = seed_predictability(seed_pred_by_cfg, chance=chance, eps=eps)
    # alpha_star_by_cfg is already-reduced α* scalars, so trend them directly (do NOT
    # re-run the per-alpha reduction — that path is exercised by alpha_star()).
    star_trend = _trend(alpha_star_by_cfg, eps=eps)
    share_trend = _trend(seed_share_by_cfg, eps=eps)

    p1 = bool(seed_pred["grows"])                       # seed-predictability grows
    p2 = bool(star_trend["increasing"])                 # α*(cfg) increases
    p3 = bool(share_trend["increasing"])                # seed share grows monotonically
    met = {"seed_predictability_grows": p1, "alpha_star_increases": p2,
           "seed_share_grows": p3}
    trends = {"seed_predictability": seed_pred["trend"], "alpha_star": star_trend,
              "seed_share": share_trend}

    def _reversed(tr: dict) -> bool:
        return bool(np.isfinite(tr["slope"]) and tr["slope"] < -eps and tr["delta"] < 0.0)

    any_reversed = any(_reversed(t) for t in trends.values())

    if p1 and p2 and p3:
        token = F1_SUPPORTED
        rationale = ("All three pre-registered F-1 predictions met: seed-predictability "
                     f"(slope {seed_pred['trend']['slope']:+.3g}), α*(cfg) "
                     f"(slope {star_trend['slope']:+.3g}), and seed share "
                     f"(slope {share_trend['slope']:+.3g}) all increase with cfg — guidance "
                     "moves the decision into the seed.")
    elif any_reversed:
        token = F1_REFUTED
        rev = [name for name, t in trends.items() if _reversed(t)]
        rationale = ("F-1 contradicted: " + ", ".join(rev) + " trends DOWN with cfg "
                     "(seed share/seed-predictability shrinks as guidance rises), opposite "
                     "to the pre-registered direction.")
    else:
        token = F1_INCONCLUSIVE
        rationale = ("Predictions met " f"{sum(met.values())}/3 with no clear reversal "
                     "(flat / mixed / too few cfg points); evidence does not license a "
                     "directional F-1 call.")

    return {"suggested_token": token, "rationale": rationale, "predictions_met": met,
            "trends": trends, "seed_predictability": seed_pred,
            "provenance": {"chance": chance, "diversity_min": diversity_min, "eps": eps,
                           "n_cfg": {"seed_pred": seed_pred["trend"]["n"],
                                     "alpha_star": star_trend["n"],
                                     "seed_share": share_trend["n"]}}}
