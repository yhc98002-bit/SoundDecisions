"""Validity sidecar aggregation for the real measurement path (foley_cw/sidecar.py).

Serves manual §3.3 (experiment/LONG_RANGE_EXPERIMENT_PLAN.md, Phase 0.5 reliability
gate): validity is judged against a calibration sidecar of gold labels (MLLM-judged
via foley_cw.mllm_judge and/or human-judged), with Cohen's kappa as the categorical
agreement statistic ("validity κ ≥ 0.6 against a sidecar ...").

Public API:
  * cohens_kappa(a, b)        — nominal Cohen's kappa between two equal-length label
                                lists; NaN for empty input, 1.0 when raters agree on
                                every item, 0.0 guard when expected agreement is 1.
  * run_real_reliability(...) — per-clip determinism/robustness via
                                foley_cw.reliability, validity vs a gold dict
                                (kappa for categorical axes, mean cosine for
                                embedding axes), aggregated by mean and gated with
                                EXACTLY the pass/demotion semantics of
                                reliability.reliability_gate (same thresholds, same
                                AXIS_DEMOTED reason format).

Scientific contract:
  * Determinism/robustness remain self-target repeatability on FIXED audio (never
    fork agreement); only validity consults the gold sidecar.
  * A missing sidecar (gold=None or no overlapping clip ids) yields validity=NaN,
    which the gate counts as a FAILURE — an axis cannot pass validity uncalibrated.
  * Gold labels must share the measurer's label vocabulary; mapping (e.g. the MLLM
    judge's "present"/"absent" vs a detector's 0/1) is the CALLER's responsibility.

numpy-only at import time (no httpx, no torch): safe for the CPU/CI core.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Optional

import numpy as np

from .reliability import determinism, robustness
from .types import Axis, AxisKind, AxisTier, ReliabilityResult, SelfTarget, Thresholds


# ---------------------------------------------------------------------------
# Cohen's kappa (nominal)
# ---------------------------------------------------------------------------

def cohens_kappa(a: list, b: list) -> float:
    """Nominal Cohen's kappa between two equal-length label lists.

    kappa = (p_o - p_e) / (1 - p_e), with p_o the observed per-item agreement and
    p_e the chance agreement from the two raters' marginal label distributions.

    Edge cases:
      * empty input                  -> NaN (undefined, not a free pass)
      * p_o == 1 (incl. both raters constant AND identical) -> 1.0
      * p_e == 1 with p_o < 1        -> 0.0 (div-by-zero guard; degenerate marginals)
    Labels may be any hashable values; ints and strings are NOT conflated.
    """
    if len(a) != len(b):
        raise ValueError(
            f"cohens_kappa: rater lists must have equal length; got {len(a)} vs {len(b)}"
        )
    n = len(a)
    if n == 0:
        return float("nan")

    p_o = sum(1 for x, y in zip(a, b) if x == y) / n
    if p_o == 1.0:
        return 1.0

    counts_a = Counter(a)
    counts_b = Counter(b)
    p_e = sum(
        (counts_a[k] / n) * (counts_b.get(k, 0) / n) for k in counts_a
    )
    denom = 1.0 - p_e
    if denom <= 1e-12:
        return 0.0
    return float((p_o - p_e) / denom)


def gwet_ac1(a: list, b: list) -> float:
    """Gwet's AC1 — a chance-corrected agreement coefficient that is ROBUST to the
    marginal skew that collapses Cohen's kappa (the "kappa paradox").

    AC1 = (p_o - p_e) / (1 - p_e), identical in form to kappa but with a different
    chance term:
        p_e(AC1) = (1 / (q - 1)) * sum_k pi_k * (1 - pi_k)
    where pi_k = (P^a_k + P^b_k) / 2 is the mean of the two raters' marginal
    proportions for category k, and q is the number of categories.

    The contrast with kappa: kappa's p_e = sum_k P^a_k * P^b_k blows UP toward 1 as
    both raters concentrate on one category, so (p_o - p_e)/(1 - p_e) collapses (and
    can go negative) even when p_o is high.  AC1's p_e instead goes to 0 as the
    marginals concentrate (pi_k*(1-pi_k) -> 0), so AC1 -> p_o.  This is exactly the
    regime of the presence/timing axes here (~90% one category): AC1 reports the high
    raw agreement that kappa hides.  Reference: Gwet (2008), BJMSP 61:29-48.

    q is taken as the number of DISTINCT categories observed in the union of the two
    label lists (the realised rating scale); this is documented because AC1's p_e
    depends on q.  Edge cases mirror cohens_kappa:
      * empty input            -> NaN
      * p_o == 1               -> 1.0
      * q <= 1 (one category)  -> 1.0 (everyone agrees on the sole category)
    """
    if len(a) != len(b):
        raise ValueError(
            f"gwet_ac1: rater lists must have equal length; got {len(a)} vs {len(b)}"
        )
    n = len(a)
    if n == 0:
        return float("nan")
    p_o = sum(1 for x, y in zip(a, b) if x == y) / n
    if p_o == 1.0:
        return 1.0

    cats = set(a) | set(b)
    q = len(cats)
    if q <= 1:
        return 1.0

    counts_a = Counter(a)
    counts_b = Counter(b)
    p_e = sum(
        (pi := ((counts_a.get(k, 0) / n) + (counts_b.get(k, 0) / n)) / 2.0) * (1.0 - pi)
        for k in cats
    ) / (q - 1)
    denom = 1.0 - p_e
    if denom <= 1e-12:
        return 0.0
    return float((p_o - p_e) / denom)


def pabak(a: list, b: list) -> float:
    """Prevalence-Adjusted Bias-Adjusted Kappa — the third skew-robust read in the
    §3.3 validity suite.

    PABAK assumes uniform marginals, so it depends ONLY on the observed agreement and
    the category count, sidestepping both the prevalence and the bias that distort
    Cohen's kappa under skew:
        PABAK = (q * p_o - 1) / (q - 1)
    with q = number of categories. For binary (q = 2) this is the familiar 2*p_o - 1.
    Reference: Byrt, Bishop & Carlin (1993), J Clin Epidemiol 46:423-9.

    q is the number of DISTINCT categories observed in the rater union (the same
    diagnostic choice documented for gwet_ac1; a future GATE should preregister the
    full rating-scale q). Edge cases mirror gwet_ac1: empty -> NaN; p_o == 1 -> 1.0;
    q <= 1 -> 1.0.
    """
    if len(a) != len(b):
        raise ValueError(
            f"pabak: rater lists must have equal length; got {len(a)} vs {len(b)}"
        )
    n = len(a)
    if n == 0:
        return float("nan")
    p_o = sum(1 for x, y in zip(a, b) if x == y) / n
    if p_o == 1.0:
        return 1.0
    q = len(set(a) | set(b))
    if q <= 1:
        return 1.0
    return float((q * p_o - 1.0) / (q - 1.0))


def confusion_matrix(a: list, b: list, labels: Optional[list] = None) -> dict:
    """Confusion matrix of two equal-length nominal label lists — the §3.3
    "truth-teller" the manual asks for alongside the chance-corrected scalars.

    Returns a JSON-serialisable dict:
        {"labels": [...], "matrix": [[...], ...], "n": int}
    where matrix[i][j] = #{k : a[k] == labels[i] AND b[k] == labels[j]} (rows = the
    `a` rater, columns = the `b` rater). ``labels`` defaults to the sorted union of
    observed labels (stringified for stable ordering); passing an explicit label set
    fixes the axes (and drops pairs whose label is not in the set).
    """
    if len(a) != len(b):
        raise ValueError(
            f"confusion_matrix: lists must have equal length; got {len(a)} vs {len(b)}"
        )
    if labels is None:
        labels = sorted({str(x) for x in a} | {str(x) for x in b})
    else:
        labels = [str(x) for x in labels]
    index = {lab: i for i, lab in enumerate(labels)}
    mat = [[0 for _ in labels] for _ in labels]
    used = 0
    for x, y in zip(a, b):
        i, j = index.get(str(x)), index.get(str(y))
        if i is None or j is None:
            continue
        mat[i][j] += 1
        used += 1
    return {"labels": labels, "matrix": mat, "n": used}


# ---------------------------------------------------------------------------
# Validity helpers
# ---------------------------------------------------------------------------

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity in [-1, 1]; 0.0 for a zero-norm input (no direction)."""
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))


def _validity_vs_gold(
    axis: Axis,
    wavs_by_clip: dict[str, np.ndarray],
    measurer: Any,
    gold: Optional[dict[str, SelfTarget]],
) -> float:
    """Validity of the measurer against the gold sidecar; NaN without calibration."""
    if not gold:
        return float("nan")
    gold_clips = [cid for cid in wavs_by_clip if cid in gold]
    if not gold_clips:
        return float("nan")

    if axis.kind is AxisKind.EMBEDDING:
        cosines = [
            _cosine(
                measurer.measure(wavs_by_clip[cid], axis).embedding,
                gold[cid].embedding,
            )
            for cid in gold_clips
        ]
        return float(np.mean(cosines))

    measured = [measurer.measure(wavs_by_clip[cid], axis).label for cid in gold_clips]
    gold_labels = [gold[cid].label for cid in gold_clips]
    return cohens_kappa(measured, gold_labels)


# ---------------------------------------------------------------------------
# run_real_reliability
# ---------------------------------------------------------------------------

def run_real_reliability(
    axis: Axis,
    wavs_by_clip: dict[str, np.ndarray],
    measurer: Any,
    thresholds: Thresholds,
    rng: np.random.Generator,
    gold: Optional[dict[str, SelfTarget]] = None,
    perturbations: Optional[dict] = None,
) -> ReliabilityResult:
    """Three-part reliability for the REAL path, gated like reliability_gate.

    Per clip: reliability.determinism and reliability.robustness (custom
    ``perturbations`` passed through); both aggregated by mean over clips.
    Validity: clips present in ``gold`` only — Cohen's kappa of (measured labels,
    gold labels) for categorical axes, mean cosine(measured, gold) for embedding
    axes; NaN when no gold is available.

    The pass/demotion decision MIRRORS reliability.reliability_gate exactly
    (theta_rel / theta_robust / theta_cal; non-finite score = failure; any failure
    -> demoted with an "AXIS_DEMOTED:<axis>" reason; TIER2 requires all three
    strong).  Returns the shared ReliabilityResult dataclass.
    """
    if len(wavs_by_clip) == 0:
        return ReliabilityResult(
            axis_id=axis.id,
            determinism=0.0,
            robustness=0.0,
            validity=0.0,
            passed=False,
            demoted=True,
            reason="no clips provided",
        )

    clips = list(wavs_by_clip.items())

    det_scores = [determinism(measurer, audio, axis) for _cid, audio in clips]
    det_score = float(np.mean(det_scores))

    # Fresh RNG split per clip (same scheme as reliability_gate) for reproducibility.
    rob_scores: list[float] = []
    for i, (_cid, audio) in enumerate(clips):
        rob_rng = np.random.default_rng(int(rng.integers(0, 2**31)) + i)
        rob_scores.append(
            robustness(measurer, audio, axis, rob_rng, perturbations=perturbations)
        )
    rob_score = float(np.mean(rob_scores))

    val_score = _validity_vs_gold(axis, wavs_by_clip, measurer, gold)

    # -- Demotion rule: verbatim mirror of reliability.reliability_gate.
    #    A non-finite score (e.g. validity with no gold sidecar) counts as a FAILURE.
    fails_det = (not np.isfinite(det_score)) or (det_score < thresholds.theta_rel)
    fails_rob = (not np.isfinite(rob_score)) or (rob_score < thresholds.theta_robust)
    fails_val = (not np.isfinite(val_score)) or (val_score < thresholds.theta_cal)
    any_fail = fails_det or fails_rob or fails_val

    is_material = axis.tier is AxisTier.TIER2
    demoted = any_fail
    if is_material and not any_fail:
        demoted = False
    elif is_material and any_fail:
        demoted = True

    passed = not demoted

    reasons: list[str] = []
    if fails_det:
        reasons.append(
            f"determinism {det_score:.3f} < theta_rel {thresholds.theta_rel:.3f}"
        )
    if fails_rob:
        reasons.append(
            f"robustness {rob_score:.3f} < theta_robust {thresholds.theta_robust:.3f}"
        )
    if fails_val:
        reasons.append(
            f"validity {val_score:.3f} < theta_cal {thresholds.theta_cal:.3f}"
        )
    if demoted and not reasons:
        reasons.append("demoted (material/fine-class tier2 demotion policy)")
    reason = "; ".join(reasons) if reasons else ""
    if demoted:
        reason = f"AXIS_DEMOTED:{axis.id}" + (f" — {reason}" if reason else "")

    return ReliabilityResult(
        axis_id=axis.id,
        determinism=det_score,
        robustness=rob_score,
        validity=val_score,
        passed=passed,
        demoted=demoted,
        reason=reason,
    )
