"""Gate A — seed-marginalized fork-kernel exchangeability (revised manual 1.2).

The velocity->score identity is exact only at cfg=1.0; at cfg>1 the CFG-mixed
velocity yields a tilted pseudo-score. Gate A tests, per (cfg, schedule, model)
tuple, whether the fork kernel preserves the conditional marginal — WITHOUT the
first-run mistake of per-seed-cell embedding tests (the seed legitimately
fingerprints fine texture even at the exact kernel, so those conflate
seed-conditioning with kernel error; they are now forbidden by the spec).

Design (pressure-tested; frozen interpretations #1/#6 in
experiment/preregistered/stage_m_rerun_interpretations.md):

  * SAMPLING — one fork tail per independent: at test point s, fork ONE tail from
    each of the N independents' states x_s; pool {fork_i} and permutation-test
    against N FRESH reference independents (generated for this purpose; textbook
    exchangeability, no prefix pairing with the reference set).
  * FEATURE SPACE — sqrt-transformed 527-dim tagger probability vectors
    (granularity matched to the science; sqrt equalizes per-coordinate SNR
    without the logit blow-up on near-zero probs), plus label-marginal total
    variation on the EXTENDED alphabet (event coarse classes + 'other' +
    'abstain' — mode-locking onto abstain is a marginal break).
  * cfg=1.0 (HARD, internal null): the kernel is exact, so cells are null draws;
    per test s-point at most LOW_P_MAX_CELLS of n_clips cells may have p < 0.05
    (Binomial(16, 0.05): P(>=3) = 4.3%). Uniformity of high p-values is NOT
    required. Guards (power positive control, half-split null sanity, threshold
    separation) must be healthy. Failure -> STOP-level instrument review.
  * cfg=4.5 (adjudicated + reported, NOT a Stage-M pass requirement): judged
    against thresholds calibrated per s from the cfg=1.0 cells — (i) one-sided
    Mann-Whitney vs the 1.0 cells (p >= 0.01); (ii) exceedance over the 95th-pct
    threshold <= EXCEEDANCE_MAX_CELLS of n_clips; (iii) no gross localized cell
    (> GROSS_FACTOR x threshold; TV additionally capped at 0.9). Both statistics
    (MMD, TV) at both test s-points must hold for CFG_KERNEL_OK.

Tokens: CFG_KERNEL_OK(cfg=x[, schedule=g]) / CFG_KERNEL_FAIL(...) /
GATE_A_UNDERPOWERED. Guards never pass silently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

#: Significance level for permutation-rejection counting.
ALPHA_SIG = 0.05
#: cfg=1.0 internal null: max low-p cells per s-point out of 16 (Binom guard).
LOW_P_MAX_CELLS = 2
#: cfg=4.5 calibrated: max cells above the 95th-pct threshold out of 16.
EXCEEDANCE_MAX_CELLS = 3
#: Gross localized break: any single cell above this multiple of the threshold.
GROSS_FACTOR = 3.0
#: TV is bounded in [0,1]; the gross rule is additionally capped here.
TV_GROSS_CAP = 0.9
#: Guard g1: minimum fraction of cross-clip pairs the MMD test must reject.
POWER_MIN_REJECT_FRAC = 0.80
#: Guard g2: minimum KS-uniformity p for half-split null p-values (rough at 4v4).
NULL_KS_MIN_P = 0.01
#: Guard g3: threshold must sit below this fraction of the cross-clip MMD level.
THETA_SEPARATION_MAX_FRAC = 0.9
#: Mann-Whitney distribution-shift significance for the calibrated verdict.
MW_MIN_P = 0.01


# --------------------------------------------------------------------------------------
# Feature transform
# --------------------------------------------------------------------------------------
def sqrt_probs(P: np.ndarray) -> np.ndarray:
    """Elementwise sqrt of tagger probability vectors (frozen interpretation #6:
    equalizes per-coordinate SNR, Var(sqrt p) ~ Var(p)/4p, without the logit
    variance blow-up on the ~500 near-zero coordinates)."""
    return np.sqrt(np.clip(np.asarray(P, dtype=float), 0.0, None))


# --------------------------------------------------------------------------------------
# MMD^2 and permutation test (unchanged machinery)
# --------------------------------------------------------------------------------------
def _pairwise_sq_dists(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    a2 = np.sum(A * A, axis=1)[:, None]
    b2 = np.sum(B * B, axis=1)[None, :]
    return np.maximum(a2 + b2 - 2.0 * (A @ B.T), 0.0)


def median_heuristic_bandwidth(X: np.ndarray, Y: np.ndarray) -> float:
    Z = np.vstack([X, Y])
    d2 = _pairwise_sq_dists(Z, Z)
    iu = np.triu_indices(Z.shape[0], k=1)
    med = float(np.median(np.sqrt(d2[iu])))
    return med if med > 0.0 else 1.0


def mmd2_unbiased(X: np.ndarray, Y: np.ndarray, bandwidth: Optional[float] = None) -> float:
    """Unbiased RBF MMD^2; can be negative near the null (never clipped — the
    calibration sees the same estimator)."""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    m, n = X.shape[0], Y.shape[0]
    if m < 2 or n < 2:
        raise ValueError(f"mmd2_unbiased needs >=2 samples per side, got {m}, {n}")
    h = bandwidth if bandwidth is not None else median_heuristic_bandwidth(X, Y)
    gamma = 1.0 / (2.0 * h * h)
    Kxx = np.exp(-gamma * _pairwise_sq_dists(X, X))
    Kyy = np.exp(-gamma * _pairwise_sq_dists(Y, Y))
    Kxy = np.exp(-gamma * _pairwise_sq_dists(X, Y))
    np.fill_diagonal(Kxx, 0.0)
    np.fill_diagonal(Kyy, 0.0)
    return float(Kxx.sum() / (m * (m - 1)) + Kyy.sum() / (n * (n - 1)) - 2.0 * Kxy.mean())


def mmd_permutation_p(X: np.ndarray, Y: np.ndarray, n_perm: int = 200,
                      rng: Optional[np.random.Generator] = None,
                      bandwidth: Optional[float] = None) -> tuple[float, float]:
    """(mmd2_observed, permutation p); bandwidth fixed across permutations."""
    rng = rng or np.random.default_rng(0)
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    h = bandwidth if bandwidth is not None else median_heuristic_bandwidth(X, Y)
    obs = mmd2_unbiased(X, Y, bandwidth=h)
    Z = np.vstack([X, Y])
    m = X.shape[0]
    count = 0
    for _ in range(int(n_perm)):
        idx = rng.permutation(Z.shape[0])
        if mmd2_unbiased(Z[idx[:m]], Z[idx[m:]], bandwidth=h) >= obs:
            count += 1
    return obs, float((count + 1.0) / (n_perm + 1.0))


# --------------------------------------------------------------------------------------
# Label-marginal TV on the extended alphabet
# --------------------------------------------------------------------------------------
def label_marginal_tv(labels_a: Sequence, labels_b: Sequence) -> float:
    """Total-variation distance of the label marginals; abstain is a category
    (extended alphabet) — mode-locking onto abstain is a marginal break."""
    a, b = list(labels_a), list(labels_b)
    support = sorted(set(a) | set(b), key=repr)
    if not support:
        return float("nan")
    pa = np.array([a.count(c) for c in support], dtype=float)
    pb = np.array([b.count(c) for c in support], dtype=float)
    pa /= max(pa.sum(), 1.0)
    pb /= max(pb.sum(), 1.0)
    return float(0.5 * np.abs(pa - pb).sum())


# --------------------------------------------------------------------------------------
# Cells, thresholds, results
# --------------------------------------------------------------------------------------
@dataclass
class GateACell:
    """Seed-marginalized exchangeability statistics for one (clip, s, cfg) cell:
    pooled one-fork-per-independent finals vs FRESH reference independents."""

    clip_id: str
    s: float
    cfg: float
    mmd2: float          # on sqrt tagger-prob vectors
    p_value: float       # permutation p for the MMD
    tv: float            # label-marginal TV, extended alphabet
    n_fork: int = 0
    n_ref: int = 0
    schedule: str = "constant"
    extra: dict = field(default_factory=dict)


def build_cell(clip_id: str, s: float, cfg: float,
               fork_probs: np.ndarray, ref_probs: np.ndarray,
               fork_labels: Sequence, ref_labels: Sequence,
               rng: Optional[np.random.Generator] = None, n_perm: int = 200,
               schedule: str = "constant") -> GateACell:
    """Compute one cell's statistics from raw 527-dim prob matrices + labels."""
    X, Y = sqrt_probs(fork_probs), sqrt_probs(ref_probs)
    mmd2, p = mmd_permutation_p(X, Y, n_perm=n_perm, rng=rng)
    return GateACell(clip_id=clip_id, s=float(s), cfg=float(cfg), mmd2=mmd2, p_value=p,
                     tv=label_marginal_tv(fork_labels, ref_labels),
                     n_fork=int(np.asarray(fork_probs).shape[0]),
                     n_ref=int(np.asarray(ref_probs).shape[0]), schedule=schedule)


@dataclass
class GateAThresholds:
    """Per-s thresholds calibrated from the cfg=1.0 cells (the same-design null)."""

    theta_mmd: dict[str, float]              # s -> 95th pct of null MMD^2
    theta_tv: dict[str, float]               # s -> 95th pct of null TV
    null_mmd: dict[str, list[float]]         # s -> raw null values (for MW)
    null_tv: dict[str, list[float]]
    n_ref_cells: int = 0
    source: str = "cfg=1.0 same-design internal null"


@dataclass
class GateAResult:
    token: str
    passed: bool
    cfg: float
    schedule: str = "constant"
    detail: str = ""
    guards: dict = field(default_factory=dict)
    per_s: dict = field(default_factory=dict)   # s -> {checks, stats}
    extra: dict = field(default_factory=dict)


def _skey(s: float) -> str:
    return f"{s:g}"


# --------------------------------------------------------------------------------------
# Guards (computed on sqrt-prob vectors of independents)
# --------------------------------------------------------------------------------------
def power_positive_control(ref_probs_by_clip: dict[str, np.ndarray],
                           rng: Optional[np.random.Generator] = None,
                           n_perm: int = 200, max_pairs: int = 60
                           ) -> tuple[float, float, float]:
    """g1: cross-clip reference pools come from different conditionals; the test
    must reject for >= POWER_MIN_REJECT_FRAC of pairs or it certifies nothing.
    Returns (reject fraction, median cross-clip MMD^2, 95th-pct cross-clip MMD^2)."""
    rng = rng or np.random.default_rng(0)
    clips = sorted(ref_probs_by_clip)
    pairs = [(a, b) for i, a in enumerate(clips) for b in clips[i + 1:]]
    if len(pairs) > max_pairs:
        idx = rng.choice(len(pairs), size=max_pairs, replace=False)
        pairs = [pairs[i] for i in idx]
    rejects, mmds = [], []
    for a, b in pairs:
        obs, p = mmd_permutation_p(sqrt_probs(ref_probs_by_clip[a]),
                                   sqrt_probs(ref_probs_by_clip[b]),
                                   n_perm=n_perm, rng=rng)
        rejects.append(p < ALPHA_SIG)
        mmds.append(obs)
    if not mmds:
        return float("nan"), float("nan"), float("nan")
    return (float(np.mean(rejects)), float(np.median(mmds)),
            float(np.percentile(mmds, 95)))


def null_sanity(ref_probs_by_clip: dict[str, np.ndarray],
                rng: Optional[np.random.Generator] = None,
                n_perm: int = 200) -> tuple[float, list[float]]:
    """g2: half-split ref-vs-ref p-values ~ uniform (KS; rough at 4-vs-4 — only
    35 distinct permutation values — logged as a caveat, threshold lenient)."""
    from scipy.stats import kstest

    rng = rng or np.random.default_rng(0)
    pvals: list[float] = []
    for clip in sorted(ref_probs_by_clip):
        E = sqrt_probs(ref_probs_by_clip[clip])
        n = E.shape[0]
        if n < 4:
            continue
        idx = rng.permutation(n)
        half = n // 2
        _, p = mmd_permutation_p(E[idx[:half]], E[idx[half:half * 2]], n_perm=n_perm, rng=rng)
        pvals.append(p)
    if len(pvals) < 5:
        return float("nan"), pvals
    return float(kstest(pvals, "uniform").pvalue), pvals


def check_guards(power_reject_frac: float, cross_clip_mmd_median: float,
                 null_ks_p: float, theta_mmd: dict[str, float],
                 cross_clip_mmd_p95: float = float("nan")) -> tuple[bool, dict, str]:
    """g3 (refined per frozen interpretation #10, BEFORE run 3): theta_mmd[s] is
    compared to the cross-clip 95TH PERCENTILE — raw 8v8 MMD magnitudes in the
    sqrt-prob space are noisy and discrimination is p-based (g1); the median
    comparison mis-fired on run 2 while g1/g2 were healthy. Both readings are
    reported in the guards dict."""
    guards = {"power_reject_frac": power_reject_frac,
              "cross_clip_mmd_median": cross_clip_mmd_median,
              "cross_clip_mmd_p95": cross_clip_mmd_p95,
              "null_ks_p": null_ks_p}
    ref_level = cross_clip_mmd_p95 if np.isfinite(cross_clip_mmd_p95) else cross_clip_mmd_median
    reasons = []
    if not (np.isfinite(power_reject_frac) and power_reject_frac >= POWER_MIN_REJECT_FRAC):
        reasons.append(f"g1 power {power_reject_frac:.2f} < {POWER_MIN_REJECT_FRAC}")
    if not (np.isfinite(null_ks_p) and null_ks_p >= NULL_KS_MIN_P):
        reasons.append(f"g2 null KS p {null_ks_p:.3f} < {NULL_KS_MIN_P}")
    for sk, th in theta_mmd.items():
        if not np.isfinite(th) or th <= 0:
            reasons.append(f"g3 degenerate theta_mmd[s={sk}]={th:.3g}")
        elif np.isfinite(ref_level) and ref_level > 0 and \
                th > THETA_SEPARATION_MAX_FRAC * ref_level:
            reasons.append(f"g3 theta_mmd[s={sk}]={th:.3g} not separated from "
                           f"cross-clip p95 {ref_level:.3g}")
    return (not reasons), guards, "; ".join(reasons)


# --------------------------------------------------------------------------------------
# Calibration + the two verdict paths
# --------------------------------------------------------------------------------------
def calibrate_from_internal_null(null_cells: Sequence[GateACell]) -> GateAThresholds:
    """Per-s 95th-percentile thresholds + raw null values from the cfg=1.0 cells."""
    if not null_cells:
        raise ValueError("no cfg=1.0 cells; run the internal-null arm first")
    theta_mmd, theta_tv, null_mmd, null_tv = {}, {}, {}, {}
    for s in sorted({c.s for c in null_cells}):
        sk = _skey(s)
        mm = [c.mmd2 for c in null_cells if c.s == s]
        tv = [c.tv for c in null_cells if c.s == s and np.isfinite(c.tv)]
        theta_mmd[sk] = float(np.percentile(mm, 95)) if mm else float("nan")
        theta_tv[sk] = float(np.percentile(tv, 95)) if tv else float("nan")
        null_mmd[sk], null_tv[sk] = list(map(float, mm)), list(map(float, tv))
    return GateAThresholds(theta_mmd=theta_mmd, theta_tv=theta_tv,
                           null_mmd=null_mmd, null_tv=null_tv, n_ref_cells=len(null_cells))


def _underpowered(cfg: float, schedule: str, guards: dict, reason: str) -> GateAResult:
    """Registry-conformant failure token (revised manual section 14 lists only
    CFG_KERNEL_OK|FAIL); the underpowered/instrument nature is carried in detail
    and extra for routing — it is a FAIL that must not be read as a kernel
    verdict."""
    sched = f", schedule={schedule}" if schedule != "constant" else ""
    return GateAResult(token=f"CFG_KERNEL_FAIL(cfg={cfg:g}{sched})", passed=False,
                       cfg=cfg, schedule=schedule, guards=guards,
                       detail="UNDERPOWERED/INVALID-DESIGN (not a kernel verdict): "
                              + reason + " — fix instrument, never pass silently",
                       extra={"underpowered": True})


def _check_design(cells: Sequence[GateACell], expected_s: tuple[float, ...],
                  expected_cells_per_s: int) -> Optional[str]:
    """The registered test grid must be fully present: every expected s-point
    with exactly the registered number of clip cells (Codex pass-A finding:
    the 2-of-16 Binomial rule is only calibrated at the registered denominator)."""
    by_s = {}
    for c in cells:
        by_s.setdefault(c.s, []).append(c)
    missing = [s for s in expected_s if s not in by_s]
    if missing:
        return f"missing test s-points {missing} (expected {list(expected_s)})"
    bad_n = {s: len(v) for s, v in by_s.items() if len(v) != expected_cells_per_s}
    if bad_n:
        return (f"cell counts {bad_n} != registered {expected_cells_per_s} per s "
                "(the low-p cap is Binomial-calibrated at that denominator)")
    extra_s = [s for s in by_s if s not in expected_s]
    if extra_s:
        return f"unregistered s-points {extra_s} present"
    return None


def evaluate_internal_null(cells: Sequence[GateACell], thresholds: GateAThresholds,
                           power_reject_frac: float, cross_clip_mmd_median: float,
                           null_ks_p: float, cfg: float = 1.0,
                           schedule: str = "constant",
                           cross_clip_mmd_p95: float = float("nan"),
                           low_p_max: int = LOW_P_MAX_CELLS,
                           expected_s: tuple[float, ...] = (0.05, 0.90),
                           expected_cells_per_s: int = 16) -> GateAResult:
    """cfg=1.0 HARD rule: registered design complete AND guards healthy AND per
    s-point at most low_p_max cells with p < ALPHA_SIG. High-p pile-up is
    expected (paired forks) and benign."""
    design_err = _check_design(cells, expected_s, expected_cells_per_s)
    if design_err:
        return _underpowered(cfg, schedule, {}, design_err)
    g_ok, guards, g_reason = check_guards(power_reject_frac, cross_clip_mmd_median,
                                          null_ks_p, thresholds.theta_mmd,
                                          cross_clip_mmd_p95=cross_clip_mmd_p95)
    if not g_ok:
        return _underpowered(cfg, schedule, guards, g_reason)
    per_s, failed = {}, []
    for s in sorted({c.s for c in cells}):
        sk = _skey(s)
        sub = [c for c in cells if c.s == s]
        n_low = sum(c.p_value < ALPHA_SIG for c in sub)
        ok = n_low <= low_p_max
        per_s[sk] = {"n_cells": len(sub), "n_low_p": n_low, "cap": low_p_max, "ok": ok,
                     "median_mmd2": float(np.median([c.mmd2 for c in sub])),
                     "median_tv": float(np.nanmedian([c.tv for c in sub]))}
        if not ok:
            failed.append(f"s={sk}: {n_low}/{len(sub)} low-p cells > {low_p_max}")
    passed = not failed
    token = (f"CFG_KERNEL_OK(cfg={cfg:g}" + (f", schedule={schedule}" if schedule != "constant" else "") + ")"
             if passed else
             f"CFG_KERNEL_FAIL(cfg={cfg:g}" + (f", schedule={schedule}" if schedule != "constant" else "") + ")")
    detail = ("internal null consistent with an exact kernel" if passed else
              "; ".join(failed) + " — STOP-level instrument review (revised manual 2 routing)")
    return GateAResult(token=token, passed=passed, cfg=cfg, schedule=schedule,
                       guards=guards, per_s=per_s, detail=detail)


def evaluate_calibrated(cells: Sequence[GateACell], thresholds: GateAThresholds,
                        cfg: float, schedule: str = "constant",
                        exceed_max: int = EXCEEDANCE_MAX_CELLS,
                        expected_s: tuple[float, ...] = (0.05, 0.90),
                        expected_cells_per_s: int = 16) -> GateAResult:
    """cfg>1 adjudication against the cfg=1.0-calibrated per-s thresholds.

    Per s-point, per statistic (MMD, TV): (i) one-sided Mann-Whitney vs the null
    cells (p >= MW_MIN_P); (ii) exceedance over the 95th-pct threshold
    <= exceed_max; (iii) no gross localized cell. ALL must hold at ALL s-points.
    NOT a Stage-M pass requirement — reported and routed per revised manual 1.2.
    """
    from scipy.stats import mannwhitneyu

    design_err = _check_design(cells, expected_s, expected_cells_per_s)
    if design_err:
        return _underpowered(cfg, schedule, {}, design_err)
    per_s, failed = {}, []
    for s in sorted({c.s for c in cells}):
        sk = _skey(s)
        sub = [c for c in cells if c.s == s]
        checks: dict[str, bool] = {}
        stats: dict[str, float] = {}
        for stat_name, vals, null_vals, theta, gross_cap in (
                ("mmd", [c.mmd2 for c in sub], thresholds.null_mmd.get(sk, []),
                 thresholds.theta_mmd.get(sk, float("nan")), float("inf")),
                ("tv", [c.tv for c in sub if np.isfinite(c.tv)],
                 thresholds.null_tv.get(sk, []),
                 thresholds.theta_tv.get(sk, float("nan")), TV_GROSS_CAP)):
            if not vals or len(null_vals) < 3 or not np.isfinite(theta):
                checks[f"{stat_name}:data"] = False
                continue
            mw_p = float(mannwhitneyu(vals, null_vals, alternative="greater").pvalue)
            n_exceed = sum(v > theta for v in vals)
            gross = min(GROSS_FACTOR * theta, gross_cap)
            n_gross = sum(v > gross for v in vals)
            stats[f"{stat_name}_mw_p"] = mw_p
            stats[f"{stat_name}_n_exceed"] = n_exceed
            stats[f"{stat_name}_worst"] = float(max(vals))
            checks[f"{stat_name}:mw"] = mw_p >= MW_MIN_P
            checks[f"{stat_name}:exceedance"] = n_exceed <= exceed_max
            checks[f"{stat_name}:gross"] = n_gross == 0
        ok = all(checks.values())
        per_s[sk] = {"checks": checks, "stats": stats, "n_cells": len(sub), "ok": ok}
        if not ok:
            failed.append(f"s={sk}: " + ",".join(k for k, v in checks.items() if not v))
    passed = bool(per_s) and not failed
    sched_part = f", schedule={schedule}" if schedule != "constant" else ""
    token = (f"CFG_KERNEL_OK(cfg={cfg:g}{sched_part})" if passed
             else f"CFG_KERNEL_FAIL(cfg={cfg:g}{sched_part})")
    detail = ("exchangeable with the cfg=1.0 null at all test points" if passed else
              "; ".join(failed) + " — route per revised manual 1.2 (headline stays cfg=1.0; "
              "re-entry requires a passing (cfg, schedule) tuple)")
    return GateAResult(token=token, passed=passed, cfg=cfg, schedule=schedule,
                       per_s=per_s, detail=detail)
