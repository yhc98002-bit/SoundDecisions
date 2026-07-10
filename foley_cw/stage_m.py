"""Stage-M micro-map evaluation — revised criteria (revised manual section 2).

Stage-M outputs are diagnostics, never evidence. Pass criteria are evaluated at
the HEADLINE cfg=1.0; the cfg=4.5 arm serves Gate-A adjudication and schedule
pilots and is NOT a pass requirement (the central change from the first run).

The five criteria (revised manual 2; frozen interpretations #3/#5/#6 in
experiment/preregistered/stage_m_rerun_interpretations.md):

  1. Endpoints, granularity-aware (June-13 manual 2.1; amendment #12).
     Label level (confident subset, exact-match gating):
       LATE  — A_fork(0.90) >= 0.90.
       EARLY — WASHOUT DIRECTION on the SIGNED seed floor g0 = A_fork(0.05) -
       A_independent: (i) g0 >= G0_MIN (a floor, not anti-correlation),
       (ii) the headline Gate-A is exchangeable at s=0.05, (iii) g0 <= G0_MAX
       (present but not dominant). The old |g0| <= 0.10 band is RETIRED — the
       seed floor is a first-class positive quantity (manual 1.1/4), not a
       distance to drive to zero. Means over SCORABLE clips (n_conf >= 2);
       >= MIN_SCORABLE_CLIPS of 16 must be scorable per endpoint. Abstain rates
       reported at every s.
     Embedding level (per-seed fork ensembles, PANNs-2048 cosine):
       E1 seed floor: paired per-clip bootstrap CI of
          [A_fork_emb(0.05) - A_ind_emb] has lower bound > 0, AND the CI
          half-width of mean A_fork_emb(0.05) is <= 0.05 (stability);
       E2 growth: paired CI of [A_fork_emb(0.90) - A_fork_emb(0.05)] lower > 0.
       The early-endpoint identity is NOT required — the seed floor is real.
  2. Monotonicity: commit(s) non-decreasing within CI tolerance (label level,
     confident subset, headline cfg) — violation iff the 95% bootstrap CI upper
     bound of an adjacent mean difference lies below -MONOTONE_TOL.
  3. Kernel: Gate A passes at cfg=1.0 (HARD; internal-null rule). At cfg=4.5
     Gate A is adjudicated and reported; a 4.5 failure routes to the manual-1.2
     fallback, not a halt.
  4. Measurer determinism = 1.0 on identical wavs (extended alphabet — abstain
     counts as a label value here).
  5. Informativeness: < 12/16 clips video-pinned (A_independent > 0.9) on class
     at the headline cfg, else widen/re-stratify; abstain rate <= 30% of
     fork-final class labels at s = 0.90 (headline cfg), else revisit delta or
     trigger the BEATs contingency.

Tokens: MICROMAP_PASS / MICROMAP_FAIL(reason) plus the Gate-A tokens.
Routing (revised manual 2): late-endpoint failure on the confident subset ->
genuinely suspect kernel/terminal-time numerics; early-endpoint label failure ->
normalization or A_independent estimation; Gate-A failure at cfg=1.0 ->
STOP-level instrument review.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .commitment import commit_gain
from .gate_a import GateAResult
from .types import Thresholds

S_EARLY = 0.05
S_LATE = 0.90
ENDPOINT_LATE_MIN = 0.90
# Early-endpoint WASHOUT-DIRECTION rule (June-13 manual section 2.1; amendment #12).
# The signed early gap g0 = A_fork(0.05) - A_independent is a SEED FLOOR, a
# first-class positive quantity (section 1.1/4), not a distance to drive to zero.
# It must be (i) non-negative (a floor, not anti-correlation; G0_MIN is numerical
# slack on the 16-clip mean), (ii) Gate-A exchangeable at s=0.05 (already the
# operative kernel test), and (iii) bounded by G0_MAX (a floor that is present but
# not dominant — above it the model is near-deterministic from noise, leaving no
# trajectory phase to map). Replaces the old |g0| <= 0.10 band.
G0_MIN = -0.02
G0_MAX = 0.25
MONOTONE_TOL = 0.02
INFORMATIVENESS_FRAC = 12 / 16
ABSTAIN_LATE_MAX = 0.30
DETERMINISM_MIN = 1.0 - 1e-9
EXPECTED_N_CLIPS = 16
MIN_SCORABLE_CLIPS = 12
EMB_FLOOR_HALFWIDTH_MAX = 0.05
HEADLINE_CFG = 1.0
DEPLOYED_CFG = 4.5


class IncompleteRecordsError(ValueError):
    """Stage-M records are missing cells — a partial run must never be scored."""


@dataclass
class StageMRecords:
    """Persisted Stage-M records (runner writes; evaluator judges).

    Label-level values are CONFIDENT-SUBSET exact-match agreements (NaN when a
    cell has < 2 confident labels); abstain rates and confident counts are
    stored alongside. Embedding-level values are mean pairwise cosines of the
    per-seed fork ensembles / independent finals (PANNs-2048).
    """

    s_grid: tuple[float, ...]
    cfgs: tuple[float, ...]
    axis_ids: tuple[str, ...]
    clips: tuple[str, ...]
    #: (clip, cfg, axis, s) -> confident-subset fork agreement (NaN if unscorable)
    a_fork: dict[tuple[str, float, str, float], float] = field(default_factory=dict)
    #: (clip, cfg, axis) -> confident-subset independent agreement
    a_independent: dict[tuple[str, float, str], float] = field(default_factory=dict)
    #: (clip, cfg, axis, s) -> abstain fraction among K fork labels
    abstain_fork: dict[tuple[str, float, str, float], float] = field(default_factory=dict)
    #: (clip, cfg, axis) -> abstain fraction among N independent labels
    abstain_ind: dict[tuple[str, float, str], float] = field(default_factory=dict)
    #: (clip, cfg, axis, s) -> n_confident among K fork labels
    n_conf_fork: dict[tuple[str, float, str, float], int] = field(default_factory=dict)
    #: (clip, cfg, axis) -> n_confident among N independents
    n_conf_ind: dict[tuple[str, float, str], int] = field(default_factory=dict)
    #: (clip, cfg, s) -> mean pairwise cosine of the per-seed fork embeddings
    a_fork_emb: dict[tuple[str, float, float], float] = field(default_factory=dict)
    #: (clip, cfg) -> mean pairwise cosine of the independent final embeddings
    a_ind_emb: dict[tuple[str, float], float] = field(default_factory=dict)

    # -- persistence -----------------------------------------------------------
    def save(self, path: Path) -> None:
        def enc4(d):
            return {f"{c}|{g:g}|{a}|{s:g}": v for (c, g, a, s), v in d.items()}

        def enc3(d):
            return {f"{c}|{g:g}|{a}": v for (c, g, a), v in d.items()}

        payload = {
            "s_grid": list(self.s_grid), "cfgs": list(self.cfgs),
            "axis_ids": list(self.axis_ids), "clips": list(self.clips),
            "a_fork": enc4(self.a_fork), "a_independent": enc3(self.a_independent),
            "abstain_fork": enc4(self.abstain_fork), "abstain_ind": enc3(self.abstain_ind),
            "n_conf_fork": enc4(self.n_conf_fork), "n_conf_ind": enc3(self.n_conf_ind),
            "a_fork_emb": {f"{c}|{g:g}|{s:g}": v for (c, g, s), v in self.a_fork_emb.items()},
            "a_ind_emb": {f"{c}|{g:g}": v for (c, g), v in self.a_ind_emb.items()},
        }
        tmp = Path(str(path) + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> "StageMRecords":
        d = json.loads(Path(path).read_text())
        rec = cls(s_grid=tuple(d["s_grid"]), cfgs=tuple(d["cfgs"]),
                  axis_ids=tuple(d["axis_ids"]), clips=tuple(d["clips"]))
        for name in ("a_fork", "abstain_fork", "n_conf_fork"):
            for k, v in d[name].items():
                c, g, a, s = k.split("|")
                getattr(rec, name)[(c, float(g), a, float(s))] = (
                    int(v) if name == "n_conf_fork" else float(v))
        for name in ("a_independent", "abstain_ind", "n_conf_ind"):
            for k, v in d[name].items():
                c, g, a = k.split("|")
                getattr(rec, name)[(c, float(g), a)] = (
                    int(v) if name == "n_conf_ind" else float(v))
        for k, v in d["a_fork_emb"].items():
            c, g, s = k.split("|")
            rec.a_fork_emb[(c, float(g), float(s))] = float(v)
        for k, v in d["a_ind_emb"].items():
            c, g = k.split("|")
            rec.a_ind_emb[(c, float(g))] = float(v)
        return rec

    # -- views -----------------------------------------------------------------
    def commit_curve_per_clip(self, cfg: float, axis_id: str) -> dict[str, list[float]]:
        out: dict[str, list[float]] = {}
        for c in self.clips:
            a_ind = self.a_independent.get((c, cfg, axis_id), float("nan"))
            if not np.isfinite(a_ind):
                continue
            curve = []
            for s in self.s_grid:
                af = self.a_fork.get((c, cfg, axis_id, s), float("nan"))
                curve.append(commit_gain(af, a_ind) if np.isfinite(af) else float("nan"))
            out[c] = curve
        return out


@dataclass
class CriterionResult:
    name: str
    passed: bool
    detail: str
    values: dict = field(default_factory=dict)


@dataclass
class StageMReport:
    criteria: list[CriterionResult]
    tokens: list[str]
    informativeness_warning: bool
    failure_routing: str = ""

    def to_markdown(self) -> str:
        lines = ["# Stage-M Micro-Map Report (revised manual section 2)", "",
                 "Stage-M outputs are engineering diagnostics, never scientific evidence.",
                 f"Pass criteria evaluated at the headline cfg={HEADLINE_CFG:g}; the "
                 f"cfg={DEPLOYED_CFG:g} arm is adjudicated, not gating.", "",
                 "| # | Criterion | Pass | Detail |", "|---|---|---|---|"]
        for i, c in enumerate(self.criteria, 1):
            lines.append(f"| {i} | {c.name} | {'PASS' if c.passed else 'FAIL'} | {c.detail} |")
        lines += ["", f"**Tokens:** {', '.join(self.tokens)}", ""]
        if self.informativeness_warning:
            lines.append("**Informativeness warning fired** — see criterion 5 detail.")
        if self.failure_routing:
            lines += ["", f"**Failure routing:** {self.failure_routing}"]
        return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def _nanmean(vals) -> float:
    arr = np.asarray([v for v in vals if np.isfinite(v)], dtype=float)
    return float(arr.mean()) if arr.size else float("nan")


def _paired_boot_ci(diffs: np.ndarray, n_boot: int = 1000, seed: int = 0,
                    ci: float = 0.95) -> tuple[float, float, float]:
    """(point, lo, hi) percentile bootstrap over clips of a paired difference."""
    diffs = np.asarray([d for d in diffs if np.isfinite(d)], dtype=float)
    if diffs.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, diffs.size, size=diffs.size)
        boots[b] = diffs[idx].mean()
    a = (1 - ci) / 2
    return float(diffs.mean()), float(np.percentile(boots, 100 * a)), \
        float(np.percentile(boots, 100 * (1 - a)))


# --------------------------------------------------------------------------------------
# completeness
# --------------------------------------------------------------------------------------
def validate_completeness(rec: StageMRecords, determinism_scores: dict[str, float],
                          expected_clips: int = EXPECTED_N_CLIPS,
                          expected_cfgs: tuple[float, ...] = (HEADLINE_CFG, DEPLOYED_CFG),
                          expected_s_grid: tuple[float, ...] = (0.05, 0.30, 0.60, 0.90),
                          expected_axes: tuple[str, ...] = ("presence", "class")) -> None:
    """Refuse to score a partial run. NaN agreements are legitimate (unscorable
    confident subsets) — completeness requires the CELL to exist (key present),
    not the value to be finite. Embedding cells must exist and be finite."""
    problems: list[str] = []
    if len(rec.clips) != expected_clips:
        problems.append(f"{len(rec.clips)} clips, expected {expected_clips}")
    if tuple(sorted(rec.cfgs)) != tuple(sorted(expected_cfgs)):
        problems.append(f"cfgs {rec.cfgs}, expected {expected_cfgs}")
    if tuple(sorted(rec.s_grid)) != tuple(sorted(expected_s_grid)):
        problems.append(f"s_grid {rec.s_grid}, expected {expected_s_grid}")
    if tuple(sorted(rec.axis_ids)) != tuple(sorted(expected_axes)):
        problems.append(f"axes {rec.axis_ids}, expected {expected_axes}")
    miss_ind = [(c, g, a) for c in rec.clips for g in rec.cfgs for a in rec.axis_ids
                if (c, g, a) not in rec.a_independent]
    miss_fork = [(c, g, a, s) for c in rec.clips for g in rec.cfgs for a in rec.axis_ids
                 for s in rec.s_grid if (c, g, a, s) not in rec.a_fork]
    miss_emb = [(c, g, s) for c in rec.clips for g in rec.cfgs for s in rec.s_grid
                if not np.isfinite(rec.a_fork_emb.get((c, g, s), float("nan")))]
    miss_iemb = [(c, g) for c in rec.clips for g in rec.cfgs
                 if not np.isfinite(rec.a_ind_emb.get((c, g), float("nan")))]
    if miss_ind:
        problems.append(f"{len(miss_ind)} missing a_independent cells (first: {miss_ind[0]})")
    if miss_fork:
        problems.append(f"{len(miss_fork)} missing a_fork cells (first: {miss_fork[0]})")
    if miss_emb:
        problems.append(f"{len(miss_emb)} missing/non-finite a_fork_emb cells "
                        f"(first: {miss_emb[0]})")
    if miss_iemb:
        problems.append(f"{len(miss_iemb)} missing a_ind_emb cells (first: {miss_iemb[0]})")
    miss_det = [a for a in rec.axis_ids
                if not np.isfinite(determinism_scores.get(a, float("nan")))]
    if miss_det:
        problems.append(f"missing determinism scores for axes {miss_det}")
    if problems:
        raise IncompleteRecordsError("Stage-M records incomplete — refusing to emit tokens: "
                                     + "; ".join(problems))


# --------------------------------------------------------------------------------------
# criteria
# --------------------------------------------------------------------------------------
def _criterion_endpoints(rec: StageMRecords, gate_a_headline: GateAResult,
                         seed: int) -> CriterionResult:
    cfg = HEADLINE_CFG
    details, values, ok = [], {}, True
    # Gate-A exchangeability at s=0.05 (criterion-1 sub-condition (ii), amendment
    # #12): the early washout test is meaningful only where the kernel is
    # marginally valid. Read the headline Gate-A per-s verdict (key is f"{s:g}").
    s05_ok = bool(gate_a_headline.per_s.get(f"{S_EARLY:g}", {}).get("ok", False))

    # --- label level, confident subset, scorability-gated ----------------------
    # Scorability is judged on n_conf >= 2 EXPLICITLY (frozen interpretation #3),
    # for the fork cells at both endpoints AND the A_independent baseline (the
    # early-gap comparison needs a scorable baseline too — Codex pass-A finding).
    for ax in rec.axis_ids:
        for s, rule in ((S_LATE, "late"), (S_EARLY, "early")):
            n_scorable = sum(1 for c in rec.clips
                             if rec.n_conf_fork.get((c, cfg, ax, s), 0) >= 2)
            values[f"{ax}:n_scorable_{rule}"] = n_scorable
            if n_scorable < MIN_SCORABLE_CLIPS:
                ok = False
                details.append(f"{ax}: only {n_scorable}/{len(rec.clips)} scorable fork "
                               f"cells at s={s:g} (< {MIN_SCORABLE_CLIPS}) — "
                               "delta/BEATs routing")
        n_scorable_ind = sum(1 for c in rec.clips
                             if rec.n_conf_ind.get((c, cfg, ax), 0) >= 2)
        values[f"{ax}:n_scorable_ind"] = n_scorable_ind
        if n_scorable_ind < MIN_SCORABLE_CLIPS:
            ok = False
            details.append(f"{ax}: only {n_scorable_ind}/{len(rec.clips)} scorable "
                           f"A_independent baselines (< {MIN_SCORABLE_CLIPS}) — "
                           "delta/BEATs routing")
        late = _nanmean([rec.a_fork.get((c, cfg, ax, S_LATE), float("nan"))
                         for c in rec.clips])
        early = _nanmean([rec.a_fork.get((c, cfg, ax, S_EARLY), float("nan"))
                          for c in rec.clips])
        a_ind = _nanmean([rec.a_independent.get((c, cfg, ax), float("nan"))
                          for c in rec.clips])
        g0 = early - a_ind  # SIGNED early gap = the seed floor (amendment #12)
        values[ax] = {"a_fork_late": late, "a_fork_early": early,
                      "a_independent": a_ind, "g0_seed_floor": g0,
                      "s05_gate_a_ok": s05_ok,
                      "abstain_late": _nanmean([rec.abstain_fork.get((c, cfg, ax, S_LATE),
                                                                     float("nan"))
                                                for c in rec.clips])}
        if not (np.isfinite(late) and late >= ENDPOINT_LATE_MIN):
            ok = False
            details.append(f"{ax}: A_fork({S_LATE})={late:.3f} < {ENDPOINT_LATE_MIN} "
                           "(confident subset)")
        # Early WASHOUT-DIRECTION rule (amendment #12), three sub-conditions:
        if not np.isfinite(g0):
            ok = False
            details.append(f"{ax}: early gap g0 not finite (A_independent/fork unscorable)")
        else:
            if g0 < G0_MIN:
                ok = False
                details.append(f"{ax}: g0={g0:.3f} < {G0_MIN} — anti-correlation, not a "
                               "floor (suspect normalization / A_independent estimation)")
            if g0 > G0_MAX:
                ok = False
                details.append(f"{ax}: g0={g0:.3f} > {G0_MAX} — near-deterministic from "
                               "noise, no trajectory phase to map")

    # sub-condition (ii) is axis-independent (one headline-Gate-A verdict), checked once
    if not s05_ok:
        ok = False
        details.append(f"Gate-A NOT exchangeable at s={S_EARLY} (kernel marginally "
                       "invalid there) — STOP-level instrument review")

    # --- embedding level: E1 seed floor + E2 growth (paired bootstrap) ---------
    floor_diffs = np.array([rec.a_fork_emb.get((c, cfg, S_EARLY), np.nan)
                            - rec.a_ind_emb.get((c, cfg), np.nan) for c in rec.clips])
    growth_diffs = np.array([rec.a_fork_emb.get((c, cfg, S_LATE), np.nan)
                             - rec.a_fork_emb.get((c, cfg, S_EARLY), np.nan)
                             for c in rec.clips])
    early_vals = np.array([rec.a_fork_emb.get((c, cfg, S_EARLY), np.nan)
                           for c in rec.clips])
    f_pt, f_lo, f_hi = _paired_boot_ci(floor_diffs, seed=seed)
    g_pt, g_lo, g_hi = _paired_boot_ci(growth_diffs, seed=seed + 1)
    e_pt, e_lo, e_hi = _paired_boot_ci(early_vals, seed=seed + 2)
    halfwidth = (e_hi - e_lo) / 2 if np.isfinite(e_hi) else float("nan")
    values["emb"] = {"floor_diff": [f_pt, f_lo, f_hi], "growth_diff": [g_pt, g_lo, g_hi],
                     "early_mean_ci": [e_pt, e_lo, e_hi], "early_halfwidth": halfwidth}
    if not (np.isfinite(f_lo) and f_lo > 0):
        ok = False
        details.append(f"E1: seed-floor paired CI [{f_lo:.3f}, {f_hi:.3f}] does not exclude 0")
    if not (np.isfinite(halfwidth) and halfwidth <= EMB_FLOOR_HALFWIDTH_MAX):
        ok = False
        details.append(f"E1: A_fork_emb({S_EARLY}) CI half-width {halfwidth:.3f} > "
                       f"{EMB_FLOOR_HALFWIDTH_MAX} (unstable floor)")
    if not (np.isfinite(g_lo) and g_lo > 0):
        ok = False
        details.append(f"E2: emb growth paired CI [{g_lo:.3f}, {g_hi:.3f}] does not exclude 0")

    return CriterionResult("endpoints", ok, "; ".join(details) or
                           "label endpoints (confident subset) + embedding seed floor/growth OK",
                           values)


def _criterion_monotonicity(rec: StageMRecords, n_boot: int = 1000, seed: int = 0) -> CriterionResult:
    """Violation iff the 95% CI upper bound of an adjacent mean commit difference
    lies below -MONOTONE_TOL (pre-registered tolerance band); headline cfg,
    confident subset, NaN-aware."""
    cfg = HEADLINE_CFG
    rng = np.random.default_rng(seed)
    violations, values = [], {}
    for ax in rec.axis_ids:
        per_clip = rec.commit_curve_per_clip(cfg, ax)
        curves = np.array(list(per_clip.values()), dtype=float)
        if curves.size == 0:
            violations.append(f"{ax}: no scorable curves")
            continue
        values[ax] = {"mean_commit_curve": np.nanmean(curves, axis=0).tolist(),
                      "n_clips": curves.shape[0]}
        n = curves.shape[0]
        for i in range(len(rec.s_grid) - 1):
            diffs = curves[:, i + 1] - curves[:, i]
            if np.all(~np.isfinite(diffs)):
                continue
            point = float(np.nanmean(diffs))
            if point >= -MONOTONE_TOL:
                continue
            boots = np.empty(n_boot)
            for b in range(n_boot):
                idx = rng.integers(0, n, size=n)
                boots[b] = np.nanmean(diffs[idx])
            hi = float(np.nanpercentile(boots, 97.5))
            if hi < -MONOTONE_TOL:
                violations.append(f"{ax}: commit({rec.s_grid[i + 1]:g}) < "
                                  f"commit({rec.s_grid[i]:g}) by {-point:.3f} (CI excl. tol)")
    return CriterionResult("monotonicity", not violations,
                           "; ".join(violations) or
                           "commit(s) non-decreasing within CI tolerance (headline cfg)",
                           values)


def _criterion_kernel(gate_a_headline: GateAResult,
                      gate_a_deployed: Optional[GateAResult]) -> CriterionResult:
    ok = gate_a_headline.passed
    detail = f"Gate-A internal null @ cfg={HEADLINE_CFG:g}: {gate_a_headline.token}"
    if not ok:
        detail += f" — {gate_a_headline.detail}"
    if gate_a_deployed is not None:
        detail += (f" | adjudicated (non-gating) @ cfg={DEPLOYED_CFG:g}: "
                   f"{gate_a_deployed.token}")
    return CriterionResult("kernel_headline", ok, detail,
                           {"headline_token": gate_a_headline.token,
                            "deployed_token": gate_a_deployed.token if gate_a_deployed else None})


def _criterion_determinism(determinism_scores: dict[str, float]) -> CriterionResult:
    bad = {a: v for a, v in determinism_scores.items()
           if not (np.isfinite(v) and v >= DETERMINISM_MIN)}
    return CriterionResult("measurer_determinism", not bad,
                           "; ".join(f"{a}: {v:.4f} < 1.0" for a, v in bad.items()) or
                           "determinism == 1.0 on identical wavs (extended alphabet)",
                           dict(determinism_scores))


def _criterion_informativeness(rec: StageMRecords, class_axis_id: str) -> CriterionResult:
    cfg = HEADLINE_CFG
    pinned = [c for c in rec.clips
              if rec.a_independent.get((c, cfg, class_axis_id), float("nan")) > 0.9]
    frac_pinned = len(pinned) / max(len(rec.clips), 1)
    pinned_fired = frac_pinned >= INFORMATIVENESS_FRAC
    abstain_late = _nanmean([rec.abstain_fork.get((c, cfg, class_axis_id, S_LATE), float("nan"))
                             for c in rec.clips])
    abstain_fired = np.isfinite(abstain_late) and abstain_late > ABSTAIN_LATE_MAX
    detail = (f"{len(pinned)}/{len(rec.clips)} video-pinned on {class_axis_id!r} "
              f"@ cfg={cfg:g}; abstain@s={S_LATE} = {abstain_late:.2f} "
              f"(cap {ABSTAIN_LATE_MAX})")
    if pinned_fired:
        detail += "; POOL MUST BE WIDENED/RE-STRATIFIED before Stage 0"
    if abstain_fired:
        detail += "; ABSTAIN CAP EXCEEDED — revisit delta or trigger BEATs contingency"
    # Both halves are PASS CRITERIA per revised manual 2 (criterion 5): a fired
    # pinned check fails Stage M with the widen/re-stratify routing (Codex
    # pass-B finding); the abstain cap fails with the delta/BEATs routing.
    return CriterionResult("informativeness", not (abstain_fired or pinned_fired), detail,
                           {"n_pinned": len(pinned), "frac_pinned": frac_pinned,
                            "pinned_fired": pinned_fired,
                            "abstain_late": abstain_late, "abstain_fired": abstain_fired})


def evaluate_stage_m(rec: StageMRecords, gate_a_headline: GateAResult,
                     gate_a_deployed: Optional[GateAResult],
                     determinism_scores: dict[str, float], thresholds: Thresholds,
                     class_axis_id: str = "class", seed: int = 0,
                     expected_clips: int = EXPECTED_N_CLIPS) -> StageMReport:
    """Apply the revised five criteria (headline cfg gating) and emit tokens."""
    if not thresholds.frozen:
        raise ValueError("thresholds must be frozen before Stage-M evaluation")
    validate_completeness(rec, determinism_scores, expected_clips=expected_clips)

    c1 = _criterion_endpoints(rec, gate_a_headline, seed)
    c2 = _criterion_monotonicity(rec, seed=seed)
    c3 = _criterion_kernel(gate_a_headline, gate_a_deployed)
    c4 = _criterion_determinism(determinism_scores)
    c5 = _criterion_informativeness(rec, class_axis_id)
    criteria = [c1, c2, c3, c4, c5]

    hard_pass = all(c.passed for c in criteria)
    warning = bool(c5.values.get("pinned_fired", False))

    tokens, routing = [], ""
    if hard_pass:
        tokens.append("MICROMAP_PASS")
    else:
        failed = [c.name for c in criteria if not c.passed]
        tokens.append(f"MICROMAP_FAIL({','.join(failed)})")
        routes = []
        if not c1.passed:
            if "A_fork(0.9" in c1.detail:
                routes.append("late-endpoint failure on the CONFIDENT subset -> genuinely "
                              "suspect kernel/terminal-time numerics (manual 2)")
            if "near-deterministic from" in c1.detail:
                routes.append("early seed-floor g0 > G0_MAX -> model is near-deterministic "
                              "from noise; no trajectory phase to map (route to NEGATIVE/F-1)")
            if "anti-correlation" in c1.detail or "scorable" in c1.detail:
                routes.append("early-endpoint/scorability failure -> normalization, "
                              "A_independent estimation, or delta/BEATs routing")
            if "marginally invalid" in c1.detail:
                routes.append("Gate-A not exchangeable at s=0.05 -> STOP-level instrument review")
            if "E1" in c1.detail or "E2" in c1.detail:
                routes.append("embedding seed-floor/growth failure -> inspect per-clip "
                              "cohesion distributions before re-running")
        if not c3.passed:
            routes.append("Gate-A failure at the headline cfg -> STOP-level instrument review")
        if not c4.passed:
            routes.append("determinism failure -> fix the measurer first")
        if not c5.passed:
            if c5.values.get("abstain_fired"):
                routes.append("abstain cap exceeded -> revisit delta or BEATs contingency")
            if c5.values.get("pinned_fired"):
                routes.append("video-pinned >= 12/16 -> widen/re-stratify the Stage-0 pool")
        routing = " | ".join(routes)
    tokens.append(gate_a_headline.token)
    if gate_a_deployed is not None:
        tokens.append(gate_a_deployed.token)
    return StageMReport(criteria=criteria, tokens=tokens,
                        informativeness_warning=warning, failure_routing=routing)
