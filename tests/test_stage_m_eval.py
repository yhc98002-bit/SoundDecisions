"""Tests for foley_cw.stage_m — REVISED Stage-M criteria (revised plan section 2).

Key revision under test: pass criteria are evaluated at the HEADLINE cfg=1.0;
the cfg=4.5 arm is Gate-A adjudication + schedule pilots and is NOT a pass
requirement. Label-level endpoints are confident-subset, scorability-gated;
embedding-level endpoints test the seed floor (E1) and growth (E2) instead of
the early-endpoint identity.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pytest

from foley_cw.gate_a import GateAResult
from foley_cw.stage_m import (
    IncompleteRecordsError,
    StageMRecords,
    evaluate_stage_m,
)
from foley_cw.types import Thresholds

S_GRID = (0.05, 0.30, 0.60, 0.90)
CFGS = (1.0, 4.5)
AXES = ("presence", "class")
CLIPS = tuple(f"clip{i:02d}" for i in range(16))
HEADLINE, DEPLOYED = 1.0, 4.5

FROZEN = Thresholds(theta_commit=0.7, theta_read=0.7, theta_rel=0.95,
                    theta_robust=0.85, theta_cal=0.6, frozen=True,
                    frozen_from="test fixture")
UNFROZEN = Thresholds(theta_commit=0.7, theta_read=0.7, theta_rel=0.95,
                      theta_robust=0.85, theta_cal=0.6, frozen=False)


def gate_a(passed: bool, cfg: float, s05_ok: Optional[bool] = None) -> GateAResult:
    """Test stub. per_s carries the per-s-point 'ok' verdict the criterion-1
    sub-condition (ii) reads at s=0.05; defaults to the overall pass state."""
    tok = f"CFG_KERNEL_{'OK' if passed else 'FAIL'}(cfg={cfg:g})"
    s05 = passed if s05_ok is None else s05_ok
    per_s = {"0.05": {"ok": bool(s05)}, "0.9": {"ok": bool(passed)}}
    return GateAResult(token=tok, passed=passed, cfg=cfg, per_s=per_s,
                       detail="" if passed else "exchangeability rejected")


GATE_HEAD_OK = gate_a(True, HEADLINE)
GATE_HEAD_FAIL = gate_a(False, HEADLINE)
GATE_DEP_OK = gate_a(True, DEPLOYED)
GATE_DEP_FAIL = gate_a(False, DEPLOYED)

DET_OK = {"presence": 1.0, "class": 1.0}

# Confident-subset label curve: early ~= A_independent, late >= 0.90.
FORK_CURVE = {0.05: None, 0.30: 0.60, 0.60: 0.85, 0.90: 0.97}
# Embedding curve: real seed floor ~0.87 (vs independents ~0.76) rising to ~0.94.
EMB_CURVE = {0.05: 0.87, 0.30: 0.89, 0.60: 0.92, 0.90: 0.94}
IND_EMB = 0.76


def healthy_records(a_ind: float = 0.40, seed: int = 0) -> StageMRecords:
    """NEW StageMRecords schema: confident-subset a_fork/a_independent plus
    n_conf / abstain bookkeeping plus per-clip embedding cohesion cells."""
    rng = np.random.default_rng(seed)
    rec = StageMRecords(s_grid=S_GRID, cfgs=CFGS, axis_ids=AXES, clips=CLIPS)
    for c in CLIPS:
        for g in CFGS:
            rec.a_ind_emb[(c, g)] = IND_EMB + float(rng.uniform(-0.01, 0.01))
            for s in S_GRID:
                rec.a_fork_emb[(c, g, s)] = EMB_CURVE[s] + float(rng.uniform(-0.01, 0.01))
            for a in AXES:
                rec.a_independent[(c, g, a)] = a_ind + float(rng.uniform(-0.02, 0.02))
                rec.abstain_ind[(c, g, a)] = 0.0 if a == "presence" else 0.125
                rec.n_conf_ind[(c, g, a)] = 8 if a == "presence" else 7
                for s in S_GRID:
                    base = a_ind if FORK_CURVE[s] is None else FORK_CURVE[s]
                    rec.a_fork[(c, g, a, s)] = base + float(rng.uniform(-0.02, 0.02))
                    rec.abstain_fork[(c, g, a, s)] = 0.0 if a == "presence" else 0.125
                    rec.n_conf_fork[(c, g, a, s)] = 8 if a == "presence" else 7
    return rec


def crit(rep, name):
    return next(c for c in rep.criteria if c.name == name)


# ----------------------------------------------------------------- (1) all-pass
def test_all_pass():
    rep = evaluate_stage_m(healthy_records(), GATE_HEAD_OK, GATE_DEP_OK, DET_OK, FROZEN)
    assert "MICROMAP_PASS" in rep.tokens
    assert "CFG_KERNEL_OK(cfg=1)" in rep.tokens
    assert "CFG_KERNEL_OK(cfg=4.5)" in rep.tokens
    assert all(c.passed for c in rep.criteria)
    assert not rep.informativeness_warning
    assert rep.failure_routing == ""


# ------------------------------------------ (2) deployed cfg=4.5 is NOT gating
def test_deployed_gate_a_fail_still_micromap_pass():
    """The central revision: a Gate-A failure at cfg=4.5 routes to the section-1.2
    fallback and is reported, but Stage-M still passes on the headline kernel."""
    rep = evaluate_stage_m(healthy_records(), GATE_HEAD_OK, GATE_DEP_FAIL, DET_OK, FROZEN)
    assert "MICROMAP_PASS" in rep.tokens
    assert not any(t.startswith("MICROMAP_FAIL") for t in rep.tokens)
    assert "CFG_KERNEL_FAIL(cfg=4.5)" in rep.tokens  # adjudicated + reported
    kernel = crit(rep, "kernel_headline")
    assert kernel.passed
    assert "4.5" in kernel.detail  # deployed verdict surfaced, non-gating


def test_deployed_gate_a_optional():
    rep = evaluate_stage_m(healthy_records(), GATE_HEAD_OK, None, DET_OK, FROZEN)
    assert "MICROMAP_PASS" in rep.tokens
    assert "CFG_KERNEL_OK(cfg=1)" in rep.tokens
    assert not any("4.5" in t for t in rep.tokens)


# -------------------------------------------------- (3) headline Gate-A is HARD
def test_headline_gate_a_fail_is_stop_level():
    rep = evaluate_stage_m(healthy_records(), GATE_HEAD_FAIL, GATE_DEP_OK, DET_OK, FROZEN)
    assert any(t.startswith("MICROMAP_FAIL") and "kernel_headline" in t for t in rep.tokens)
    assert "STOP-level" in rep.failure_routing
    assert "CFG_KERNEL_FAIL(cfg=1)" in rep.tokens
    assert not crit(rep, "kernel_headline").passed


# ------------------------------------------------- (4) late-endpoint label fail
def test_late_endpoint_failure_routes_to_terminal_numerics():
    rec = healthy_records()
    for c in CLIPS:
        rec.a_fork[(c, HEADLINE, "presence", 0.90)] = 0.5  # confident A_fork(0.90) low
    rep = evaluate_stage_m(rec, GATE_HEAD_OK, GATE_DEP_OK, DET_OK, FROZEN)
    assert any(t.startswith("MICROMAP_FAIL") and "endpoints" in t for t in rep.tokens)
    assert not crit(rep, "endpoints").passed
    assert "terminal-time" in rep.failure_routing


def test_late_endpoint_failure_at_deployed_cfg_only_is_ignored():
    """Symmetric check on the revision: criteria read the headline arm only."""
    rec = healthy_records()
    for c in CLIPS:
        for a in AXES:
            rec.a_fork[(c, DEPLOYED, a, 0.90)] = 0.5
    rep = evaluate_stage_m(rec, GATE_HEAD_OK, GATE_DEP_OK, DET_OK, FROZEN)
    assert "MICROMAP_PASS" in rep.tokens


# -------------------------------- (5) early WASHOUT-DIRECTION rule (amendment #12)
def test_early_seed_floor_in_band_passes():
    """g0 = A_fork(0.05) - A_ind is a SEED FLOOR: a moderate positive value
    (here ~0.15, like Run-3) is the intended outcome, not a failure."""
    rec = healthy_records(a_ind=0.40)
    for c in CLIPS:
        for a in AXES:
            rec.a_fork[(c, HEADLINE, a, 0.05)] = 0.55  # g0 ~= 0.15, within (G0_MIN, G0_MAX]
    rep = evaluate_stage_m(rec, GATE_HEAD_OK, GATE_DEP_OK, DET_OK, FROZEN)
    assert "MICROMAP_PASS" in rep.tokens
    assert crit(rep, "endpoints").passed
    ep = crit(rep, "endpoints").values
    assert ep["class"]["g0_seed_floor"] == pytest.approx(0.15, abs=0.03)


def test_early_gap_too_large_fails_near_deterministic():
    """g0 > G0_MAX (0.25): the model is near-deterministic from noise, no
    trajectory phase to map -> fail with the near-deterministic routing."""
    rec = healthy_records(a_ind=0.40)
    for c in CLIPS:
        for a in AXES:
            rec.a_fork[(c, HEADLINE, a, 0.05)] = 0.8  # g0 ~= 0.40 > 0.25
    rep = evaluate_stage_m(rec, GATE_HEAD_OK, GATE_DEP_OK, DET_OK, FROZEN)
    assert any(t.startswith("MICROMAP_FAIL") and "endpoints" in t for t in rep.tokens)
    assert "near-deterministic from" in crit(rep, "endpoints").detail
    assert "no trajectory phase to map" in rep.failure_routing


def test_early_gap_negative_fails_anti_correlation():
    """g0 < G0_MIN: forks LESS self-consistent than independents (anti-correlation)
    -> suspect normalization / A_independent estimation."""
    rec = healthy_records(a_ind=0.80)
    for c in CLIPS:
        for a in AXES:
            rec.a_fork[(c, HEADLINE, a, 0.05)] = 0.40  # g0 ~= -0.40 < G0_MIN
    rep = evaluate_stage_m(rec, GATE_HEAD_OK, GATE_DEP_OK, DET_OK, FROZEN)
    assert any(t.startswith("MICROMAP_FAIL") and "endpoints" in t for t in rep.tokens)
    assert "anti-correlation" in crit(rep, "endpoints").detail
    assert "normalization" in rep.failure_routing


def test_early_gate_a_not_exchangeable_at_s05_fails():
    """Sub-condition (ii): if the headline Gate-A is not exchangeable at s=0.05
    the early washout test is meaningless -> STOP-level instrument review."""
    gate_head_s05_bad = gate_a(True, HEADLINE, s05_ok=False)  # overall ok, s=0.05 not
    rep = evaluate_stage_m(healthy_records(), gate_head_s05_bad, GATE_DEP_OK, DET_OK, FROZEN)
    assert any(t.startswith("MICROMAP_FAIL") and "endpoints" in t for t in rep.tokens)
    assert "marginally invalid" in crit(rep, "endpoints").detail
    assert "STOP-level" in rep.failure_routing


# -------------------------------------------------- (6) scorability gate (NaN)
def test_unscorable_endpoint_cells_fail_with_delta_beats_routing():
    """NaN a_fork = confident subset too small (n_conf < 2). More than 4 such
    clips at an endpoint leaves < MIN_SCORABLE_CLIPS scorable -> criterion 1
    FAILS with the delta/BEATs routing rather than silently averaging."""
    rec = healthy_records()
    for c in CLIPS[:5]:  # 11/16 scorable < 12
        rec.a_fork[(c, HEADLINE, "class", 0.90)] = float("nan")
        rec.n_conf_fork[(c, HEADLINE, "class", 0.90)] = 1
        rec.abstain_fork[(c, HEADLINE, "class", 0.90)] = 0.875
    rep = evaluate_stage_m(rec, GATE_HEAD_OK, GATE_DEP_OK, DET_OK, FROZEN)
    endpoints = crit(rep, "endpoints")
    assert not endpoints.passed
    assert "scorable" in endpoints.detail
    assert "delta/BEATs" in rep.failure_routing
    assert any(t.startswith("MICROMAP_FAIL") and "endpoints" in t for t in rep.tokens)


# ---------------------------------------------------------- (7) E1: seed floor
def test_e1_no_seed_floor_fails():
    rec = healthy_records()
    for c in CLIPS:
        for g in CFGS:
            rec.a_fork_emb[(c, g, 0.05)] = rec.a_ind_emb[(c, g)]  # floor diff == 0
    rep = evaluate_stage_m(rec, GATE_HEAD_OK, GATE_DEP_OK, DET_OK, FROZEN)
    endpoints = crit(rep, "endpoints")
    assert not endpoints.passed
    assert "E1" in endpoints.detail
    assert "E2" not in endpoints.detail  # growth (0.94 vs 0.76) still real
    assert any(t.startswith("MICROMAP_FAIL") and "endpoints" in t for t in rep.tokens)


# -------------------------------------------------------------- (8) E2: growth
def test_e2_flat_embedding_curve_fails():
    rec = healthy_records()
    for c in CLIPS:
        for g in CFGS:
            rec.a_fork_emb[(c, g, 0.90)] = rec.a_fork_emb[(c, g, 0.05)]  # no growth
    rep = evaluate_stage_m(rec, GATE_HEAD_OK, GATE_DEP_OK, DET_OK, FROZEN)
    endpoints = crit(rep, "endpoints")
    assert not endpoints.passed
    assert "E2" in endpoints.detail
    assert "E1" not in endpoints.detail  # seed floor (0.87 vs 0.76) untouched


# ----------------------------------------------------------- (9) abstain cap
def test_abstain_cap_at_late_s_fails_criterion_5():
    rec = healthy_records()
    for c in CLIPS:
        rec.abstain_fork[(c, HEADLINE, "class", 0.90)] = 0.5  # > 0.30 cap
    rep = evaluate_stage_m(rec, GATE_HEAD_OK, GATE_DEP_OK, DET_OK, FROZEN)
    info = crit(rep, "informativeness")
    assert not info.passed
    assert info.values["abstain_fired"]
    assert any(t.startswith("MICROMAP_FAIL") and "informativeness" in t for t in rep.tokens)
    assert "BEATs" in rep.failure_routing


# ----------------------- (10) video-pinned >= 12/16 FAILS criterion 5 (spec §2)
def _pin_clips(rec: StageMRecords, n: int) -> None:
    for c in CLIPS[:n]:
        rec.a_independent[(c, HEADLINE, "class")] = 0.95
        for s in S_GRID:  # keep endpoints consistent: forks at least as high
            rec.a_fork[(c, HEADLINE, "class", s)] = max(
                rec.a_fork[(c, HEADLINE, "class", s)], 0.95)


def test_video_pinned_12_of_16_fails_with_pool_routing():
    """Revised manual 2 lists informativeness among the PASS criteria: a fired
    pinned check fails Stage M with the widen/re-stratify routing (Codex
    pass-B finding superseding the earlier warning-only reading)."""
    rec = healthy_records()
    _pin_clips(rec, 12)  # exactly 12/16 fires the >= rule
    rep = evaluate_stage_m(rec, GATE_HEAD_OK, GATE_DEP_OK, DET_OK, FROZEN)
    assert rep.informativeness_warning
    assert not crit(rep, "informativeness").passed
    assert any(t.startswith("MICROMAP_FAIL") and "informativeness" in t for t in rep.tokens)
    assert "widen/re-stratify" in rep.failure_routing

    rec2 = healthy_records()
    _pin_clips(rec2, 11)  # 11/16 does not fire
    rep2 = evaluate_stage_m(rec2, GATE_HEAD_OK, GATE_DEP_OK, DET_OK, FROZEN)
    assert not rep2.informativeness_warning
    assert crit(rep2, "informativeness").passed


# ------------------------------------------------------- (11) completeness gate
def test_missing_a_fork_cell_refused():
    rec = healthy_records()
    del rec.a_fork[(CLIPS[3], DEPLOYED, "class", 0.60)]
    with pytest.raises(IncompleteRecordsError, match="a_fork"):
        evaluate_stage_m(rec, GATE_HEAD_OK, GATE_DEP_OK, DET_OK, FROZEN)


def test_nan_a_fork_cell_is_complete():
    """NaN agreements are legitimate (unscorable confident subset) — completeness
    requires the key, not a finite value. One NaN cell must not raise."""
    rec = healthy_records()
    rec.a_fork[(CLIPS[3], HEADLINE, "class", 0.60)] = float("nan")
    rep = evaluate_stage_m(rec, GATE_HEAD_OK, GATE_DEP_OK, DET_OK, FROZEN)
    assert rep.tokens


def test_missing_or_nan_embedding_cells_refused():
    rec = healthy_records()
    del rec.a_fork_emb[(CLIPS[0], HEADLINE, 0.30)]
    with pytest.raises(IncompleteRecordsError, match="a_fork_emb"):
        evaluate_stage_m(rec, GATE_HEAD_OK, GATE_DEP_OK, DET_OK, FROZEN)
    rec2 = healthy_records()
    rec2.a_fork_emb[(CLIPS[0], HEADLINE, 0.30)] = float("nan")  # emb must be finite
    with pytest.raises(IncompleteRecordsError, match="a_fork_emb"):
        evaluate_stage_m(rec2, GATE_HEAD_OK, GATE_DEP_OK, DET_OK, FROZEN)
    rec3 = healthy_records()
    del rec3.a_ind_emb[(CLIPS[5], DEPLOYED)]
    with pytest.raises(IncompleteRecordsError, match="a_ind_emb"):
        evaluate_stage_m(rec3, GATE_HEAD_OK, GATE_DEP_OK, DET_OK, FROZEN)


def test_missing_determinism_axis_refused():
    with pytest.raises(IncompleteRecordsError, match="determinism"):
        evaluate_stage_m(healthy_records(), GATE_HEAD_OK, GATE_DEP_OK,
                         {"presence": 1.0}, FROZEN)


def _subset_records(rec: StageMRecords, clips: tuple[str, ...]) -> StageMRecords:
    sub = StageMRecords(s_grid=rec.s_grid, cfgs=rec.cfgs, axis_ids=rec.axis_ids,
                        clips=clips)
    for name in ("a_fork", "a_independent", "abstain_fork", "abstain_ind",
                 "n_conf_fork", "n_conf_ind", "a_fork_emb", "a_ind_emb"):
        src, dst = getattr(rec, name), getattr(sub, name)
        for k, v in src.items():
            if k[0] in clips:
                dst[k] = v
    return sub


def test_wrong_clip_count_refused_but_expected_clips_override_scores():
    rec = healthy_records()
    partial = _subset_records(rec, rec.clips[:2])
    with pytest.raises(IncompleteRecordsError, match="clips"):
        evaluate_stage_m(partial, GATE_HEAD_OK, GATE_DEP_OK, DET_OK, FROZEN)
    # explicit override scores the small fixture instead of raising
    rep = evaluate_stage_m(partial, GATE_HEAD_OK, GATE_DEP_OK, DET_OK, FROZEN,
                           expected_clips=2)
    assert rep.tokens and rep.tokens[0].startswith("MICROMAP")


# ------------------------------------------------------ (12) records roundtrip
def test_records_roundtrip_with_nan(tmp_path):
    rec = healthy_records()
    rec.a_fork[(CLIPS[2], HEADLINE, "class", 0.30)] = float("nan")
    p = tmp_path / "records.json"
    rec.save(p)
    back = StageMRecords.load(p)
    assert back.s_grid == rec.s_grid and back.cfgs == rec.cfgs
    assert back.axis_ids == rec.axis_ids and back.clips == rec.clips
    # NaN survives as NaN (not dropped, not coerced)
    assert math.isnan(back.a_fork[(CLIPS[2], HEADLINE, "class", 0.30)])
    for name in ("a_fork", "a_independent", "abstain_fork", "abstain_ind",
                 "a_fork_emb", "a_ind_emb"):
        src, dst = getattr(rec, name), getattr(back, name)
        assert set(src) == set(dst)
        for k, v in src.items():
            if math.isnan(v):
                assert math.isnan(dst[k])
            else:
                assert dst[k] == pytest.approx(v)
    assert back.n_conf_fork == rec.n_conf_fork
    assert back.n_conf_ind == rec.n_conf_ind
    assert all(isinstance(v, int) for v in back.n_conf_fork.values())


def test_loaded_records_evaluate_identically(tmp_path):
    rec = healthy_records()
    p = tmp_path / "records.json"
    rec.save(p)
    back = StageMRecords.load(p)
    rep = evaluate_stage_m(back, GATE_HEAD_OK, GATE_DEP_OK, DET_OK, FROZEN)
    assert "MICROMAP_PASS" in rep.tokens


# --------------------------------------------------------- (13) monotonicity
def test_monotonicity_violation_fails():
    rec = healthy_records()
    for c in CLIPS:
        rec.a_fork[(c, HEADLINE, "class", 0.60)] = 0.30  # deep dip below s=0.30
    rep = evaluate_stage_m(rec, GATE_HEAD_OK, GATE_DEP_OK, DET_OK, FROZEN)
    mono = crit(rep, "monotonicity")
    assert not mono.passed
    assert any(t.startswith("MICROMAP_FAIL") and "monotonicity" in t for t in rep.tokens)


def test_tiny_dip_within_tolerance_passes_monotonicity():
    """Decreases inside the pre-registered tolerance band pass even if their CI
    excludes zero (the band absorbs estimator noise)."""
    rec = healthy_records(seed=7)
    for c in CLIPS:
        for g in CFGS:
            rec.a_fork[(c, g, "presence", 0.60)] = rec.a_fork[(c, g, "presence", 0.30)] - 0.01
            rec.a_fork[(c, g, "presence", 0.90)] = 0.97
    rep = evaluate_stage_m(rec, GATE_HEAD_OK, GATE_DEP_OK, DET_OK, FROZEN)
    assert crit(rep, "monotonicity").passed


# ------------------------------------------------------- (14) frozen thresholds
def test_requires_frozen_thresholds():
    with pytest.raises(ValueError, match="frozen"):
        evaluate_stage_m(healthy_records(), GATE_HEAD_OK, GATE_DEP_OK, DET_OK, UNFROZEN)


# ------------------------------------------------------------------ reporting
def test_report_markdown_mentions_non_gating_deployed_arm():
    rep = evaluate_stage_m(healthy_records(), GATE_HEAD_OK, GATE_DEP_FAIL, DET_OK, FROZEN)
    md = rep.to_markdown()
    assert "MICROMAP_PASS" in md
    assert "not gating" in md  # cfg=4.5 arm is adjudicated, not gating
    assert "never scientific evidence" in md
