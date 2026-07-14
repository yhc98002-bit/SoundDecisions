"""Tests for foley_cw.gate_a — V2 seed-marginalized exchangeability instrument.

Targets the revised manual (experiment/LONG_RANGE_EXPERIMENT_PLAN_revised.md
sections 1.2 / 2): sqrt-prob feature transform, label-marginal TV on the
extended alphabet (incl. 'abstain'), one-fork-per-independent cells, internal-
null calibration, guards g1-g3, the cfg=1.0 HARD internal-null verdict, the
calibrated cfg>1 adjudication (MW / exceedance / gross per statistic), and
token formats. CPU-only, deterministic seeds, n_perm=99 throughout.
"""

from __future__ import annotations

import numpy as np
import pytest

from foley_cw.gate_a import (ALPHA_SIG, EXCEEDANCE_MAX_CELLS, GROSS_FACTOR,
                             LOW_P_MAX_CELLS, MW_MIN_P, NULL_KS_MIN_P,
                             POWER_MIN_REJECT_FRAC, THETA_SEPARATION_MAX_FRAC,
                             TV_GROSS_CAP, GateACell, build_cell,
                             calibrate_from_internal_null, check_guards,
                             evaluate_calibrated, evaluate_internal_null,
                             label_marginal_tv, median_heuristic_bandwidth,
                             mmd2_unbiased, mmd_permutation_p, null_sanity,
                             power_positive_control, sqrt_probs)

D = 20  # stand-in for the 527-dim tagger probability space
N_CLIPS = 16

# Deterministic per-s null statistic levels (positive, so theta_mmd > 0).
MMD_NULL = np.linspace(1e-3, 1e-2, N_CLIPS)
TV_NULL = np.linspace(0.05, 0.3, N_CLIPS)


def _pool(rng, n=8, d=D, peak=0, conc=8.0):
    """Synthetic tagger probability pool: Dirichlet concentrated on `peak`."""
    alpha = np.full(d, 0.3)
    alpha[peak % d] = conc
    return rng.dirichlet(alpha, size=n)


def _cells(cfg, mmd_vals=MMD_NULL, tv_vals=TV_NULL, s_grid=(0.05, 0.3),
           p_vals=None, schedule="constant"):
    """Cells with controlled statistics (p_vals indexed by clip, same at each s)."""
    cells = []
    for s in s_grid:
        for i in range(len(mmd_vals)):
            p = 0.5 if p_vals is None else float(p_vals[i])
            cells.append(GateACell(clip_id=f"c{i:02d}", s=s, cfg=cfg,
                                   mmd2=float(mmd_vals[i]), p_value=p,
                                   tv=float(tv_vals[i]), n_fork=8, n_ref=8,
                                   schedule=schedule))
    return cells


def _null_like_p():
    """p-values for an exact-kernel arm: exactly LOW_P_MAX_CELLS low-p cells."""
    p = (np.arange(N_CLIPS) + 1.0) / (N_CLIPS + 1.0)   # all > ALPHA_SIG
    p[0], p[1] = 0.01, 0.02                            # exactly 2 < ALPHA_SIG
    return p


# ------------------------------------------------------------------ sqrt_probs
def test_sqrt_probs_zeros_stay_zero_and_values():
    P = np.array([[0.0, 0.25, 1.0], [0.0, 0.04, 0.81]])
    S = sqrt_probs(P)
    assert S.shape == P.shape
    np.testing.assert_allclose(S[:, 0], 0.0)
    np.testing.assert_allclose(S, [[0.0, 0.5, 1.0], [0.0, 0.2, 0.9]])


def test_sqrt_probs_monotone():
    p = np.linspace(0.0, 1.0, 50)
    s = sqrt_probs(p)
    assert np.all(np.diff(s) > 0)            # strictly increasing on (0, 1]
    assert np.all((s >= 0) & (s <= 1))


def test_sqrt_probs_clips_negative_noise():
    # tiny negative numerical noise must not produce NaNs
    S = sqrt_probs(np.array([-1e-9, 0.0, 0.5]))
    assert np.all(np.isfinite(S))
    assert S[0] == pytest.approx(0.0)


# ------------------------------------------------------------ label-marginal TV
def test_tv_identical_and_disjoint():
    assert label_marginal_tv(["dog", "cat"], ["dog", "cat"]) == pytest.approx(0.0)
    assert label_marginal_tv(["dog"] * 4, ["cat"] * 4) == pytest.approx(1.0)


def test_tv_extended_alphabet_with_abstain():
    a = ["dog", "dog", "cat", "abstain"]
    b = ["dog", "cat", "cat", "abstain"]
    # support {abstain, cat, dog}: pa=(.25,.25,.5), pb=(.25,.5,.25) -> TV=0.25
    assert label_marginal_tv(a, b) == pytest.approx(0.25)


def test_tv_mode_lock_onto_abstain_is_a_marginal_break():
    forks = ["abstain"] * 8
    refs = ["dog"] * 3 + ["cat"] * 3 + ["other"] + ["abstain"]
    # pa=(1,0,0,0) vs pb=(.125,.375,.375,.125) on the extended alphabet
    assert label_marginal_tv(forks, refs) == pytest.approx(0.875)


def test_tv_empty_support_is_nan():
    assert np.isnan(label_marginal_tv([], []))


# ----------------------------------------------------------- MMD machinery
def test_mmd2_unbiased_mean_near_zero_under_null():
    rng = np.random.default_rng(0)
    vals = [mmd2_unbiased(rng.standard_normal((8, 16)), rng.standard_normal((8, 16)))
            for _ in range(20)]
    assert abs(float(np.mean(vals))) < 0.05  # unbiased: can be negative, mean ~0


def test_mmd2_detects_shift_and_needs_two_samples():
    rng = np.random.default_rng(1)
    null = np.mean([mmd2_unbiased(rng.standard_normal((8, 16)),
                                  rng.standard_normal((8, 16))) for _ in range(10)])
    shifted = np.mean([mmd2_unbiased(rng.standard_normal((8, 16)),
                                     rng.standard_normal((8, 16)) + 2.0)
                       for _ in range(10)])
    assert shifted > null + 0.1
    with pytest.raises(ValueError):
        mmd2_unbiased(rng.standard_normal((1, 4)), rng.standard_normal((5, 4)))


def test_mmd_permutation_p_small_under_separation():
    rng = np.random.default_rng(2)
    assert median_heuristic_bandwidth(_pool(rng), _pool(rng)) > 0
    _, p = mmd_permutation_p(sqrt_probs(_pool(rng, n=12, peak=2)),
                             sqrt_probs(_pool(rng, n=12, peak=9)),
                             n_perm=99, rng=rng)
    assert p <= ALPHA_SIG


# ------------------------------------------------------------------ build_cell
def test_build_cell_same_pool_is_null_like():
    rng = np.random.default_rng(12)
    fork, ref = _pool(rng, peak=3), _pool(rng, peak=3)
    cell = build_cell("c0", 0.3, 1.0, fork, ref,
                      ["dog"] * 5 + ["abstain"] * 3, ["dog"] * 6 + ["abstain"] * 2,
                      rng=np.random.default_rng(0), n_perm=99)
    assert cell.clip_id == "c0"
    assert (cell.s, cell.cfg) == pytest.approx((0.3, 1.0))
    assert cell.n_fork == 8 and cell.n_ref == 8 and cell.schedule == "constant"
    assert np.isfinite(cell.mmd2)
    assert 0.0 < cell.p_value <= 1.0 and cell.p_value > ALPHA_SIG
    assert cell.tv == pytest.approx(0.125)


def test_build_cell_separated_pools_reject():
    rng = np.random.default_rng(12)
    same_fork, same_ref = _pool(rng, peak=3), _pool(rng, peak=3)
    null_cell = build_cell("c0", 0.3, 1.0, same_fork, same_ref, ["dog"] * 8,
                           ["dog"] * 8, rng=np.random.default_rng(0), n_perm=99)
    fork, ref = _pool(rng, peak=3), _pool(rng, peak=11)
    cell = build_cell("c1", 0.6, 4.5, fork, ref, ["dog"] * 8, ["cat"] * 8,
                      rng=np.random.default_rng(0), n_perm=99, schedule="linear_down")
    assert cell.p_value <= ALPHA_SIG
    assert cell.mmd2 > null_cell.mmd2
    assert cell.tv == pytest.approx(1.0)     # disjoint label marginals
    assert cell.schedule == "linear_down"


# ----------------------------------------------------------------- calibration
def test_calibrate_per_s_thresholds_and_raw_null_lists():
    th = calibrate_from_internal_null(_cells(1.0))
    assert set(th.theta_mmd) == {"0.05", "0.3"} == set(th.theta_tv)
    for sk in ("0.05", "0.3"):
        assert th.theta_mmd[sk] == pytest.approx(float(np.percentile(MMD_NULL, 95)))
        assert th.theta_tv[sk] == pytest.approx(float(np.percentile(TV_NULL, 95)))
        assert sorted(th.null_mmd[sk]) == pytest.approx(sorted(map(float, MMD_NULL)))
        assert len(th.null_tv[sk]) == N_CLIPS
    assert th.n_ref_cells == 2 * N_CLIPS
    assert "cfg=1.0" in th.source


def test_calibrate_excludes_nan_tv_from_null():
    cells = _cells(1.0, s_grid=(0.05,))
    cells.append(GateACell(clip_id="cx", s=0.05, cfg=1.0, mmd2=0.005,
                           p_value=0.5, tv=float("nan")))
    th = calibrate_from_internal_null(cells)
    assert len(th.null_mmd["0.05"]) == N_CLIPS + 1   # mmd kept
    assert len(th.null_tv["0.05"]) == N_CLIPS        # nan tv dropped
    assert np.isfinite(th.theta_tv["0.05"])


def test_calibrate_empty_raises():
    with pytest.raises(ValueError):
        calibrate_from_internal_null([])


# ---------------------------------------------------------------------- guards
def test_check_guards_pass():
    ok, guards, reason = check_guards(0.95, 1.0, 0.5, {"0.05": 0.01})
    assert ok and reason == ""
    assert guards["power_reject_frac"] == pytest.approx(0.95)
    assert guards["cross_clip_mmd_median"] == pytest.approx(1.0)
    assert guards["null_ks_p"] == pytest.approx(0.5)
    assert "cross_clip_mmd_p95" in guards  # frozen interpretation #10 reading


def test_check_guards_g3_uses_p95_when_available():
    """Frozen interpretation #10: g3 compares theta to the cross-clip 95th pct;
    a theta above the median but below the p95 passes (run-2 misfire mode)."""
    ok, _, reason = check_guards(0.95, 0.32, 0.5, {"0.05": 0.34},
                                 cross_clip_mmd_p95=0.60)
    assert ok and reason == ""
    ok2, _, reason2 = check_guards(0.95, 0.32, 0.5, {"0.05": 0.59},
                                   cross_clip_mmd_p95=0.60)
    assert not ok2 and "g3" in reason2  # above 0.9 * p95 still fails


def test_check_guards_g1_power_failure():
    ok, _, reason = check_guards(POWER_MIN_REJECT_FRAC - 0.1, 1.0, 0.5, {"0.05": 0.01})
    assert not ok and "g1" in reason
    ok, _, reason = check_guards(float("nan"), 1.0, 0.5, {"0.05": 0.01})
    assert not ok and "g1" in reason


def test_check_guards_g2_null_ks_failure():
    ok, _, reason = check_guards(0.95, 1.0, NULL_KS_MIN_P / 10.0, {"0.05": 0.01})
    assert not ok and "g2" in reason


def test_check_guards_g3_degenerate_theta():
    for bad in (0.0, -0.002, float("nan")):
        ok, _, reason = check_guards(0.95, 1.0, 0.5, {"0.3": bad})
        assert not ok and "g3 degenerate" in reason


def test_check_guards_g3_threshold_not_separated():
    theta = THETA_SEPARATION_MAX_FRAC + 0.05   # > 0.9 x cross-clip median of 1.0
    ok, _, reason = check_guards(0.95, 1.0, 0.5, {"0.3": theta})
    assert not ok and "not separated" in reason


# ------------------------------------------------- internal null (cfg=1.0 HARD)
def test_internal_null_passes_with_low_p_at_cap():
    cells = _cells(1.0, p_vals=_null_like_p())
    th = calibrate_from_internal_null(cells)
    res = evaluate_internal_null(cells, th, power_reject_frac=0.95,
                                 expected_s=(0.05, 0.3), expected_cells_per_s=N_CLIPS,
                                 cross_clip_mmd_median=1.0, null_ks_p=0.5)
    assert res.passed
    assert res.token == "CFG_KERNEL_OK(cfg=1)"
    for sk in ("0.05", "0.3"):
        assert res.per_s[sk]["n_low_p"] == LOW_P_MAX_CELLS
        assert res.per_s[sk]["cap"] == LOW_P_MAX_CELLS
        assert res.per_s[sk]["ok"]
    assert res.guards["power_reject_frac"] == pytest.approx(0.95)


def test_internal_null_fails_on_low_p_pileup_at_one_s():
    cells = _cells(1.0, p_vals=(np.arange(N_CLIPS) + 1.0) / (N_CLIPS + 1.0))
    n_broken = 0
    for c in cells:                      # 4 low-p cells at s=0.3 only
        if c.s == 0.3 and n_broken < 4:
            c.p_value = 0.001
            n_broken += 1
    th = calibrate_from_internal_null(cells)
    res = evaluate_internal_null(cells, th, power_reject_frac=0.95,
                                 expected_s=(0.05, 0.3), expected_cells_per_s=N_CLIPS,
                                 cross_clip_mmd_median=1.0, null_ks_p=0.5)
    assert not res.passed
    assert res.token == "CFG_KERNEL_FAIL(cfg=1)"
    assert res.per_s["0.05"]["ok"] and not res.per_s["0.3"]["ok"]
    assert res.per_s["0.3"]["n_low_p"] == 4
    assert "STOP-level" in res.detail    # routing per revised manual section 2


def test_internal_null_guard_failure_is_underpowered_not_fail():
    cells = _cells(1.0, p_vals=_null_like_p())
    th = calibrate_from_internal_null(cells)
    res = evaluate_internal_null(cells, th, power_reject_frac=0.3,
                                 expected_s=(0.05, 0.3), expected_cells_per_s=N_CLIPS,
                                 cross_clip_mmd_median=1.0, null_ks_p=0.5)
    assert res.token == "CFG_KERNEL_FAIL(cfg=1)" and not res.passed
    assert res.extra.get("underpowered") is True
    assert "UNDERPOWERED" in res.detail and "never pass silently" in res.detail


def test_internal_null_degenerate_theta_is_underpowered():
    # unbiased-MMD null entirely <= 0 -> theta_mmd <= 0 -> g3 fires
    cells = _cells(1.0, mmd_vals=-MMD_NULL, p_vals=_null_like_p())
    th = calibrate_from_internal_null(cells)
    res = evaluate_internal_null(cells, th, power_reject_frac=0.95,
                                 expected_s=(0.05, 0.3), expected_cells_per_s=N_CLIPS,
                                 cross_clip_mmd_median=1.0, null_ks_p=0.5)
    assert not res.passed and res.extra.get("underpowered") is True


def test_internal_null_schedule_token_suffix():
    cells = _cells(1.0, p_vals=(np.arange(N_CLIPS) + 1.0) / (N_CLIPS + 1.0),
                   schedule="linear_down")
    th = calibrate_from_internal_null(cells)
    res = evaluate_internal_null(cells, th, power_reject_frac=0.95,
                                 expected_s=(0.05, 0.3), expected_cells_per_s=N_CLIPS,
                                 cross_clip_mmd_median=1.0, null_ks_p=0.5,
                                 schedule="linear_down")
    assert res.token == "CFG_KERNEL_OK(cfg=1, schedule=linear_down)"


# ------------------------------------------- calibrated adjudication (cfg > 1)
THETA = float(np.percentile(MMD_NULL, 95))


def _calibrated(target_cells, cfg=4.5, schedule="constant",
                expected_s=(0.3,), expected_cells_per_s=N_CLIPS):
    th = calibrate_from_internal_null(_cells(1.0))
    return evaluate_calibrated(target_cells, th, cfg=cfg, schedule=schedule,
                               expected_s=expected_s,
                               expected_cells_per_s=expected_cells_per_s)


def test_calibrated_same_distribution_passes():
    res = _calibrated(_cells(4.5), expected_s=(0.05, 0.3))  # identical levels as null
    assert res.passed
    assert res.token == "CFG_KERNEL_OK(cfg=4.5)"
    for sk in ("0.05", "0.3"):
        checks = res.per_s[sk]["checks"]
        assert checks == {"mmd:mw": True, "mmd:exceedance": True, "mmd:gross": True,
                          "tv:mw": True, "tv:exceedance": True, "tv:gross": True}
        assert res.per_s[sk]["stats"]["mmd_mw_p"] >= MW_MIN_P


def test_calibrated_mw_shift_fails_mmd_mw_only():
    # stochastically larger but every cell below theta: pure distribution shift
    shifted = np.linspace(0.008, 0.0095, N_CLIPS)
    assert float(shifted.max()) < THETA
    res = _calibrated(_cells(4.5, mmd_vals=shifted, s_grid=(0.3,)))
    assert not res.passed and res.token == "CFG_KERNEL_FAIL(cfg=4.5)"
    checks = res.per_s["0.3"]["checks"]
    assert not checks["mmd:mw"]
    assert checks["mmd:exceedance"] and checks["mmd:gross"]
    assert all(checks[k] for k in checks if k.startswith("tv:"))
    assert "mmd:mw" in res.detail


def test_calibrated_localized_exceedance_fails():
    # 5 of 16 cells above theta (but < gross) at one s; MW barely moves
    vals = MMD_NULL.copy()
    vals[[2, 5, 8, 11, 15]] = 1.5 * THETA
    assert np.sum(vals > THETA) == 5 > EXCEEDANCE_MAX_CELLS
    res = _calibrated(_cells(4.5, mmd_vals=vals, s_grid=(0.3,)))
    assert not res.passed
    checks = res.per_s["0.3"]["checks"]
    assert not checks["mmd:exceedance"]
    assert checks["mmd:mw"] and checks["mmd:gross"]
    assert res.per_s["0.3"]["stats"]["mmd_n_exceed"] == 5


def test_calibrated_single_gross_cell_fails():
    vals = MMD_NULL.copy()
    vals[7] = (GROSS_FACTOR + 1.0) * THETA   # one cell >> 3 x theta
    res = _calibrated(_cells(4.5, mmd_vals=vals, s_grid=(0.3,)))
    assert not res.passed
    checks = res.per_s["0.3"]["checks"]
    assert not checks["mmd:gross"]
    assert checks["mmd:exceedance"]          # 2 cells above theta <= cap
    assert checks["mmd:mw"]                  # rank shift from one cell is tiny
    assert res.per_s["0.3"]["stats"]["mmd_worst"] == pytest.approx(vals[7])


def test_calibrated_tv_only_break_with_healthy_mmd():
    # mmd identical to null; labels mode-locked -> TV ~0.95 above the gross cap
    tv_locked = np.full(N_CLIPS, 0.95)
    theta_tv = float(np.percentile(TV_NULL, 95))
    assert 0.95 > min(GROSS_FACTOR * theta_tv, TV_GROSS_CAP)
    res = _calibrated(_cells(4.5, tv_vals=tv_locked, s_grid=(0.3,)))
    assert not res.passed
    checks = res.per_s["0.3"]["checks"]
    assert all(checks[k] for k in checks if k.startswith("mmd:"))
    assert not checks["tv:mw"] and not checks["tv:exceedance"] and not checks["tv:gross"]


def test_calibrated_abstain_mode_lock_end_to_end():
    """Mode-locking fork labels onto 'abstain' breaks TV while the sqrt-prob MMD
    stays exchangeable (same prob matrices in both arms)."""
    rng = np.random.default_rng(13)
    lab_rng = np.random.default_rng(14)
    alphabet = ["dog", "cat", "other", "abstain"]
    pdist = [0.4, 0.3, 0.15, 0.15]
    null_cells, target_cells = [], []
    for i in range(N_CLIPS):
        fork, ref = _pool(rng, peak=i), _pool(rng, peak=i)
        null_cells.append(build_cell(
            f"c{i}", 0.6, 1.0, fork, ref,
            list(lab_rng.choice(alphabet, p=pdist, size=8)),
            list(lab_rng.choice(alphabet, p=pdist, size=8)),
            rng=np.random.default_rng(100 + i), n_perm=99))
        target_cells.append(build_cell(   # same matrices, abstain-locked forks
            f"c{i}", 0.6, 4.5, fork, ref, ["abstain"] * 8,
            list(lab_rng.choice(alphabet, p=pdist, size=8)),
            rng=np.random.default_rng(200 + i), n_perm=99))
    th = calibrate_from_internal_null(null_cells)
    res = evaluate_calibrated(target_cells, th, cfg=4.5,
                              expected_s=(0.6,), expected_cells_per_s=N_CLIPS)
    assert not res.passed and res.token == "CFG_KERNEL_FAIL(cfg=4.5)"
    checks = res.per_s["0.6"]["checks"]
    assert all(checks[k] for k in checks if k.startswith("mmd:"))
    assert not all(checks[k] for k in checks if k.startswith("tv:"))
    assert res.per_s["0.6"]["stats"]["tv_worst"] >= 0.8


def test_calibrated_schedule_token_suffix():
    res = _calibrated(_cells(2.5, schedule="sqrt_down"), cfg=2.5, schedule="sqrt_down",
                      expected_s=(0.05, 0.3))
    assert res.passed
    assert res.token == "CFG_KERNEL_OK(cfg=2.5, schedule=sqrt_down)"
    bad = _cells(2.5, mmd_vals=MMD_NULL + 0.05, s_grid=(0.3,), schedule="sqrt_down")
    res = _calibrated(bad, cfg=2.5, schedule="sqrt_down")
    assert res.token == "CFG_KERNEL_FAIL(cfg=2.5, schedule=sqrt_down)"


def test_calibrated_no_cells_or_missing_null_never_passes():
    th = calibrate_from_internal_null(_cells(1.0))
    # empty cells -> design check fires (registered grid absent) -> underpowered FAIL
    empty = evaluate_calibrated([], th, cfg=4.5)
    assert not empty.passed and empty.extra.get("underpowered") is True
    # cells at an s-point with no calibrated null -> data checks fail, never passes
    orphan = _cells(4.5, s_grid=(0.9,))   # no null cells at s=0.9
    res = evaluate_calibrated(orphan, th, cfg=4.5,
                              expected_s=(0.9,), expected_cells_per_s=N_CLIPS)
    assert not res.passed
    assert res.per_s["0.9"]["checks"] == {"mmd:data": False, "tv:data": False}


def test_design_enforcement_default_grid():
    """Codex pass-A finding: the registered {0.05, 0.90} x 16-cell design is
    enforced by default — wrong grids or short counts never pass."""
    cells = _cells(1.0, p_vals=_null_like_p())  # s_grid (0.05, 0.3) != registered
    th = calibrate_from_internal_null(cells)
    res = evaluate_internal_null(cells, th, power_reject_frac=0.95,
                                 cross_clip_mmd_median=1.0, null_ks_p=0.5)
    assert not res.passed and res.extra.get("underpowered") is True
    short = [c for c in _cells(1.0, s_grid=(0.05, 0.90), p_vals=_null_like_p())
             if not (c.s == 0.90 and c.clip_id == "c00")]  # 15 cells at s=0.90
    res2 = evaluate_internal_null(short, calibrate_from_internal_null(short),
                                  power_reject_frac=0.95,
                                  cross_clip_mmd_median=1.0, null_ks_p=0.5)
    assert not res2.passed and res2.extra.get("underpowered") is True


# -------------------------------------------------------- guard computations
def test_power_positive_control_separated_clips_reject():
    rng = np.random.default_rng(10)
    pools = {f"c{i}": _pool(rng, peak=2 * i, conc=10.0) for i in range(5)}
    frac, med, p95 = power_positive_control(pools, rng=rng, n_perm=99)
    assert frac >= POWER_MIN_REJECT_FRAC
    assert med > 0


def test_power_positive_control_needs_pairs():
    rng = np.random.default_rng(10)
    frac, med, p95 = power_positive_control({"c0": _pool(rng)}, rng=rng, n_perm=99)
    assert np.isnan(frac) and np.isnan(med)


def test_null_sanity_same_pool_half_splits_uniform():
    rng = np.random.default_rng(11)
    pools = {f"c{i}": _pool(rng, peak=i) for i in range(8)}
    ks_p, pvals = null_sanity(pools, rng=rng, n_perm=99)
    assert len(pvals) == 8
    assert all(0.0 < p <= 1.0 for p in pvals)
    assert np.isfinite(ks_p) and ks_p > NULL_KS_MIN_P


def test_null_sanity_too_few_pools_is_nan():
    rng = np.random.default_rng(11)
    pools = {f"c{i}": _pool(rng, peak=i) for i in range(3)}
    ks_p, pvals = null_sanity(pools, rng=rng, n_perm=99)
    assert np.isnan(ks_p) and len(pvals) == 3
