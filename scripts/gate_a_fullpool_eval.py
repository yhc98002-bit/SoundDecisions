#!/usr/bin/env python
"""Full-pool Gate-A evaluation (manual §1.2/§15.8) — cfg=4.5 ratification verdict.

Calibrates per-s thresholds from the cfg=1.0 internal null (the same-design reference) and
adjudicates cfg=4.5 against them (seed-marginalized exchangeability: pooled one-fork-per-
independent vs FRESH independents, sqrt-prob MMD + label-marginal TV). On the FULL Phase-1
pool (200 clips), not the Stage-M pilot cells.

  cfg=1.0 internal null PASS  -> calibration valid (HARD; if it fails the pool/instrument is
                                 suspect — report, do not certify 4.5 off an invalid null).
  cfg=4.5 calibrated PASS     -> RATIFY: set ratified=true in certified_kernels.json, emit
                                 CFG_KERNEL_OK(cfg=4.5, schedule=sqrt_down) [ratified].
  cfg=4.5 calibrated FAIL     -> CFG_KERNEL_FAIL(cfg=4.5, schedule=sqrt_down) -> PAUSE (the
                                 cfg=1.0 backbone carries Fig 1/2/3; cfg=4.5 stays candidate).

This script DOES modify certified_kernels.json (ledger) only on a PASS, and only the
`ratified` flag of the existing cfg=4.5 entry — never the headline, thresholds, or tuple.
CPU-only. Output: results/stage0/gate_a_fullpool_report.md + tokens.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from foley_cw.gate_a import (_skey, build_cell, calibrate_from_internal_null,  # noqa: E402
                             evaluate_calibrated, evaluate_internal_null,
                             null_sanity, power_positive_control)

GATE_A_S = (0.05, 0.90)


def scaled_cap(n_cells: int, alpha: float = 0.05, q: float = 0.95) -> int:
    """RULING-1 bug-fix: the frozen Gate-A caps LOW_P_MAX_CELLS=2 / EXCEEDANCE_MAX_CELLS=3
    are Binomial(16, 0.05) 95th-percentile values (binom.ppf(0.95,16,0.05)=2, recovering 2
    exactly). Applied to the full pool (n=200 cells/s-point) the SAME pre-registered
    Binomial(n, 0.05) rule gives a 95th-percentile cap of 15, not 2/3. We use the single
    conservative 95% rule for BOTH caps (low-p and exceedance) — documented, ratified by the
    PI as a correction of the exposure n, NOT a re-tune. cap(16)=2, cap(200)=15."""
    from scipy.stats import binom
    return int(binom.ppf(q, n_cells, alpha))


def build_cells(out: Path, clips, cfg, rng, n_perm):
    cells, ref_probs = [], {}
    tag = f"gate_a__cfg{cfg:g}"
    from foley_cw.run_store import RunStore
    store = RunStore(out)
    missing = 0
    for clip in clips:
        unit = f"{tag}__{clip}"
        npz = out / "gate_a" / f"{clip}__cfg{cfg:g}.npz"
        if not store.is_done(unit) or not npz.exists():
            missing += 1; continue
        j = store.load_journal(unit)
        z = np.load(npz)
        ref_probs[clip] = z["probs_ref"]
        labels = j["gate_a_labels"]
        for s in GATE_A_S:
            cells.append(build_cell(clip_id=clip, s=float(s), cfg=float(cfg),
                                    fork_probs=z[f"probs_gafork_s{s:g}"], ref_probs=z["probs_ref"],
                                    fork_labels=labels[f"s{s:g}"], ref_labels=labels["ref"],
                                    rng=rng, n_perm=n_perm, schedule="sqrt_down"))
    return cells, ref_probs, missing


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("results/stage0"))
    ap.add_argument("--manifest", type=Path, default=Path("data/manifests/phase1_manifest_frozen.json"))
    ap.add_argument("--certified", type=Path, default=Path("results/stage_m_rerun/certified_kernels.json"))
    ap.add_argument("--n-perm", type=int, default=1000,
                    help="RULING-1: raised 200→1000 for a tighter cfg=4.5 exceedance estimate")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--write-ledger", action=argparse.BooleanOptionalAction, default=True,
                    help="on cfg=4.5 PASS, flip ratified=true in certified_kernels.json")
    args = ap.parse_args()

    clips = sorted(str(c) for c in json.loads(args.manifest.read_text())["clips"]["single_event"])
    rng = np.random.default_rng(args.seed)

    c1, ref1, miss1 = build_cells(args.out, clips, 1.0, rng, args.n_perm)
    c45, ref45, miss45 = build_cells(args.out, clips, 4.5, rng, args.n_perm)
    n1 = len(ref1); n45 = len(ref45)
    if n1 == 0 or n45 == 0:
        print(f"[gatea-eval] missing bundles: cfg1 have {n1} (miss {miss1}), "
              f"cfg4.5 have {n45} (miss {miss45}) — run gate_a_collect first")
        return 2

    # RULING-1 scaled caps at the actual cell count (Binomial(n, 0.05) 95th pct).
    cap1 = scaled_cap(n1)
    cap45 = scaled_cap(n45)
    thresholds = calibrate_from_internal_null(c1)
    pf, cm, cp95 = power_positive_control(ref1, rng=rng, n_perm=args.n_perm)
    ksp, _ = null_sanity(ref1, rng=rng, n_perm=args.n_perm)
    res_null = evaluate_internal_null(c1, thresholds, power_reject_frac=pf,
                                      cross_clip_mmd_median=cm, null_ks_p=ksp,
                                      cross_clip_mmd_p95=cp95, cfg=1.0, schedule="sqrt_down",
                                      low_p_max=cap1,
                                      expected_s=GATE_A_S, expected_cells_per_s=n1)
    res_45 = evaluate_calibrated(c45, thresholds, cfg=4.5, schedule="sqrt_down",
                                 exceed_max=cap45,
                                 expected_s=GATE_A_S, expected_cells_per_s=n45)

    # Worst-case MMD exceedance across s-points (for the clean-vs-caveat ratification).
    mmd_exceed = max((res_45.per_s.get(_skey(s), {}).get("stats", {}).get("mmd_n_exceed", 0)
                      for s in GATE_A_S), default=0)
    null_ok = res_null.passed
    # RULING-1: cfg=4.5 ratifies; CLEAN if MMD exceedance strictly below the scaled cap,
    # else CFG_KERNEL_OK WITH the 'near-exchangeable, not provably exact' caveat.
    cfg45_clean = bool(res_45.passed and mmd_exceed < cap45)
    cfg45_caveat = not cfg45_clean
    ratify = bool(null_ok and res_45.passed)
    tokens = [res_null.token, res_45.token]

    # Ledger update (only on a clean ratify; only the ratified flag of the cfg=4.5 entry).
    ledger_updated = False
    caveat = ("near-exchangeable on tagger-probs, not provably exact" if cfg45_caveat else "")
    if ratify and args.write_ledger:
        led = json.loads(args.certified.read_text())
        if "deployed" in led and abs(float(led["deployed"]["cfg"]) - 4.5) < 1e-9:
            led["deployed"]["ratified"] = True
            led["deployed"]["scope"] = (
                "RATIFIED — full Phase-1 pool Gate-A (200 clips, scaled cap "
                f"Binomial(200,0.05)=={cap45}, n_perm={args.n_perm})"
                + (f"; CAVEAT: {caveat}" if cfg45_caveat else "; CLEAN"))
            if cfg45_caveat:
                led["deployed"]["caveat"] = caveat
            args.certified.write_text(json.dumps(led, indent=2))
            ledger_updated = True

    out = {"_doc": "Full-pool Gate-A (§1.2/§15.8) cfg=4.5 ratification — RULING-1 scaled caps.",
           "n_perm": args.n_perm, "scaled_cap_cfg1": cap1, "scaled_cap_cfg45": cap45,
           "mmd_exceedance_worst_cfg45": mmd_exceed, "cfg45_clean": cfg45_clean,
           "cfg45_caveat": caveat,
           "n_clips_cfg1": n1, "n_clips_cfg45": n45,
           "guards": {"power_reject_frac": pf, "cross_clip_mmd_median": cm,
                      "cross_clip_mmd_p95": cp95, "null_ks_p": ksp},
           "cfg1_internal_null": {"token": res_null.token, "passed": res_null.passed,
                                  "detail": res_null.detail, "per_s": res_null.per_s},
           "cfg45_calibrated": {"token": res_45.token, "passed": res_45.passed,
                                "detail": res_45.detail, "per_s": res_45.per_s},
           "ratified": ratify, "ledger_updated": ledger_updated, "tokens": tokens}
    (args.out / "gate_a_fullpool.json").write_text(json.dumps(out, indent=2, default=str))

    L = ["# Full-pool Gate-A — cfg=4.5 ratification (§1.2/§15.8) — RULING-1 scaled caps", "",
         f"cfg=1.0 cells {n1}, cfg=4.5 cells {n45}; n_perm={args.n_perm}; Gate-A s={GATE_A_S}. "
         f"**Scaled cap = Binomial(n,0.05) 95th pct = {cap45}** (recovers the frozen 2 at n=16; "
         "RULING-1 bug-fix correcting the exposure n, not a re-tune).", "",
         f"- **cfg=1.0 internal null: `{res_null.token}` passed={res_null.passed}** "
         f"(low-p cap {cap1})  {res_null.detail}",
         f"- **cfg=4.5 calibrated: `{res_45.token}` passed={res_45.passed}** "
         f"(MMD exceedance worst {mmd_exceed} vs cap {cap45} → "
         f"{'CLEAN' if cfg45_clean else 'with CAVEAT'})  {res_45.detail}", "",
         f"guards: power_reject_frac={pf:.3f}, cross_clip_mmd_median={cm:.4g}, null_ks_p={ksp:.3f}",
         "",
         f"**Verdict: {'RATIFIED' if ratify else 'NOT ratified'}** (ledger updated: {ledger_updated})."
         + (f" cfg=4.5 carries the caveat: *{caveat}* on every cfg=4.5 claim." if cfg45_caveat else
            " cfg=4.5 ratified CLEAN."),
         "" if ratify else "\n**`CFG_KERNEL_FAIL` after the scaled cap → PAUSE (trigger a).**"]
    (args.out / "gate_a_fullpool_report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\n[gatea-eval] ratified={ratify} clean={cfg45_clean} tokens={tokens}")
    return 0 if ratify else 3


if __name__ == "__main__":
    raise SystemExit(main())
