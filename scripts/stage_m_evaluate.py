#!/usr/bin/env python
"""Stage-M CPU evaluation — REVISED protocol (revised manual section 2).

Consumes results/stage_m_rerun/{journal,gate_a,measurements} and produces:
  micromap_curves.csv        clip x cfg x axis x s: confident A_fork, A_ind,
                             commit, abstain rates, n_conf, embedding cosines
  gate_a_report.md           internal null @1.0 (HARD) + calibrated @4.5
  micromap_report.md         five revised criteria + tokens + routing
  tokens.json, logging_contract_audit.md

Gating happens at the headline cfg=1.0; the cfg=4.5 Gate-A verdict is
adjudicated and reported (manual-1.2 routing on failure, never a halt by
itself). Exit 0 on MICROMAP_PASS, 2 otherwise.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from foley_cw.commitment import commit_gain  # noqa: E402
from foley_cw.config import load_config  # noqa: E402
from foley_cw.gate_a import (GateAResult, build_cell, calibrate_from_internal_null,  # noqa: E402
                             evaluate_calibrated, evaluate_internal_null, null_sanity,
                             power_positive_control)
from foley_cw.stage_m import (DEPLOYED_CFG, HEADLINE_CFG, StageMRecords,  # noqa: E402
                              evaluate_stage_m)

STAGE_M_AXIS_IDS = ("presence", "class")


def load_journals(out: Path) -> tuple[list[dict], dict]:
    """Units + a MERGED extras across all shards (Codex T1 High finding: a
    non-OK SDE token from ANY shard must count; the worst token per cfg wins)."""
    units, extras_shards = [], []
    for p in sorted((out / "journal").glob("*.json")):
        d = json.loads(p.read_text())
        (extras_shards if p.stem.startswith("extras__") else units).append(d)
    if not units:
        raise SystemExit(f"no unit journals under {out}/journal")
    if not extras_shards:
        raise SystemExit("no extras journal (SDE re-validation + determinism)")
    extras = dict(extras_shards[0])
    for key in ("sde_token_cfg1", "sde_token_cfg45"):
        toks = [e.get(key) for e in extras_shards if e.get(key) is not None]
        # any non-OK token across shards downgrades the merged verdict
        non_ok = [t for t in toks if t != "OK"]
        extras[key] = non_ok[0] if non_ok else (toks[0] if toks else "MISSING")
    return units, extras


def build_records(units: list[dict], s_grid, cfgs) -> StageMRecords:
    clips = tuple(sorted({u["clip"] for u in units}))
    rec = StageMRecords(s_grid=tuple(s_grid), cfgs=tuple(cfgs),
                        axis_ids=STAGE_M_AXIS_IDS, clips=clips)
    for u in units:
        c, g = u["clip"], float(u["cfg"])
        for a, v in u["a_independent"].items():
            rec.a_independent[(c, g, a)] = float(v) if v is not None else float("nan")
            rec.n_conf_ind[(c, g, a)] = int(u["n_conf_ind"][a])
            rec.abstain_ind[(c, g, a)] = float(u["abstain_ind"][a])
        for key, v in u["a_fork"].items():
            a, s = key.split("|")
            k4 = (c, g, a, float(s))
            rec.a_fork[k4] = float(v) if v is not None else float("nan")
            rec.n_conf_fork[k4] = int(u["n_conf_fork"][key])
            rec.abstain_fork[k4] = float(u["abstain_fork"][key])
        for sk, v in u["a_fork_emb"].items():
            rec.a_fork_emb[(c, g, float(sk))] = float(v)
        rec.a_ind_emb[(c, g)] = float(u["a_ind_emb"])
    return rec


def build_gate_a(out: Path, units: list[dict], rng: np.random.Generator,
                 n_perm: int) -> tuple[dict, dict]:
    """cells_by_cfg, ref_probs_by_cfg_clip (for the guards)."""
    cells = defaultdict(list)
    ref_probs = defaultdict(dict)
    for u in units:
        c, g = u["clip"], float(u["cfg"])
        npz_path = out / "gate_a" / f"{c}__cfg{g:g}.npz"
        if not npz_path.exists():
            raise SystemExit(f"missing Gate-A bundle for journaled unit {c} cfg={g:g}")
        z = np.load(npz_path)
        ref_probs[g][c] = z["probs_ref"]
        labels = u["gate_a_labels"]
        for s in u["gate_a_s"]:
            cells[g].append(build_cell(
                clip_id=c, s=float(s), cfg=g,
                fork_probs=z[f"probs_gafork_s{float(s):g}"], ref_probs=z["probs_ref"],
                fork_labels=labels[f"s{float(s):g}"], ref_labels=labels["ref"],
                rng=rng, n_perm=n_perm))
    return cells, ref_probs


def write_curves_csv(path: Path, rec: StageMRecords) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip", "cfg", "axis_id", "s", "a_fork_conf", "n_conf", "abstain_fork",
                    "a_independent_conf", "abstain_ind", "commit_gain", "a_fork_emb",
                    "a_ind_emb"])
        for (c, g, a, s), af in sorted(rec.a_fork.items()):
            ai = rec.a_independent.get((c, g, a), float("nan"))
            cg = (commit_gain(af, ai)
                  if np.isfinite(af) and np.isfinite(ai) else float("nan"))
            w.writerow([c, g, a, s,
                        f"{af:.6f}" if np.isfinite(af) else "NaN",
                        rec.n_conf_fork.get((c, g, a, s), ""),
                        f"{rec.abstain_fork.get((c, g, a, s), float('nan')):.4f}",
                        f"{ai:.6f}" if np.isfinite(ai) else "NaN",
                        f"{rec.abstain_ind.get((c, g, a), float('nan')):.4f}",
                        f"{cg:.6f}" if np.isfinite(cg) else "NaN",
                        f"{rec.a_fork_emb.get((c, g, s), float('nan')):.6f}",
                        f"{rec.a_ind_emb.get((c, g), float('nan')):.6f}"])


def write_gate_a_report(out: Path, thresholds, res_null: GateAResult,
                        res_dep: GateAResult) -> None:
    lines = ["# Gate-A Report — seed-marginalized exchangeability (revised manual 1.2)", "",
             f"Feature space: sqrt 527-dim tagger-prob vectors + extended-alphabet label TV.",
             f"Null: {thresholds.n_ref_cells} cfg={HEADLINE_CFG:g} cells; per-s theta_mmd="
             f"{ {k: round(v, 4) for k, v in thresholds.theta_mmd.items()} }, theta_tv="
             f"{ {k: round(v, 4) for k, v in thresholds.theta_tv.items()} }", "",
             f"## Internal null @ cfg={HEADLINE_CFG:g} (HARD): **{res_null.token}**",
             f"Guards: {json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in res_null.guards.items()})}",
             f"Per-s: {json.dumps(res_null.per_s, default=str)}",
             res_null.detail, "",
             f"## Adjudicated @ cfg={DEPLOYED_CFG:g} (non-gating): **{res_dep.token}**",
             f"Per-s: {json.dumps(res_dep.per_s, default=str)}",
             res_dep.detail, ""]
    (out / "gate_a_report.md").write_text("\n".join(lines) + "\n")


def write_logging_audit(out: Path, units: list[dict]) -> None:
    cats = {"features": "features/*.npz", "previews": "previews/*.wav",
            "finals": "finals/*.wav", "audit_wavs": "audit_wavs/*.wav",
            "measurements": "measurements/*.jsonl", "journal": "journal/*.json",
            "gate_a": "gate_a/*.npz"}
    lines = ["# Logging Contract Audit — Stage-M re-run (revised manual 1.4)", "",
             "| category | files | bytes |", "|---|---|---|"]
    total = 0
    for name, pat in cats.items():
        files = list(out.glob(pat))
        nbytes = sum(p.stat().st_size for p in files)
        total += nbytes
        lines.append(f"| {name} | {len(files)} | {nbytes:,} |")
    lines += ["", f"**Total: {total/1e9:.3f} GB of the 100 GB hard cap.**", ""]
    if units and "budget" in units[-1]:
        lines.append(f"Runner budget snapshot: `{json.dumps(units[-1]['budget'])}`")
    (out / "logging_contract_audit.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("results/stage_m_rerun"))
    ap.add_argument("--s-grid", default="0.05,0.30,0.60,0.90")
    ap.add_argument("--cfgs", default="1.0,4.5")
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--expected-clips", type=int, default=16,
                    help="override ONLY for plumbing smoke; smoke tokens are SMOKE_ONLY")
    args = ap.parse_args()

    s_grid = tuple(float(x) for x in args.s_grid.split(","))
    cfgs = tuple(float(x) for x in args.cfgs.split(","))
    rng = np.random.default_rng(args.seed)

    cfg_all = load_config()
    if not cfg_all.thresholds.frozen:
        raise SystemExit("thresholds not frozen — refusing to evaluate")

    units, extras = load_journals(args.out)
    rec = build_records(units, s_grid, cfgs)
    rec.save(args.out / "stage_m_records.json")
    write_curves_csv(args.out / "micromap_curves.csv", rec)

    # Schedule provenance (June-13 manual 1.2/15.8): the CFG_KERNEL token MUST
    # carry the schedule= suffix and a commitment grid may only run under the
    # exact (cfg, schedule) its kernel certification used. Extract g_kind per cfg
    # from the journals and assert it is consistent within each cfg.
    schedule_by_cfg = {}
    for cfg in cfgs:
        gks = {u.get("g_kind", "constant") for u in units if float(u["cfg"]) == cfg}
        if len(gks) > 1:
            raise SystemExit(f"cfg={cfg:g} units mix schedules {gks} — refusing to certify "
                             "a kernel without a single (cfg, schedule) provenance")
        schedule_by_cfg[cfg] = gks.pop() if gks else "constant"
    sched_h = schedule_by_cfg.get(HEADLINE_CFG, "constant")
    sched_d = schedule_by_cfg.get(DEPLOYED_CFG, "constant")

    cells, ref_probs = build_gate_a(args.out, units, rng, args.n_perm)
    thresholds = calibrate_from_internal_null(cells.get(HEADLINE_CFG, []))
    power_frac, cross_mmd, cross_p95 = power_positive_control(
        ref_probs.get(HEADLINE_CFG, {}), rng=rng, n_perm=args.n_perm)
    ks_p, _ = null_sanity(ref_probs.get(HEADLINE_CFG, {}), rng=rng, n_perm=args.n_perm)
    res_null = evaluate_internal_null(cells.get(HEADLINE_CFG, []), thresholds,
                                      power_reject_frac=power_frac,
                                      cross_clip_mmd_median=cross_mmd, null_ks_p=ks_p,
                                      cross_clip_mmd_p95=cross_p95, cfg=HEADLINE_CFG,
                                      schedule=sched_h)
    # Gate-A items (i)-(iii) at the headline cfg come from the SDE re-validation;
    # a non-OK token fails the headline kernel criterion regardless of (iv).
    sde1 = extras.get("sde_token_cfg1", "MISSING")
    if sde1 != "OK" and res_null.passed:
        sfx = f", schedule={sched_h}" if sched_h != "constant" else ""
        res_null = GateAResult(token=f"CFG_KERNEL_FAIL(cfg={HEADLINE_CFG:g}{sfx})",
                               passed=False, cfg=HEADLINE_CFG, schedule=sched_h,
                               guards=res_null.guards, per_s=res_null.per_s,
                               detail=f"SDE re-validation at headline cfg returned {sde1} "
                                      "(Gate-A items i-iii)")
    res_dep = evaluate_calibrated(cells.get(DEPLOYED_CFG, []), thresholds, cfg=DEPLOYED_CFG,
                                  schedule=sched_d)
    # cfg=4.5 Gate-A items (i)-(iii) also come from the SDE re-validation; a
    # non-OK token forces the deployed kernel FAIL (Codex T1 High finding).
    sde45 = extras.get("sde_token_cfg45", "MISSING")
    if sde45 != "OK" and res_dep.passed:
        sfx = f", schedule={sched_d}" if sched_d != "constant" else ""
        res_dep = GateAResult(token=f"CFG_KERNEL_FAIL(cfg={DEPLOYED_CFG:g}{sfx})",
                              passed=False, cfg=DEPLOYED_CFG, schedule=sched_d,
                              per_s=res_dep.per_s,
                              detail=f"SDE re-validation at deployed cfg returned {sde45} "
                                     "(Gate-A items i-iii)")
    write_gate_a_report(args.out, thresholds, res_null, res_dep)

    # Certified-kernel provenance ledger (manual 15.8). The cfg=4.5 entry is a
    # CANDIDATE: Stage-M cells only; ratification to "headline arm" requires the
    # full Phase-1 independent pool (NOT these pilot cells).
    certified = {
        "_doc": "Kernel certifications from Stage-M (manual 1.2/15.8). A commitment "
                "grid may only run under a (cfg, schedule) listed here as ok=true. The "
                "cfg=4.5 entry is candidate-only until full-Phase-1-pool Gate-A confirms it.",
        "headline": {"cfg": HEADLINE_CFG, "schedule": sched_h, "token": res_null.token,
                     "ok": bool(res_null.passed), "ratified": bool(res_null.passed),
                     "scope": "Stage-M cells (headline backbone)"},
        "deployed": {"cfg": DEPLOYED_CFG, "schedule": sched_d, "token": res_dep.token,
                     "ok": bool(res_dep.passed), "ratified": False,
                     "scope": "CANDIDATE — Stage-M pilot cells only; full-Phase-1-pool "
                              "Gate-A required to ratify the deployed-cfg headline arm"},
    }
    (args.out / "certified_kernels.json").write_text(json.dumps(certified, indent=2))

    report = evaluate_stage_m(rec, res_null, res_dep, extras["determinism"],
                              cfg_all.thresholds, class_axis_id="class", seed=args.seed,
                              expected_clips=args.expected_clips)
    if args.expected_clips != 16:
        report.tokens = [f"SMOKE_ONLY:{t}" for t in report.tokens]
    (args.out / "micromap_report.md").write_text(report.to_markdown())
    tokens_payload = {
        "tokens": report.tokens,
        "informativeness_warning": report.informativeness_warning,
        "failure_routing": report.failure_routing,
        "sde_tokens": {"cfg1.0": sde1, "cfg4.5": extras.get("sde_token_cfg45", "MISSING")},
        "deployed_cfg_token_status": "CANDIDATE (Stage-M cells; full-Phase-1-pool "
                                     "Gate-A required to ratify the cfg=4.5 headline arm)",
        "criteria": [{"name": c.name, "passed": c.passed, "detail": c.detail}
                     for c in report.criteria],
    }
    (args.out / "tokens.json").write_text(json.dumps(tokens_payload, indent=2))
    write_logging_audit(args.out, units)

    print(json.dumps(tokens_payload["tokens"], indent=2))
    # Every human-facing emission of the cfg=4.5 OK token carries the candidate
    # marking (Codex T1 Medium finding): it is NOT yet a ratified headline arm.
    if any(t.startswith("CFG_KERNEL_OK") and "cfg=4.5" in t for t in report.tokens):
        print("[stage-m-eval] NOTE: CFG_KERNEL_OK(cfg=4.5, ...) is a CANDIDATE re-entry "
              "token from Stage-M pilot cells — NOT ratified; a full-Phase-1-pool Gate-A "
              "must confirm it before the deployed-cfg commitment grid runs (manual 1.2/15.8).")
    # SMOKE_ONLY tokens never unlock anything (Codex pass-B finding): a plumbing
    # smoke exits 2 even when its criteria happen to pass.
    passed = "MICROMAP_PASS" in report.tokens
    if args.expected_clips != 16:
        print("[stage-m-eval] SMOKE plumbing run — tokens are SMOKE_ONLY, nothing unlocked")
        return 2
    print(f"[stage-m-eval] {'PASS — Stage 0 unlocked' if passed else 'FAIL — HALT for PI review'}")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
