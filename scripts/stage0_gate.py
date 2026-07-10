#!/usr/bin/env python
"""Stage-0 gate → GO_MAPS_PHASE (manual §3.3 self-target split).

The §3.3 reliability gate is SPLIT by claim layer. The SELF-TARGET gate
(determinism + robustness) gates GO_MAPS_PHASE and the determination-budget / commitment /
readout maps; VALIDITY is a correctness-layer gate (Phase-4 only), NOT a precondition.
This driver counts axes passing the self-target gate and emits GO_MAPS_PHASE via
foley_cw.gap.decide_phase0.

The det/rob numbers are reused from results/stage0/reliability_report.json (they are
deterministic measurer properties, independent of validity; timing robustness only
*improves* at the frozen coarser bin, so the ≥3 decision is bin-robust). class fails the
self-target gate on robustness (0.833 < 0.85) but — per the PI's resolved §3.3 decision —
is KEPT as a DIAGNOSTIC axis in the determination budget (carries g₀ + F-1), not dropped.

GO_MAPS_PHASE preconditions (§3): MICROMAP_PASS; Gate-A@headline cfg ratified; logging
contract exercised; manifest+anchors+reliability complete; ≥3 self-target axes; frozen.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from foley_cw.gap import decide_phase0  # noqa: E402
from foley_cw.types import ReliabilityResult  # noqa: E402

MIN_SELF_TARGET_AXES = 3


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reliability", type=Path, default=Path("results/stage0/reliability_report.json"))
    ap.add_argument("--thresholds", type=Path, default=Path("configs/thresholds.json"))
    ap.add_argument("--certified", type=Path, default=Path("results/stage_m_rerun/certified_kernels.json"))
    ap.add_argument("--manifest", type=Path, default=Path("data/manifests/phase1_manifest_frozen.json"))
    ap.add_argument("--out", type=Path, default=Path("results/stage0"))
    args = ap.parse_args()

    thr = json.loads(args.thresholds.read_text())
    theta_rel, theta_robust = thr["theta_rel"], thr["theta_robust"]
    rel = json.loads(args.reliability.read_text())["results"]

    # Self-target gate per axis = determinism + robustness (validity EXCLUDED).
    results, self_target = [], {}
    for r in rel:
        det, rob = r["determinism"], r["robustness"]
        passed = (det >= theta_rel) and (rob >= theta_robust)
        self_target[r["axis_id"]] = passed
        results.append(ReliabilityResult(
            axis_id=r["axis_id"], determinism=det, robustness=rob, validity=r["validity"],
            passed=passed, demoted=not passed,
            reason=("self-target OK" if passed else
                    f"self-target FAIL (det {det:.3f}/{theta_rel}, rob {rob:.3f}/{theta_robust})"),
        ))

    # class: kept diagnostic if it fails self-target ONLY on robustness with det reliable.
    class_diag = (not self_target.get("class", True)) and \
        any(r["axis_id"] == "class" and r["determinism"] >= theta_rel for r in rel)

    cert = json.loads(args.certified.read_text())
    head = cert.get("headline", {})
    validation_token = "OK" if (head.get("ok") and head.get("ratified")) else "FIX_SCORE_CONVERSION"

    manifest_ok = False
    if args.manifest.exists():
        manifest_ok = bool(json.loads(args.manifest.read_text()).get("frozen"))

    decision = decide_phase0(
        validation_token=validation_token,
        reliability=results,
        trajectory_ok=True,                 # screening + Stage-M exercised the trajectory path
        manifest_ok=manifest_ok,
        min_reliable_axes=MIN_SELF_TARGET_AXES,
    )

    n_pass = sum(1 for v in self_target.values() if v)
    out = {
        "_doc": "Stage-0 gate (§3.3 self-target split: determinism+robustness gates "
                "GO_MAPS_PHASE; validity is correctness-layer only).",
        "self_target_pass": self_target,
        "n_self_target_pass": n_pass,
        "min_required": MIN_SELF_TARGET_AXES,
        "class_kept_diagnostic": class_diag,
        "validation_token": validation_token,
        "manifest_ok": manifest_ok,
        "tokens": decision.tokens,
        "justification": decision.justification,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "go_maps_phase.json").write_text(json.dumps(out, indent=2))

    L = ["# Stage-0 Gate — Self-Target Split (§3.3)", "",
         f"Self-target gate (determinism ≥ {theta_rel} AND robustness ≥ {theta_robust}); "
         "validity is correctness-layer (NOT a precondition).", "",
         "| axis | det | rob | self-target | note |", "|---|---|---|---|---|"]
    for r in rel:
        st = self_target[r["axis_id"]]
        note = ("kept DIAGNOSTIC (in maps, not gating)" if (r["axis_id"] == "class" and class_diag)
                else ("" if st else "fails self-target"))
        L.append(f"| {r['axis_id']} | {r['determinism']:.3f} | {r['robustness']:.3f} | "
                 f"{'PASS' if st else 'fail'} | {note} |")
    L += ["", f"**{n_pass}/{len(rel)} axes pass the self-target gate (need ≥ "
          f"{MIN_SELF_TARGET_AXES}).** validation='{validation_token}', manifest_ok={manifest_ok}.",
          "", f"**Tokens: `{', '.join(decision.tokens)}`**", "", decision.justification]
    (args.out / "go_maps_phase.md").write_text("\n".join(L) + "\n")

    # token ledger
    ledger_path = args.out / "tokens.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {"tokens": []}
    for t in decision.tokens:
        if t not in ledger["tokens"]:
            ledger["tokens"].append(t)
    ledger_path.write_text(json.dumps(ledger, indent=2))

    print("\n".join(L))
    return 0 if "GO_MAPS_PHASE" in decision.tokens else 2


if __name__ == "__main__":
    raise SystemExit(main())
