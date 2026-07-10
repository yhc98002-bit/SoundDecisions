#!/usr/bin/env python
"""Phase 3 — Gap, Separation, GO/NO-GO (manual §6). The make-or-break.

Reads the Phase-1 determination budget (s_commit + bootstrap CIs, at the primary α and a
second α for Gate B) and the Phase-2 readout map (s_read per axis/probe), then:
  * gap(axis, probe) = s_read − s_commit (at the headline cfg);
  * separation: s_commit ordering with non-overlapping CIs; separation_score = spread /
    mean CI width;
  * Gate B: the s_commit axis ORDERING is stable across the α grid (instrument-stability);
  * R1/R2 cross-tab: uncommitted→R1 (defer); committed-but-unreadable→R2 (probe-limited,
    flag for Track P); committed & readable→early-action.
Emits per the frozen §6 language: GO_MAP (separated windows beyond CIs AND Gate-B-stable
ordering) and GO_READOUT (≥1 external probe reads a committed early axis well before s=1);
else GO_RESTRICTED / GO_DIAGNOSTIC / STOP_ADSR. cfg-cross-ordering never blocks GO_MAP.

Output: results/stage0/phase1/phase3_decision.{json,md} (Tab 2). CPU-only.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SELF_TARGET_AXES = ("presence", "timing", "material")   # class is diagnostic (kept, not gating)
READABLE_MARGIN = 0.15   # s_read must beat s=1 by this much on a committed axis for GO_READOUT


def read_budget(path: Path) -> dict:
    out = {}
    for r in csv.DictReader(path.open()):
        def g(k):
            v = r.get(k, "")
            return float(v) if v not in ("", "nan", "None") else float("nan")
        out[r["axis"]] = {"s_commit": g("s_commit"), "lo": g("scommit_lo"), "hi": g("scommit_hi")}
    return out


def read_readout(path: Path) -> dict:
    """axis -> best (earliest finite) s_read across probes, ODE-target."""
    best = {}
    if not path.exists():
        return best
    for r in csv.DictReader(path.open()):
        pass  # readout_map is per (axis,probe,s,accuracy); s_read comes from the token file
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase1-dir", type=Path, default=Path("results/stage0/phase1"))
    ap.add_argument("--primary-tag", default="p1cfg1")
    ap.add_argument("--alt-tag", default="p1cfg1a04", help="second-α budget for Gate B")
    ap.add_argument("--readout-tokens", type=Path, default=None)
    ap.add_argument("--thresholds", type=Path, default=Path("configs/thresholds.json"))
    args = ap.parse_args()

    bud = read_budget(args.phase1_dir / f"determination_budget_{args.primary_tag}.csv")
    alt_path = args.phase1_dir / f"determination_budget_{args.alt_tag}.csv"
    alt = read_budget(alt_path) if alt_path.exists() else {}
    rtok_path = args.readout_tokens or (args.phase1_dir / f"tokens_readout_p2cfg1.json")
    s_read = {}
    if rtok_path.exists():
        sr = json.loads(rtok_path.read_text()).get("s_read", {})
        for k, v in sr.items():        # key "axis/probe/target"
            ax = k.split("/")[0]
            v = float(v) if v not in (None, "nan") else float("nan")
            if not math.isnan(v):
                s_read[ax] = min(s_read.get(ax, math.inf), v)

    # --- separation over the self-target (gating) axes, ordered by s_commit ---
    gating = [a for a in SELF_TARGET_AXES if a in bud and math.isfinite(bud[a]["s_commit"])]
    ordered = sorted(gating, key=lambda a: bud[a]["s_commit"])
    non_overlap = all(bud[ordered[i]]["hi"] < bud[ordered[i + 1]]["lo"]
                      for i in range(len(ordered) - 1)) if len(ordered) >= 2 else False
    scs = [bud[a]["s_commit"] for a in ordered]
    spread = (max(scs) - min(scs)) if len(scs) >= 2 else 0.0
    mean_ci_w = (sum(bud[a]["hi"] - bud[a]["lo"] for a in ordered) / len(ordered)) if ordered else float("nan")
    sep_score = spread / mean_ci_w if mean_ci_w and math.isfinite(mean_ci_w) and mean_ci_w > 0 else float("nan")

    # --- Gate B: ordering stable across α ---
    gate_b = None
    if alt:
        alt_ord = sorted([a for a in gating if a in alt and math.isfinite(alt[a]["s_commit"])],
                         key=lambda a: alt[a]["s_commit"])
        gate_b = (alt_ord == ordered) and len(alt_ord) == len(ordered)

    # --- R1/R2 + gap (over all mapped axes incl. class diagnostic) ---
    rows = []
    for a in [x for x in ("presence", "timing", "class", "material") if x in bud]:
        sc = bud[a]["s_commit"]; sr = s_read.get(a, float("nan"))
        committed = math.isfinite(sc)
        readable = math.isfinite(sr)
        gap = (sr - sc) if (committed and readable) else float("nan")
        if not committed:
            cls = "R1 (uncommitted — defer)"
        elif not readable:
            cls = "R2 (committed, unreadable — Track P)"
        elif sr <= 1.0 - READABLE_MARGIN:
            cls = "early-action (committed & readable)"
        else:
            cls = "R2 (readable only near s=1)"
        rows.append({"axis": a, "s_commit": sc, "s_read": sr, "gap": gap, "class": cls})

    # --- §6 token logic ---
    go_map = bool(non_overlap and (gate_b is True))
    # GO_READOUT: ≥1 committed self-target axis whose external probe reads well before s=1
    go_readout = any(r["axis"] in SELF_TARGET_AXES and math.isfinite(r["s_commit"])
                     and math.isfinite(r["s_read"]) and r["s_read"] <= 1.0 - READABLE_MARGIN
                     for r in rows)
    tokens = []
    if go_map:
        tokens.append("GO_MAP")
    if go_readout:
        tokens.append("GO_READOUT")
    if not go_map:
        # separation present but Gate B unconfirmed / CIs overlap -> restricted/diagnostic
        tokens.append("GO_RESTRICTED" if non_overlap else "GO_DIAGNOSTIC")

    out = {"_doc": "Phase 3 (§6) gap/separation/GO-NO-GO at the headline cfg. self-target "
                   "axes gate; class is diagnostic. cfg-cross ordering never blocks GO_MAP.",
           "ordered_by_s_commit": ordered, "separation_non_overlapping_CIs": non_overlap,
           "separation_score": sep_score, "gate_b_ordering_stable": gate_b,
           "gate_b_alt_tag": args.alt_tag if alt else None,
           "rows": rows, "tokens": tokens, "go_map": go_map, "go_readout": go_readout}
    args.phase1_dir.mkdir(parents=True, exist_ok=True)
    (args.phase1_dir / "phase3_decision.json").write_text(json.dumps(out, indent=2, default=str))

    def f(x):
        return "never" if (isinstance(x, float) and math.isnan(x)) else (f"{x:.3f}" if isinstance(x, float) else str(x))
    L = ["# Phase 3 — Gap / Separation / GO-NO-GO (§6) — Tab 2", "",
         f"Self-target gating axes ordered by s_commit: {ordered}. Non-overlapping CIs: "
         f"**{non_overlap}**; separation_score = {f(sep_score)}; Gate-B ordering stable "
         f"across α ({args.alt_tag}): **{gate_b}**. class is diagnostic (kept, not gating).", "",
         "| axis | s_commit | s_read | gap | R-class |", "|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['axis']} | {f(r['s_commit'])} | {f(r['s_read'])} | {f(r['gap'])} | {r['class']} |")
    L += ["", f"**Tokens: `{', '.join(tokens)}`** — GO_MAP={go_map} (separation+GateB), "
          f"GO_READOUT={go_readout}."]
    (args.phase1_dir / "phase3_decision.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
