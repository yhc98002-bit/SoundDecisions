#!/usr/bin/env python
"""Phase-1 aggregate (manual §4) — determination budget + Fig-1 taxonomy + A(axis,s) surface.

Reads results/stage0/phase1/commitment_map_<tag>.csv (the per-clip A_fork/A_independent/
commit_gain produced by phase1_commitment.py --aggregate) and builds, via
foley_cw.determination (bootstrap by video):
  * determination_budget_<tag>.csv  — per axis: conditioning / seed / trajectory / residual
    shares (mean + 95% CI), s_commit, taxonomy counts (Fig 1).
  * taxonomy_report_<tag>.md        — the Fig-1 narrative.
  * commitment_surface_<tag>.csv    — per (axis, s): mean A_fork, mean commit_gain (the
    A(axis, s) curve at the primary α; the full A(axis,s,α) surface is assembled once the
    secondary-α runs land).
Emits COMMITMENT_MAP_DONE when every manifest clip is present. CPU-only.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from foley_cw.config import load_config  # noqa: E402
from foley_cw.determination import build_determination_budget  # noqa: E402
from foley_cw.types import AxisKind  # noqa: E402


def _f(x) -> str:
    return "nan" if (x is None or (isinstance(x, float) and math.isnan(x))) else f"{x:.3f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cmap", type=Path, default=None, help="commitment_map_<tag>.csv")
    ap.add_argument("--tag", default="p1cfg1")
    ap.add_argument("--phase1-dir", type=Path, default=Path("results/stage0/phase1"))
    ap.add_argument("--manifest", type=Path, default=Path("data/manifests/phase1_manifest_frozen.json"))
    ap.add_argument("--thresholds", type=Path, default=Path("configs/thresholds.json"))
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()

    cmap = args.cmap or (args.phase1_dir / f"commitment_map_{args.tag}.csv")
    theta_commit = json.loads(args.thresholds.read_text())["theta_commit"]
    embedding_axes = {a.id for a in load_config().axes if a.kind is AxisKind.EMBEDDING}

    rows = list(csv.DictReader(cmap.open()))
    per_clip: dict[str, dict[str, dict[float, float]]] = defaultdict(lambda: defaultdict(dict))
    a_ind: dict[str, dict[str, float]] = defaultdict(dict)
    surf: dict[tuple, list] = defaultdict(lambda: {"a_fork": [], "commit": []})
    clips_seen = set()
    for r in rows:
        ax, clip, s = r["axis_id"], r["clip"], float(r["s"])
        clips_seen.add(clip)
        af = float(r["a_fork"]) if r["a_fork"] not in ("", "None", None) else float("nan")
        ai = float(r["a_independent"]) if r["a_independent"] not in ("", "None", None) else float("nan")
        cg = float(r["commit_gain"]) if r["commit_gain"] not in ("", "None", None) else float("nan")
        per_clip[ax][clip][s] = af
        a_ind[ax][clip] = ai
        surf[(ax, s)]["a_fork"].append(af)
        surf[(ax, s)]["commit"].append(cg)

    budget = build_determination_budget(per_clip, a_ind, embedding_axes, theta_commit,
                                        n_boot=args.n_boot)

    args.phase1_dir.mkdir(parents=True, exist_ok=True)
    # determination budget csv
    with (args.phase1_dir / f"determination_budget_{args.tag}.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["axis", "n_clips", "is_embedding", "conditioning", "cond_lo", "cond_hi",
                    "seed", "seed_lo", "seed_hi", "trajectory", "traj_lo", "traj_hi",
                    "residual", "s_commit", "scommit_lo", "scommit_hi"])
        for ax, d in budget.items():
            b = d["budget"]
            def trip(k):
                return [b[k]["mean"], b[k]["ci95"][0], b[k]["ci95"][1]]
            sc = d["s_commit"]
            w.writerow([ax, d["n_clips"], d["is_embedding"], *trip("conditioning_share"),
                        *trip("seed_share"), *trip("trajectory_share"),
                        b["residual"]["mean"], sc["mean"], sc["ci95"][0], sc["ci95"][1]])

    # A(axis,s) surface csv
    with (args.phase1_dir / f"commitment_surface_{args.tag}.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["axis", "s", "mean_a_fork", "mean_commit_gain", "n"])
        for (ax, s) in sorted(surf):
            af = [x for x in surf[(ax, s)]["a_fork"] if np.isfinite(x)]
            cg = [x for x in surf[(ax, s)]["commit"] if np.isfinite(x)]
            w.writerow([ax, s, _f(np.mean(af) if af else float("nan")),
                        _f(np.mean(cg) if cg else float("nan")), len(af)])

    # taxonomy report
    L = [f"# Phase-1 Determination Budget + Taxonomy ({args.tag}) — Fig 1", "",
         f"θ_commit = {theta_commit}; bootstrap by video. Shares clipped at 0; s_min = seed "
         "floor. class is DIAGNOSTIC (kept).", "",
         "| axis | n | conditioning | seed | trajectory | residual | s_commit |",
         "|---|---|---|---|---|---|---|"]
    for ax, d in budget.items():
        b = d["budget"]
        L.append(f"| {ax}{' (emb)' if d['is_embedding'] else ''} | {d['n_clips']} | "
                 f"{_f(b['conditioning_share']['mean'])} | {_f(b['seed_share']['mean'])} | "
                 f"{_f(b['trajectory_share']['mean'])} | {_f(b['residual']['mean'])} | "
                 f"{_f(d['s_commit']['mean'])} |")
    L += ["", "## Taxonomy (clip counts per axis)",
          "| axis | video-det | seed-det | traj-early | traj-mid | traj-late | never |",
          "|---|---|---|---|---|---|---|"]
    for ax, d in budget.items():
        t = d["taxonomy"]
        L.append(f"| {ax} | {t['video_determined']} | {t['seed_determined']} | "
                 f"{t['trajectory_early']} | {t['trajectory_mid']} | {t['trajectory_late']} | "
                 f"{t['never_commits']} |")
    (args.phase1_dir / f"taxonomy_report_{args.tag}.md").write_text("\n".join(L) + "\n")

    # completion token
    manifest = json.loads(args.manifest.read_text())
    want = set(str(c) for c in manifest["clips"]["single_event"])
    done = clips_seen & want
    complete = want.issubset(clips_seen)
    tokens = ["COMMITMENT_MAP_DONE"] if complete else []
    ledger = args.phase1_dir / f"tokens_{args.tag}.json"
    ledger.write_text(json.dumps({"tokens": tokens, "n_clips": len(done),
                                  "n_expected": len(want), "complete": complete}, indent=2))
    print("\n".join(L))
    print(f"\n[aggregate] {len(done)}/{len(want)} single-event clips; "
          f"tokens={tokens or '(incomplete)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
