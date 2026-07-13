#!/usr/bin/env python
"""Phase 4 — OFFLINE policy Pareto (manual §9, Fig. 6). CPU-only, NO new generation.

REPLAY gating on the cached deployed-cfg (cfg=4.5) Phase-1 independents (measurements.jsonl
role 'p1cfg45_independent'). Builds one ClipPool per clip — its up-to-16 cached candidates,
their per-axis final self-target labels, the ORACLE-PROXY per-axis correctness (agreement
with the per-clip majority self-target; see experiment/preregistered/policy_preregistration.md),
and a deterministic cache-derived scalar reward (cosine of the candidate's final-grid pooled
feature to the per-clip mean independent feature — a self-consistency reward, distinct from the
label-majority proxy). Runs every preregistered policy with matched generator-NFE AND matched
scoring-call accounting and writes:

  results/stage0/phase1/policy_pareto.csv   one row per policy (metrics + two Pareto axes)
  results/stage0/phase1/policy_report.md    the §9 report + the proxy-correctness caveat

Oracle gate windows come from s_commit in determination_budget_p1cfg45.csv. The eval clips are
the frozen split (split_60_40_by_clip.eval). Nothing here writes finals, features, or touches
frozen configs / certified_kernels.json. Deterministic given --seed.

  python scripts/phase4_policy.py            # full offline run, writes csv + report
  python scripts/phase4_policy.py --split all   # use all clips, not just the eval split
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from foley_cw import policy_offline as P  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MEAS = ROOT / "results/stage0/measurements/measurements.jsonl"
FEATS = ROOT / "results/stage0/features"
BUDGET_CSV = ROOT / "results/stage0/phase1/determination_budget_p1cfg45.csv"
MANIFEST = ROOT / "data/manifests/phase1_manifest_frozen.json"
OUT_CSV = ROOT / "results/stage0/phase1/policy_pareto.csv"
OUT_MD = ROOT / "results/stage0/phase1/policy_report.md"

ROLE = "p1cfg45_independent"
AXES = P.DEFAULT_AXES
PHASE1_S_GRID = (0.05, 0.15, 0.25, 0.35, 0.45, 0.60, 0.75, 0.90)
FINAL_GRID_S = 0.90               # latest cached grid point; the final-window feature source
NUM_STEPS = P.NFE_FULL_DEFAULT    # deployed cfg=4.5 integration budget


def headroom_supported(
    gated_final: float,
    gated_nfe: float,
    scbon_final: float,
    scbon_nfe: float,
) -> bool:
    """Return the corrected offline headroom screen."""
    return bool(
        gated_final >= scbon_final + 0.01
        and gated_nfe <= scbon_nfe * 1.02
    )


# ---------------------------------------------------------------------------
# Cache loading
# ---------------------------------------------------------------------------
def load_pool_records() -> dict[str, dict[int, dict[str, object]]]:
    """clip -> j -> axis_id -> (label or embedding array). One pass over the jsonl."""
    out: dict[str, dict[int, dict[str, object]]] = collections.defaultdict(
        lambda: collections.defaultdict(dict)
    )
    with open(MEAS, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or '"p1cfg45_independent"' not in line:
                continue
            r = json.loads(line)
            ex = r.get("extra", {})
            if ex.get("role") != ROLE:
                continue
            clip = str(ex["clip"]); j = int(ex["j"]); ax = r["axis_id"]; tgt = r["target"]
            if tgt.get("embedding") is not None:
                out[clip][j][ax] = np.asarray(tgt["embedding"], dtype=float)
            else:
                out[clip][j][ax] = tgt.get("label")
    return out


def load_scommit() -> dict[str, float]:
    sc: dict[str, float] = {}
    with open(BUDGET_CSV, "r", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                sc[row["axis"]] = float(row["s_commit"])
            except (KeyError, ValueError):
                pass
    return sc


def load_final_feature(clip: str, j: int) -> np.ndarray | None:
    p = FEATS / f"{clip}__p1cfg45_ind{j}__s{FINAL_GRID_S:.2f}.npz"
    if not p.exists():
        return None
    return np.load(p)["pooled"].astype(np.float64).ravel()


# ---------------------------------------------------------------------------
# Oracle proxy + cache-derived scalar reward
# ---------------------------------------------------------------------------
def _majority_label(labels: list) -> object:
    """Plurality label, ties broken by sorted string order (deterministic)."""
    cnt = collections.Counter(labels)
    best = max(cnt.values())
    winners = sorted((k for k, v in cnt.items() if v == best), key=str)
    return winners[0]


def build_pool(clip: str, per_j: dict[int, dict[str, object]]) -> P.ClipPool | None:
    """Build a ClipPool: proxy labels (majority agreement) + a cache-derived scalar reward."""
    js = sorted(per_j.keys())
    if len(js) < 2:
        return None
    n = len(js)

    # ---- per-axis proxy-correctness = agreement with the per-clip majority ----
    labels: dict[str, np.ndarray] = {}
    axis_score: dict[str, np.ndarray] = {}
    for ax in AXES:
        if ax not in per_j[js[0]]:
            continue
        vals = [per_j[j].get(ax) for j in js]
        if isinstance(vals[0], np.ndarray):
            # embedding axis (material): consensus = closer-than-median to the mean embedding
            M = np.stack(vals, axis=0)
            mean = M.mean(axis=0, keepdims=True)
            mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
            cn = mean / (np.linalg.norm(mean) + 1e-12)
            cos = (mn @ cn.T).ravel()
            thr = float(np.median(cos))
            ok = cos >= thr
            labels[ax] = ok
            axis_score[ax] = (cos - cos.min()) / (np.ptp(cos) + 1e-12)
        else:
            maj = _majority_label(vals)
            ok = np.array([v == maj for v in vals], dtype=bool)
            labels[ax] = ok
            axis_score[ax] = ok.astype(float)
    if not labels:
        return None

    # ---- scalar reward: cosine of the final-grid pooled feature to the clip mean ----
    feats = [load_final_feature(clip, j) for j in js]
    if all(f is not None for f in feats):
        F = np.stack(feats, axis=0)
        mean = F.mean(axis=0, keepdims=True)
        fn = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-12)
        cn = mean / (np.linalg.norm(mean) + 1e-12)
        final_score = (fn @ cn.T).ravel()
    else:
        # feature-free fallback: number of axes on which the candidate matches consensus
        final_score = np.sum([labels[a].astype(float) for a in labels], axis=0)

    return P.ClipPool(clip=clip, labels=labels, final_score=final_score, axis_score=axis_score)


def eval_clips(split: str) -> set[str] | None:
    if split == "all":
        return None
    m = json.loads(MANIFEST.read_text())
    return set(str(c) for c in m["split_60_40_by_clip"]["eval"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Phase-4 offline policy Pareto (manual §9, Fig. 6).")
    ap.add_argument("--split", choices=["eval", "all"], default="eval",
                    help="clip set: frozen eval split (default) or all 200 clips.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--diffrs-tau", type=float, default=None,
                    help="DiffRS rejection threshold; default = global median final_score.")
    ap.add_argument("--smc-temp", type=float, default=0.1)
    ap.add_argument("--random-prune-frac", type=float, default=0.5)
    ap.add_argument("--out-csv", type=Path, default=OUT_CSV)
    ap.add_argument("--out-md", type=Path, default=OUT_MD)
    args = ap.parse_args()

    keep = eval_clips(args.split)
    records = load_pool_records()
    s_commit = load_scommit()
    gates = P.gates_from_scommit(s_commit, PHASE1_S_GRID, AXES)

    pools: list[P.ClipPool] = []
    for clip in sorted(records):
        if keep is not None and clip not in keep:
            continue
        pool = build_pool(clip, records[clip])
        if pool is not None:
            pools.append(pool)
    if not pools:
        raise SystemExit("phase4_policy: no eligible pools found in cache.")

    # default DiffRS threshold = global median final_score (a frozen-by-data choice).
    if args.diffrs_tau is None:
        all_scores = np.concatenate([p.final_score for p in pools])
        diffrs_tau = float(np.median(all_scores))
    else:
        diffrs_tau = args.diffrs_tau

    metrics = P.run_all_policies(
        pools, gates=gates, axes=AXES, num_steps=NUM_STEPS, seed=args.seed,
        diffrs_tau=diffrs_tau, smc_temp=args.smc_temp,
        random_prune_frac=args.random_prune_frac,
    )

    # ---- write policy_pareto.csv ----
    cols = ["policy", "n_clips", "final_correctness"] + [f"correct_{a}" for a in AXES] + [
        "completed_candidates", "total_nfe", "scoring_calls",
        "winner_retention", "false_prune_rate", "regret",
    ]
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for policy in P.POLICIES:
            row = {c: metrics[policy].get(c, "") for c in cols}
            w.writerow(row)

    # ---- write policy_report.md ----
    gated = metrics["oracle_axis_gated"]; scbon = metrics["same_compute_bon"]
    full = metrics["full_bon"]; rnd = metrics["random_prune"]
    lines = [
        "# Phase-4 OFFLINE policy Pareto (manual §9, Fig. 6)",
        "",
        f"Split: **{args.split}** ({len(pools)} clips). Seed {args.seed}. "
        f"Deployed cfg=4.5, num_steps={NUM_STEPS}. DiffRS τ={diffrs_tau:.4f}, "
        f"SMC T={args.smc_temp}, random-prune frac={args.random_prune_frac}.",
        "",
        "**PROXY-CORRECTNESS CAVEAT.** `correctness` here is the ORACLE PROXY = agreement with "
        "the per-clip MAJORITY self-target across independents (preregistration). It is a "
        "self-consistency proxy, NOT human/MLLM correctness-vs-video. No correctness claim "
        "follows from these numbers alone; the offline pass is a headroom screen + method "
        "illustration (manual §9, §1.7).",
        "",
        "Oracle gate windows from s_commit (determination_budget_p1cfg45.csv):",
    ]
    for g in gates:
        lines.append(f"- window s={g.s:.2f}: prune on {', '.join(g.axes)}")
    lines += [
        "",
        "| policy | final_corr | NFE | scoring | winner_ret | false_prune | regret |",
        "|---|---|---|---|---|---|---|",
    ]
    for policy in P.POLICIES:
        m = metrics[policy]
        lines.append(
            f"| {policy} | {m['final_correctness']:.3f} | {m['total_nfe']} | "
            f"{m['scoring_calls']} | {m['winner_retention']:.3f} | "
            f"{m['false_prune_rate']:.3f} | {m['regret']:.3f} |"
        )
    lines += [
        "",
        "**Two-axis Pareto points** `(generator-NFE, final proxy-correctness)` and "
        "`(scoring-calls, final proxy-correctness)`:",
        "",
        "| policy | (NFE, final_corr) | (scoring, final_corr) |",
        "|---|---|---|",
    ]
    for policy in P.POLICIES:
        m = metrics[policy]
        lines.append(
            f"| {policy} | ({m['total_nfe']}, {m['final_correctness']:.3f}) | "
            f"({m['scoring_calls']}, {m['final_correctness']:.3f}) |"
        )
    headroom = headroom_supported(
        gated["final_correctness"], gated["total_nfe"],
        scbon["final_correctness"], scbon["total_nfe"],
    )
    lines += [
        "",
        "## Matched-compute read (oracle_axis_gated vs same_compute_bon)",
        f"- same_compute_bon NFE budget is matched per clip to oracle_axis_gated.",
        f"- oracle_axis_gated: final_corr={gated['final_correctness']:.3f}, "
        f"NFE={gated['total_nfe']}, false_prune={gated['false_prune_rate']:.3f}.",
        f"- same_compute_bon: final_corr={scbon['final_correctness']:.3f}, "
        f"NFE={scbon['total_nfe']}.",
        f"- full_bon ceiling: final_corr={full['final_correctness']:.3f}, NFE={full['total_nfe']}.",
        f"- random_prune control: final_corr={rnd['final_correctness']:.3f}, "
        f"false_prune={rnd['false_prune_rate']:.3f}.",
        "",
        f"**Offline headroom (proxy): {'YES' if headroom else 'NO'}** — "
        f"gated_final={gated['final_correctness']:.3f}, "
        f"same_compute_bon_final={scbon['final_correctness']:.3f}, "
        f"gated_nfe={gated['total_nfe']}, "
        f"same_compute_bon_nfe={scbon['total_nfe']}; "
        "criterion: gated_final >= same_compute_bon_final + 0.01 and "
        "gated_nfe <= same_compute_bon_nfe * 1.02. "
        "Token routing (§9) is a human STOP decision; this report does not emit GO_POLICY / "
        "GO_RESTRICTED / DIAGNOSTIC_ONLY on its own.",
    ]
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"wrote {args.out_csv} and {args.out_md} "
          f"({len(pools)} clips, {len(P.POLICIES)} policies)")
    for policy in P.POLICIES:
        m = metrics[policy]
        print(f"  {policy:18s} final_corr={m['final_correctness']:.3f} "
              f"NFE={m['total_nfe']:6d} scoring={m['scoring_calls']:5d} "
              f"regret={m['regret']:.3f}")


if __name__ == "__main__":
    main()
