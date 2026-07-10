#!/usr/bin/env python
"""Figure assembly (manual §11 deliverable map). matplotlib Agg (no display).

Populates figures as each phase lands; missing inputs are skipped with a note (the
consolidated report states which figures are populated). Currently wired:
  Fig 1  — three-share determination budget (stacked bars per axis × cfg)  [Phase 1]
  Fig 1b — share migration cfg=1.0 ↔ 4.5 (when both arms exist)            [Phase 1/§8.3]
  Fig 2  — commitment surface A(axis, s) at the primary α                  [Phase 1]
Later figures (3 gap, 4 internal, 5 cond-swap, 6 policy) are added as their inputs land.

Usage: python scripts/make_figures.py --tags p1cfg1 [p1cfg4.5]
Output: results/figures/*.png
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SHARE_COLORS = {"conditioning": "#4c72b0", "seed": "#dd8452", "trajectory": "#55a868",
                "residual": "#cccccc"}


def _read_budget(path: Path) -> dict:
    out = {}
    if not path.exists():
        return out
    for r in csv.DictReader(path.open()):
        out[r["axis"]] = {k: (float(v) if v not in ("", "nan", "None") else float("nan"))
                          for k, v in r.items() if k not in ("axis", "is_embedding")}
    return out


def fig1_budget(tags: list[str], phase1_dir: Path, out: Path) -> bool:
    budgets = {t: _read_budget(phase1_dir / f"determination_budget_{t}.csv") for t in tags}
    budgets = {t: b for t, b in budgets.items() if b}
    if not budgets:
        return False
    axes_order = ["presence", "timing", "class", "material"]
    ntags = len(budgets)
    fig, axarr = plt.subplots(1, ntags, figsize=(5.5 * ntags, 4.2), squeeze=False)
    for ti, (tag, b) in enumerate(budgets.items()):
        ax = axarr[0][ti]
        axis_ids = [a for a in axes_order if a in b]
        x = np.arange(len(axis_ids))
        bottom = np.zeros(len(axis_ids))
        for share in ("conditioning", "seed", "trajectory", "residual"):
            vals = np.array([max(0.0, b[a].get(share, 0.0)) for a in axis_ids])
            ax.bar(x, vals, bottom=bottom, color=SHARE_COLORS[share],
                   label=share if ti == 0 else None)
            bottom += vals
        ax.set_xticks(x); ax.set_xticklabels([a[:4] for a in axis_ids])
        ax.set_ylim(0, 1.05); ax.set_title(f"determination budget — {tag}")
        ax.set_ylabel("share of the decision")
    fig.legend(loc="upper center", ncol=4, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "fig1_determination_budget.png", dpi=140)
    plt.close(fig)
    return True


def fig1b_migration(tags: list[str], phase1_dir: Path, out: Path) -> bool:
    budgets = {t: _read_budget(phase1_dir / f"determination_budget_{t}.csv") for t in tags}
    budgets = {t: b for t, b in budgets.items() if b}
    if len(budgets) < 2:
        return False
    axes_order = ["presence", "timing", "class", "material"]
    tagl = list(budgets)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    common = [a for a in axes_order if all(a in budgets[t] for t in tagl)]
    x = np.arange(len(common)); w = 0.35
    for i, share in enumerate(("seed", "trajectory")):
        for j, t in enumerate(tagl):
            vals = [max(0.0, budgets[t][a].get(share, 0.0)) for a in common]
            ax.bar(x + (j - 0.5) * w, vals, w, label=f"{share}/{t}",
                   alpha=0.7 if share == "seed" else 1.0)
    ax.set_xticks(x); ax.set_xticklabels(common)
    ax.set_title("share migration: seed vs trajectory across cfg")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(out / "fig1b_share_migration.png", dpi=140)
    plt.close(fig)
    return True


def fig2_surface(tags: list[str], phase1_dir: Path, out: Path) -> bool:
    drawn = False
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for tag in tags:
        path = phase1_dir / f"commitment_surface_{tag}.csv"
        if not path.exists():
            continue
        by_axis: dict[str, list] = {}
        for r in csv.DictReader(path.open()):
            by_axis.setdefault(r["axis"], []).append((float(r["s"]), r["mean_commit_gain"]))
        for axis, pts in by_axis.items():
            pts = sorted(pts)
            s = [p[0] for p in pts]
            cg = [float(p[1]) if p[1] not in ("nan", "", "None") else np.nan for p in pts]
            ax.plot(s, cg, marker="o", label=f"{axis} ({tag})")
            drawn = True
    if not drawn:
        plt.close(fig); return False
    ax.set_xlabel("progress s"); ax.set_ylabel("commit_gain"); ax.set_ylim(-0.02, 1.05)
    ax.set_title("commitment surface A(axis, s) at primary α"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "fig2_commitment_surface.png", dpi=140)
    plt.close(fig)
    return True


def fig5_condswap(out: Path) -> bool:
    import csv as _csv
    p = Path("results/stage0/stage_r/cond_swap_map_cswap.csv")
    if not p.exists():
        return False
    by = {}
    for r in _csv.DictReader(p.open()):
        ax = r.get("axis_id", r.get("axis"))
        by.setdefault(ax, []).append((float(r["s"]), float(r.get("follow_rate", "nan") or "nan"),
                                      float(r.get("retention_rate", "nan") or "nan")))
    if not by:
        return False
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for axis, pts in by.items():
        pts = sorted(pts)
        ax.plot([x[0] for x in pts], [x[1] for x in pts], marker="o", label=f"{axis} follow")
    ax.axhline(0.5, ls="--", color="grey", lw=1)
    ax.set_xlabel("swap progress s"); ax.set_ylabel("follow-rate (tracks donor video)")
    ax.set_ylim(-0.02, 1.05); ax.set_title("Fig 5 — condition-swap follow-rate (class stays low = not video-driven)")
    ax.legend(fontsize=8)
    out.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out / "fig5_condition_swap.png", dpi=140); plt.close(fig)
    return True


def fig6_pareto(out: Path) -> bool:
    import csv as _csv
    p = Path("results/stage0/phase1/policy_pareto.csv")
    if not p.exists():
        return False
    rows = list(_csv.DictReader(p.open()))
    if not rows:
        return False
    fig, ax = plt.subplots(figsize=(7, 4.8))
    for r in rows:
        nfe = float(r.get("total_nfe", r.get("nfe", "nan")))
        corr = float(r.get("final_correctness", r.get("final_corr", "nan")))
        name = r.get("policy", "?")
        mk = "*" if "oracle" in name else "o"
        ax.scatter(nfe, corr, s=120 if "oracle" in name else 60, marker=mk,
                   label=name, zorder=3 if "oracle" in name else 2)
        ax.annotate(name, (nfe, corr), fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("total generator-NFE"); ax.set_ylabel("final correctness (oracle-proxy)")
    ax.set_title("Fig 6 — policy compute–quality Pareto (oracle gating ≫ matched-compute scalars)")
    fig.tight_layout(); fig.savefig(out / "fig6_policy_pareto.png", dpi=140); plt.close(fig)
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["p1cfg1"])
    ap.add_argument("--phase1-dir", type=Path, default=Path("results/stage0/phase1"))
    ap.add_argument("--out", type=Path, default=Path("results/figures"))
    args = ap.parse_args()
    status = {
        "fig1_determination_budget": fig1_budget(args.tags, args.phase1_dir, args.out),
        "fig1b_share_migration": fig1b_migration(args.tags, args.phase1_dir, args.out),
        "fig2_commitment_surface": fig2_surface(args.tags, args.phase1_dir, args.out),
        "fig5_condition_swap": fig5_condswap(args.out),
        "fig6_policy_pareto": fig6_pareto(args.out),
    }
    for k, v in status.items():
        print(f"  {k}: {'WROTE' if v else 'skipped (no input)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
