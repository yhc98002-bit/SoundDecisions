#!/usr/bin/env python
"""Cached class-diversity and abstention lens. CPU-only; no generation or decision token.

The default command writes a descriptive WP-A2 artifact with clip-bootstrap confidence
intervals. The withdrawn Arc-3 mechanism narrative is retained for audit reproduction only and
can be emitted explicitly with ``--legacy-arc3`` to its historical results path.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BUDGET_CFG1 = ROOT / "results/stage0/phase1/determination_budget_p1cfg1.csv"
BUDGET_CFG45 = ROOT / "results/stage0/phase1/determination_budget_p1cfg45.csv"
CSWAP_MAP = ROOT / "results/stage0/stage_r/cond_swap_map_cswap.csv"
CSWAP_SUMMARY = ROOT / "results/stage0/stage_r/cond_swap_summary_cswap.json"
DIAL_GLOB = "results/stage0/gate_a/dial_noise__dial_cfg{C}__*.npz"
OUT_DIR = ROOT / "results/stage0/arc3"
OUT_JSON = OUT_DIR / "two_budgets.json"
OUT_MD = OUT_DIR / "two_budgets.md"
ARC4_OUT_DIR = ROOT / "results/arc4_wpA"
ARC4_OUT_JSON = ARC4_OUT_DIR / "entropy_lens_v2.json"
ARC4_OUT_MD = ARC4_OUT_DIR / "entropy_lens_v2.md"
ARC4_WPA2_OUT_DIR = ROOT / "results/arc4_wpA2"
ARC4_WPA2_OUT_JSON = ARC4_WPA2_OUT_DIR / "entropy_lens_v3.json"
ARC4_WPA2_OUT_MD = ARC4_WPA2_OUT_DIR / "entropy_lens_v3.md"

AXES = ("presence", "timing", "class", "material")
CFG_LIST = ("1", "1.5", "2", "2.5", "3", "4.5")  # cfg-dial grid for the entropy lens
SHARE_KEYS = ("conditioning", "seed", "trajectory", "residual")
ABSTAIN_LABEL = "abstain"
EXPECTED_DIAL_CLIPS = 24


# ---------------------------------------------------------------------------
# (i) Observational determination budget
# ---------------------------------------------------------------------------
def load_budget(path: Path) -> dict[str, dict[str, float]]:
    """axis -> {conditioning, seed, trajectory, residual, s_commit, + CI bounds} (floats)."""
    out: dict[str, dict[str, float]] = {}
    with open(path, "r", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            ax = r["axis"]
            rec: dict[str, float] = {}
            for k, v in r.items():
                if k == "axis":
                    continue
                try:
                    rec[k] = float(v)
                except (TypeError, ValueError):
                    rec[k] = v  # is_embedding bool-ish strings etc.
            out[ax] = rec
    return out


# ---------------------------------------------------------------------------
# (ii) Causal conditioning responsiveness (cond-swap)
# ---------------------------------------------------------------------------
def load_cswap_map(path: Path) -> dict[str, dict[float, dict[str, float]]]:
    """axis -> s -> {follow_rate, retention_rate, neither_rate, n}."""
    out: dict[str, dict[float, dict[str, float]]] = {}
    with open(path, "r", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            ax = r["axis_id"]
            s = float(r["s"])
            out.setdefault(ax, {})[s] = {
                "follow_rate": float(r["follow_rate"]),
                "retention_rate": float(r["retention_rate"]),
                "neither_rate": float(r["neither_rate"]),
                "n": int(r["n"]),
                "kind": r["kind"],
            }
    return out


def load_cswap_summary(path: Path) -> dict:
    txt = path.read_text(encoding="utf-8")
    # the file uses bare NaN (json5-ish); make it strict-JSON loadable
    txt = txt.replace(": NaN", ": null")
    return json.loads(txt)


# ---------------------------------------------------------------------------
# (iii) Entropy lens — distinct-class-count vs cfg, recomputed from caches
# ---------------------------------------------------------------------------
def distinct_class_count_by_cfg() -> dict[str, dict[str, float]]:
    """cfg -> {mean_distinct, n_clips, per_clip_min, per_clip_max}.

    Per clip, count the distinct final-class labels among its 16 independents in the cfg-dial
    cache; report the mean across clips. Drop in this count as cfg rises = MODE COLLAPSE.
    """
    out: dict[str, dict[str, float]] = {}
    for C in CFG_LIST:
        pat = str(ROOT / DIAL_GLOB.format(C=C))
        files = sorted(glob.glob(pat))
        dcs: list[int] = []
        for fp in files:
            d = np.load(fp, allow_pickle=True)
            labels = np.asarray(d["labels"]).tolist()
            dcs.append(len(set(labels)))
        if not dcs:
            out[C] = {"mean_distinct": float("nan"), "n_clips": 0,
                      "per_clip_min": float("nan"), "per_clip_max": float("nan")}
            continue
        out[C] = {
            "mean_distinct": float(np.mean(dcs)),
            "n_clips": int(len(dcs)),
            "per_clip_min": int(np.min(dcs)),
            "per_clip_max": int(np.max(dcs)),
        }
    return out


def binomial_wilson_ci(n_success: int, n_total: int, z: float = 1.959963984540054) \
        -> tuple[float, float]:
    """Two-sided Wilson score interval for a binomial proportion (95% by default)."""
    if n_total <= 0:
        return float("nan"), float("nan")
    if not 0 <= n_success <= n_total:
        raise ValueError("n_success must lie in [0, n_total]")
    p = n_success / n_total
    z2 = z * z
    denom = 1.0 + z2 / n_total
    center = (p + z2 / (2.0 * n_total)) / denom
    half_width = z * math.sqrt(
        p * (1.0 - p) / n_total + z2 / (4.0 * n_total * n_total)
    ) / denom
    return max(0.0, center - half_width), min(1.0, center + half_width)


def build_entropy_lens_v2() -> dict:
    """Recompute class diversity both with and without abstentions from dial caches."""
    by_cfg: dict[str, dict[str, float | int]] = {}
    for C in CFG_LIST:
        files = sorted(glob.glob(str(ROOT / DIAL_GLOB.format(C=C))))
        distinct_including: list[int] = []
        distinct_excluding: list[int] = []
        n_abstain = 0
        n_labels = 0
        for fp in files:
            labels = np.asarray(np.load(fp, allow_pickle=True)["labels"]).tolist()
            distinct_including.append(len(set(labels)))
            confident = [label for label in labels if label != ABSTAIN_LABEL]
            distinct_excluding.append(len(set(confident)))
            n_abstain += sum(label == ABSTAIN_LABEL for label in labels)
            n_labels += len(labels)

        if len(files) != EXPECTED_DIAL_CLIPS:
            raise FileNotFoundError(
                f"cfg={C}: expected {EXPECTED_DIAL_CLIPS} cfg-dial caches, found {len(files)}"
            )
        ci_lo, ci_hi = binomial_wilson_ci(n_abstain, n_labels)
        by_cfg[C] = {
            "n_clips": len(files),
            "n_labels": n_labels,
            "n_abstain": n_abstain,
            "mean_distinct_including_abstain": float(np.mean(distinct_including)),
            "mean_distinct_excluding_abstain": float(np.mean(distinct_excluding)),
            "per_clip_min_including_abstain": int(np.min(distinct_including)),
            "per_clip_max_including_abstain": int(np.max(distinct_including)),
            "per_clip_min_excluding_abstain": int(np.min(distinct_excluding)),
            "per_clip_max_excluding_abstain": int(np.max(distinct_excluding)),
            "abstain_rate": n_abstain / n_labels,
            "abstain_ci_lo": ci_lo,
            "abstain_ci_hi": ci_hi,
        }

    return {
        "analysis": "Arc-4 WP-A abstain-filtered entropy lens",
        "source": DIAL_GLOB,
        "cfg_grid": list(CFG_LIST),
        "abstain_label": ABSTAIN_LABEL,
        "distinct_count_unit": "clip",
        "abstain_rate_unit": "independent final-class label",
        "abstain_ci": "two-sided Wilson score interval, 95%",
        "by_cfg": by_cfg,
        "series": {
            "mean_distinct_including_abstain": [
                by_cfg[C]["mean_distinct_including_abstain"] for C in CFG_LIST
            ],
            "mean_distinct_excluding_abstain": [
                by_cfg[C]["mean_distinct_excluding_abstain"] for C in CFG_LIST
            ],
            "abstain_rate": [by_cfg[C]["abstain_rate"] for C in CFG_LIST],
        },
        "decision_token": None,
    }


def bootstrap_clip_mean(
    values: list[float], n_boot: int = 1000, seed: int = 0
) -> tuple[float, float, float]:
    """Mean and percentile CI from a bootstrap over clips."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, arr.size, size=(n_boot, arr.size))
    draws = arr[indices].mean(axis=1)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return float(arr.mean()), float(lo), float(hi)


def build_entropy_lens_v3(n_boot: int = 1000, seed: int = 0) -> dict:
    """Publish inclusive and abstain-filtered diversity with clip-bootstrap intervals."""
    by_cfg: dict[str, dict[str, float | int]] = {}
    for C in CFG_LIST:
        files = sorted(glob.glob(str(ROOT / DIAL_GLOB.format(C=C))))
        if len(files) != EXPECTED_DIAL_CLIPS:
            raise FileNotFoundError(
                f"cfg={C}: expected {EXPECTED_DIAL_CLIPS} cfg-dial caches, found {len(files)}"
            )

        distinct_including: list[float] = []
        distinct_excluding: list[float] = []
        abstain_rates: list[float] = []
        n_labels = 0
        n_abstain = 0
        for fp in files:
            labels = np.asarray(np.load(fp, allow_pickle=True)["labels"]).tolist()
            if not labels:
                raise ValueError(f"empty label array in {fp}")
            confident = [label for label in labels if label != ABSTAIN_LABEL]
            distinct_including.append(float(len(set(labels))))
            distinct_excluding.append(float(len(set(confident))))
            n_clip_abstain = sum(label == ABSTAIN_LABEL for label in labels)
            abstain_rates.append(n_clip_abstain / len(labels))
            n_labels += len(labels)
            n_abstain += n_clip_abstain

        inc, inc_lo, inc_hi = bootstrap_clip_mean(distinct_including, n_boot, seed)
        exc, exc_lo, exc_hi = bootstrap_clip_mean(distinct_excluding, n_boot, seed)
        abstain, abstain_lo, abstain_hi = bootstrap_clip_mean(abstain_rates, n_boot, seed)
        by_cfg[C] = {
            "n_clips": len(files),
            "n_labels": n_labels,
            "n_abstain": n_abstain,
            "mean_distinct_including_abstain": inc,
            "mean_distinct_including_abstain_ci_lo": inc_lo,
            "mean_distinct_including_abstain_ci_hi": inc_hi,
            "mean_distinct_excluding_abstain": exc,
            "mean_distinct_excluding_abstain_ci_lo": exc_lo,
            "mean_distinct_excluding_abstain_ci_hi": exc_hi,
            "abstain_rate": abstain,
            "abstain_ci_lo": abstain_lo,
            "abstain_ci_hi": abstain_hi,
        }

    return {
        "analysis": "Arc-4 WP-A2 descriptive entropy lens",
        "source": DIAL_GLOB,
        "cfg_grid": list(CFG_LIST),
        "abstain_label": ABSTAIN_LABEL,
        "ci_method": "clip_bootstrap",
        "ci_level": 0.95,
        "bootstrap_unit": "clip",
        "n_boot": n_boot,
        "bootstrap_seed": seed,
        "interpretation": "descriptive_only",
        "by_cfg": by_cfg,
        "series": {
            "mean_distinct_including_abstain": [
                by_cfg[C]["mean_distinct_including_abstain"] for C in CFG_LIST
            ],
            "mean_distinct_excluding_abstain": [
                by_cfg[C]["mean_distinct_excluding_abstain"] for C in CFG_LIST
            ],
            "abstain_rate": [by_cfg[C]["abstain_rate"] for C in CFG_LIST],
        },
        "decision_token": None,
        "mechanism_claim": None,
    }


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
def build() -> dict:
    b1 = load_budget(BUDGET_CFG1)
    b45 = load_budget(BUDGET_CFG45)
    cmap = load_cswap_map(CSWAP_MAP)
    csum = load_cswap_summary(CSWAP_SUMMARY)
    dcc = distinct_class_count_by_cfg()

    s_grid = sorted({s for ax in cmap for s in cmap[ax]})
    s_low = s_grid[0]                     # 0.05
    s_mid = 0.45 if 0.45 in s_grid else s_grid[len(s_grid) // 2]

    per_axis = {}
    for a in AXES:
        cond1 = b1[a]["conditioning"]
        cond45 = b45[a]["conditioning"]
        follow_low = cmap[a][s_low]["follow_rate"]
        follow_mid = cmap[a][s_mid]["follow_rate"]
        ret_high = cmap[a][max(s_grid)]["retention_rate"]
        s_cond = csum["axes"][a]["s_cond"]
        sanity = csum["axes"][a]["sanity"]
        # observational budget shares (cfg=1 and cfg=4.5)
        shares1 = {k: b1[a][k] for k in SHARE_KEYS}
        shares45 = {k: b45[a][k] for k in SHARE_KEYS}
        per_axis[a] = {
            "is_embedding": b1[a].get("is_embedding"),
            "observational": {
                "cfg1": shares1, "cfg45": shares45,
                "conditioning_share_cfg1": cond1,
                "conditioning_share_cfg45": cond45,
                "conditioning_share_delta_cfg1_to_cfg45": cond45 - cond1,
                "s_commit_cfg1": b1[a]["s_commit"],
                "s_commit_cfg45": b45[a]["s_commit"],
            },
            "causal_conditioning_responsiveness": {
                "_label": "cond-swap FOLLOW/RETENTION; NOT the observational conditioning share",
                "follow_rate_low_s": follow_low,
                "follow_rate_mid_s": follow_mid,
                "retention_rate_high_s": ret_high,
                "s_cond": s_cond,
                "sanity_passed": bool(sanity["passed"]),
                "follow_ok": bool(sanity["follow_ok"]),
                "retention_ok": bool(sanity["retention_ok"]),
                "by_s": cmap[a],
            },
            # The headline divergence = high/rising OBSERVATIONAL conditioning-share whose
            # causal cond-swap FAILS its sanity (follow does not hold up under a direct video
            # swap). This is the 'apparent share but not video-driven' case (class). An axis can
            # have a low mid-s follow yet a PASSING sanity because it commits LATE (material:
            # high follow at low s, sanity PASS) — that is late commitment, NOT spurious share.
            "divergence_flag": (
                (cond45 >= 0.5 or cond45 > cond1) and (not sanity["passed"])
            ),
        }

    entropy_lens = {
        "_meaning": "mean #distinct final-class labels across 16 independents per clip; "
                    "DROP with cfg = mode collapse = explainer for (i)/(ii) divergence",
        "distinct_class_count_by_cfg": dcc,
        "sequence_cfg_1_to_4p5": [round(dcc[c]["mean_distinct"], 4) for c in CFG_LIST],
        "collapse_delta": (dcc["1"]["mean_distinct"] - dcc["4.5"]["mean_distinct"]),
    }

    return {
        "part": "C — two budgets + entropy lens (pre-reg §C)",
        "pre_registered_as": "DESCRIPTIVE; emits NO token (no decision rule fires here)",
        "bootstrap_unit": "video",
        "s_grid": s_grid,
        "s_low": s_low,
        "s_mid": s_mid,
        "per_axis": per_axis,
        "entropy_lens": entropy_lens,
        "headline": {
            "axis": "class",
            "observational_conditioning_share_cfg1": per_axis["class"]["observational"][
                "conditioning_share_cfg1"],
            "observational_conditioning_share_cfg45": per_axis["class"]["observational"][
                "conditioning_share_cfg45"],
            "causal_follow_rate_low_s": per_axis["class"][
                "causal_conditioning_responsiveness"]["follow_rate_low_s"],
            "cond_swap_sanity_passed": per_axis["class"][
                "causal_conditioning_responsiveness"]["sanity_passed"],
            "statement": (
                "CLASS: observational conditioning-share RISES "
                f"{per_axis['class']['observational']['conditioning_share_cfg1']:.3f} -> "
                f"{per_axis['class']['observational']['conditioning_share_cfg45']:.3f} with cfg "
                "(looks MORE video-driven) BUT causal follow-rate is only "
                f"{per_axis['class']['causal_conditioning_responsiveness']['follow_rate_low_s']:.2f} "
                "and cond-swap sanity FAILS => NOT actually video-driven. The rise is an artifact "
                f"of mode collapse (distinct classes {entropy_lens['sequence_cfg_1_to_4p5'][0]} -> "
                f"{entropy_lens['sequence_cfg_1_to_4p5'][-1]}), not of the video."
            ),
        },
    }


def fmt(x, nd=3):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "n/a"
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def write_md(data: dict) -> None:
    pa = data["per_axis"]
    el = data["entropy_lens"]
    L: list[str] = []
    L.append("# Arc-3 Tier-B Part C — Two budgets + entropy lens (pre-reg §C)")
    L.append("")
    L.append("**Pre-registered as DESCRIPTIVE. No token is emitted from Part C.** "
             "Bootstrap unit = video. Assembly of cached artifacts only — NO new generation.")
    L.append("")
    L.append("Three views, side by side:")
    L.append("- **(i) Observational determination budget** — `conditioning` share of the final "
             "(determination decomposition), cfg=1.0 and cfg=4.5.")
    L.append("- **(ii) Causal conditioning responsiveness** — cond-swap FOLLOW / RETENTION / "
             "s_cond. *This is the causal follow-rate under a SWAPPED video conditioning, "
             "explicitly NOT the observational conditioning share of (i).*")
    L.append("- **(iii) Entropy lens** — distinct-class-count vs cfg (mode collapse), the "
             "explainer for why (i) and (ii) diverge at high cfg.")
    L.append("")

    # --- Table 1: observational budget shares ---
    L.append("## (i) Observational determination budget (shares)")
    L.append("")
    L.append("| axis | cfg | conditioning | seed | trajectory | residual | s_commit |")
    L.append("|---|---|---|---|---|---|---|")
    for a in AXES:
        for tag, key in (("1.0", "cfg1"), ("4.5", "cfg45")):
            sh = pa[a]["observational"][key]
            sc = pa[a]["observational"][f"s_commit_{key}"]
            L.append(f"| {a} | {tag} | {fmt(sh['conditioning'])} | {fmt(sh['seed'])} | "
                     f"{fmt(sh['trajectory'])} | {fmt(sh['residual'])} | {fmt(sc)} |")
    L.append("")

    # --- Table 2: causal cond-swap ---
    L.append("## (ii) Causal conditioning responsiveness (cond-swap; NOT the conditioning share)")
    L.append("")
    L.append(f"FOLLOW = output tracks a SWAPPED video conditioning; RETENTION = output keeps its "
             f"own conditioning when the swap is late. s_low={fmt(data['s_low'],2)}, "
             f"s_mid={fmt(data['s_mid'],2)}.")
    L.append("")
    L.append("| axis | follow@low_s | follow@mid_s | retention@high_s | s_cond | sanity |")
    L.append("|---|---|---|---|---|---|")
    for a in AXES:
        c = pa[a]["causal_conditioning_responsiveness"]
        san = "PASS" if c["sanity_passed"] else "**FAIL**"
        L.append(f"| {a} | {fmt(c['follow_rate_low_s'],2)} | {fmt(c['follow_rate_mid_s'],2)} | "
                 f"{fmt(c['retention_rate_high_s'],2)} | {fmt(c['s_cond'],2)} | {san} |")
    L.append("")

    # --- Table 3: THE CONTRAST (observational share vs causal follow) ---
    L.append("## CONTRAST — observational conditioning-share vs causal follow-rate (per axis)")
    L.append("")
    L.append("| axis | cond-share cfg1 | cond-share cfg4.5 | Δ share | causal follow@low_s | "
             "cond-swap sanity | reading |")
    L.append("|---|---|---|---|---|---|---|")
    for a in AXES:
        o = pa[a]["observational"]
        c = pa[a]["causal_conditioning_responsiveness"]
        d = o["conditioning_share_delta_cfg1_to_cfg45"]
        san = "PASS" if c["sanity_passed"] else "**FAIL**"
        if pa[a]["divergence_flag"]:
            reading = "**apparent share up/high BUT cond-swap FAILS -> not video-driven**"
        elif c["sanity_passed"] and c["follow_rate_low_s"] >= 0.5 and c["follow_rate_mid_s"] >= 0.5:
            reading = "consistent (truly video-responsive)"
        elif c["sanity_passed"] and c["follow_rate_low_s"] >= 0.5:
            reading = "consistent; late commitment (high early follow, sanity PASS)"
        else:
            reading = "see per-axis"
        L.append(f"| {a} | {fmt(o['conditioning_share_cfg1'])} | "
                 f"{fmt(o['conditioning_share_cfg45'])} | {fmt(d,3)} | "
                 f"{fmt(c['follow_rate_low_s'],2)} | {san} | {reading} |")
    L.append("")
    L.append(f"**Headline (class):** {data['headline']['statement']}")
    L.append("")

    # --- Table 4: entropy lens ---
    L.append("## (iii) Entropy lens — distinct-class-count vs cfg (the explainer)")
    L.append("")
    L.append("Mean # distinct final-class labels across the 16 independents per clip "
             "(cfg-dial cache, 24 clips/cfg). A DROP as cfg rises = mode collapse.")
    L.append("")
    L.append("| cfg | mean distinct classes | n_clips |")
    L.append("|---|---|---|")
    for C in CFG_LIST:
        d = el["distinct_class_count_by_cfg"][C]
        L.append(f"| {C} | {fmt(d['mean_distinct'],2)} | {d['n_clips']} |")
    L.append("")
    L.append(f"Sequence cfg 1.0->4.5: {el['sequence_cfg_1_to_4p5']} "
             f"(collapse Δ = {fmt(el['collapse_delta'],2)}).")
    L.append("")
    L.append("## Interpretation (descriptive)")
    L.append("")
    L.append("- **presence / timing / material** are CONSISTENT across the two budgets: high "
             "observational conditioning-share AND high causal follow-rate at EARLY s with a "
             "PASSING cond-swap sanity — these axes are genuinely video-driven. (Material's "
             "mid-s follow is low and its s_commit is late ~0.65, i.e. LATE commitment with an "
             "early follow of 0.80 and a PASSING sanity — not spurious share.)")
    L.append("- **class is the divergence case.** Its observational conditioning-share actually "
             f"RISES with cfg ({fmt(pa['class']['observational']['conditioning_share_cfg1'])} -> "
             f"{fmt(pa['class']['observational']['conditioning_share_cfg45'])}), which naively "
             "reads as 'class becomes MORE video-determined under guidance'. The causal cond-swap "
             "contradicts that: class follow-rate is low "
             f"({fmt(pa['class']['causal_conditioning_responsiveness']['follow_rate_low_s'],2)} at "
             "low s, falling toward 0 at high s) and the cond-swap sanity FAILS.")
    L.append("- **The entropy lens reconciles them.** Distinct-class-count collapses "
             f"{el['sequence_cfg_1_to_4p5'][0]} -> {el['sequence_cfg_1_to_4p5'][-1]} as cfg rises. "
             "Under mode collapse the generator concentrates on fewer classes, so the final class "
             "becomes more PREDICTABLE/self-consistent within a clip — inflating the OBSERVATIONAL "
             "conditioning-share — without the class actually FOLLOWING the video. The two budgets "
             "diverge precisely because observational determination conflates 'video-driven' with "
             "'low-entropy/collapsed', whereas the causal cond-swap isolates the video channel.")
    L.append("")
    L.append("_Generated by scripts/c_two_budgets.py — assembly of cached artifacts, no new "
             "generation, no token (pre-reg §C is descriptive)._")
    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")


def write_entropy_lens_v2_md(data: dict) -> None:
    """Write the Arc-4 abstain-aware lens without touching the Arc-3 report."""
    lines = [
        "# Arc-4 WP-A abstain-filtered entropy lens",
        "",
        "Cached analysis only; no generation and no decision token. Distinct-class counts are "
        "clip-level means. Abstain rates pool the independent final-class labels at each cfg; "
        "intervals are two-sided Wilson score 95% binomial intervals.",
        "",
        "| cfg | clips | labels | distinct incl. abstain | distinct excl. abstain | "
        "abstain rate (95% CI) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for C in CFG_LIST:
        row = data["by_cfg"][C]
        lines.append(
            f"| {C} | {row['n_clips']} | {row['n_labels']} | "
            f"{row['mean_distinct_including_abstain']:.4f} | "
            f"{row['mean_distinct_excluding_abstain']:.4f} | "
            f"{row['abstain_rate']:.4f} "
            f"[{row['abstain_ci_lo']:.4f}, {row['abstain_ci_hi']:.4f}] |"
        )
    lines.extend([
        "",
        "The inclusive series is the legacy Arc-3 entropy lens. The filtered series removes "
        "`abstain` from each clip's distinct-label set; clips with no confident label contribute "
        "a distinct count of zero.",
        "",
        "_Generated by `scripts/c_two_budgets.py --exclude-abstain`._",
    ])
    ARC4_OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_entropy_lens_v3_md(data: dict) -> None:
    """Write the descriptive WP-A2 lens; no causal or mechanism interpretation."""
    lines = [
        "# Arc-4 WP-A2 descriptive entropy lens",
        "",
        "Cached analysis only; no generation and no decision token. Means and 95% intervals "
        "use a 1,000-draw bootstrap over the 24 clips at each cfg (seed 0).",
        "",
        "| cfg | distinct incl. abstain (95% CI) | distinct excl. abstain (95% CI) | "
        "abstain rate (95% CI) |",
        "|---|---:|---:|---:|",
    ]
    for C in CFG_LIST:
        row = data["by_cfg"][C]
        lines.append(
            f"| {C} | {row['mean_distinct_including_abstain']:.4f} "
            f"[{row['mean_distinct_including_abstain_ci_lo']:.4f}, "
            f"{row['mean_distinct_including_abstain_ci_hi']:.4f}] | "
            f"{row['mean_distinct_excluding_abstain']:.4f} "
            f"[{row['mean_distinct_excluding_abstain_ci_lo']:.4f}, "
            f"{row['mean_distinct_excluding_abstain_ci_hi']:.4f}] | "
            f"{row['abstain_rate']:.4f} "
            f"[{row['abstain_ci_lo']:.4f}, {row['abstain_ci_hi']:.4f}] |"
        )
    lines.extend([
        "",
        "The two distinct-count series are reported side by side for continuity. They are "
        "descriptive diversity summaries and do not establish a causal mechanism.",
        "",
        "_Generated by `scripts/c_two_budgets.py`._",
    ])
    ARC4_WPA2_OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--exclude-abstain",
        action="store_true",
        help="deprecated alias for the default abstain-aware WP-A2 output",
    )
    parser.add_argument(
        "--legacy-arc3",
        action="store_true",
        help="reproduce the withdrawn Arc-3 narrative at its historical output path",
    )
    args = parser.parse_args([] if argv is None else argv)

    def _rel(p: Path) -> str:
        try:
            return str(p.relative_to(ROOT))
        except ValueError:
            return str(p)

    if args.legacy_arc3:
        for p in (BUDGET_CFG1, BUDGET_CFG45, CSWAP_MAP, CSWAP_SUMMARY):
            if not p.exists():
                raise SystemExit(f"c_two_budgets: missing cached input {p}")
        data = build()
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        OUT_JSON.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        write_md(data)
        print(f"wrote historical {_rel(OUT_JSON)} and {_rel(OUT_MD)}")
        return

    data = build_entropy_lens_v3()
    ARC4_WPA2_OUT_DIR.mkdir(parents=True, exist_ok=True)
    ARC4_WPA2_OUT_JSON.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    write_entropy_lens_v3_md(data)
    print(f"wrote {_rel(ARC4_WPA2_OUT_JSON)} and {_rel(ARC4_WPA2_OUT_MD)}")


if __name__ == "__main__":
    main(sys.argv[1:])
