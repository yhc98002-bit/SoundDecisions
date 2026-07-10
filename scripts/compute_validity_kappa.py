#!/usr/bin/env python
"""Validity FULL SUITE (June-13 manual §3.3) — human ↔ measurer ↔ qwen.

Joins the human-labeled validity JSONL (exported from the labeling bundle) with the
measurer's own self-target labels on the SAME generated clips and the qwen sidecar
labels, and reports — per axis, per rater-pair — the FULL suite the manual now
mandates: raw agreement, marginals, confusion matrix, Cohen κ, Gwet AC1, PABAK.
"Never a lone chance-corrected number (under the ~90% skew here κ is too harsh and
AC1 too lenient; the confusion matrix is the truth)."

Rater pairs:
  * human vs measurer — the validity quantity (correctness-layer gate, θ_cal = 0.6;
    NOT a GO_MAPS_PHASE precondition under the §3.3 split — diagnostic here);
  * human vs qwen      — MLLM-as-proxy quality;
  * qwen vs measurer   — context (already in the reliability gate).

Label provenance: measurer labels = the screen_ind0 self-target from
results/stage0/measurements/measurements.jsonl (the model's measured value on the
generation the human heard). Timing: human onset_s is SECONDS (binned here against the
frozen bin width); measurer/qwen timing labels are ALREADY bin indices (used as-is).
Class abstain is shown in the confusion matrix but excluded from the chance-corrected
scalars (computed on the confident subset); abstain counts reported alongside.

CPU-only. Output: results/stage0/validity_suite.{json,md}.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from foley_cw.sidecar import (cohens_kappa, confusion_matrix, gwet_ac1,  # noqa: E402
                              pabak)

AXES = ("presence", "class", "timing")
ABSTAIN = "abstain"


def load_human(jsonl: Path, bin_s: float) -> dict[str, dict]:
    """clip -> {presence, class, timing(bin)} from the exported human JSONL."""
    out: dict[str, dict] = {}
    for line in Path(jsonl).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        rec = {"presence": d.get("presence"), "class": d.get("class")}
        onset = d.get("onset_s")
        rec["timing"] = int(float(onset) // bin_s) if onset is not None else None
        out[str(d["clip_id"])] = rec
    return out


def load_qwen(sidecar_csv: Path) -> dict[str, dict]:
    """clip -> {axis: label}; timing/qwen labels are already bin indices."""
    q: dict[str, dict] = defaultdict(dict)
    if sidecar_csv.exists():
        for r in csv.DictReader(sidecar_csv.open()):
            q[r["clip"]][r["axis_id"]] = r["label"]
    return q


def load_measurer(measurements_jsonl: Path) -> dict[str, dict]:
    """clip -> {axis: label} from the screen_ind0 measurements (self-target)."""
    m: dict[str, dict] = defaultdict(dict)
    if not measurements_jsonl.exists():
        return m
    for line in measurements_jsonl.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        e = d.get("extra") or {}
        gid = d.get("gen_id", "")
        # screen-task independents, ind0 = representative self-target
        if e.get("j") != 0 or not gid.endswith("__screen_ind0"):
            continue
        clip = e.get("clip") or gid.split("__")[0]
        tgt = d.get("target") or {}
        if d["axis_id"] in AXES and tgt.get("label") is not None:
            m[str(clip)][d["axis_id"]] = str(tgt["label"])
    return m


def _norm(axis: str, v) -> str | None:
    """Canonical comparable label (timing already binned on both sides upstream)."""
    if v is None:
        return None
    if axis == "timing":
        try:
            return str(int(float(v)))
        except (TypeError, ValueError):
            return None
    return str(v)


def suite_for(axis: str, ra: dict, rb: dict) -> dict:
    """Full suite for one axis over the clip-intersection of two raters."""
    full_a, full_b = [], []          # both-present (abstain kept) — for the matrix
    conf_a, conf_b = [], []          # confident subset — for scalars
    for clip in sorted(set(ra) & set(rb)):
        va, vb = _norm(axis, ra[clip].get(axis)), _norm(axis, rb[clip].get(axis))
        if va is None or vb is None:
            continue
        full_a.append(va)
        full_b.append(vb)
        if va == ABSTAIN or vb == ABSTAIN:
            continue
        conf_a.append(va)
        conf_b.append(vb)
    n_conf = len(conf_a)
    p_o = (sum(1 for x, y in zip(conf_a, conf_b) if x == y) / n_conf) if n_conf else float("nan")
    # Timing: exact-bin agreement is harsh (the measurer's flux onset fires at t≈0 while
    # humans tap the later salient event); report the ±1-bin agreement the robustness gate
    # uses, so the timing read is honest in both directions.
    within1 = None
    if axis == "timing" and n_conf:
        within1 = sum(1 for x, y in zip(conf_a, conf_b)
                      if abs(int(x) - int(y)) <= 1) / n_conf
    return {
        "n_overlap": len(full_a),
        "n_confident": n_conf,
        "n_abstain_a": sum(1 for x in full_a if x == ABSTAIN),
        "n_abstain_b": sum(1 for x in full_b if x == ABSTAIN),
        "raw_agreement": p_o,
        "within_1_bin_agreement": within1,
        "cohens_kappa": cohens_kappa(conf_a, conf_b) if n_conf >= 2 else float("nan"),
        "gwet_ac1": gwet_ac1(conf_a, conf_b) if n_conf >= 2 else float("nan"),
        "pabak": pabak(conf_a, conf_b) if n_conf >= 2 else float("nan"),
        "marginal_a": dict(Counter(conf_a)),
        "marginal_b": dict(Counter(conf_b)),
        "confusion_matrix": confusion_matrix(full_a, full_b),
    }


def _fmt(x) -> str:
    return "nan" if (isinstance(x, float) and math.isnan(x)) else f"{x:.3f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--human-jsonl", type=Path,
                    default=Path("results/labeling/labels_validity_v1.jsonl"))
    ap.add_argument("--sidecar-csv", type=Path,
                    default=Path("results/stage0/mllm_sidecar/sidecar.csv"))
    ap.add_argument("--measurements", type=Path,
                    default=Path("results/stage0/measurements/measurements.jsonl"))
    ap.add_argument("--timing-bin-s", type=float, default=0.5)
    ap.add_argument("--out", type=Path, default=Path("results/stage0"))
    args = ap.parse_args()

    human = load_human(args.human_jsonl, args.timing_bin_s)
    qwen = load_qwen(args.sidecar_csv)
    meas = load_measurer(args.measurements)

    pairs = {"human_vs_measurer": (human, meas),
             "human_vs_qwen": (human, qwen),
             "qwen_vs_measurer": (qwen, meas)}
    out = {
        "_doc": "Validity full suite (§3.3): raw agreement, marginals, confusion matrix, "
                "Cohen κ, Gwet AC1, PABAK. DIAGNOSTIC (correctness layer); θ_cal=0.6 is the "
                "human-vs-measurer gate but NOT a GO_MAPS_PHASE precondition (self-target split).",
        "timing_bin_s": args.timing_bin_s,
        "n_human_clips": len(human),
        "pairs": {},
    }
    for name, (ra, rb) in pairs.items():
        out["pairs"][name] = {ax: suite_for(ax, ra, rb) for ax in AXES}

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "validity_suite.json").write_text(json.dumps(out, indent=2))

    L = ["# Validity Full Suite — §3.3 (DIAGNOSTIC; confusion matrix is the truth-teller)",
         "",
         f"Human clips: {len(human)}; timing bin = {args.timing_bin_s}s. The human-vs-measurer "
         "κ ≥ 0.6 is the correctness-layer gate (NOT a GO_MAPS_PHASE precondition under the "
         "self-target split). Scalars are on the confident subset (class abstain dropped); the "
         "confusion matrix keeps abstain.", ""]
    for name, (ra, rb) in pairs.items():
        L += [f"## {name}",
              "| axis | n_conf | raw | Cohen κ | Gwet AC1 | PABAK | abstain a/b |",
              "|---|---|---|---|---|---|---|"]
        for ax in AXES:
            s = out["pairs"][name][ax]
            L.append(f"| {ax} | {s['n_confident']} | {_fmt(s['raw_agreement'])} | "
                     f"{_fmt(s['cohens_kappa'])} | {_fmt(s['gwet_ac1'])} | {_fmt(s['pabak'])} | "
                     f"{s['n_abstain_a']}/{s['n_abstain_b']} |")
        t1 = out["pairs"][name]["timing"].get("within_1_bin_agreement")
        if t1 is not None:
            L.append(f"\n_timing ±1-bin agreement: {t1:.3f} (exact-bin is harsh — measurer "
                     f"onset fires at t≈0)._")
        # class confusion matrix inline (the most informative axis)
        cm = out["pairs"][name]["class"]["confusion_matrix"]
        L += ["", f"_class confusion ({name}; rows={name.split('_vs_')[0]}, "
              f"cols={name.split('_vs_')[1]}):_", "",
              "| r\\c | " + " | ".join(cm["labels"]) + " |",
              "|" + "---|" * (len(cm["labels"]) + 1)]
        for i, lab in enumerate(cm["labels"]):
            L.append(f"| {lab} | " + " | ".join(str(c) for c in cm["matrix"][i]) + " |")
        L.append("")
    (args.out / "validity_suite.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
