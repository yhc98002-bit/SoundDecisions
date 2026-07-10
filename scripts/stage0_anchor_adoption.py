#!/usr/bin/env python
"""Stage-0 anchor adoption decision (manual §3.2) — human marks vs audio-track onset.

The 30-clip human anchor check has two jobs: report MAE/coverage, and arbitrate the
**audio-only adoption** rule. `data/manifests/anchor_check_30.csv` carries
`proposed_onset_s` = the primary audio-track (spectral-flux) onset of each ORIGINAL clip;
`results/labeling/labels_anchor_v1.jsonl` carries the human `human_onset_s` (blank = no
discrete event, excluded). We compute Δ = human − audio onset over the marked clips and
the dispersion σ_anchor, then apply the PRE-REGISTERED rule:

    if σ_anchor (human vs audio-track) ≤ 0.35 s  →  ADOPT audio-only anchors,
        re-derive timing bins as max(0.5 s, 2·σ_anchor);
    else  →  keep the approved chain (foleybench_metadata → visual → light human),
        timing bin = max(0.5 s, 2·median audio-vs-visual σ) from anchors.json.

σ_anchor headline = median |Δ| (robust to the occasional human mis-tap, matching the
manual's median-σ framing); MAE / RMSE / std / bias reported alongside. Applying this
rule is executing a frozen decision (§3.2), not changing a frozen quantity.

CPU-only. Output: results/stage0/anchor_adoption.{json,md} + the chosen timing_bin_s.
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

SIGMA_ADOPT_THRESHOLD_S = 0.35   # frozen §3.2
MIN_BIN_S = 0.5                  # frozen floor


def load_proposed(check_csv: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    for r in csv.DictReader(check_csv.open()):
        if r.get("proposed_onset_s"):
            out[str(r["key"])] = float(r["proposed_onset_s"])
    return out


def load_human(jsonl: Path) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for line in Path(jsonl).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        v = d.get("human_onset_s")
        out[str(d["clip_id"])] = (float(v) if v is not None else None)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--human-jsonl", type=Path,
                    default=Path("results/labeling/labels_anchor_v1.jsonl"))
    ap.add_argument("--check-csv", type=Path,
                    default=Path("data/manifests/anchor_check_30.csv"))
    ap.add_argument("--anchors-json", type=Path,
                    default=Path("results/stage0/anchors.json"))
    ap.add_argument("--out", type=Path, default=Path("results/stage0"))
    args = ap.parse_args()

    proposed = load_proposed(args.check_csv)
    human = load_human(args.human_jsonl)

    n_csv = len(proposed)
    marked = {k: human[k] for k in human if human.get(k) is not None and k in proposed}
    n_marked = len(marked)
    n_no_event = sum(1 for k in human if human.get(k) is None)

    diffs = [marked[k] - proposed[k] for k in marked]
    import numpy as np
    d = np.array(diffs, dtype=float)
    abs_d = np.abs(d)
    sigma_anchor = float(np.median(abs_d)) if len(d) else float("nan")   # robust headline
    stats = {
        "n_check_csv": n_csv,
        "n_human_marked": n_marked,
        "n_human_no_event": n_no_event,
        "coverage": (n_marked / n_csv) if n_csv else float("nan"),
        "mae_s": float(np.mean(abs_d)) if len(d) else float("nan"),
        "median_abs_dev_s": sigma_anchor,
        "rmse_s": float(np.sqrt(np.mean(d ** 2))) if len(d) else float("nan"),
        "std_s": float(np.std(d)) if len(d) else float("nan"),
        "bias_s_human_minus_audio": float(np.mean(d)) if len(d) else float("nan"),
        "sigma_anchor_s": sigma_anchor,
    }

    adopt_audio_only = bool(np.isfinite(sigma_anchor) and sigma_anchor <= SIGMA_ADOPT_THRESHOLD_S)
    if adopt_audio_only:
        timing_bin_s = max(MIN_BIN_S, 2.0 * sigma_anchor)
        anchor_source = "foleybench_audio_onset"
        token = "AUDIO_ANCHOR_ADOPTED"
    else:
        # keep approved chain; bin from the audio-vs-visual median σ in anchors.json
        median_av_sigma = float("nan")
        if args.anchors_json.exists():
            a = json.loads(args.anchors_json.read_text())
            median_av_sigma = float((a.get("summary") or {}).get("median_sigma_s") or float("nan"))
        timing_bin_s = max(MIN_BIN_S, 2.0 * median_av_sigma) if np.isfinite(median_av_sigma) else MIN_BIN_S
        anchor_source = "approved_chain(metadata>visual>light_human)"
        token = "AUDIO_ANCHOR_NOT_ADOPTED"
        stats["median_audio_vs_visual_sigma_s"] = median_av_sigma

    out = {
        "_doc": "Anchor adoption decision (§3.2): human marks vs audio-track onset; the "
                "audio-only adoption rule is pre-registered (σ_anchor ≤ 0.35s).",
        "sigma_adopt_threshold_s": SIGMA_ADOPT_THRESHOLD_S,
        "stats": stats,
        "decision": {
            "adopt_audio_only": adopt_audio_only,
            "token": token,
            "anchor_source": anchor_source,
            "timing_bin_s": round(timing_bin_s, 4),
            "rule": "bins >= 2*sigma_anchor, floored at 0.5s",
        },
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "anchor_adoption.json").write_text(json.dumps(out, indent=2))

    def f(x):
        return "nan" if (isinstance(x, float) and math.isnan(x)) else f"{x:.4f}"
    L = ["# Anchor Adoption Decision — §3.2 (human marks vs audio-track onset)", "",
         f"30-clip human check: {n_marked} marked, {n_no_event} marked 'no event', "
         f"coverage {f(stats['coverage'])}.", "",
         "| stat | value (s) |", "|---|---|",
         f"| MAE | {f(stats['mae_s'])} |",
         f"| median \\|Δ\\| (= σ_anchor) | {f(stats['median_abs_dev_s'])} |",
         f"| RMSE | {f(stats['rmse_s'])} |",
         f"| std | {f(stats['std_s'])} |",
         f"| bias (human − audio) | {f(stats['bias_s_human_minus_audio'])} |", "",
         f"**Decision: `{token}`** — σ_anchor = {f(sigma_anchor)}s "
         f"{'≤' if adopt_audio_only else '>'} {SIGMA_ADOPT_THRESHOLD_S}s → "
         f"{'ADOPT audio-only' if adopt_audio_only else 'keep approved chain'}; "
         f"anchor source = `{anchor_source}`; **timing_bin_s = "
         f"{out['decision']['timing_bin_s']}** (bins ≥ 2·σ_anchor, floor 0.5s).", ""]
    (args.out / "anchor_adoption.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
