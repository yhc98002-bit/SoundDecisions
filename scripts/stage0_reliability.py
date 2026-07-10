#!/usr/bin/env python
"""Stage-0 reliability gate on REAL generated audio (manual section 3.3, Tab. 1).

Per axis: determinism (test-retest on identical wavs), robustness (the five real
waveform perturbations), validity (Cohen's kappa vs the MLLM sidecar gold for
categorical axes). Thresholds come frozen from configs/thresholds.json
(theta_rel >= 0.95, theta_robust >= 0.85, kappa >= 0.6).

Honest gaps, reported not hidden:
  * material (embedding axis) has no MLLM-judgeable gold -> validity NaN ->
    demoted with an explicit reason; whether to validate it via a second
    embedder is a PI-checkpoint question.
  * the ~50-clip HUMAN sidecar is pending (PI checkpoint); kappa here is vs
    MLLM-only and is labelled as such.

Emits AXIS_DEMOTED:<axis> tokens; >= 3 surviving axes is a GO_MAPS_PHASE
precondition. Output: results/stage0/reliability_report.md + .json.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from foley_cw.config import load_config  # noqa: E402
from foley_cw.real_perturbations import REAL_PERTURBATIONS  # noqa: E402
from foley_cw.sidecar import run_real_reliability  # noqa: E402
from foley_cw.types import AxisKind, SelfTarget  # noqa: E402

GATE_AXIS_IDS = ("presence", "timing", "class", "material")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--finals-dir", type=Path, default=Path("results/stage0/finals"))
    ap.add_argument("--sidecar-csv", type=Path,
                    default=Path("results/stage0/mllm_sidecar/sidecar.csv"))
    ap.add_argument("--out", type=Path, default=Path("results/stage0"))
    ap.add_argument("--n-clips", type=int, default=50,
                    help="determinism-subset size; validity kappa is scored on THIS "
                         "subset's gold-overlap (not all gold) — see run_real_reliability")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    import soundfile as sf

    from foley_cw.real_measurer import RealFoleyMeasurer

    cfg_all = load_config()
    thresholds = cfg_all.thresholds
    if not thresholds.frozen:
        raise SystemExit("thresholds not frozen — refusing to run the reliability gate")
    axes = {a.id: a for a in cfg_all.axes if a.id in GATE_AXIS_IDS}

    gold: dict[str, dict[str, SelfTarget]] = {a: {} for a in GATE_AXIS_IDS}
    if args.sidecar_csv.exists():
        for r in csv.DictReader(args.sidecar_csv.open()):
            ax = axes.get(r["axis_id"])
            if ax is None:
                continue
            label = int(r["label"]) if r["label"].lstrip("-").isdigit() else r["label"]
            gold[r["axis_id"]][r["clip"]] = SelfTarget(axis_id=r["axis_id"], kind=ax.kind,
                                                       label=label)
    else:
        print(f"[reliability] WARNING: no sidecar at {args.sidecar_csv}; validity will be NaN")

    gold_clips = sorted({c for by in gold.values() for c in by})
    rng = np.random.default_rng(args.seed)
    det_clips = (list(rng.permutation(gold_clips))[: args.n_clips] if gold_clips else [])
    use_clips = sorted(set(det_clips) | set(gold_clips))

    wavs_by_clip: dict[str, np.ndarray] = {}
    for clip in use_clips:
        p = args.finals_dir / f"{clip}__screen_ind0.wav"
        if p.exists():
            wav, _sr = sf.read(p, dtype="float32")
            wavs_by_clip[clip] = np.asarray(wav, dtype=np.float32)
    print(f"[reliability] {len(wavs_by_clip)} generated wavs loaded "
          f"({len(gold_clips)} with MLLM gold)", flush=True)
    if not wavs_by_clip:
        raise SystemExit("no screened finals found — run stage0_screening first")

    measurer = RealFoleyMeasurer(device=args.device)
    results, tokens = [], []
    for axis_id in GATE_AXIS_IDS:
        ax = axes[axis_id]
        axis_gold = gold[axis_id] if ax.kind is AxisKind.CATEGORICAL else None
        det_subset = {c: wavs_by_clip[c] for c in det_clips if c in wavs_by_clip}
        res = run_real_reliability(ax, det_subset, measurer, thresholds,
                                   np.random.default_rng(args.seed + 1),
                                   gold=axis_gold, perturbations=REAL_PERTURBATIONS)
        if ax.kind is not AxisKind.CATEGORICAL and not np.isfinite(res.validity):
            res.reason = (res.reason + "; " if res.reason else "") + \
                "no MLLM-judgeable gold for an embedding axis — second-embedder " \
                "validation is a PI-checkpoint question"
        results.append(res)
        if res.demoted:
            tokens.append(f"AXIS_DEMOTED:{axis_id}")
        print(f"  {axis_id}: det={res.determinism:.3f} rob={res.robustness:.3f} "
              f"val={res.validity:.3f} passed={res.passed} demoted={res.demoted} "
              f"{('(' + res.reason + ')') if res.reason else ''}", flush=True)

    survivors = [r.axis_id for r in results if not r.demoted]
    gate_ok = len(survivors) >= 3
    lines = ["# Axis Reliability Report — Stage 0 (manual section 3.3; feeds Tab. 1)", "",
             f"Thresholds (frozen): determinism >= {thresholds.theta_rel}, robustness >= "
             f"{thresholds.theta_robust}, validity kappa >= {thresholds.theta_cal} "
             f"(vs MLLM sidecar; HUMAN sidecar pending at the PI checkpoint).", "",
             "| axis | determinism | robustness | validity (kappa vs MLLM) | passed | demoted | reason |",
             "|---|---|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r.axis_id} | {r.determinism:.3f} | {r.robustness:.3f} | "
                     f"{r.validity:.3f} | {r.passed} | {r.demoted} | {r.reason or '—'} |")
    lines += ["", f"**Surviving axes: {survivors} ({len(survivors)}; need >= 3 for "
              f"GO_MAPS_PHASE) — {'OK' if gate_ok else 'INSUFFICIENT'}**", ""]
    if tokens:
        lines.append(f"**Tokens:** {', '.join(tokens)}")
    (args.out / "reliability_report.md").write_text("\n".join(lines) + "\n")
    (args.out / "reliability_report.json").write_text(json.dumps({
        "results": [{"axis_id": r.axis_id, "determinism": r.determinism,
                     "robustness": r.robustness, "validity": r.validity,
                     "passed": r.passed, "demoted": r.demoted, "reason": r.reason}
                    for r in results],
        "tokens": tokens, "survivors": survivors, "gate_ok": gate_ok,
        "validity_source": "MLLM (qwen3.5-omni-plus) only; human sidecar pending",
    }, indent=2))
    print(f"[reliability] survivors={survivors} gate_ok={gate_ok}")
    return 0 if gate_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
