#!/usr/bin/env python
"""Stage-0 MLLM validity sidecar (manual section 3.3) — qwen3.5-omni-plus judgments.

Judges ~100 screening-clip GENERATED audios (finals/<clip>__screen_ind0.wav) on
{presence, class, timing}, plus a 20-clip test-retest repeat for MLLM-judge
stability. Cached + journaled (resumable); call budget enforced. Login node only
(needs the qwen proxy). The ~50-clip HUMAN sidecar is a PI-checkpoint item.

Output: results/stage0/mllm_sidecar/sidecar.csv with one gold SelfTarget per
(clip, axis), and retest_agreement per axis in sidecar_summary.json.
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
from foley_cw.mllm_judge import QwenOmniJudge  # noqa: E402

SIDECAR_AXES = ("presence", "class", "timing")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=Path("data/manifests/screening_manifest.json"))
    ap.add_argument("--finals-dir", type=Path, default=Path("results/stage0/finals"))
    ap.add_argument("--out", type=Path, default=Path("results/stage0/mllm_sidecar"))
    ap.add_argument("--n-clips", type=int, default=100)
    ap.add_argument("--n-retest", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=int, default=500)
    ap.add_argument("--axes", default=",".join(SIDECAR_AXES),
                    help="comma list of axes to (re-)judge; merges into the existing csv")
    ap.add_argument("--prompt-version", default="v1",
                    help="cache-key component; bump to bust the cache when the prompt/"
                         "label-space changes (the cache key does NOT include the text)")
    args = ap.parse_args()

    run_axes = tuple(a for a in args.axes.split(",") if a)
    cfg_all = load_config()
    axes = {a.id: a for a in cfg_all.axes if a.id in SIDECAR_AXES}
    # EVENT-RESTRICTED class label space (matches the measurer + the human bundle;
    # the full 15-class list let qwen pick excluded classes — speech/music/ambient —
    # which the event-restricted measurer can never match, confounding the class kappa).
    cmap = json.loads(Path("configs/coarse_class_map.json").read_text())
    excluded = set(cmap.get("class_excluded_coarse", []))
    coarse = [c for c in cmap["coarse_classes"] if c not in excluded]

    clips = sorted(json.loads(args.manifest.read_text())["clips"])
    rng = np.random.default_rng(args.seed)
    have = [c for c in clips if (args.finals_dir / f"{c}__screen_ind0.wav").exists()]
    if len(have) < args.n_clips:
        print(f"[sidecar] only {len(have)} screened finals available "
              f"(want {args.n_clips}); judging what exists", flush=True)
    chosen = list(rng.permutation(have))[: args.n_clips]
    retest = list(rng.permutation(chosen))[: args.n_retest]

    judge = QwenOmniJudge(coarse_classes=coarse, budget_max_calls=args.budget,
                          prompt_version=args.prompt_version)
    args.out.mkdir(parents=True, exist_ok=True)

    # merge mode: keep existing rows for axes we are NOT re-judging
    existing = {}
    csv_path = args.out / "sidecar.csv"
    if csv_path.exists():
        for r in csv.DictReader(csv_path.open()):
            existing[(r["clip"], r["axis_id"])] = r
    new_rows = []
    for i, clip in enumerate(chosen):
        wav = args.finals_dir / f"{clip}__screen_ind0.wav"
        for axis_id in run_axes:
            tgt = judge.judge(wav, axes[axis_id])
            new_rows.append({"clip": clip, "axis_id": axis_id, "label": str(tgt.label),
                             "wav": str(wav)})
        if (i + 1) % 10 == 0:
            print(f"[sidecar] {i + 1}/{len(chosen)} clips judged ({run_axes})", flush=True)
    for r in new_rows:
        existing[(r["clip"], r["axis_id"])] = r

    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["clip", "axis_id", "label", "wav"])
        w.writeheader()
        w.writerows([existing[k] for k in sorted(existing)])

    # test-retest: cache-bypassed repeated judgments, agreement per axis. Only the
    # (re-)judged axes are retested — and only when --n-retest > 0 — so a partial,
    # cache-busting re-run (e.g. --axes class --prompt-version v2er) does NOT silently
    # re-charge the budget for axes it is not touching. Prior agreements are preserved.
    summ_path = args.out / "sidecar_summary.json"
    prior_ra = (json.loads(summ_path.read_text()).get("retest_agreement", {})
                if summ_path.exists() else {})
    retest_agreement = dict(prior_ra)
    if args.n_retest > 0:
        retest_wavs = [args.finals_dir / f"{c}__screen_ind0.wav" for c in retest]
        for axis_id in run_axes:
            retest_agreement[axis_id] = judge.test_retest(retest_wavs, axes[axis_id], n=2)

    summary = {"n_clips": len(chosen), "n_retest": len(retest) if args.n_retest > 0 else 0,
               "retest_agreement": retest_agreement,
               "axes_judged": list(run_axes),
               "model": judge.model, "prompt_version": judge.prompt_version,
               "human_sidecar": "PENDING — ~50 clips at the PI checkpoint (manual 3.3)"}
    (args.out / "sidecar_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
