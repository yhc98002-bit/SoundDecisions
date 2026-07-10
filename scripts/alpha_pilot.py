#!/usr/bin/env python
"""Alpha pilot (manual section 1.3) — select the primary fork-noise scale.

Primary alpha = the SMALLEST alpha in the pilot grid satisfying, averaged over
pilot clips, at the PRIMARY cfg=4.5:
  (a) early-endpoint washout: |A_fork(s=0.05) - A_independent| <= 0.10 on the
      class axis (the Stage-M criterion-1 early endpoint: a fork from near-noise
      must recover full independent diversity, or commitment curves are
      kernel-artifacts);
  (b) fork audio validity >= audio_validity_min (finite, non-trivial RMS);
  (c) late anchoring preserved: A_fork(s=0.90) >= 0.90 on both axes.

If no alpha satisfies all three -> FORK_ALPHA_NO_VALID_OPERATING_POINT (manual
section 1.3 routing: kernel redesign or diagnostic framing; NEVER silently raise
alpha beyond the grid).

GPU, an17; ~3 clips x grid x (8 forks at s in {0.05, 0.90}) ~= minutes.
Output: results/stage_m/alpha_pilot.json + chosen alpha printed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from foley_cw import score_sde as K  # noqa: E402
from foley_cw.agreement import agreement, confident_agreement  # noqa: E402
from foley_cw.real_measurer import ABSTAIN  # noqa: E402
from foley_cw.types import AgreementMetric  # noqa: E402
from foley_cw.config import load_config  # noqa: E402
from foley_cw.types import ScheduleSpec  # noqa: E402

PILOT_AXIS_IDS = ("presence", "class")
EARLY_S, LATE_S = 0.05, 0.90
EARLY_GAP_MAX = 0.10
LATE_MIN = 0.90


def rng_for(seed: int, *parts) -> np.random.Generator:
    entropy = [seed] + [zlib.crc32(str(p).encode()) for p in parts]
    return np.random.default_rng(np.random.SeedSequence(entropy))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips-json", type=Path, default=Path("data/manifests/stage_m_clips.json"))
    ap.add_argument("--clips-root", type=Path, default=Path("data/FoleyBench/clips"))
    ap.add_argument("--out", type=Path, default=Path("results/stage_m/alpha_pilot.json"))
    ap.add_argument("--n-clips", type=int, default=3)
    ap.add_argument("--alphas", default=None,
                    help="comma list; default = nonzero pilot grid from configs/alpha_grid.json")
    ap.add_argument("--cfg", type=float, default=4.5)
    ap.add_argument("--k-forks", type=int, default=8)
    ap.add_argument("--n-independent", type=int, default=8)
    ap.add_argument("--num-steps", type=int, default=20)
    ap.add_argument("--g-kind", default="constant", choices=["constant", "linear_down", "sqrt_down"],
                    help="diffusion-scale family g(s); early-heavy schedules (linear_down/"
                         "sqrt_down) are the manual-1.3 kernel-redesign route when constant "
                         "g has no valid alpha")
    ap.add_argument("--duration", type=float, default=8.0)
    ap.add_argument("--variant", default="small_16k")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg_all = load_config()
    axes = [a for a in cfg_all.axes if a.id in PILOT_AXIS_IDS]
    grid_spec = cfg_all.alpha_grid
    alphas = ([float(x) for x in args.alphas.split(",")] if args.alphas
              else [a for a in grid_spec.pilot_grid if a > 0])
    validity_min = grid_spec.audio_validity_min

    clips = json.loads(args.clips_json.read_text())["clips"][: args.n_clips]
    schedule = ScheduleSpec(n_steps=args.num_steps, scan_points=(EARLY_S, LATE_S),
                            K_forks=args.k_forks, N_independent=args.n_independent,
                            g_kind=args.g_kind)
    g = K.make_g(schedule.g_kind, schedule.g_value)

    from foley_cw.mmaudio_backend import MMAudioBackend
    from foley_cw.real_measurer import RealFoleyMeasurer

    backend = MMAudioBackend(variant=args.variant, device=args.device, full_precision=True,
                             cfg_strength=args.cfg, num_steps=args.num_steps,
                             duration_sec=args.duration, enable_conditions=True)
    measurer = RealFoleyMeasurer(device=args.device)
    print(f"[alpha-pilot] backend ready; clips={clips} alphas={alphas} cfg={args.cfg}", flush=True)

    records = []
    for clip in clips:
        t0 = time.time()
        cond = backend.make_video_cond(str(args.clips_root / f"{clip}.mp4"), video_id=clip)
        base = K.generate_trajectory(backend, cond, schedule, rng_for(args.seed, clip, "base"),
                                     alpha=0.0, record_points=(EARLY_S, LATE_S))
        ind_targets = {a.id: [] for a in axes}
        for j in range(args.n_independent):
            tr = K.generate_trajectory(backend, cond, schedule,
                                       rng_for(args.seed, clip, "ind", j), alpha=0.0,
                                       record_points=(EARLY_S, LATE_S))
            for a in axes:
                ind_targets[a.id].append(measurer.measure(tr["audio"], a))
        # confident-subset agreement (abstains excluded; frozen interpretation #3)
        def _agree(targets):
            labels = [t.label for t in targets]
            v, _n = confident_agreement(labels, AgreementMetric.EXACT_MATCH, abstain=ABSTAIN)
            return float(v)
        a_ind = {a.id: _agree(ind_targets[a.id]) for a in axes}

        for alpha in alphas:
            row = {"clip": clip, "alpha": alpha, "a_independent": a_ind}
            for s in (EARLY_S, LATE_S):
                fork_targets = {a.id: [] for a in axes}
                valid = []
                rng_f = rng_for(args.seed, clip, "pilotfork", alpha, s)
                for _k in range(args.k_forks):
                    x1 = K.integrate_segment(backend, base["states"][s], cond, s, 1.0,
                                             schedule, alpha, g, rng_f)
                    audio = backend.decode(x1)
                    rms = float(np.sqrt(np.mean(audio ** 2)))
                    valid.append(bool(np.isfinite(audio).all() and rms > 1e-4))
                    for a in axes:
                        fork_targets[a.id].append(measurer.measure(audio, a))
                row[f"a_fork_s{s:g}"] = {a.id: _agree(fork_targets[a.id]) for a in axes}
                row[f"valid_frac_s{s:g}"] = float(np.mean(valid))
            records.append(row)
            print(f"  {clip} alpha={alpha}: early={row[f'a_fork_s{EARLY_S:g}']} "
                  f"late={row[f'a_fork_s{LATE_S:g}']} A_ind={a_ind} "
                  f"valid={row[f'valid_frac_s{LATE_S:g}']:.2f}", flush=True)
        print(f"[alpha-pilot] {clip} done in {time.time()-t0:.0f}s", flush=True)

    # selection: smallest alpha meeting (a)+(b)+(c) on clip-mean values
    summary = {}
    chosen = None
    for alpha in alphas:
        rows = [r for r in records if r["alpha"] == alpha]
        # NaN-aware (Codex finding): an unscorable confident subset on one clip
        # must not poison selection; require >= 2 scorable clips per quantity.
        gaps = [abs(r[f"a_fork_s{EARLY_S:g}"]["class"] - r["a_independent"]["class"])
                for r in rows]
        gaps = [v for v in gaps if np.isfinite(v)]
        early_gap = float(np.mean(gaps)) if len(gaps) >= 2 else float("nan")
        late_vals = []
        for a in axes:
            vals = [r[f"a_fork_s{LATE_S:g}"][a.id] for r in rows]
            vals = [v for v in vals if np.isfinite(v)]
            late_vals.append(float(np.mean(vals)) if len(vals) >= 2 else float("nan"))
        late_min = min(late_vals)
        valid = float(np.mean([min(r[f"valid_frac_s{EARLY_S:g}"],
                                   r[f"valid_frac_s{LATE_S:g}"]) for r in rows]))
        ok = (np.isfinite(early_gap) and np.isfinite(late_min)
              and early_gap <= EARLY_GAP_MAX and late_min >= LATE_MIN
              and valid >= validity_min)
        summary[str(alpha)] = {"early_gap_class": early_gap, "late_min": late_min,
                               "valid_frac": valid, "ok": bool(ok)}
        if ok and chosen is None:
            chosen = alpha

    token = (f"PRIMARY_ALPHA={chosen}" if chosen is not None
             else "FORK_ALPHA_NO_VALID_OPERATING_POINT")
    payload = {"records": records, "summary": summary, "chosen_alpha": chosen,
               "token": token, "criteria": {"early_gap_max": EARLY_GAP_MAX,
                                            "late_min": LATE_MIN,
                                            "valid_min": validity_min},
               "cfg": args.cfg, "g_kind": args.g_kind,
               "n_clips": len(clips), "k_forks": args.k_forks}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"[alpha-pilot] {token}")
    return 0 if chosen is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
