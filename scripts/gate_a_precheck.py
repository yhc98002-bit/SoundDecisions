#!/usr/bin/env python
"""Gate-A internal-null PRECHECK for a candidate (cfg=1.0, schedule, alpha,
n_steps) tuple — instrument calibration before committing the single remaining
Stage-M attempt (frozen interpretation #10 iteration bound).

Reuses the re-run's stored reference/independent prob vectors (alpha=0
generations are schedule-independent) and regenerates ONLY the seed-marginalized
gate forks under the candidate tuple: per clip, re-derive the 8 independents'
states at the test s-points (deterministic seeds -> identical trajectories),
fork one tail per independent under (schedule, alpha, n_steps), tag the finals,
build cells, and run the internal-null verdict (refined g3).

~330 calls/clip => ~264 FGE per tuple; minutes on one GPU. Diagnostics only.
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
from foley_cw.config import load_config  # noqa: E402
from foley_cw.gate_a import (build_cell, calibrate_from_internal_null,  # noqa: E402
                             evaluate_internal_null, null_sanity, power_positive_control)
from foley_cw.real_measurer import RealFoleyMeasurer  # noqa: E402
from foley_cw.types import ScheduleSpec  # noqa: E402

GATE_S = (0.05, 0.90)


def rng_for(seed: int, *parts) -> np.random.Generator:
    entropy = [seed] + [zlib.crc32(str(p).encode()) for p in parts]
    return np.random.default_rng(np.random.SeedSequence(entropy))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerun-out", type=Path, default=Path("results/stage_m_rerun"))
    ap.add_argument("--clips-json", type=Path, default=Path("data/manifests/stage_m_clips.json"))
    ap.add_argument("--clips-root", type=Path, default=Path("data/FoleyBench/clips"))
    ap.add_argument("--g-kind", required=True, choices=["constant", "linear_down", "sqrt_down"])
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--num-steps", type=int, default=20)
    ap.add_argument("--cfg", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1, help="must match the re-run seed")
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg_all = load_config()
    class_axis = next(a for a in cfg_all.axes if a.id == "class")
    clips = json.loads(args.clips_json.read_text())["clips"]
    schedule = ScheduleSpec(n_steps=args.num_steps, scan_points=GATE_S,
                            K_forks=8, N_independent=8, g_kind=args.g_kind)
    grid = schedule.integration_s_grid()
    for s in GATE_S:
        assert np.any(np.isclose(grid, s, atol=1e-9)), f"s={s} off grid (n={args.num_steps})"
    g = K.make_g(schedule.g_kind, schedule.g_value)

    from foley_cw.mmaudio_backend import MMAudioBackend
    backend = MMAudioBackend(variant="small_16k", device=args.device, full_precision=True,
                             cfg_strength=args.cfg, num_steps=args.num_steps,
                             duration_sec=8.0, enable_conditions=True)
    measurer = RealFoleyMeasurer(device=args.device)
    print(f"[precheck] tuple=(cfg={args.cfg:g}, {args.g_kind}, alpha={args.alpha}, "
          f"n={args.num_steps}); {len(clips)} clips", flush=True)

    cells, ref_probs_by_clip = [], {}
    for clip in clips:
        t0 = time.time()
        journal = json.loads((args.rerun_out / "journal" /
                              f"{clip}__cfg{args.cfg:g}.json").read_text())
        z = np.load(args.rerun_out / "gate_a" / f"{clip}__cfg{args.cfg:g}.npz")
        ref_probs = z["probs_ref"]
        ref_probs_by_clip[clip] = ref_probs
        ref_labels = journal["gate_a_labels"]["ref"]

        cond = backend.make_video_cond(str(args.clips_root / f"{clip}.mp4"), video_id=clip)
        # re-derive independents' states deterministically (same RNG streams as
        # the re-run; alpha=0 trajectories are schedule-independent)
        ind_states = []
        for j in range(8):
            tr = K.generate_trajectory(backend, cond, schedule,
                                       rng_for(args.seed, clip, args.cfg, "ind", j),
                                       alpha=0.0, record_points=GATE_S)
            ind_states.append({s: tr["states"][s] for s in GATE_S})

        for s in GATE_S:
            probs_pool, labels_pool = [], []
            for j, st in enumerate(ind_states):
                rng_j = rng_for(args.seed, clip, args.cfg, "gatea", args.g_kind,
                                args.alpha, j, s)
                x = K.integrate_segment(backend, np.array(st[s], copy=True), cond,
                                        s, 1.0, schedule, args.alpha, g, rng_j)
                audio = backend.decode(x)
                probs, _ = measurer._panns_forward(audio)
                probs_pool.append(probs)
                labels_pool.append(measurer.class_diagnostics(probs)["label"])
            cells.append(build_cell(clip, s, args.cfg, np.stack(probs_pool), ref_probs,
                                    labels_pool, ref_labels,
                                    rng=np.random.default_rng(args.seed),
                                    n_perm=args.n_perm, schedule=args.g_kind))
        print(f"[precheck] {clip} done in {time.time()-t0:.0f}s", flush=True)

    rng = np.random.default_rng(args.seed)
    thresholds = calibrate_from_internal_null(cells)
    power, cross_med, cross_p95 = power_positive_control(ref_probs_by_clip, rng=rng,
                                                         n_perm=args.n_perm)
    ks_p, _ = null_sanity(ref_probs_by_clip, rng=rng, n_perm=args.n_perm)
    res = evaluate_internal_null(cells, thresholds, power_reject_frac=power,
                                 cross_clip_mmd_median=cross_med, null_ks_p=ks_p,
                                 cross_clip_mmd_p95=cross_p95, cfg=args.cfg,
                                 schedule=args.g_kind,
                                 expected_cells_per_s=len(clips))
    payload = {"tuple": {"cfg": args.cfg, "g_kind": args.g_kind, "alpha": args.alpha,
                         "num_steps": args.num_steps},
               "token": res.token, "passed": res.passed, "detail": res.detail,
               "guards": res.guards, "per_s": res.per_s}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, default=str))
    print(json.dumps({"token": res.token, "per_s": res.per_s}, indent=2, default=str))
    return 0 if res.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
