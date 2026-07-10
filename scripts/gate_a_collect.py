#!/usr/bin/env python
"""Full-pool Gate-A collection (manual §1.2/§15.8) — prob-vector bundles for ratification.

The cfg=4.5 deployed kernel is promoted candidate→ratified only after Gate-A passes on the
FULL Phase-1 independent pool (not the Stage-M pilot cells). Gate-A consumes 527-dim PANNs
prob vectors (seed-marginalized exchangeability: one tail-fork per independent at s vs FRESH
reference independents) — which the commitment runner journaled as labels only. This script
collects the prob-vector bundles, mirroring the Stage-M structure exactly so the same
evaluator (foley_cw.gate_a) certifies them:

  per clip × cfg, npz results/stage0/gate_a/<clip>__cfg<g>.npz with keys:
    probs_ind            (n_independent, 527)  the fork-parent independents' final probs
    probs_ref            (n_ref, 527)          FRESH reference independents' final probs
    probs_gafork_s<s>    (n_independent, 527)  one tail-fork per independent at gate_a s
  journal gate_a__<clip>__cfg<g>: {gate_a_labels: {ref, s<s>}, gate_a_s, cfg, schedule}.

Run BOTH cfgs (1.0 = internal-null calibration reference; 4.5 = candidate under test):
  scripts/run_on_node.sh an17 'for i in 0..7; do CUDA_VISIBLE_DEVICES=$i \
    python scripts/gate_a_collect.py --shard $i/16 --cfg 1.0 ... & done; wait'
Forks run under the (cfg, schedule) tuple with --no-require-ratified (candidate collection).
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
from foley_cw.kernel_provenance import assert_certified_kernel  # noqa: E402
from foley_cw.run_store import RunStore  # noqa: E402
from foley_cw.storage_budget import StorageBudget  # noqa: E402
from foley_cw.types import ScheduleSpec  # noqa: E402

GATE_A_S = (0.05, 0.90)
N_INDEPENDENT = 16
N_REF = 16


def rng_for(seed, *parts):
    return np.random.default_rng(np.random.SeedSequence([seed] + [zlib.crc32(str(p).encode()) for p in parts]))


def run_clip(ib, measurer, store, clip, video, schedule, seed, cfg, alpha, tag):
    t0 = time.time(); nfe0 = ib.nfe
    ib.cfg_strength = cfg
    cond = ib.make_video_cond(str(video), video_id=clip)
    g = K.make_g(schedule.g_kind, schedule.g_value)
    rec_pts = tuple(sorted(set(GATE_A_S) | {0.0, 1.0}))

    # fork-parent independents: final probs + states at the Gate-A s-points
    ind_probs, ind_states = [], []
    ref_class = []
    for j in range(N_INDEPENDENT):
        tr = K.generate_trajectory(ib, cond, schedule, rng_for(seed, clip, cfg, "ind", j),
                                   alpha=0.0, record_points=rec_pts)
        ind_probs.append(measurer._panns_forward(tr["audio"])[0])
        ind_states.append({s: tr["states"][s] for s in GATE_A_S})

    # FRESH reference independents (distinct seed stream)
    ref_probs = []
    for j in range(N_REF):
        tr = K.generate_trajectory(ib, cond, schedule, rng_for(seed, clip, cfg, "gateref", j),
                                   alpha=0.0, record_points=(0.0, 1.0))
        p = measurer._panns_forward(tr["audio"])[0]
        ref_probs.append(p)
        ref_class.append(measurer._coarse_from_probs(p))

    # one tail-fork per independent at each Gate-A s (seed-marginalized pool)
    ga_probs, ga_labels = {}, {}
    for s in GATE_A_S:
        probs_pool, labels_pool = [], []
        for j, st in enumerate(ind_states):
            x = K.integrate_segment(ib, np.array(st[s], copy=True), cond, s, 1.0,
                                    schedule, alpha, g, rng_for(seed, clip, cfg, "gatea", j, s))
            audio = ib.decode(x)
            p = measurer._panns_forward(audio)[0]
            probs_pool.append(p); labels_pool.append(measurer._coarse_from_probs(p))
        ga_probs[s] = np.stack(probs_pool); ga_labels[f"s{s:g}"] = labels_pool

    store.put_npz("gate_a", f"{clip}__cfg{cfg:g}",
                  probs_ind=np.stack(ind_probs).astype(np.float32),
                  probs_ref=np.stack(ref_probs).astype(np.float32),
                  **{f"probs_gafork_s{s:g}": ga_probs[s].astype(np.float32) for s in GATE_A_S})
    nfe = ib.nfe - nfe0
    print(f"[gatea {clip} cfg{cfg:g}] {time.time()-t0:.0f}s nfe={nfe}", flush=True)
    return {"clip": clip, "cfg": cfg, "schedule": schedule.g_kind, "gate_a_s": list(GATE_A_S),
            "gate_a_labels": {"ref": ref_class, **ga_labels}, "nfe_velocity_calls": int(nfe)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=Path("data/manifests/phase1_manifest_frozen.json"))
    ap.add_argument("--clips-root", type=Path, default=Path("data/FoleyBench/clips"))
    ap.add_argument("--out", type=Path, default=Path("results/stage0"))
    ap.add_argument("--certified", type=Path, default=Path("results/stage_m_rerun/certified_kernels.json"))
    ap.add_argument("--cfg", type=float, default=4.5)
    ap.add_argument("--schedule", default="sqrt_down")
    ap.add_argument("--alpha", type=float, default=0.8)
    ap.add_argument("--variant", default="small_16k")
    ap.add_argument("--num-steps", type=int, default=20)
    ap.add_argument("--duration", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--require-ratified", action=argparse.BooleanOptionalAction, default=False)
    args = ap.parse_args()

    cert = assert_certified_kernel(args.cfg, args.schedule, args.certified,
                                   require_ratified=args.require_ratified)
    print(f"[gatea] kernel {cert['token']} (ratified={cert['ratified']})", flush=True)

    clips = sorted(str(c) for c in json.loads(args.manifest.read_text())["clips"]["single_event"])
    shard_i, shard_n = (int(x) for x in args.shard.split("/"))
    schedule = ScheduleSpec(n_steps=args.num_steps, scan_points=GATE_A_S, K_forks=1,
                            N_independent=N_INDEPENDENT, g_kind=args.schedule, g_value=1.0)
    store = RunStore(args.out, budget=StorageBudget(cap_gb=100.0))
    store.account_preexisting_tree()
    tag = f"gate_a__cfg{args.cfg:g}"
    todo = [c for i, c in enumerate(clips) if i % shard_n == shard_i]
    if args.limit:
        todo = todo[: args.limit]
    todo = [c for c in todo if not store.is_done(f"{tag}__{c}")]
    print(f"[gatea] shard {args.shard}: {len(todo)} clips (cfg={args.cfg})", flush=True)
    if not todo:
        return 0

    from foley_cw.feature_tap import InstrumentedBackend
    from foley_cw.mmaudio_backend import MMAudioBackend
    from foley_cw.real_measurer import RealFoleyMeasurer
    backend = MMAudioBackend(variant=args.variant, device=args.device, full_precision=True,
                             cfg_strength=args.cfg, num_steps=args.num_steps,
                             duration_sec=args.duration, enable_conditions=True)
    ib = InstrumentedBackend(backend)
    measurer = RealFoleyMeasurer(device=args.device)
    for clip in todo:
        payload = run_clip(ib, measurer, store, clip, args.clips_root / f"{clip}.mp4",
                           schedule, args.seed, args.cfg, args.alpha, tag)
        store.journal_done(f"{tag}__{clip}", payload)
    print(f"[gatea] shard {args.shard} complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
