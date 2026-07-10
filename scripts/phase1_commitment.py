#!/usr/bin/env python
"""Phase 1 — Commitment maps (manual §4). The REAL GPU runner.

Per clip (journal unit), on the frozen Phase-1 manifest, under a CERTIFIED (cfg, schedule)
tuple (assert_certified_kernel guards — cfg=4.5 is refused until its full-pool Gate-A
ratifies it, §1.2/§15.8):
  * N_independent = 16 independent ODE generations (alpha=0); features + x̂0(s) previews
    stored at the Phase-1 s-grid; finals measured (4 axes) → A_independent per axis.
  * 1 base ODE trajectory (the fork source); features + previews stored.
  * at each s in the grid: K = 12 marginal-preserving tail-forks from the base state x_s
    under (alpha=0.8, sqrt_down); fork finals measured → A_fork(s) per axis; 10% wav audit.
  * commit_gain(s) = clip((A_fork − A_independent)/(1 − A_independent), 0, 1) [label axes];
    embedding axes carry A_fork(s_min) for the trajectory-share normalization (aggregate).

Reuses the proven stage0_screening pattern (InstrumentedBackend + RealFoleyMeasurer +
RunStore + StorageBudget, --shard i/n over an17/an29). The determination budget + Fig-1
taxonomy are built by the --aggregate step via foley_cw.determination.

Self-target axes (det+rob) are mapped; class is mapped as a DIAGNOSTIC axis (kept, not
gated) per §3.3 — axis selection here is NOT the old combined gate.

Run sharded:
  scripts/run_on_node.sh an17 'for i in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$i \
    python scripts/phase1_commitment.py --shard $i/8 > logs/p1_$i.log 2>&1 & done; wait'
Aggregate (CPU): python scripts/phase1_commitment.py --aggregate
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from foley_cw import score_sde as K  # noqa: E402
from foley_cw.agreement import agreement, confident_agreement  # noqa: E402
from foley_cw.config import load_config  # noqa: E402
from foley_cw.kernel_provenance import assert_certified_kernel  # noqa: E402
from foley_cw.real_measurer import ABSTAIN  # noqa: E402
from foley_cw.run_store import RunStore  # noqa: E402
from foley_cw.storage_budget import StorageBudget  # noqa: E402
from foley_cw.types import AgreementMetric, AxisKind, ScheduleSpec  # noqa: E402

AXIS_IDS = ("presence", "timing", "class", "material")
PHASE1_S_GRID = (0.05, 0.15, 0.25, 0.35, 0.45, 0.60, 0.75, 0.90)
N_INDEPENDENT = 16
K_FORKS = 12


def rng_for(seed: int, *parts) -> np.random.Generator:
    entropy = [seed] + [zlib.crc32(str(p).encode()) for p in parts]
    return np.random.default_rng(np.random.SeedSequence(entropy))


def _axis_agreement(axis, targets) -> tuple[float, int, float]:
    """(agreement, n_confident, abstain_rate) — confident-subset for categorical."""
    if axis.kind is AxisKind.CATEGORICAL:
        labels = [t.label for t in targets]
        v, n = confident_agreement(labels, AgreementMetric.EXACT_MATCH, abstain=ABSTAIN)
        ar = sum(1 for l in labels if l == ABSTAIN) / max(len(labels), 1)
        return float(v), int(n), ar
    return float(agreement(targets, axis.agreement)), len(targets), 0.0


def run_clip(ib, measurer, axes, store: RunStore, clip: str, video: Path,
             schedule: ScheduleSpec, seed: int, cfg: float, alpha: float,
             tag: str, n_independent: int = N_INDEPENDENT, k_forks: int = K_FORKS,
             audit_frac: float = 0.10, gate_a_forks: bool = False) -> dict:
    t0 = time.time(); nfe0 = ib.nfe
    ib.cfg_strength = cfg
    cond = ib.make_video_cond(str(video), video_id=clip)
    g = K.make_g(schedule.g_kind, schedule.g_value)

    # --- N independents (A_independent); optionally one Gate-A tail-fork per independent ---
    ind_targets = {a.id: [] for a in axes}
    for j in range(n_independent):
        gid = f"{clip}__{tag}_ind{j}"
        tr = K.generate_trajectory(ib, cond, schedule, rng_for(seed, clip, "ind", j),
                                   alpha=0.0, record_points=PHASE1_S_GRID)
        for s in PHASE1_S_GRID:
            feats, v = ib.tap_features_at(tr["states"][s], s, cond)
            store.put_features(gid, s, feats)
            t = ib.s_to_t.s_to_t(s)
            store.put_preview(gid, s, ib.decode(K.tweedie_x0(v, tr["states"][s], t)))
        store.put_final_wav(gid, tr["audio"])
        for a in axes:
            tgt = measurer.measure(tr["audio"], a)
            ind_targets[a.id].append(tgt)
            store.record_measurement(gid, a.id, tgt,
                                     extra={"role": f"{tag}_independent", "j": j, "clip": clip,
                                            "cfg": cfg})
        if gate_a_forks:
            # ONE marginal-preserving tail-fork per independent at each s (§1.2 Gate-A:
            # exchangeability of fork-from-independent vs fresh independents).
            for s in PHASE1_S_GRID:
                fa = K.fork_tail(ib, tr["states"][s], s, cond, alpha, 1, schedule,
                                 rng_for(seed, clip, "gaf", j, s), g=g)[0]
                for a in axes:
                    store.record_measurement(f"{clip}__{tag}_gaf_j{j}_s{s:.2f}", a.id,
                                             measurer.measure(fa, a),
                                             extra={"role": f"{tag}_gate_a_fork", "j": j,
                                                    "s": s, "clip": clip, "cfg": cfg})

    a_ind, n_conf, abstain = {}, {}, {}
    for a in axes:
        a_ind[a.id], n_conf[a.id], abstain[a.id] = _axis_agreement(a, ind_targets[a.id])

    # --- base trajectory (fork source) ---
    base_gid = f"{clip}__{tag}_base"
    base = K.generate_trajectory(ib, cond, schedule, rng_for(seed, clip, "base"),
                                 alpha=0.0, record_points=PHASE1_S_GRID)
    for s in PHASE1_S_GRID:
        feats, v = ib.tap_features_at(base["states"][s], s, cond)
        store.put_features(base_gid, s, feats)
        store.put_preview(base_gid, s, ib.decode(K.tweedie_x0(v, base["states"][s],
                                                              ib.s_to_t.s_to_t(s))))
    store.put_final_wav(base_gid, base["audio"])

    # --- K forks from the base state at each s (A_fork(s)) ---
    a_fork = {a.id: {} for a in axes}
    commit = {a.id: {} for a in axes}
    g = K.make_g(schedule.g_kind, schedule.g_value)
    audit_rng = rng_for(seed, clip, "audit")
    for s in PHASE1_S_GRID:
        audios = K.fork_tail(ib, base["states"][s], s, cond, alpha, k_forks, schedule,
                             rng_for(seed, clip, "fork", s), g=g)
        fork_targets = {a.id: [] for a in axes}
        for k, audio in enumerate(audios):
            fgid = f"{clip}__{tag}_fork_s{s:.2f}_k{k}"
            if audit_rng.random() < audit_frac:
                store.put_final_wav(fgid, audio)        # 10% wav audit sample (§1.4)
            for a in axes:
                tgt = measurer.measure(audio, a)
                fork_targets[a.id].append(tgt)
                store.record_measurement(fgid, a.id, tgt,
                                         extra={"role": f"{tag}_fork", "s": s, "k": k,
                                                "clip": clip, "cfg": cfg})
        for a in axes:
            af, _n, _ar = _axis_agreement(a, fork_targets[a.id])
            a_fork[a.id][f"{s:.2f}"] = af
            # commit_gain only meaningful for label axes here; embedding axes use the
            # trajectory-share normalization at aggregate time (needs A_fork(s_min)).
            if a.kind is AxisKind.CATEGORICAL and np.isfinite(af) and np.isfinite(a_ind[a.id]):
                denom = 1.0 - a_ind[a.id]
                commit[a.id][f"{s:.2f}"] = float(np.clip((af - a_ind[a.id]) / denom, 0.0, 1.0)) \
                    if denom > 1e-9 else 0.0

    nfe = ib.nfe - nfe0
    elapsed = time.time() - t0
    print(f"[p1 {clip}] {elapsed:.0f}s nfe={nfe} A_ind="
          f"{ {k: round(v, 3) for k, v in a_ind.items()} }", flush=True)
    return {"clip": clip, "cfg": cfg, "alpha": alpha, "schedule": schedule.g_kind,
            "a_independent": a_ind, "a_fork": a_fork, "commit_gain": commit,
            "n_conf": n_conf, "abstain": abstain,
            "elapsed_s": round(elapsed, 1), "nfe_velocity_calls": int(nfe)}


def aggregate_raw(out: Path, clips: list[str], tag: str) -> int:
    """Flatten journals → commitment_map.csv (per-clip A_fork/A_independent/commit_gain).
    The determination budget + taxonomy are built by foley_cw.determination from this."""
    store = RunStore(out)
    rows, missing = [], []
    for clip in clips:
        unit = f"{tag}__{clip}"
        if not store.is_done(unit):
            missing.append(clip); continue
        d = store.load_journal(unit)
        for ax in AXIS_IDS:
            aind = d["a_independent"].get(ax)
            for s in PHASE1_S_GRID:
                sk = f"{s:.2f}"
                rows.append({
                    "clip": clip, "cfg": d["cfg"], "alpha": d["alpha"],
                    "schedule": d["schedule"], "axis_id": ax, "s": s,
                    "a_independent": aind, "a_fork": d["a_fork"].get(ax, {}).get(sk),
                    "commit_gain": d.get("commit_gain", {}).get(ax, {}).get(sk),
                })
    out_dir = out / "phase1"; out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"commitment_map_{tag}.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["clip", "cfg", "alpha", "schedule", "axis_id", "s",
                                          "a_independent", "a_fork", "commit_gain"])
        w.writeheader(); w.writerows(rows)
    print(f"[aggregate] {len(rows)} rows → {csv_path}; missing {len(missing)} clips "
          f"(first {missing[:5]})")
    return 0 if not missing else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=Path("data/manifests/phase1_manifest_frozen.json"))
    ap.add_argument("--clips-root", type=Path, default=Path("data/FoleyBench/clips"))
    ap.add_argument("--out", type=Path, default=Path("results/stage0"))
    ap.add_argument("--certified", type=Path, default=Path("results/stage_m_rerun/certified_kernels.json"))
    ap.add_argument("--cfg", type=float, default=1.0)
    ap.add_argument("--schedule", default="sqrt_down")
    ap.add_argument("--alpha", type=float, default=0.8)
    ap.add_argument("--variant", default="small_16k")
    ap.add_argument("--duration", type=float, default=8.0)
    ap.add_argument("--num-steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-independent", type=int, default=N_INDEPENDENT)
    ap.add_argument("--k-forks", type=int, default=K_FORKS)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--clip-set", default="single_event",
                    choices=["single_event", "two_event", "both"])
    ap.add_argument("--tag", default=None, help="default p1cfg<cfg>")
    ap.add_argument("--require-ratified", action=argparse.BooleanOptionalAction, default=True,
                    help="--no-require-ratified for the cfg=4.5 full-pool Gate-A ratification "
                         "run itself (candidate kernel collection); commitment grids keep True")
    ap.add_argument("--gate-a-forks", action="store_true",
                    help="also collect ONE tail-fork per independent at each s (Gate-A data)")
    ap.add_argument("--aggregate", action="store_true")
    args = ap.parse_args()

    tag = args.tag or f"p1cfg{args.cfg:g}"
    man = json.loads(args.manifest.read_text())
    if args.clip_set == "both":
        clips = sorted(set(man["clips"]["single_event"]) | set(man["clips"]["two_event"]))
    else:
        clips = sorted(str(c) for c in man["clips"][args.clip_set])

    if args.aggregate:
        return aggregate_raw(args.out, clips, tag)

    # Provenance guard (§15.8): refuse an uncertified or candidate-only (cfg, schedule).
    cert = assert_certified_kernel(args.cfg, args.schedule, args.certified,
                                   require_ratified=args.require_ratified)
    print(f"[p1] kernel OK: {cert['token']} (ratified={cert['ratified']}, "
          f"require_ratified={args.require_ratified}, gate_a_forks={args.gate_a_forks})", flush=True)

    shard_i, shard_n = (int(x) for x in args.shard.split("/"))
    schedule = ScheduleSpec(n_steps=args.num_steps, scan_points=PHASE1_S_GRID,
                            K_forks=args.k_forks, N_independent=args.n_independent,
                            g_kind=args.schedule, g_value=1.0)
    grid = schedule.integration_s_grid()
    for s in PHASE1_S_GRID:
        assert np.any(np.isclose(grid, s, atol=1e-9)), f"s={s} off integration grid"

    budget = StorageBudget(cap_gb=100.0)
    store = RunStore(args.out, budget=budget)
    store.account_preexisting_tree()

    todo = [c for i, c in enumerate(clips) if i % shard_n == shard_i]
    if args.limit:
        todo = todo[: args.limit]
    todo = [c for c in todo if not store.is_done(f"{tag}__{c}")]
    print(f"[p1] shard {args.shard}: {len(todo)} clips (tag={tag}, cfg={args.cfg}, "
          f"alpha={args.alpha}, schedule={args.schedule})", flush=True)
    if not todo:
        return 0

    from foley_cw.feature_tap import InstrumentedBackend
    from foley_cw.mmaudio_backend import MMAudioBackend
    from foley_cw.real_measurer import RealFoleyMeasurer

    axes = [a for a in load_config().axes if a.id in AXIS_IDS]
    backend = MMAudioBackend(variant=args.variant, device=args.device, full_precision=True,
                             cfg_strength=args.cfg, num_steps=args.num_steps,
                             duration_sec=args.duration, enable_conditions=True)
    ib = InstrumentedBackend(backend)
    measurer = RealFoleyMeasurer(device=args.device)

    for clip in todo:
        payload = run_clip(ib, measurer, axes, store, clip,
                           args.clips_root / f"{clip}.mp4", schedule, args.seed,
                           cfg=args.cfg, alpha=args.alpha, tag=tag,
                           n_independent=args.n_independent, k_forks=args.k_forks,
                           gate_a_forks=args.gate_a_forks)
        payload["budget"] = budget.summary()
        store.journal_done(f"{tag}__{clip}", payload)
    print(f"[p1] shard {args.shard} complete; budget: {budget.summary()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
