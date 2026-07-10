#!/usr/bin/env python
"""Arc-3 Tier-B §B3 — seed-floor direct test (CPU dial part + sharded GPU collect).

Frozen plan: `experiment/preregistered/arc3_tierB_preregistration.md` §B3. Probe the s=0
noise seed → final class on the frozen 60/40 clip split, reduced to d=256 via a FIXED
Gaussian random projection (seed 0; JL), with ridge + MLP probes, metric = eval accuracy
vs chance, trend = OLS slope of (acc − chance) vs cfg with a clip-level bootstrap CI.

Two modes:

  (DEFAULT, CPU, RUN NOW) — the cfg-DIAL part:
    Loads the cached `dial_noise__dial_cfg<C>__<clip>.npz` bundles (noise (16,5000), labels)
    for cfg ∈ {1,1.5,2,2.5,3,4.5}, projects to d=256 with the frozen JL matrix, fits the
    ridge + MLP probes on the frozen `probe_train` clips and evaluates on the frozen `eval`
    clips, computes acc vs chance per cfg, the OLS slope + bootstrap CI, and emits the §B3
    suggested token. Writes results/stage0/arc3/b3_seed_floor_dial.json.

  (--collect, GPU, sharded, BUILD-ONLY — do NOT run here):
    The full cfg=1.0 pool (200 single-event clips) needs the s=0 prior latents, which are
    NOT cached (only pooled features are). This mode mirrors scripts/phase1_commitment.py:
    --shard i/n, rng_for, journaled/resumable, kernel-guarded via assert_certified_kernel.
    Per clip it runs N_independent ODE generations (alpha=0), captures each INITIAL s=0
    prior latent (the probe input) + measures the final class (the probe target), and stores
    one `seedfloor_noise__<tag>__<clip>.npz` bundle (noise (N,d_in), labels) the SAME way
    stage_r_cfg_dial stored its dial bundles. The orchestrator runs this on an17/an29; the
    --collect-aggregate step then re-uses probe_cfg on the frozen split exactly as the dial
    part does. This script does NOT launch GPU work; the GPU branch only runs when an
    operator invokes `--collect` on a node.

Run (CPU dial part, NOW):
  .venv/bin/python scripts/b3_seed_floor.py
Build the full-pool latents (GPU; orchestrator only — do NOT run from this agent):
  scripts/run_on_node.sh an17 'for i in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$i \
    python scripts/b3_seed_floor.py --collect --shard $i/8 > logs/b3_collect_$i.log 2>&1 & done; wait'
  .venv/bin/python scripts/b3_seed_floor.py --collect-aggregate
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

from foley_cw import seed_floor as SF  # noqa: E402

CLASS_AXIS = "class"
DIAL_CFGS = (1.0, 1.5, 2.0, 2.5, 3.0, 4.5)
PHASE1_S_GRID = (0.05, 0.15, 0.25, 0.35, 0.45, 0.60, 0.75, 0.90)
S_MIN = PHASE1_S_GRID[0]
N_INDEPENDENT = 16


def rng_for(seed: int, *parts) -> np.random.Generator:
    entropy = [seed] + [zlib.crc32(str(p).encode()) for p in parts]
    return np.random.default_rng(np.random.SeedSequence(entropy))


# ---------------------------------------------------------------------------
# CPU dial part (RUN NOW) — cached dial noise bundles, frozen split.
# ---------------------------------------------------------------------------
def _load_dial_bundle(gate_a: Path, tag: str, clip: str):
    p = gate_a / f"dial_noise__{tag}__{clip}.npz"
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=True)
    return np.asarray(z["noise"], dtype=np.float64), [str(x) for x in z["labels"].tolist()]


def run_dial(out: Path, manifest: Path, n_boot: int, lam: float, seed: int) -> dict:
    man = json.loads(manifest.read_text())
    train = [str(c) for c in man["split_60_40_by_clip"]["probe_train"]]
    eval_ = [str(c) for c in man["split_60_40_by_clip"]["eval"]]
    gate_a = out / "gate_a"

    # frozen JL matrix (regenerated deterministically; same for every cfg)
    R = SF.gaussian_projection(5000, SF.RP_DIM, SF.RP_SEED)

    results: dict[float, SF.CfgResult] = {}
    per_cfg_json = {}
    dial_clips_present = set()
    for cfg in DIAL_CFGS:
        tag = f"dial_cfg{cfg:g}"
        noise_by_clip, labels_by_clip = {}, {}
        for clip in sorted(set(train) | set(eval_)):
            b = _load_dial_bundle(gate_a, tag, clip)
            if b is None:
                continue
            noise_by_clip[clip], labels_by_clip[clip] = b
            dial_clips_present.add(clip)
        tr = [c for c in train if c in noise_by_clip]
        ev = [c for c in eval_ if c in noise_by_clip]
        r = SF.probe_cfg(noise_by_clip, labels_by_clip, tr, ev, R=R, lam=lam,
                         mlp_seed=seed, cfg=cfg)
        results[cfg] = r
        per_cfg_json[f"{cfg:g}"] = {
            "n_train_clips": len(tr), "n_eval_clips": len(ev),
            "n_train_rows": r.n_train, "n_eval_rows": r.n_eval,
            "chance": round(r.chance, 4), "ridge_acc": round(r.ridge_acc, 4),
            "mlp_acc": round(r.mlp_acc, 4), "best_acc": round(r.best_acc, 4),
            "best_probe": r.best_probe,
            "ridge_margin": round(r.ridge_acc - r.chance, 4),
            "mlp_margin": round(r.mlp_acc - r.chance, 4),
            "best_margin": round(r.best_acc - r.chance, 4),
        }
        print(f"[b3 dial cfg={cfg:g}] ntr={r.n_train} nev={r.n_eval} chance={r.chance:.3f} "
              f"ridge={r.ridge_acc:.3f} mlp={r.mlp_acc:.3f} best={r.best_acc:.3f}"
              f" ({r.best_probe}) margin={r.best_acc - r.chance:+.3f}", flush=True)

    slope_best = SF.bootstrap_slope_ci(results, n_boot=n_boot, seed=seed, probe="best")
    slope_ridge = SF.bootstrap_slope_ci(results, n_boot=n_boot, seed=seed, probe="ridge")
    decision = SF.decide(results, slope_best)

    payload = {
        "section": "B3_seed_floor_dial",
        "preregistration": "experiment/preregistered/arc3_tierB_preregistration.md#B3",
        "pool": "cfg-dial clips (frozen 60/40 split)",
        "n_dial_clips_present": len(dial_clips_present),
        "rp_dim": SF.RP_DIM, "rp_seed": SF.RP_SEED, "lam": lam, "n_boot": n_boot,
        "cfgs": list(DIAL_CFGS),
        "per_cfg": per_cfg_json,
        "slope_best_acc_minus_chance_vs_cfg": slope_best,
        "slope_ridge_acc_minus_chance_vs_cfg": slope_ridge,
        "decision": decision,
        "suggested_token": decision["suggested_token"],
        "notes": (
            "Bootstrap unit = video (eval clips resampled, chance recomputed per draw). "
            "Abstain rows dropped (confident class subset). Probes trained ONLY on "
            "probe_train clips, evaluated ONLY on eval clips (no clip in both). "
            "Full-pool cfg=1.0 part requires the 200-clip s=0 latents via --collect "
            "(GPU, not run here)."),
    }
    out_dir = out / "arc3"; out_dir.mkdir(parents=True, exist_ok=True)
    res_path = out_dir / "b3_seed_floor_dial.json"
    res_path.write_text(json.dumps(payload, indent=2, default=lambda o: (
        None if (isinstance(o, float) and not np.isfinite(o)) else
        float(o) if isinstance(o, np.floating) else str(o))))
    print(f"[b3 dial] suggested={decision['suggested_token']} "
          f"slope_best={slope_best['point']:+.4f} "
          f"CI=[{slope_best['lo']},{slope_best['hi']}] → {res_path}", flush=True)
    return payload


# ---------------------------------------------------------------------------
# GPU collect mode (BUILD-ONLY) — full-pool s=0 latents per clip.
# ---------------------------------------------------------------------------
def collect_clip(ib, measurer, class_axis, store, clip: str, video: Path, schedule,
                 seed: int, cfg: float, tag: str, n_independent: int) -> dict:
    """Run N independents (alpha=0); store each INITIAL s=0 prior latent + final class.

    Mirrors stage_r_cfg_dial.run_clip_cfg's seed-predictability collection: the probe input
    is the flattened s=0 prior latent, the target is the measured final class. The bundle is
    written via RunStore.put_npz (budget-accounted), keyed `seedfloor_noise__<tag>__<clip>`.
    """
    from foley_cw import score_sde as K
    t0 = time.time(); nfe0 = getattr(ib, "nfe", 0)
    ib.cfg_strength = cfg
    cond = ib.make_video_cond(str(video), video_id=clip)
    noise_rows, label_rows = [], []
    for j in range(n_independent):
        tr = K.generate_trajectory(ib, cond, schedule, rng_for(seed, clip, cfg, "ind", j),
                                   alpha=0.0, record_points=(S_MIN,))
        x0_noise = np.asarray(tr["states"][0.0], dtype=np.float32).ravel()    # s=0 prior latent
        tgt = measurer.measure(tr["audio"], class_axis)
        noise_rows.append(x0_noise)
        label_rows.append(str(tgt.label))
        store.record_measurement(f"{clip}__{tag}_ind{j}", CLASS_AXIS, tgt,
                                 extra={"role": f"{tag}_independent", "j": j,
                                        "clip": clip, "cfg": cfg, "s": 0.0})
    store.put_npz("arc3", f"seedfloor_noise__{tag}__{clip}",
                  noise=np.stack(noise_rows, axis=0),
                  labels=np.array(label_rows, dtype=object))
    nfe = getattr(ib, "nfe", 0) - nfe0
    elapsed = time.time() - t0
    print(f"[b3 collect {clip} cfg={cfg:g}] {elapsed:.0f}s nfe={nfe} "
          f"n_indep={n_independent} dim={noise_rows[0].shape[0]}", flush=True)
    return {"clip": clip, "cfg": cfg, "n_independent": n_independent,
            "noise_dim": int(noise_rows[0].shape[0]),
            "elapsed_s": round(elapsed, 1), "nfe_velocity_calls": int(nfe)}


def run_collect(args, clips: list[str]) -> int:
    from foley_cw.kernel_provenance import assert_certified_kernel
    from foley_cw.run_store import RunStore
    from foley_cw.storage_budget import StorageBudget
    from foley_cw.types import ScheduleSpec

    # Provenance guard (§15.8): the full-pool seed-floor latents are collected under the
    # SAME certified (cfg, schedule) tuple as the headline grid (cfg=1.0 is ratified).
    cert = assert_certified_kernel(args.cfg, args.schedule, args.certified,
                                   require_ratified=args.require_ratified)
    print(f"[b3 collect] kernel OK: {cert['token']} (ratified={cert['ratified']})", flush=True)

    shard_i, shard_n = (int(x) for x in args.shard.split("/"))
    schedule = ScheduleSpec(n_steps=args.num_steps, scan_points=PHASE1_S_GRID,
                            K_forks=1, N_independent=args.n_independent,
                            g_kind=args.schedule, g_value=1.0)
    grid = schedule.integration_s_grid()
    assert np.any(np.isclose(grid, S_MIN, atol=1e-9)), f"s_min={S_MIN} off integration grid"

    tag = args.tag or f"seedfloor_cfg{args.cfg:g}"
    budget = StorageBudget(cap_gb=100.0)
    store = RunStore(args.out, budget=budget)
    store.account_preexisting_tree()
    todo = [c for i, c in enumerate(clips) if i % shard_n == shard_i]
    if args.limit:
        todo = todo[: args.limit]
    todo = [c for c in todo if not store.is_done(f"{tag}__{c}")]
    print(f"[b3 collect] shard {args.shard}: {len(todo)} clips (tag={tag}, cfg={args.cfg})",
          flush=True)
    if not todo:
        return 0

    from foley_cw.config import load_config
    from foley_cw.feature_tap import InstrumentedBackend
    from foley_cw.mmaudio_backend import MMAudioBackend
    from foley_cw.real_measurer import RealFoleyMeasurer

    class_axis = next(a for a in load_config().axes if a.id == CLASS_AXIS)
    backend = MMAudioBackend(variant=args.variant, device=args.device, full_precision=True,
                             cfg_strength=args.cfg, num_steps=args.num_steps,
                             duration_sec=args.duration, enable_conditions=True)
    ib = InstrumentedBackend(backend)
    measurer = RealFoleyMeasurer(device=args.device)
    for clip in todo:
        payload = collect_clip(ib, measurer, class_axis, store, clip,
                               args.clips_root / f"{clip}.mp4", schedule, args.seed,
                               cfg=args.cfg, tag=tag, n_independent=args.n_independent)
        payload["budget"] = budget.summary()
        store.journal_done(f"{tag}__{clip}", payload)
    print(f"[b3 collect] shard {args.shard} complete; budget: {budget.summary()}", flush=True)
    return 0


def run_collect_aggregate(args, clips: list[str]) -> int:
    """CPU: probe the full-pool collected latents on the frozen split (cfg=1.0 part)."""
    from foley_cw.run_store import RunStore
    man = json.loads(args.manifest.read_text())
    train = [str(c) for c in man["split_60_40_by_clip"]["probe_train"]]
    eval_ = [str(c) for c in man["split_60_40_by_clip"]["eval"]]
    tag = args.tag or f"seedfloor_cfg{args.cfg:g}"
    store = RunStore(args.out)
    arc3 = args.out / "arc3"
    noise_by_clip, labels_by_clip, missing = {}, {}, []
    d_in = None
    for clip in clips:
        if not store.is_done(f"{tag}__{clip}"):
            missing.append(clip); continue
        p = arc3 / f"seedfloor_noise__{tag}__{clip}.npz"
        if not p.exists():
            missing.append(clip); continue
        z = np.load(p, allow_pickle=True)
        noise_by_clip[clip] = np.asarray(z["noise"], dtype=np.float64)
        labels_by_clip[clip] = [str(x) for x in z["labels"].tolist()]
        d_in = noise_by_clip[clip].shape[1]
    if d_in is None:
        print("[b3 collect-aggregate] no collected latents yet; run --collect first")
        return 1
    R = SF.gaussian_projection(d_in, SF.RP_DIM, SF.RP_SEED)
    tr = [c for c in train if c in noise_by_clip]
    ev = [c for c in eval_ if c in noise_by_clip]
    r = SF.probe_cfg(noise_by_clip, labels_by_clip, tr, ev, R=R, lam=args.lam,
                     mlp_seed=args.seed, cfg=args.cfg)
    seed_floor = bool(np.isfinite(r.best_acc) and np.isfinite(r.chance)
                      and (r.best_acc - r.chance) > SF.SEED_FLOOR_MARGIN)
    payload = {
        "section": "B3_seed_floor_fullpool", "cfg": args.cfg, "noise_dim_in": int(d_in),
        "rp_dim": SF.RP_DIM, "n_train_clips": len(tr), "n_eval_clips": len(ev),
        "n_train_rows": r.n_train, "n_eval_rows": r.n_eval,
        "chance": round(r.chance, 4), "ridge_acc": round(r.ridge_acc, 4),
        "mlp_acc": round(r.mlp_acc, 4), "best_acc": round(r.best_acc, 4),
        "best_probe": r.best_probe, "best_margin": round(r.best_acc - r.chance, 4),
        "seed_floor_confirmed": seed_floor,
        "suggested_token": "SEED_FLOOR_CONFIRMED" if seed_floor else "NO_SEED_FLOOR",
        "n_missing_clips": len(missing),
    }
    arc3.mkdir(parents=True, exist_ok=True)
    res_path = arc3 / "b3_seed_floor_fullpool.json"
    res_path.write_text(json.dumps(payload, indent=2, default=lambda o: (
        None if (isinstance(o, float) and not np.isfinite(o)) else str(o))))
    print(f"[b3 collect-aggregate] cfg={args.cfg:g} best={r.best_acc:.3f} chance={r.chance:.3f}"
          f" seed_floor={seed_floor} (missing {len(missing)}) → {res_path}")
    return 0 if not missing else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path,
                    default=Path("data/manifests/phase1_manifest_frozen.json"))
    ap.add_argument("--clips-root", type=Path, default=Path("data/FoleyBench/clips"))
    ap.add_argument("--out", type=Path, default=Path("results/stage0"))
    ap.add_argument("--certified", type=Path,
                    default=Path("results/stage_m_rerun/certified_kernels.json"))
    ap.add_argument("--cfg", type=float, default=1.0)
    ap.add_argument("--schedule", default="sqrt_down")
    ap.add_argument("--variant", default="small_16k")
    ap.add_argument("--duration", type=float, default=8.0)
    ap.add_argument("--num-steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-independent", type=int, default=N_INDEPENDENT)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default=None, help="default seedfloor_cfg<cfg> (collect modes)")
    ap.add_argument("--lam", type=float, default=1.0, help="ridge regularization")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--require-ratified", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--collect", action="store_true",
                    help="GPU: regenerate+store the s=0 prior latent per clip (orchestrator)")
    ap.add_argument("--collect-aggregate", action="store_true",
                    help="CPU: probe the collected full-pool latents on the frozen split")
    args = ap.parse_args()

    if args.collect or args.collect_aggregate:
        man = json.loads(args.manifest.read_text())
        clips = sorted(str(c) for c in man["clips"]["single_event"])
        if args.collect_aggregate:
            return run_collect_aggregate(args, clips)
        return run_collect(args, clips)

    # DEFAULT: CPU dial part (RUN NOW).
    run_dial(args.out, args.manifest, args.n_boot, args.lam, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
