#!/usr/bin/env python
"""Stage-0 A_independent screening (manual section 3.1) — ~400 candidate clips.

Per clip (journal unit = clip): N=8 independent generations at cfg=4.5, alpha=0;
per generation: grid features + Tweedie previews recorded at the PHASE-1 s-grid
(so these become the first 8 of Phase 1's N=16 independents and the Phase-4
candidate-pool seed), final wav stored, all four axes measured. Per-axis
A_independent = agreement over the 8 finals.

--aggregate builds results/stage0/screening/a_independent.csv and the per-axis
video_determined_registry.json (A_independent > 0.9 exclusions, manual 3.1).

Run sharded across an12 + an29 in single SSH sessions:
  scripts/run_on_node.sh an12 'for i in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$i \
    python scripts/stage0_screening.py --shard $i/8 > logs/screen_$i.log 2>&1 & done; wait'
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
from foley_cw.real_measurer import ABSTAIN  # noqa: E402
from foley_cw.config import load_config  # noqa: E402
from foley_cw.run_store import RunStore, to_jsonable_target  # noqa: E402
from foley_cw.storage_budget import StorageBudget  # noqa: E402
from foley_cw.types import AgreementMetric, AxisKind, ScheduleSpec  # noqa: E402

SCREEN_AXIS_IDS = ("presence", "timing", "class", "material")
PHASE1_S_GRID = (0.05, 0.15, 0.25, 0.35, 0.45, 0.60, 0.75, 0.90)
VIDEO_DETERMINED_MIN = 0.9
DEFAULT_SCREEN_CFG = 1.0  # revised manual 3.1: screening at the HEADLINE cfg
                          # (exclusions must share the conditional distribution
                          # of the curves they gate); a 60-clip cfg=4.5
                          # sub-screen is logged separately, non-gating
N_INDEPENDENT = 8


def rng_for(seed: int, *parts) -> np.random.Generator:
    entropy = [seed] + [zlib.crc32(str(p).encode()) for p in parts]
    return np.random.default_rng(np.random.SeedSequence(entropy))


def run_clip(ib, measurer, axes, store: RunStore, clip: str, video: Path,
             schedule: ScheduleSpec, seed: int, screen_cfg: float,
             tag: str = "screen") -> dict:
    t0 = time.time()
    nfe0 = ib.nfe
    ib.cfg_strength = screen_cfg
    assert float(ib.cfg_strength) == screen_cfg
    cond = ib.make_video_cond(str(video), video_id=clip)

    targets_by_axis = {a.id: [] for a in axes}
    for j in range(N_INDEPENDENT):
        gid = f"{clip}__{tag}_ind{j}"
        tr = K.generate_trajectory(ib, cond, schedule, rng_for(seed, clip, tag, j),
                                   alpha=0.0, record_points=PHASE1_S_GRID)
        for s in PHASE1_S_GRID:
            feats, v = ib.tap_features_at(tr["states"][s], s, cond)
            store.put_features(gid, s, feats)
            t = ib.s_to_t.s_to_t(s)
            store.put_preview(gid, s, ib.decode(K.tweedie_x0(v, tr["states"][s], t)))
        audio = tr["audio"]
        store.put_final_wav(gid, audio)
        for a in axes:
            tgt = measurer.measure(audio, a)
            targets_by_axis[a.id].append(tgt)
            store.record_measurement(gid, a.id, tgt,
                                     extra={"role": f"{tag}_independent", "j": j,
                                            "clip": clip, "cfg": screen_cfg})

    # Confident-subset agreement for categorical axes (abstains excluded; frozen
    # interpretation #3); embedding axes use the plain metric. Abstain rates and
    # confident counts journaled for the manifest's per-axis usable-n accounting.
    a_ind, n_conf, abstain = {}, {}, {}
    for a in axes:
        if a.kind is AxisKind.CATEGORICAL:
            labels_a = [t.label for t in targets_by_axis[a.id]]
            v, n = confident_agreement(labels_a, AgreementMetric.EXACT_MATCH,
                                       abstain=ABSTAIN)
            a_ind[a.id] = float(v)
            n_conf[a.id] = int(n)
            abstain[a.id] = sum(1 for l in labels_a if l == ABSTAIN) / max(len(labels_a), 1)
        else:
            a_ind[a.id] = float(agreement(targets_by_axis[a.id], a.agreement))
            n_conf[a.id] = len(targets_by_axis[a.id])
            abstain[a.id] = 0.0
    labels = {a.id: [str(t.label) for t in targets_by_axis[a.id]]
              for a in axes if a.kind is AxisKind.CATEGORICAL}
    nfe = ib.nfe - nfe0
    elapsed = time.time() - t0
    print(f"[screen {clip}] {elapsed:.0f}s nfe={nfe} "
          f"A_ind={ {k: round(v, 3) for k, v in a_ind.items()} }", flush=True)
    return {"clip": clip, "cfg": screen_cfg, "a_independent": a_ind, "n_conf": n_conf, "abstain": abstain, "labels": labels,
            "elapsed_s": round(elapsed, 1), "nfe_velocity_calls": int(nfe)}


def aggregate(out: Path, manifest_clips: list[str], tag: str = "screen") -> int:
    store = RunStore(out)
    rows, registry = [], defaultdict(list)
    missing = []
    for clip in manifest_clips:
        unit = f"{tag}__{clip}"
        if not store.is_done(unit):
            missing.append(clip)
            continue
        d = store.load_journal(unit)
        for axis_id, v in d["a_independent"].items():
            rows.append({"clip": clip, "axis_id": axis_id, "a_independent": v})
            if np.isfinite(v) and v > VIDEO_DETERMINED_MIN:
                registry[axis_id].append(clip)
    if missing:
        print(f"[aggregate] {len(missing)} clips not yet screened (first: {missing[:5]})")
    screen_dir = out / "screening"
    screen_dir.mkdir(parents=True, exist_ok=True)
    # Tag-specific filenames so cfg-specific registries never collide (manual 3.1:
    # A_independent and the video-pinned registry differ by cfg).
    csv_path = screen_dir / f"a_independent_{tag}.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["clip", "axis_id", "a_independent"])
        w.writeheader()
        w.writerows(rows)
    reg_payload = {
        "_doc": "Per-axis video-determined exclusions (A_independent > 0.9, manual 3.1). "
                "Excluded from that axis's normalized curve only; usable for other axes. "
                "Seed-determined vs trajectory-early/late classification is a Phase-1 "
                "output, NOT a screening output.",
        "tag": tag,
        "threshold": VIDEO_DETERMINED_MIN,
        "n_screened": len(manifest_clips) - len(missing),
        "video_determined_by_axis": {k: sorted(v) for k, v in registry.items()},
    }
    (out / f"video_determined_registry_{tag}.json").write_text(json.dumps(reg_payload, indent=2))
    print(f"[aggregate] {len(rows)} axis-rows; registry: "
          f"{ {k: len(v) for k, v in registry.items()} } -> {out}/video_determined_registry_{tag}.json")
    return 0 if not missing else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=Path("data/manifests/screening_manifest.json"))
    ap.add_argument("--clips-root", type=Path, default=Path("data/FoleyBench/clips"))
    ap.add_argument("--out", type=Path, default=Path("results/stage0"))
    ap.add_argument("--variant", default="small_16k")
    ap.add_argument("--duration", type=float, default=8.0)
    ap.add_argument("--num-steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--aggregate", action="store_true", help="CPU: build csv + registry")
    ap.add_argument("--cfg", type=float, default=DEFAULT_SCREEN_CFG)
    ap.add_argument("--tag", default="screen",
                    help="journal/gen-id prefix; use e.g. subscreen45 for the\n"
                         "non-gating 60-clip cfg=4.5 sub-screen")
    args = ap.parse_args()

    clips = json.loads(args.manifest.read_text())["clips"]
    if args.aggregate:
        return aggregate(args.out, clips, tag=args.tag)

    shard_i, shard_n = (int(x) for x in args.shard.split("/"))
    schedule = ScheduleSpec(n_steps=args.num_steps, scan_points=PHASE1_S_GRID,
                            K_forks=0, N_independent=N_INDEPENDENT)
    grid = schedule.integration_s_grid()
    for s in PHASE1_S_GRID:
        assert np.any(np.isclose(grid, s, atol=1e-9)), f"s={s} off integration grid"

    budget = StorageBudget(cap_gb=100.0)
    store = RunStore(args.out, budget=budget)
    store.account_preexisting_tree()

    todo = [c for i, c in enumerate(sorted(clips)) if i % shard_n == shard_i]
    if args.limit:
        todo = todo[: args.limit]
    todo = [c for c in todo if not store.is_done(f"{args.tag}__{c}")]
    print(f"[screen] shard {args.shard}: {len(todo)} clips to run", flush=True)
    if not todo:
        return 0

    from foley_cw.feature_tap import InstrumentedBackend
    from foley_cw.mmaudio_backend import MMAudioBackend
    from foley_cw.real_measurer import RealFoleyMeasurer

    cfg_all = load_config()
    axes = [a for a in cfg_all.axes if a.id in SCREEN_AXIS_IDS]
    assert len(axes) == len(SCREEN_AXIS_IDS)

    backend = MMAudioBackend(variant=args.variant, device=args.device, full_precision=True,
                             cfg_strength=args.cfg, num_steps=args.num_steps,
                             duration_sec=args.duration, enable_conditions=True)
    ib = InstrumentedBackend(backend)
    measurer = RealFoleyMeasurer(device=args.device)

    for clip in todo:
        payload = run_clip(ib, measurer, axes, store, clip,
                           args.clips_root / f"{clip}.mp4", schedule, args.seed,
                           screen_cfg=args.cfg, tag=args.tag)
        payload["budget"] = budget.summary()
        store.journal_done(f"{args.tag}__{clip}", payload)
    print(f"[screen] shard {args.shard} complete; budget: {budget.summary()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
