#!/usr/bin/env python
"""Arc-4 B6 raw condition-swap generation from the frozen pair manifest.

The script deliberately has no aggregation or reporting mode. Each ordered
pair is an atomic, resumable journal unit under the quarantine tree.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from foley_cw import condition_swap as CS  # noqa: E402
from foley_cw import score_sde as K  # noqa: E402
from foley_cw.arc4_gpu import (  # noqa: E402
    B6_S_GRID,
    sha256_file,
    validate_pair_manifest,
)
from foley_cw.config import load_config  # noqa: E402
from foley_cw.kernel_provenance import assert_certified_kernel  # noqa: E402
from foley_cw.run_store import RunStore, to_jsonable_target  # noqa: E402
from foley_cw.types import ScheduleSpec  # noqa: E402

AXES = ("class", "timing", "presence", "material")


def rng_for(seed: int, *parts) -> np.random.Generator:
    entropy = [seed] + [zlib.crc32(str(part).encode("utf-8")) for part in parts]
    return np.random.default_rng(np.random.SeedSequence(entropy))


def _measure_all(measurer, axes, audio: np.ndarray) -> dict:
    targets = {axis.id: to_jsonable_target(measurer.measure(audio, axis)) for axis in axes}
    return {
        "targets": targets,
        "panns527_posterior": measurer.panns_posterior(audio).tolist(),
    }


def _journal_complete(store: RunStore, pair: dict, cfg: float) -> bool:
    if not store.is_done(pair["pair_id"]):
        return False
    try:
        row = store.load_journal(pair["pair_id"])
        return (
            row.get("pair_id") == pair["pair_id"]
            and math.isclose(float(row.get("cfg")), cfg)
            and row.get("source") == pair["source"]
            and row.get("donor") == pair["donor"]
            and tuple(float(s) for s in row.get("s_grid", [])) == B6_S_GRID
            and set(row.get("swap_targets", {})) == {f"{s:.2f}" for s in B6_S_GRID}
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair-manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--clips-root", type=Path, required=True)
    ap.add_argument("--certified", type=Path,
                    default=Path("results/stage_m_rerun/certified_kernels.json"))
    ap.add_argument("--cfg", type=float, choices=(1.0, 4.5), required=True)
    ap.add_argument("--schedule", default="sqrt_down")
    ap.add_argument("--variant", default="small_16k")
    ap.add_argument("--duration", type=float, default=8.0)
    ap.add_argument("--num-steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    if args.seed != 0:
        raise ValueError("Arc-4 B6 generation seed is frozen at 0")
    if "arc4_quarantine" not in args.out.parts:
        raise ValueError("B6 raw outputs must live under results/arc4_quarantine/")

    manifest = json.loads(args.pair_manifest.read_text())
    validate_pair_manifest(manifest, expected_pairs_per_cfg=128)
    pairs = [pair for pair in manifest["pairs"]
             if math.isclose(float(pair["cfg"]), args.cfg)]
    if len(pairs) != 128:
        raise ValueError(f"cfg={args.cfg:g}: expected 128 manifest pairs, got {len(pairs)}")

    cert = assert_certified_kernel(args.cfg, args.schedule, args.certified,
                                   require_ratified=True)
    print(f"[b6] kernel={cert['token']} ratified={cert['ratified']}", flush=True)

    shard_i, shard_n = (int(value) for value in args.shard.split("/"))
    assigned = [pair for index, pair in enumerate(pairs) if index % shard_n == shard_i]
    if args.limit:
        assigned = assigned[:args.limit]
    store = RunStore(args.out)
    todo = []
    for pair in assigned:
        if store.is_done(pair["pair_id"]):
            if not _journal_complete(store, pair, args.cfg):
                raise RuntimeError(
                    f"invalid existing B6 journal for {pair['pair_id']}; refusing to replace")
        else:
            todo.append(pair)
    print(f"[b6] cfg={args.cfg:g} shard={args.shard} todo={len(todo)} "
          f"assigned={len(assigned)}", flush=True)
    if not todo:
        return 0

    from foley_cw.feature_tap import InstrumentedBackend
    from foley_cw.mmaudio_backend import MMAudioBackend
    from foley_cw.real_measurer import RealFoleyMeasurer

    backend = MMAudioBackend(
        variant=args.variant,
        device=args.device,
        full_precision=True,
        cfg_strength=args.cfg,
        num_steps=args.num_steps,
        duration_sec=args.duration,
        enable_conditions=True,
    )
    ib = InstrumentedBackend(backend)
    measurer = RealFoleyMeasurer(device=args.device)
    axes = [axis for axis in load_config().axes if axis.id in AXES]
    schedule = ScheduleSpec(
        n_steps=args.num_steps,
        scan_points=B6_S_GRID,
        g_kind=args.schedule,
        g_value=1.0,
    )
    integration_grid = schedule.integration_s_grid()
    for s in B6_S_GRID:
        if not np.any(np.isclose(integration_grid, s, atol=1e-9)):
            raise ValueError(f"frozen B6 s={s} is off the integration grid")

    cond_cache = {}
    source_cache = {}
    donor_cache = {}
    manifest_sha = sha256_file(args.pair_manifest)

    def condition(clip: str):
        if clip not in cond_cache:
            video = args.clips_root / f"{clip}.mp4"
            if not video.is_file():
                raise FileNotFoundError(video)
            cond_cache[clip] = ib.make_video_cond(str(video), video_id=clip)
        return cond_cache[clip]

    def source_record(clip: str):
        if clip not in source_cache:
            trajectory = K.generate_trajectory(
                ib,
                condition(clip),
                schedule,
                rng_for(args.seed, clip, "src"),
                alpha=0.0,
                record_points=B6_S_GRID,
            )
            source_cache[clip] = {
                "states": trajectory["states"],
                "targets": _measure_all(measurer, axes, trajectory["audio"]),
            }
        return source_cache[clip]

    def donor_record(clip: str):
        if clip not in donor_cache:
            trajectory = K.generate_trajectory(
                ib,
                condition(clip),
                schedule,
                rng_for(args.seed, clip, "don"),
                alpha=0.0,
                record_points=(),
            )
            donor_cache[clip] = {
                "targets": _measure_all(measurer, axes, trajectory["audio"]),
            }
        return donor_cache[clip]

    for pair in todo:
        started = time.time()
        nfe_before = ib.nfe
        source_cache_hit = pair["source"] in source_cache
        donor_cache_hit = pair["donor"] in donor_cache
        source = source_record(pair["source"])
        donor = donor_record(pair["donor"])
        donor_cond = condition(pair["donor"])
        swaps = {}
        for s in B6_S_GRID:
            audio = CS.cond_swap_complete(
                ib, source["states"][s], s, donor_cond, schedule)
            swaps[f"{s:.2f}"] = _measure_all(measurer, axes, audio)
        payload = {
            "_doc": "Arc-4 B6 raw pair journal; no aggregate estimand or decision token.",
            "pair_id": pair["pair_id"],
            "cfg": args.cfg,
            "source": pair["source"],
            "donor": pair["donor"],
            "source_cached_label": pair["source_cached_label"],
            "donor_cached_label": pair["donor_cached_label"],
            "cached_label_role": pair["cached_label_role"],
            "seed": args.seed,
            "schedule": args.schedule,
            "variant": args.variant,
            "duration": args.duration,
            "num_steps": args.num_steps,
            "s_grid": list(B6_S_GRID),
            "pair_manifest_sha256": manifest_sha,
            "source_targets": source["targets"],
            "donor_targets": donor["targets"],
            "swap_targets": swaps,
            "nfe_velocity_calls_since_previous_pair": int(ib.nfe - nfe_before),
            "source_cache_hit": source_cache_hit,
            "donor_cache_hit": donor_cache_hit,
            "elapsed_s": round(time.time() - started, 3),
        }
        store.journal_done(pair["pair_id"], payload)
        print(f"[b6 {pair['pair_id']}] complete elapsed={payload['elapsed_s']:.1f}s "
              f"nfe_delta={payload['nfe_velocity_calls_since_previous_pair']}", flush=True)
    print(f"[b6] cfg={args.cfg:g} shard={args.shard} complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
