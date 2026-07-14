#!/usr/bin/env python
"""Freeze the outcome-blind Arc-4 B2 multi-seed generation manifest."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from foley_cw.arc4_gpu import (  # noqa: E402
    B2_ALPHA,
    B2_BASE_SEEDS,
    B2_CFG,
    B2_K_FORKS,
    B2_N_CLIPS,
    B2_SCHEDULE,
    B2_S_GRID,
    atomic_json_create,
    select_b2_clips,
    sha256_file,
    validate_b2_generation_manifest,
)


def build_manifest(source_manifest: Path) -> dict:
    source = json.loads(source_manifest.read_text())
    if source.get("frozen") is not True:
        raise ValueError(f"source manifest is not frozen: {source_manifest}")
    pool = [str(clip) for clip in source.get("clips", {}).get("single_event", [])]
    clips = select_b2_clips(pool, n_clips=B2_N_CLIPS, seed=0)
    return {
        "_doc": (
            "Frozen Arc-4 B2 axis-agnostic raw multi-seed generation manifest. "
            "No outcome or measurement artifact was read during selection."
        ),
        "schema_version": 1,
        "selection_seed": 0,
        "selection_rule": (
            "ascending SHA256('arc4-b2-v1|<selection_seed>|<clip>') over the "
            "frozen single_event clip IDs; first 48"
        ),
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": sha256_file(source_manifest),
        "clip_pool": "single_event",
        "source_pool_size": len(set(pool)),
        "n_clips": B2_N_CLIPS,
        "clips": clips,
        "base_seeds": list(B2_BASE_SEEDS),
        "cfg": B2_CFG,
        "schedule": B2_SCHEDULE,
        "alpha": B2_ALPHA,
        "s_grid": list(B2_S_GRID),
        "k_forks": B2_K_FORKS,
        "variant": "small_16k",
        "duration_sec": 8.0,
        "num_steps": 20,
        "conditioning": "full_video_clip_synchformer_empty_text",
        "audio_format": "WAV",
        "audio_subtype": "FLOAT",
        "sample_rate": 16000,
        "expected_frames": 128000,
        "expected_artifacts": {
            "base_units": B2_N_CLIPS * len(B2_BASE_SEEDS),
            "base_wavs": B2_N_CLIPS * len(B2_BASE_SEEDS),
            "fork_cells": B2_N_CLIPS * len(B2_BASE_SEEDS) * len(B2_S_GRID),
            "fork_wavs": (
                B2_N_CLIPS * len(B2_BASE_SEEDS) * len(B2_S_GRID) * B2_K_FORKS
            ),
        },
        "analysis": "forbidden_in_generation_queue",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=Path("data/manifests/phase1_manifest_frozen.json"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/arc4_quarantine/b2/generation_manifest.json"),
    )
    args = parser.parse_args()
    if "arc4_quarantine" not in args.out.parts:
        raise ValueError("B2 generation manifest must live under results/arc4_quarantine/")

    manifest = build_manifest(args.source_manifest)
    validate_b2_generation_manifest(manifest)
    expected = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if args.out.exists():
        if args.out.read_text() != expected:
            raise FileExistsError(f"refusing to replace frozen manifest {args.out}")
    else:
        atomic_json_create(args.out, manifest)

    digest = sha256_file(args.out)
    sidecar = args.out.with_suffix(".sha256")
    content = f"{digest}  {args.out.name}\n"
    if sidecar.exists():
        if sidecar.read_text() != content:
            raise FileExistsError(f"refusing to replace frozen hash {sidecar}")
    else:
        tmp = sidecar.parent / f".{sidecar.name}.tmp.{os.getpid()}"
        try:
            with tmp.open("x", encoding="utf-8") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.link(tmp, sidecar)
        finally:
            tmp.unlink(missing_ok=True)
    print(
        f"wrote {args.out} clips={len(manifest['clips'])} "
        f"fork_wavs={manifest['expected_artifacts']['fork_wavs']} sha256={digest}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
