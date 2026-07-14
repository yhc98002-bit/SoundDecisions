#!/usr/bin/env python
"""Generate quarantined Arc-4 B2 multi-seed fork WAVs.

This queue is deliberately generation-only: it writes raw IEEE-float WAVs and
integrity journals, and has no measurement, aggregation, or reporting path.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import shlex
import socket
import subprocess
import sys
import time
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from foley_cw import score_sde as K  # noqa: E402
from foley_cw.arc4_gpu import (  # noqa: E402
    B2_BASE_SEEDS,
    B2_S_GRID,
    atomic_json_create,
    atomic_wav_create,
    sha256_file,
    validate_b2_generation_manifest,
    wav_metadata,
)
from foley_cw.kernel_provenance import assert_certified_kernel  # noqa: E402
from foley_cw.types import ScheduleSpec  # noqa: E402


def rng_for(seed: int, *parts) -> np.random.Generator:
    entropy = [seed] + [zlib.crc32(str(part).encode("utf-8")) for part in parts]
    return np.random.default_rng(np.random.SeedSequence(entropy))


def _unit_id(clip: str, base_seed: int) -> str:
    return f"{clip}__seed{base_seed}"


def _base_path(root: Path, clip: str, base_seed: int) -> Path:
    return root / "raw" / clip / f"seed{base_seed}" / "base.wav"


def _fork_path(root: Path, clip: str, base_seed: int, s: float, k: int) -> Path:
    return root / "raw" / clip / f"seed{base_seed}" / f"s{s:.2f}" / f"fork{k:02d}.wav"


def _cell_journal_path(root: Path, clip: str, base_seed: int, s: float) -> Path:
    return root / "journal" / "cells" / f"{_unit_id(clip, base_seed)}__s{s:.2f}.json"


def _unit_journal_path(root: Path, clip: str, base_seed: int) -> Path:
    return root / "journal" / "units" / f"{_unit_id(clip, base_seed)}.json"


def _read_frozen_hash(manifest_path: Path) -> str:
    digest = sha256_file(manifest_path)
    sidecar = manifest_path.with_suffix(".sha256")
    if not sidecar.is_file():
        raise FileNotFoundError(f"missing frozen B2 manifest hash: {sidecar}")
    fields = sidecar.read_text().strip().split()
    if len(fields) != 2 or fields[0] != digest or fields[1] != manifest_path.name:
        raise ValueError(f"B2 manifest hash mismatch: {sidecar}")
    return digest


def _artifact_metadata(
    path: Path,
    root: Path,
    manifest: dict,
    *,
    role: str,
    s: float | None = None,
    k: int | None = None,
) -> dict:
    meta = wav_metadata(path, expected_subtype=manifest["audio_subtype"])
    if meta["sample_rate"] != int(manifest["sample_rate"]):
        raise ValueError(f"wrong sample rate in {path}: {meta['sample_rate']}")
    if meta["frames"] != int(manifest["expected_frames"]):
        raise ValueError(f"wrong frame count in {path}: {meta['frames']}")
    meta.update({"path": str(path.relative_to(root)), "role": role})
    if s is not None:
        meta["s"] = float(s)
    if k is not None:
        meta["fork_index"] = int(k)
    return meta


def _assert_metadata_equal(saved: dict, actual: dict, path: Path) -> None:
    if saved != actual:
        raise RuntimeError(f"artifact metadata mismatch for {path}; refusing to replace")


def _validate_cell(
    root: Path,
    manifest: dict,
    manifest_sha: str,
    clip: str,
    base_seed: int,
    s: float,
) -> dict | None:
    path = _cell_journal_path(root, clip, base_seed, s)
    if not path.exists():
        return None
    try:
        row = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid B2 cell journal {path}; refusing to replace") from exc
    if (
        row.get("clip") != clip
        or int(row.get("base_seed", -1)) != base_seed
        or not math.isclose(float(row.get("s", -1)), s)
        or row.get("generation_manifest_sha256") != manifest_sha
        or len(row.get("artifacts", [])) != int(manifest["k_forks"])
    ):
        raise RuntimeError(f"B2 cell journal design mismatch: {path}")
    for k, saved in enumerate(row["artifacts"]):
        wav = _fork_path(root, clip, base_seed, s, k)
        actual = _artifact_metadata(
            wav, root, manifest, role="fork", s=s, k=k
        )
        _assert_metadata_equal(saved, actual, wav)
    return row


def _validate_unit(
    root: Path,
    manifest: dict,
    manifest_sha: str,
    clip: str,
    base_seed: int,
) -> bool:
    path = _unit_journal_path(root, clip, base_seed)
    if not path.exists():
        return False
    try:
        row = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid B2 unit journal {path}; refusing to replace") from exc
    if (
        row.get("clip") != clip
        or int(row.get("base_seed", -1)) != base_seed
        or row.get("generation_manifest_sha256") != manifest_sha
        or tuple(float(s) for s in row.get("s_grid", [])) != B2_S_GRID
        or int(row.get("fork_wavs", -1))
        != len(B2_S_GRID) * int(manifest["k_forks"])
    ):
        raise RuntimeError(f"B2 unit journal design mismatch: {path}")
    base_path = _base_path(root, clip, base_seed)
    actual_base = _artifact_metadata(base_path, root, manifest, role="base")
    _assert_metadata_equal(row.get("base_artifact", {}), actual_base, base_path)
    for s in B2_S_GRID:
        if _validate_cell(root, manifest, manifest_sha, clip, base_seed, s) is None:
            raise RuntimeError(f"B2 unit journal {path} is missing its s={s:.2f} cell")
    return True


def _git_commit(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()


def _physical_gpu(device: str) -> str:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if not device.startswith("cuda:"):
        return "unknown"
    logical = int(device.split(":", 1)[1])
    return visible[logical].strip() if logical < len(visible) else "unknown"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--clips-root", type=Path, required=True)
    parser.add_argument(
        "--certified",
        type=Path,
        default=Path("results/stage_m_rerun/certified_kernels.json"),
    )
    parser.add_argument("--shard", default="0/1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit-clips", type=int, default=0)
    args = parser.parse_args()

    if "arc4_quarantine" not in args.out.parts:
        raise ValueError("B2 raw outputs must live under results/arc4_quarantine/")
    manifest = json.loads(args.generation_manifest.read_text())
    validate_b2_generation_manifest(manifest)
    manifest_sha = _read_frozen_hash(args.generation_manifest)
    cert = assert_certified_kernel(
        float(manifest["cfg"]),
        str(manifest["schedule"]),
        args.certified,
        require_ratified=True,
    )
    kernel_sha = sha256_file(args.certified)
    print(
        f"[b2] manifest={manifest_sha} kernel={cert['token']} "
        f"ratified={cert['ratified']}",
        flush=True,
    )

    shard_i, shard_n = (int(value) for value in args.shard.split("/"))
    if not 0 <= shard_i < shard_n:
        raise ValueError(f"invalid shard {args.shard}")
    assigned_clips = [
        clip for index, clip in enumerate(manifest["clips"]) if index % shard_n == shard_i
    ]
    if args.limit_clips:
        assigned_clips = assigned_clips[: args.limit_clips]

    todo_units = []
    for clip in assigned_clips:
        for base_seed in B2_BASE_SEEDS:
            if not _validate_unit(
                args.out, manifest, manifest_sha, clip, base_seed
            ):
                todo_units.append((clip, base_seed))
    print(
        f"[b2] shard={args.shard} clips={len(assigned_clips)} "
        f"todo_units={len(todo_units)} expected_fork_wavs="
        f"{len(assigned_clips) * len(B2_BASE_SEEDS) * len(B2_S_GRID) * manifest['k_forks']}",
        flush=True,
    )
    if not todo_units:
        return 0

    from foley_cw.feature_tap import InstrumentedBackend
    from foley_cw.mmaudio_backend import MMAudioBackend

    backend = MMAudioBackend(
        variant=manifest["variant"],
        device=args.device,
        full_precision=True,
        cfg_strength=float(manifest["cfg"]),
        num_steps=int(manifest["num_steps"]),
        duration_sec=float(manifest["duration_sec"]),
        enable_conditions=True,
    )
    ib = InstrumentedBackend(backend)
    schedule = ScheduleSpec(
        n_steps=int(manifest["num_steps"]),
        scan_points=B2_S_GRID,
        K_forks=int(manifest["k_forks"]),
        g_kind=manifest["schedule"],
        g_value=1.0,
    )
    grid = schedule.integration_s_grid()
    for s in B2_S_GRID:
        if not np.any(np.isclose(grid, s, atol=1e-9)):
            raise ValueError(f"frozen B2 s={s} is off the integration grid")
    g = K.make_g(manifest["schedule"], 1.0)

    repo = Path(__file__).resolve().parent.parent
    provenance = {
        "git_commit": _git_commit(repo),
        "command": shlex.join(sys.argv),
        "node": socket.gethostname(),
        "logical_device": args.device,
        "physical_gpu_id": _physical_gpu(args.device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "python": sys.executable,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "weights_source": os.environ.get("FOLEY_CW_WEIGHTS_SOURCE", ""),
        "hf_offline": os.environ.get("HF_HUB_OFFLINE", ""),
    }
    condition_cache = {}

    for clip in assigned_clips:
        clip_units = [unit for unit in todo_units if unit[0] == clip]
        if not clip_units:
            continue
        video = args.clips_root / f"{clip}.mp4"
        if not video.is_file():
            raise FileNotFoundError(video)
        if clip not in condition_cache:
            condition_cache[clip] = ib.make_video_cond(str(video), video_id=clip)
        cond = condition_cache[clip]

        for _, base_seed in clip_units:
            started = time.time()
            unit_nfe_start = ib.nfe
            complete_cells = {
                s: _validate_cell(
                    args.out, manifest, manifest_sha, clip, base_seed, s
                )
                for s in B2_S_GRID
            }
            base_path = _base_path(args.out, clip, base_seed)
            need_trajectory = not base_path.exists() or any(
                row is None for row in complete_cells.values()
            )
            trajectory = None
            if need_trajectory:
                trajectory = K.generate_trajectory(
                    ib,
                    cond,
                    schedule,
                    rng_for(base_seed, clip, "base"),
                    alpha=0.0,
                    record_points=B2_S_GRID,
                )
                audio = np.asarray(trajectory["audio"], dtype=np.float32).reshape(-1)
                if audio.size != int(manifest["expected_frames"]):
                    raise RuntimeError(
                        f"base audio {clip}/seed{base_seed} has {audio.size} frames; "
                        f"expected {manifest['expected_frames']}"
                    )
                if not base_path.exists():
                    atomic_wav_create(
                        base_path,
                        audio,
                        sample_rate=int(manifest["sample_rate"]),
                        subtype=manifest["audio_subtype"],
                    )
            base_meta = _artifact_metadata(
                base_path, args.out, manifest, role="base"
            )

            cell_records = []
            for s in B2_S_GRID:
                row = complete_cells[s]
                if row is None:
                    if trajectory is None:
                        raise AssertionError("missing trajectory for incomplete B2 cell")
                    cell_nfe_start = ib.nfe
                    cell_started = time.time()
                    audios = K.fork_tail(
                        ib,
                        trajectory["states"][s],
                        s,
                        cond,
                        float(manifest["alpha"]),
                        int(manifest["k_forks"]),
                        schedule,
                        rng_for(base_seed, clip, "fork", s),
                        g=g,
                    )
                    if len(audios) != int(manifest["k_forks"]):
                        raise RuntimeError(f"B2 fork count mismatch at {clip}/seed{base_seed}/s{s}")
                    artifacts = []
                    for k, audio in enumerate(audios):
                        wav = _fork_path(args.out, clip, base_seed, s, k)
                        samples = np.asarray(audio, dtype=np.float32).reshape(-1)
                        if samples.size != int(manifest["expected_frames"]):
                            raise RuntimeError(
                                f"fork audio {clip}/seed{base_seed}/s{s}/k{k} has "
                                f"{samples.size} frames; expected {manifest['expected_frames']}"
                            )
                        if not wav.exists():
                            atomic_wav_create(
                                wav,
                                samples,
                                sample_rate=int(manifest["sample_rate"]),
                                subtype=manifest["audio_subtype"],
                            )
                        artifacts.append(
                            _artifact_metadata(
                                wav, args.out, manifest, role="fork", s=s, k=k
                            )
                        )
                    expected_cell_nfe = int(
                        round((1.0 - s) * int(manifest["num_steps"]))
                    ) * int(manifest["k_forks"])
                    row = {
                        "_doc": "Arc-4 B2 raw fork-cell journal; no measurements.",
                        "clip": clip,
                        "base_seed": base_seed,
                        "s": s,
                        "cfg": manifest["cfg"],
                        "schedule": manifest["schedule"],
                        "alpha": manifest["alpha"],
                        "k_forks": manifest["k_forks"],
                        "conditioning": manifest["conditioning"],
                        "generation_manifest_sha256": manifest_sha,
                        "kernel_ledger_sha256": kernel_sha,
                        "kernel": cert,
                        "rng_lineage": {
                            "algorithm": "numpy.SeedSequence([base_seed, crc32(parts)...])",
                            "parts": [clip, "fork", s],
                        },
                        "expected_nfe_velocity_calls": expected_cell_nfe,
                        "actual_nfe_velocity_calls": int(ib.nfe - cell_nfe_start),
                        "elapsed_s": round(time.time() - cell_started, 3),
                        "artifacts": artifacts,
                        "provenance": provenance,
                    }
                    atomic_json_create(
                        _cell_journal_path(args.out, clip, base_seed, s), row
                    )
                cell_path = _cell_journal_path(args.out, clip, base_seed, s)
                cell_records.append(
                    {
                        "s": s,
                        "journal": str(cell_path.relative_to(args.out)),
                        "sha256": sha256_file(cell_path),
                    }
                )

            payload = {
                "_doc": "Arc-4 B2 raw clip/base-seed rollup; no measurements.",
                "clip": clip,
                "base_seed": base_seed,
                "cfg": manifest["cfg"],
                "schedule": manifest["schedule"],
                "alpha": manifest["alpha"],
                "s_grid": list(B2_S_GRID),
                "k_forks": manifest["k_forks"],
                "fork_wavs": len(B2_S_GRID) * int(manifest["k_forks"]),
                "generation_manifest_sha256": manifest_sha,
                "kernel_ledger_sha256": kernel_sha,
                "kernel": cert,
                "base_rng_lineage": {
                    "algorithm": "numpy.SeedSequence([base_seed, crc32(parts)...])",
                    "parts": [clip, "base"],
                },
                "base_artifact": base_meta,
                "cell_journals": cell_records,
                "expected_full_unit_nfe_velocity_calls": 1100,
                "actual_nfe_velocity_calls_this_attempt": int(ib.nfe - unit_nfe_start),
                "elapsed_s_this_attempt": round(time.time() - started, 3),
                "provenance": provenance,
            }
            atomic_json_create(_unit_journal_path(args.out, clip, base_seed), payload)
            print(
                f"[b2 {clip} seed={base_seed}] complete "
                f"elapsed={payload['elapsed_s_this_attempt']:.1f}s "
                f"nfe={payload['actual_nfe_velocity_calls_this_attempt']}",
                flush=True,
            )
    print(f"[b2] shard={args.shard} complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
