#!/usr/bin/env python
"""Arc-4 B6 raw condition-swap generation from the frozen pair manifest.

The script deliberately has no measurer, aggregation, or reporting path. It
writes measurement-ready IEEE-float WAVs and integrity journals only.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import socket
import subprocess
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
    atomic_json_create,
    atomic_wav_create,
    sha256_file,
    validate_pair_manifest,
    wav_metadata,
)
from foley_cw.kernel_provenance import assert_certified_kernel  # noqa: E402
from foley_cw.types import ScheduleSpec  # noqa: E402

SAMPLE_RATE = 16000
EXPECTED_FRAMES = 128000
AUDIO_SUBTYPE = "FLOAT"


def rng_for(seed: int, *parts) -> np.random.Generator:
    entropy = [seed] + [zlib.crc32(str(part).encode("utf-8")) for part in parts]
    return np.random.default_rng(np.random.SeedSequence(entropy))


def _journal_path(root: Path, pair_id: str) -> Path:
    return root / "journal" / f"{pair_id}.json"


def _audio_path(root: Path, pair_id: str, role: str, s: float | None = None) -> Path:
    name = f"swap_s{s:.2f}.wav" if role == "swap" else f"{role}.wav"
    return root / "raw" / pair_id / name


def _artifact_metadata(path: Path, root: Path, *, role: str, s: float | None = None) -> dict:
    meta = wav_metadata(path, expected_subtype=AUDIO_SUBTYPE)
    if meta["sample_rate"] != SAMPLE_RATE or meta["frames"] != EXPECTED_FRAMES:
        raise ValueError(
            f"invalid B6 audio shape {path}: sr={meta['sample_rate']} frames={meta['frames']}"
        )
    meta.update({"path": str(path.relative_to(root)), "role": role})
    if s is not None:
        meta["s"] = float(s)
    return meta


def _persist_audio(
    path: Path,
    root: Path,
    audio: np.ndarray,
    *,
    role: str,
    s: float | None = None,
) -> dict:
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    if samples.size != EXPECTED_FRAMES:
        raise RuntimeError(f"B6 {role} audio has {samples.size} frames; expected {EXPECTED_FRAMES}")
    if not path.exists():
        atomic_wav_create(path, samples, sample_rate=SAMPLE_RATE, subtype=AUDIO_SUBTYPE)
    return _artifact_metadata(path, root, role=role, s=s)


def _assert_saved_artifact(saved: dict, actual: dict, path: Path) -> None:
    if saved != actual:
        raise RuntimeError(f"B6 artifact metadata mismatch for {path}; refusing to replace")


def _journal_complete(
    root: Path,
    pair: dict,
    cfg: float,
    pair_manifest_sha: str,
) -> bool:
    path = _journal_path(root, pair["pair_id"])
    if not path.exists():
        return False
    try:
        row = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid B6 journal {path}; refusing to replace") from exc
    if (
        row.get("pair_id") != pair["pair_id"]
        or not math.isclose(float(row.get("cfg", -1)), cfg)
        or row.get("source") != pair["source"]
        or row.get("donor") != pair["donor"]
        or row.get("pair_manifest_sha256") != pair_manifest_sha
        or tuple(float(s) for s in row.get("s_grid", [])) != B6_S_GRID
    ):
        raise RuntimeError(f"B6 journal design mismatch: {path}")
    raw = row.get("raw_audio", {})
    for role in ("source", "donor"):
        wav = _audio_path(root, pair["pair_id"], role)
        actual = _artifact_metadata(wav, root, role=role)
        _assert_saved_artifact(raw.get(role, {}), actual, wav)
    swaps = raw.get("swaps", {})
    if set(swaps) != {f"{s:.2f}" for s in B6_S_GRID}:
        raise RuntimeError(f"B6 journal has incomplete swap WAVs: {path}")
    for s in B6_S_GRID:
        wav = _audio_path(root, pair["pair_id"], "swap", s)
        actual = _artifact_metadata(wav, root, role="swap", s=s)
        _assert_saved_artifact(swaps[f"{s:.2f}"], actual, wav)
    return True


def _git_commit(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--clips-root", type=Path, required=True)
    parser.add_argument(
        "--certified",
        type=Path,
        default=Path("results/stage_m_rerun/certified_kernels.json"),
    )
    parser.add_argument("--cfg", type=float, choices=(1.0, 4.5), required=True)
    parser.add_argument("--schedule", default="sqrt_down")
    parser.add_argument("--variant", default="small_16k")
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shard", default="0/1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    if args.seed != 0:
        raise ValueError("Arc-4 B6 generation seed is frozen at 0")
    if "arc4_quarantine" not in args.out.parts:
        raise ValueError("B6 raw outputs must live under results/arc4_quarantine/")

    manifest = json.loads(args.pair_manifest.read_text())
    validate_pair_manifest(manifest, expected_pairs_per_cfg=128)
    pair_manifest_sha = sha256_file(args.pair_manifest)
    pairs = [
        pair for pair in manifest["pairs"] if math.isclose(float(pair["cfg"]), args.cfg)
    ]
    if len(pairs) != 128:
        raise ValueError(f"cfg={args.cfg:g}: expected 128 manifest pairs, got {len(pairs)}")

    cert = assert_certified_kernel(
        args.cfg, args.schedule, args.certified, require_ratified=True
    )
    kernel_sha = sha256_file(args.certified)
    print(f"[b6] kernel={cert['token']} ratified={cert['ratified']}", flush=True)

    shard_i, shard_n = (int(value) for value in args.shard.split("/"))
    assigned = [pair for index, pair in enumerate(pairs) if index % shard_n == shard_i]
    if args.limit:
        assigned = assigned[: args.limit]
    todo = [
        pair
        for pair in assigned
        if not _journal_complete(args.out, pair, args.cfg, pair_manifest_sha)
    ]
    print(
        f"[b6] cfg={args.cfg:g} shard={args.shard} todo={len(todo)} assigned={len(assigned)}",
        flush=True,
    )
    if not todo:
        return 0

    from foley_cw.feature_tap import InstrumentedBackend
    from foley_cw.mmaudio_backend import MMAudioBackend

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

    repo = Path(__file__).resolve().parent.parent
    provenance = {
        "git_commit": _git_commit(repo),
        "command": shlex.join(sys.argv),
        "node": socket.gethostname(),
        "logical_device": args.device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "weights_source": os.environ.get("FOLEY_CW_WEIGHTS_SOURCE", ""),
        "hf_offline": os.environ.get("HF_HUB_OFFLINE", ""),
    }
    cond_cache = {}
    source_cache = {}
    donor_cache = {}

    def condition(clip: str):
        if clip not in cond_cache:
            video = args.clips_root / f"{clip}.mp4"
            if not video.is_file():
                raise FileNotFoundError(video)
            cond_cache[clip] = ib.make_video_cond(str(video), video_id=clip)
        return cond_cache[clip]

    def source_record(clip: str):
        if clip not in source_cache:
            source_cache[clip] = K.generate_trajectory(
                ib,
                condition(clip),
                schedule,
                rng_for(args.seed, clip, "src"),
                alpha=0.0,
                record_points=B6_S_GRID,
            )
        return source_cache[clip]

    def donor_record(clip: str):
        if clip not in donor_cache:
            donor_cache[clip] = K.generate_trajectory(
                ib,
                condition(clip),
                schedule,
                rng_for(args.seed, clip, "don"),
                alpha=0.0,
                record_points=(),
            )
        return donor_cache[clip]

    for pair in todo:
        started = time.time()
        nfe_before = ib.nfe
        source_cache_hit = pair["source"] in source_cache
        donor_cache_hit = pair["donor"] in donor_cache
        source = source_record(pair["source"])
        donor = donor_record(pair["donor"])
        donor_cond = condition(pair["donor"])
        pair_id = pair["pair_id"]
        source_meta = _persist_audio(
            _audio_path(args.out, pair_id, "source"),
            args.out,
            source["audio"],
            role="source",
        )
        donor_meta = _persist_audio(
            _audio_path(args.out, pair_id, "donor"),
            args.out,
            donor["audio"],
            role="donor",
        )
        swaps = {}
        for s in B6_S_GRID:
            wav = _audio_path(args.out, pair_id, "swap", s)
            if wav.exists():
                swaps[f"{s:.2f}"] = _artifact_metadata(
                    wav, args.out, role="swap", s=s
                )
                continue
            audio = CS.cond_swap_complete(
                ib, source["states"][s], s, donor_cond, schedule
            )
            swaps[f"{s:.2f}"] = _persist_audio(
                wav, args.out, audio, role="swap", s=s
            )

        payload = {
            "_doc": "Arc-4 B6 raw pair journal; no measurements or aggregate estimands.",
            "pair_id": pair_id,
            "cfg": args.cfg,
            "source": pair["source"],
            "donor": pair["donor"],
            "source_cached_label": pair["source_cached_label"],
            "donor_cached_label": pair["donor_cached_label"],
            "cached_label_role": pair["cached_label_role"],
            "cached_labels_usage": "frozen_stratification_provenance_only",
            "seed": args.seed,
            "schedule": args.schedule,
            "variant": args.variant,
            "duration": args.duration,
            "num_steps": args.num_steps,
            "s_grid": list(B6_S_GRID),
            "pair_manifest_sha256": pair_manifest_sha,
            "kernel_ledger_sha256": kernel_sha,
            "kernel": cert,
            "rng_lineage": {
                "source": [args.seed, pair["source"], "src"],
                "donor": [args.seed, pair["donor"], "don"],
                "algorithm": "numpy.SeedSequence([seed, crc32(parts)...])",
            },
            "raw_audio": {
                "source": source_meta,
                "donor": donor_meta,
                "swaps": swaps,
            },
            "nfe_velocity_calls_since_previous_pair": int(ib.nfe - nfe_before),
            "source_cache_hit": source_cache_hit,
            "donor_cache_hit": donor_cache_hit,
            "elapsed_s": round(time.time() - started, 3),
            "provenance": provenance,
        }
        atomic_json_create(_journal_path(args.out, pair_id), payload)
        print(
            f"[b6 {pair_id}] complete elapsed={payload['elapsed_s']:.1f}s "
            f"nfe_delta={payload['nfe_velocity_calls_since_previous_pair']}",
            flush=True,
        )
    print(f"[b6] cfg={args.cfg:g} shard={args.shard} complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
