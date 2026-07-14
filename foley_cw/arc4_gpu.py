"""CPU-only helpers for the isolated Arc-4 GPU appendix.

This module contains manifest construction and cache-integrity code only.  It
does not import torch, open evaluation reports, or compute experiment metrics.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

ABSTAIN = "abstain"
B2_S_GRID = (0.05, 0.15, 0.25, 0.35, 0.45, 0.60, 0.75, 0.90)
B2_BASE_SEEDS = (0, 1, 2, 3, 4)
B2_EXTENSION_BASE_SEEDS = (5, 6, 7, 8)
B2_BACKFILL_BASE_SEEDS = (9, 10, 11, 12)
B2_SECOND_BACKFILL_BASE_SEEDS = (13, 14, 15, 16)
B2_CFG = 4.5
B2_SCHEDULE = "sqrt_down"
B2_ALPHA = 0.8
B2_K_FORKS = 12
B2_N_CLIPS = 48
B6_S_GRID = (0.05, 0.15, 0.25, 0.35, 0.45, 0.60, 0.75, 0.90)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_npz_create(path: Path, **arrays: np.ndarray) -> Path:
    """Create a compressed NPZ atomically, refusing to replace any artifact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.tmp.{os.getpid()}"
    try:
        with tmp.open("xb") as fh:
            np.savez_compressed(fh, **arrays)
            fh.flush()
            os.fsync(fh.fileno())
        os.link(tmp, path)  # atomic create; raises FileExistsError on collision
    finally:
        tmp.unlink(missing_ok=True)
    return path


def atomic_json_create(path: Path, payload: dict) -> Path:
    """Create sorted JSON atomically and refuse replacement."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    tmp = path.parent / f".{path.name}.tmp.{os.getpid()}"
    try:
        with tmp.open("x", encoding="utf-8") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.link(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


def atomic_wav_create(
    path: Path,
    audio: np.ndarray,
    *,
    sample_rate: int,
    subtype: str = "FLOAT",
) -> Path:
    """Create a measurement-ready WAV atomically and refuse replacement."""
    import soundfile as sf

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.tmp.{os.getpid()}"
    try:
        with tmp.open("xb"):
            pass
        sf.write(
            str(tmp),
            np.asarray(audio, dtype=np.float32).reshape(-1),
            int(sample_rate),
            format="WAV",
            subtype=subtype,
        )
        with tmp.open("rb") as fh:
            os.fsync(fh.fileno())
        os.link(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


def wav_metadata(path: Path, *, expected_subtype: str = "FLOAT") -> dict:
    """Validate one raw B2 WAV and return integrity metadata."""
    import soundfile as sf

    path = Path(path)
    info = sf.info(str(path))
    if (
        info.format != "WAV"
        or info.subtype != expected_subtype
        or info.samplerate <= 0
        or info.channels != 1
        or info.frames <= 0
    ):
        raise ValueError(
            f"invalid WAV {path}: format={info.format} subtype={info.subtype} "
            f"sr={info.samplerate} channels={info.channels} frames={info.frames}"
        )
    return {
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "sample_rate": int(info.samplerate),
        "frames": int(info.frames),
        "channels": int(info.channels),
        "format": info.format,
        "subtype": info.subtype,
    }


def _b2_clip_rank(seed: int, clip: str) -> str:
    return hashlib.sha256(f"arc4-b2-v1|{seed}|{clip}".encode("utf-8")).hexdigest()


def select_b2_clips(
    clips: Iterable[str],
    *,
    n_clips: int = B2_N_CLIPS,
    seed: int = 0,
) -> list[str]:
    """Select clips by an outcome-blind, seeded SHA256 rank."""
    pool = sorted({str(clip) for clip in clips})
    if len(pool) < n_clips:
        raise ValueError(f"only {len(pool)} unique clips; need {n_clips}")
    return sorted(pool, key=lambda clip: (_b2_clip_rank(seed, clip), clip))[:n_clips]


def validate_b2_generation_manifest(manifest: dict) -> None:
    """Enforce the frozen Arc-4 B2 raw-generation design."""
    expected = {
        "selection_seed": 0,
        "n_clips": B2_N_CLIPS,
        "cfg": B2_CFG,
        "schedule": B2_SCHEDULE,
        "alpha": B2_ALPHA,
        "k_forks": B2_K_FORKS,
        "variant": "small_16k",
        "duration_sec": 8.0,
        "num_steps": 20,
        "conditioning": "full_video_clip_synchformer_empty_text",
        "audio_format": "WAV",
        "audio_subtype": "FLOAT",
        "sample_rate": 16000,
        "expected_frames": 128000,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"B2 manifest {key}={manifest.get(key)!r}; expected {value!r}")
    base_seeds = tuple(int(seed) for seed in manifest.get("base_seeds", []))
    if base_seeds not in (
        B2_BASE_SEEDS,
        B2_EXTENSION_BASE_SEEDS,
        B2_BACKFILL_BASE_SEEDS,
        B2_SECOND_BACKFILL_BASE_SEEDS,
    ):
        raise ValueError("B2 manifest has an unregistered base-seed series")
    if tuple(float(s) for s in manifest.get("s_grid", [])) != B2_S_GRID:
        raise ValueError("B2 manifest has the wrong s-grid")
    clips = [str(clip) for clip in manifest.get("clips", [])]
    if len(clips) != B2_N_CLIPS or len(set(clips)) != B2_N_CLIPS:
        raise ValueError("B2 manifest must contain 48 unique clips")
    counts = manifest.get("expected_artifacts", {})
    if counts != {
        "base_units": B2_N_CLIPS * len(base_seeds),
        "base_wavs": B2_N_CLIPS * len(base_seeds),
        "fork_cells": B2_N_CLIPS * len(base_seeds) * len(B2_S_GRID),
        "fork_wavs": B2_N_CLIPS * len(base_seeds) * len(B2_S_GRID) * B2_K_FORKS,
    }:
        raise ValueError("B2 manifest artifact cardinalities are inconsistent")


def valid_b1_bundle(path: Path) -> bool:
    """Validate only the frozen B1 array schema and numeric integrity."""
    try:
        with np.load(path, allow_pickle=False) as z:
            required = {"token_mean", "token_mean_max", "tokens_sub",
                        "xattn_clip", "xattn_frac"}
            if set(z.files) != required:
                return False
            mean = z["token_mean"]
            meanmax = z["token_mean_max"]
            tokens = z["tokens_sub"]
            xattn = z["xattn_clip"]
            frac = z["xattn_frac"]
            if mean.ndim != 2 or mean.shape != (12, 448):
                return False
            if meanmax.shape != (12, 896):
                return False
            if tokens.shape != (12, 64, 448):
                return False
            if xattn.ndim != 2 or xattn.shape[0] != 4:
                return False
            if frac.shape != (4,):
                return False
            return all(np.isfinite(a).all() for a in (mean, meanmax, tokens, xattn, frac))
    except (OSError, ValueError, KeyError):
        return False


def valid_b2_bundle(path: Path) -> bool:
    """Validate a raw conditioning bundle without deriving any probe statistic."""
    try:
        with np.load(path, allow_pickle=False) as z:
            required = {"pooled", "clip_f", "sync_f", "clip_f_c",
                        "cond_keys", "raw_shapes"}
            if set(z.files) != required:
                return False
            if z["pooled"].shape != (2688,):
                return False
            if any(z[key].shape != (896,) for key in ("clip_f", "sync_f", "clip_f_c")):
                return False
            if tuple(z["cond_keys"].tolist()) != ("clip_f", "sync_f", "clip_f_c"):
                return False
            return all(np.isfinite(z[key]).all()
                       for key in ("pooled", "clip_f", "sync_f", "clip_f_c"))
    except (OSError, ValueError, KeyError):
        return False


def load_confident_clip_labels(
    measurements: Path,
    role: str,
    *,
    expected_votes: int = 16,
) -> dict[str, str]:
    """Return a deterministic non-abstain majority class for complete clips.

    Each individual non-abstain cached tagger output has already passed the
    frozen cross-group confidence margin.  Abstentions are discarded; a clip is
    eligible only when all expected cached independent rows are present and at
    least one row is confident.  Count ties are broken lexicographically.
    """
    votes: dict[str, list[str]] = defaultdict(list)
    with Path(measurements).open() as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            extra = row.get("extra") or {}
            if row.get("axis_id") != "class" or extra.get("role") != role:
                continue
            clip = extra.get("clip")
            label = (row.get("target") or {}).get("label")
            if clip is not None and label is not None:
                votes[str(clip)].append(str(label))

    labels: dict[str, str] = {}
    for clip, clip_votes in votes.items():
        if len(clip_votes) != expected_votes:
            continue
        confident = [label for label in clip_votes if label != ABSTAIN]
        if not confident:
            continue
        counts = Counter(confident)
        labels[clip] = sorted(counts, key=lambda label: (-counts[label], label))[0]
    return labels


def _tie_hash(seed: int, cfg: float, source: str, donor: str) -> str:
    text = f"arc4-b6-v1|{seed}|{cfg:g}|{source}|{donor}"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def select_balanced_pairs(
    labels: dict[str, str],
    *,
    cfg: float,
    n_pairs: int = 128,
    seed: int = 0,
) -> list[dict]:
    """Select deterministic unique ordered cross-class pairs.

    Greedy scores first balance source-class and donor-class marginals, then
    class-pair cells, then individual clip reuse.  A seeded SHA256 order breaks
    otherwise exact ties.  The full candidate set is fixed before selection.
    """
    candidates = [
        (source, donor)
        for source in sorted(labels)
        for donor in sorted(labels)
        if source != donor and labels[source] != labels[donor]
    ]
    if len(candidates) < n_pairs:
        raise ValueError(
            f"only {len(candidates)} unique confident cross-class pairs; need {n_pairs}")

    src_class = Counter()
    donor_class = Counter()
    class_pair = Counter()
    src_clip = Counter()
    donor_clip = Counter()
    remaining = set(candidates)
    selected: list[dict] = []

    for index in range(n_pairs):
        def score(pair: tuple[str, str]):
            source, donor = pair
            sl, dl = labels[source], labels[donor]
            return (
                src_class[sl],
                donor_class[dl],
                class_pair[(sl, dl)],
                src_clip[source],
                donor_clip[donor],
                src_clip[source] + donor_clip[source]
                + src_clip[donor] + donor_clip[donor],
                _tie_hash(seed, cfg, source, donor),
            )

        source, donor = min(remaining, key=score)
        remaining.remove((source, donor))
        sl, dl = labels[source], labels[donor]
        src_class[sl] += 1
        donor_class[dl] += 1
        class_pair[(sl, dl)] += 1
        src_clip[source] += 1
        donor_clip[donor] += 1
        selected.append({
            "pair_id": f"cfg{cfg:g}_pair{index:03d}",
            "cfg": float(cfg),
            "source": source,
            "donor": donor,
            "source_cached_label": sl,
            "donor_cached_label": dl,
        })
    return selected


def validate_pair_manifest(manifest: dict, *, expected_pairs_per_cfg: int = 128) -> None:
    if manifest.get("seed") != 0:
        raise ValueError("B6 pair manifest seed must be 0")
    if tuple(float(s) for s in manifest.get("s_grid", [])) != B6_S_GRID:
        raise ValueError("B6 pair manifest has the wrong frozen s-grid")
    by_cfg: dict[float, list[dict]] = defaultdict(list)
    ids = set()
    for pair in manifest.get("pairs", []):
        pair_id = str(pair["pair_id"])
        if pair_id in ids:
            raise ValueError(f"duplicate pair_id {pair_id}")
        ids.add(pair_id)
        cfg = float(pair["cfg"])
        if pair["source"] == pair["donor"]:
            raise ValueError(f"self pair {pair_id}")
        sl = pair["source_cached_label"]
        dl = pair["donor_cached_label"]
        if sl == ABSTAIN or dl == ABSTAIN or sl == dl:
            raise ValueError(f"non-confident or same-class pair {pair_id}")
        by_cfg[cfg].append(pair)
    if set(by_cfg) != {1.0, 4.5}:
        raise ValueError(f"expected cfgs 1.0 and 4.5; got {sorted(by_cfg)}")
    for cfg, pairs in by_cfg.items():
        if len(pairs) != expected_pairs_per_cfg:
            raise ValueError(
                f"cfg={cfg:g}: expected {expected_pairs_per_cfg} pairs, got {len(pairs)}")
        ordered = {(pair["source"], pair["donor"]) for pair in pairs}
        if len(ordered) != len(pairs):
            raise ValueError(f"cfg={cfg:g}: duplicate ordered source/donor pair")


def class_marginal_range(pairs: Iterable[dict], key: str) -> int:
    """Design-only balance diagnostic used by tests and manifest validation."""
    counts = Counter(str(pair[key]) for pair in pairs)
    return max(counts.values()) - min(counts.values()) if counts else 0
