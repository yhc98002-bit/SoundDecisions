"""Fail-closed B2 Class measurement and exploratory multi-seed analysis.

This module is deliberately split from the B2 generation queue.  It can only
inventory already-journaled WAVs, measure immutable assignments, validate and
reduce immutable shards, and analyse synthetic or measured posterior bundles.
There is no generation or replay entry point here.

The continuity decision is the frozen legacy rule: event-restricted PANNs
top-1, mapped through ``configs/coarse_class_map.json``, with a cross-group
margin abstention at delta=0.05.  Full 527-way sigmoid outputs are retained.
For auditability the derived coarse posterior is the normalized sum of all 527
class scores in each frozen coarse group.  The hard decision remains the
registered event-restricted top-1 rule; it is not a group-sum argmax.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import platform
import socket
import subprocess
import tempfile
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from .agreement import confident_agreement
from .real_measurer import ABSTAIN, CLASS_ABSTAIN_DELTA, load_coarse_map
from .types import AgreementMetric


INVENTORY_SCHEMA = "sounddecisions.b2_inventory.v1"
POSTERIOR_SCHEMA = "sounddecisions.b2_class_posterior.v1"
MERGE_SCHEMA = "sounddecisions.b2_class_posterior_merge.v1"
ANALYSIS_SCHEMA = "sounddecisions.b2_class_multiseed.v1"
PROTOCOL_SCHEMA = "sounddecisions_non_human_closure_protocol_v1"
ABSTENTION_RULE_ID = "legacy_event_top1_cross_group_margin_delta_0.05_v1"
COARSE_SCORE_SUMS_RULE_ID = "group_sum_all_527_sigmoid_outputs_v1"
COARSE_POSTERIOR_RULE_ID = "normalized_group_sum_all_527_v1"
EXPECTED_S_GRID = (0.05, 0.15, 0.25, 0.35, 0.45, 0.60, 0.75, 0.90)
EXPECTED_SEEDS = tuple(range(17))
EXPECTED_SEED_SERIES = (
    (0, 1, 2, 3, 4),
    (5, 6, 7, 8),
    (9, 10, 11, 12),
    (13, 14, 15, 16),
)
EXPECTED_N_CLIPS = 48
EXPECTED_K_FORKS = 12
EXPECTED_527 = 527
SENSITIVITY_THRESHOLDS = (0.50, 0.60, 0.70, 0.80, 0.90)
DEFAULT_BOOTSTRAP_DRAWS = 5000
DEFAULT_BOOTSTRAP_SEED = 20260717
VIDEO_DETERMINED_MIN = 0.90
PINNED_CHECKPOINT_SHA256 = (
    "e2ee543a27919542c2ea03eabaa70b24dcd4e6c8e05621de6b67a94e4c5058e6"
)
PINNED_PROTOCOL_SHA256 = (
    "5c4fc4025995c16e355feb8cc02fbb3627891d47f6df052becde4845eaa7bd09"
)
PINNED_COARSE_MAP_SHA256 = (
    "55b5a1d4116caa4503a6b4b17192425da487a9c4385a287e343d850795be4fe7"
)
PINNED_HISTORICAL_COMPARATOR_SHA256 = (
    "4d191b1c53ce21c4e9b68b9c36e385b0fc7ea2edd607fc528e2e034047773bc0"
)
PINNED_HISTORICAL_S_COMMIT = 0.34593023255813954
EXPECTED_GENERATION_MANIFEST_SHA256 = frozenset(
    {
        "5c3a334ecfcfb3e91504354c14c8e8dbae71b3bade088b21bec26fb06fd68ed3",
        "b6e176949f531528ccb669759d2057fa0b1b1a14567633d3dd6a2d47e0a8a9e4",
        "72bcd677376b1ca44278d7cb6e9ea61910cc07f06fc135b239cdbb54aa4ee6ee",
        "ae3dfb2e0022043206d8d4fcf748a3ce68ba1a3898af560ff3e059ba217e3c51",
    }
)
EXPECTED_GENERATION_REVISIONS = frozenset(
    {
        "dbd40d94d4867a53bdaad6d2524f4534817fddbf",
        "dd7fdc006fe1f5b3baca4024854d37d533606f74",
        "6ec5c0dbdfb2b45ca8a27d2a193015d97607d8db",
        "07718809024a674bb938684e6cfdc520026d3122",
    }
)
ALLOWED_SCIENTIFIC_STATUSES = frozenset(
    {
        "SUPPORTED_EXPLORATORILY",
        "NOT_SUPPORTED",
        "UNRESOLVED",
        "NOT_TESTED",
        "ENGINEERING_FAILURE",
        "INCOMPLETE_ARTIFACTS",
    }
)
REPLICATION_LABELS = (
    "stable_across_seeds",
    "heterogeneous_but_directionally_consistent",
    "strongly_seed_dependent",
    "not_reproduced",
)
CUDA_DETERMINISTIC_WORKSPACE_CONFIGS = frozenset({":4096:8", ":16:8"})


class B2ClosureError(RuntimeError):
    """Raised when an integrity or scientific-contract check fails."""


def validate_cuda_determinism_environment(device: str) -> str | None:
    """Fail before model construction if deterministic CUDA GEMM is not configured."""
    if not str(device).startswith("cuda"):
        return None
    observed = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if observed not in CUDA_DETERMINISTIC_WORKSPACE_CONFIGS:
        allowed = ", ".join(sorted(CUDA_DETERMINISTIC_WORKSPACE_CONFIGS))
        raise B2ClosureError(
            "deterministic CUDA PANNs requires CUBLAS_WORKSPACE_CONFIG to be one of "
            f"{allowed}; observed {observed!r}"
        )
    return observed


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json_bytes(payload: Any, *, indent: int | None = 2) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            indent=indent,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":") if indent is None else None,
        )
        + "\n"
    ).encode("utf-8")


def _atomic_create(path: Path, writer: Callable[[Any], None], *, binary: bool) -> Path:
    """Atomically create ``path`` and refuse to replace an existing artifact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_tmp = tempfile.mkstemp(prefix=f".{path.name}.tmp.", dir=path.parent)
    os.close(fd)
    tmp = Path(raw_tmp)
    try:
        mode = "wb" if binary else "w"
        kwargs = {} if binary else {"encoding": "utf-8", "newline": ""}
        with tmp.open(mode, **kwargs) as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


def atomic_json_create(path: Path, payload: Any) -> Path:
    data = canonical_json_bytes(payload)
    return _atomic_create(path, lambda handle: handle.write(data), binary=True)


def atomic_jsonl_create(path: Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    def write(handle: Any) -> None:
        for row in rows:
            handle.write(canonical_json_bytes(dict(row), indent=None))

    return _atomic_create(path, write, binary=True)


def atomic_csv_create(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> Path:
    def write(handle: Any) -> None:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    return _atomic_create(path, write, binary=False)


def deterministic_npz_create(path: Path, arrays: Mapping[str, np.ndarray]) -> Path:
    """Write a byte-deterministic compressed NPZ with no pickle payloads."""
    normalized: dict[str, np.ndarray] = {}
    for key, value in arrays.items():
        if not key or "/" in key or key.endswith(".npy"):
            raise ValueError(f"invalid NPZ key {key!r}")
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise ValueError(f"object dtype is forbidden for {key}")
        normalized[key] = array

    def write(handle: Any) -> None:
        with zipfile.ZipFile(
            handle, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
        ) as archive:
            for key in sorted(normalized):
                info = zipfile.ZipInfo(f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o600 << 16
                info.create_system = 3
                with archive.open(info, "w", force_zip64=True) as member:
                    np.lib.format.write_array(member, normalized[key], allow_pickle=False)

    return _atomic_create(path, write, binary=True)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise B2ClosureError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise B2ClosureError(f"expected JSON object: {path}")
    return value


def validate_class_protocol(
    protocol_path: Path,
    *,
    checkpoint_path: Path | None = None,
    coarse_map_path: Path | None = None,
    abstain_delta: float = CLASS_ABSTAIN_DELTA,
    canonical: bool,
) -> dict[str, Any]:
    """Validate the outcome-independent closure protocol and pinned assets.

    Canonical callers invoke this before constructing a model or output file.
    Noncanonical use is an explicit synthetic-test path and still records a
    supplied protocol hash when one is provided.
    """
    path = Path(protocol_path)
    protocol = _load_json(path)
    digest = sha256_file(path)
    if not canonical:
        return {"path": str(path.resolve()), "sha256": digest, "payload": protocol}
    if protocol.get("schema") != PROTOCOL_SCHEMA:
        raise B2ClosureError(f"unexpected closure protocol schema: {protocol.get('schema')!r}")
    if digest != PINNED_PROTOCOL_SHA256:
        raise B2ClosureError("canonical closure protocol hash does not match the frozen Git artifact")
    if protocol.get("status") != "EXPLORATORY_CONTINUITY":
        raise B2ClosureError("canonical Class measurement requires exploratory continuity protocol")
    freeze = protocol.get("outcome_freeze", {})
    if freeze.get("b2_posteriors_inspected") is not False:
        raise B2ClosureError("protocol is not outcome-blind for B2 posteriors")
    config = protocol.get("class_measurement", {})
    expected = {
        "checkpoint_sha256": PINNED_CHECKPOINT_SHA256,
        "coarse_map_path": "configs/coarse_class_map.json",
        "coarse_map_sha256": PINNED_COARSE_MAP_SHA256,
        "decision_rule": "legacy_event_restricted_top1_then_map_cross_group_margin",
        "abstain_delta": CLASS_ABSTAIN_DELTA,
        "full_posterior_dimension": EXPECTED_527,
        "unscorable_if_confident_lt": 2,
        "theta_commit": 0.7,
        "video_determined_if_baseline_gte": VIDEO_DETERMINED_MIN,
        "sensitivity_thresholds": list(SENSITIVITY_THRESHOLDS),
        "progress": list(EXPECTED_S_GRID),
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise B2ClosureError(
                f"canonical protocol Class field {key}={config.get(key)!r}; expected {value!r}"
            )
    bootstrap = config.get("bootstrap", {})
    if (
        bootstrap.get("unit") != "video"
        or int(bootstrap.get("draws", -1)) != DEFAULT_BOOTSTRAP_DRAWS
        or int(bootstrap.get("seed", -1)) != DEFAULT_BOOTSTRAP_SEED
        or bootstrap.get("interval") != [0.025, 0.975]
    ):
        raise B2ClosureError("canonical protocol bootstrap settings are not pinned")
    labels = config.get("exploratory_replication_labels", {})
    if tuple(labels) != REPLICATION_LABELS:
        raise B2ClosureError("canonical protocol replication labels/order changed")
    comparator = config.get("historical_comparators", {})
    if not math.isclose(
        float(comparator.get("canonical_cfg1_confident_s_commit", float("nan"))),
        PINNED_HISTORICAL_S_COMMIT,
        abs_tol=1e-15,
        rel_tol=0.0,
    ):
        raise B2ClosureError("canonical historical Class comparator changed")
    if not math.isclose(float(abstain_delta), CLASS_ABSTAIN_DELTA, abs_tol=0.0):
        raise B2ClosureError("canonical measurement must use abstain delta 0.05")
    if checkpoint_path is not None:
        checkpoint = Path(checkpoint_path)
        if not checkpoint.is_file() or sha256_file(checkpoint) != PINNED_CHECKPOINT_SHA256:
            raise B2ClosureError("canonical measurement checkpoint is absent or has the wrong hash")
    if coarse_map_path is not None:
        coarse_map = Path(coarse_map_path)
        if not coarse_map.is_file() or sha256_file(coarse_map) != PINNED_COARSE_MAP_SHA256:
            raise B2ClosureError("canonical measurement coarse map is absent or has the wrong hash")
    return {"path": str(path.resolve()), "sha256": digest, "payload": protocol}


def _safe_resolve(root: Path, relative: str) -> Path:
    rel = Path(relative)
    if rel.is_absolute():
        raise B2ClosureError(f"journal path must be relative: {relative}")
    root_resolved = root.resolve()
    path = (root / rel).resolve()
    if not path.is_relative_to(root_resolved):
        raise B2ClosureError(f"journal path escapes root: {relative}")
    return path


def _check_manifest_sidecar(path: Path) -> str:
    digest = sha256_file(path)
    sidecar = path.with_suffix(".sha256")
    if not sidecar.is_file():
        raise B2ClosureError(f"missing frozen manifest sidecar: {sidecar}")
    fields = sidecar.read_text(encoding="utf-8").strip().split()
    if fields != [digest, path.name]:
        raise B2ClosureError(f"frozen manifest sidecar mismatch: {sidecar}")
    return digest


def _validate_generation_manifest(manifest: Mapping[str, Any], *, canonical: bool) -> None:
    required = {
        "schema_version",
        "clips",
        "base_seeds",
        "s_grid",
        "k_forks",
        "cfg",
        "alpha",
        "schedule",
        "variant",
        "sample_rate",
        "expected_frames",
        "audio_subtype",
        "expected_artifacts",
    }
    missing = sorted(required - set(manifest))
    if missing:
        raise B2ClosureError(f"generation manifest missing fields: {missing}")
    clips = [str(value) for value in manifest["clips"]]
    seeds = [int(value) for value in manifest["base_seeds"]]
    grid = [float(value) for value in manifest["s_grid"]]
    if len(clips) != len(set(clips)) or not clips:
        raise B2ClosureError("generation manifest clips are empty or duplicated")
    if len(seeds) != len(set(seeds)) or not seeds:
        raise B2ClosureError("generation manifest seeds are empty or duplicated")
    if len(grid) != len(set(grid)) or not grid:
        raise B2ClosureError("generation manifest progress grid is empty or duplicated")
    k = int(manifest["k_forks"])
    expected_counts = {
        "base_units": len(clips) * len(seeds),
        "base_wavs": len(clips) * len(seeds),
        "fork_cells": len(clips) * len(seeds) * len(grid),
        "fork_wavs": len(clips) * len(seeds) * len(grid) * k,
    }
    if manifest["expected_artifacts"] != expected_counts:
        raise B2ClosureError("generation manifest cardinalities are inconsistent")
    if canonical:
        expected = {
            "schema_version": 1,
            "cfg": 4.5,
            "alpha": 0.8,
            "schedule": "sqrt_down",
            "variant": "small_16k",
            "sample_rate": 16000,
            "expected_frames": 128000,
            "audio_subtype": "FLOAT",
            "k_forks": EXPECTED_K_FORKS,
        }
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise B2ClosureError(
                    f"noncanonical B2 {key}: {manifest.get(key)!r} != {value!r}"
                )
        if len(clips) != EXPECTED_N_CLIPS:
            raise B2ClosureError(f"canonical B2 requires {EXPECTED_N_CLIPS} clips")
        if tuple(grid) != EXPECTED_S_GRID:
            raise B2ClosureError("canonical B2 progress grid mismatch")
        if tuple(seeds) not in EXPECTED_SEED_SERIES:
            raise B2ClosureError(f"unregistered canonical seed series: {seeds}")


def _validate_wav_header(path: Path, saved: Mapping[str, Any]) -> None:
    try:
        import soundfile as sf
    except ImportError as exc:  # pragma: no cover - production environment has it
        raise B2ClosureError("soundfile is required for WAV header validation") from exc
    info = sf.info(str(path))
    actual = {
        "sample_rate": int(info.samplerate),
        "frames": int(info.frames),
        "channels": int(info.channels),
        "format": info.format,
        "subtype": info.subtype,
    }
    for key, value in actual.items():
        if saved.get(key) != value:
            raise B2ClosureError(
                f"WAV header mismatch {path}: {key}={value!r}, journal={saved.get(key)!r}"
            )


def _validate_artifact(
    root: Path,
    saved: Mapping[str, Any],
    *,
    expected_path: str,
    expected_role: str,
    expected_progress: float | None,
    expected_fork: int | None,
    verify_wav_headers: bool,
) -> Path:
    if saved.get("path") != expected_path or saved.get("role") != expected_role:
        raise B2ClosureError(f"artifact identity mismatch under {root}: {saved}")
    if expected_progress is not None and not math.isclose(
        float(saved.get("s", -1.0)), expected_progress, abs_tol=1e-12
    ):
        raise B2ClosureError(f"artifact progress mismatch: {saved}")
    if expected_fork is not None and int(saved.get("fork_index", -1)) != expected_fork:
        raise B2ClosureError(f"artifact fork index mismatch: {saved}")
    path = _safe_resolve(root, expected_path)
    if not path.is_file():
        raise B2ClosureError(f"missing banked WAV: {path}")
    actual_sha = sha256_file(path)
    if saved.get("sha256") != actual_sha:
        raise B2ClosureError(f"banked WAV hash mismatch: {path}")
    if int(saved.get("bytes", -1)) != path.stat().st_size:
        raise B2ClosureError(f"banked WAV byte-count mismatch: {path}")
    if verify_wav_headers:
        _validate_wav_header(path, saved)
    return path


def _validate_artifact_design(saved: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    expected = {
        "sample_rate": int(manifest["sample_rate"]),
        "frames": int(manifest["expected_frames"]),
        "channels": 1,
        "format": "WAV",
        "subtype": str(manifest["audio_subtype"]),
    }
    for key, value in expected.items():
        if saved.get(key) != value:
            raise B2ClosureError(
                f"artifact metadata violates generation manifest: "
                f"{key}={saved.get(key)!r}, expected {value!r}"
            )


def _record_id(video_id: str, seed: int, role: str, progress: float | None, fork: int | None) -> str:
    if role == "base":
        return f"{video_id}__seed{seed}__base"
    assert progress is not None and fork is not None
    return f"{video_id}__seed{seed}__s{progress:.2f}__fork{fork:02d}"


def inventory_b2_roots(
    roots: Sequence[Path],
    *,
    canonical: bool = True,
    verify_wav_headers: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate generation/unit/cell journals and return immutable WAV records.

    The function hashes every referenced WAV and rejects missing, duplicated,
    unjournaled, or path-escaping artifacts.  It never writes inside a B2 root.
    """
    if not roots:
        raise B2ClosureError("at least one B2 root is required")
    root_rows: list[dict[str, Any]] = []
    all_records: list[dict[str, Any]] = []
    seen_seeds: set[int] = set()
    reference_design: dict[str, Any] | None = None

    for root_input in roots:
        root = Path(root_input).resolve()
        manifest_path = root / "generation_manifest.json"
        if not manifest_path.is_file():
            raise B2ClosureError(f"missing generation manifest: {manifest_path}")
        manifest_sha = _check_manifest_sidecar(manifest_path)
        manifest = _load_json(manifest_path)
        _validate_generation_manifest(manifest, canonical=canonical)
        clips = [str(value) for value in manifest["clips"]]
        seeds = [int(value) for value in manifest["base_seeds"]]
        grid = [float(value) for value in manifest["s_grid"]]
        k_forks = int(manifest["k_forks"])
        overlap = seen_seeds.intersection(seeds)
        if overlap:
            raise B2ClosureError(f"base seeds occur in multiple roots: {sorted(overlap)}")
        seen_seeds.update(seeds)
        design = {
            key: manifest[key]
            for key in (
                "clips",
                "s_grid",
                "k_forks",
                "cfg",
                "alpha",
                "schedule",
                "variant",
                "sample_rate",
                "expected_frames",
                "audio_subtype",
            )
        }
        if reference_design is None:
            reference_design = design
        elif design != reference_design:
            raise B2ClosureError(f"B2 roots mix incompatible designs: {root}")

        expected_units: set[Path] = set()
        expected_cells: set[Path] = set()
        expected_wavs: set[Path] = set()
        journal_hashes: list[str] = []
        root_records_before = len(all_records)
        for video_id in clips:
            for seed in seeds:
                unit_rel = Path("journal") / "units" / f"{video_id}__seed{seed}.json"
                unit_path = root / unit_rel
                expected_units.add(unit_path)
                unit = _load_json(unit_path)
                unit_sha = sha256_file(unit_path)
                journal_hashes.append(unit_sha)
                if (
                    str(unit.get("clip")) != video_id
                    or int(unit.get("base_seed", -1)) != seed
                    or unit.get("generation_manifest_sha256") != manifest_sha
                    or [float(value) for value in unit.get("s_grid", [])] != grid
                    or int(unit.get("fork_wavs", -1)) != len(grid) * k_forks
                    or float(unit.get("cfg", manifest["cfg"])) != float(manifest["cfg"])
                    or float(unit.get("alpha", manifest["alpha"])) != float(manifest["alpha"])
                    or str(unit.get("schedule", manifest["schedule"]))
                    != str(manifest["schedule"])
                ):
                    raise B2ClosureError(f"unit journal design mismatch: {unit_path}")
                generation_revision = str(unit.get("provenance", {}).get("git_commit", ""))
                if not generation_revision:
                    raise B2ClosureError(f"unit journal lacks generation revision: {unit_path}")
                base_rel = f"raw/{video_id}/seed{seed}/base.wav"
                base_saved = unit.get("base_artifact")
                if not isinstance(base_saved, dict):
                    raise B2ClosureError(f"unit journal lacks base artifact: {unit_path}")
                _validate_artifact_design(base_saved, manifest)
                base_path = _validate_artifact(
                    root,
                    base_saved,
                    expected_path=base_rel,
                    expected_role="base",
                    expected_progress=None,
                    expected_fork=None,
                    verify_wav_headers=verify_wav_headers,
                )
                expected_wavs.add(base_path)
                all_records.append(
                    {
                        "record_id": _record_id(video_id, seed, "base", None, None),
                        "video_id": video_id,
                        "base_seed": seed,
                        "role": "base",
                        "fork_index": None,
                        "progress": None,
                        "audio_path": str(base_path),
                        "audio_sha256": str(base_saved["sha256"]),
                        "audio_bytes": int(base_saved["bytes"]),
                        "sample_rate": int(base_saved["sample_rate"]),
                        "frames": int(base_saved["frames"]),
                        "audio_subtype": str(base_saved["subtype"]),
                        "source_root": str(root),
                        "generation_manifest_sha256": manifest_sha,
                        "source_unit_journal": str(unit_path),
                        "source_unit_journal_sha256": unit_sha,
                        "source_cell_journal": None,
                        "source_cell_journal_sha256": None,
                        "cfg": float(manifest["cfg"]),
                        "alpha": float(manifest["alpha"]),
                        "schedule": str(manifest["schedule"]),
                        "generation_model_variant": str(manifest["variant"]),
                        "generation_revision": generation_revision,
                    }
                )
                cell_refs = unit.get("cell_journals")
                if not isinstance(cell_refs, list) or len(cell_refs) != len(grid):
                    raise B2ClosureError(f"unit journal cell-reference mismatch: {unit_path}")
                refs_by_s: dict[float, Mapping[str, Any]] = {}
                for ref in cell_refs:
                    progress = float(ref.get("s", -1.0))
                    if progress in refs_by_s:
                        raise B2ClosureError(f"duplicate cell reference: {unit_path}")
                    refs_by_s[progress] = ref
                if set(refs_by_s) != set(grid):
                    raise B2ClosureError(f"unit journal progress-reference mismatch: {unit_path}")
                for progress in grid:
                    cell_rel = Path("journal") / "cells" / (
                        f"{video_id}__seed{seed}__s{progress:.2f}.json"
                    )
                    cell_path = root / cell_rel
                    expected_cells.add(cell_path)
                    ref = refs_by_s[progress]
                    if ref.get("journal") != cell_rel.as_posix():
                        raise B2ClosureError(f"cell journal path mismatch: {unit_path}")
                    cell_sha = sha256_file(cell_path)
                    if ref.get("sha256") != cell_sha:
                        raise B2ClosureError(f"cell journal hash mismatch: {cell_path}")
                    journal_hashes.append(cell_sha)
                    cell = _load_json(cell_path)
                    if (
                        str(cell.get("clip")) != video_id
                        or int(cell.get("base_seed", -1)) != seed
                        or not math.isclose(float(cell.get("s", -1)), progress, abs_tol=1e-12)
                        or cell.get("generation_manifest_sha256") != manifest_sha
                        or int(cell.get("k_forks", -1)) != k_forks
                        or float(cell.get("cfg", manifest["cfg"])) != float(manifest["cfg"])
                        or float(cell.get("alpha", manifest["alpha"])) != float(manifest["alpha"])
                        or str(cell.get("schedule", manifest["schedule"]))
                        != str(manifest["schedule"])
                    ):
                        raise B2ClosureError(f"cell journal design mismatch: {cell_path}")
                    artifacts = cell.get("artifacts")
                    if not isinstance(artifacts, list) or len(artifacts) != k_forks:
                        raise B2ClosureError(f"cell artifact cardinality mismatch: {cell_path}")
                    for fork_index, saved in enumerate(artifacts):
                        _validate_artifact_design(saved, manifest)
                        fork_rel = (
                            f"raw/{video_id}/seed{seed}/s{progress:.2f}/"
                            f"fork{fork_index:02d}.wav"
                        )
                        fork_path = _validate_artifact(
                            root,
                            saved,
                            expected_path=fork_rel,
                            expected_role="fork",
                            expected_progress=progress,
                            expected_fork=fork_index,
                            verify_wav_headers=verify_wav_headers,
                        )
                        expected_wavs.add(fork_path)
                        all_records.append(
                            {
                                "record_id": _record_id(
                                    video_id, seed, "fork", progress, fork_index
                                ),
                                "video_id": video_id,
                                "base_seed": seed,
                                "role": "fork",
                                "fork_index": fork_index,
                                "progress": progress,
                                "audio_path": str(fork_path),
                                "audio_sha256": str(saved["sha256"]),
                                "audio_bytes": int(saved["bytes"]),
                                "sample_rate": int(saved["sample_rate"]),
                                "frames": int(saved["frames"]),
                                "audio_subtype": str(saved["subtype"]),
                                "source_root": str(root),
                                "generation_manifest_sha256": manifest_sha,
                                "source_unit_journal": str(unit_path),
                                "source_unit_journal_sha256": unit_sha,
                                "source_cell_journal": str(cell_path),
                                "source_cell_journal_sha256": cell_sha,
                                "cfg": float(manifest["cfg"]),
                                "alpha": float(manifest["alpha"]),
                                "schedule": str(manifest["schedule"]),
                                "generation_model_variant": str(manifest["variant"]),
                                "generation_revision": generation_revision,
                            }
                        )

        actual_units = set((root / "journal" / "units").glob("*.json"))
        actual_cells = set((root / "journal" / "cells").glob("*.json"))
        actual_wavs = set((root / "raw").glob("**/*.wav"))
        if actual_units != expected_units:
            raise B2ClosureError(
                f"unit journal set mismatch under {root}: missing={len(expected_units-actual_units)}, "
                f"extra={len(actual_units-expected_units)}"
            )
        if actual_cells != expected_cells:
            raise B2ClosureError(
                f"cell journal set mismatch under {root}: missing={len(expected_cells-actual_cells)}, "
                f"extra={len(actual_cells-expected_cells)}"
            )
        if actual_wavs != expected_wavs:
            raise B2ClosureError(
                f"WAV set mismatch under {root}: missing={len(expected_wavs-actual_wavs)}, "
                f"extra={len(actual_wavs-expected_wavs)}"
            )
        root_rows.append(
            {
                "root": str(root),
                "generation_manifest": str(manifest_path),
                "generation_manifest_sha256": manifest_sha,
                "base_seeds": seeds,
                "records": len(all_records) - root_records_before,
                "unit_journals": len(expected_units),
                "cell_journals": len(expected_cells),
                "base_wavs": len(clips) * len(seeds),
                "fork_wavs": len(clips) * len(seeds) * len(grid) * k_forks,
                "journal_hash_set_sha256": sha256_bytes(
                    ("\n".join(sorted(journal_hashes)) + "\n").encode("ascii")
                ),
            }
        )

    if canonical and (
        len(root_rows) != 4 or tuple(sorted(seen_seeds)) != EXPECTED_SEEDS
    ):
        raise B2ClosureError(
            f"canonical inventory requires four roots and seeds 0..16; "
            f"got roots={len(root_rows)}, seeds={sorted(seen_seeds)}"
        )
    all_records.sort(key=lambda row: row["record_id"])
    ids = [str(row["record_id"]) for row in all_records]
    if len(ids) != len(set(ids)):
        duplicates = [key for key, count in Counter(ids).items() if count > 1]
        raise B2ClosureError(f"duplicate inventory record IDs: {duplicates[:3]}")
    if canonical:
        expected_count = EXPECTED_N_CLIPS * len(EXPECTED_SEEDS) * (
            1 + len(EXPECTED_S_GRID) * EXPECTED_K_FORKS
        )
        if len(all_records) != expected_count:
            raise B2ClosureError(
                f"canonical inventory has {len(all_records)} records, expected {expected_count}"
            )
    manifest_summary = {
        "schema_version": INVENTORY_SCHEMA,
        "status": "COMPLETE",
        "read_only_inventory": True,
        "canonical_b2": bool(canonical),
        "verify_wav_headers": bool(verify_wav_headers),
        "roots": sorted(root_rows, key=lambda row: min(row["base_seeds"])),
        "record_count": len(all_records),
        "base_record_count": sum(row["role"] == "base" for row in all_records),
        "fork_record_count": sum(row["role"] == "fork" for row in all_records),
        "video_ids": list(reference_design["clips"] if reference_design else []),
        "base_seeds": sorted(seen_seeds),
        "progress_grid": list(reference_design["s_grid"] if reference_design else []),
        "k_forks": int(reference_design["k_forks"] if reference_design else 0),
        "record_ids_sha256": sha256_bytes(("\n".join(ids) + "\n").encode("utf-8")),
    }
    return all_records, manifest_summary


def write_inventory(
    roots: Sequence[Path],
    out_dir: Path,
    *,
    canonical: bool = True,
    verify_wav_headers: bool = True,
) -> dict[str, Any]:
    records, summary = inventory_b2_roots(
        roots, canonical=canonical, verify_wav_headers=verify_wav_headers
    )
    out_dir = Path(out_dir)
    records_path = out_dir / "B2_WAV_INVENTORY.jsonl"
    manifest_path = out_dir / "B2_WAV_INVENTORY_MANIFEST.json"
    atomic_jsonl_create(records_path, records)
    summary = {
        **summary,
        "records_file": records_path.name,
        "records_sha256": sha256_file(records_path),
        "records_bytes": records_path.stat().st_size,
        "inventory_code_sha256": sha256_file(Path(__file__)),
        "inventory_git_commit": _git_commit(Path(__file__).resolve().parents[1]),
    }
    atomic_json_create(manifest_path, summary)
    return {**summary, "manifest_path": str(manifest_path)}


def load_inventory(manifest_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_path = Path(manifest_path)
    manifest = _load_json(manifest_path)
    if manifest.get("schema_version") != INVENTORY_SCHEMA or manifest.get("status") != "COMPLETE":
        raise B2ClosureError(f"not a complete B2 inventory manifest: {manifest_path}")
    records_path = _safe_resolve(manifest_path.parent, str(manifest.get("records_file", "")))
    if sha256_file(records_path) != manifest.get("records_sha256"):
        raise B2ClosureError(f"inventory record hash mismatch: {records_path}")
    records: list[dict[str, Any]] = []
    with records_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise B2ClosureError(
                    f"invalid inventory JSONL {records_path}:{line_number}"
                ) from exc
            if not isinstance(row, dict):
                raise B2ClosureError(f"non-object inventory row at line {line_number}")
            records.append(row)
    if len(records) != int(manifest.get("record_count", -1)):
        raise B2ClosureError("inventory record cardinality mismatch")
    ids = [str(row.get("record_id", "")) for row in records]
    if ids != sorted(ids) or len(ids) != len(set(ids)):
        raise B2ClosureError("inventory record IDs are unsorted or duplicated")
    digest = sha256_bytes(("\n".join(ids) + "\n").encode("utf-8"))
    if digest != manifest.get("record_ids_sha256"):
        raise B2ClosureError("inventory record-ID digest mismatch")
    return records, manifest


def parse_shard(value: str) -> tuple[int, int]:
    try:
        index, count = (int(part) for part in value.split("/", 1))
    except (ValueError, TypeError) as exc:
        raise B2ClosureError(f"invalid shard {value!r}; expected INDEX/COUNT") from exc
    if count <= 0 or not 0 <= index < count:
        raise B2ClosureError(f"invalid shard {value!r}; expected 0 <= INDEX < COUNT")
    return index, count


def assigned_inventory_records(
    records: Sequence[Mapping[str, Any]], shard_index: int, shard_count: int
) -> list[Mapping[str, Any]]:
    if shard_count <= 0 or not 0 <= shard_index < shard_count:
        raise B2ClosureError("invalid shard assignment")
    return [row for index, row in enumerate(records) if index % shard_count == shard_index]


def validate_coarse_map(coarse_map: Mapping[str, Any]) -> tuple[list[str], np.ndarray]:
    names = [str(value) for value in coarse_map.get("coarse_classes", [])]
    mapping = coarse_map.get("index_to_coarse", {})
    if not names or len(names) != len(set(names)):
        raise B2ClosureError("coarse map names are empty or duplicated")
    if set(int(key) for key in mapping) != set(range(EXPECTED_527)):
        raise B2ClosureError("coarse map must assign every AudioSet index 0..526")
    name_to_index = {name: index for index, name in enumerate(names)}
    try:
        lookup = np.asarray(
            [name_to_index[str(mapping[index])] for index in range(EXPECTED_527)],
            dtype=np.int16,
        )
    except KeyError as exc:
        raise B2ClosureError(f"coarse map references an unknown group: {exc}") from exc
    return names, lookup


def derive_coarse_scores(
    probabilities: np.ndarray, coarse_map: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    probs = np.asarray(probabilities, dtype=np.float32)
    if probs.ndim != 2 or probs.shape[1] != EXPECTED_527:
        raise B2ClosureError(f"expected posterior shape (N,527), got {probs.shape}")
    if not np.isfinite(probs).all() or np.any(probs < 0.0) or np.any(probs > 1.0):
        raise B2ClosureError("PANNs probabilities must be finite and in [0,1]")
    names, lookup = validate_coarse_map(coarse_map)
    sums = np.zeros((probs.shape[0], len(names)), dtype=np.float64)
    for class_index, group_index in enumerate(lookup):
        sums[:, int(group_index)] += probs[:, class_index]
    totals = sums.sum(axis=1, keepdims=True)
    if np.any(totals <= 0.0):
        raise B2ClosureError("cannot normalize an all-zero 527-way posterior")
    coarse = (sums / totals).astype(np.float32)
    if not np.allclose(coarse.sum(axis=1), 1.0, atol=1e-6, rtol=0.0):
        raise B2ClosureError("coarse posterior does not conserve normalized mass")
    return sums.astype(np.float32), coarse, names


def derive_coarse_posterior(
    probabilities: np.ndarray, coarse_map: Mapping[str, Any]
) -> tuple[np.ndarray, list[str]]:
    """Backward-compatible normalized coarse posterior helper."""
    _raw_sums, normalized, names = derive_coarse_scores(probabilities, coarse_map)
    return normalized, names


def class_diagnostics_batch(
    probabilities: np.ndarray,
    coarse_map: Mapping[str, Any],
    *,
    abstain_delta: float = CLASS_ABSTAIN_DELTA,
) -> dict[str, np.ndarray]:
    probs = np.asarray(probabilities, dtype=np.float32)
    if probs.ndim != 2 or probs.shape[1] != EXPECTED_527:
        raise B2ClosureError(f"expected posterior shape (N,527), got {probs.shape}")
    names, lookup = validate_coarse_map(coarse_map)
    excluded_groups = set(str(value) for value in coarse_map.get("class_excluded_coarse", []))
    mask = np.ones(EXPECTED_527, dtype=bool)
    for index in coarse_map.get("non_event_indices", []):
        mask[int(index)] = False
    for index, group_index in enumerate(lookup):
        if names[int(group_index)] in excluded_groups:
            mask[index] = False
    event_indices = np.flatnonzero(mask)
    if event_indices.size < 2:
        raise B2ClosureError("coarse map leaves fewer than two event classes")
    n = probs.shape[0]
    top1_index = np.empty(n, dtype=np.int16)
    runner_index = np.empty(n, dtype=np.int16)
    top_class: list[str] = []
    confident_label: list[str] = []
    confidence = np.empty(n, dtype=np.float32)
    margin = np.empty(n, dtype=np.float32)
    entropy = np.empty(n, dtype=np.float32)
    concentration = np.empty(n, dtype=np.float32)
    abstain = np.empty(n, dtype=bool)
    for row_index, row in enumerate(probs):
        event_probs = row[event_indices]
        # Match RealFoleyMeasurer.class_diagnostics exactly, including NumPy's
        # default argsort tie handling.  Ties remain visible via the margin.
        order = np.argsort(event_probs)[::-1]
        top = int(event_indices[int(order[0])])
        top_group_index = int(lookup[top])
        runner = -1
        for position in order[1:]:
            candidate = int(event_indices[int(position)])
            if int(lookup[candidate]) != top_group_index:
                runner = candidate
                break
        if runner < 0:
            raise B2ClosureError("no cross-group runner-up exists")
        top_prob = float(row[top])
        runner_prob = float(row[runner])
        row_margin = top_prob - runner_prob
        if row_margin < -1e-7:
            raise B2ClosureError("negative cross-group margin")
        total = float(event_probs.sum(dtype=np.float64))
        if total <= 0.0:
            normalized = np.full(event_probs.shape, 1.0 / event_probs.size)
        else:
            normalized = event_probs.astype(np.float64) / total
        row_entropy = -float(np.sum(normalized * np.log(np.maximum(normalized, 1e-12))))
        group = names[top_group_index]
        is_abstain = row_margin < abstain_delta
        top1_index[row_index] = top
        runner_index[row_index] = runner
        top_class.append(group)
        confident_label.append(ABSTAIN if is_abstain else group)
        confidence[row_index] = top_prob
        margin[row_index] = max(0.0, row_margin)
        entropy[row_index] = row_entropy
        concentration[row_index] = top_prob / max(total, 1e-12)
        abstain[row_index] = is_abstain
    return {
        "top1_index": top1_index,
        "cross_group_runner_index": runner_index,
        "top_class": _unicode_array(top_class),
        "confident_label": _unicode_array(confident_label),
        "confidence": confidence,
        "margin": margin,
        "entropy": entropy,
        "concentration": concentration,
        "abstain": abstain,
    }


def _unicode_array(values: Sequence[Any]) -> np.ndarray:
    strings = [str(value) for value in values]
    width = max((len(value) for value in strings), default=1)
    return np.asarray(strings, dtype=f"U{width}")


def _scalar_text(value: str) -> np.ndarray:
    return np.asarray(str(value), dtype=f"U{max(1, len(str(value)))}")


def build_posterior_arrays(
    records: Sequence[Mapping[str, Any]],
    probabilities: np.ndarray,
    *,
    coarse_map: Mapping[str, Any],
    coarse_map_sha256: str,
    tagger_revision: str,
    tagger_checkpoint_sha256: str,
    measurer_revision: str,
    protocol_sha256: str,
    abstain_delta: float = CLASS_ABSTAIN_DELTA,
) -> dict[str, np.ndarray]:
    probs = np.asarray(probabilities, dtype=np.float32)
    if probs.shape != (len(records), EXPECTED_527):
        raise B2ClosureError(
            f"probability cardinality mismatch: {probs.shape} for {len(records)} records"
        )
    coarse_sums, coarse, coarse_names = derive_coarse_scores(probs, coarse_map)
    diagnostics = class_diagnostics_batch(
        probs, coarse_map, abstain_delta=abstain_delta
    )
    arrays: dict[str, np.ndarray] = {
        "schema_version": _scalar_text(POSTERIOR_SCHEMA),
        "record_id": _unicode_array([row["record_id"] for row in records]),
        "video_id": _unicode_array([row["video_id"] for row in records]),
        "base_seed": np.asarray([row["base_seed"] for row in records], dtype=np.int16),
        "role": _unicode_array([row["role"] for row in records]),
        "fork_index": np.asarray(
            [-1 if row["fork_index"] is None else row["fork_index"] for row in records],
            dtype=np.int16,
        ),
        "progress": np.asarray(
            [np.nan if row["progress"] is None else row["progress"] for row in records],
            dtype=np.float32,
        ),
        "audio_path": _unicode_array([row["audio_path"] for row in records]),
        "audio_sha256": _unicode_array([row["audio_sha256"] for row in records]),
        "source_unit_journal_sha256": _unicode_array(
            [row["source_unit_journal_sha256"] for row in records]
        ),
        "source_cell_journal_sha256": _unicode_array(
            [row["source_cell_journal_sha256"] or "" for row in records]
        ),
        "generation_manifest_sha256": _unicode_array(
            [row["generation_manifest_sha256"] for row in records]
        ),
        "generation_revision": _unicode_array(
            [row["generation_revision"] for row in records]
        ),
        "generation_model_variant": _unicode_array(
            [row["generation_model_variant"] for row in records]
        ),
        "cfg": np.asarray([row["cfg"] for row in records], dtype=np.float32),
        "alpha": np.asarray([row["alpha"] for row in records], dtype=np.float32),
        "schedule": _unicode_array([row["schedule"] for row in records]),
        "sample_rate": np.asarray([row["sample_rate"] for row in records], dtype=np.int32),
        "clipwise_output_527": probs,
        "coarse_score_sums": coarse_sums,
        "coarse_posterior": coarse,
        "coarse_class_names": _unicode_array(coarse_names),
        "coarse_map_sha256": _scalar_text(coarse_map_sha256),
        "coarse_map_revision": _scalar_text(str(coarse_map.get("version", ""))),
        "coarse_score_sums_rule_id": _scalar_text(COARSE_SCORE_SUMS_RULE_ID),
        "coarse_posterior_rule_id": _scalar_text(COARSE_POSTERIOR_RULE_ID),
        "abstention_rule_id": _scalar_text(ABSTENTION_RULE_ID),
        "abstain_delta": np.asarray(float(abstain_delta), dtype=np.float32),
        "tagger_revision": _scalar_text(tagger_revision),
        "tagger_checkpoint_sha256": _scalar_text(tagger_checkpoint_sha256),
        "measurer_revision": _scalar_text(measurer_revision),
        "protocol_sha256": _scalar_text(protocol_sha256),
        **diagnostics,
    }
    return arrays


def _git_commit(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _array_schema(arrays: Mapping[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    return {
        key: {"dtype": str(np.asarray(value).dtype), "shape": list(np.asarray(value).shape)}
        for key, value in sorted(arrays.items())
    }


def load_wav_batch(records: Sequence[Mapping[str, Any]]) -> np.ndarray:
    try:
        import soundfile as sf
    except ImportError as exc:  # pragma: no cover
        raise B2ClosureError("soundfile is required for B2 measurement") from exc
    waveforms: list[np.ndarray] = []
    expected_frames: int | None = None
    for row in records:
        path = Path(str(row["audio_path"]))
        if sha256_file(path) != row["audio_sha256"]:
            raise B2ClosureError(f"audio changed after inventory: {path}")
        audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
        if audio.ndim != 1:
            raise B2ClosureError(f"expected mono audio: {path}")
        if int(sample_rate) != int(row["sample_rate"]):
            raise B2ClosureError(f"sample-rate mismatch after inventory: {path}")
        if expected_frames is None:
            expected_frames = int(audio.size)
        if int(audio.size) != expected_frames or int(audio.size) != int(row["frames"]):
            raise B2ClosureError(f"frame-count mismatch after inventory: {path}")
        waveforms.append(np.asarray(audio, dtype=np.float32))
    return np.stack(waveforms, axis=0)


class PannsBatchPredictor:
    """Local-only deterministic PANNs batch predictor."""

    revision = "panns_cnn14_16k_upstream_port_v1"

    def __init__(self, checkpoint: Path, device: str) -> None:
        checkpoint = Path(checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"PANNs checkpoint not found locally: {checkpoint}; downloads are forbidden"
            )
        validate_cuda_determinism_environment(device)
        import torch

        from .measurers_panns_cnn14 import load_cnn14_16k

        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        self.torch = torch
        self.device = device
        self.checkpoint = checkpoint.resolve()
        self.checkpoint_sha256 = sha256_file(self.checkpoint)
        self.model = load_cnn14_16k(self.checkpoint, device=device)

    def __call__(self, waveforms: np.ndarray) -> np.ndarray:
        tensor = self.torch.from_numpy(np.asarray(waveforms, dtype=np.float32)).to(self.device)
        with self.torch.inference_mode():
            output = self.model(tensor)["clipwise_output"]
        probabilities = output.float().cpu().numpy().astype(np.float32, copy=False)
        if probabilities.shape != (waveforms.shape[0], EXPECTED_527):
            raise B2ClosureError(f"unexpected PANNs output shape: {probabilities.shape}")
        return probabilities


def measure_inventory_shard(
    inventory_manifest_path: Path,
    out_dir: Path,
    *,
    protocol_path: Path,
    canonical: bool,
    shard_index: int,
    shard_count: int,
    coarse_map_path: Path,
    posterior_fn: Callable[[np.ndarray], np.ndarray],
    audio_loader: Callable[[Sequence[Mapping[str, Any]]], np.ndarray] = load_wav_batch,
    batch_size: int = 8,
    tagger_revision: str,
    tagger_checkpoint_sha256: str,
    measurer_revision: str,
    abstain_delta: float = CLASS_ABSTAIN_DELTA,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    protocol_info = validate_class_protocol(
        protocol_path,
        coarse_map_path=coarse_map_path,
        abstain_delta=abstain_delta,
        canonical=canonical,
    )
    records, inventory = load_inventory(inventory_manifest_path)
    if canonical:
        if inventory.get("canonical_b2") is not True:
            raise B2ClosureError("canonical measurement requires a canonical B2 inventory")
        if (
            int(inventory.get("record_count", -1)) != 79152
            or int(inventory.get("base_record_count", -1)) != 816
            or int(inventory.get("fork_record_count", -1)) != 78336
            or inventory.get("base_seeds") != list(EXPECTED_SEEDS)
            or inventory.get("progress_grid") != list(EXPECTED_S_GRID)
            or int(inventory.get("k_forks", -1)) != EXPECTED_K_FORKS
        ):
            raise B2ClosureError("canonical B2 inventory cardinality/progress mismatch")
        if tagger_checkpoint_sha256 != PINNED_CHECKPOINT_SHA256:
            raise B2ClosureError("canonical measurement tagger checkpoint hash is not pinned")
        required_provenance = {
            "node",
            "device",
            "command",
            "python_executable",
            "python_version",
            "numpy_version",
            "torch_version",
            "git_commit",
        }
        if not provenance or not required_provenance.issubset(provenance):
            raise B2ClosureError("canonical measurement runtime provenance is incomplete")
    assigned = assigned_inventory_records(records, shard_index, shard_count)
    if not assigned:
        raise B2ClosureError(f"shard {shard_index}/{shard_count} has no assigned records")
    if batch_size <= 0:
        raise B2ClosureError("batch_size must be positive")
    map_path = Path(coarse_map_path)
    coarse_map_sha = sha256_file(map_path)
    coarse_map = load_coarse_map(map_path)
    probabilities: list[np.ndarray] = []
    for start in range(0, len(assigned), batch_size):
        batch_records = assigned[start : start + batch_size]
        waveforms = audio_loader(batch_records)
        posterior = np.asarray(posterior_fn(waveforms), dtype=np.float32)
        if posterior.shape != (len(batch_records), EXPECTED_527):
            raise B2ClosureError(
                f"posterior function returned {posterior.shape}; "
                f"expected {(len(batch_records), EXPECTED_527)}"
            )
        probabilities.append(posterior)
    arrays = build_posterior_arrays(
        assigned,
        np.concatenate(probabilities, axis=0),
        coarse_map=coarse_map,
        coarse_map_sha256=coarse_map_sha,
        tagger_revision=tagger_revision,
        tagger_checkpoint_sha256=tagger_checkpoint_sha256,
        measurer_revision=measurer_revision,
        protocol_sha256=protocol_info["sha256"],
        abstain_delta=abstain_delta,
    )
    out_dir = Path(out_dir)
    stem = f"CLASS_POSTERIOR_SHARD_{shard_index:05d}_OF_{shard_count:05d}"
    data_path = out_dir / f"{stem}.npz"
    completion_path = out_dir / f"{stem}.completion.json"
    deterministic_npz_create(data_path, arrays)
    ids = [str(row["record_id"]) for row in assigned]
    completion = {
        "schema_version": POSTERIOR_SCHEMA,
        "status": "COMPLETE",
        "immutable": True,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "record_count": len(assigned),
        "batch_size": batch_size,
        "record_ids_sha256": sha256_bytes(("\n".join(ids) + "\n").encode("utf-8")),
        "inventory_manifest": str(Path(inventory_manifest_path).resolve()),
        "inventory_manifest_sha256": sha256_file(inventory_manifest_path),
        "inventory_records_sha256": inventory["records_sha256"],
        "canonical_b2": bool(canonical),
        "protocol": protocol_info["path"],
        "protocol_sha256": protocol_info["sha256"],
        "data_file": data_path.name,
        "data_sha256": sha256_file(data_path),
        "data_bytes": data_path.stat().st_size,
        "array_schema": _array_schema(arrays),
        "coarse_map": str(map_path.resolve()),
        "coarse_map_sha256": coarse_map_sha,
        "coarse_map_revision": str(coarse_map["version"]),
        "coarse_score_sums_rule_id": COARSE_SCORE_SUMS_RULE_ID,
        "coarse_posterior_rule_id": COARSE_POSTERIOR_RULE_ID,
        "abstention_rule_id": ABSTENTION_RULE_ID,
        "abstain_delta": float(abstain_delta),
        "tagger_revision": tagger_revision,
        "tagger_checkpoint_sha256": tagger_checkpoint_sha256,
        "measurer_revision": measurer_revision,
        "provenance": dict(provenance or {}),
    }
    atomic_json_create(completion_path, completion)
    return {**completion, "completion_path": str(completion_path)}


RECORD_ARRAY_KEYS = (
    "record_id",
    "video_id",
    "base_seed",
    "role",
    "fork_index",
    "progress",
    "audio_path",
    "audio_sha256",
    "source_unit_journal_sha256",
    "source_cell_journal_sha256",
    "generation_manifest_sha256",
    "generation_revision",
    "generation_model_variant",
    "cfg",
    "alpha",
    "schedule",
    "sample_rate",
    "clipwise_output_527",
    "coarse_score_sums",
    "coarse_posterior",
    "top1_index",
    "cross_group_runner_index",
    "top_class",
    "confident_label",
    "confidence",
    "margin",
    "entropy",
    "concentration",
    "abstain",
)

SCALAR_ARRAY_KEYS = (
    "schema_version",
    "coarse_map_sha256",
    "coarse_map_revision",
    "coarse_score_sums_rule_id",
    "coarse_posterior_rule_id",
    "abstention_rule_id",
    "abstain_delta",
    "tagger_revision",
    "tagger_checkpoint_sha256",
    "measurer_revision",
    "protocol_sha256",
)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            return {key: archive[key] for key in archive.files}
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        raise B2ClosureError(f"invalid posterior NPZ: {path}") from exc


def validate_posterior_arrays(
    arrays: Mapping[str, np.ndarray],
    *,
    expected_ids: Sequence[str] | None = None,
    coarse_map: Mapping[str, Any] | None = None,
) -> None:
    required = set(RECORD_ARRAY_KEYS) | set(SCALAR_ARRAY_KEYS) | {"coarse_class_names"}
    if set(arrays) != required:
        raise B2ClosureError(
            f"posterior array keys mismatch: missing={sorted(required-set(arrays))}, "
            f"extra={sorted(set(arrays)-required)}"
        )
    for key in SCALAR_ARRAY_KEYS:
        scalar = np.asarray(arrays[key])
        if scalar.ndim != 0 or scalar.dtype.hasobject:
            raise B2ClosureError(f"posterior provenance {key} must be a non-object scalar")
    if str(np.asarray(arrays["schema_version"]).item()) != POSTERIOR_SCHEMA:
        raise B2ClosureError("posterior bundle schema mismatch")
    if str(np.asarray(arrays["coarse_posterior_rule_id"]).item()) != COARSE_POSTERIOR_RULE_ID:
        raise B2ClosureError("posterior bundle coarse-posterior rule mismatch")
    if str(np.asarray(arrays["coarse_score_sums_rule_id"]).item()) != COARSE_SCORE_SUMS_RULE_ID:
        raise B2ClosureError("posterior bundle raw coarse-sum rule mismatch")
    if str(np.asarray(arrays["abstention_rule_id"]).item()) != ABSTENTION_RULE_ID:
        raise B2ClosureError("posterior bundle abstention rule mismatch")
    for key in (
        "coarse_map_sha256",
        "tagger_checkpoint_sha256",
        "protocol_sha256",
    ):
        digest = str(np.asarray(arrays[key]).item())
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise B2ClosureError(f"posterior bundle has invalid {key}")
    for key in ("coarse_map_revision", "tagger_revision", "measurer_revision"):
        if not str(np.asarray(arrays[key]).item()):
            raise B2ClosureError(f"posterior bundle has empty {key}")
    ids = np.asarray(arrays["record_id"])
    if ids.ndim != 1 or ids.dtype.kind != "U":
        raise B2ClosureError("record_id must be a one-dimensional Unicode array")
    n = ids.size
    if n == 0 or len(set(ids.tolist())) != n:
        raise B2ClosureError("posterior record IDs are empty or duplicated")
    for key in RECORD_ARRAY_KEYS:
        value = np.asarray(arrays[key])
        if value.ndim == 0 or value.shape[0] != n:
            raise B2ClosureError(f"record array {key} has incompatible shape {value.shape}")
        if value.dtype.hasobject:
            raise B2ClosureError(f"record array {key} has forbidden object dtype")
    probs = np.asarray(arrays["clipwise_output_527"])
    coarse_sums = np.asarray(arrays["coarse_score_sums"])
    coarse = np.asarray(arrays["coarse_posterior"])
    names = np.asarray(arrays["coarse_class_names"])
    if (
        names.ndim != 1
        or names.dtype.kind != "U"
        or names.size == 0
        or len(set(names.tolist())) != names.size
    ):
        raise B2ClosureError("coarse class names must be a unique nonempty Unicode vector")
    if probs.dtype != np.float32 or probs.shape != (n, EXPECTED_527):
        raise B2ClosureError(f"invalid full posterior schema: {probs.dtype} {probs.shape}")
    if coarse_sums.dtype != np.float32 or coarse_sums.shape != (n, names.size):
        raise B2ClosureError(f"invalid raw coarse-sum schema: {coarse_sums.dtype} {coarse_sums.shape}")
    if coarse.dtype != np.float32 or coarse.shape != (n, names.size):
        raise B2ClosureError(f"invalid coarse posterior schema: {coarse.dtype} {coarse.shape}")
    if not np.isfinite(probs).all() or np.any((probs < 0) | (probs > 1)):
        raise B2ClosureError("full posterior contains invalid probabilities")
    if not np.isfinite(coarse).all() or np.any((coarse < 0) | (coarse > 1)):
        raise B2ClosureError("coarse posterior contains invalid probabilities")
    if not np.isfinite(coarse_sums).all() or np.any(coarse_sums < 0):
        raise B2ClosureError("raw coarse sums contain invalid scores")
    if not np.allclose(
        coarse_sums.sum(axis=1), probs.sum(axis=1), atol=5e-5, rtol=1e-7
    ):
        raise B2ClosureError("raw coarse sums do not conserve full-posterior score mass")
    if not np.allclose(coarse.sum(axis=1), 1.0, atol=1e-6, rtol=0.0):
        raise B2ClosureError("coarse posterior fails mass conservation")
    normalized_sums = coarse_sums / coarse_sums.sum(axis=1, keepdims=True)
    if not np.allclose(coarse, normalized_sums, atol=1e-7, rtol=0.0):
        raise B2ClosureError("coarse posterior is not the normalized raw coarse-sum vector")
    if expected_ids is not None and ids.tolist() != list(expected_ids):
        raise B2ClosureError("posterior shard assignment does not match inventory")
    if coarse_map is not None:
        if str(np.asarray(arrays["coarse_map_revision"]).item()) != str(
            coarse_map.get("version", "")
        ):
            raise B2ClosureError("stored coarse-map revision does not match the map")
        expected_sums, expected_coarse, expected_names = derive_coarse_scores(probs, coarse_map)
        if names.tolist() != expected_names or not np.allclose(
            coarse_sums, expected_sums, atol=1e-6, rtol=0.0
        ) or not np.allclose(
            coarse, expected_coarse, atol=1e-7, rtol=0.0
        ):
            raise B2ClosureError("raw/normalized coarse scores do not match the frozen map")
        expected_diag = class_diagnostics_batch(
            probs,
            coarse_map,
            abstain_delta=float(np.asarray(arrays["abstain_delta"]).item()),
        )
        for key, expected in expected_diag.items():
            actual = np.asarray(arrays[key])
            if actual.dtype.kind in "fc":
                equal = np.allclose(actual, expected, atol=1e-7, rtol=0.0)
            else:
                equal = np.array_equal(actual, expected)
            if not equal:
                raise B2ClosureError(f"stored Class diagnostics mismatch for {key}")
    role = np.asarray(arrays["role"])
    fork = np.asarray(arrays["fork_index"])
    progress = np.asarray(arrays["progress"])
    if np.any(~np.isin(role, ["base", "fork"])):
        raise B2ClosureError("posterior bundle contains an unknown role")
    if np.any((role == "base") & ((fork != -1) | np.isfinite(progress))):
        raise B2ClosureError("base records must have fork=-1 and progress=NaN")
    if np.any((role == "fork") & ((fork < 0) | ~np.isfinite(progress))):
        raise B2ClosureError("fork records require fork index and progress")


def _scalar_item(arrays: Mapping[str, np.ndarray], key: str) -> Any:
    value = np.asarray(arrays[key])
    if value.ndim != 0:
        raise B2ClosureError(f"posterior provenance {key} must be scalar")
    return value.item()


def _validate_completion_provenance(
    completion: Mapping[str, Any], arrays: Mapping[str, np.ndarray]
) -> None:
    expected = {
        "coarse_map_sha256": str(_scalar_item(arrays, "coarse_map_sha256")),
        "coarse_map_revision": str(_scalar_item(arrays, "coarse_map_revision")),
        "coarse_score_sums_rule_id": str(
            _scalar_item(arrays, "coarse_score_sums_rule_id")
        ),
        "coarse_posterior_rule_id": str(
            _scalar_item(arrays, "coarse_posterior_rule_id")
        ),
        "abstention_rule_id": str(_scalar_item(arrays, "abstention_rule_id")),
        "tagger_revision": str(_scalar_item(arrays, "tagger_revision")),
        "tagger_checkpoint_sha256": str(
            _scalar_item(arrays, "tagger_checkpoint_sha256")
        ),
        "measurer_revision": str(_scalar_item(arrays, "measurer_revision")),
        "protocol_sha256": str(_scalar_item(arrays, "protocol_sha256")),
    }
    for key, value in expected.items():
        if completion.get(key) != value:
            raise B2ClosureError(f"completion/NPZ provenance mismatch for {key}")
    if not math.isclose(
        float(completion.get("abstain_delta", float("nan"))),
        float(_scalar_item(arrays, "abstain_delta")),
        abs_tol=1e-8,
        rel_tol=0.0,
    ):
        raise B2ClosureError("completion/NPZ provenance mismatch for abstain_delta")
    for key in ("coarse_map_sha256", "tagger_checkpoint_sha256", "protocol_sha256"):
        value = expected[key]
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise B2ClosureError(f"invalid SHA256 provenance scalar {key}")
    if int(completion.get("record_count", -1)) != int(np.asarray(arrays["record_id"]).size):
        raise B2ClosureError("completion/NPZ record cardinality mismatch")


def validate_shard_completion(
    completion_path: Path,
    *,
    inventory_manifest_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    completion_path = Path(completion_path)
    completion = _load_json(completion_path)
    if completion.get("schema_version") != POSTERIOR_SCHEMA or completion.get("status") != "COMPLETE":
        raise B2ClosureError(f"not a complete posterior shard: {completion_path}")
    data_path = _safe_resolve(completion_path.parent, str(completion.get("data_file", "")))
    if sha256_file(data_path) != completion.get("data_sha256"):
        raise B2ClosureError(f"posterior shard hash mismatch: {data_path}")
    if data_path.stat().st_size != int(completion.get("data_bytes", -1)):
        raise B2ClosureError(f"posterior shard byte-count mismatch: {data_path}")
    inventory_path = Path(
        inventory_manifest_path or str(completion.get("inventory_manifest", ""))
    )
    if sha256_file(inventory_path) != completion.get("inventory_manifest_sha256"):
        raise B2ClosureError("posterior shard inventory-manifest hash mismatch")
    records, inventory = load_inventory(inventory_path)
    if inventory["records_sha256"] != completion.get("inventory_records_sha256"):
        raise B2ClosureError("posterior shard inventory-record hash mismatch")
    index = int(completion.get("shard_index", -1))
    count = int(completion.get("shard_count", -1))
    assigned = assigned_inventory_records(records, index, count)
    expected_ids = [str(row["record_id"]) for row in assigned]
    digest = sha256_bytes(("\n".join(expected_ids) + "\n").encode("utf-8"))
    if (
        len(expected_ids) != int(completion.get("record_count", -1))
        or digest != completion.get("record_ids_sha256")
    ):
        raise B2ClosureError("posterior shard assigned-ID manifest mismatch")
    map_path = Path(str(completion.get("coarse_map", "")))
    if sha256_file(map_path) != completion.get("coarse_map_sha256"):
        raise B2ClosureError("posterior shard coarse-map hash mismatch")
    coarse_map = load_coarse_map(map_path)
    arrays = _load_npz(data_path)
    validate_posterior_arrays(arrays, expected_ids=expected_ids, coarse_map=coarse_map)
    _validate_completion_provenance(completion, arrays)
    protocol_path = Path(str(completion.get("protocol", "")))
    if not protocol_path.is_file() or sha256_file(protocol_path) != completion.get(
        "protocol_sha256"
    ):
        raise B2ClosureError("posterior shard protocol path/hash mismatch")
    if _array_schema(arrays) != completion.get("array_schema"):
        raise B2ClosureError("posterior shard array-schema manifest mismatch")
    return completion, arrays


def merge_posterior_shards(
    inventory_manifest_path: Path,
    completion_paths: Sequence[Path],
    out_dir: Path,
) -> dict[str, Any]:
    records, inventory = load_inventory(inventory_manifest_path)
    if not completion_paths:
        raise B2ClosureError("no posterior shard completions supplied")
    validated = [
        validate_shard_completion(path, inventory_manifest_path=inventory_manifest_path)
        for path in completion_paths
    ]
    shard_counts = {int(item[0]["shard_count"]) for item in validated}
    if len(shard_counts) != 1:
        raise B2ClosureError("posterior shards use inconsistent shard counts")
    shard_count = shard_counts.pop()
    indices = [int(item[0]["shard_index"]) for item in validated]
    if sorted(indices) != list(range(shard_count)):
        raise B2ClosureError(
            f"posterior shard set is incomplete or duplicated: {sorted(indices)}"
        )
    invariant_fields = (
        "inventory_manifest_sha256",
        "inventory_records_sha256",
        "batch_size",
        "canonical_b2",
        "protocol",
        "protocol_sha256",
        "coarse_map_sha256",
        "coarse_map_revision",
        "coarse_score_sums_rule_id",
        "coarse_posterior_rule_id",
        "abstention_rule_id",
        "abstain_delta",
        "tagger_revision",
        "tagger_checkpoint_sha256",
        "measurer_revision",
    )
    first_completion = validated[0][0]
    for completion, _arrays in validated[1:]:
        for field in invariant_fields:
            if completion.get(field) != first_completion.get(field):
                raise B2ClosureError(f"posterior shard provenance mismatch: {field}")
    by_id: dict[str, tuple[dict[str, np.ndarray], int]] = {}
    for _completion, arrays in validated:
        for row_index, record_id in enumerate(arrays["record_id"].tolist()):
            if record_id in by_id:
                raise B2ClosureError(f"duplicate record across posterior shards: {record_id}")
            by_id[str(record_id)] = (arrays, row_index)
    expected_ids = [str(row["record_id"]) for row in records]
    if set(by_id) != set(expected_ids):
        raise B2ClosureError(
            f"posterior merge coverage mismatch: missing={len(set(expected_ids)-set(by_id))}, "
            f"extra={len(set(by_id)-set(expected_ids))}"
        )
    merged: dict[str, np.ndarray] = {}
    for key in RECORD_ARRAY_KEYS:
        values = [by_id[record_id][0][key][by_id[record_id][1]] for record_id in expected_ids]
        merged[key] = np.asarray(values, dtype=validated[0][1][key].dtype)
    for key in SCALAR_ARRAY_KEYS:
        merged[key] = np.asarray(validated[0][1][key])
    merged["coarse_class_names"] = np.asarray(validated[0][1]["coarse_class_names"])
    validate_posterior_arrays(merged, expected_ids=expected_ids)
    out_dir = Path(out_dir)
    data_path = out_dir / "CLASS_POSTERIORS_MERGED.npz"
    completion_path = out_dir / "CLASS_POSTERIORS_MERGED.completion.json"
    deterministic_npz_create(data_path, merged)
    completion = {
        "schema_version": MERGE_SCHEMA,
        "status": "COMPLETE",
        "immutable": True,
        "record_count": len(expected_ids),
        "record_ids_sha256": inventory["record_ids_sha256"],
        "inventory_manifest": str(Path(inventory_manifest_path).resolve()),
        "inventory_manifest_sha256": sha256_file(inventory_manifest_path),
        "inventory_records_sha256": inventory["records_sha256"],
        "data_file": data_path.name,
        "data_sha256": sha256_file(data_path),
        "data_bytes": data_path.stat().st_size,
        "array_schema": _array_schema(merged),
        "input_shards": [
            {
                "shard_index": int(completion["shard_index"]),
                "completion": str(Path(path).resolve()),
                "completion_sha256": sha256_file(path),
                "data_sha256": completion["data_sha256"],
                "record_count": completion["record_count"],
            }
            for path, (completion, _arrays) in sorted(
                zip(completion_paths, validated), key=lambda item: int(item[1][0]["shard_index"])
            )
        ],
        **{field: first_completion[field] for field in invariant_fields},
    }
    atomic_json_create(completion_path, completion)
    return {**completion, "completion_path": str(completion_path)}


def load_merged_posteriors(completion_path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    completion_path = Path(completion_path)
    completion = _load_json(completion_path)
    if completion.get("schema_version") != MERGE_SCHEMA or completion.get("status") != "COMPLETE":
        raise B2ClosureError(f"not a complete merged posterior artifact: {completion_path}")
    data_path = _safe_resolve(completion_path.parent, str(completion.get("data_file", "")))
    if sha256_file(data_path) != completion.get("data_sha256"):
        raise B2ClosureError("merged posterior hash mismatch")
    arrays = _load_npz(data_path)
    validate_posterior_arrays(arrays)
    _validate_completion_provenance(completion, arrays)
    protocol_path = Path(str(completion.get("protocol", "")))
    if not protocol_path.is_file() or sha256_file(protocol_path) != completion.get(
        "protocol_sha256"
    ):
        raise B2ClosureError("merged posterior protocol path/hash mismatch")
    if _array_schema(arrays) != completion.get("array_schema"):
        raise B2ClosureError("merged posterior array-schema mismatch")
    return arrays, completion


def _confident_pairwise(labels: Sequence[str]) -> tuple[float, int]:
    return confident_agreement(
        list(labels), AgreementMetric.EXACT_MATCH, abstain=ABSTAIN
    )


def _commit_gain(a_fork: float, a_independent: float) -> float:
    if not (math.isfinite(a_fork) and math.isfinite(a_independent)):
        return float("nan")
    denominator = 1.0 - a_independent
    if denominator <= 1e-9:
        return 0.0
    return float(np.clip((a_fork - a_independent) / denominator, 0.0, 1.0))


def _first_crossing(curve: Mapping[float, float], theta: float, *, sustained: bool) -> float | None:
    grid = sorted(curve)
    for index, progress in enumerate(grid):
        value = float(curve[progress])
        if not math.isfinite(value) or value < theta:
            continue
        if sustained and any(
            math.isfinite(float(curve[later])) and float(curve[later]) < theta
            for later in grid[index + 1 :]
        ):
            continue
        return float(progress)
    return None


def _majority_label(labels: Sequence[str]) -> tuple[str | None, float]:
    confident = [str(value) for value in labels if str(value) != ABSTAIN]
    if len(confident) < 2:
        return None, float("nan")
    counts = Counter(confident)
    maximum = max(counts.values())
    winners = sorted(label for label, count in counts.items() if count == maximum)
    if len(winners) != 1:
        return None, maximum / len(confident)
    return winners[0], maximum / len(confident)


def _canonical_progress(value: float) -> float:
    """Undo harmless float32 serialization noise on the registered grid."""
    return round(float(value), 6)


def build_commitment_cells(arrays: Mapping[str, np.ndarray]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    validate_posterior_arrays(arrays)
    role = arrays["role"].tolist()
    videos = sorted(set(str(value) for value in arrays["video_id"].tolist()))
    seeds = sorted(set(int(value) for value in arrays["base_seed"].tolist()))
    progress_grid = sorted(
        set(
            _canonical_progress(value)
            for value in arrays["progress"][np.isfinite(arrays["progress"])]
        )
    )
    all_fork_indices = np.asarray(arrays["fork_index"])[np.asarray(role) == "fork"]
    if all_fork_indices.size == 0:
        raise B2ClosureError("posterior bundle has no fork records")
    k_forks = int(all_fork_indices.max()) + 1
    labels = arrays["confident_label"].tolist()
    index_by_key: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for index in range(len(role)):
        video = str(arrays["video_id"][index])
        seed = int(arrays["base_seed"][index])
        if role[index] == "base":
            index_by_key[("base", video)].append(index)
        else:
            progress = _canonical_progress(float(arrays["progress"][index]))
            index_by_key[("fork", video, seed, progress)].append(index)
    baselines: list[dict[str, Any]] = []
    baseline_by_video: dict[str, float] = {}
    for video in videos:
        indices = index_by_key[("base", video)]
        if len(indices) != len(seeds):
            raise B2ClosureError(
                f"video {video} has {len(indices)} base finals, expected {len(seeds)}"
            )
        observed_seeds = sorted(int(arrays["base_seed"][index]) for index in indices)
        if observed_seeds != seeds:
            raise B2ClosureError(f"video-conditioned baseline seed mismatch for {video}")
        base_labels = [str(labels[index]) for index in indices]
        agreement, n_confident = _confident_pairwise(base_labels)
        baseline_by_video[video] = agreement
        baselines.append(
            {
                "video_id": video,
                "a_independent": agreement,
                "n_base_finals": len(indices),
                "n_confident_base_finals": n_confident,
                "base_abstention_rate": 1.0 - n_confident / len(indices),
                "video_determined": bool(
                    math.isfinite(agreement) and agreement >= VIDEO_DETERMINED_MIN
                ),
            }
        )
    cells: list[dict[str, Any]] = []
    for video in videos:
        baseline = baseline_by_video[video]
        for seed in seeds:
            for progress in progress_grid:
                indices = index_by_key[("fork", video, seed, progress)]
                fork_indices = sorted(int(arrays["fork_index"][index]) for index in indices)
                expected_forks = list(range(k_forks))
                if not indices or fork_indices != expected_forks or len(fork_indices) < 2:
                    raise B2ClosureError(
                        f"incomplete fork cell {video}/seed{seed}/s={progress}: {fork_indices}"
                    )
                cell_labels = [str(labels[index]) for index in indices]
                agreement, n_confident = _confident_pairwise(cell_labels)
                majority, majority_share = _majority_label(cell_labels)
                cells.append(
                    {
                        "video_id": video,
                        "base_seed": seed,
                        "progress": progress,
                        "a_independent": baseline,
                        "a_fork_confident": agreement,
                        "commitment_gain": _commit_gain(agreement, baseline),
                        "n_forks": len(indices),
                        "n_confident_forks": n_confident,
                        "fork_abstention_rate": 1.0 - n_confident / len(indices),
                        "fork_majority_label": majority,
                        "fork_majority_share": majority_share,
                        "fork_labels": cell_labels,
                    }
                )
    return cells, baselines


def _finite_mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    finite = array[np.isfinite(array)]
    return float(finite.mean()) if finite.size else float("nan")


def _percentile_interval(values: Sequence[float]) -> tuple[float, float]:
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
    if finite.size == 0:
        return float("nan"), float("nan")
    low, high = np.quantile(finite, [0.025, 0.975])
    return float(low), float(high)


def _video_cluster_curve_bootstrap(
    cells: Sequence[Mapping[str, Any]],
    *,
    progress_grid: Sequence[float],
    n_boot: int,
    seed: int,
) -> dict[float, dict[str, tuple[float, float]]]:
    videos = sorted({str(row["video_id"]) for row in cells})
    per_video: dict[str, dict[float, dict[str, float]]] = defaultdict(dict)
    for video in videos:
        for progress in progress_grid:
            selected = [
                row for row in cells
                if str(row["video_id"]) == video and float(row["progress"]) == progress
            ]
            per_video[video][progress] = {
                "commitment_gain": _finite_mean(float(row["commitment_gain"]) for row in selected),
                "a_fork_confident": _finite_mean(float(row["a_fork_confident"]) for row in selected),
                "fork_abstention_rate": _finite_mean(
                    float(row["fork_abstention_rate"]) for row in selected
                ),
            }
    rng = np.random.default_rng(seed)
    draws: dict[float, dict[str, list[float]]] = {
        progress: defaultdict(list) for progress in progress_grid
    }
    for _ in range(n_boot):
        sampled = rng.integers(0, len(videos), size=len(videos))
        for progress in progress_grid:
            for metric in ("commitment_gain", "a_fork_confident", "fork_abstention_rate"):
                value = _finite_mean(
                    per_video[videos[int(index)]][progress][metric] for index in sampled
                )
                draws[progress][metric].append(value)
    return {
        progress: {
            metric: _percentile_interval(values) for metric, values in by_metric.items()
        }
        for progress, by_metric in draws.items()
    }


def _cell_curves(cells: Sequence[Mapping[str, Any]]) -> dict[tuple[str, int], dict[float, float]]:
    curves: dict[tuple[str, int], dict[float, float]] = defaultdict(dict)
    for row in cells:
        key = (str(row["video_id"]), int(row["base_seed"]))
        progress = float(row["progress"])
        if progress in curves[key]:
            raise B2ClosureError(f"duplicate commitment cell {key}/s={progress}")
        curves[key][progress] = float(row["commitment_gain"])
    return dict(curves)


def summarize_thresholds(
    cells: Sequence[Mapping[str, Any]],
    baselines: Sequence[Mapping[str, Any]],
    thresholds: Sequence[float],
    *,
    n_boot: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    curves = _cell_curves(cells)
    baseline_by_video = {str(row["video_id"]): row for row in baselines}
    videos = sorted(baseline_by_video)
    seed_values = sorted({key[1] for key in curves})
    summaries: list[dict[str, Any]] = []
    video_rows: list[dict[str, Any]] = []
    all_unit_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    for threshold in thresholds:
        unit_rows: list[dict[str, Any]] = []
        for (video, base_seed), curve in sorted(curves.items()):
            determined = bool(baseline_by_video[video]["video_determined"])
            n_scorable = sum(math.isfinite(float(value)) for value in curve.values())
            first = _first_crossing(curve, threshold, sustained=False)
            sustained = _first_crossing(curve, threshold, sustained=True)
            if determined:
                status = "VIDEO_DETERMINED"
            elif n_scorable == 0:
                status = "UNSCORABLE"
            elif first is not None:
                status = "CROSSING"
            else:
                status = "NONCROSSING"
            unit_rows.append(
                {
                    "video_id": video,
                    "base_seed": base_seed,
                    "theta_commit": float(threshold),
                    "status": status,
                    "n_scorable_progress_points": n_scorable,
                    "first_crossing": first,
                    "sustained_crossing": sustained,
                }
            )
        all_unit_rows.extend(unit_rows)
        nondetermined = [
            row for row in unit_rows if row["status"] != "VIDEO_DETERMINED"
        ]
        eligible = [row for row in nondetermined if row["status"] != "UNSCORABLE"]
        crossings = [
            float(row["first_crossing"])
            for row in eligible
            if row["first_crossing"] is not None
        ]
        per_video_fraction = {
            video: _finite_mean(
                1.0 if row["status"] == "CROSSING" else 0.0
                for row in eligible
                if row["video_id"] == video
            )
            for video in videos
            if not bool(baseline_by_video[video]["video_determined"])
        }
        boot_cross: list[float] = []
        boot_non: list[float] = []
        nondetermined_videos = sorted(per_video_fraction)
        if nondetermined_videos:
            for _ in range(n_boot):
                sampled = rng.integers(0, len(nondetermined_videos), size=len(nondetermined_videos))
                value = _finite_mean(
                    per_video_fraction[nondetermined_videos[int(index)]] for index in sampled
                )
                boot_cross.append(value)
                boot_non.append(1.0 - value)
        cross_ci = _percentile_interval(boot_cross)
        non_ci = _percentile_interval(boot_non)
        summaries.append(
            {
                "theta_commit": float(threshold),
                "n_videos": len(videos),
                "n_video_determined": sum(
                    bool(row["video_determined"]) for row in baselines
                ),
                "video_determined_fraction": _finite_mean(
                    float(bool(row["video_determined"])) for row in baselines
                ),
                "n_video_seed_units": len(unit_rows),
                "n_nondetermined_video_seed_units": len(nondetermined),
                "n_scorable_nondetermined_video_seed_units": len(eligible),
                "n_unscorable": sum(row["status"] == "UNSCORABLE" for row in nondetermined),
                "n_crossing": len(crossings),
                "n_noncrossing": sum(row["status"] == "NONCROSSING" for row in eligible),
                "crossing_fraction": len(crossings) / len(eligible) if eligible else float("nan"),
                "crossing_fraction_ci_low": cross_ci[0],
                "crossing_fraction_ci_high": cross_ci[1],
                "noncrossing_fraction": 1.0 - len(crossings) / len(eligible)
                if eligible else float("nan"),
                "noncrossing_fraction_ci_low": non_ci[0],
                "noncrossing_fraction_ci_high": non_ci[1],
                "mean_first_crossing_crossers": _finite_mean(crossings),
                "median_first_crossing_crossers": float(np.median(crossings))
                if crossings else float("nan"),
                "noncrossers_are_right_censored_without_numeric_imputation": True,
                "n_sustained_crossing": sum(
                    row["sustained_crossing"] is not None for row in eligible
                ),
                "registered_sustained_crossing_fraction": _finite_mean(
                    float(row["sustained_crossing"] is not None) for row in eligible
                ),
            }
        )
        for video in videos:
            selected = [row for row in unit_rows if row["video_id"] == video]
            crossing_values = [
                float(row["first_crossing"])
                for row in selected if row["first_crossing"] is not None
            ]
            unscorable_count = sum(row["status"] == "UNSCORABLE" for row in selected)
            noncrossing_count = sum(row["status"] == "NONCROSSING" for row in selected)
            video_rows.append(
                {
                    "video_id": video,
                    "theta_commit": float(threshold),
                    "a_independent": float(baseline_by_video[video]["a_independent"]),
                    "n_confident_base_finals": int(
                        baseline_by_video[video]["n_confident_base_finals"]
                    ),
                    "base_abstention_rate": float(
                        baseline_by_video[video]["base_abstention_rate"]
                    ),
                    "video_determined": bool(baseline_by_video[video]["video_determined"]),
                    "n_base_seeds": len(seed_values),
                    "n_crossing_seeds": len(crossing_values),
                    "n_noncrossing_seeds": noncrossing_count,
                    "n_unscorable_seeds": unscorable_count,
                    "crossing_seed_fraction_scorable": len(crossing_values)
                    / max(len(crossing_values) + noncrossing_count, 1),
                    "mean_first_crossing": _finite_mean(crossing_values),
                    "median_first_crossing": float(np.median(crossing_values))
                    if crossing_values else float("nan"),
                    "n_sustained_crossing_seeds": sum(
                        row["sustained_crossing"] is not None for row in selected
                    ),
                }
            )
    return summaries, video_rows, all_unit_rows


def pooled_and_seed_crossings(
    cells: Sequence[Mapping[str, Any]],
    thresholds: Sequence[float],
    *,
    n_boot: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    progress_grid = sorted({float(row["progress"]) for row in cells})
    videos = sorted({str(row["video_id"]) for row in cells})
    seeds = sorted({int(row["base_seed"]) for row in cells})
    pooled_curve = {
        progress: _finite_mean(
            float(row["commitment_gain"])
            for row in cells
            if float(row["progress"]) == progress
        )
        for progress in progress_grid
    }
    seed_curves = {
        base_seed: {
            progress: _finite_mean(
                float(row["commitment_gain"])
                for row in cells
                if int(row["base_seed"]) == base_seed
                and float(row["progress"]) == progress
            )
            for progress in progress_grid
        }
        for base_seed in seeds
    }
    per_video_curve = {
        video: {
            progress: _finite_mean(
                float(row["commitment_gain"])
                for row in cells
                if str(row["video_id"]) == video
                and float(row["progress"]) == progress
            )
            for progress in progress_grid
        }
        for video in videos
    }
    pooled_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    for threshold in thresholds:
        first = _first_crossing(pooled_curve, threshold, sustained=False)
        sustained = _first_crossing(pooled_curve, threshold, sustained=True)
        n_scorable = sum(math.isfinite(value) for value in pooled_curve.values())
        bootstrap_first: list[float] = []
        bootstrap_sustained: list[float] = []
        n_boot_unscorable = 0
        n_boot_noncrossing = 0
        for _ in range(n_boot):
            sampled = rng.integers(0, len(videos), size=len(videos))
            curve = {
                progress: _finite_mean(
                    per_video_curve[videos[int(index)]][progress] for index in sampled
                )
                for progress in progress_grid
            }
            if not any(math.isfinite(value) for value in curve.values()):
                n_boot_unscorable += 1
                continue
            draw_first = _first_crossing(curve, threshold, sustained=False)
            draw_sustained = _first_crossing(curve, threshold, sustained=True)
            if draw_first is None:
                n_boot_noncrossing += 1
            else:
                bootstrap_first.append(draw_first)
            if draw_sustained is not None:
                bootstrap_sustained.append(draw_sustained)
        first_ci = _percentile_interval(bootstrap_first)
        sustained_ci = _percentile_interval(bootstrap_sustained)
        pooled_rows.append(
            {
                "theta_commit": float(threshold),
                "n_scorable_progress_points": n_scorable,
                "status": "UNSCORABLE"
                if n_scorable == 0
                else ("CROSSING" if first is not None else "NONCROSSING"),
                "first_crossing": first,
                "first_crossing_bootstrap_ci_low": first_ci[0],
                "first_crossing_bootstrap_ci_high": first_ci[1],
                "sustained_crossing": sustained,
                "sustained_crossing_bootstrap_ci_low": sustained_ci[0],
                "sustained_crossing_bootstrap_ci_high": sustained_ci[1],
                "bootstrap_draws": n_boot,
                "bootstrap_unscorable_draws": n_boot_unscorable,
                "bootstrap_noncrossing_draws": n_boot_noncrossing,
                "bootstrap_sustained_crossing_draws": len(bootstrap_sustained),
                "noncrossing_draws_not_numerically_imputed": True,
            }
        )
        for base_seed in seeds:
            curve = seed_curves[base_seed]
            seed_n_scorable = sum(math.isfinite(value) for value in curve.values())
            seed_first = _first_crossing(curve, threshold, sustained=False)
            seed_sustained = _first_crossing(curve, threshold, sustained=True)
            seed_rows.append(
                {
                    "base_seed": base_seed,
                    "theta_commit": float(threshold),
                    "n_scorable_progress_points": seed_n_scorable,
                    "status": "UNSCORABLE"
                    if seed_n_scorable == 0
                    else ("CROSSING" if seed_first is not None else "NONCROSSING"),
                    "first_crossing": seed_first,
                    "sustained_crossing": seed_sustained,
                }
            )
    return pooled_rows, seed_rows


def _bootstrap_cell_measurement_variance(
    labels: Sequence[str],
    baseline: float,
    *,
    n_boot: int,
    rng: np.random.Generator,
) -> tuple[float, float, float, float]:
    text_labels = [str(value) for value in labels]
    confident_names = sorted({value for value in text_labels if value != ABSTAIN})
    codebook = {value: index for index, value in enumerate(confident_names)}
    codes = np.asarray(
        [-1 if value == ABSTAIN else codebook[value] for value in text_labels],
        dtype=np.int16,
    )
    confident = codes[codes >= 0]

    def row_agreement(sampled: np.ndarray) -> np.ndarray:
        n_confident = np.sum(sampled >= 0, axis=1)
        matching_pairs = np.zeros(sampled.shape[0], dtype=np.float64)
        for code in range(len(confident_names)):
            count = np.sum(sampled == code, axis=1)
            matching_pairs += count * (count - 1) / 2.0
        all_pairs = n_confident * (n_confident - 1) / 2.0
        return np.divide(
            matching_pairs,
            all_pairs,
            out=np.full(sampled.shape[0], np.nan, dtype=np.float64),
            where=all_pairs > 0,
        )

    def gain(values: np.ndarray) -> np.ndarray:
        if not math.isfinite(baseline):
            return np.full(values.shape, np.nan, dtype=np.float64)
        denominator = 1.0 - baseline
        if denominator <= 1e-9:
            return np.zeros(values.shape, dtype=np.float64)
        return np.clip((values - baseline) / denominator, 0.0, 1.0)

    sampled = codes[rng.integers(0, codes.size, size=(n_boot, codes.size))]
    full_values = gain(row_agreement(sampled))
    full_values = full_values[np.isfinite(full_values)]
    if confident.size >= 2:
        conditional = confident[
            rng.integers(0, confident.size, size=(n_boot, confident.size))
        ]
        conditional_values = gain(row_agreement(conditional))
        conditional_values = conditional_values[np.isfinite(conditional_values)]
    else:
        conditional_values = np.asarray([], dtype=np.float64)
    full_var = float(np.var(full_values, ddof=1)) if full_values.size >= 2 else float("nan")
    conditional_var = (
        float(np.var(conditional_values, ddof=1))
        if conditional_values.size >= 2 else float("nan")
    )
    abstention_increment = (
        max(full_var - conditional_var, 0.0)
        if math.isfinite(full_var) and math.isfinite(conditional_var)
        else float("nan")
    )
    valid_fraction = int(full_values.size) / n_boot
    return full_var, conditional_var, abstention_increment, valid_fraction


def _crossed_variance_components(
    rows: Sequence[Mapping[str, Any]],
    *,
    fork_variance: float,
    abstention_variance: float,
) -> dict[str, Any]:
    finite = [row for row in rows if math.isfinite(float(row["commitment_gain"]))]
    videos = sorted({str(row["video_id"]) for row in rows})
    seeds = sorted({int(row["base_seed"]) for row in rows})
    values = np.asarray([float(row["commitment_gain"]) for row in finite], dtype=float)
    if values.size < 4:
        return {
            "status": "UNRESOLVED",
            "reason": "fewer_than_four_scorable_video_seed_cells",
            "n_scorable": int(values.size),
        }
    complete = len(finite) == len(videos) * len(seeds)
    if complete:
        matrix = np.full((len(videos), len(seeds)), np.nan)
        video_index = {value: index for index, value in enumerate(videos)}
        seed_index = {value: index for index, value in enumerate(seeds)}
        for row in finite:
            matrix[video_index[str(row["video_id"])], seed_index[int(row["base_seed"])]] = float(
                row["commitment_gain"]
            )
        grand = float(matrix.mean())
        video_means = matrix.mean(axis=1)
        seed_means = matrix.mean(axis=0)
        residual = matrix - video_means[:, None] - seed_means[None, :] + grand
        ms_video = len(seeds) * float(np.var(video_means, ddof=1))
        ms_seed = len(videos) * float(np.var(seed_means, ddof=1))
        ms_residual = float(np.sum(residual**2) / ((len(videos) - 1) * (len(seeds) - 1)))
        video_raw = (ms_video - ms_residual) / len(seeds)
        seed_raw = (ms_seed - ms_residual) / len(videos)
        video_variance = max(video_raw, 0.0)
        seed_variance = max(seed_raw, 0.0)
        interaction_plus_measurement = max(ms_residual, 0.0)
        method = "balanced_crossed_random_effects_anova_moments"
    else:
        grand = float(values.mean())
        centered_values = np.asarray(
            [float(row["commitment_gain"]) - grand for row in finite], dtype=float
        )

        def shared_effect_covariance(group_values: Sequence[Any]) -> float:
            unique = {value: index for index, value in enumerate(sorted(set(group_values)))}
            codes = np.asarray([unique[value] for value in group_values], dtype=np.int32)
            counts = np.bincount(codes, minlength=len(unique)).astype(float)
            sums = np.bincount(
                codes, weights=centered_values, minlength=len(unique)
            )
            square_sums = np.bincount(
                codes, weights=centered_values**2, minlength=len(unique)
            )
            pair_products = float(np.sum((sums**2 - square_sums) / 2.0))
            pair_count = float(np.sum(counts * (counts - 1.0) / 2.0))
            return pair_products / pair_count if pair_count > 0 else float("nan")

        video_raw = shared_effect_covariance(
            [str(row["video_id"]) for row in finite]
        )
        seed_raw = shared_effect_covariance(
            [int(row["base_seed"]) for row in finite]
        )
        total = float(np.var(values, ddof=1))
        video_variance = max(video_raw, 0.0) if math.isfinite(video_raw) else 0.0
        seed_variance = max(seed_raw, 0.0) if math.isfinite(seed_raw) else 0.0
        interaction_plus_measurement = max(total - video_variance - seed_variance, 0.0)
        ms_residual = interaction_plus_measurement
        method = "unbalanced_crossed_pair_covariance_moments"
    usable_fork = min(max(fork_variance, 0.0), interaction_plus_measurement)
    usable_abstention = min(max(abstention_variance, 0.0), usable_fork)
    fork_nonabstention = max(usable_fork - usable_abstention, 0.0)
    interaction = max(interaction_plus_measurement - usable_fork, 0.0)
    components = {
        "video": video_variance,
        "base_seed": seed_variance,
        "video_by_seed_interaction": interaction,
        "fork_monte_carlo_nonabstention": fork_nonabstention,
        "abstention_within_fork_monte_carlo": usable_abstention,
    }
    component_total = sum(components.values())
    return {
        "scientific_status": "SUPPORTED_EXPLORATORILY",
        "method": method,
        "n_scorable": len(finite),
        "n_total": len(rows),
        "complete_crossed_table": complete,
        "raw_video_component": video_raw,
        "raw_base_seed_component": seed_raw,
        "interaction_plus_measurement_residual": interaction_plus_measurement,
        "fork_monte_carlo_total": usable_fork,
        "abstention_is_subcomponent_of_fork_monte_carlo": True,
        "measurer_repeatability_variance": None,
        "measurer_repeatability_identifiability": (
            "not identifiable from one deterministic posterior measurement per WAV"
        ),
        "components": components,
        "component_total": component_total,
        "component_fractions": {
            key: value / component_total if component_total > 0 else float("nan")
            for key, value in components.items()
        },
    }


def variance_decomposition(
    cells: Sequence[Mapping[str, Any]],
    *,
    n_fork_boot: int,
    n_video_boot: int,
    seed: int,
) -> dict[str, Any]:
    progress_grid = sorted({float(row["progress"]) for row in cells})
    rng = np.random.default_rng(seed)
    video_rng = np.random.default_rng(seed)
    all_videos = sorted({str(row["video_id"]) for row in cells})
    video_bootstrap_samples = video_rng.integers(
        0, len(all_videos), size=(n_video_boot, len(all_videos))
    )
    overall_draw_components: list[dict[str, list[float]]] = [
        defaultdict(list) for _ in range(n_video_boot)
    ]
    by_progress: list[dict[str, Any]] = []
    for progress in progress_grid:
        selected = [row for row in cells if float(row["progress"]) == progress]
        full_variances: list[float] = []
        conditional_variances: list[float] = []
        abstention_variances: list[float] = []
        valid_fractions: list[float] = []
        selected_with_variance: list[dict[str, Any]] = []
        for row in selected:
            full, conditional, abstention, valid = _bootstrap_cell_measurement_variance(
                row["fork_labels"],
                float(row["a_independent"]),
                n_boot=n_fork_boot,
                rng=rng,
            )
            full_variances.append(full)
            conditional_variances.append(conditional)
            abstention_variances.append(abstention)
            valid_fractions.append(valid)
            selected_with_variance.append(
                {
                    **row,
                    "_fork_variance": full,
                    "_abstention_variance": abstention,
                }
            )
        full_mean = _finite_mean(full_variances)
        conditional_mean = _finite_mean(conditional_variances)
        abstention_mean = _finite_mean(abstention_variances)
        components = _crossed_variance_components(
            selected,
            fork_variance=full_mean if math.isfinite(full_mean) else 0.0,
            abstention_variance=abstention_mean if math.isfinite(abstention_mean) else 0.0,
        )
        videos = sorted({str(row["video_id"]) for row in selected})
        if videos != all_videos:
            raise B2ClosureError("variance strata do not contain the same video clusters")
        component_draws: dict[str, list[float]] = defaultdict(list)
        for draw_index, sampled in enumerate(video_bootstrap_samples):
            boot_rows: list[dict[str, Any]] = []
            for cluster_index, sampled_index in enumerate(sampled):
                source_video = videos[int(sampled_index)]
                for row in selected_with_variance:
                    if str(row["video_id"]) == source_video:
                        boot_rows.append(
                            {**row, "video_id": f"cluster{cluster_index:05d}:{source_video}"}
                        )
            boot_fork = _finite_mean(float(row["_fork_variance"]) for row in boot_rows)
            boot_abstention = _finite_mean(
                float(row["_abstention_variance"]) for row in boot_rows
            )
            boot_result = _crossed_variance_components(
                boot_rows,
                fork_variance=boot_fork if math.isfinite(boot_fork) else 0.0,
                abstention_variance=(
                    boot_abstention if math.isfinite(boot_abstention) else 0.0
                ),
            )
            for name, value in boot_result.get("components", {}).items():
                if math.isfinite(float(value)):
                    component_draws[name].append(float(value))
                    overall_draw_components[draw_index][name].append(float(value))
        component_ci = {
            name: {
                "ci_low": _percentile_interval(values)[0],
                "ci_high": _percentile_interval(values)[1],
                "valid_draws": len(values),
            }
            for name, values in sorted(component_draws.items())
        }
        by_progress.append(
            {
                "progress": progress,
                "fork_bootstrap_draws": n_fork_boot,
                "video_cluster_bootstrap_draws": n_video_boot,
                "video_cluster_bootstrap_seed": seed,
                "fork_monte_carlo_variance_mean": full_mean,
                "fork_conditional_confident_label_variance_mean": conditional_mean,
                "abstention_increment_variance_mean": abstention_mean,
                "fork_bootstrap_scorable_fraction_mean": _finite_mean(valid_fractions),
                **components,
                "component_video_bootstrap_ci": component_ci,
            }
        )
    component_names = (
        "video",
        "base_seed",
        "video_by_seed_interaction",
        "fork_monte_carlo_nonabstention",
        "abstention_within_fork_monte_carlo",
    )
    overall_components = {
        name: _finite_mean(
            float(row.get("components", {}).get(name, float("nan"))) for row in by_progress
        )
        for name in component_names
    }
    overall_component_draws = {
        name: [
            _finite_mean(draw.get(name, []))
            for draw in overall_draw_components
        ]
        for name in component_names
    }
    overall_component_ci = {
        name: {
            "ci_low": _percentile_interval(values)[0],
            "ci_high": _percentile_interval(values)[1],
            "valid_draws": sum(math.isfinite(value) for value in values),
        }
        for name, values in overall_component_draws.items()
    }
    total = sum(value for value in overall_components.values() if math.isfinite(value))
    return {
        "model": "progress-stratified crossed random-effects method-of-moments",
        "progress_treated_as_fixed_stratum": True,
        "fork_variance_method": (
            "within-cell nonparametric fork bootstrap; abstention increment is full-label "
            "bootstrap variance minus fixed-confident-count label bootstrap variance"
        ),
        "variance_component_uncertainty": (
            "video-cluster percentile bootstrap; all base seeds retained within sampled video"
        ),
        "video_cluster_bootstrap_draws": n_video_boot,
        "video_cluster_bootstrap_seed": seed,
        "by_progress": by_progress,
        "overall_mean_components": overall_components,
        "overall_mean_component_video_bootstrap_ci": overall_component_ci,
        "overall_mean_component_fractions": {
            name: value / total if total > 0 and math.isfinite(value) else float("nan")
            for name, value in overall_components.items()
        },
        "measurer_repeatability_variance": None,
        "measurement_limit": (
            "posterior inference was run once per immutable WAV; repeat-inference or "
            "cross-device measurement variance is not identifiable"
        ),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _historical_inputs(
    paths: Sequence[Path], *, canonical: bool
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        data = _load_json(path)
        curve: list[dict[str, Any]] = []
        for row in data.get("curve_by_s", []):
            if isinstance(row, dict) and "s" in row:
                curve.append(
                    {
                        "progress": float(row["s"]),
                        "commitment_gain_confident": row.get("commit_gain_confident"),
                        "a_fork_confident": row.get("a_fork_confident"),
                        "abstain_rate": row.get("abstain_rate"),
                    }
                )
        rows.append(
            {
                "path": str(Path(path).resolve()),
                "sha256": sha256_file(path),
                "s_commit_confident": data.get("s_commit_confident"),
                "theta_commit": data.get("theta_commit"),
                "n_crossing_confident": data.get("n_crossing_confident"),
                "reproduces_committed_csv": data.get("reproduces_committed_csv"),
                "max_abs_delta": data.get("max_abs_delta"),
                "curve_by_progress": curve,
            }
        )
    if canonical:
        pinned = [row for row in rows if row["sha256"] == PINNED_HISTORICAL_COMPARATOR_SHA256]
        if len(pinned) != 1:
            raise B2ClosureError(
                "canonical analysis requires exactly one pinned WP-A2 Class comparator"
            )
        comparator = pinned[0]
        if (
            comparator["reproduces_committed_csv"] is not True
            or float(comparator["max_abs_delta"]) != 0.0
            or not math.isclose(
                float(comparator["s_commit_confident"]),
                PINNED_HISTORICAL_S_COMMIT,
                abs_tol=1e-15,
                rel_tol=0.0,
            )
            or not math.isclose(float(comparator["theta_commit"]), 0.7, abs_tol=0.0)
        ):
            raise B2ClosureError("pinned historical comparator content is inconsistent")
    return rows


def validate_canonical_analysis_inputs(
    arrays: Mapping[str, np.ndarray],
    merged: Mapping[str, Any],
    *,
    protocol_path: Path,
    historical_jsons: Sequence[Path],
) -> dict[str, Any]:
    protocol = validate_class_protocol(protocol_path, canonical=True)
    if merged.get("canonical_b2") is not True:
        raise B2ClosureError("production analysis requires canonical_b2=true")
    if merged.get("protocol_sha256") != protocol["sha256"]:
        raise B2ClosureError("merged posterior was measured under a different protocol")
    inventory_path = Path(str(merged.get("inventory_manifest", "")))
    if not inventory_path.is_file() or sha256_file(inventory_path) != merged.get(
        "inventory_manifest_sha256"
    ):
        raise B2ClosureError("canonical merged inventory provenance is invalid")
    records, inventory = load_inventory(inventory_path)
    expected_inventory = {
        "canonical_b2": True,
        "record_count": 79152,
        "base_record_count": 816,
        "fork_record_count": 78336,
        "base_seeds": list(EXPECTED_SEEDS),
        "progress_grid": list(EXPECTED_S_GRID),
        "k_forks": EXPECTED_K_FORKS,
    }
    for key, value in expected_inventory.items():
        if inventory.get(key) != value:
            raise B2ClosureError(
                f"canonical inventory provenance mismatch for {key}: {inventory.get(key)!r}"
            )
    expected_ids = [str(row["record_id"]) for row in records]
    if np.asarray(arrays["record_id"]).tolist() != expected_ids:
        raise B2ClosureError("canonical merged posterior IDs/order differ from inventory")
    for key, expected in (
        ("record_count", len(expected_ids)),
        ("record_ids_sha256", inventory["record_ids_sha256"]),
        ("inventory_records_sha256", inventory["records_sha256"]),
    ):
        if merged.get(key) != expected:
            raise B2ClosureError(f"canonical merged completion provenance mismatch for {key}")
    if len(inventory.get("video_ids", [])) != EXPECTED_N_CLIPS or len(
        inventory.get("roots", [])
    ) != 4:
        raise B2ClosureError("canonical inventory video/root cardinality mismatch")
    n = int(np.asarray(arrays["record_id"]).size)
    role = np.asarray(arrays["role"])
    if (
        n != 79152
        or int(np.sum(role == "base")) != 816
        or int(np.sum(role == "fork")) != 78336
        or len(set(arrays["video_id"].tolist())) != EXPECTED_N_CLIPS
        or sorted(set(int(value) for value in arrays["base_seed"].tolist()))
        != list(EXPECTED_SEEDS)
        or sorted(
            {
                _canonical_progress(value)
                for value in arrays["progress"][np.isfinite(arrays["progress"])]
            }
        )
        != list(EXPECTED_S_GRID)
    ):
        raise B2ClosureError("canonical merged posterior population cardinality mismatch")
    for key, expected in (("cfg", 4.5), ("alpha", 0.8)):
        values = np.asarray(arrays[key], dtype=float)
        if not np.allclose(values, expected, atol=1e-7, rtol=0.0):
            raise B2ClosureError(f"canonical merged posterior provenance mismatch for {key}")
    for key, expected in (
        ("schedule", "sqrt_down"),
        ("sample_rate", 16000),
        ("generation_model_variant", "small_16k"),
    ):
        if set(np.asarray(arrays[key]).tolist()) != {expected}:
            raise B2ClosureError(f"canonical merged posterior provenance mismatch for {key}")
    if set(arrays["generation_manifest_sha256"].tolist()) != set(
        EXPECTED_GENERATION_MANIFEST_SHA256
    ):
        raise B2ClosureError("canonical generation-manifest hash set mismatch")
    if set(arrays["generation_revision"].tolist()) != set(EXPECTED_GENERATION_REVISIONS):
        raise B2ClosureError("canonical generation revision mismatch")
    if str(_scalar_item(arrays, "coarse_map_sha256")) != PINNED_COARSE_MAP_SHA256:
        raise B2ClosureError("canonical merged coarse-map hash mismatch")
    if (
        str(_scalar_item(arrays, "tagger_checkpoint_sha256"))
        != PINNED_CHECKPOINT_SHA256
    ):
        raise B2ClosureError("canonical merged checkpoint hash mismatch")
    if not str(_scalar_item(arrays, "tagger_revision")) or not str(
        _scalar_item(arrays, "measurer_revision")
    ):
        raise B2ClosureError("canonical merged measurer revisions are missing")
    historical = _historical_inputs(historical_jsons, canonical=True)
    return {"protocol": protocol, "historical": historical, "inventory": inventory}


def classify_frozen_replication(
    pooled_crossings: Sequence[Mapping[str, Any]],
    seed_crossings: Sequence[Mapping[str, Any]],
    variance: Mapping[str, Any],
    *,
    progress_grid: Sequence[float],
) -> dict[str, Any]:
    pooled = next(
        row for row in pooled_crossings if math.isclose(float(row["theta_commit"]), 0.7)
    )
    seed_rows = [
        row for row in seed_crossings if math.isclose(float(row["theta_commit"]), 0.7)
    ]
    pooled_sustained = pooled.get("sustained_crossing")
    seed_cross_by_045 = sum(
        row.get("sustained_crossing") is not None
        and float(row["sustained_crossing"]) <= 0.45
        for row in seed_rows
    ) / max(len(seed_rows), 1)
    components = variance.get("overall_mean_components", {})
    seed_variance = float(components.get("base_seed") or 0.0)
    video_variance = float(components.get("video") or 0.0)
    identifiable_total = sum(
        float(value) for value in components.values() if value is not None and math.isfinite(float(value))
    )
    seed_variance_fraction = (
        seed_variance / identifiable_total if identifiable_total > 0 else float("inf")
    )
    ci_low = pooled.get("sustained_crossing_bootstrap_ci_low")
    ci_high = pooled.get("sustained_crossing_bootstrap_ci_high")
    historical_in_ci = (
        ci_low is not None
        and ci_high is not None
        and math.isfinite(float(ci_low))
        and math.isfinite(float(ci_high))
        and float(ci_low) <= PINNED_HISTORICAL_S_COMMIT <= float(ci_high)
    )
    nearest_historical_index = min(
        range(len(progress_grid)),
        key=lambda index: abs(float(progress_grid[index]) - PINNED_HISTORICAL_S_COMMIT),
    )
    point_within_one_step = False
    if pooled_sustained is not None:
        point_index = min(
            range(len(progress_grid)),
            key=lambda index: abs(float(progress_grid[index]) - float(pooled_sustained)),
        )
        point_within_one_step = abs(point_index - nearest_historical_index) <= 1

    pooled_by_045 = pooled_sustained is not None and float(pooled_sustained) <= 0.45
    if not pooled_by_045 or seed_cross_by_045 < 0.25:
        label = "not_reproduced"
        scientific_status = "NOT_SUPPORTED"
    elif seed_variance >= video_variance or seed_cross_by_045 < 0.60:
        label = "strongly_seed_dependent"
        scientific_status = "UNRESOLVED"
    elif (
        seed_cross_by_045 >= 0.80
        and seed_variance_fraction < 0.25
        and (historical_in_ci or point_within_one_step)
    ):
        label = "stable_across_seeds"
        scientific_status = "SUPPORTED_EXPLORATORILY"
    else:
        label = "heterogeneous_but_directionally_consistent"
        scientific_status = "SUPPORTED_EXPLORATORILY"
    if label not in REPLICATION_LABELS or scientific_status not in ALLOWED_SCIENTIFIC_STATUSES:
        raise AssertionError("frozen replication classifier emitted an invalid label/status")
    return {
        "replication_label": label,
        "scientific_status": scientific_status,
        "rule_id": "protocol.class_measurement.exploratory_replication_labels.v1",
        "pooled_sustained_crossing_theta_0.70": pooled_sustained,
        "seed_curves_sustained_crossing_by_0.45_fraction": seed_cross_by_045,
        "seed_variance": seed_variance,
        "video_variance": video_variance,
        "seed_variance_fraction_identifiable_total": seed_variance_fraction,
        "historical_s_commit": PINNED_HISTORICAL_S_COMMIT,
        "historical_within_pooled_bootstrap_interval": historical_in_ci,
        "historical_within_one_sampled_step_of_point_estimate": point_within_one_step,
    }


def analyze_multiseed(
    arrays: Mapping[str, np.ndarray],
    *,
    protocol_path: Path,
    canonical: bool,
    thresholds: Sequence[float] = SENSITIVITY_THRESHOLDS,
    n_video_boot: int = DEFAULT_BOOTSTRAP_DRAWS,
    n_fork_boot: int = DEFAULT_BOOTSTRAP_DRAWS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    historical_jsons: Sequence[Path] = (),
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    protocol_info = validate_class_protocol(protocol_path, canonical=canonical)
    if canonical and (
        tuple(float(value) for value in thresholds) != SENSITIVITY_THRESHOLDS
        or n_video_boot != DEFAULT_BOOTSTRAP_DRAWS
        or n_fork_boot != DEFAULT_BOOTSTRAP_DRAWS
        or seed != DEFAULT_BOOTSTRAP_SEED
    ):
        raise B2ClosureError("canonical analysis settings differ from the frozen protocol")
    historical = _historical_inputs(historical_jsons, canonical=canonical)
    cells, baselines = build_commitment_cells(arrays)
    progress_grid = sorted({float(row["progress"]) for row in cells})
    curve_ci = _video_cluster_curve_bootstrap(
        cells, progress_grid=progress_grid, n_boot=n_video_boot, seed=seed
    )
    curve_rows: list[dict[str, Any]] = []
    for progress in progress_grid:
        selected = [row for row in cells if float(row["progress"]) == progress]
        row = {
            "progress": progress,
            "n_video_seed_cells": len(selected),
            "n_scorable_cells": sum(
                math.isfinite(float(item["commitment_gain"])) for item in selected
            ),
            "mean_a_fork_confident": _finite_mean(
                float(item["a_fork_confident"]) for item in selected
            ),
            "mean_commitment_gain": _finite_mean(
                float(item["commitment_gain"]) for item in selected
            ),
            "mean_fork_abstention_rate": _finite_mean(
                float(item["fork_abstention_rate"]) for item in selected
            ),
        }
        for metric, output_name in (
            ("a_fork_confident", "a_fork_confident"),
            ("commitment_gain", "commitment_gain"),
            ("fork_abstention_rate", "fork_abstention_rate"),
        ):
            low, high = curve_ci[progress][metric]
            row[f"{output_name}_ci_low"] = low
            row[f"{output_name}_ci_high"] = high
        curve_rows.append(row)
    seed_curve_rows: list[dict[str, Any]] = []
    for base_seed in sorted({int(row["base_seed"]) for row in cells}):
        for progress in progress_grid:
            selected = [
                row for row in cells
                if int(row["base_seed"]) == base_seed and float(row["progress"]) == progress
            ]
            seed_curve_rows.append(
                {
                    "base_seed": base_seed,
                    "progress": progress,
                    "n_videos": len(selected),
                    "n_scorable": sum(
                        math.isfinite(float(item["commitment_gain"])) for item in selected
                    ),
                    "mean_a_fork_confident": _finite_mean(
                        float(item["a_fork_confident"]) for item in selected
                    ),
                    "mean_commitment_gain": _finite_mean(
                        float(item["commitment_gain"]) for item in selected
                    ),
                    "mean_fork_abstention_rate": _finite_mean(
                        float(item["fork_abstention_rate"]) for item in selected
                    ),
                }
            )
    threshold_summary, video_crossings, video_seed_crossings = summarize_thresholds(
        cells,
        baselines,
        thresholds,
        n_boot=n_video_boot,
        seed=seed,
    )
    pooled_crossings, seed_crossings = pooled_and_seed_crossings(
        cells,
        thresholds,
        n_boot=n_video_boot,
        seed=seed,
    )
    variance = variance_decomposition(
        cells,
        n_fork_boot=n_fork_boot,
        n_video_boot=n_video_boot,
        seed=seed,
    )
    primary = next(
        (row for row in threshold_summary if math.isclose(row["theta_commit"], 0.70)),
        None,
    )
    historical_comparisons: list[dict[str, Any]] = []
    curve_lookup = {row["progress"]: row for row in curve_rows}
    for item in historical:
        historical_curve_differences = []
        for old in item["curve_by_progress"]:
            current = curve_lookup.get(old["progress"])
            if current is not None and old["commitment_gain_confident"] is not None:
                historical_curve_differences.append(
                    {
                        "progress": old["progress"],
                        "b2_minus_historical_commitment_gain": (
                            current["mean_commitment_gain"]
                            - float(old["commitment_gain_confident"])
                        ),
                    }
                )
        historical_comparisons.append(
            {**item, "b2_curve_differences": historical_curve_differences}
        )
    clean_cells = [
        {key: value for key, value in row.items() if key != "fork_labels"} for row in cells
    ]
    replication = classify_frozen_replication(
        pooled_crossings,
        seed_crossings,
        variance,
        progress_grid=progress_grid,
    )
    summary = {
        "schema_version": ANALYSIS_SCHEMA,
        "scientific_status": replication["scientific_status"],
        "replication_label": replication["replication_label"],
        "replication_classification": replication,
        "scientific_scope": "clip-level legacy Class continuity; not event-centered v2 PASS",
        "protocol": protocol_info["path"],
        "protocol_sha256": protocol_info["sha256"],
        "legacy_decision_rule": ABSTENTION_RULE_ID,
        "baseline": "video-conditioned confident pairwise agreement across 17 base finals",
        "commitment_gain": "clip((A_fork-A_ind)/(1-A_ind),0,1)",
        "video_determined_rule": "A_ind >= 0.90",
        "first_crossing_rule": "earliest sampled progress meeting theta",
        "registered_sustained_rule": (
            "earliest sampled progress meeting theta and all later scorable finite points; "
            "unscorable later points are ignored and remain reported missing"
        ),
        "thresholds": [float(value) for value in thresholds],
        "video_cluster_bootstrap": {
            "draws": n_video_boot,
            "seed": seed,
            "confidence_interval": "percentile 95%",
            "cluster": "video; all seeds retained together",
        },
        "cardinality": {
            "videos": len(baselines),
            "base_seeds": len({int(row["base_seed"]) for row in cells}),
            "progress_points": len(progress_grid),
            "video_seed_progress_cells": len(cells),
        },
        "baseline_summary": {
            "mean_a_independent": _finite_mean(
                float(row["a_independent"]) for row in baselines
            ),
            "mean_base_abstention_rate": _finite_mean(
                float(row["base_abstention_rate"]) for row in baselines
            ),
            "n_video_determined": sum(bool(row["video_determined"]) for row in baselines),
        },
        "curves_by_progress": curve_rows,
        "curves_by_base_seed": seed_curve_rows,
        "threshold_sensitivity": threshold_summary,
        "pooled_crossings": pooled_crossings,
        "base_seed_crossings": seed_crossings,
        "theta_0.70_summary": primary,
        "variance_decomposition": variance,
        "historical_comparison_inputs": historical_comparisons,
        "limitations": [
            "posterior inference repeatability variance is not identifiable from one measurement per WAV",
            "fork bootstrap quantifies finite K=12 Monte Carlo uncertainty, not semantic label validity",
            "abstention variance is an identifiable subcomponent of fork resampling, not an independent measurer repeat",
        ],
    }
    if summary["scientific_status"] not in ALLOWED_SCIENTIFIC_STATUSES:
        raise AssertionError("analysis emitted a noncanonical scientific status")
    return (
        _json_safe(summary),
        _json_safe(clean_cells),
        _json_safe(video_crossings),
        _json_safe(baselines),
        _json_safe(video_seed_crossings),
    )


def write_multiseed_analysis(
    merged_completion_path: Path,
    out_dir: Path,
    *,
    protocol_path: Path,
    canonical: bool,
    thresholds: Sequence[float] = SENSITIVITY_THRESHOLDS,
    n_video_boot: int = DEFAULT_BOOTSTRAP_DRAWS,
    n_fork_boot: int = DEFAULT_BOOTSTRAP_DRAWS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    historical_jsons: Sequence[Path] = (),
) -> dict[str, Any]:
    arrays, merged = load_merged_posteriors(merged_completion_path)
    if canonical:
        validate_canonical_analysis_inputs(
            arrays,
            merged,
            protocol_path=protocol_path,
            historical_jsons=historical_jsons,
        )
    summary, cells, video_crossings, baselines, video_seed_crossings = analyze_multiseed(
        arrays,
        protocol_path=protocol_path,
        canonical=canonical,
        thresholds=thresholds,
        n_video_boot=n_video_boot,
        n_fork_boot=n_fork_boot,
        seed=seed,
        historical_jsons=historical_jsons,
    )
    out_dir = Path(out_dir)
    summary_path = out_dir / "CLASS_MULTISEED_COMMITMENT.json"
    cells_path = out_dir / "CLASS_MULTISEED_COMMITMENT.csv"
    video_path = out_dir / "CLASS_VIDEO_CROSSING_DISTRIBUTIONS.csv"
    video_seed_path = out_dir / "CLASS_VIDEO_SEED_CROSSINGS.csv"
    pooled_path = out_dir / "CLASS_POOLED_CROSSINGS.csv"
    seed_path = out_dir / "CLASS_BASE_SEED_CROSSINGS.csv"
    baseline_path = out_dir / "CLASS_VIDEO_BASELINES.csv"
    variance_path = out_dir / "CLASS_VARIANCE_DECOMPOSITION.json"
    atomic_json_create(summary_path, summary)
    atomic_csv_create(cells_path, cells, list(cells[0]))
    atomic_csv_create(video_path, video_crossings, list(video_crossings[0]))
    atomic_csv_create(
        video_seed_path, video_seed_crossings, list(video_seed_crossings[0])
    )
    atomic_csv_create(
        pooled_path, summary["pooled_crossings"], list(summary["pooled_crossings"][0])
    )
    atomic_csv_create(
        seed_path,
        summary["base_seed_crossings"],
        list(summary["base_seed_crossings"][0]),
    )
    atomic_csv_create(baseline_path, baselines, list(baselines[0]))
    atomic_json_create(variance_path, summary["variance_decomposition"])
    outputs = []
    for path in (
        summary_path,
        cells_path,
        video_path,
        video_seed_path,
        pooled_path,
        seed_path,
        baseline_path,
        variance_path,
    ):
        outputs.append(
            {"path": path.name, "sha256": sha256_file(path), "bytes": path.stat().st_size}
        )
    completion = {
        "schema_version": ANALYSIS_SCHEMA,
        "status": "COMPLETE",
        "merged_completion": str(Path(merged_completion_path).resolve()),
        "merged_completion_sha256": sha256_file(merged_completion_path),
        "merged_data_sha256": merged["data_sha256"],
        "canonical_b2": bool(canonical),
        "protocol": str(Path(protocol_path).resolve()),
        "protocol_sha256": sha256_file(protocol_path),
        "scientific_status": summary["scientific_status"],
        "replication_label": summary["replication_label"],
        "thresholds": [float(value) for value in thresholds],
        "n_video_boot": n_video_boot,
        "n_fork_boot": n_fork_boot,
        "seed": seed,
        "historical_inputs": [
            {"path": str(Path(path).resolve()), "sha256": sha256_file(path)}
            for path in historical_jsons
        ],
        "outputs": outputs,
    }
    completion_path = out_dir / "CLASS_MULTISEED_ANALYSIS.completion.json"
    atomic_json_create(completion_path, completion)
    return {**completion, "completion_path": str(completion_path)}


def runtime_provenance(repo: Path, *, command: Sequence[str], device: str) -> dict[str, Any]:
    try:
        import torch

        torch_version = torch.__version__
        cuda_version = torch.version.cuda
    except ImportError:
        torch_version = None
        cuda_version = None
    return {
        "node": socket.gethostname(),
        "device": device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "command": list(command),
        "python_executable": os.path.realpath(os.sys.executable),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "torch_version": torch_version,
        "cuda_version": cuda_version,
        "git_commit": _git_commit(repo),
    }
