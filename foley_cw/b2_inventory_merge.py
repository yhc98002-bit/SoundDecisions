"""Fail-closed reduction of independently validated B2 WAV inventories.

The four B2 banks are large enough that their read-only inventory is run once
per bank, potentially on different nodes.  This module combines those partial
inventories without opening any WAV.  It treats the partial JSONL hashes as the
audio-integrity boundary and revalidates every recorded identity/design field,
the pinned generation manifests, and exact population coverage before writing
a canonical inventory.

There is intentionally no noncanonical command-line mode.  Small tests replace
module constants in-process; production callers can only request the frozen
48-video, 17-seed B2 population.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from .b2_class_closure import (
    B2ClosureError,
    EXPECTED_K_FORKS,
    EXPECTED_N_CLIPS,
    EXPECTED_S_GRID,
    EXPECTED_SEEDS,
    EXPECTED_SEED_SERIES,
    INVENTORY_SCHEMA,
    atomic_json_create,
    atomic_jsonl_create,
    canonical_json_bytes,
    load_inventory,
    sha256_bytes,
    sha256_file,
)


INVENTORY_MERGE_SCHEMA = "sounddecisions.b2_inventory_merge.v1"

# The manifest digest identifies both the seed series and the only accepted
# generation revision for that series.  Keeping this mapping (rather than two
# unordered sets) prevents a valid revision from being reassigned to a
# different bank.
PINNED_GENERATION_LINEAGE: dict[str, dict[str, Any]] = {
    "5c3a334ecfcfb3e91504354c14c8e8dbae71b3bade088b21bec26fb06fd68ed3": {
        "base_seeds": (0, 1, 2, 3, 4),
        "generation_revision": "dbd40d94d4867a53bdaad6d2524f4534817fddbf",
    },
    "b6e176949f531528ccb669759d2057fa0b1b1a14567633d3dd6a2d47e0a8a9e4": {
        "base_seeds": (5, 6, 7, 8),
        "generation_revision": "dd7fdc006fe1f5b3baca4024854d37d533606f74",
    },
    "72bcd677376b1ca44278d7cb6e9ea61910cc07f06fc135b239cdbb54aa4ee6ee": {
        "base_seeds": (9, 10, 11, 12),
        "generation_revision": "6ec5c0dbdfb2b45ca8a27d2a193015d97607d8db",
    },
    "ae3dfb2e0022043206d8d4fcf748a3ce68ba1a3898af560ff3e059ba217e3c51": {
        "base_seeds": (13, 14, 15, 16),
        "generation_revision": "07718809024a674bb938684e6cfdc520026d3122",
    },
}

_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_SHA = re.compile(r"[0-9a-f]{40}")
_RECORD_FIELDS = frozenset(
    {
        "record_id",
        "video_id",
        "base_seed",
        "role",
        "fork_index",
        "progress",
        "audio_path",
        "audio_sha256",
        "audio_bytes",
        "sample_rate",
        "frames",
        "audio_subtype",
        "source_root",
        "generation_manifest_sha256",
        "source_unit_journal",
        "source_unit_journal_sha256",
        "source_cell_journal",
        "source_cell_journal_sha256",
        "cfg",
        "alpha",
        "schedule",
        "generation_model_variant",
        "generation_revision",
    }
)


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _is_git_sha(value: Any) -> bool:
    return isinstance(value, str) and _GIT_SHA.fullmatch(value) is not None


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise B2ClosureError(f"invalid partial inventory manifest: {path}") from exc
    if not isinstance(payload, dict):
        raise B2ClosureError(f"partial inventory manifest is not an object: {path}")
    return payload


def _safe_child(parent: Path, relative: Any) -> Path:
    if not isinstance(relative, str) or not relative:
        raise B2ClosureError("partial inventory records_file is empty")
    rel = Path(relative)
    if rel.is_absolute():
        raise B2ClosureError("partial inventory records_file must be relative")
    parent = parent.resolve()
    child = (parent / rel).resolve()
    if not child.is_relative_to(parent):
        raise B2ClosureError("partial inventory records_file escapes its directory")
    return child


def _load_partial_inventory(
    manifest_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], Path]:
    """Load a partial inventory while independently checking all file digests."""
    manifest_path = Path(manifest_path).resolve()
    manifest = _load_json(manifest_path)
    if manifest.get("schema_version") != INVENTORY_SCHEMA:
        raise B2ClosureError(f"unexpected partial inventory schema: {manifest_path}")
    if (
        manifest.get("status") != "COMPLETE"
        or manifest.get("read_only_inventory") is not True
    ):
        raise B2ClosureError(f"partial inventory is not complete/read-only: {manifest_path}")
    if manifest.get("canonical_b2") is not False:
        raise B2ClosureError(
            f"expected one-root --noncanonical inventory, got canonical_b2={manifest.get('canonical_b2')!r}"
        )
    if manifest.get("verify_wav_headers") is not True:
        raise B2ClosureError("canonical merge rejects partials that skipped WAV-header validation")
    if not _is_sha256(manifest.get("inventory_code_sha256")):
        raise B2ClosureError("partial inventory lacks a valid inventory-code digest")
    if not _is_git_sha(manifest.get("inventory_git_commit")):
        raise B2ClosureError("partial inventory lacks a valid inventory Git revision")

    records_path = _safe_child(manifest_path.parent, manifest.get("records_file"))
    if not records_path.is_file():
        raise B2ClosureError(f"partial inventory JSONL is missing: {records_path}")
    if records_path.stat().st_size != int(manifest.get("records_bytes", -1)):
        raise B2ClosureError(f"partial inventory JSONL byte-count mismatch: {records_path}")
    if sha256_file(records_path) != manifest.get("records_sha256"):
        raise B2ClosureError(f"partial inventory JSONL hash mismatch: {records_path}")

    records: list[dict[str, Any]] = []
    with records_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise B2ClosureError(f"blank partial inventory row: {records_path}:{line_number}")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise B2ClosureError(
                    f"invalid partial inventory JSONL: {records_path}:{line_number}"
                ) from exc
            if not isinstance(row, dict):
                raise B2ClosureError(
                    f"non-object partial inventory row: {records_path}:{line_number}"
                )
            records.append(row)

    if len(records) != int(manifest.get("record_count", -1)):
        raise B2ClosureError("partial inventory record cardinality mismatch")
    ids = [row.get("record_id") for row in records]
    if any(not isinstance(value, str) or not value for value in ids):
        raise B2ClosureError("partial inventory has an invalid record ID")
    if ids != sorted(ids) or len(ids) != len(set(ids)):
        raise B2ClosureError("partial inventory record IDs are unsorted or duplicated")
    ids_digest = sha256_bytes(("\n".join(ids) + "\n").encode("utf-8"))
    if ids_digest != manifest.get("record_ids_sha256"):
        raise B2ClosureError("partial inventory record-ID digest mismatch")
    return records, manifest, records_path


def _normal_absolute(value: Any, *, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise B2ClosureError(f"invalid {field}")
    path = Path(value)
    if not path.is_absolute() or str(path.resolve()) != value:
        raise B2ClosureError(f"{field} is not a normalized absolute path: {value!r}")
    return path


def _record_id(
    video_id: str,
    seed: int,
    role: str,
    progress: float | None,
    fork_index: int | None,
) -> str:
    if role == "base":
        return f"{video_id}__seed{seed}__base"
    assert progress is not None and fork_index is not None
    return f"{video_id}__seed{seed}__s{progress:.2f}__fork{fork_index:02d}"


def _validate_record(
    row: Mapping[str, Any],
    *,
    root: Path,
    videos: set[str],
    seeds: set[int],
    manifest_sha256: str,
    generation_revision: str,
) -> tuple[str, int, str, float | None, int | None]:
    if set(row) != _RECORD_FIELDS:
        missing = sorted(_RECORD_FIELDS - set(row))
        extra = sorted(set(row) - _RECORD_FIELDS)
        raise B2ClosureError(f"inventory row field mismatch: missing={missing}, extra={extra}")
    video_id = row["video_id"]
    if not isinstance(video_id, str) or video_id not in videos:
        raise B2ClosureError(f"invalid inventory video_id: {video_id!r}")
    seed = row["base_seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed not in seeds:
        raise B2ClosureError(f"invalid inventory base_seed: {seed!r}")
    role = row["role"]
    if role not in {"base", "fork"}:
        raise B2ClosureError(f"invalid inventory role: {role!r}")

    if row["source_root"] != str(root):
        raise B2ClosureError("record source_root does not match its partial root")
    if row["generation_manifest_sha256"] != manifest_sha256:
        raise B2ClosureError("record generation-manifest digest mismatch")
    if row["generation_revision"] != generation_revision:
        raise B2ClosureError("record generation revision is not pinned for its bank")
    fixed = {
        "sample_rate": 16000,
        "frames": 128000,
        "audio_subtype": "FLOAT",
        "cfg": 4.5,
        "alpha": 0.8,
        "schedule": "sqrt_down",
        "generation_model_variant": "small_16k",
    }
    for field, expected in fixed.items():
        value = row[field]
        if isinstance(expected, float):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isclose(
                float(value), expected, rel_tol=0.0, abs_tol=1e-12
            ):
                raise B2ClosureError(
                    f"record {field} mismatch: {value!r} != {expected!r}"
                )
        elif value != expected:
            raise B2ClosureError(f"record {field} mismatch: {value!r} != {expected!r}")
    if isinstance(row["audio_bytes"], bool) or not isinstance(row["audio_bytes"], int):
        raise B2ClosureError("record audio_bytes must be an integer")
    if row["audio_bytes"] <= 44:
        raise B2ClosureError("record audio_bytes is not a plausible WAV size")
    if not _is_sha256(row["audio_sha256"]):
        raise B2ClosureError("record audio_sha256 is invalid")
    if not _is_sha256(row["source_unit_journal_sha256"]):
        raise B2ClosureError("record unit-journal digest is invalid")

    expected_unit = root / "journal" / "units" / f"{video_id}__seed{seed}.json"
    if row["source_unit_journal"] != str(expected_unit):
        raise B2ClosureError("record unit-journal path mismatch")

    progress: float | None
    fork_index: int | None
    if role == "base":
        progress = None
        fork_index = None
        if row["progress"] is not None or row["fork_index"] is not None:
            raise B2ClosureError("base record contains fork identity")
        if row["source_cell_journal"] is not None or row["source_cell_journal_sha256"] is not None:
            raise B2ClosureError("base record contains cell-journal metadata")
        expected_audio = root / "raw" / video_id / f"seed{seed}" / "base.wav"
    else:
        raw_progress = row["progress"]
        if isinstance(raw_progress, bool) or not isinstance(raw_progress, (int, float)):
            raise B2ClosureError("fork progress is not numeric")
        progress = float(raw_progress)
        if progress not in EXPECTED_S_GRID:
            raise B2ClosureError(f"fork progress is outside the frozen grid: {progress}")
        raw_fork = row["fork_index"]
        if isinstance(raw_fork, bool) or not isinstance(raw_fork, int):
            raise B2ClosureError("fork_index is not an integer")
        fork_index = raw_fork
        if not 0 <= fork_index < EXPECTED_K_FORKS:
            raise B2ClosureError(f"fork_index is outside the frozen range: {fork_index}")
        expected_cell = (
            root
            / "journal"
            / "cells"
            / f"{video_id}__seed{seed}__s{progress:.2f}.json"
        )
        if row["source_cell_journal"] != str(expected_cell):
            raise B2ClosureError("record cell-journal path mismatch")
        if not _is_sha256(row["source_cell_journal_sha256"]):
            raise B2ClosureError("record cell-journal digest is invalid")
        expected_audio = (
            root
            / "raw"
            / video_id
            / f"seed{seed}"
            / f"s{progress:.2f}"
            / f"fork{fork_index:02d}.wav"
        )

    if row["audio_path"] != str(expected_audio):
        raise B2ClosureError("record audio path does not match its identity")
    expected_id = _record_id(video_id, seed, role, progress, fork_index)
    if row["record_id"] != expected_id:
        raise B2ClosureError(f"record ID mismatch: {row['record_id']!r} != {expected_id!r}")
    return video_id, seed, role, progress, fork_index


def _validate_partial(
    manifest_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    records, manifest, records_path = _load_partial_inventory(manifest_path)
    roots = manifest.get("roots")
    if not isinstance(roots, list) or len(roots) != 1 or not isinstance(roots[0], dict):
        raise B2ClosureError("each partial inventory must contain exactly one root")
    root_row = roots[0]
    root = _normal_absolute(root_row.get("root"), field="partial source root")
    generation_manifest = _normal_absolute(
        root_row.get("generation_manifest"), field="generation manifest path"
    )
    if generation_manifest != root / "generation_manifest.json":
        raise B2ClosureError("generation manifest path is not rooted in the source bank")
    if not generation_manifest.is_file():
        raise B2ClosureError(f"generation manifest is missing: {generation_manifest}")
    generation_manifest_sha256 = root_row.get("generation_manifest_sha256")
    if generation_manifest_sha256 not in PINNED_GENERATION_LINEAGE:
        raise B2ClosureError("partial inventory generation manifest is not pinned")
    if sha256_file(generation_manifest) != generation_manifest_sha256:
        raise B2ClosureError("on-disk generation manifest no longer matches the pinned digest")

    lineage = PINNED_GENERATION_LINEAGE[generation_manifest_sha256]
    seeds = tuple(manifest.get("base_seeds", []))
    if seeds != tuple(lineage["base_seeds"]) or seeds not in EXPECTED_SEED_SERIES:
        raise B2ClosureError(f"partial inventory has an unexpected seed series: {seeds}")
    if root_row.get("base_seeds") != list(seeds):
        raise B2ClosureError("partial root seed metadata disagrees with its manifest")
    videos = manifest.get("video_ids")
    if (
        not isinstance(videos, list)
        or len(videos) != EXPECTED_N_CLIPS
        or len(set(videos)) != EXPECTED_N_CLIPS
        or any(not isinstance(value, str) or not value for value in videos)
    ):
        raise B2ClosureError("partial inventory does not contain 48 unique video IDs")
    if manifest.get("progress_grid") != list(EXPECTED_S_GRID):
        raise B2ClosureError("partial inventory progress grid mismatch")
    if manifest.get("k_forks") != EXPECTED_K_FORKS:
        raise B2ClosureError("partial inventory fork count mismatch")

    expected_base = EXPECTED_N_CLIPS * len(seeds)
    expected_fork = expected_base * len(EXPECTED_S_GRID) * EXPECTED_K_FORKS
    expected_total = expected_base + expected_fork
    count_fields = {
        "record_count": expected_total,
        "base_record_count": expected_base,
        "fork_record_count": expected_fork,
    }
    for field, expected in count_fields.items():
        if manifest.get(field) != expected:
            raise B2ClosureError(f"partial {field} mismatch: {manifest.get(field)!r}")
    root_counts = {
        "records": expected_total,
        "unit_journals": expected_base,
        "cell_journals": expected_base * len(EXPECTED_S_GRID),
        "base_wavs": expected_base,
        "fork_wavs": expected_fork,
    }
    for field, expected in root_counts.items():
        if root_row.get(field) != expected:
            raise B2ClosureError(f"partial root {field} mismatch: {root_row.get(field)!r}")
    if not _is_sha256(root_row.get("journal_hash_set_sha256")):
        raise B2ClosureError("partial root lacks a valid journal-hash-set digest")

    identities: set[tuple[str, int, str, float | None, int | None]] = set()
    unit_hashes: dict[tuple[str, int], str] = {}
    cell_hashes: dict[tuple[str, int, float], str] = {}
    role_counts: Counter[str] = Counter()
    for row in records:
        identity = _validate_record(
            row,
            root=root,
            videos=set(videos),
            seeds=set(seeds),
            manifest_sha256=generation_manifest_sha256,
            generation_revision=str(lineage["generation_revision"]),
        )
        if identity in identities:
            raise B2ClosureError(f"duplicate B2 record identity: {identity}")
        identities.add(identity)
        video_id, seed, role, progress, _fork = identity
        unit_key = (video_id, seed)
        unit_sha = str(row["source_unit_journal_sha256"])
        if unit_key in unit_hashes and unit_hashes[unit_key] != unit_sha:
            raise B2ClosureError(f"unit-journal digest changes within {unit_key}")
        unit_hashes[unit_key] = unit_sha
        if role == "fork":
            assert progress is not None
            cell_key = (video_id, seed, progress)
            cell_sha = str(row["source_cell_journal_sha256"])
            if cell_key in cell_hashes and cell_hashes[cell_key] != cell_sha:
                raise B2ClosureError(f"cell-journal digest changes within {cell_key}")
            cell_hashes[cell_key] = cell_sha
        role_counts[role] += 1
    if len(identities) != expected_total:
        raise B2ClosureError("partial inventory is not an exact identity partition")
    if role_counts != Counter({"base": expected_base, "fork": expected_fork}):
        raise B2ClosureError("partial inventory role cardinalities are inconsistent")
    if len(unit_hashes) != expected_base or len(cell_hashes) != expected_base * len(
        EXPECTED_S_GRID
    ):
        raise B2ClosureError("partial journal identity coverage is incomplete")

    source = {
        "manifest_path": str(Path(manifest_path).resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "records_path": str(records_path),
        "records_sha256": manifest["records_sha256"],
        "records_bytes": manifest["records_bytes"],
        "root": str(root),
        "base_seeds": list(seeds),
        "generation_manifest_sha256": generation_manifest_sha256,
        "generation_revision": lineage["generation_revision"],
        "inventory_code_sha256": manifest["inventory_code_sha256"],
        "inventory_git_commit": manifest["inventory_git_commit"],
    }
    return records, manifest, source


def _git_commit(repo: Path) -> str:
    try:
        value = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise B2ClosureError("cannot record inventory-merger Git revision") from exc
    if not _is_git_sha(value):
        raise B2ClosureError("inventory-merger Git revision is invalid")
    return value


def merge_partial_inventories(
    manifest_paths: Sequence[Path], out_dir: Path
) -> dict[str, Any]:
    """Validate four one-root parts and create one canonical B2 inventory.

    The operation is deterministic and create-only.  It does not stat, read,
    decode, or hash any WAV; all audio checks were completed by the partial
    inventories and are anchored here by their JSONL digests.
    """
    resolved_inputs = [Path(path).resolve() for path in manifest_paths]
    if len(resolved_inputs) != 4 or len(set(resolved_inputs)) != 4:
        raise B2ClosureError("canonical B2 merge requires four distinct partial manifests")
    out_dir = Path(out_dir)
    records_path = out_dir / "B2_WAV_INVENTORY.jsonl"
    manifest_path = out_dir / "B2_WAV_INVENTORY_MANIFEST.json"
    existing = [str(path) for path in (records_path, manifest_path) if path.exists()]
    if existing:
        raise B2ClosureError(f"canonical inventory outputs are create-only: {existing}")

    parts = [_validate_partial(path) for path in resolved_inputs]
    parts.sort(key=lambda item: min(item[1]["base_seeds"]))
    reference_videos = parts[0][1]["video_ids"]
    roots = [item[1]["roots"][0] for item in parts]
    root_paths = [row["root"] for row in roots]
    if len(root_paths) != len(set(root_paths)):
        raise B2ClosureError("partial inventories do not refer to four disjoint roots")
    if any(item[1]["video_ids"] != reference_videos for item in parts[1:]):
        raise B2ClosureError("partial inventories disagree on video identity/order")
    if any(item[1]["progress_grid"] != list(EXPECTED_S_GRID) for item in parts):
        raise B2ClosureError("partial inventories disagree on the progress design")
    if any(item[1]["k_forks"] != EXPECTED_K_FORKS for item in parts):
        raise B2ClosureError("partial inventories disagree on the fork design")

    observed_series = tuple(tuple(item[1]["base_seeds"]) for item in parts)
    if observed_series != tuple(EXPECTED_SEED_SERIES):
        raise B2ClosureError(f"partial seed-series partition mismatch: {observed_series}")
    union_seeds = sorted(seed for series in observed_series for seed in series)
    if tuple(union_seeds) != tuple(EXPECTED_SEEDS) or len(union_seeds) != len(
        set(union_seeds)
    ):
        raise B2ClosureError("partial seed union is not exactly 0..16")
    observed_manifests = {
        item[2]["generation_manifest_sha256"] for item in parts
    }
    if observed_manifests != set(PINNED_GENERATION_LINEAGE):
        raise B2ClosureError("partial generation-manifest lineage is incomplete")

    records = sorted(
        (row for item in parts for row in item[0]), key=lambda row: row["record_id"]
    )
    ids = [str(row["record_id"]) for row in records]
    expected_base = EXPECTED_N_CLIPS * len(EXPECTED_SEEDS)
    expected_fork = expected_base * len(EXPECTED_S_GRID) * EXPECTED_K_FORKS
    expected_total = expected_base + expected_fork
    if len(records) != expected_total or len(ids) != len(set(ids)):
        raise B2ClosureError(
            f"canonical union cardinality mismatch: records={len(records)}, unique={len(set(ids))}"
        )
    if sum(row["role"] == "base" for row in records) != expected_base:
        raise B2ClosureError("canonical union base cardinality mismatch")
    if sum(row["role"] == "fork" for row in records) != expected_fork:
        raise B2ClosureError("canonical union fork cardinality mismatch")

    source_parts = [item[2] for item in parts]
    source_parts_sha256 = sha256_bytes(canonical_json_bytes(source_parts, indent=None))
    atomic_jsonl_create(records_path, records)
    repo = Path(__file__).resolve().parents[1]
    summary = {
        "schema_version": INVENTORY_SCHEMA,
        "status": "COMPLETE",
        "read_only_inventory": True,
        "canonical_b2": True,
        "verify_wav_headers": True,
        "roots": roots,
        "record_count": expected_total,
        "base_record_count": expected_base,
        "fork_record_count": expected_fork,
        "video_ids": reference_videos,
        "base_seeds": list(EXPECTED_SEEDS),
        "progress_grid": list(EXPECTED_S_GRID),
        "k_forks": EXPECTED_K_FORKS,
        "record_ids_sha256": sha256_bytes(("\n".join(ids) + "\n").encode("utf-8")),
        "records_file": records_path.name,
        "records_sha256": sha256_file(records_path),
        "records_bytes": records_path.stat().st_size,
        "inventory_code_sha256": sha256_file(Path(__file__)),
        "inventory_git_commit": _git_commit(repo),
        "inventory_merge_schema": INVENTORY_MERGE_SCHEMA,
        "source_part_count": len(source_parts),
        "source_parts": source_parts,
        "source_parts_sha256": source_parts_sha256,
        "wav_rehash_during_merge": False,
    }
    atomic_json_create(manifest_path, summary)

    # Prove compatibility with the consumer loader before returning.  The
    # artifacts remain immutable if this final validation fails.
    loaded_records, loaded_manifest = load_inventory(manifest_path)
    if len(loaded_records) != expected_total or loaded_manifest != summary:
        raise B2ClosureError("new canonical inventory failed compatibility validation")
    return {
        **summary,
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
    }


__all__ = [
    "INVENTORY_MERGE_SCHEMA",
    "PINNED_GENERATION_LINEAGE",
    "merge_partial_inventories",
]
