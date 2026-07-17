"""Outcome-blind feasibility gate for the legacy Material continuity 2AFC.

This module constructs candidate/positive/negative *identities and metadata*
before any preview is replayed or any candidate/reference similarity is read.
It deliberately does not import the Material embedder and never decodes the
legacy Phase-2 ``correct`` field.  The only independent material evidence is
an exact FoleyBench UCS category named in :data:`ADMISSIBLE_MATERIALS`.

The high-level contract is:

* inventory the exact 200 x 4 x 8 legacy cells without parsing their outcomes;
* obtain each candidate subject's non-abstaining retained ``p1cfg1`` Class;
* choose a disjoint same-video retained final as the positive;
* choose a different-video retained final with the same Class, different
  admissible material, exact scene metadata, and frozen timing/loudness bands;
* retain only videos for which all four subjects have a strict match;
* freeze a manifest only at >=20 complete videos and >=640 cells.

Source-audio loudness is an outcome-independent covariate.  It is calculated
from the original MP4 audio with a recorded ffmpeg command and hashes.  No
network access, model inference, preview replay, or embedding access occurs.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = "sounddecisions_legacy_material_reference_manifest_v1"
REPORT_SCHEMA_VERSION = "sounddecisions_material_reference_feasibility_v1"
LOUDNESS_SCHEMA_VERSION = "sounddecisions_source_audio_loudness_v1"

ADMISSIBLE_MATERIALS: tuple[str, ...] = (
    "CERAMICS",
    "CLOTH",
    "GLASS",
    "METAL",
    "PAPER",
    "PLASTIC",
    "ROCKS",
    "WOOD",
)
PROGRESS_POINTS: tuple[float, ...] = (0.05, 0.15, 0.25, 0.35, 0.45, 0.60, 0.75, 0.90)
SUBJECT_INDICES: tuple[int, ...] = (0, 1, 2, 3)
EXPECTED_LEGACY_VIDEOS = 200
EXPECTED_CELLS_PER_VIDEO = len(SUBJECT_INDICES) * len(PROGRESS_POINTS)
EXPECTED_LEGACY_CELLS = EXPECTED_LEGACY_VIDEOS * EXPECTED_CELLS_PER_VIDEO
MIN_VALID_VIDEOS = 20
MIN_VALID_CELLS = 640
ABSTAIN_LABEL = "abstain"

MATCHING_BANDS: tuple[dict[str, float | str], ...] = (
    {"difficulty": "hard", "timing_seconds_lte": 0.25, "loudness_db_lte": 3.0},
    {"difficulty": "medium", "timing_seconds_lte": 0.50, "loudness_db_lte": 6.0},
    {"difficulty": "easy", "timing_seconds_lte": 1.00, "loudness_db_lte": 9.0},
)

_FINAL_ID_RE = re.compile(r"^(?P<clip>.+)__p1cfg1_ind(?P<j>\d+)$")
_GEN_ID_BYTES_RE = re.compile(br'"gen_id"\s*:\s*"([^"\\]+__p1cfg1_ind\d+)"')
_MATERIAL_CELL_RE = re.compile(
    br'\{"axis_id"\s*:\s*"material"\s*,\s*'
    br'"clip"\s*:\s*"([^"\\]+)"\s*,\s*'
    br'"correct"\s*:\s*[^,}\]]+\s*,\s*'
    br'"j"\s*:\s*(\d+)\s*,\s*'
    br'"probe"\s*:\s*"[^"\\]*"\s*,\s*'
    br'"s"\s*:\s*([0-9.eE+\-]+)\s*,\s*'
    br'"target"\s*:\s*"[^"\\]*"\s*\}',
)

# Any of these in a frozen reference manifest indicates outcome leakage.  The
# deterministic A/B field ``correct_choice`` is intentionally not in this set:
# it records which slot contains the positive, not a measured response.
FORBIDDEN_OUTCOME_KEYS = frozenset({
    "embedding",
    "candidate_embedding",
    "positive_embedding",
    "negative_embedding",
    "similarity",
    "positive_similarity",
    "negative_similarity",
    "similarity_margin",
    "cosine",
    "margin",
    "decision",
    "observed_choice",
    "correct",
})


class FeasibilityInputError(ValueError):
    """Raised when an input violates the outcome-blind feasibility contract."""


@dataclass(frozen=True)
class FinalInventory:
    """Retained final IDs and confident Class labels from the measurement log."""

    class_labels: dict[str, str]
    material_final_ids: frozenset[str]
    measurement_sha256: str
    counters: dict[str, int]
    errors: tuple[str, ...]


@dataclass(frozen=True)
class Phase2Inventory:
    """Outcome-blind inventory of the legacy Phase-2 candidate cells."""

    clips: tuple[str, ...]
    cells_by_clip: dict[str, frozenset[tuple[int, float]]]
    journal_hashes: dict[str, str]
    aggregate_sha256: str
    errors: tuple[str, ...]


@dataclass(frozen=True)
class LoudnessCollection:
    """Source-audio loudness records plus extraction failures/provenance."""

    records: dict[str, dict[str, Any]]
    failures: dict[str, str]
    ffmpeg: dict[str, Any]


def _canonical_json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False,
                       allow_nan=False) + "\n").encode("utf-8")


def sha256_file(path: Path, chunk_bytes: int = 4 * 1024 * 1024) -> str:
    """Return the SHA256 of *path* without loading it all into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _natural_key(value: str) -> tuple[int, int | str]:
    return (0, int(value)) if value.isdigit() else (1, value)


def scan_retained_finals(measurements_jsonl: Path) -> FinalInventory:
    """Scan retained ``p1cfg1`` finals without reading Material embeddings.

    Class rows are JSON-decoded because the confident coarse label is an input
    to matching.  Material rows are *not* JSON-decoded: a byte-level scan only
    records the generation ID and exact retained-final role.  Consequently the
    embedding array is neither accessed nor returned.
    """
    path = Path(measurements_jsonl)
    if not path.is_file():
        raise FileNotFoundError(f"measurement JSONL not found: {path}")

    digest = hashlib.sha256()
    class_labels: dict[str, str] = {}
    material_ids: set[str] = set()
    counters: Counter[str] = Counter()
    errors: list[str] = []

    with path.open("rb") as handle:
        for line_no, line in enumerate(handle, start=1):
            digest.update(line)
            if b"__p1cfg1_ind" not in line:
                continue
            if b'"role": "p1cfg1_independent"' not in line:
                continue
            match = _GEN_ID_BYTES_RE.search(line)
            if match is None:
                continue
            gen_id = match.group(1).decode("utf-8")
            id_match = _FINAL_ID_RE.fullmatch(gen_id)
            if id_match is None:
                continue

            if re.search(br'"axis_id"\s*:\s*"material"', line):
                counters["material_rows_seen"] += 1
                if gen_id in material_ids:
                    errors.append(f"duplicate Material retained final at line {line_no}: {gen_id}")
                material_ids.add(gen_id)
                continue

            if not re.search(br'"axis_id"\s*:\s*"class"', line):
                continue
            counters["class_rows_seen"] += 1
            try:
                row = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                errors.append(f"malformed Class row at line {line_no}: {exc}")
                continue
            target = row.get("target")
            extra = row.get("extra")
            if not isinstance(target, dict) or not isinstance(extra, dict):
                errors.append(f"Class row missing target/extra objects at line {line_no}")
                continue
            label = target.get("label")
            if not isinstance(label, str) or not label.strip():
                errors.append(f"Class row has no label at line {line_no}: {gen_id}")
                continue
            expected_clip = id_match.group("clip")
            expected_j = int(id_match.group("j"))
            if (str(extra.get("role", "")) != "p1cfg1_independent"
                    or str(extra.get("clip", "")) != expected_clip
                    or int(extra.get("j", -1)) != expected_j):
                errors.append(f"Class row lineage mismatch at line {line_no}: {gen_id}")
                continue
            old = class_labels.get(gen_id)
            if old is not None and old != label:
                errors.append(f"conflicting Class labels for {gen_id}: {old!r} vs {label!r}")
            elif old is not None:
                errors.append(f"duplicate Class retained final at line {line_no}: {gen_id}")
            class_labels[gen_id] = label

    counters["class_final_ids"] = len(class_labels)
    counters["material_final_ids"] = len(material_ids)
    counters["confident_class_final_ids"] = sum(
        label != ABSTAIN_LABEL for label in class_labels.values()
    )
    return FinalInventory(
        class_labels=class_labels,
        material_final_ids=frozenset(material_ids),
        measurement_sha256=digest.hexdigest(),
        counters=dict(sorted(counters.items())),
        errors=tuple(errors),
    )


def inventory_phase2_journals(journal_dir: Path) -> Phase2Inventory:
    """Inventory exact Material cells without decoding legacy outcomes.

    The journals contain an obsolete scalar in ``correct``.  This function
    uses a byte regular expression that captures only ``clip``, ``j`` and ``s``;
    the scalar is syntactically skipped and never converted to a number.
    """
    root = Path(journal_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Phase-2 journal directory not found: {root}")
    paths = sorted(root.glob("p2cfg1__*.json"), key=lambda p: _natural_key(
        p.stem.removeprefix("p2cfg1__")
    ))
    errors: list[str] = []
    cells_by_clip: dict[str, frozenset[tuple[int, float]]] = {}
    journal_hashes: dict[str, str] = {}
    aggregate = hashlib.sha256()
    expected_cells = frozenset((j, s) for j in SUBJECT_INDICES for s in PROGRESS_POINTS)

    for path in paths:
        clip = path.stem.removeprefix("p2cfg1__")
        raw = path.read_bytes()
        cells: list[tuple[int, float]] = []
        for match in _MATERIAL_CELL_RE.finditer(raw):
            row_clip = match.group(1).decode("utf-8")
            j = int(match.group(2))
            s = float(match.group(3))
            if row_clip != clip:
                errors.append(f"journal {path.name} contains Material clip {row_clip!r}")
            cells.append((j, s))
        cell_set = frozenset(cells)
        if len(cells) != EXPECTED_CELLS_PER_VIDEO:
            errors.append(
                f"journal {path.name}: found {len(cells)} outcome-blind Material cells; "
                f"expected {EXPECTED_CELLS_PER_VIDEO}"
            )
        if len(cell_set) != len(cells):
            errors.append(f"journal {path.name}: duplicate Material candidate cells")
        if cell_set != expected_cells:
            missing = sorted(expected_cells - cell_set)
            extra = sorted(cell_set - expected_cells)
            errors.append(f"journal {path.name}: cell grid mismatch; missing={missing}, extra={extra}")
        cells_by_clip[clip] = cell_set
        # Deliberately hash only the outcome-blind inventory.  A raw-file hash
        # would change with the obsolete ``correct`` cosine and make a frozen
        # reference manifest byte-dependent on an observed outcome.
        inventory_text = "\n".join(
            f"{clip}\tj{j}\ts{s:.2f}" for j, s in sorted(cell_set)
        ) + "\n"
        inventory_hash = stable_hash(inventory_text)
        journal_hashes[clip] = inventory_hash
        aggregate.update(f"{clip}\0{inventory_hash}\n".encode("utf-8"))

    if len(paths) != EXPECTED_LEGACY_VIDEOS:
        errors.append(
            f"legacy journal count {len(paths)} != required {EXPECTED_LEGACY_VIDEOS}"
        )
    return Phase2Inventory(
        clips=tuple(sorted(cells_by_clip, key=_natural_key)),
        cells_by_clip=cells_by_clip,
        journal_hashes=journal_hashes,
        aggregate_sha256=aggregate.hexdigest(),
        errors=tuple(errors),
    )


def load_clip_metadata(index_csv: Path) -> tuple[dict[str, dict[str, str]], str, tuple[str, ...]]:
    """Load only matching fields from ``clips_index.csv``.

    Captions are intentionally omitted so they cannot become material truth.
    Exact category/scene strings are stripped but not case-normalized.
    """
    path = Path(index_csv)
    required = {
        "key", "path", "ucs_category", "audioset_category", "source_type",
        "discrete_vs_rest", "status",
    }
    rows: dict[str, dict[str, str]] = {}
    errors: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing_columns = sorted(required - set(reader.fieldnames or ()))
        if missing_columns:
            raise FeasibilityInputError(
                f"metadata CSV missing required columns: {missing_columns}"
            )
        for line_no, row in enumerate(reader, start=2):
            clip = str(row.get("key", "")).strip()
            if not clip:
                errors.append(f"metadata row {line_no} has empty key")
                continue
            if clip in rows:
                errors.append(f"duplicate metadata key {clip!r} at row {line_no}")
                continue
            rows[clip] = {
                "path": str(row.get("path", "")).strip(),
                "ucs_category": str(row.get("ucs_category", "")).strip(),
                "audioset_category": str(row.get("audioset_category", "")).strip(),
                "source_type": str(row.get("source_type", "")).strip(),
                "discrete_vs_rest": str(row.get("discrete_vs_rest", "")).strip(),
                "status": str(row.get("status", "")).strip(),
            }
    return rows, sha256_file(path), tuple(errors)


def load_primary_detector_timing(
    anchors_json: Path,
) -> tuple[dict[str, dict[str, Any]], str, tuple[str, ...]]:
    """Load the primary salience-ranked automatic audio onset per clip."""
    path = Path(anchors_json)
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("clips")
    if not isinstance(rows, list):
        raise FeasibilityInputError("anchors JSON must contain a clips array")
    timings: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            errors.append(f"anchors clips[{index}] is not an object")
            continue
        clip = str(row.get("key", "")).strip()
        audio = row.get("audio")
        if not clip or not isinstance(audio, dict):
            continue
        stamps = audio.get("timestamps")
        source = str(audio.get("source", "")).strip()
        if not isinstance(stamps, list) or not stamps:
            continue
        try:
            timing_s = float(stamps[0])
        except (TypeError, ValueError):
            errors.append(f"clip {clip}: invalid primary audio detector timestamp")
            continue
        if not math.isfinite(timing_s) or timing_s < 0:
            errors.append(f"clip {clip}: non-finite/negative detector timestamp")
            continue
        if clip in timings:
            errors.append(f"duplicate anchor key {clip!r}")
            continue
        timings[clip] = {
            "timing_s": timing_s,
            "source": source,
            "selection": "timestamps[0] (salience-descending primary audio onset)",
        }
    return timings, sha256_file(path), tuple(errors)


def resolve_video_path(metadata_path: str, metadata_csv: Path, clips_root: Path, clip: str) -> Path:
    """Resolve retained source MP4 without downloading or searching broadly."""
    raw = Path(metadata_path)
    candidates = [raw] if raw.is_absolute() else [Path(metadata_csv).parent / raw, raw]
    candidates.append(Path(clips_root) / f"{clip}.mp4")
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return candidates[-1]


def ffmpeg_provenance(ffmpeg_binary: str = "ffmpeg") -> dict[str, Any]:
    """Resolve and fingerprint ffmpeg; raise if it is unavailable."""
    resolved = shutil.which(ffmpeg_binary)
    if resolved is None:
        candidate = Path(ffmpeg_binary)
        if not candidate.is_file():
            raise FileNotFoundError(f"ffmpeg executable not found: {ffmpeg_binary}")
        resolved = str(candidate)
    resolved_path = Path(resolved).resolve()
    proc = subprocess.run(
        [str(resolved_path), "-version"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg -version failed ({proc.returncode}): "
            f"{proc.stderr.decode('utf-8', errors='replace')[:500]}"
        )
    first_line = proc.stdout.decode("utf-8", errors="replace").splitlines()
    return {
        "resolved_path": str(resolved_path),
        "executable_sha256": sha256_file(resolved_path),
        "version_first_line": first_line[0] if first_line else "",
        "version_command": [str(resolved_path), "-version"],
    }


def source_audio_loudness(
    video_path: Path,
    ffmpeg_info: Mapping[str, Any],
    *,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> dict[str, Any]:
    """Decode original MP4 audio as mono 16 kHz float32 and compute RMS dBFS."""
    path = Path(video_path)
    if not path.is_file():
        raise FileNotFoundError(f"source video not found: {path}")
    command = [
        str(ffmpeg_info["resolved_path"]),
        "-hide_banner", "-loglevel", "error", "-nostdin",
        "-i", str(path),
        "-map", "0:a:0", "-vn", "-ac", "1", "-ar", "16000",
        "-c:a", "pcm_f32le", "-f", "f32le", "pipe:1",
    ]
    proc = runner(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=120,
    )
    if proc.returncode != 0:
        message = proc.stderr.decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(f"source-audio decode failed ({proc.returncode}): {message}")
    pcm_bytes = bytes(proc.stdout)
    if not pcm_bytes or len(pcm_bytes) % 4:
        raise RuntimeError(
            f"decoded PCM byte count must be positive and divisible by four; got {len(pcm_bytes)}"
        )
    pcm = np.frombuffer(pcm_bytes, dtype="<f4")
    if not np.all(np.isfinite(pcm)):
        raise RuntimeError("decoded source audio contains non-finite samples")
    rms = float(np.sqrt(np.mean(np.square(pcm, dtype=np.float64))))
    rms_dbfs = float(20.0 * np.log10(max(rms, 1e-12)))
    return {
        "source_video_path": str(path.resolve()),
        "source_video_sha256": sha256_file(path),
        "decode_command": command,
        "decode_command_sha256": stable_hash("\0".join(command)),
        "decoded_pcm_sha256": hashlib.sha256(pcm_bytes).hexdigest(),
        "sample_rate_hz": 16000,
        "channels": 1,
        "sample_format": "pcm_f32le",
        "n_samples": int(pcm.size),
        "rms": rms,
        "rms_dbfs": rms_dbfs,
    }


def collect_source_loudness(
    clips: Iterable[str],
    metadata: Mapping[str, Mapping[str, str]],
    metadata_csv: Path,
    clips_root: Path,
    *,
    ffmpeg_binary: str = "ffmpeg",
    workers: int = 1,
    ffmpeg_info: Mapping[str, Any] | None = None,
    extractor: Callable[[Path, Mapping[str, Any]], dict[str, Any]] = source_audio_loudness,
) -> LoudnessCollection:
    """Collect deterministic source loudness, recording every failure by clip."""
    if workers < 1:
        raise ValueError("workers must be >=1")
    wanted = sorted(set(str(c) for c in clips), key=_natural_key)
    try:
        info = dict(ffmpeg_info or ffmpeg_provenance(ffmpeg_binary))
    except Exception as exc:  # an absent decoder is a recorded blocker
        return LoudnessCollection(
            records={},
            failures={clip: f"FFMPEG_UNAVAILABLE: {exc}" for clip in wanted},
            ffmpeg={"requested_binary": ffmpeg_binary, "error": str(exc)},
        )

    def one(clip: str) -> tuple[str, dict[str, Any]]:
        row = metadata.get(clip)
        if row is None:
            raise KeyError("metadata row missing")
        path = resolve_video_path(str(row.get("path", "")), metadata_csv, clips_root, clip)
        record = extractor(path, info)
        return clip, record

    records: dict[str, dict[str, Any]] = {}
    failures: dict[str, str] = {}
    if workers == 1:
        for clip in wanted:
            try:
                key, record = one(clip)
                records[key] = record
            except Exception as exc:
                failures[clip] = f"SOURCE_LOUDNESS_UNAVAILABLE: {exc}"
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(one, clip): clip for clip in wanted}
            for future in as_completed(futures):
                clip = futures[future]
                try:
                    key, record = future.result()
                    records[key] = record
                except Exception as exc:
                    failures[clip] = f"SOURCE_LOUDNESS_UNAVAILABLE: {exc}"
    return LoudnessCollection(
        records={key: records[key] for key in sorted(records, key=_natural_key)},
        failures={key: failures[key] for key in sorted(failures, key=_natural_key)},
        ffmpeg=info,
    )


def validate_protocol(protocol: Mapping[str, Any]) -> None:
    """Fail closed if the checked-in protocol differs from this matcher."""
    material = protocol.get("material_continuity")
    if not isinstance(material, Mapping):
        raise FeasibilityInputError("protocol lacks material_continuity")
    evidence = material.get("admissible_material_evidence")
    if not isinstance(evidence, Mapping):
        raise FeasibilityInputError("protocol lacks admissible_material_evidence")
    if tuple(evidence.get("exact_categories", ())) != ADMISSIBLE_MATERIALS:
        raise FeasibilityInputError("protocol material categories differ from implementation")
    if evidence.get("caption_inference_allowed") is not False:
        raise FeasibilityInputError("caption inference must remain forbidden")
    if evidence.get("embedding_defined_truth_allowed") is not False:
        raise FeasibilityInputError("embedding-defined truth must remain forbidden")
    if tuple(material.get("matching_bands", ())) != MATCHING_BANDS:
        raise FeasibilityInputError("protocol matching bands differ from implementation")
    if int(material.get("minimum_valid_candidate_videos", -1)) != MIN_VALID_VIDEOS:
        raise FeasibilityInputError("protocol minimum candidate-video coverage differs")
    if int(material.get("minimum_valid_videos_per_progress", -1)) != MIN_VALID_VIDEOS:
        raise FeasibilityInputError("protocol per-progress coverage differs")
    if int(material.get("minimum_valid_cells", -1)) != MIN_VALID_CELLS:
        raise FeasibilityInputError("protocol minimum cell coverage differs")


def _difficulty(timing_distance_s: float, loudness_distance_db: float) -> tuple[int, dict[str, Any]] | None:
    for rank, band in enumerate(MATCHING_BANDS):
        timing_limit = float(band["timing_seconds_lte"])
        loudness_limit = float(band["loudness_db_lte"])
        if timing_distance_s <= timing_limit and loudness_distance_db <= loudness_limit:
            return rank, {
                "difficulty": str(band["difficulty"]),
                "timing_seconds_lte": timing_limit,
                "loudness_db_lte": loudness_limit,
                "normalized_distance": (
                    timing_distance_s / timing_limit + loudness_distance_db / loudness_limit
                ),
            }
    return None


def _final_ids_by_clip(final_ids: Iterable[str]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = defaultdict(list)
    for final_id in final_ids:
        match = _FINAL_ID_RE.fullmatch(final_id)
        if match:
            result[match.group("clip")].append(final_id)
    for ids in result.values():
        ids.sort()
    return dict(result)


def _orientation(trial_id: str, seed: int) -> dict[str, Any]:
    digest = stable_hash(f"material-orientation-v1:{seed}:{trial_id}")
    positive_slot = "A" if int(digest, 16) % 2 == 0 else "B"
    negative_slot = "B" if positive_slot == "A" else "A"
    return {
        "randomization_seed": int(seed),
        "algorithm": "SHA256 parity of material-orientation-v1:<seed>:<trial_id>",
        "orientation_sha256": digest,
        "positive_slot": positive_slot,
        "negative_slot": negative_slot,
        "correct_choice": positive_slot,
    }


def build_reference_manifest(
    *,
    phase2: Phase2Inventory,
    finals: FinalInventory,
    metadata: Mapping[str, Mapping[str, str]],
    timings: Mapping[str, Mapping[str, Any]],
    loudness: LoudnessCollection,
    source_provenance: Mapping[str, Any],
    protocol_sha256: str,
    orientation_seed: int = 20260717,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Build an insufficiency report and, only on success, a frozen manifest."""
    material_ids_by_clip = _final_ids_by_clip(finals.material_final_ids)
    all_reference_clips = sorted(set(phase2.clips), key=_natural_key)
    subject_failures: dict[str, list[str]] = {}
    negative_match_funnels: dict[str, dict[str, int]] = {}
    exclusion_counts: Counter[str] = Counter()
    subject_matches: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)

    def fail(subject_key: str, reason: str) -> None:
        subject_failures.setdefault(subject_key, []).append(reason)
        exclusion_counts[reason] += 1

    for clip in all_reference_clips:
        row = metadata.get(clip)
        for j in SUBJECT_INDICES:
            subject_key = f"{clip}:j{j}"
            candidate_id = f"{clip}__p1cfg1_ind{j}"
            if row is None:
                fail(subject_key, "MISSING_METADATA_ROW")
                continue
            material = str(row.get("ucs_category", ""))
            if material not in ADMISSIBLE_MATERIALS:
                fail(subject_key, "UCS_MATERIAL_NOT_ADMISSIBLE")
                continue
            missing_scene = [field for field in (
                "audioset_category", "source_type", "discrete_vs_rest"
            ) if not str(row.get(field, "")).strip()]
            if missing_scene:
                for field in missing_scene:
                    fail(subject_key, f"MISSING_SCENE_FIELD:{field}")
                continue
            timing = timings.get(clip)
            if timing is None:
                fail(subject_key, "MISSING_DETECTOR_TIMING")
                continue
            loud_record = loudness.records.get(clip)
            if loud_record is None:
                reason = loudness.failures.get(clip, "SOURCE_LOUDNESS_UNAVAILABLE")
                fail(subject_key, reason.split(":", 1)[0])
                continue
            class_label = finals.class_labels.get(candidate_id)
            if class_label is None:
                fail(subject_key, "MISSING_CANDIDATE_CLASS_FINAL")
                continue
            if class_label == ABSTAIN_LABEL:
                fail(subject_key, "CANDIDATE_CLASS_ABSTAINED")
                continue
            if candidate_id not in finals.material_final_ids:
                fail(subject_key, "MISSING_CANDIDATE_MATERIAL_FINAL")
                continue

            positive_options = [
                final_id for final_id in material_ids_by_clip.get(clip, ())
                if final_id != candidate_id
            ]
            if not positive_options:
                fail(subject_key, "NO_DISJOINT_POSITIVE_REFERENCE")
                continue
            positive_id = min(
                positive_options,
                key=lambda ref: stable_hash(f"material-positive-v1:{candidate_id}:{ref}"),
            )

            candidates: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
            funnel: dict[str, set[str]] = {
                "different_admissible_material_videos": set(),
                "exact_scene_match_videos": set(),
                "negative_covariates_available_videos": set(),
                "within_easy_timing_loudness_band_videos": set(),
                "same_class_reference_ids": set(),
            }
            candidate_timing = float(timing["timing_s"])
            candidate_loudness = float(loud_record["rms_dbfs"])
            for negative_clip in all_reference_clips:
                if negative_clip == clip:
                    continue
                neg_meta = metadata.get(negative_clip)
                if neg_meta is None:
                    continue
                neg_material = str(neg_meta.get("ucs_category", ""))
                if neg_material not in ADMISSIBLE_MATERIALS or neg_material == material:
                    continue
                funnel["different_admissible_material_videos"].add(negative_clip)
                if any(str(neg_meta.get(field, "")) != str(row.get(field, "")) for field in (
                    "audioset_category", "source_type", "discrete_vs_rest"
                )):
                    continue
                funnel["exact_scene_match_videos"].add(negative_clip)
                neg_timing = timings.get(negative_clip)
                neg_loudness = loudness.records.get(negative_clip)
                if neg_timing is None or neg_loudness is None:
                    continue
                funnel["negative_covariates_available_videos"].add(negative_clip)
                timing_distance = abs(candidate_timing - float(neg_timing["timing_s"]))
                loudness_distance = abs(candidate_loudness - float(neg_loudness["rms_dbfs"]))
                band_result = _difficulty(timing_distance, loudness_distance)
                if band_result is None:
                    continue
                funnel["within_easy_timing_loudness_band_videos"].add(negative_clip)
                band_rank, band = band_result
                for negative_id in material_ids_by_clip.get(negative_clip, ()):
                    if finals.class_labels.get(negative_id) != class_label:
                        continue
                    funnel["same_class_reference_ids"].add(negative_id)
                    tie_hash = stable_hash(
                        f"material-negative-v1:{candidate_id}:{negative_id}"
                    )
                    selection_key = (band_rank, float(band["normalized_distance"]), tie_hash)
                    candidates.append((selection_key, {
                        "negative_id": negative_id,
                        "negative_clip": negative_clip,
                        "negative_material": neg_material,
                        "negative_timing_s": float(neg_timing["timing_s"]),
                        "negative_timing_source": str(neg_timing.get("source", "")),
                        "negative_loudness_dbfs": float(neg_loudness["rms_dbfs"]),
                        "negative_metadata": {
                            field: str(neg_meta.get(field, "")) for field in (
                                "audioset_category", "source_type", "discrete_vs_rest"
                            )
                        },
                        "timing_distance_s": timing_distance,
                        "loudness_distance_db": loudness_distance,
                        "band": band,
                        "tie_break_sha256": tie_hash,
                    }))
            negative_match_funnels[subject_key] = {
                key: len(value) for key, value in funnel.items()
            }
            if not candidates:
                if not funnel["different_admissible_material_videos"]:
                    reason = "NO_DIFFERENT_ADMISSIBLE_MATERIAL_VIDEO"
                elif not funnel["exact_scene_match_videos"]:
                    reason = "NO_EXACT_SCENE_METADATA_MATCH"
                elif not funnel["negative_covariates_available_videos"]:
                    reason = "NO_NEGATIVE_WITH_TIMING_AND_LOUDNESS"
                elif not funnel["within_easy_timing_loudness_band_videos"]:
                    reason = "NO_TIMING_LOUDNESS_BAND_MATCH"
                elif not funnel["same_class_reference_ids"]:
                    reason = "NO_SAME_CLASS_NEGATIVE_FINAL"
                else:  # defensive: a non-empty same-class set must make a candidate
                    reason = "NEGATIVE_MATCH_INTERNAL_INCONSISTENCY"
                fail(subject_key, reason)
                continue
            candidates.sort(key=lambda item: item[0])
            negative = candidates[0][1]
            subject_matches[clip][j] = {
                "candidate_id": candidate_id,
                "class_label": class_label,
                "positive_id": positive_id,
                "candidate_material": material,
                "candidate_timing_s": candidate_timing,
                "candidate_timing_source": str(timing.get("source", "")),
                "candidate_loudness_dbfs": candidate_loudness,
                "candidate_metadata": {
                    field: str(row.get(field, "")) for field in (
                        "audioset_category", "source_type", "discrete_vs_rest"
                    )
                },
                **negative,
            }

    complete_videos = [
        clip for clip in all_reference_clips
        if set(subject_matches.get(clip, ())) == set(SUBJECT_INDICES)
    ]
    partial_videos = [
        clip for clip in all_reference_clips
        if subject_matches.get(clip) and clip not in complete_videos
    ]
    trial_rows: list[dict[str, Any]] = []
    per_progress_videos: dict[str, set[str]] = {
        f"{s:.2f}": set() for s in PROGRESS_POINTS
    }
    for clip in complete_videos:
        for j in SUBJECT_INDICES:
            match = subject_matches[clip][j]
            for s in PROGRESS_POINTS:
                progress_key = f"{s:.2f}"
                trial_id = f"legacy-material:{clip}:j{j}:s{progress_key}"
                per_progress_videos[progress_key].add(clip)
                orientation = _orientation(trial_id, orientation_seed)
                trial_rows.append({
                    "trial_id": trial_id,
                    "candidate": {
                        "candidate_id": f"{match['candidate_id']}__s{progress_key}",
                        "video_id": clip,
                        "subject_index": j,
                        "progress": float(s),
                        "retained_final_generation_id": match["candidate_id"],
                        "coarse_class": match["class_label"],
                    },
                    "positive": {
                        "reference_id": f"positive:{match['positive_id']}",
                        "video_id": clip,
                        "generation_id": match["positive_id"],
                    },
                    "negative": {
                        "reference_id": f"negative:{match['negative_id']}",
                        "video_id": match["negative_clip"],
                        "generation_id": match["negative_id"],
                        "coarse_class": match["class_label"],
                    },
                    "matching": {
                        "candidate_material": match["candidate_material"],
                        "negative_material": match["negative_material"],
                        "material_difference_source": (
                            "exact distinct FoleyBench ucs_category values from the "
                            "predeclared unambiguous material set"
                        ),
                        "candidate_scene_metadata": match["candidate_metadata"],
                        "negative_scene_metadata": match["negative_metadata"],
                        "candidate_timing_s": match["candidate_timing_s"],
                        "negative_timing_s": match["negative_timing_s"],
                        "candidate_timing_source": match["candidate_timing_source"],
                        "negative_timing_source": match["negative_timing_source"],
                        "timing_distance_s": match["timing_distance_s"],
                        "candidate_loudness_dbfs": match["candidate_loudness_dbfs"],
                        "negative_loudness_dbfs": match["negative_loudness_dbfs"],
                        "loudness_distance_db": match["loudness_distance_db"],
                        "difficulty": match["band"]["difficulty"],
                        "timing_seconds_lte": match["band"]["timing_seconds_lte"],
                        "loudness_db_lte": match["band"]["loudness_db_lte"],
                        "normalized_distance": match["band"]["normalized_distance"],
                        "selection_tiebreak_sha256": match["tie_break_sha256"],
                    },
                    "ab_orientation": orientation,
                })

    requirements = {
        "exact_legacy_inventory": (
            len(phase2.clips) == EXPECTED_LEGACY_VIDEOS
            and sum(len(cells) for cells in phase2.cells_by_clip.values())
            == EXPECTED_LEGACY_CELLS
            and not phase2.errors
        ),
        "minimum_valid_candidate_videos": len(complete_videos) >= MIN_VALID_VIDEOS,
        "minimum_valid_videos_every_progress": all(
            len(videos) >= MIN_VALID_VIDEOS for videos in per_progress_videos.values()
        ),
        "minimum_valid_cells": len(trial_rows) >= MIN_VALID_CELLS,
        "retained_final_inventory_valid": not finals.errors,
        "source_metadata_and_timing_inputs_valid": not (
            source_provenance.get("metadata_input_errors")
            or source_provenance.get("timing_input_errors")
        ),
    }
    sufficient = all(requirements.values())
    source_summary = dict(source_provenance)
    source_summary["protocol_sha256"] = protocol_sha256
    source_summary["measurements_sha256"] = finals.measurement_sha256
    source_summary["phase2_journals_aggregate_sha256"] = phase2.aggregate_sha256

    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA_VERSION,
        "status": "READY_TO_FREEZE" if sufficient else "INCOMPLETE_ARTIFACTS",
        "outcome_blind": True,
        "candidate_margins_inspected": False,
        "candidate_embeddings_inspected": False,
        "caption_inference_used": False,
        "embedding_defined_truth_used": False,
        "rules": {
            "admissible_materials": list(ADMISSIBLE_MATERIALS),
            "matching_bands": list(MATCHING_BANDS),
            "minimum_valid_candidate_videos": MIN_VALID_VIDEOS,
            "minimum_valid_videos_per_progress": MIN_VALID_VIDEOS,
            "minimum_valid_cells": MIN_VALID_CELLS,
            "complete_video_rule": "all four subjects have a strict match",
        },
        "coverage": {
            "legacy_journal_videos": len(phase2.clips),
            "legacy_cells_inventoried": sum(
                len(cells) for cells in phase2.cells_by_clip.values()
            ),
            "subjects_with_strict_match": sum(len(v) for v in subject_matches.values()),
            "strict_subject_videos_before_complete_video_rule": sum(
                bool(matches) for matches in subject_matches.values()
            ),
            "strict_subject_cells_before_complete_video_rule": (
                sum(len(v) for v in subject_matches.values()) * len(PROGRESS_POINTS)
            ),
            "complete_candidate_videos": len(complete_videos),
            "partial_candidate_videos_excluded": len(partial_videos),
            "valid_cells": len(trial_rows),
            "valid_videos_by_progress": {
                key: len(value) for key, value in per_progress_videos.items()
            },
        },
        "complete_candidate_video_ids": complete_videos,
        "partial_candidate_video_ids": partial_videos,
        "requirements": requirements,
        "exclusion_reason_counts": dict(sorted(exclusion_counts.items())),
        "subject_exclusions": {
            key: sorted(set(reasons)) for key, reasons in sorted(subject_failures.items())
        },
        "negative_match_funnels": {
            key: value for key, value in sorted(negative_match_funnels.items())
        },
        "input_errors": {
            "phase2": list(phase2.errors),
            "retained_finals": list(finals.errors),
        },
        "loudness_failures": dict(loudness.failures),
        "source_provenance": source_summary,
    }
    if not sufficient:
        report["blocker_summary"] = [
            name for name, passed in requirements.items() if not passed
        ]
        return report, None

    manifest = {
        "schema": SCHEMA_VERSION,
        "frozen": True,
        "exploratory_scope": "legacy clip-level continuity; not event-centered Material v2",
        "outcome_blind": True,
        "candidate_margins_inspected": False,
        "candidate_embeddings_inspected": False,
        "protocol_sha256": protocol_sha256,
        "source_provenance": source_summary,
        "rules": report["rules"],
        "coverage": report["coverage"],
        "trials": trial_rows,
    }
    validate_reference_manifest(manifest)
    return report, manifest


def _walk_keys(value: Any, path: str = "$") -> Iterable[tuple[str, str]]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            yield str(key), child_path
            yield from _walk_keys(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk_keys(child, f"{path}[{index}]")


def validate_reference_manifest(manifest: Mapping[str, Any]) -> None:
    """Schema-like and semantic validation, including leakage/disjointness."""
    if manifest.get("schema") != SCHEMA_VERSION:
        raise FeasibilityInputError("wrong Material reference manifest schema")
    if manifest.get("frozen") is not True or manifest.get("outcome_blind") is not True:
        raise FeasibilityInputError("reference manifest must be frozen and outcome-blind")
    for key, path in _walk_keys(manifest):
        if key in FORBIDDEN_OUTCOME_KEYS:
            raise FeasibilityInputError(f"outcome/embedding field forbidden at {path}")
    trials = manifest.get("trials")
    if not isinstance(trials, list) or not trials:
        raise FeasibilityInputError("reference manifest must contain non-empty trials")
    ids: set[str] = set()
    videos_by_progress: dict[str, set[str]] = defaultdict(set)
    for index, trial in enumerate(trials):
        if not isinstance(trial, Mapping):
            raise FeasibilityInputError(f"trials[{index}] is not an object")
        for field in ("trial_id", "candidate", "positive", "negative", "matching", "ab_orientation"):
            if field not in trial:
                raise FeasibilityInputError(f"trials[{index}] missing {field}")
        trial_id = str(trial["trial_id"])
        if trial_id in ids:
            raise FeasibilityInputError(f"duplicate trial_id {trial_id}")
        ids.add(trial_id)
        candidate = trial["candidate"]
        positive = trial["positive"]
        negative = trial["negative"]
        matching = trial["matching"]
        orientation = trial["ab_orientation"]
        if not all(isinstance(x, Mapping) for x in (
            candidate, positive, negative, matching, orientation
        )):
            raise FeasibilityInputError(f"trials[{index}] nested fields must be objects")
        if candidate.get("video_id") != positive.get("video_id"):
            raise FeasibilityInputError(f"{trial_id}: positive is not same-video")
        if candidate.get("video_id") == negative.get("video_id"):
            raise FeasibilityInputError(f"{trial_id}: negative is not different-video")
        if candidate.get("retained_final_generation_id") == positive.get("generation_id"):
            raise FeasibilityInputError(f"{trial_id}: candidate contributes to positive")
        if candidate.get("coarse_class") != negative.get("coarse_class"):
            raise FeasibilityInputError(f"{trial_id}: negative Class mismatch")
        candidate_material = matching.get("candidate_material")
        negative_material = matching.get("negative_material")
        if (candidate_material not in ADMISSIBLE_MATERIALS
                or negative_material not in ADMISSIBLE_MATERIALS
                or candidate_material == negative_material):
            raise FeasibilityInputError(f"{trial_id}: invalid independent material difference")
        if matching.get("candidate_scene_metadata") != matching.get("negative_scene_metadata"):
            raise FeasibilityInputError(f"{trial_id}: scene metadata mismatch")
        timing_distance = abs(
            float(matching["candidate_timing_s"]) - float(matching["negative_timing_s"])
        )
        loudness_distance = abs(
            float(matching["candidate_loudness_dbfs"])
            - float(matching["negative_loudness_dbfs"])
        )
        if not math.isclose(timing_distance, float(matching["timing_distance_s"]), abs_tol=1e-12):
            raise FeasibilityInputError(f"{trial_id}: timing distance mismatch")
        if not math.isclose(loudness_distance, float(matching["loudness_distance_db"]), abs_tol=1e-12):
            raise FeasibilityInputError(f"{trial_id}: loudness distance mismatch")
        band = _difficulty(timing_distance, loudness_distance)
        if band is None or band[1]["difficulty"] != matching.get("difficulty"):
            raise FeasibilityInputError(f"{trial_id}: matching band mismatch")
        if orientation.get("correct_choice") != orientation.get("positive_slot"):
            raise FeasibilityInputError(f"{trial_id}: A/B correct choice is not positive slot")
        if {orientation.get("positive_slot"), orientation.get("negative_slot")} != {"A", "B"}:
            raise FeasibilityInputError(f"{trial_id}: invalid A/B orientation")
        expected_orientation = _orientation(trial_id, int(orientation["randomization_seed"]))
        if dict(orientation) != expected_orientation:
            raise FeasibilityInputError(f"{trial_id}: A/B orientation hash/assignment mismatch")
        progress = float(candidate["progress"])
        if progress not in PROGRESS_POINTS:
            raise FeasibilityInputError(f"{trial_id}: unregistered progress {progress}")
        videos_by_progress[f"{progress:.2f}"].add(str(candidate["video_id"]))

    coverage = manifest.get("coverage")
    if not isinstance(coverage, Mapping):
        raise FeasibilityInputError("manifest coverage must be an object")
    if len(trials) != int(coverage.get("valid_cells", -1)):
        raise FeasibilityInputError("coverage valid_cells disagrees with trials")
    if len(trials) < MIN_VALID_CELLS:
        raise FeasibilityInputError("manifest does not meet minimum valid cells")
    if any(len(videos_by_progress.get(f"{s:.2f}", set())) < MIN_VALID_VIDEOS
           for s in PROGRESS_POINTS):
        raise FeasibilityInputError("manifest does not meet per-progress video coverage")


def immutable_write(path: Path, payload: Any) -> str:
    """Atomically create *path*, permitting only a byte-identical rerun."""
    target = Path(path)
    data = _canonical_json_bytes(payload)
    digest = hashlib.sha256(data).hexdigest()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.read_bytes() != data:
            raise FileExistsError(f"refusing to overwrite non-identical immutable output: {target}")
        return digest
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, target)
    except FileExistsError:
        if target.read_bytes() != data:
            raise FileExistsError(
                f"concurrent non-identical immutable output exists: {target}"
            )
    finally:
        temporary.unlink(missing_ok=True)
    return digest


def _immutable_write_text(path: Path, text: str) -> None:
    """Atomically create a small text sidecar, allowing identical reruns."""
    target = Path(path)
    data = text.encode("utf-8")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.read_bytes() != data:
            raise FileExistsError(f"refusing to overwrite non-identical sidecar: {target}")
        return
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, target)
    except FileExistsError:
        if target.read_bytes() != data:
            raise FileExistsError(f"concurrent non-identical sidecar exists: {target}")
    finally:
        temporary.unlink(missing_ok=True)


def write_feasibility_outputs(
    out_dir: Path,
    report: Mapping[str, Any],
    manifest: Mapping[str, Any] | None,
    loudness: LoudnessCollection,
) -> dict[str, str]:
    """Write immutable loudness + either insufficiency or frozen manifest."""
    root = Path(out_dir)
    manifest_path = root / "MATERIAL_REFERENCE_MANIFEST.json"
    insufficiency_path = root / "MATERIAL_REFERENCE_INSUFFICIENCY.json"
    if manifest is None and manifest_path.exists():
        raise FileExistsError(
            f"stale frozen manifest exists while current run is insufficient: {manifest_path}"
        )
    if manifest is not None and insufficiency_path.exists():
        raise FileExistsError(
            f"stale insufficiency report exists while current run passes: {insufficiency_path}"
        )
    loudness_payload = {
        "schema": LOUDNESS_SCHEMA_VERSION,
        "outcome_independent": True,
        "candidate_audio_used": False,
        "ffmpeg": loudness.ffmpeg,
        "records": loudness.records,
        "failures": loudness.failures,
    }
    outputs = {
        "loudness": str(root / "SOURCE_AUDIO_LOUDNESS.json"),
    }
    immutable_write(Path(outputs["loudness"]), loudness_payload)
    if manifest is None:
        outputs["insufficiency"] = str(insufficiency_path)
        immutable_write(insufficiency_path, report)
    else:
        outputs["feasibility"] = str(root / "MATERIAL_REFERENCE_FEASIBILITY.json")
        outputs["manifest"] = str(manifest_path)
        immutable_write(Path(outputs["feasibility"]), report)
        digest = immutable_write(manifest_path, manifest)
        checksum_path = Path(str(manifest_path) + ".sha256")
        checksum_data = f"{digest}  {manifest_path.name}\n"
        _immutable_write_text(checksum_path, checksum_data)
        outputs["manifest_sha256"] = digest
    return outputs
