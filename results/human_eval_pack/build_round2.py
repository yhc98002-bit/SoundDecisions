#!/usr/bin/env python3
"""Build an immutable, offline Round-2 Presence package from an event catalog."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Mapping

from jsonschema import Draft202012Validator, FormatChecker


HERE = Path(__file__).resolve().parent
SOURCE_DIR = HERE / "release_src"
CATALOG_SCHEMA_PATH = HERE / "event_catalog.schema.json"
MANIFEST_SCHEMA_PATH = SOURCE_DIR / "round2_manifest.schema.json"
RATINGS_SCHEMA_PATH = SOURCE_DIR / "round2_ratings.schema.json"
SUMMARY_SCHEMA_PATH = SOURCE_DIR / "round2_summary.schema.json"
HTML_TEMPLATE_PATH = SOURCE_DIR / "rate_round2.html"
INSTRUCTIONS_PATH = SOURCE_DIR / "INSTRUCTIONS_ROUND2.md"
RUBRIC = (
    "For one specified visible event, determine whether a corresponding audio event "
    "occurs near its anchor: present, absent, or uncertain. Salient unrelated background "
    "audio does not count. The unit is the event."
)


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: cannot read JSON: {exc}") from exc


def _canonical_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalized_rater_hash(rater_id: str) -> str:
    return hashlib.sha256(rater_id.strip().lower().encode("utf-8")).hexdigest()


def _validate(payload: Any, schema_path: Path, context: str) -> None:
    schema = _load_json(schema_path)
    Draft202012Validator.check_schema(schema)
    errors = sorted(
        Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(payload),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if errors:
        error = errors[0]
        location = "$" + "".join(
            f"[{part}]" if isinstance(part, int) else f".{part}"
            for part in error.absolute_path
        )
        raise ValueError(f"{context}: schema validation failed at {location}: {error.message}")


def _catalog_interval(value: Any, duration_s: float, context: str) -> dict[str, float]:
    if not isinstance(value, Mapping) or set(value) != {"start_s", "end_s"}:
        raise ValueError(f"{context}: anchor must contain exactly start_s and end_s")
    start, end = value["start_s"], value["end_s"]
    numeric = lambda bound: (
        isinstance(bound, (int, float))
        and not isinstance(bound, bool)
        and math.isfinite(float(bound))
    )
    if not numeric(start) or not numeric(end):
        raise ValueError(f"{context}: anchor bounds must be finite numbers")
    start_f, end_f = float(start), float(end)
    if not 0.0 <= start_f <= end_f <= duration_s:
        raise ValueError(f"{context}: anchor is outside the video bounds")
    return {"start_s": start_f, "end_s": end_f}


def _catalog_event(
    event: Any,
    *,
    blind_id: str,
    suffix: str,
    source: str,
    duration_s: float,
) -> dict[str, Any]:
    context = f"{blind_id}-{suffix}"
    if not isinstance(event, Mapping):
        raise ValueError(f"{context}: event must be an object")
    expected_id = f"{blind_id}-{suffix}"
    if event.get("event_id") != expected_id:
        raise ValueError(f"{context}: event ID prefix/suffix does not match its item")
    if event.get("event_source") != source:
        raise ValueError(f"{context}: event suffix does not match event_source")
    description = event.get("description")
    if (
        not isinstance(description, str)
        or not description.strip()
        or description != description.strip()
    ):
        raise ValueError(f"{context}: description must be nonempty and trimmed")
    anchor = _catalog_interval(event.get("anchor"), duration_s, context)
    return {
        "event_id": expected_id,
        "description": description,
        "anchor": anchor,
        "event_source": source,
    }


def validate_catalog_semantics(catalog: Mapping[str, Any]) -> None:
    """Validate cross-field contracts that JSON Schema cannot express."""

    if not isinstance(catalog, Mapping):
        raise ValueError("event catalog must be an object")
    items = catalog.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("event catalog must contain items")

    recomputed = {
        "items_total": len(items),
        "anchor_eligible": 0,
        "anchor_unresolved": 0,
        "pair_eligible": 0,
        "pair_rejected": 0,
        "events_eligible": 0,
    }
    seen_blind_ids: set[str] = set()
    seen_event_ids: set[str] = set()
    for item in items:
        if not isinstance(item, Mapping):
            raise ValueError("catalog item must be an object")
        blind_id = item.get("blind_id")
        if not isinstance(blind_id, str) or blind_id in seen_blind_ids:
            raise ValueError(f"duplicate or invalid blind_id: {blind_id}")
        seen_blind_ids.add(blind_id)
        duration = item.get("duration_s")
        if (
            not isinstance(duration, (int, float))
            or isinstance(duration, bool)
            or not math.isfinite(float(duration))
            or float(duration) <= 0
        ):
            raise ValueError(f"{blind_id}: duration_s must be finite and positive")
        duration_s = float(duration)
        expected_events: list[dict[str, Any]] = []

        anchor = item.get("anchor_curation")
        if anchor is not None:
            status = anchor.get("status") if isinstance(anchor, Mapping) else None
            description = anchor.get("description") if isinstance(anchor, Mapping) else None
            if not isinstance(description, str) or description != description.strip():
                raise ValueError(f"{blind_id}: anchor description must be trimmed")
            if status == "eligible":
                if not description:
                    raise ValueError(f"{blind_id}: eligible anchor description is empty")
                event = _catalog_event(
                    {
                        "event_id": f"{blind_id}-A1",
                        "description": description,
                        "anchor": anchor.get("anchor"),
                        "event_source": "anchor_curation",
                    },
                    blind_id=blind_id,
                    suffix="A1",
                    source="anchor_curation",
                    duration_s=duration_s,
                )
                expected_events.append(event)
                recomputed["anchor_eligible"] += 1
            elif status == "unresolved":
                if anchor.get("anchor") is not None:
                    raise ValueError(f"{blind_id}: unresolved anchor must have a null anchor")
                recomputed["anchor_unresolved"] += 1
            else:
                raise ValueError(f"{blind_id}: invalid anchor curation status")

        pair = item.get("two_event_curation")
        if pair is not None:
            status = pair.get("status") if isinstance(pair, Mapping) else None
            pair_events = pair.get("events") if isinstance(pair, Mapping) else None
            if status == "eligible":
                if not isinstance(pair_events, list) or len(pair_events) != 2:
                    raise ValueError(f"{blind_id}: eligible pair must contain P1 and P2")
                checked_pair = [
                    _catalog_event(
                        pair_events[index - 1],
                        blind_id=blind_id,
                        suffix=f"P{index}",
                        source="two_event_curation",
                        duration_s=duration_s,
                    )
                    for index in (1, 2)
                ]
                if checked_pair[0]["anchor"]["start_s"] > checked_pair[1]["anchor"]["start_s"]:
                    raise ValueError(f"{blind_id}: P1 must not start after P2")
                expected_events.extend(checked_pair)
                recomputed["pair_eligible"] += 1
            elif status == "rejected":
                if pair_events != []:
                    raise ValueError(f"{blind_id}: rejected pair must contain no events")
                recomputed["pair_rejected"] += 1
            else:
                raise ValueError(f"{blind_id}: invalid two-event curation status")

        if anchor is None and pair is None:
            raise ValueError(f"{blind_id}: item has no curation subrecord")
        flattened = item.get("events")
        if not isinstance(flattened, list):
            raise ValueError(f"{blind_id}: flattened events must be an array")
        checked_flattened: list[dict[str, Any]] = []
        for event in flattened:
            event_id = event.get("event_id") if isinstance(event, Mapping) else None
            if not isinstance(event_id, str) or not event_id.startswith(f"{blind_id}-"):
                raise ValueError(f"{blind_id}: event ID prefix/suffix does not match its item")
            suffix = event_id.removeprefix(f"{blind_id}-")
            if suffix == "A1":
                source = "anchor_curation"
            elif suffix in {"P1", "P2"}:
                source = "two_event_curation"
            else:
                raise ValueError(f"{blind_id}: event ID prefix/suffix does not match its item")
            checked_flattened.append(
                _catalog_event(
                    event,
                    blind_id=blind_id,
                    suffix=suffix,
                    source=source,
                    duration_s=duration_s,
                )
            )
        if checked_flattened != expected_events:
            raise ValueError(f"{blind_id}: flattened events do not equal the eligible subrecord union")
        for event in expected_events:
            event_id = event["event_id"]
            if event_id in seen_event_ids:
                raise ValueError(f"duplicate event_id in catalog: {event_id}")
            seen_event_ids.add(event_id)
        recomputed["events_eligible"] += len(expected_events)

    if catalog.get("counts") != recomputed:
        raise ValueError("catalog counts do not match semantic item totals")


def build_manifest(catalog: Mapping[str, Any], catalog_sha256: str) -> dict[str, Any]:
    validate_catalog_semantics(catalog)
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    for catalog_item in catalog["items"]:
        for event in catalog_item["events"]:
            event_id = event["event_id"]
            if event_id in seen:
                raise ValueError(f"duplicate event_id in catalog: {event_id}")
            seen.add(event_id)
            items.append(
                {
                    "event_id": event_id,
                    "blind_id": catalog_item["blind_id"],
                    "media_path": catalog_item["media_path"],
                    "fps": catalog_item["fps"],
                    "duration_s": catalog_item["duration_s"],
                    "event_description": event["description"],
                    "anchor": event["anchor"],
                }
            )
    if not items:
        raise ValueError("event catalog contains no eligible events")
    manifest = {
        "schema_version": "sounddecisions-human-presence-items-v1-1.0",
        "instrument_version": "human-eval-round2-presence-1.0",
        "manifest_id": f"{catalog['catalog_id']}-round2-presence-v1",
        "status": "RATING_AUTHORIZED_SINGLE_RATER",
        "source_catalog_id": catalog["catalog_id"],
        "source_catalog_sha256": catalog_sha256,
        "curator_rater_id_sha256": _normalized_rater_hash(catalog["curator_id"]),
        "counts": {
            "events": len(items),
            "unique_videos": len({item["blind_id"] for item in items}),
        },
        "rubric": RUBRIC,
        "items": items,
    }
    _validate(manifest, MANIFEST_SCHEMA_PATH, "generated Round-2 manifest")
    return manifest


def _write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _copy_verified(source: Path, target: Path) -> None:
    """Copy one blinded video without sharing an inode and verify stable bytes."""

    target.parent.mkdir(parents=True, exist_ok=True)
    expected = _sha256(source)
    partial = target.with_name(f".{target.name}.partial")
    shutil.copyfile(source, partial)
    if _sha256(source) != expected or _sha256(partial) != expected:
        partial.unlink(missing_ok=True)
        raise OSError(f"media changed or failed hash verification while copying: {source}")
    os.replace(partial, target)


def _probe_media_streams(source: Path) -> tuple[int, int]:
    """Return video/audio stream counts, failing closed when ffprobe cannot inspect input."""

    executable = shutil.which("ffprobe")
    if executable is None:
        raise RuntimeError("ffprobe is required to verify Round-2 audio-bearing media")
    completed = subprocess.run(
        [
            executable,
            "-v", "error",
            "-show_entries", "stream=codec_type",
            "-of", "json",
            str(source),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise ValueError(f"ffprobe could not inspect {source}: {completed.stderr.strip()}")
    try:
        streams = json.loads(completed.stdout).get("streams", [])
    except json.JSONDecodeError as exc:
        raise ValueError(f"ffprobe returned invalid JSON for {source}") from exc
    video_count = sum(stream.get("codec_type") == "video" for stream in streams)
    audio_count = sum(stream.get("codec_type") == "audio" for stream in streams)
    return video_count, audio_count


def build_package(catalog_path: Path, output_dir: Path, media_root: Path) -> dict[str, Any]:
    """Build a new package directory without overwriting any prior output."""

    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output_dir}")
    catalog = _load_json(catalog_path)
    _validate(catalog, CATALOG_SCHEMA_PATH, str(catalog_path))
    catalog_sha256 = _sha256(catalog_path)
    manifest = build_manifest(catalog, catalog_sha256)
    manifest_bytes = _canonical_bytes(manifest)
    manifest_sha256 = _sha256_bytes(manifest_bytes)

    template = HTML_TEMPLATE_PATH.read_text(encoding="utf-8")
    for marker in ("__ROUND2_MANIFEST_JSON__", "__ROUND2_MANIFEST_SHA256__"):
        if template.count(marker) != 1:
            raise ValueError(f"HTML template must contain exactly one {marker} marker")
    inline_manifest = json.dumps(manifest, sort_keys=True).replace("<", "\\u003c")
    html = template.replace("__ROUND2_MANIFEST_JSON__", inline_manifest).replace(
        "__ROUND2_MANIFEST_SHA256__", manifest_sha256
    )

    output_parent = output_dir.parent
    output_parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_parent))
    try:
        _write(temporary / "round2_manifest.json", manifest_bytes)
        _write(temporary / "rate.html", html.encode("utf-8"))
        for source, destination in (
            (RATINGS_SCHEMA_PATH, "ratings.schema.json"),
            (SUMMARY_SCHEMA_PATH, "summary.schema.json"),
            (INSTRUCTIONS_PATH, "INSTRUCTIONS.md"),
        ):
            _write(temporary / destination, source.read_bytes())

        copied_media: set[str] = set()
        for item in manifest["items"]:
            relative = item["media_path"]
            if relative in copied_media:
                continue
            source = media_root / relative
            if not source.is_file():
                raise FileNotFoundError(f"required blinded video is missing: {source}")
            video_count, audio_count = _probe_media_streams(source)
            if video_count < 1 or audio_count < 1:
                raise ValueError(
                    f"Round-2 source must contain video and audio streams: {source} "
                    f"(video={video_count}, audio={audio_count})"
                )
            _copy_verified(source, temporary / relative)
            copied_media.add(relative)

        sums: dict[str, str] = {}
        for path in sorted(temporary.rglob("*")):
            if path.is_file() and path.name != "SHA256SUMS.json":
                sums[path.relative_to(temporary).as_posix()] = _sha256(path)
        _write(temporary / "SHA256SUMS.json", _canonical_bytes(sums))
        temporary.rename(output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-catalog", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--media-root",
        type=Path,
        required=True,
        help=(
            "audio-bearing blinded pack root containing the catalog's relative media/ "
            "paths; do not use the silent Round-1 release"
        ),
    )
    args = parser.parse_args()
    manifest = build_package(args.event_catalog, args.output_dir, args.media_root)
    print(json.dumps(manifest["counts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
