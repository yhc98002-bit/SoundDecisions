#!/usr/bin/env python3
"""Build an immutable, offline Round-2 Presence package from an event catalog."""

from __future__ import annotations

import argparse
import hashlib
import json
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


def build_manifest(catalog: Mapping[str, Any], catalog_sha256: str) -> dict[str, Any]:
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
