#!/usr/bin/env python3
"""Build the blinded, pre-freeze human-evaluation candidate package.

This reads only source-video metadata and the two concrete candidate manifests
named by the Goal-1 cohort plan. It does not open quarantined audio, run a
measurer, select a development cohort, or inspect a model condition.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import hmac
import json
import os
import re
import shutil
import subprocess
import tempfile
from fractions import Fraction
from pathlib import Path


INSTRUMENT_VERSION = "human-eval-pack-1.0"
MANIFEST_ID = "axis-spec-v2-candidate-curation-2026-07"
PRESENCE_RUBRIC = (
    "For one specified visible event, determine whether a corresponding audio "
    "event occurs near its anchor: present, absent, or uncertain. Salient "
    "unrelated background audio does not count. The unit is the event."
)
ANCHOR_RUBRIC = (
    "Mark the closed visual-event interval [start, end]. Use Too uncertain when "
    "a defensible interval cannot be marked."
)
TWO_EVENT_RUBRIC = (
    "Confirm only when two separable target events exist, then mark one closed "
    "visual interval for each event; otherwise reject the item."
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _probe_video(path: Path) -> tuple[float, float]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate:format=duration",
        "-of",
        "json",
        str(path),
    ]
    proc = subprocess.run(command, check=True, capture_output=True, text=True)
    payload = json.loads(proc.stdout)
    streams = payload.get("streams", [])
    if len(streams) != 1:
        raise RuntimeError(f"expected one video stream in {path}, found {len(streams)}")
    fps = float(Fraction(streams[0]["avg_frame_rate"]))
    duration = float(payload["format"]["duration"])
    if fps <= 0 or duration <= 0:
        raise RuntimeError(f"invalid video metadata for {path}: fps={fps}, duration={duration}")
    return fps, duration


def _load_index(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    index = {row["key"]: row for row in rows}
    if len(index) != len(rows):
        raise RuntimeError(f"duplicate clip keys in {path}")
    return index


def _copy_media(source: Path, destination: Path, source_sha256: str) -> None:
    """Publish an independent, hash-verified media copy atomically."""

    if source.absolute() == destination.absolute():
        raise RuntimeError("blinded media destination must differ from its source path")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination_sha256 = _sha256(destination)
        if destination_sha256 == source_sha256 and not os.path.samefile(source, destination):
            return

    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        shutil.copyfile(source, temporary_path)
        if _sha256(temporary_path) != source_sha256:
            raise RuntimeError("blinded media copy failed source-hash verification")
        temporary_path.chmod(0o644)
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)

    if os.path.samefile(source, destination):
        raise RuntimeError("blinded media copy is not independent from its source")
    if _sha256(destination) != source_sha256:
        raise RuntimeError("published blinded media failed source-hash verification")


def _atomic_write_text(destination: Path, text: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.chmod(0o644)
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def _append_public_preflight_flag(out_dir: Path, failures: list[dict[str, object]]) -> Path:
    flag_path = out_dir / "FLAGS.json"
    if flag_path.is_file():
        existing = json.loads(flag_path.read_text(encoding="utf-8"))
        if isinstance(existing, list):
            flags = existing
        elif isinstance(existing, dict):
            flags = [existing]
        else:
            raise RuntimeError(f"existing flag ledger is not a JSON object or list: {flag_path}")
    else:
        flags = []

    flags.append(
        {
            "id": "HUMAN_EVAL_MEDIA_PREFLIGHT_FAILED",
            "status": "INCOMPLETE_ARTIFACTS",
            "reason": "one or more named calibration-candidate videos failed preflight",
            "evidence": failures,
            "resolution": (
                "restore the exact indexed source MP4s and rerun the builder; "
                "do not substitute clips"
            ),
        }
    )
    _atomic_write_text(flag_path, json.dumps(flags, indent=2) + "\n")
    return flag_path


def _seal_mapping(mapping: dict, destination: Path, key_file: Path) -> dict:
    if not key_file.exists():
        raise RuntimeError(f"blinding/sealing key is missing: {key_file}")
    mode = key_file.stat().st_mode & 0o777
    if mode & 0o077:
        raise RuntimeError(f"sealing key must not be group/world accessible: {key_file}")

    plaintext = (json.dumps(mapping, indent=2, sort_keys=True) + "\n").encode("utf-8")
    plaintext_sha256 = hashlib.sha256(plaintext).hexdigest()
    key_id = hashlib.sha256(key_file.read_bytes()).hexdigest()[:16]
    if destination.is_file():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if (
            existing.get("format") == "sounddecisions-sealed-map-v1"
            and existing.get("key_id") == key_id
            and existing.get("plaintext_sha256") == plaintext_sha256
        ):
            return existing

    command = [
        "openssl",
        "enc",
        "-aes-256-cbc",
        "-a",
        "-A",
        "-salt",
        "-pbkdf2",
        "-iter",
        "200000",
        "-pass",
        f"file:{key_file}",
    ]
    proc = subprocess.run(command, input=plaintext, check=True, capture_output=True)
    envelope = {
        "format": "sounddecisions-sealed-map-v1",
        "cipher": "AES-256-CBC",
        "kdf": "PBKDF2",
        "iterations": 200000,
        "key_id": key_id,
        "plaintext_sha256": plaintext_sha256,
        "ciphertext_base64": proc.stdout.decode("ascii"),
    }
    destination.write_text(json.dumps(envelope, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return envelope


def _load_key(key_file: Path) -> bytes:
    if not key_file.exists():
        raise RuntimeError(
            f"blinding/sealing key is missing: {key_file}; restore the package key "
            "instead of generating new blinded IDs"
        )
    mode = key_file.stat().st_mode & 0o777
    if mode & 0o077:
        raise RuntimeError(f"sealing key must not be group/world accessible: {key_file}")
    return key_file.read_bytes()


def _blind_id(clip_id: str, key: bytes) -> str:
    digest = hmac.new(key, f"human-eval-v2:{clip_id}".encode("ascii"), hashlib.sha256)
    return f"HEV2-{digest.hexdigest()[:12].upper()}"


def _embed_manifest(rate_html: Path, manifest_text: str, manifest_sha256: str) -> None:
    if not rate_html.is_file():
        raise RuntimeError(f"offline instrument is missing: {rate_html}")
    document = rate_html.read_text(encoding="utf-8")
    embedded = manifest_text.replace("</", "<\\/")
    pattern = re.compile(
        r'(<script id="manifest-data" type="application/json" data-sha256=")'
        r'[^"]*(">).*?(</script>)',
        re.DOTALL,
    )
    replacement = rf"\g<1>{manifest_sha256}\g<2>{embedded}\g<3>"
    updated, count = pattern.subn(replacement, document, count=1)
    if count != 1:
        raise RuntimeError(f"manifest embed point is missing or ambiguous in {rate_html}")
    rate_html.write_text(updated, encoding="utf-8")


def build(args: argparse.Namespace) -> dict:
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    index = _load_index(args.clips_index)

    with args.anchor_csv.open(newline="", encoding="utf-8") as handle:
        anchor_rows = list(csv.DictReader(handle))
    two_payload = json.loads(args.two_event_json.read_text(encoding="utf-8"))
    two_ids = [str(value) for value in two_payload["clips"]]
    pilot_payload = json.loads(args.pilot_manifest.read_text(encoding="utf-8"))
    pilot_ids = {str(value) for value in pilot_payload["clips"]}
    if len(anchor_rows) != 30 or len(two_ids) != 60:
        raise RuntimeError(
            f"candidate contract changed: anchors={len(anchor_rows)}, two_event={len(two_ids)}"
        )

    key = _load_key(args.key_file)
    tasks_by_clip: dict[str, list[str]] = {}
    positions_by_clip: dict[str, dict[str, int]] = {}
    for position, row in enumerate(anchor_rows, 1):
        clip_id = str(row["key"])
        tasks_by_clip.setdefault(clip_id, []).append("anchor_presence")
        positions_by_clip.setdefault(clip_id, {})["anchor_presence"] = position
    for position, clip_id in enumerate(two_ids, 1):
        tasks_by_clip.setdefault(clip_id, []).append("two_event")
        positions_by_clip.setdefault(clip_id, {})["two_event"] = position

    prepared = []
    failures: list[dict[str, object]] = []
    for clip_id in tasks_by_clip:
        blind_id = _blind_id(clip_id, key)
        tasks = tasks_by_clip[clip_id]
        metadata = index.get(clip_id)
        if metadata is None:
            failures.append(
                {
                    "blind_id": blind_id,
                    "reason": "index_join_missing",
                    "checksum_verified": False,
                }
            )
            continue
        source = Path(metadata["path"])
        if not source.is_file():
            failures.append(
                {
                    "blind_id": blind_id,
                    "reason": "video_missing",
                    "checksum_verified": False,
                }
            )
            continue
        source_sha256 = _sha256(source)
        if source_sha256 != metadata["sha256"]:
            failures.append(
                {
                    "blind_id": blind_id,
                    "reason": "video_checksum_mismatch",
                    "checksum_verified": False,
                }
            )
            continue
        try:
            fps, duration = _probe_video(source)
        except Exception:
            failures.append(
                {
                    "blind_id": blind_id,
                    "reason": "video_probe_failed",
                    "checksum_verified": True,
                }
            )
            continue
        prepared.append(
            {
                "blind_id": blind_id,
                "clip_id": clip_id,
                "tasks": tasks,
                "metadata": metadata,
                "source": source,
                "source_sha256": source_sha256,
                "fps": fps,
                "duration": duration,
            }
        )

    if failures:
        flag_path = _append_public_preflight_flag(out_dir, failures)
        raise RuntimeError(
            f"media preflight failed for {len(failures)} item(s); "
            f"redacted evidence written to {flag_path}"
        )

    items = []
    private_items = []
    for row in prepared:
        blind_id = row["blind_id"]
        clip_id = row["clip_id"]
        tasks = row["tasks"]
        metadata = row["metadata"]
        source = row["source"]
        source_sha256 = row["source_sha256"]
        fps = row["fps"]
        duration = row["duration"]
        media_path = Path("media") / f"{blind_id}.mp4"
        _copy_media(source, out_dir / media_path, source_sha256)
        target_prompt = (
            "Identify and briefly describe the visible target event before marking it."
            if tasks == ["anchor_presence"]
            else "Assess whether this clip contains two separable target events."
        )
        if set(tasks) == {"anchor_presence", "two_event"}:
            target_prompt = "Complete the single-event and two-event curation tasks for this clip."
        items.append(
            {
                "blind_id": blind_id,
                "tasks": tasks,
                "media_path": media_path.as_posix(),
                "fps": fps,
                "duration_s": duration,
                "target_prompt": target_prompt,
            }
        )
        private_items.append(
            {
                "blind_id": blind_id,
                "tasks": tasks,
                "source_manifest_positions": positions_by_clip[clip_id],
                "source_clip_id": clip_id,
                "source_path": str(source),
                "source_sha256": source_sha256,
                "caption": metadata["caption"],
                "ucs_category": metadata["ucs_category"],
                "source_type": metadata["source_type"],
                "candidate_status": "CURATION_ONLY_PENDING_FREEZE",
                "development_tuning_eligible": clip_id not in pilot_ids,
                "role_constraints": (
                    ["BANKED_PILOT_NOT_FOR_DEVELOPMENT_TUNING"]
                    if clip_id in pilot_ids
                    else []
                ),
            }
        )

    manifest = {
        "schema_version": "sounddecisions-human-eval-items-v2-1.0",
        "instrument_version": INSTRUMENT_VERSION,
        "manifest_id": MANIFEST_ID,
        "status": "INCOMPLETE_ARTIFACTS_PRE_FREEZE_DO_NOT_RATE",
        "default_fps": 30.0,
        "rubric": {
            "anchor": ANCHOR_RUBRIC,
            "presence": PRESENCE_RUBRIC,
            "two_event": TWO_EVENT_RUBRIC,
        },
        "counts": {
            "anchor_presence": len(anchor_rows),
            "two_event": len(two_ids),
            "total_tasks": len(anchor_rows) + len(two_ids),
            "unique_videos": len({row["source_clip_id"] for row in private_items}),
        },
        "items": items,
    }
    manifest_path = out_dir / "blinded_items.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    manifest_sha256 = _sha256(manifest_path)
    _embed_manifest(out_dir / "rate.html", manifest_path.read_text(encoding="utf-8"), manifest_sha256)

    mapping = {
        "schema_version": "sounddecisions-human-eval-unblinding-v1",
        "manifest_id": MANIFEST_ID,
        "source_contract": {
            "anchor_candidates_sha256": _sha256(args.anchor_csv),
            "two_event_candidates_sha256": _sha256(args.two_event_json),
            "clips_index_sha256": _sha256(args.clips_index),
            "pilot_manifest_sha256": _sha256(args.pilot_manifest),
        },
        "role_policy": "candidate curation does not admit an item to development; banked-pilot overlaps cannot tune a measurer",
        "items": private_items,
    }
    sealed_path = out_dir / "unblinding_map.sealed.json"
    envelope = _seal_mapping(mapping, sealed_path, args.key_file)
    return {
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "sealed_map": str(sealed_path),
        "sealed_map_sha256": _sha256(sealed_path),
        "key_id": envelope["key_id"],
        "key_file": str(args.key_file),
        "counts": manifest["counts"],
        "pilot_overlap_count": sum(not row["development_tuning_eligible"] for row in private_items),
    }


def main() -> int:
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor-csv", type=Path, default=repo / "data/manifests/anchor_check_30.csv")
    parser.add_argument("--two-event-json", type=Path, default=repo / "data/manifests/two_event_manifest.json")
    parser.add_argument("--clips-index", type=Path, required=True)
    parser.add_argument("--pilot-manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument(
        "--key-file",
        type=Path,
        default=Path.home() / ".config/sounddecisions/human_eval_pack.key",
    )
    args = parser.parse_args()
    summary = build(args)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
