#!/usr/bin/env python3
"""Build the deterministic, blinded Round-1 human-curation release.

The builder verifies each source video, remuxes only its first video stream, and
never opens generated audio or runs a measurer. Rater artifacts contain no
source/model/condition fields. All joins and media checks complete before
publication.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import errno
import hashlib
import hmac
import io
import json
import os
import shutil
import subprocess
import tempfile
import zipfile
from fractions import Fraction
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "sounddecisions-human-curation-items-v1-1.3"
INSTRUMENT_VERSION = "human-eval-round1-curation-1.3"
MANIFEST_ID = "axis-spec-v2-round1-curation-2026-07-v4"
RELEASE_ID = "round1_curation_v4"
RELEASE_STATUS = "CURATION_AUTHORIZED"
AUDIO_MEDIA_REGISTRY_SCHEMA_VERSION = (
    "sounddecisions-human-curation-audio-media-registry-v4-1.0"
)
AUDIO_MEDIA_REGISTRY_FILENAME = "round1_v4_audio_media_registry.json"
EXPECTED_ANCHOR_TASKS = 30
EXPECTED_TWO_EVENT_TASKS = 60
EXPECTED_UNIQUE_VIDEOS = 82
FIXED_ZIP_TIMESTAMP = (2026, 7, 19, 0, 0, 0)
SEALED_MAP_FORMAT = "sounddecisions-human-curation-sealed-map-v4"
SEALED_MAP_ITERATIONS = 200_000
_AT_FDCWD = -100
_RENAME_NOREPLACE = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(payload: object) -> bytes:
    return (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def _canonical_private_json(payload: object) -> bytes:
    return (
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    ).encode("utf-8")


def _atomic_write_bytes(destination: Path, content: bytes) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(0o644)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _rename_noreplace(source: Path, destination: Path) -> bool:
    """Atomically rename without replacement, or return False if unsupported.

    Linux ``renameat2(RENAME_NOREPLACE)`` closes the exists-check/rename race.
    Callers use creation-exclusive fallbacks when the kernel or libc does not
    expose it; they must never fall back to ``os.replace``.
    """

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        return False
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        _AT_FDCWD,
        os.fsencode(source),
        _AT_FDCWD,
        os.fsencode(destination),
        _RENAME_NOREPLACE,
    )
    if result == 0:
        return True
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise FileExistsError(error_number, os.strerror(error_number), destination)
    if error_number in {errno.ENOSYS, errno.EINVAL, errno.ENOTSUP}:
        return False
    raise OSError(error_number, os.strerror(error_number), destination)


def _copy_file_exclusive(source: Path, destination: Path) -> None:
    """Creation-exclusive cross-filesystem fallback for a staged file."""

    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with source.open("rb") as input_handle, os.fdopen(
            descriptor, "wb", closefd=True
        ) as output_handle:
            descriptor = -1
            shutil.copyfileobj(input_handle, output_handle, length=1024 * 1024)
            output_handle.flush()
            os.fsync(output_handle.fileno())
        destination.chmod(source.stat().st_mode & 0o777)
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        destination.unlink(missing_ok=True)
        raise
    source.unlink()


def _copy_directory_exclusive(source: Path, destination: Path) -> None:
    """Reserve a destination then copy when renameat2 is unavailable.

    The directory may be briefly incomplete, but no file is overwritten and
    the release record is not published until this copy has completed.
    """

    destination.mkdir(mode=source.stat().st_mode & 0o777)
    try:
        for path in sorted(source.rglob("*")):
            relative = path.relative_to(source)
            target = destination / relative
            if path.is_dir():
                target.mkdir(mode=path.stat().st_mode & 0o777)
            elif path.is_file():
                target.parent.mkdir(parents=True, exist_ok=True)
                with path.open("rb") as input_handle, target.open("xb") as output_handle:
                    shutil.copyfileobj(input_handle, output_handle, length=1024 * 1024)
                    output_handle.flush()
                    os.fsync(output_handle.fileno())
                target.chmod(path.stat().st_mode & 0o777)
            else:
                raise RuntimeError(f"unsupported staged release entry: {path}")
    except BaseException:
        shutil.rmtree(destination, ignore_errors=True)
        raise
    shutil.rmtree(source)


def _read_key(path: Path) -> bytes:
    if not path.is_file():
        raise RuntimeError(f"blinding key is missing: {path}")
    if path.stat().st_mode & 0o077:
        raise RuntimeError(f"blinding key must not be group/world accessible: {path}")
    key = path.read_bytes()
    if not key.strip():
        raise RuntimeError(f"blinding key is empty: {path}")
    return key


def _decrypt_sealed_envelope(envelope: dict[str, Any], key_file: Path) -> dict[str, Any]:
    key = _read_key(key_file)
    expected_key_id = hashlib.sha256(key).hexdigest()[:16]
    required = {
        "format": SEALED_MAP_FORMAT,
        "cipher": "AES-256-CBC",
        "kdf": "PBKDF2",
        "iterations": SEALED_MAP_ITERATIONS,
        "key_id": expected_key_id,
        "manifest_id": MANIFEST_ID,
    }
    for field, expected in required.items():
        if envelope.get(field) != expected:
            raise RuntimeError(f"sealed map {field} mismatch: expected {expected!r}")
    ciphertext = envelope.get("ciphertext_base64")
    plaintext_sha256 = envelope.get("plaintext_sha256")
    if not isinstance(ciphertext, str) or not isinstance(plaintext_sha256, str):
        raise RuntimeError("sealed map envelope is incomplete")
    command = [
        "openssl",
        "enc",
        "-d",
        "-aes-256-cbc",
        "-a",
        "-A",
        "-pbkdf2",
        "-iter",
        str(SEALED_MAP_ITERATIONS),
        "-pass",
        f"file:{key_file}",
    ]
    try:
        proc = subprocess.run(
            command, input=ciphertext.encode("ascii"), check=True, capture_output=True
        )
    except (UnicodeEncodeError, subprocess.CalledProcessError) as error:
        raise RuntimeError("sealed map decryption failed") from error
    if hashlib.sha256(proc.stdout).hexdigest() != plaintext_sha256:
        raise RuntimeError("sealed map plaintext checksum mismatch")
    try:
        payload = json.loads(proc.stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("sealed map plaintext is not valid JSON") from error
    if not isinstance(payload, dict):
        raise RuntimeError("sealed map plaintext must be a JSON object")
    return payload


def _assert_private_mapping_matches(
    actual: dict[str, Any], expected: dict[str, Any]
) -> None:
    if actual.get("manifest_id") != MANIFEST_ID:
        raise RuntimeError("sealed map manifest ID mismatch")
    actual_items = actual.get("items")
    expected_items = expected.get("items")
    if not isinstance(actual_items, list) or not isinstance(expected_items, list):
        raise RuntimeError("sealed map item registry is invalid")
    actual_by_id = {row.get("blind_id"): row for row in actual_items if isinstance(row, dict)}
    expected_by_id = {row["blind_id"]: row for row in expected_items}
    if set(actual_by_id) != set(expected_by_id):
        raise RuntimeError("sealed map blind-ID set mismatch")
    if actual.get("source_contract_sha256") != expected.get("source_contract_sha256"):
        raise RuntimeError("sealed map source-contract hashes mismatch")
    for blind_id, expected_item in expected_by_id.items():
        actual_item = actual_by_id[blind_id]
        for field in ("source_sha256", "delivered_media_sha256"):
            if actual_item.get(field) != expected_item[field]:
                raise RuntimeError(f"sealed map {field} mismatch for {blind_id}")
    if actual != expected:
        raise RuntimeError("sealed map plaintext differs from the exact Round-1 v4 registry")


def _stage_sealed_mapping(
    mapping: dict[str, Any],
    staged_destination: Path,
    existing_destination: Path,
    key_file: Path,
) -> str:
    plaintext = _canonical_private_json(mapping)
    plaintext_sha256 = hashlib.sha256(plaintext).hexdigest()
    key = _read_key(key_file)
    key_id = hashlib.sha256(key).hexdigest()[:16]
    if existing_destination.exists():
        try:
            envelope = json.loads(existing_destination.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError("existing Round-1 v4 sealed map is unreadable") from error
        actual = _decrypt_sealed_envelope(envelope, key_file)
        _assert_private_mapping_matches(actual, mapping)
        if envelope.get("plaintext_sha256") != plaintext_sha256:
            raise RuntimeError("existing Round-1 v4 sealed map plaintext hash differs")
        shutil.copyfile(existing_destination, staged_destination)
        staged_destination.chmod(0o600)
        return _sha256(staged_destination)

    command = [
        "openssl",
        "enc",
        "-aes-256-cbc",
        "-a",
        "-A",
        "-salt",
        "-pbkdf2",
        "-iter",
        str(SEALED_MAP_ITERATIONS),
        "-pass",
        f"file:{key_file}",
    ]
    proc = subprocess.run(command, input=plaintext, check=True, capture_output=True)
    envelope = {
        "format": SEALED_MAP_FORMAT,
        "cipher": "AES-256-CBC",
        "kdf": "PBKDF2",
        "iterations": SEALED_MAP_ITERATIONS,
        "key_id": key_id,
        "manifest_id": MANIFEST_ID,
        "plaintext_sha256": plaintext_sha256,
        "ciphertext_base64": proc.stdout.decode("ascii"),
    }
    _atomic_write_bytes(staged_destination, _canonical_private_json(envelope))
    staged_destination.chmod(0o600)
    _assert_private_mapping_matches(
        _decrypt_sealed_envelope(envelope, key_file), mapping
    )
    return _sha256(staged_destination)


def _blind_id(clip_id: str, key: bytes) -> str:
    digest = hmac.new(key, f"human-eval-v2:{clip_id}".encode("ascii"), hashlib.sha256)
    return f"HEV2-{digest.hexdigest()[:12].upper()}"


def _load_clip_index(content: bytes, path: Path) -> dict[str, dict[str, str]]:
    rows = list(csv.DictReader(io.StringIO(content.decode("utf-8"), newline="")))
    required = {"key", "path", "sha256", "caption"}
    if not rows or not required.issubset(rows[0]):
        raise RuntimeError(f"clips index lacks required columns {sorted(required)}: {path}")
    result = {row["key"]: row for row in rows}
    if len(result) != len(rows):
        raise RuntimeError(f"duplicate clip keys in {path}")
    return result


def _load_candidate_tasks(
    anchor_content: bytes,
    two_event_content: bytes,
    anchor_csv: Path,
    two_event_json: Path,
) -> dict[str, list[str]]:
    anchor_rows = list(
        csv.DictReader(io.StringIO(anchor_content.decode("utf-8"), newline=""))
    )
    two_payload = json.loads(two_event_content.decode("utf-8"))
    two_ids = [str(value) for value in two_payload.get("clips", [])]

    if len(anchor_rows) != EXPECTED_ANCHOR_TASKS:
        raise RuntimeError(
            f"anchor candidate contract changed: expected {EXPECTED_ANCHOR_TASKS}, "
            f"found {len(anchor_rows)}"
        )
    if not anchor_rows or "key" not in anchor_rows[0]:
        raise RuntimeError(f"anchor candidate manifest lacks key column: {anchor_csv}")
    anchor_ids = [str(row["key"]) for row in anchor_rows]
    if len(set(anchor_ids)) != len(anchor_ids):
        raise RuntimeError(f"duplicate anchor candidate IDs in {anchor_csv}")
    if len(two_ids) != EXPECTED_TWO_EVENT_TASKS or two_payload.get("n") != len(two_ids):
        raise RuntimeError(
            f"two-event candidate contract changed: expected {EXPECTED_TWO_EVENT_TASKS}, "
            f"found {len(two_ids)} with declared n={two_payload.get('n')!r}"
        )
    if len(set(two_ids)) != len(two_ids):
        raise RuntimeError(f"duplicate two-event candidate IDs in {two_event_json}")

    tasks_by_clip: dict[str, list[str]] = {}
    for clip_id in anchor_ids:
        tasks_by_clip.setdefault(clip_id, []).append("anchor_curation")
    for clip_id in two_ids:
        tasks_by_clip.setdefault(clip_id, []).append("two_event_curation")
    if len(tasks_by_clip) != EXPECTED_UNIQUE_VIDEOS:
        raise RuntimeError(
            f"candidate overlap contract changed: expected {EXPECTED_UNIQUE_VIDEOS} unique "
            f"videos, found {len(tasks_by_clip)}"
        )
    return tasks_by_clip


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
        raise RuntimeError(f"expected exactly one video stream: {path}")
    fps = float(Fraction(streams[0]["avg_frame_rate"]))
    duration = float(payload["format"]["duration"])
    if fps <= 0 or duration <= 0:
        raise RuntimeError(f"invalid video metadata: {path}")
    return fps, duration


def _verify_video_only(path: Path) -> tuple[float, float]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "stream=codec_type,avg_frame_rate:format=duration",
        "-of",
        "json",
        str(path),
    ]
    proc = subprocess.run(command, check=True, capture_output=True, text=True)
    payload = json.loads(proc.stdout)
    streams = payload.get("streams", [])
    video_streams = [stream for stream in streams if stream.get("codec_type") == "video"]
    audio_streams = [stream for stream in streams if stream.get("codec_type") == "audio"]
    if len(video_streams) != 1:
        raise RuntimeError(f"delivered media must have exactly one video stream: {path.name}")
    if audio_streams:
        raise RuntimeError(f"delivered media contains an audio stream: {path.name}")
    if len(streams) != 1:
        raise RuntimeError(f"delivered media contains a non-video stream: {path.name}")
    fps = float(Fraction(video_streams[0]["avg_frame_rate"]))
    duration = float(payload["format"]["duration"])
    if fps <= 0 or duration <= 0:
        raise RuntimeError(f"delivered media has invalid video metadata: {path.name}")
    return fps, duration


def _preflight(
    tasks_by_clip: dict[str, list[str]],
    index: dict[str, dict[str, str]],
    key: bytes,
) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    blind_ids: set[str] = set()

    for clip_id, tasks in tasks_by_clip.items():
        blind_id = _blind_id(clip_id, key)
        metadata = index.get(clip_id)
        reason = ""
        if metadata is None:
            reason = "index_join_missing"
        elif not metadata.get("caption", "").strip():
            reason = "caption_missing"
        else:
            source = Path(metadata["path"])
            if not source.is_file():
                reason = "video_missing"
            elif _sha256(source) != metadata["sha256"]:
                reason = "video_checksum_mismatch"
            else:
                try:
                    fps, duration = _probe_video(source)
                except Exception:
                    reason = "video_probe_failed"
                else:
                    if blind_id in blind_ids:
                        reason = "blind_id_collision"
                    else:
                        blind_ids.add(blind_id)
                        prepared.append(
                            {
                                "blind_id": blind_id,
                                "source_clip_id": clip_id,
                                "tasks": list(tasks),
                                "source": source,
                                "source_sha256": metadata["sha256"],
                                "candidate_caption": metadata["caption"].strip(),
                                "fps": fps,
                                "duration_s": duration,
                            }
                        )
        if reason:
            failures.append({"blind_id": blind_id, "reason": reason})

    if failures:
        evidence = ", ".join(f"{row['blind_id']}:{row['reason']}" for row in failures)
        raise RuntimeError(f"release preflight failed for {len(failures)} item(s): {evidence}")
    if len(prepared) != EXPECTED_UNIQUE_VIDEOS:
        raise RuntimeError(
            f"release preflight produced {len(prepared)} items, expected {EXPECTED_UNIQUE_VIDEOS}"
        )
    return sorted(prepared, key=lambda row: row["blind_id"])


def _validate_json(instance: object, schema_bytes: bytes) -> None:
    try:
        import jsonschema
    except ImportError as error:  # pragma: no cover - dependency failure path
        raise RuntimeError("jsonschema is required to build the human-eval release") from error
    schema = json.loads(schema_bytes.decode("utf-8"))
    jsonschema.Draft202012Validator(schema).validate(instance)


def _render_html(template_bytes: bytes, manifest_bytes: bytes, manifest_sha256: str) -> bytes:
    template = template_bytes.decode("utf-8")
    json_marker = "__ROUND1_MANIFEST_JSON__"
    sha_marker = "__ROUND1_MANIFEST_SHA256__"
    if template.count(json_marker) != 1 or template.count(sha_marker) != 1:
        raise RuntimeError("Round-1 HTML template must contain each manifest marker exactly once")
    safe_manifest = manifest_bytes.decode("utf-8").replace("</", "<\\/")
    rendered = template.replace(json_marker, safe_manifest).replace(sha_marker, manifest_sha256)
    if json_marker in rendered or sha_marker in rendered:
        raise RuntimeError("Round-1 HTML template markers were not fully replaced")
    return rendered.encode("utf-8")


def _snapshot_contract_files(paths: dict[str, Path]) -> tuple[dict[str, bytes], dict[str, str]]:
    snapshots: dict[str, bytes] = {}
    hashes: dict[str, str] = {}
    for label, path in paths.items():
        try:
            content = path.read_bytes()
        except OSError as error:
            raise RuntimeError(f"release contract file is unreadable ({label}): {path}") from error
        snapshots[label] = content
        hashes[label] = hashlib.sha256(content).hexdigest()
    return snapshots, hashes


def _assert_contract_files_unchanged(
    paths: dict[str, Path], expected_hashes: dict[str, str]
) -> None:
    for label, path in paths.items():
        try:
            actual = _sha256(path)
        except OSError as error:
            raise RuntimeError(
                f"release contract file changed during build ({label}): {path}"
            ) from error
        if actual != expected_hashes[label]:
            raise RuntimeError(
                f"release contract file changed during build ({label}): {path}"
            )


def _remux_video_only(source: Path, destination: Path) -> str:
    if source.absolute() == destination.absolute():
        raise RuntimeError("delivered media path must differ from its source")
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-nostdin",
        "-y",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-c:v",
        "copy",
        "-an",
        "-map_metadata",
        "-1",
        "-map_metadata:s",
        "-1",
        "-map_chapters",
        "-1",
        "-fflags",
        "+bitexact",
        str(destination),
    ]
    subprocess.run(command, check=True, capture_output=True)
    destination.chmod(0o644)
    _verify_video_only(destination)
    if os.path.samefile(source, destination):
        raise RuntimeError(f"release media is not an independent copy: {destination.name}")
    return _sha256(destination)


def _directory_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _write_deterministic_zip(source_dir: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_STORED) as archive:
        for path in sorted(candidate for candidate in source_dir.rglob("*") if candidate.is_file()):
            relative = path.relative_to(source_dir).as_posix()
            info = zipfile.ZipInfo(relative, date_time=FIXED_ZIP_TIMESTAMP)
            info.create_system = 3
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o100644 << 16
            info.flag_bits |= 0x800
            with path.open("rb") as handle, archive.open(info, "w") as zipped:
                shutil.copyfileobj(handle, zipped, length=1024 * 1024)


def _publish_immutable_file(staged: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        if not _rename_noreplace(staged, destination):
            _copy_file_exclusive(staged, destination)
        return
    except FileExistsError:
        pass
    if not destination.is_file() or _sha256(staged) != _sha256(destination):
        raise RuntimeError(f"refusing to overwrite non-identical release artifact: {destination}")
    staged.unlink()


def _publish_immutable_directory(staged: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        if not _rename_noreplace(staged, destination):
            _copy_directory_exclusive(staged, destination)
        return
    except FileExistsError:
        pass
    if not destination.is_dir() or _directory_digest(staged) != _directory_digest(destination):
        raise RuntimeError(f"refusing to overwrite non-identical release directory: {destination}")
    shutil.rmtree(staged)


def _publication_checkpoint(label: str) -> None:
    """Test hook for proving that partial publication never commits a release."""


def _assert_publishable(
    staged_dir: Path,
    staged_zip: Path,
    staged_sealed_map: Path,
    staged_audio_registry: Path,
    release_dir: Path,
    zip_path: Path,
    sealed_map_path: Path,
    audio_registry_path: Path,
    public_manifest_path: Path,
    release_record_path: Path,
    staged_public_manifest: Path,
    staged_release_record: Path,
) -> None:
    if release_dir.exists() and (
        not release_dir.is_dir() or _directory_digest(staged_dir) != _directory_digest(release_dir)
    ):
        raise RuntimeError(f"refusing to overwrite non-identical release directory: {release_dir}")
    if zip_path.exists() and (
        not zip_path.is_file() or _sha256(staged_zip) != _sha256(zip_path)
    ):
        raise RuntimeError(f"refusing to overwrite non-identical release artifact: {zip_path}")
    if sealed_map_path.exists() and (
        not sealed_map_path.is_file()
        or _sha256(staged_sealed_map) != _sha256(sealed_map_path)
    ):
        raise RuntimeError(f"refusing to overwrite non-identical sealed map: {sealed_map_path}")
    if audio_registry_path.exists() and (
        not audio_registry_path.is_file()
        or _sha256(staged_audio_registry) != _sha256(audio_registry_path)
    ):
        raise RuntimeError(
            f"refusing to overwrite non-identical audio-media registry: {audio_registry_path}"
        )
    if public_manifest_path.exists() and (
        not public_manifest_path.is_file()
        or _sha256(staged_public_manifest) != _sha256(public_manifest_path)
    ):
        raise RuntimeError(
            f"refusing to overwrite non-identical public manifest: {public_manifest_path}"
        )
    if release_record_path.exists() and (
        not release_record_path.is_file()
        or _sha256(staged_release_record) != _sha256(release_record_path)
    ):
        raise RuntimeError(
            f"refusing to overwrite non-identical release record: {release_record_path}"
        )


def build(args: argparse.Namespace) -> dict[str, Any]:
    contract_paths = {
        "anchor_candidates": args.anchor_csv,
        "two_event_candidates": args.two_event_json,
        "clips_index": args.clips_index,
        "html_template": args.html_template,
        "instructions": args.instructions,
        "manifest_schema": args.manifest_schema,
        "ratings_schema": args.ratings_schema,
        "audio_media_registry_schema": args.audio_media_registry_schema,
    }
    contract_bytes, contract_hashes = _snapshot_contract_files(contract_paths)
    tasks_by_clip = _load_candidate_tasks(
        contract_bytes["anchor_candidates"],
        contract_bytes["two_event_candidates"],
        args.anchor_csv,
        args.two_event_json,
    )
    index = _load_clip_index(contract_bytes["clips_index"], args.clips_index)
    key = _read_key(args.key_file)
    prepared = _preflight(tasks_by_clip, index, key)

    items = [
        {
            "blind_id": row["blind_id"],
            "tasks": row["tasks"],
            "media_path": f"media/{row['blind_id']}.mp4",
            "fps": row["fps"],
            "duration_s": row["duration_s"],
            "candidate_caption": row["candidate_caption"],
        }
        for row in prepared
    ]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "instrument_version": INSTRUMENT_VERSION,
        "manifest_id": MANIFEST_ID,
        "status": RELEASE_STATUS,
        "default_fps": 30.0,
        "counts": {
            "anchor_curation": EXPECTED_ANCHOR_TASKS,
            "two_event_curation": EXPECTED_TWO_EVENT_TASKS,
            "total_tasks": EXPECTED_ANCHOR_TASKS + EXPECTED_TWO_EVENT_TASKS,
            "unique_videos": EXPECTED_UNIQUE_VIDEOS,
        },
        "items": items,
    }
    _validate_json(manifest, contract_bytes["manifest_schema"])
    manifest_bytes = _canonical_json(manifest)
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    rate_html = _render_html(
        contract_bytes["html_template"], manifest_bytes, manifest_sha256
    )
    audio_media_registry = {
        "schema_version": AUDIO_MEDIA_REGISTRY_SCHEMA_VERSION,
        "manifest_id": MANIFEST_ID,
        "manifest_sha256": manifest_sha256,
        "items": [
            {
                "blind_id": row["blind_id"],
                "media_path": f"media/{row['blind_id']}.mp4",
                "sha256": row["source_sha256"],
            }
            for row in prepared
        ],
    }
    _validate_json(
        audio_media_registry, contract_bytes["audio_media_registry_schema"]
    )
    audio_registry_bytes = _canonical_json(audio_media_registry)
    audio_registry_sha256 = hashlib.sha256(audio_registry_bytes).hexdigest()
    private_source_contract = {
        "anchor_candidates": contract_hashes["anchor_candidates"],
        "two_event_candidates": contract_hashes["two_event_candidates"],
        "clips_index": contract_hashes["clips_index"],
        "public_manifest": manifest_sha256,
        "audio_media_registry": audio_registry_sha256,
        "audio_media_registry_schema": contract_hashes["audio_media_registry_schema"],
    }

    release_parent = args.release_dir.parent.resolve()
    release_parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{RELEASE_ID}.", dir=release_parent))
    zip_descriptor, zip_temporary_name = tempfile.mkstemp(
        prefix=f".{RELEASE_ID}.", suffix=".zip.tmp", dir=release_parent
    )
    os.close(zip_descriptor)
    staged_zip = Path(zip_temporary_name)
    sealed_descriptor, sealed_temporary_name = tempfile.mkstemp(
        prefix=f".{RELEASE_ID}.", suffix=".sealed.tmp", dir=release_parent
    )
    os.close(sealed_descriptor)
    staged_sealed_map = Path(sealed_temporary_name)
    staged_sealed_map.unlink()
    registry_descriptor, registry_temporary_name = tempfile.mkstemp(
        prefix=f".{RELEASE_ID}.", suffix=".audio-registry.tmp", dir=release_parent
    )
    os.close(registry_descriptor)
    staged_audio_registry = Path(registry_temporary_name)
    _atomic_write_bytes(staged_audio_registry, audio_registry_bytes)
    staged_audio_registry.chmod(0o600)
    manifest_descriptor, manifest_temporary_name = tempfile.mkstemp(
        prefix=f".{RELEASE_ID}.", suffix=".manifest.tmp", dir=release_parent
    )
    os.close(manifest_descriptor)
    staged_public_manifest = Path(manifest_temporary_name)
    _atomic_write_bytes(staged_public_manifest, manifest_bytes)
    record_descriptor, record_temporary_name = tempfile.mkstemp(
        prefix=f".{RELEASE_ID}.", suffix=".record.tmp", dir=release_parent
    )
    os.close(record_descriptor)
    staged_release_record = Path(record_temporary_name)
    release_record: dict[str, Any]
    try:
        _atomic_write_bytes(staging / "blinded_items.json", manifest_bytes)
        _atomic_write_bytes(staging / "rate.html", rate_html)
        _atomic_write_bytes(staging / "INSTRUCTIONS.md", contract_bytes["instructions"])
        _atomic_write_bytes(
            staging / "blinded_items.schema.json", contract_bytes["manifest_schema"]
        )
        _atomic_write_bytes(
            staging / "ratings.schema.json", contract_bytes["ratings_schema"]
        )
        for path in (
            staging / "INSTRUCTIONS.md",
            staging / "blinded_items.schema.json",
            staging / "ratings.schema.json",
        ):
            path.chmod(0o644)
        private_items = []
        for row in prepared:
            delivered_path = staging / "media" / f"{row['blind_id']}.mp4"
            if _sha256(row["source"]) != row["source_sha256"]:
                raise RuntimeError(
                    f"source video changed before remux: {row['blind_id']}"
                )
            delivered_sha256 = _remux_video_only(row["source"], delivered_path)
            if _sha256(row["source"]) != row["source_sha256"]:
                raise RuntimeError(
                    f"source video changed during remux: {row['blind_id']}"
                )
            private_items.append(
                {
                    "blind_id": row["blind_id"],
                    "tasks": row["tasks"],
                    "source_clip_id": row["source_clip_id"],
                    "source_path": str(row["source"]),
                    "source_sha256": row["source_sha256"],
                    "delivered_media_path": f"media/{row['blind_id']}.mp4",
                    "delivered_media_sha256": delivered_sha256,
                }
            )

        private_mapping = {
            "schema_version": "sounddecisions-human-curation-unblinding-v4-1.0",
            "manifest_id": MANIFEST_ID,
            "manifest_sha256": manifest_sha256,
            "source_contract_sha256": private_source_contract,
            "items": private_items,
        }
        sealed_destination = args.sealed_map.resolve()
        sealed_map_sha256 = _stage_sealed_mapping(
            private_mapping,
            staged_sealed_map,
            sealed_destination,
            args.key_file,
        )

        checksum_rows = []
        for path in sorted(candidate for candidate in staging.rglob("*") if candidate.is_file()):
            checksum_rows.append(f"{_sha256(path)}  {path.relative_to(staging).as_posix()}")
        _atomic_write_bytes(staging / "SHA256SUMS.txt", ("\n".join(checksum_rows) + "\n").encode("utf-8"))
        _write_deterministic_zip(staging, staged_zip)

        zip_sha256 = _sha256(staged_zip)
        zip_size_bytes = staged_zip.stat().st_size
        release_digest = _directory_digest(staging)
        release_destination = args.release_dir.resolve()
        zip_destination = args.zip_path.resolve()
        audio_registry_destination = args.audio_media_registry.resolve()
        public_manifest_path = args.public_manifest.resolve()
        release_record_path = args.release_record.resolve()
        release_record = {
            "schema_version": "sounddecisions-human-curation-release-v4",
            "release_id": RELEASE_ID,
            "status": RELEASE_STATUS,
            "counts": manifest["counts"],
            "manifest": {
                "path": public_manifest_path.name,
                "sha256": manifest_sha256,
            },
            "sealed_unblinding_map": {
                "path": args.sealed_map.name,
                "sha256": sealed_map_sha256,
            },
            "audio_media_registry": {
                "storage": "operator_private",
                "sha256": audio_registry_sha256,
                "item_count": len(audio_media_registry["items"]),
            },
            "zip": {
                "path": f"releases/{args.zip_path.name}",
                "sha256": zip_sha256,
                "size_bytes": zip_size_bytes,
            },
            "release_directory_sha256": release_digest,
            "source_contract_sha256": {
                **private_source_contract,
                "html_template": contract_hashes["html_template"],
                "instructions": contract_hashes["instructions"],
                "manifest_schema": contract_hashes["manifest_schema"],
                "ratings_schema": contract_hashes["ratings_schema"],
            },
        }
        _atomic_write_bytes(staged_release_record, _canonical_json(release_record))
        _assert_contract_files_unchanged(contract_paths, contract_hashes)
        _assert_publishable(
            staging,
            staged_zip,
            staged_sealed_map,
            staged_audio_registry,
            release_destination,
            zip_destination,
            sealed_destination,
            audio_registry_destination,
            public_manifest_path,
            release_record_path,
            staged_public_manifest,
            staged_release_record,
        )
        _publication_checkpoint("audio_media_registry")
        _publish_immutable_file(staged_audio_registry, audio_registry_destination)
        _publication_checkpoint("sealed_unblinding_map")
        _publish_immutable_file(staged_sealed_map, sealed_destination)
        _publication_checkpoint("release_directory")
        _publish_immutable_directory(staging, release_destination)
        _publication_checkpoint("zip")
        _publish_immutable_file(staged_zip, zip_destination)
        _publication_checkpoint("public_manifest")
        _publish_immutable_file(staged_public_manifest, public_manifest_path)
        _publication_checkpoint("release_record")
        _publish_immutable_file(staged_release_record, release_record_path)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
        staged_zip.unlink(missing_ok=True)
        staged_sealed_map.unlink(missing_ok=True)
        staged_audio_registry.unlink(missing_ok=True)
        staged_public_manifest.unlink(missing_ok=True)
        staged_release_record.unlink(missing_ok=True)
    return release_record


def main() -> int:
    pack_dir = Path(__file__).resolve().parent
    repo = pack_dir.parents[1]
    source_dir = pack_dir / "release_src"
    release_dir = pack_dir / "releases" / RELEASE_ID
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-csv", type=Path, default=repo / "data/manifests/anchor_check_30.csv")
    parser.add_argument(
        "--two-event-json", type=Path, default=repo / "data/manifests/two_event_manifest.json"
    )
    parser.add_argument("--clips-index", type=Path, required=True)
    parser.add_argument(
        "--key-file",
        type=Path,
        default=Path.home() / ".config/sounddecisions/human_eval_pack.key",
    )
    parser.add_argument(
        "--sealed-map",
        type=Path,
        default=pack_dir / "round1_v4_unblinding_map.sealed.json",
    )
    parser.add_argument("--html-template", type=Path, default=source_dir / "round1_rate.template.html")
    parser.add_argument("--instructions", type=Path, default=source_dir / "INSTRUCTIONS_ROUND1.md")
    parser.add_argument("--manifest-schema", type=Path, default=source_dir / "round1_manifest.schema.json")
    parser.add_argument("--ratings-schema", type=Path, default=source_dir / "round1_ratings.schema.json")
    parser.add_argument(
        "--audio-media-registry-schema",
        type=Path,
        default=source_dir / "audio_media_registry.schema.json",
    )
    parser.add_argument("--release-dir", type=Path, default=release_dir)
    parser.add_argument("--zip-path", type=Path, default=release_dir.with_suffix(".zip"))
    parser.add_argument(
        "--public-manifest", type=Path, default=pack_dir / "round1_v4_blinded_items.json"
    )
    parser.add_argument(
        "--audio-media-registry",
        type=Path,
        default=pack_dir / "private" / AUDIO_MEDIA_REGISTRY_FILENAME,
    )
    parser.add_argument(
        "--release-record", type=Path, default=pack_dir / "ROUND1_V4_RELEASE.json"
    )
    args = parser.parse_args()
    print(json.dumps(build(args), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
