"""Determinism, privacy, and fail-closed tests for the Round-1 release builder."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
BUILD_PATH = ROOT / "results" / "human_eval_pack" / "build_release.py"
BUILD_SPEC = importlib.util.spec_from_file_location("human_eval_release_builder", BUILD_PATH)
assert BUILD_SPEC is not None and BUILD_SPEC.loader is not None
BUILDER = importlib.util.module_from_spec(BUILD_SPEC)
BUILD_SPEC.loader.exec_module(BUILDER)
SOURCE_DIR = ROOT / "results" / "human_eval_pack" / "release_src"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> tuple[argparse.Namespace, list[str]]:
    anchor_ids = [f"A{index:03d}" for index in range(30)]
    two_ids = anchor_ids[:8] + [f"T{index:03d}" for index in range(52)]
    all_ids = list(dict.fromkeys([*anchor_ids, *two_ids]))

    anchor_csv = tmp_path / "anchor.csv"
    with anchor_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["key"])
        writer.writeheader()
        writer.writerows({"key": clip_id} for clip_id in anchor_ids)

    two_event_json = tmp_path / "two_event.json"
    two_event_json.write_text(json.dumps({"clips": two_ids, "n": 60}), encoding="utf-8")

    source_video = tmp_path / "private" / "source_video.mp4"
    source_video.parent.mkdir()
    source_video.write_bytes(b"synthetic-mp4-content")
    clips_index = tmp_path / "clips_index.csv"
    with clips_index.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["key", "path", "sha256", "caption"])
        writer.writeheader()
        for clip_id in all_ids:
            writer.writerow(
                {
                    "key": clip_id,
                    "path": source_video,
                    "sha256": _sha256(source_video),
                    "caption": "A person performs the caption-guided visible action.",
                }
            )

    key_file = tmp_path / "human_eval.key"
    key_file.write_text("fixture-blinding-key\n", encoding="ascii")
    key_file.chmod(0o600)
    sealed_map = tmp_path / "round1_v3_unblinding_map.sealed.json"
    template = tmp_path / "rate.template.html"
    template.write_text(
        '<script id="manifest-data" type="application/json" '
        'data-sha256="__ROUND1_MANIFEST_SHA256__">'
        "__ROUND1_MANIFEST_JSON__</script>\n",
        encoding="utf-8",
    )
    instructions = tmp_path / "INSTRUCTIONS.md"
    instructions.write_text("Curate visible events with audio muted.\n", encoding="utf-8")

    out = tmp_path / "out"
    args = argparse.Namespace(
        anchor_csv=anchor_csv,
        two_event_json=two_event_json,
        clips_index=clips_index,
        key_file=key_file,
        sealed_map=sealed_map,
        html_template=template,
        instructions=instructions,
        manifest_schema=SOURCE_DIR / "round1_manifest.schema.json",
        ratings_schema=SOURCE_DIR / "round1_ratings.schema.json",
        release_dir=out / "release",
        zip_path=out / "release.zip",
        public_manifest=out / "round1_v3_blinded_items.json",
        release_record=out / "ROUND1_V3_RELEASE.json",
    )
    return args, all_ids


def _all_json_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {key for child in value.values() for key in _all_json_keys(child)}
    if isinstance(value, list):
        return {key for child in value for key in _all_json_keys(child)}
    return set()


def _install_fake_video_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(BUILDER, "_probe_video", lambda _: (24.0, 10.0))
    monkeypatch.setattr(BUILDER, "SEALED_MAP_ITERATIONS", 1_000)

    def remux(source: Path, destination: Path) -> str:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"video-only-fixture\n" + source.read_bytes())
        destination.chmod(0o644)
        return _sha256(destination)

    monkeypatch.setattr(BUILDER, "_remux_video_only", remux)


def test_release_is_authorized_blinded_and_contains_independent_media(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, source_ids = _fixture(tmp_path)
    _install_fake_video_pipeline(monkeypatch)

    record = BUILDER.build(args)
    manifest = json.loads(args.public_manifest.read_text(encoding="utf-8"))

    assert manifest["status"] == "CURATION_AUTHORIZED"
    assert manifest["counts"] == {
        "anchor_curation": 30,
        "two_event_curation": 60,
        "total_tasks": 90,
        "unique_videos": 82,
    }
    assert len(manifest["items"]) == 82
    assert sum("anchor_curation" in item["tasks"] for item in manifest["items"]) == 30
    assert sum("two_event_curation" in item["tasks"] for item in manifest["items"]) == 60
    assert all(set(item) == {
        "blind_id", "tasks", "media_path", "fps", "duration_s", "candidate_caption"
    } for item in manifest["items"])
    assert _all_json_keys(manifest).isdisjoint(
        {"source", "source_id", "source_path", "model", "condition", "cfg", "seed"}
    )
    serialized = args.public_manifest.read_text(encoding="utf-8")
    assert all(source_id not in serialized for source_id in source_ids)
    assert record["manifest"]["sha256"] == _sha256(args.public_manifest)
    assert record["sealed_unblinding_map"]["sha256"] == _sha256(args.sealed_map)
    envelope = json.loads(args.sealed_map.read_text(encoding="utf-8"))
    private_mapping = BUILDER._decrypt_sealed_envelope(envelope, args.key_file)
    assert private_mapping["manifest_id"] == BUILDER.MANIFEST_ID
    assert private_mapping["manifest_sha256"] == _sha256(args.public_manifest)
    assert {row["source_clip_id"] for row in private_mapping["items"]} == set(source_ids)
    assert len(private_mapping["items"]) == 82
    for row in private_mapping["items"]:
        assert _sha256(args.release_dir / row["delivered_media_path"]) == row[
            "delivered_media_sha256"
        ]
        assert _sha256(Path(row["source_path"])) == row["source_sha256"]

    for item in manifest["items"]:
        media = args.release_dir / item["media_path"]
        assert media.is_file()
        assert not os.path.samefile(media, Path(next(csv.DictReader(args.clips_index.open()))["path"]))

    html = (args.release_dir / "rate.html").read_text(encoding="utf-8")
    assert "__ROUND1_MANIFEST_JSON__" not in html
    assert "__ROUND1_MANIFEST_SHA256__" not in html
    assert f'data-sha256="{_sha256(args.public_manifest)}"' in html

    first_sealed_bytes = args.sealed_map.read_bytes()
    BUILDER.build(args)
    assert args.sealed_map.read_bytes() == first_sealed_bytes


def test_zip_is_byte_deterministic_with_fixed_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, _ = _fixture(tmp_path)
    _install_fake_video_pipeline(monkeypatch)

    first = BUILDER.build(args)
    first_zip_sha = _sha256(args.zip_path)
    second_args = argparse.Namespace(**vars(args))
    second_args.release_dir = tmp_path / "second" / "release"
    second_args.zip_path = tmp_path / "second" / "release.zip"
    second_args.sealed_map = tmp_path / "second" / "round1_v3_unblinding_map.sealed.json"
    second_args.public_manifest = tmp_path / "second" / "round1_v3_blinded_items.json"
    second_args.release_record = tmp_path / "second" / "ROUND1_V3_RELEASE.json"
    second = BUILDER.build(second_args)

    assert _sha256(second_args.zip_path) == first_zip_sha
    assert first["zip"]["sha256"] == second["zip"]["sha256"] == first_zip_sha
    with zipfile.ZipFile(args.zip_path) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        assert names == sorted(names)
        assert all(info.date_time == BUILDER.FIXED_ZIP_TIMESTAMP for info in infos)
        assert "rate.html" in names
        assert "INSTRUCTIONS.md" in names
        assert "blinded_items.json" in names
        assert "blinded_items.schema.json" in names
        assert "ratings.schema.json" in names
        assert "SHA256SUMS.txt" in names
        assert len([name for name in names if name.startswith("media/")]) == 82
        assert all("sealed" not in name.lower() and "unblind" not in name.lower() for name in names)


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("checksum", "video_checksum_mismatch"),
        ("missing_join", "index_join_missing"),
        ("missing_file", "video_missing"),
        ("probe", "video_probe_failed"),
    ],
)
def test_preflight_failure_publishes_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
    message: str,
) -> None:
    args, _ = _fixture(tmp_path)
    rows = list(csv.DictReader(args.clips_index.open(newline="", encoding="utf-8")))
    target = rows[0]
    if failure == "checksum":
        target["sha256"] = "f" * 64
    elif failure == "missing_join":
        rows.pop(0)
    elif failure == "missing_file":
        target["path"] = str(tmp_path / "does-not-exist.mp4")
    fields = ["key", "path", "sha256", "caption"]
    with args.clips_index.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    if failure == "probe":
        monkeypatch.setattr(BUILDER, "_probe_video", lambda _: (_ for _ in ()).throw(ValueError()))
    else:
        monkeypatch.setattr(BUILDER, "_probe_video", lambda _: (24.0, 10.0))

    with pytest.raises(RuntimeError, match=message):
        BUILDER.build(args)

    assert not args.release_dir.exists()
    assert not args.zip_path.exists()
    assert not args.public_manifest.exists()
    assert not args.release_record.exists()


def test_existing_nonidentical_release_is_never_overwritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, _ = _fixture(tmp_path)
    _install_fake_video_pipeline(monkeypatch)
    BUILDER.build(args)
    original_zip = args.zip_path.read_bytes()
    (args.release_dir / "rate.html").write_text("tampered release\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        BUILDER.build(args)

    assert args.zip_path.read_bytes() == original_zip
    assert (args.release_dir / "rate.html").read_text(encoding="utf-8") == "tampered release\n"


def test_existing_sealed_map_is_decrypted_and_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, _ = _fixture(tmp_path)
    _install_fake_video_pipeline(monkeypatch)
    BUILDER.build(args)
    original_zip = args.zip_path.read_bytes()
    envelope = json.loads(args.sealed_map.read_text(encoding="utf-8"))
    envelope["manifest_id"] = "wrong-manifest"
    args.sealed_map.write_text(json.dumps(envelope) + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="sealed map manifest_id mismatch"):
        BUILDER.build(args)

    assert args.zip_path.read_bytes() == original_zip


def test_remux_strips_audio_and_metadata_and_is_byte_deterministic(tmp_path: Path) -> None:
    assert shutil.which("ffmpeg"), "ffmpeg is required by the release builder"
    assert shutil.which("ffprobe"), "ffprobe is required by the release builder"
    source = tmp_path / "source-with-audio.mp4"
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-nostdin",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "color=c=black:s=32x32:r=10:d=0.4",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:sample_rate=8000:duration=0.4",
        "-c:v",
        "mpeg4",
        "-c:a",
        "aac",
        "-metadata",
        "title=PRIVATE_CONDITION",
        "-shortest",
        str(source),
    ]
    subprocess.run(command, check=True, capture_output=True)
    stream_probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "json",
            str(source),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert any(
        stream["codec_type"] == "audio"
        for stream in json.loads(stream_probe.stdout)["streams"]
    )

    first = tmp_path / "first.mp4"
    second = tmp_path / "second.mp4"
    first_sha = BUILDER._remux_video_only(source, first)
    second_sha = BUILDER._remux_video_only(source, second)
    assert first_sha == second_sha == _sha256(first) == _sha256(second)
    BUILDER._verify_video_only(first)
    format_probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format_tags",
            "-of",
            "json",
            str(first),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    tags = json.loads(format_probe.stdout).get("format", {}).get("tags", {})
    assert tags.get("title") != "PRIVATE_CONDITION"


def test_template_markers_are_required_exactly_once(tmp_path: Path) -> None:
    template = tmp_path / "bad.html"
    template.write_text("__ROUND1_MANIFEST_JSON__", encoding="utf-8")
    with pytest.raises(RuntimeError, match="each manifest marker exactly once"):
        BUILDER._render_html(template, b"{}\n", "a" * 64)
