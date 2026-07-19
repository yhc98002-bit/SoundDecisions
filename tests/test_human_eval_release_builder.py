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
PACK_DIR = ROOT / "results" / "human_eval_pack"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_audio_registry_default_is_private_untracked_and_ignored() -> None:
    private_registry = PACK_DIR / "private" / BUILDER.AUDIO_MEDIA_REGISTRY_FILENAME
    relative = private_registry.relative_to(ROOT)
    assert private_registry.parent == PACK_DIR / "private"
    assert (private_registry.parent / ".gitignore").read_text(encoding="utf-8") == (
        "*\n!.gitignore\n"
    )
    assert subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(relative)],
        cwd=ROOT,
        capture_output=True,
    ).returncode != 0
    assert subprocess.run(
        ["git", "check-ignore", "-q", str(relative)], cwd=ROOT
    ).returncode == 0
    assert not private_registry.exists() or subprocess.run(
        ["git", "check-ignore", "-q", str(relative)], cwd=ROOT
    ).returncode == 0


def test_committed_v4_registry_matches_tracked_release_contract() -> None:
    registry = json.loads(
        (PACK_DIR / "ROUND1_V4_SHA256SUMS.json").read_text(encoding="utf-8")
    )
    record = json.loads((PACK_DIR / "ROUND1_V4_RELEASE.json").read_text(encoding="utf-8"))

    assert registry["release_id"] == record["release_id"] == "round1_curation_v4"
    assert record["status"] == "CURATION_AUTHORIZED"
    assert (
        registry["files"]["results/human_eval_pack/releases/round1_curation_v4.zip"]
        == record["zip"]["sha256"]
    )
    assert (
        registry["files"]["results/human_eval_pack/round1_v4_blinded_items.json"]
        == record["manifest"]["sha256"]
    )
    assert (
        registry["files"][
            "results/human_eval_pack/round1_v4_unblinding_map.sealed.json"
        ]
        == record["sealed_unblinding_map"]["sha256"]
    )
    assert record["audio_media_registry"]["storage"] == "operator_private"
    assert record["audio_media_registry"]["item_count"] == 82
    assert not any("audio_media_registry" in path for path in registry["files"])

    for relative, expected in registry["files"].items():
        path = ROOT / relative
        if relative.endswith("round1_curation_v4.zip") and not path.exists():
            continue  # Large reviewer media are intentionally absent from GitHub.
        assert path.is_file(), relative
        assert _sha256(path) == expected, relative

    private_registry = PACK_DIR / "private" / BUILDER.AUDIO_MEDIA_REGISTRY_FILENAME
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(private_registry.relative_to(ROOT))],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    ignored = subprocess.run(
        ["git", "check-ignore", "-q", str(private_registry.relative_to(ROOT))],
        cwd=ROOT,
    )
    assert tracked.returncode != 0
    assert not private_registry.exists() or ignored.returncode == 0
    if private_registry.exists():
        assert _sha256(private_registry) == record["audio_media_registry"]["sha256"]


def test_retirement_registry_preserves_tracked_audit_hashes() -> None:
    registry = json.loads((PACK_DIR / "RETIRED_RELEASES.json").read_text(encoding="utf-8"))

    assert registry["current_release"] == "round1_curation_v4"
    assert {row["release_id"] for row in registry["retired"]} == {
        "round1_curation_v1",
        "round1_curation_v2",
        "round1_curation_v3",
    }
    for row in registry["retired"]:
        assert row["status"] == "RETIRED_BEFORE_DELIVERY"
        builder = PACK_DIR / row["builder_record"]["path"]
        assert _sha256(builder) == row["builder_record"]["sha256"]
        archive = PACK_DIR / row["zip"]["path"]
        if archive.exists():
            assert _sha256(archive) == row["zip"]["sha256"]


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
    sealed_map = tmp_path / "round1_v4_unblinding_map.sealed.json"
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
        audio_media_registry_schema=SOURCE_DIR / "audio_media_registry.schema.json",
        release_dir=out / "release",
        zip_path=out / "release.zip",
        public_manifest=out / "round1_v4_blinded_items.json",
        audio_media_registry=out / "private" / "round1_v4_audio_media_registry.json",
        release_record=out / "ROUND1_V4_RELEASE.json",
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
    assert record["audio_media_registry"]["sha256"] == _sha256(
        args.audio_media_registry
    )
    assert record["audio_media_registry"] == {
        "storage": "operator_private",
        "sha256": _sha256(args.audio_media_registry),
        "item_count": 82,
    }
    assert args.audio_media_registry.stat().st_mode & 0o077 == 0
    audio_registry = json.loads(args.audio_media_registry.read_text(encoding="utf-8"))
    assert set(audio_registry) == {
        "schema_version",
        "manifest_id",
        "manifest_sha256",
        "items",
    }
    assert audio_registry["manifest_id"] == BUILDER.MANIFEST_ID
    assert audio_registry["manifest_sha256"] == _sha256(args.public_manifest)
    assert len(audio_registry["items"]) == 82
    assert all(set(row) == {"blind_id", "media_path", "sha256"} for row in audio_registry["items"])
    assert _all_json_keys(audio_registry).isdisjoint(
        {"source_clip_id", "source_path", "caption", "model", "condition", "cfg", "seed"}
    )
    source_sha256 = _sha256(Path(next(csv.DictReader(args.clips_index.open()))["path"]))
    assert all(row["sha256"] == source_sha256 for row in audio_registry["items"])
    envelope = json.loads(args.sealed_map.read_text(encoding="utf-8"))
    private_mapping = BUILDER._decrypt_sealed_envelope(envelope, args.key_file)
    assert private_mapping["manifest_id"] == BUILDER.MANIFEST_ID
    assert private_mapping["manifest_sha256"] == _sha256(args.public_manifest)
    assert {row["source_clip_id"] for row in private_mapping["items"]} == set(source_ids)
    assert len(private_mapping["items"]) == 82
    assert (
        private_mapping["source_contract_sha256"]["audio_media_registry"]
        == _sha256(args.audio_media_registry)
    )
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
    first_registry_bytes = args.audio_media_registry.read_bytes()
    BUILDER.build(args)
    assert args.sealed_map.read_bytes() == first_sealed_bytes
    assert args.audio_media_registry.read_bytes() == first_registry_bytes


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
    second_args.sealed_map = tmp_path / "second" / "round1_v4_unblinding_map.sealed.json"
    second_args.public_manifest = tmp_path / "second" / "round1_v4_blinded_items.json"
    second_args.audio_media_registry = (
        tmp_path / "second" / "private" / "round1_v4_audio_media_registry.json"
    )
    second_args.release_record = tmp_path / "second" / "ROUND1_V4_RELEASE.json"
    second = BUILDER.build(second_args)

    assert _sha256(second_args.zip_path) == first_zip_sha
    assert first["zip"]["sha256"] == second["zip"]["sha256"] == first_zip_sha
    assert args.audio_media_registry.read_bytes() == second_args.audio_media_registry.read_bytes()
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
        assert all("audio_media_registry" not in name for name in names)


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
    assert not args.audio_media_registry.exists()
    assert not args.release_record.exists()


def test_contract_file_mutation_during_remux_publishes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, _ = _fixture(tmp_path)
    monkeypatch.setattr(BUILDER, "_probe_video", lambda _: (24.0, 10.0))
    monkeypatch.setattr(BUILDER, "SEALED_MAP_ITERATIONS", 1_000)
    mutated = False

    def remux(source: Path, destination: Path) -> str:
        nonlocal mutated
        if not mutated:
            args.instructions.write_text("mutated during remux\n", encoding="utf-8")
            mutated = True
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"video-only-fixture\n" + source.read_bytes())
        return _sha256(destination)

    monkeypatch.setattr(BUILDER, "_remux_video_only", remux)

    with pytest.raises(RuntimeError, match="contract file changed during build.*instructions"):
        BUILDER.build(args)

    assert not args.release_dir.exists()
    assert not args.zip_path.exists()
    assert not args.sealed_map.exists()
    assert not args.public_manifest.exists()
    assert not args.audio_media_registry.exists()
    assert not args.release_record.exists()


def test_source_video_mutation_during_remux_publishes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, _ = _fixture(tmp_path)
    monkeypatch.setattr(BUILDER, "_probe_video", lambda _: (24.0, 10.0))
    source = Path(next(csv.DictReader(args.clips_index.open()))["path"])

    def remux(source_path: Path, destination: Path) -> str:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"video-only-fixture\n" + source_path.read_bytes())
        source_path.write_bytes(b"mutated source media")
        return _sha256(destination)

    monkeypatch.setattr(BUILDER, "_remux_video_only", remux)

    with pytest.raises(RuntimeError, match="source video changed during remux"):
        BUILDER.build(args)

    assert source.read_bytes() == b"mutated source media"
    assert not args.release_dir.exists()
    assert not args.zip_path.exists()
    assert not args.sealed_map.exists()
    assert not args.audio_media_registry.exists()


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


def test_existing_audio_registry_is_never_overwritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, _ = _fixture(tmp_path)
    _install_fake_video_pipeline(monkeypatch)
    BUILDER.build(args)
    args.audio_media_registry.write_text("tampered registry\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="non-identical audio-media registry"):
        BUILDER.build(args)

    assert args.audio_media_registry.read_text(encoding="utf-8") == "tampered registry\n"


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
        BUILDER._render_html(template.read_bytes(), b"{}\n", "a" * 64)


@pytest.mark.parametrize("kind", ["file", "directory"])
def test_racing_nonidentical_destination_is_never_overwritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    destination = tmp_path / "published"
    staged = tmp_path / "staged"
    if kind == "file":
        staged.write_bytes(b"builder-content")
    else:
        staged.mkdir()
        (staged / "payload.txt").write_bytes(b"builder-content")

    def racing_publish(_source: Path, raced_destination: Path) -> bool:
        if kind == "file":
            raced_destination.write_bytes(b"racer-content")
        else:
            raced_destination.mkdir()
            (raced_destination / "payload.txt").write_bytes(b"racer-content")
        raise FileExistsError(raced_destination)

    monkeypatch.setattr(BUILDER, "_rename_noreplace", racing_publish)
    publisher = (
        BUILDER._publish_immutable_file
        if kind == "file"
        else BUILDER._publish_immutable_directory
    )
    with pytest.raises(RuntimeError, match="refusing to overwrite non-identical"):
        publisher(staged, destination)

    if kind == "file":
        assert destination.read_bytes() == b"racer-content"
    else:
        assert (destination / "payload.txt").read_bytes() == b"racer-content"
    assert staged.exists()


@pytest.mark.parametrize("kind", ["file", "directory"])
def test_creation_exclusive_fallback_publishes_without_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    destination = tmp_path / "published"
    staged = tmp_path / "staged"
    if kind == "file":
        staged.write_bytes(b"builder-content")
    else:
        staged.mkdir()
        (staged / "payload.txt").write_bytes(b"builder-content")
    monkeypatch.setattr(BUILDER, "_rename_noreplace", lambda *_: False)

    if kind == "file":
        BUILDER._publish_immutable_file(staged, destination)
        assert destination.read_bytes() == b"builder-content"
    else:
        BUILDER._publish_immutable_directory(staged, destination)
        assert (destination / "payload.txt").read_bytes() == b"builder-content"
    assert not staged.exists()


def test_partial_publication_has_no_commit_record_and_rerun_completes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, _ = _fixture(tmp_path)
    _install_fake_video_pipeline(monkeypatch)
    checkpoints: list[str] = []

    def fail_before_record(label: str) -> None:
        checkpoints.append(label)
        if label == "release_record":
            raise RuntimeError("injected publication failure")

    monkeypatch.setattr(BUILDER, "_publication_checkpoint", fail_before_record)
    with pytest.raises(RuntimeError, match="injected publication failure"):
        BUILDER.build(args)

    assert checkpoints[-2:] == ["public_manifest", "release_record"]
    assert args.public_manifest.is_file()
    assert args.release_dir.is_dir()
    assert args.zip_path.is_file()
    assert args.sealed_map.is_file()
    assert args.audio_media_registry.is_file()
    assert not args.release_record.exists()

    artifact_hashes = {
        path: _sha256(path)
        for path in (
            args.public_manifest,
            args.zip_path,
            args.sealed_map,
            args.audio_media_registry,
        )
    }
    monkeypatch.setattr(BUILDER, "_publication_checkpoint", lambda _label: None)
    record = BUILDER.build(args)
    assert args.release_record.is_file()
    assert json.loads(args.release_record.read_text(encoding="utf-8")) == record
    assert all(_sha256(path) == expected for path, expected in artifact_hashes.items())


def test_identical_rerun_keeps_every_published_byte(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, _ = _fixture(tmp_path)
    _install_fake_video_pipeline(monkeypatch)
    first = BUILDER.build(args)
    paths = (
        args.public_manifest,
        args.release_record,
        args.zip_path,
        args.sealed_map,
        args.audio_media_registry,
    )
    first_bytes = {path: path.read_bytes() for path in paths}
    first_directory_digest = BUILDER._directory_digest(args.release_dir)

    second = BUILDER.build(args)

    assert second == first
    assert {path: path.read_bytes() for path in paths} == first_bytes
    assert BUILDER._directory_digest(args.release_dir) == first_directory_digest
