"""Fail-closed and blinding-boundary tests for the human-eval pack builder."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
BUILD_PATH = ROOT / "results" / "human_eval_pack" / "build_pack.py"
BUILD_SPEC = importlib.util.spec_from_file_location("human_eval_pack_builder", BUILD_PATH)
assert BUILD_SPEC is not None and BUILD_SPEC.loader is not None
BUILDER = importlib.util.module_from_spec(BUILD_SPEC)
BUILD_SPEC.loader.exec_module(BUILDER)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_media_copy_replaces_a_hardlink_with_an_independent_atomic_copy(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    destination = tmp_path / "media" / "HEV2-0123456789AB.mp4"
    source.write_bytes(b"blinded-media-fixture")
    destination.parent.mkdir()
    os.link(source, destination)
    assert os.path.samefile(source, destination)

    BUILDER._copy_media(source, destination, _sha256(source))

    assert destination.read_bytes() == source.read_bytes()
    assert _sha256(destination) == _sha256(source)
    assert not os.path.samefile(source, destination)
    assert destination.stat().st_ino != source.stat().st_ino


def test_failed_preflight_appends_redacted_flag_without_publishing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    anchor_ids = [f"A{index:03d}" for index in range(29)] + ["SECRET_CLIP_29"]
    two_event_ids = [f"T{index:03d}" for index in range(60)]

    anchor_csv = tmp_path / "anchors.csv"
    with anchor_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["key"])
        writer.writeheader()
        writer.writerows({"key": clip_id} for clip_id in anchor_ids)

    two_event_json = tmp_path / "two_event.json"
    two_event_json.write_text(json.dumps({"clips": two_event_ids}), encoding="utf-8")
    pilot_manifest = tmp_path / "pilot.json"
    pilot_manifest.write_text(json.dumps({"clips": []}), encoding="utf-8")

    valid_source = tmp_path / "valid.mp4"
    valid_source.write_bytes(b"valid-source")
    secret_source = tmp_path / "private" / "SECRET_SOURCE_PATH" / "SECRET_CLIP_29.mp4"
    secret_source.parent.mkdir(parents=True)
    secret_source.write_bytes(b"mismatched-source")
    actual_secret_sha256 = _sha256(secret_source)
    claimed_secret_sha256 = "f" * 64

    clips_index = tmp_path / "clips_index.csv"
    fields = ["key", "path", "sha256", "caption", "ucs_category", "source_type"]
    with clips_index.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for clip_id in [*anchor_ids, *two_event_ids]:
            is_secret = clip_id == "SECRET_CLIP_29"
            source = secret_source if is_secret else valid_source
            writer.writerow(
                {
                    "key": clip_id,
                    "path": source,
                    "sha256": claimed_secret_sha256 if is_secret else _sha256(source),
                    "caption": "private caption",
                    "ucs_category": "private category",
                    "source_type": "private source",
                }
            )

    key_file = tmp_path / "human_eval.key"
    key_file.write_text("01" * 32 + "\n", encoding="ascii")
    key_file.chmod(0o600)

    out_dir = tmp_path / "pack"
    media_dir = out_dir / "media"
    media_dir.mkdir(parents=True)
    sentinels = {
        "rate.html": b"prior authorized instrument",
        "blinded_items.json": b"prior manifest",
        "unblinding_map.sealed.json": b"prior sealed map",
        "media/prior.mp4": b"prior media",
    }
    for relative_path, content in sentinels.items():
        path = out_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    prior_flag = {"id": "PRIOR_FLAG", "status": "INCOMPLETE_ARTIFACTS"}
    (out_dir / "FLAGS.json").write_text(json.dumps([prior_flag]) + "\n", encoding="utf-8")

    monkeypatch.setattr(BUILDER, "_probe_video", lambda _: (24.0, 1.0))
    args = argparse.Namespace(
        anchor_csv=anchor_csv,
        two_event_json=two_event_json,
        clips_index=clips_index,
        pilot_manifest=pilot_manifest,
        out_dir=out_dir,
        key_file=key_file,
    )

    with pytest.raises(RuntimeError, match="media preflight failed"):
        BUILDER.build(args)

    for relative_path, content in sentinels.items():
        assert (out_dir / relative_path).read_bytes() == content
    assert sorted(path.name for path in media_dir.iterdir()) == ["prior.mp4"]

    flag_text = (out_dir / "FLAGS.json").read_text(encoding="utf-8")
    flags = json.loads(flag_text)
    assert flags[0] == prior_flag
    failure = flags[-1]
    assert failure["id"] == "HUMAN_EVAL_MEDIA_PREFLIGHT_FAILED"
    assert failure["resolution"]
    assert failure["evidence"] == [
        {
            "blind_id": BUILDER._blind_id("SECRET_CLIP_29", key_file.read_bytes()),
            "reason": "video_checksum_mismatch",
            "checksum_verified": False,
        }
    ]
    for private_value in (
        "SECRET_CLIP_29",
        "SECRET_SOURCE_PATH",
        str(secret_source),
        actual_secret_sha256,
        claimed_secret_sha256,
    ):
        assert private_value not in flag_text
    assert set(failure["evidence"][0]) == {"blind_id", "reason", "checksum_verified"}
