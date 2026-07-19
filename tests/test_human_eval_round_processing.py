"""Round-1 curation to Round-2 Presence package contract tests."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess

import pytest
from jsonschema import Draft202012Validator, FormatChecker


ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "results" / "human_eval_pack"


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CURATE = _module("human_curate_round1", PACK / "curate_round1.py")
BUILD = _module("human_build_round2", PACK / "build_round2.py")
SCORE = _module("human_score_round2", PACK / "score_round2.py")
AC1 = _module("human_score_round2_ac1", PACK / "score_round2_ac1.py")


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _round1_fixture(tmp_path: Path) -> tuple[Path, Path, dict, dict]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    ids = [f"HEV2-{index:012X}" for index in range(1, 6)]
    tasks = [
        ["anchor_curation"],
        ["anchor_curation"],
        ["two_event_curation"],
        ["two_event_curation"],
        ["anchor_curation", "two_event_curation"],
    ]
    manifest = {
        "schema_version": "sounddecisions-human-curation-items-v1-1.3",
        "instrument_version": "human-eval-round1-curation-1.3",
        "manifest_id": "round1-test",
        "status": "CURATION_AUTHORIZED",
        "default_fps": 25.0,
        "counts": {
            "anchor_curation": 3,
            "two_event_curation": 3,
            "total_tasks": 6,
            "unique_videos": 5,
        },
        "items": [
            {
                "blind_id": blind_id,
                "tasks": item_tasks,
                "media_path": f"media/{blind_id}.mp4",
                "fps": 25.0,
                "duration_s": 10.0,
                "candidate_caption": f"  fixed caption {index}  ",
            }
            for index, (blind_id, item_tasks) in enumerate(zip(ids, tasks), start=1)
        ],
    }
    manifest["items"][2]["candidate_caption"] = (
        "Audio-bearing caption: a loud boom and two impacts are audible."
    )
    manifest["items"][4]["candidate_caption"] = (
        "Audio-bearing caption: footsteps and background music can be heard."
    )
    manifest_path = tmp_path / "round1_manifest.json"
    _write_json(manifest_path, manifest)

    ratings = {
        "schema_version": "sounddecisions-human-curation-ratings-v1-1.3",
        "instrument_version": "human-eval-round1-curation-1.3",
        "manifest_id": manifest["manifest_id"],
        "manifest_sha256": _sha(manifest_path),
        "rater_id": "curator-1",
        "started_at": "2026-07-19T00:00:00Z",
        "exported_at": "2026-07-19T01:00:00Z",
        "item_order": ids,
        "ratings": [
            {
                "blind_id": ids[0], "tasks": tasks[0], "completed": True,
                "anchor_curation": {"status": "marked", "start_s": 1.0, "end_s": 1.5, "event_description": "  curator event one  ", "note": ""},
            },
            {
                "blind_id": ids[1], "tasks": tasks[1], "completed": True,
                "anchor_curation": {"status": "too_uncertain", "start_s": 2.0, "end_s": None, "event_description": "", "note": "occluded"},
            },
            {
                "blind_id": ids[2], "tasks": tasks[2], "completed": True,
                "two_event_curation": {
                    "verdict": "confirm",
                    "event_1": {"description": "  first contact  ", "start_s": 2.0, "end_s": 2.2},
                    "event_2": {"description": "second contact", "start_s": 4.0, "end_s": 4.4},
                    "note": "",
                },
            },
            {
                "blind_id": ids[3], "tasks": tasks[3], "completed": True,
                "two_event_curation": {"verdict": "reject", "event_1": None, "event_2": None, "note": "one event"},
            },
            {
                "blind_id": ids[4], "tasks": tasks[4], "completed": True,
                "anchor_curation": {"status": "marked", "start_s": 0.5, "end_s": 0.7, "event_description": "overlap anchor", "note": ""},
                "two_event_curation": {
                    "verdict": "confirm",
                    "event_1": {"description": "pair one", "start_s": 3.0, "end_s": 3.2},
                    "event_2": {"description": "pair two", "start_s": 6.0, "end_s": 6.2},
                    "note": "distinct events on overlap",
                },
            },
        ],
    }
    ratings_path = tmp_path / "ratings_curator-1.json"
    _write_json(ratings_path, ratings)
    return manifest_path, ratings_path, manifest, ratings


def _catalog(tmp_path: Path) -> tuple[Path, dict]:
    manifest_path, ratings_path, _, _ = _round1_fixture(tmp_path)
    output = tmp_path / "event_catalog.json"
    catalog = CURATE.curate(manifest_path, ratings_path, output)
    return output, catalog


def _audio_registry(
    tmp_path: Path, catalog: dict, media_root: Path
) -> Path:
    path = tmp_path / "round1_v4_audio_media_registry.json"
    payload = {
        "schema_version": "sounddecisions-human-curation-audio-media-registry-v4-1.0",
        "manifest_id": catalog["source_manifest_id"],
        "manifest_sha256": catalog["source_manifest_sha256"],
        "items": [
            {
                "blind_id": item["blind_id"],
                "media_path": item["media_path"],
                "sha256": (
                    _sha(media_root / item["media_path"])
                    if (media_root / item["media_path"]).is_file()
                    else "0" * 64
                ),
            }
            for item in catalog["items"]
        ],
    }
    _write_json(path, payload)
    return path


def _round1_release_record(
    tmp_path: Path, catalog: dict, audio_registry: Path
) -> Path:
    path = tmp_path / "ROUND1_V4_RELEASE.json"
    registry_sha256 = _sha(audio_registry)
    manifest_sha256 = catalog["source_manifest_sha256"]
    payload = {
        "schema_version": "sounddecisions-human-curation-release-v4",
        "release_id": "round1_curation_v4",
        "status": "CURATION_AUTHORIZED",
        "manifest": {
            "path": "round1_v4_blinded_items.json",
            "sha256": manifest_sha256,
        },
        "audio_media_registry": {
            "storage": "operator_private",
            "sha256": registry_sha256,
            "item_count": len(json.loads(audio_registry.read_text(encoding="utf-8"))["items"]),
        },
        "source_contract_sha256": {
            "public_manifest": manifest_sha256,
            "audio_media_registry": registry_sha256,
        },
    }
    _write_json(path, payload)
    return path


def _fake_remux(
    source: Path, target: Path, expected_source_sha256: str
) -> str:
    if _sha(source) != expected_source_sha256:
        raise ValueError(f"audio-media registry checksum mismatch: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"sanitized:" + source.read_bytes())
    return _sha(target)


def test_round1_exact_join_emits_fixed_events_and_explicit_noneligible_states(tmp_path: Path) -> None:
    output, catalog = _catalog(tmp_path)
    schema = json.loads((PACK / "event_catalog.schema.json").read_text(encoding="utf-8"))
    Draft202012Validator(schema, format_checker=FormatChecker()).validate(catalog)

    assert output.is_file()
    assert catalog["analysis_scope"] == "single_curator_event_definition_not_agreement_evidence"
    assert catalog["counts"] == {
        "items_total": 5,
        "anchor_eligible": 2,
        "anchor_unresolved": 1,
        "pair_eligible": 2,
        "pair_rejected": 1,
        "events_eligible": 6,
    }
    by_id = {item["blind_id"]: item for item in catalog["items"]}
    ids = sorted(by_id)
    assert by_id[ids[0]]["events"][0] == {
        "event_id": f"{ids[0]}-A1",
        "description": "curator event one",
        "anchor": {"start_s": 1.0, "end_s": 1.5},
        "event_source": "anchor_curation",
    }
    assert by_id[ids[1]]["anchor_curation"]["status"] == "unresolved"
    assert by_id[ids[1]]["events"] == []
    assert by_id[ids[3]]["two_event_curation"]["status"] == "rejected"
    assert by_id[ids[3]]["events"] == []
    assert [event["event_id"] for event in by_id[ids[4]]["events"]] == [
        f"{ids[4]}-A1", f"{ids[4]}-P1", f"{ids[4]}-P2"
    ]
    assert by_id[ids[4]]["events"][0]["description"] == "overlap anchor"
    assert by_id[ids[4]]["events"][1]["description"] == "pair one"
    pair_captions = {
        item["candidate_caption"].strip()
        for item in _round1_fixture(tmp_path / "caption_check")[2]["items"]
        if "two_event_curation" in item["tasks"]
    }
    pair_descriptions = {
        event["description"]
        for item in catalog["items"]
        for event in item["events"]
        if event["event_id"].endswith(("-P1", "-P2"))
    }
    assert pair_descriptions == {"first contact", "second contact", "pair one", "pair two"}
    assert pair_descriptions.isdisjoint(pair_captions)


@pytest.mark.parametrize(
    "case,match",
    [
        ("id_prefix", "event ID prefix/suffix"),
        ("media_path", "media_path must equal"),
        ("suffix_source", "suffix does not match event_source"),
        ("description", "description"),
        ("bounds", "outside the video bounds"),
        ("eligible_pair", "eligible pair must contain P1 and P2"),
        ("unresolved", "unresolved anchor must have a null anchor"),
        ("rejected_pair", "rejected pair must contain no events"),
        ("flattened", "flattened events do not equal"),
        ("counts", "counts do not match"),
    ],
)
def test_round2_catalog_semantics_fail_closed_on_cross_field_mutations(
    tmp_path: Path, case: str, match: str
) -> None:
    _, catalog = _catalog(tmp_path)
    broken = copy.deepcopy(catalog)
    if case == "id_prefix":
        broken["items"][0]["events"][0]["event_id"] = "HEV2-000000000005-A1"
    elif case == "media_path":
        broken["items"][0]["media_path"] = "media/HEV2-000000000005.mp4"
    elif case == "suffix_source":
        broken["items"][0]["events"][0]["event_source"] = "two_event_curation"
    elif case == "description":
        broken["items"][0]["anchor_curation"]["description"] = ""
        broken["items"][0]["events"][0]["description"] = ""
    elif case == "bounds":
        broken["items"][0]["anchor_curation"]["anchor"]["end_s"] = 11.0
        broken["items"][0]["events"][0]["anchor"]["end_s"] = 11.0
    elif case == "eligible_pair":
        broken["items"][2]["two_event_curation"]["events"].pop()
    elif case == "unresolved":
        broken["items"][1]["anchor_curation"]["anchor"] = {
            "start_s": 2.0,
            "end_s": 2.5,
        }
    elif case == "rejected_pair":
        broken["items"][3]["two_event_curation"]["events"] = [
            copy.deepcopy(broken["items"][2]["two_event_curation"]["events"][0])
        ]
    elif case == "flattened":
        broken["items"][4]["events"].pop()
    elif case == "counts":
        broken["counts"]["events_eligible"] += 1
    with pytest.raises(ValueError, match=match):
        BUILD.build_manifest(broken, "a" * 64, "b" * 64)


@pytest.mark.parametrize("mutation,match", [
    (lambda payload: payload["ratings"].pop(), "ratings item set"),
    (lambda payload: payload["ratings"][0].update(completed=False), "curation is incomplete"),
    (lambda payload: payload["ratings"][0]["anchor_curation"].update(end_s=11.0), "outside the video"),
])
def test_round1_fails_closed_on_missing_incomplete_or_invalid_data(
    tmp_path: Path, mutation, match: str
) -> None:
    manifest_path, ratings_path, manifest, ratings = _round1_fixture(tmp_path)
    broken = copy.deepcopy(ratings)
    mutation(broken)
    broken_path = tmp_path / "broken.json"
    _write_json(broken_path, broken)
    broken["manifest_sha256"] = _sha(manifest_path)
    _write_json(broken_path, broken)
    with pytest.raises(ValueError, match=match):
        CURATE.curate(manifest_path, broken_path, tmp_path / "must_not_exist.json")
    assert not (tmp_path / "must_not_exist.json").exists()


def test_round2_builder_packages_only_fixed_events_and_no_condition_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    catalog_path, catalog = _catalog(tmp_path)
    media_root = tmp_path / "round1_release"
    for item in catalog["items"]:
        media = media_root / item["media_path"]
        media.parent.mkdir(parents=True, exist_ok=True)
        media.write_bytes(f"video:{item['blind_id']}".encode())
    audio_registry = _audio_registry(tmp_path, catalog, media_root)
    release_record = _round1_release_record(tmp_path, catalog, audio_registry)
    output = tmp_path / "round2_release"
    monkeypatch.setattr(BUILD, "_remux_sanitized_media", _fake_remux)
    manifest = BUILD.build_package(
        catalog_path, output, media_root, audio_registry, release_record
    )

    assert manifest["status"] == "RATING_AUTHORIZED_SINGLE_RATER"
    assert manifest["counts"] == {"events": 6, "unique_videos": 3}
    assert len({item["event_id"] for item in manifest["items"]}) == 6
    html = (output / "rate.html").read_text(encoding="utf-8")
    assert "Fixed target event" in html and "Fixed visual anchor" in html
    assert "target_present" in html and "unrelated_background" in html
    assert "__ROUND2_MANIFEST_" not in html
    assert "fetch(" not in html and "https://" not in html
    assert "curator-1" not in html
    assert 'unrelated_background: null' in html
    assert '<input type="radio" name="background" value="true">' in html
    assert '<input type="radio" name="background" value="false">' in html
    assert "condition" not in json.dumps(manifest).lower()
    assert manifest["source_audio_media_registry_sha256"] == _sha(audio_registry)
    assert manifest["source_round1_release_id"] == "round1_curation_v4"
    assert manifest["source_round1_release_record_sha256"] == _sha(release_record)
    assert manifest["source_round1_manifest_sha256"] == catalog["source_manifest_sha256"]
    assert "audio-media registry binding" in html
    assert "source_audio_media_registry_sha256" in html
    packaged_instructions = (output / "INSTRUCTIONS.md").read_text(encoding="utf-8")
    assert "--audio-media-registry" in packaged_instructions
    assert "results/human_eval_pack/private/round1_v4_audio_media_registry.json" in packaged_instructions
    assert "--round1-release-record" in packaged_instructions
    assert "COMPLETE.json" in packaged_instructions
    assert (output / "INSTRUCTIONS.md").is_file()
    sums = json.loads((output / "SHA256SUMS.json").read_text(encoding="utf-8"))
    for relative, digest in sums.items():
        assert _sha(output / relative) == digest
    completion = json.loads((output / "COMPLETE.json").read_text(encoding="utf-8"))
    assert completion["media_files"] == 3
    assert sums["COMPLETE.json"] == _sha(output / "COMPLETE.json")
    first_media = manifest["items"][0]["media_path"]
    assert (media_root / first_media).stat().st_ino != (output / first_media).stat().st_ino


def _round2_export(manifest: dict, manifest_path: Path, rater_id: str) -> dict:
    event_ids = [item["event_id"] for item in manifest["items"]]
    ratings = []
    for index, event_id in enumerate(event_ids):
        completed = index < 3
        ratings.append({
            "event_id": event_id,
            "completed": completed,
            "presence": {
                "verdict": ["target_present", "absent", "uncertain"][index] if completed else None,
                "unrelated_background": index == 0 if completed else None,
                "note": "",
            },
        })
    return {
        "schema_version": "sounddecisions-human-presence-ratings-v1-1.0",
        "instrument_version": manifest["instrument_version"],
        "manifest_id": manifest["manifest_id"],
        "manifest_sha256": _sha(manifest_path),
        "rater_id": rater_id,
        "started_at": "2026-07-19T02:00:00Z",
        "exported_at": "2026-07-19T03:00:00Z",
        "item_order": event_ids,
        "ratings": ratings,
    }


def test_round2_rejects_curator_and_emits_descriptive_single_rater_summary(tmp_path: Path) -> None:
    catalog_path, catalog = _catalog(tmp_path)
    manifest = BUILD.build_manifest(catalog, _sha(catalog_path), "b" * 64)
    manifest_path = tmp_path / "round2_manifest.json"
    _write_json(manifest_path, manifest)

    same_rater = _round2_export(manifest, manifest_path, "CURATOR-1")
    with pytest.raises(ValueError, match="different from the Round-1 curator"):
        SCORE.validate_export(manifest, same_rater, _sha(manifest_path))

    ratings = _round2_export(manifest, manifest_path, "presence-rater-2")
    summary = SCORE.summarize(
        manifest,
        ratings,
        manifest_sha256=_sha(manifest_path),
        export_sha256="a" * 64,
    )
    Draft202012Validator(
        json.loads((PACK / "release_src" / "round2_summary.schema.json").read_text()),
        format_checker=FormatChecker(),
    ).validate(summary)
    assert summary["analysis_scope"] == "single_rater_descriptive_only"
    assert summary["counts"] == {"events_total": 6, "completed": 3, "incomplete": 3}
    assert summary["coverage_rate"] == pytest.approx(0.5)
    assert summary["verdict_counts"] == {"target_present": 1, "absent": 1, "uncertain": 1}
    assert summary["unrelated_background_counts"] == {"true": 1, "false": 2}
    assert "agreement" not in summary and "ac1" not in summary
    assert "not computed" in summary["limitation"]


def test_round2_missing_media_aborts_without_partial_release(tmp_path: Path) -> None:
    catalog_path, catalog = _catalog(tmp_path)
    media_root = tmp_path / "empty_media_root"
    audio_registry = _audio_registry(tmp_path, catalog, media_root)
    release_record = _round1_release_record(tmp_path, catalog, audio_registry)
    output = tmp_path / "round2_release"
    with pytest.raises(FileNotFoundError, match="required blinded video is missing"):
        BUILD.build_package(
            catalog_path, output, media_root, audio_registry, release_record
        )
    assert not output.exists()


def test_round2_rejects_silent_media_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    catalog_path, catalog = _catalog(tmp_path)
    media_root = tmp_path / "silent_round1_release"
    for item in catalog["items"]:
        media = media_root / item["media_path"]
        media.parent.mkdir(parents=True, exist_ok=True)
        media.write_bytes(b"silent-video")
    audio_registry = _audio_registry(tmp_path, catalog, media_root)
    release_record = _round1_release_record(tmp_path, catalog, audio_registry)
    monkeypatch.setattr(
        BUILD,
        "_remux_sanitized_media",
        lambda source, target, expected: (_ for _ in ()).throw(
            ValueError(
                f"Round-2 source must contain video and audio streams: {source} "
                "(video=1, audio=0)"
            )
        ),
    )
    output = tmp_path / "must_not_exist"
    with pytest.raises(ValueError, match="must contain video and audio streams"):
        BUILD.build_package(
            catalog_path, output, media_root, audio_registry, release_record
        )
    assert not output.exists()


def test_round2_rejects_media_bytes_not_bound_by_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    catalog_path, catalog = _catalog(tmp_path)
    media_root = tmp_path / "audio_media"
    for item in catalog["items"]:
        media = media_root / item["media_path"]
        media.parent.mkdir(parents=True, exist_ok=True)
        media.write_bytes(f"original:{item['blind_id']}".encode())
    audio_registry = _audio_registry(tmp_path, catalog, media_root)
    release_record = _round1_release_record(tmp_path, catalog, audio_registry)
    first_eligible = next(item for item in catalog["items"] if item["events"])
    (media_root / first_eligible["media_path"]).write_bytes(b"substituted audio media")
    monkeypatch.setattr(BUILD, "_remux_sanitized_media", _fake_remux)
    output = tmp_path / "must_not_exist"

    with pytest.raises(ValueError, match="audio-media registry checksum mismatch"):
        BUILD.build_package(
            catalog_path, output, media_root, audio_registry, release_record
        )

    assert not output.exists()


def test_round2_accepts_exact_round1_v4_producer_record_shape(tmp_path: Path) -> None:
    catalog_path, catalog = _catalog(tmp_path)
    registry_path = _audio_registry(tmp_path, catalog, tmp_path / "audio_media")
    release_record_path = _round1_release_record(tmp_path, catalog, registry_path)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    release_record = json.loads(release_record_path.read_text(encoding="utf-8"))

    assert set(release_record["audio_media_registry"]) == {
        "storage", "sha256", "item_count"
    }
    assert release_record["audio_media_registry"]["storage"] == "operator_private"
    assert release_record["audio_media_registry"]["item_count"] == len(registry["items"])
    BUILD.validate_round1_release_record(
        release_record,
        registry,
        _sha(registry_path),
        catalog,
    )
    assert catalog_path.is_file()


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda payload: payload.update(manifest_id="wrong"), "manifest_id"),
        (lambda payload: payload.update(manifest_sha256="f" * 64), "manifest_sha256"),
        (lambda payload: payload.update(unexpected=True), "schema validation"),
        (
            lambda payload: payload["items"][1].update(
                blind_id=payload["items"][0]["blind_id"]
            ),
            "duplicate or invalid blind_id",
        ),
        (
            lambda payload: payload["items"][0].update(
                media_path=f"media/{payload['items'][1]['blind_id']}.mp4"
            ),
            "registry path",
        ),
        (lambda payload: payload["items"].pop(), "blind-ID set"),
    ],
)
def test_round2_rejects_registry_lineage_mutations(
    tmp_path: Path, mutation, match: str
) -> None:
    catalog_path, catalog = _catalog(tmp_path)
    media_root = tmp_path / "audio_media"
    registry_path = _audio_registry(tmp_path, catalog, media_root)
    release_record = _round1_release_record(tmp_path, catalog, registry_path)
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    mutation(payload)
    _write_json(registry_path, payload)

    with pytest.raises(ValueError, match=match):
        BUILD.build_package(
            catalog_path,
            tmp_path / "must_not_exist",
            media_root,
            registry_path,
            release_record,
        )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda payload: payload.update(schema_version="old"), "v4 release schema"),
        (lambda payload: payload.update(release_id="round1_curation_v3"), "round1_curation_v4"),
        (lambda payload: payload.update(status="DRAFT"), "not curation-authorized"),
        (
            lambda payload: payload["manifest"].update(sha256="e" * 64),
            "manifest SHA does not match",
        ),
        (
            lambda payload: payload["audio_media_registry"].update(sha256="e" * 64),
            "audio registry SHA does not match",
        ),
        (
            lambda payload: payload["audio_media_registry"].update(storage="public"),
            "not operator-private",
        ),
        (
            lambda payload: payload["audio_media_registry"].update(item_count=999),
            "item_count does not match",
        ),
        (
            lambda payload: payload["audio_media_registry"].update(path="private.json"),
            "must contain exactly",
        ),
        (
            lambda payload: payload["source_contract_sha256"].update(
                audio_media_registry="e" * 64
            ),
            "source contract does not bind the audio registry",
        ),
    ],
)
def test_round2_requires_untampered_round1_v4_release_record(
    tmp_path: Path, mutation, match: str
) -> None:
    catalog_path, catalog = _catalog(tmp_path)
    media_root = tmp_path / "audio_media"
    registry_path = _audio_registry(tmp_path, catalog, media_root)
    release_record = _round1_release_record(tmp_path, catalog, registry_path)
    payload = json.loads(release_record.read_text(encoding="utf-8"))
    mutation(payload)
    _write_json(release_record, payload)

    with pytest.raises(ValueError, match=match):
        BUILD.build_package(
            catalog_path,
            tmp_path / "must_not_exist",
            media_root,
            registry_path,
            release_record,
        )


def test_round2_tampering_registry_and_media_together_still_fails(
    tmp_path: Path,
) -> None:
    catalog_path, catalog = _catalog(tmp_path)
    media_root = tmp_path / "audio_media"
    for item in catalog["items"]:
        media = media_root / item["media_path"]
        media.parent.mkdir(parents=True, exist_ok=True)
        media.write_bytes(f"original:{item['blind_id']}".encode())
    registry_path = _audio_registry(tmp_path, catalog, media_root)
    release_record = _round1_release_record(tmp_path, catalog, registry_path)

    first = catalog["items"][0]
    replacement = media_root / first["media_path"]
    replacement.write_bytes(b"replacement media with replacement audio")
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry["items"][0]["sha256"] = _sha(replacement)
    _write_json(registry_path, registry)

    with pytest.raises(ValueError, match="release record audio registry SHA"):
        BUILD.build_package(
            catalog_path,
            tmp_path / "must_not_exist",
            media_root,
            registry_path,
            release_record,
        )


def test_round2_detects_input_mutation_before_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    catalog_path, catalog = _catalog(tmp_path)
    media_root = tmp_path / "audio_media"
    for item in catalog["items"]:
        media = media_root / item["media_path"]
        media.parent.mkdir(parents=True, exist_ok=True)
        media.write_bytes(f"source:{item['blind_id']}".encode())
    registry_path = _audio_registry(tmp_path, catalog, media_root)
    release_record = _round1_release_record(tmp_path, catalog, registry_path)
    instructions = tmp_path / "INSTRUCTIONS_ROUND2.md"
    instructions.write_bytes(BUILD.INSTRUCTIONS_PATH.read_bytes())
    monkeypatch.setattr(BUILD, "INSTRUCTIONS_PATH", instructions)
    calls = 0

    def mutate_after_snapshot(source: Path, target: Path, expected: str) -> str:
        nonlocal calls
        result = _fake_remux(source, target, expected)
        calls += 1
        if calls == 1:
            instructions.write_text("changed after snapshot\n", encoding="utf-8")
        return result

    monkeypatch.setattr(BUILD, "_remux_sanitized_media", mutate_after_snapshot)
    output = tmp_path / "must_not_publish"
    with pytest.raises(RuntimeError, match="snapshotted input changed before publish"):
        BUILD.build_package(
            catalog_path,
            output,
            media_root,
            registry_path,
            release_record,
        )
    assert not output.exists()


def test_round2_concurrent_output_injection_is_not_clobbered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    catalog_path, catalog = _catalog(tmp_path)
    media_root = tmp_path / "audio_media"
    for item in catalog["items"]:
        media = media_root / item["media_path"]
        media.parent.mkdir(parents=True, exist_ok=True)
        media.write_bytes(f"source:{item['blind_id']}".encode())
    registry_path = _audio_registry(tmp_path, catalog, media_root)
    release_record = _round1_release_record(tmp_path, catalog, registry_path)
    output = tmp_path / "round2_release"
    original_assert = BUILD._assert_snapshots_unchanged

    def inject_after_preflight(snapshots) -> None:
        original_assert(snapshots)
        output.mkdir()
        (output / "INJECTED_SENTINEL.txt").write_text("preserve me\n", encoding="utf-8")

    monkeypatch.setattr(BUILD, "_assert_snapshots_unchanged", inject_after_preflight)
    monkeypatch.setattr(BUILD, "_remux_sanitized_media", _fake_remux)
    with pytest.raises(FileExistsError, match="concurrently created"):
        BUILD.build_package(
            catalog_path,
            output,
            media_root,
            registry_path,
            release_record,
        )
    assert (output / "INJECTED_SENTINEL.txt").read_text(encoding="utf-8") == "preserve me\n"
    assert not (output / "COMPLETE.json").exists()


def test_round2_remux_strips_private_metadata_chapters_and_extra_streams(
    tmp_path: Path,
) -> None:
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        pytest.skip("ffmpeg and ffprobe are required for the media-sanitization test")
    source = tmp_path / "PRIVATE_CONDITION_source.mp4"
    metadata = tmp_path / "private_condition.ffmetadata"
    metadata.write_text(
        ";FFMETADATA1\n"
        "title=PRIVATE_CONDITION\n"
        "[CHAPTER]\nTIMEBASE=1/1000\nSTART=0\nEND=250\n"
        "title=PRIVATE_CONDITION_CHAPTER\n",
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            ffmpeg,
            "-v", "error",
            "-y",
            "-f", "lavfi", "-i", "color=c=red:s=64x64:d=0.5:r=10",
            "-f", "lavfi", "-i", "color=c=blue:s=64x64:d=0.5:r=10",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=0.5",
            "-f", "lavfi", "-i", "sine=frequency=880:duration=0.5",
            "-f", "ffmetadata", "-i", str(metadata),
            "-map", "0:v:0",
            "-map", "1:v:0",
            "-map", "2:a:0",
            "-map", "3:a:0",
            "-map_metadata", "4",
            "-map_chapters", "4",
            "-c:v", "mpeg4",
            "-c:a", "aac",
            "-metadata", "title=PRIVATE_CONDITION",
            "-metadata:s:v:0", "comment=PRIVATE_CONDITION_STREAM",
            "-metadata:s:a:1", "title=PRIVATE_CONDITION_AUDIO",
            str(source),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    source_probe = BUILD._probe_media(source)
    source_video, source_audio, source_total = BUILD._stream_counts(source_probe)
    assert (source_video, source_audio) == (2, 2)
    assert source_total > 4  # MP4 chapter data is an additional non-A/V stream.
    assert source_probe.get("chapters")
    assert "PRIVATE_CONDITION" in json.dumps(source_probe)
    target = tmp_path / "delivered.mp4"
    BUILD._remux_sanitized_media(source, target, _sha(source))
    probe = BUILD._probe_media(target)
    assert BUILD._stream_counts(probe) == (1, 1, 2)
    assert probe.get("chapters") == []
    assert "PRIVATE_CONDITION" not in json.dumps(probe)
    assert b"PRIVATE_CONDITION" not in target.read_bytes()


def test_round2_pure_javascript_sha256_and_import_hardening() -> None:
    template = (PACK / "release_src" / "rate_round2.html").read_text(encoding="utf-8")
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for the offline SHA-256 known-vector test")
    match = re.search(r'<script id="round2-hash-code">([\s\S]*?)</script>', template)
    assert match is not None
    program = (
        "const module={exports:{}}; const exports=module.exports;\n"
        + match.group(1)
        + '\nprocess.stdout.write(module.exports.sha256Hex("abc"));'
    )
    completed = subprocess.run([node, "-e", program], text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert "crypto.subtle" not in template
    assert "try {\n      localStorage.setItem" in template
    assert "Object.keys(payload).sort().join" in template
    assert 'Object.keys(rating).sort().join' in template
    assert 'Object.keys(presence).sort().join' in template


def test_round2_malformed_or_unavailable_storage_is_recoverable_without_overwrite() -> None:
    template = (PACK / "release_src" / "rate_round2.html").read_text(encoding="utf-8")
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for the offline local-storage test")
    match = re.search(r'<script id="round2-storage-code">([\s\S]*?)</script>', template)
    assert match is not None
    program = (
        "const module={exports:{}}; const exports=module.exports;\n"
        + match.group(1)
        + "\nlet writes=0;"
        + "const storage={getItem:()=>'{bad json',setItem:()=>{writes+=1;}};"
        + "const malformed=module.exports.readStoredState(storage,'key');"
        + "const unavailable=module.exports.readStoredState({getItem:()=>{throw new Error('blocked');}},'key');"
        + "const root={};Object.defineProperty(root,'localStorage',{get(){throw new Error('denied');}});"
        + "const getterUnavailable=module.exports.resolveStorage(root);"
        + "process.stdout.write(JSON.stringify({writes,malformed,unavailable,getterUnavailable,"
        + "valid:module.exports.isRfc3339('2026-07-19T03:04:05.123Z'),"
        + "badDay:module.exports.isRfc3339('2026-02-30T03:04:05Z'),"
        + "badZone:module.exports.isRfc3339('2026-07-19 03:04:05')}));"
    )
    completed = subprocess.run([node, "-e", program], text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["writes"] == 0
    assert result["malformed"]["status"] == "malformed"
    assert "not overwritten" in result["malformed"]["message"]
    assert result["unavailable"]["status"] == "unavailable"
    assert result["getterUnavailable"]["status"] == "unavailable"
    assert result["getterUnavailable"]["value"] is None
    assert result["valid"] is True
    assert result["badDay"] is False
    assert result["badZone"] is False
    assert 'if (result.status === "missing") return false;' in template
    assert "if (!loadLocal())" in template
    assert "Round2Storage.resolveStorage(window)" in template
    assert template.index('id="import-button"') < template.index('id="session"')
    assert "state.storageWritable = false" in template
    assert "Imported in memory" in template


def test_round2_ratings_schema_rejects_unexpected_fields(tmp_path: Path) -> None:
    catalog_path, catalog = _catalog(tmp_path)
    manifest = BUILD.build_manifest(catalog, _sha(catalog_path), "b" * 64)
    manifest_path = tmp_path / "manifest.json"
    _write_json(manifest_path, manifest)
    base = _round2_export(manifest, manifest_path, "presence-rater-2")
    schema = json.loads(
        (PACK / "release_src" / "round2_ratings.schema.json").read_text(encoding="utf-8")
    )
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    for mutate in (
        lambda payload: payload.update(unexpected=True),
        lambda payload: payload.update(started_at="2026-07-19 02:00:00"),
        lambda payload: payload.update(started_at="2026-02-30T02:00:00Z"),
        lambda payload: payload["ratings"][0].update(unexpected=True),
        lambda payload: payload["ratings"][0]["presence"].update(unexpected=True),
    ):
        payload = copy.deepcopy(base)
        mutate(payload)
        assert list(validator.iter_errors(payload))


def test_round2_multirater_ac1_known_two_export_answer_and_unique_raters(tmp_path: Path) -> None:
    catalog_path, catalog = _catalog(tmp_path)
    manifest = BUILD.build_manifest(catalog, _sha(catalog_path), "b" * 64)
    manifest["items"] = manifest["items"][:4]
    manifest["counts"] = {
        "events": 4,
        "unique_videos": len({item["blind_id"] for item in manifest["items"]}),
    }
    manifest_path = tmp_path / "manifest.json"
    _write_json(manifest_path, manifest)

    exports = []
    verdict_rows = [
        ["target_present", "target_present", "absent", "uncertain"],
        ["target_present", "absent", "absent", "uncertain"],
    ]
    background_rows = [[True, False, False, False], [True, False, True, False]]
    for rater_index in range(2):
        payload = _round2_export(manifest, manifest_path, f"presence-rater-{rater_index + 2}")
        for index, rating in enumerate(payload["ratings"]):
            rating["completed"] = True
            rating["presence"]["verdict"] = verdict_rows[rater_index][index]
            rating["presence"]["unrelated_background"] = background_rows[rater_index][index]
        path = tmp_path / f"ratings_{rater_index}.json"
        _write_json(path, payload)
        exports.append(path)

    with pytest.raises(ValueError, match="at least two"):
        AC1.compute_report(manifest_path, exports[:1])
    report = AC1.compute_report(manifest_path, exports)
    assert report["n_raters"] == 2
    assert report["questions"]["presence_verdict"]["observed_agreement"] == pytest.approx(0.75)
    assert report["questions"]["presence_verdict"]["ac1"] == pytest.approx(27 / 43)
    assert report["questions"]["unrelated_background"]["observed_agreement"] == pytest.approx(0.75)
    assert report["questions"]["unrelated_background"]["ac1"] == pytest.approx(9 / 17)
    assert [row["value"] for row in report["questions"]["presence_verdict"]["categories"]] == [
        "target_present", "absent", "uncertain"
    ]
    assert [row["value"] for row in report["questions"]["unrelated_background"]["categories"]] == [False, True]

    duplicate = json.loads(exports[1].read_text(encoding="utf-8"))
    duplicate["rater_id"] = "PRESENCE-RATER-2"
    _write_json(exports[1], duplicate)
    with pytest.raises(ValueError, match="distinct unique rater IDs"):
        AC1.compute_report(manifest_path, exports)


def _mutate_inputs_after_snapshot(
    monkeypatch: pytest.MonkeyPatch, paths: list[Path]
) -> tuple[dict[Path, bytes], dict[Path, int]]:
    real_read_bytes = Path.read_bytes
    originals = {path: real_read_bytes(path) for path in paths}
    reads = {path: 0 for path in paths}

    def read_bytes_then_mutate(path: Path) -> bytes:
        snapshot = real_read_bytes(path)
        if path in reads:
            reads[path] += 1
            if reads[path] == 1:
                path.write_bytes(b"{}\n")
        return snapshot

    monkeypatch.setattr(Path, "read_bytes", read_bytes_then_mutate)
    return originals, reads


def _round2_score_inputs(
    tmp_path: Path, *, n_raters: int = 1
) -> tuple[Path, list[Path]]:
    catalog_path, catalog = _catalog(tmp_path / "catalog")
    manifest = BUILD.build_manifest(catalog, _sha(catalog_path), "b" * 64)
    manifest_path = tmp_path / "round2_manifest.json"
    _write_json(manifest_path, manifest)
    ratings_paths: list[Path] = []
    for index in range(n_raters):
        ratings_path = tmp_path / f"round2_ratings_{index}.json"
        _write_json(
            ratings_path,
            _round2_export(manifest, manifest_path, f"presence-rater-{index + 2}"),
        )
        ratings_paths.append(ratings_path)
    return manifest_path, ratings_paths


def test_round1_parse_and_provenance_hash_share_one_byte_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path, ratings_path, _, _ = _round1_fixture(tmp_path)
    originals, reads = _mutate_inputs_after_snapshot(
        monkeypatch, [manifest_path, ratings_path]
    )

    catalog = CURATE.curate(manifest_path, ratings_path, tmp_path / "catalog.json")

    assert reads == {manifest_path: 1, ratings_path: 1}
    assert catalog["source_manifest_sha256"] == hashlib.sha256(
        originals[manifest_path]
    ).hexdigest()
    assert catalog["source_export_sha256"] == hashlib.sha256(
        originals[ratings_path]
    ).hexdigest()
    assert catalog["counts"]["events_eligible"] == 6


def test_round2_summary_parse_and_provenance_hash_share_one_byte_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path, ratings_paths = _round2_score_inputs(tmp_path)
    ratings_path = ratings_paths[0]
    originals, reads = _mutate_inputs_after_snapshot(
        monkeypatch, [manifest_path, ratings_path]
    )

    summary = SCORE.score(manifest_path, ratings_path, tmp_path / "summary.json")

    assert reads == {manifest_path: 1, ratings_path: 1}
    assert summary["source_manifest_sha256"] == hashlib.sha256(
        originals[manifest_path]
    ).hexdigest()
    assert summary["source_export_sha256"] == hashlib.sha256(
        originals[ratings_path]
    ).hexdigest()
    assert summary["counts"] == {"events_total": 6, "completed": 3, "incomplete": 3}


def test_round2_ac1_parse_and_provenance_hash_share_one_byte_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path, ratings_paths = _round2_score_inputs(tmp_path, n_raters=2)
    inputs = [manifest_path, *ratings_paths]
    originals, reads = _mutate_inputs_after_snapshot(monkeypatch, inputs)

    report = AC1.score(manifest_path, ratings_paths, tmp_path / "ac1.json")

    assert reads == {path: 1 for path in inputs}
    assert report["source_manifest_sha256"] == hashlib.sha256(
        originals[manifest_path]
    ).hexdigest()
    expected_export_hashes = {
        hashlib.sha256(originals[path]).hexdigest() for path in ratings_paths
    }
    assert {row["sha256"] for row in report["source_exports"]} == expected_export_hashes
    assert report["n_raters"] == 2


def _processing_score_invocations(tmp_path: Path):
    round1_manifest, round1_ratings, _, _ = _round1_fixture(tmp_path / "round1")
    round2_manifest, round2_ratings = _round2_score_inputs(
        tmp_path / "round2", n_raters=2
    )
    return [
        (
            "round1",
            lambda output: CURATE.curate(round1_manifest, round1_ratings, output),
        ),
        (
            "round2",
            lambda output: SCORE.score(round2_manifest, round2_ratings[0], output),
        ),
        (
            "round2_ac1",
            lambda output: AC1.score(round2_manifest, round2_ratings, output),
        ),
    ]


def test_processing_outputs_refuse_to_overwrite_preexisting_files(tmp_path: Path) -> None:
    sentinel = b"concurrent reviewer output\n"
    for name, invoke in _processing_score_invocations(tmp_path):
        output = tmp_path / f"{name}_preexisting.json"
        output.write_bytes(sentinel)
        with pytest.raises(FileExistsError, match="refusing to overwrite"):
            invoke(output)
        assert output.read_bytes() == sentinel


def test_processing_outputs_do_not_clobber_destination_created_at_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sentinel = b"racer won publication\n"
    real_open = Path.open
    for name, invoke in _processing_score_invocations(tmp_path):
        output = tmp_path / f"{name}_raced.json"
        raced = False

        def racing_open(path: Path, mode: str = "r", *args, **kwargs):
            nonlocal raced
            if path == output and mode == "xb" and not raced:
                raced = True
                with real_open(path, "xb") as handle:
                    handle.write(sentinel)
            return real_open(path, mode, *args, **kwargs)

        with monkeypatch.context() as context:
            context.setattr(Path, "open", racing_open)
            with pytest.raises(FileExistsError, match="refusing to overwrite"):
                invoke(output)
        assert raced is True
        assert output.read_bytes() == sentinel
