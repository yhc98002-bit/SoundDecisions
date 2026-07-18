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
        "schema_version": "sounddecisions-human-curation-items-v1-1.2",
        "instrument_version": "human-eval-round1-curation-1.2",
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
        "schema_version": "sounddecisions-human-curation-ratings-v1-1.2",
        "instrument_version": "human-eval-round1-curation-1.2",
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
        BUILD.build_manifest(broken, "a" * 64)


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
        if item["events"]:
            media = media_root / item["media_path"]
            media.parent.mkdir(parents=True, exist_ok=True)
            media.write_bytes(f"video:{item['blind_id']}".encode())
    output = tmp_path / "round2_release"
    monkeypatch.setattr(BUILD, "_probe_media_streams", lambda _: (1, 1))
    manifest = BUILD.build_package(catalog_path, output, media_root)

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
    assert (output / "INSTRUCTIONS.md").is_file()
    sums = json.loads((output / "SHA256SUMS.json").read_text(encoding="utf-8"))
    for relative, digest in sums.items():
        assert _sha(output / relative) == digest
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
    manifest = BUILD.build_manifest(catalog, _sha(catalog_path))
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
    catalog_path, _ = _catalog(tmp_path)
    output = tmp_path / "round2_release"
    with pytest.raises(FileNotFoundError, match="required blinded video is missing"):
        BUILD.build_package(catalog_path, output, tmp_path / "empty_media_root")
    assert not output.exists()


def test_round2_rejects_silent_media_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    catalog_path, catalog = _catalog(tmp_path)
    media_root = tmp_path / "silent_round1_release"
    for item in catalog["items"]:
        if item["events"]:
            media = media_root / item["media_path"]
            media.parent.mkdir(parents=True, exist_ok=True)
            media.write_bytes(b"silent-video")
    monkeypatch.setattr(BUILD, "_probe_media_streams", lambda _: (1, 0))
    output = tmp_path / "must_not_exist"
    with pytest.raises(ValueError, match="must contain video and audio streams"):
        BUILD.build_package(catalog_path, output, media_root)
    assert not output.exists()


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


def test_round2_ratings_schema_rejects_unexpected_fields(tmp_path: Path) -> None:
    catalog_path, catalog = _catalog(tmp_path)
    manifest = BUILD.build_manifest(catalog, _sha(catalog_path))
    manifest_path = tmp_path / "manifest.json"
    _write_json(manifest_path, manifest)
    base = _round2_export(manifest, manifest_path, "presence-rater-2")
    schema = json.loads(
        (PACK / "release_src" / "round2_ratings.schema.json").read_text(encoding="utf-8")
    )
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    for mutate in (
        lambda payload: payload.update(unexpected=True),
        lambda payload: payload["ratings"][0].update(unexpected=True),
        lambda payload: payload["ratings"][0]["presence"].update(unexpected=True),
    ):
        payload = copy.deepcopy(base)
        mutate(payload)
        assert list(validator.iter_errors(payload))


def test_round2_multirater_ac1_known_two_export_answer_and_unique_raters(tmp_path: Path) -> None:
    catalog_path, catalog = _catalog(tmp_path)
    manifest = BUILD.build_manifest(catalog, _sha(catalog_path))
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
