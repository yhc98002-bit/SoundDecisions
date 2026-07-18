"""Contracts for the offline Round-1 event-curation instrument."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

import pytest


jsonschema = pytest.importorskip("jsonschema")

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "human_eval_pack" / "release_src"
HTML = SOURCE / "round1_rate.template.html"
MANIFEST_SCHEMA = SOURCE / "round1_manifest.schema.json"
RATINGS_SCHEMA = SOURCE / "round1_ratings.schema.json"
SHA = "a" * 64


def _item(blind_id: str = "HEV2-0123456789AB", tasks: list[str] | None = None) -> dict[str, object]:
    return {
        "blind_id": blind_id,
        "tasks": tasks or ["anchor_curation", "two_event_curation"],
        "media_path": f"media/{blind_id}.mp4",
        "fps": 30.0,
        "duration_s": 10.0,
        "candidate_caption": "A person taps the metal bowl twice.",
    }


def _manifest(status: str = "CURATION_AUTHORIZED") -> dict[str, object]:
    item = _item()
    return {
        "schema_version": "sounddecisions-human-curation-items-v1-1.0",
        "instrument_version": "human-eval-round1-curation-1.0",
        "manifest_id": "round1-fixture",
        "status": status,
        "default_fps": 30.0,
        "counts": {
            "anchor_curation": 1,
            "two_event_curation": 1,
            "total_tasks": 2,
            "unique_videos": 1,
        },
        "items": [item],
    }


def _ratings() -> dict[str, object]:
    blind_id = "HEV2-0123456789AB"
    return {
        "schema_version": "sounddecisions-human-curation-ratings-v1-1.0",
        "instrument_version": "human-eval-round1-curation-1.0",
        "manifest_id": "round1-fixture",
        "manifest_sha256": SHA,
        "rater_id": "lead-17",
        "started_at": "2026-07-19T00:00:00.000Z",
        "exported_at": "2026-07-19T00:03:00.000Z",
        "item_order": [blind_id],
        "ratings": [
            {
                "blind_id": blind_id,
                "tasks": ["anchor_curation", "two_event_curation"],
                "completed": False,
                "anchor_curation": {
                    "status": "unrated",
                    "start_s": 1.0,
                    "end_s": None,
                    "note": "Draft boundary",
                },
                "two_event_curation": {
                    "verdict": "confirm",
                    "event_1": {
                        "description": "first tap",
                        "start_s": 1.0,
                        "end_s": 1.2,
                    },
                    "event_2": {
                        "description": "second tap",
                        "start_s": 2.0,
                        "end_s": None,
                    },
                    "note": "Draft second interval",
                },
            }
        ],
    }


def _run_javascript(expression: str, payload: object) -> object:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is needed to execute the instrument's pure helpers")
    program = rf"""
const fs = require("fs");
const html = fs.readFileSync({json.dumps(str(HTML))}, "utf8");
const match = html.match(/<script id="instrument-code">([\s\S]*?)<\/script>/);
if (!match) throw new Error("instrument script not found");
const instrument = {{exports: {{}}}};
new Function("module", "exports", match[1])(instrument, instrument.exports);
const api = instrument.exports;
const input = JSON.parse(fs.readFileSync(0, "utf8"));
const result = {expression};
process.stdout.write(JSON.stringify(result));
"""
    completed = subprocess.run(
        [node, "-e", program],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout)


def test_template_is_offline_blinded_and_has_exact_builder_markers() -> None:
    html = HTML.read_text(encoding="utf-8")

    assert html.count("__ROUND1_MANIFEST_JSON__") == 1
    assert html.count("__ROUND1_MANIFEST_SHA256__") == 1
    assert 'id="manifest-data"' in html
    assert 'id="instrument-code"' in html
    assert "fetch(" not in html
    assert "XMLHttpRequest" not in html
    assert "https://" not in html
    assert "unblinding" not in html.lower()
    assert "condition" not in html.lower()
    assert "model" not in html.lower()


def test_ui_has_only_curation_tasks_caption_and_permanent_mute_guards() -> None:
    html = HTML.read_text(encoding="utf-8")

    assert "candidate_caption" in html
    assert "first clearly visible discrete occurrence" in html
    assert "two separable target events" in html
    assert "anchor_curation" in html
    assert "two_event_curation" in html
    assert "presence" not in html.lower()
    assert "target_present" not in html
    assert "<video id=\"item-video\" controls muted" in html
    assert 'video.addEventListener("volumechange"' in html
    assert 'video.addEventListener("play"' in html
    assert "video.defaultMuted = true" in html
    assert "video.volume = 0" in html


def test_manifest_and_partial_export_validate_against_committed_schemas() -> None:
    manifest_schema = json.loads(MANIFEST_SCHEMA.read_text(encoding="utf-8"))
    ratings_schema = json.loads(RATINGS_SCHEMA.read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator

    validator.check_schema(manifest_schema)
    validator.check_schema(ratings_schema)
    validator(manifest_schema, format_checker=jsonschema.FormatChecker()).validate(_manifest())
    validator(ratings_schema, format_checker=jsonschema.FormatChecker()).validate(_ratings())


def test_schema_rejects_unauthorized_manifest_and_presence_payload() -> None:
    manifest_schema = json.loads(MANIFEST_SCHEMA.read_text(encoding="utf-8"))
    ratings_schema = json.loads(RATINGS_SCHEMA.read_text(encoding="utf-8"))
    unauthorized = _manifest("PRE_FREEZE")
    with_presence = _ratings()
    with_presence["ratings"][0]["presence"] = {"verdict": "target_present"}  # type: ignore[index]

    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(manifest_schema).validate(unauthorized)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(ratings_schema).validate(with_presence)


def test_seeded_shuffle_is_deterministic_and_pinned() -> None:
    result = _run_javascript(
        "({first: api.deterministicShuffle(input.items, input.rater), "
        "second: api.deterministicShuffle(input.items, input.rater), "
        "other: api.deterministicShuffle(input.items, 'another-lead')})",
        {"items": ["A", "B", "C", "D", "E"], "rater": "lead-17"},
    )

    assert result["first"] == result["second"]
    assert sorted(result["first"]) == ["A", "B", "C", "D", "E"]
    assert sorted(result["other"]) == ["A", "B", "C", "D", "E"]
    assert result["first"] != result["other"]


def test_authorization_manifest_validation_and_blank_state_have_no_presence() -> None:
    result = _run_javascript(
        "(() => { const manifest = api.validateManifest(input); "
        "return {status: manifest.status, blank: api.blankRating(manifest.items[0])}; })()",
        _manifest(),
    )

    assert result["status"] == "CURATION_AUTHORIZED"
    assert result["blank"]["anchor_curation"]["status"] == "unrated"
    assert result["blank"]["two_event_curation"]["verdict"] is None
    assert "presence" not in result["blank"]


def test_completion_requires_valid_anchor_and_ordered_described_pair() -> None:
    result = _run_javascript(
        "({partial: api.isCompleted(input.partial), "
        "complete: api.isCompleted(input.complete), "
        "reversed: api.isCompleted(input.reversed), "
        "uncertain: api.isCompleted(input.uncertain)})",
        {
            "partial": _ratings()["ratings"][0],
            "complete": {
                **_ratings()["ratings"][0],
                "anchor_curation": {"status": "marked", "start_s": 0.5, "end_s": 0.7, "note": ""},
                "two_event_curation": {
                    "verdict": "confirm",
                    "event_1": {"description": "first", "start_s": 1.0, "end_s": 1.1},
                    "event_2": {"description": "second", "start_s": 2.0, "end_s": 2.1},
                    "note": "",
                },
            },
            "reversed": {
                **_ratings()["ratings"][0],
                "anchor_curation": {"status": "marked", "start_s": 0.5, "end_s": 0.7, "note": ""},
                "two_event_curation": {
                    "verdict": "confirm",
                    "event_1": {"description": "later", "start_s": 3.0, "end_s": 3.1},
                    "event_2": {"description": "earlier", "start_s": 2.0, "end_s": 2.1},
                    "note": "",
                },
            },
            "uncertain": {
                **_ratings()["ratings"][0],
                "anchor_curation": {"status": "too_uncertain", "start_s": 0.5, "end_s": None, "note": "occluded"},
                "two_event_curation": {"verdict": "reject", "event_1": None, "event_2": None, "note": "one event"},
            },
        },
    )

    assert result == {"partial": False, "complete": True, "reversed": False, "uncertain": True}


def test_export_import_round_trip_preserves_partial_drafts_and_checks_manifest() -> None:
    payload = _ratings()
    result = _run_javascript(
        "(() => { const roundTrip = api.roundTripExport(input.payload); "
        "const prepared = api.prepareImport(roundTrip, api.validateManifest(input.manifest), input.sha); "
        "let mismatch = null; try { api.prepareImport(roundTrip, api.validateManifest(input.manifest), 'b'.repeat(64)); } "
        "catch (error) { mismatch = error.message; } return {prepared, mismatch}; })()",
        {"payload": payload, "manifest": _manifest(), "sha": SHA},
    )

    assert result["prepared"] == payload
    assert result["prepared"]["ratings"][0]["anchor_curation"]["end_s"] is None
    assert result["prepared"]["ratings"][0]["two_event_curation"]["event_2"]["end_s"] is None
    assert "do not match" in result["mismatch"]


def test_uncertain_and_reject_preserve_drafts_and_mute_helper_is_fail_closed() -> None:
    result = _run_javascript(
        "(() => { const anchor = api.markAnchorTooUncertain(input.anchor); "
        "const pair = api.setPairVerdict(input.pair, 'reject'); "
        "const video = {muted: false, defaultMuted: false, volume: 1}; "
        "return {anchor, pair, muted: api.enforceMuted(video), video}; })()",
        {
            "anchor": {"status": "marked", "start_s": 1.0, "end_s": 1.2, "note": ""},
            "pair": {
                "verdict": "confirm",
                "event_1": {"description": "first", "start_s": 1.0, "end_s": 1.2},
                "event_2": {"description": "second", "start_s": 2.0, "end_s": None},
                "note": "",
            },
        },
    )

    assert result["anchor"] == {"status": "too_uncertain", "start_s": 1.0, "end_s": 1.2, "note": ""}
    assert result["pair"]["verdict"] == "reject"
    assert result["pair"]["event_2"]["description"] == "second"
    assert result["muted"] is True
    assert result["video"] == {"muted": True, "defaultMuted": True, "volume": 0}


def test_ui_persists_and_exports_without_network_or_build_runtime() -> None:
    html = HTML.read_text(encoding="utf-8")

    assert "localStorage.setItem" in html
    assert "localStorage.getItem" in html
    assert "ratings_${payload.rater_id}.json" in html
    assert "URL.createObjectURL" in html
    assert "DOMContentLoaded" in html
