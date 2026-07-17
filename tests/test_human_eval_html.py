"""Contract tests for pure helpers embedded in the offline rating instrument."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
RATE_HTML = ROOT / "results" / "human_eval_pack" / "rate.html"


def _run_javascript(expression: str, payload: object) -> object:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is needed to execute the instrument's embedded pure helpers")
    program = rf"""
const fs = require("fs");
const html = fs.readFileSync({json.dumps(str(RATE_HTML))}, "utf8");
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


def test_seeded_shuffle_is_deterministic_and_pinned() -> None:
    result = _run_javascript(
        "({first: api.deterministicShuffle(input.items, input.rater), "
        "second: api.deterministicShuffle(input.items, input.rater), "
        "other: api.deterministicShuffle(input.items, 'other-rater')})",
        {"items": ["A", "B", "C", "D", "E"], "rater": "rater-17"},
    )

    assert result["first"] == ["C", "A", "D", "B", "E"]
    assert result["second"] == result["first"]
    assert sorted(result["other"]) == ["A", "B", "C", "D", "E"]
    assert result["other"] != result["first"]


def test_export_import_round_trip_preserves_combined_task_rating() -> None:
    payload = {
        "schema_version": "sounddecisions-human-eval-v2-1.0",
        "instrument_version": "human-eval-pack-1.0",
        "manifest_id": "fixture-manifest",
        "manifest_sha256": "0" * 64,
        "rater_id": "rater-17",
        "started_at": "2026-07-17T00:00:00.000Z",
        "exported_at": "2026-07-17T00:05:00.000Z",
        "item_order": ["HEV2-0123456789AB"],
        "ratings": [
            {
                "blind_id": "HEV2-0123456789AB",
                "tasks": ["anchor_presence", "two_event"],
                "completed": False,
                "anchor": {
                    "status": "unrated",
                    "start_s": 1.25,
                    "end_s": None,
                    "event_description": "partly occluded impact",
                },
                "presence": {
                    "verdict": "uncertain",
                    "unrelated_background": True,
                    "note": "",
                },
                "pair_curation": {
                    "verdict": "confirm",
                    "event_1": {"start_s": 2.5, "end_s": None},
                    "event_2": None,
                    "event_1_description": "first contact",
                    "event_2_description": "",
                    "note": "second interval not marked yet",
                },
            }
        ],
    }

    assert _run_javascript("api.roundTripExport(input)", payload) == payload


def test_offline_instrument_has_inline_manifest_and_no_runtime_fetch() -> None:
    html = RATE_HTML.read_text(encoding="utf-8")

    assert 'id="manifest-data"' in html
    assert "fetch(" not in html
    assert "unblinding_map" not in html
    assert "https://" not in html


def test_uncertain_and_reject_choices_preserve_draft_marks() -> None:
    result = _run_javascript(
        "({anchor: api.markAnchorTooUncertain(input.anchor), "
        "pair: api.setPairVerdict(input.pair, 'reject')})",
        {
            "anchor": {
                "status": "marked",
                "start_s": 1.0,
                "end_s": 1.2,
                "event_description": "impact",
            },
            "pair": {
                "verdict": "confirm",
                "event_1": {"start_s": 1.0, "end_s": 1.2},
                "event_2": {"start_s": 2.0, "end_s": 2.2},
                "event_1_description": "first",
                "event_2_description": "second",
                "note": "",
            },
        },
    )

    assert result["anchor"]["status"] == "too_uncertain"
    assert result["anchor"]["start_s"] == 1.0
    assert result["pair"]["verdict"] == "reject"
    assert result["pair"]["event_1"]["start_s"] == 1.0
    assert result["pair"]["event_2_description"] == "second"


def test_committed_candidate_manifest_is_locked() -> None:
    manifest = json.loads(
        (ROOT / "results" / "human_eval_pack" / "blinded_items.json").read_text(encoding="utf-8")
    )
    result = _run_javascript("api.validateManifest(input).status", manifest)
    html = RATE_HTML.read_text(encoding="utf-8")

    assert result == "INCOMPLETE_ARTIFACTS_PRE_FREEZE_DO_NOT_RATE"
    assert 'manifest.status === "RATING_AUTHORIZED"' in html
    assert 'byId("start-button").disabled = true' in html


def _import_fixture() -> dict[str, object]:
    blind_id = "HEV2-0123456789AB"
    manifest_sha256 = "a" * 64
    manifest = {
        "manifest_id": "fixture-manifest",
        "status": "RATING_AUTHORIZED",
        "items": [{"blind_id": blind_id, "tasks": ["anchor_presence"]}],
    }
    payload = {
        "schema_version": "sounddecisions-human-eval-v2-1.0",
        "instrument_version": "human-eval-pack-1.0",
        "manifest_id": manifest["manifest_id"],
        "manifest_sha256": manifest_sha256,
        "rater_id": "rater-17",
        "started_at": "2026-07-17T00:00:00.000Z",
        "exported_at": "2026-07-17T00:05:00.000Z",
        "item_order": [blind_id],
        "ratings": [
            {
                "blind_id": blind_id,
                "tasks": ["anchor_presence"],
                "completed": False,
                "anchor": {
                    "status": "unrated",
                    "start_s": None,
                    "end_s": None,
                    "event_description": "",
                },
                "presence": {
                    "verdict": None,
                    "unrelated_background": None,
                    "note": "",
                },
            }
        ],
    }
    return {"manifest": manifest, "sha": manifest_sha256, "payload": payload}


def test_enter_and_click_share_the_authorization_guard() -> None:
    locked = _run_javascript(
        "(() => { try { api.assertRatingAuthorized(input); return null; } "
        "catch (error) { return error.message; } })()",
        {"status": "INCOMPLETE_ARTIFACTS_PRE_FREEZE_DO_NOT_RATE"},
    )
    html = RATE_HTML.read_text(encoding="utf-8")
    start = html.index("function startSession()")
    guard = html.index("assertRatingAuthorized(app.manifest);", start)
    first_state_access = html.index('cleanRaterId(byId("rater-id").value)', start)

    assert "locked until the signed freeze" in locked
    assert start < guard < first_state_access
    assert 'if (event.key === "Enter") startSession();' in html
    assert 'byId("start-button").addEventListener("click", startSession);' in html


def test_import_rejects_unauthorized_task_mismatch_and_schema_parity_errors() -> None:
    result = _run_javascript(
        "(() => {\n"
        "  const clone = value => JSON.parse(JSON.stringify(value));\n"
        "  const capture = callback => { try { callback(); return null; } catch (error) { return error.message; } };\n"
        "  const unauthorized = clone(input.manifest); unauthorized.status = 'PRE_FREEZE_DO_NOT_RATE';\n"
        "  const swapped = clone(input.payload);\n"
        "  swapped.ratings[0].tasks = ['two_event']; delete swapped.ratings[0].anchor; delete swapped.ratings[0].presence;\n"
        "  swapped.ratings[0].pair_curation = {verdict: 'reject', event_1: null, event_2: null, event_1_description: '', event_2_description: '', note: ''};\n"
        "  const duplicate = clone(input.payload); duplicate.ratings[0].tasks = ['anchor_presence', 'anchor_presence'];\n"
        "  const nested = clone(input.payload); nested.ratings[0].anchor.unexpected = true;\n"
        "  const tooLong = clone(input.payload); tooLong.ratings[0].presence.note = 'x'.repeat(2001);\n"
        "  const badTime = clone(input.payload); badTime.started_at = 'July 17, 2026';\n"
        "  return {\n"
        "    valid: capture(() => api.prepareImport(input.payload, input.manifest, input.sha)),\n"
        "    unauthorized: capture(() => api.prepareImport(input.payload, unauthorized, input.sha)),\n"
        "    swapped: capture(() => api.prepareImport(swapped, input.manifest, input.sha)),\n"
        "    duplicate: capture(() => api.prepareImport(duplicate, input.manifest, input.sha)),\n"
        "    nested: capture(() => api.prepareImport(nested, input.manifest, input.sha)),\n"
        "    too_long: capture(() => api.prepareImport(tooLong, input.manifest, input.sha)),\n"
        "    bad_time: capture(() => api.prepareImport(badTime, input.manifest, input.sha))\n"
        "  };\n"
        "})()",
        _import_fixture(),
    )

    assert result["valid"] is None
    assert "locked until the signed freeze" in result["unauthorized"]
    assert "tasks do not match" in result["swapped"]
    assert "Invalid rating" in result["duplicate"]
    assert "Invalid anchor fields" in result["nested"]
    assert "Invalid Presence fields" in result["too_long"]
    assert "RFC3339" in result["bad_time"]
