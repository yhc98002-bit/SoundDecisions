"""Schema, manifest, and sealed-boundary tests for the human-eval package."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import re

import pytest
from jsonschema import Draft202012Validator, FormatChecker


ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "results" / "human_eval_pack"
FIXTURES = ROOT / "tests" / "fixtures" / "human_eval"
SCORE_SPEC = importlib.util.spec_from_file_location("human_eval_score_schema", PACK / "score_ac1.py")
assert SCORE_SPEC is not None and SCORE_SPEC.loader is not None
SCORER = importlib.util.module_from_spec(SCORE_SPEC)
SCORE_SPEC.loader.exec_module(SCORER)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_committed_rating_schema_accepts_known_exports_and_rejects_incomplete_complete_flag(
    tmp_path: Path,
) -> None:
    schema = json.loads((PACK / "ratings.schema.json").read_text(encoding="utf-8"))
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["properties"]["schema_version"]["const"] == SCORER.RATING_SCHEMA_VERSION
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema, format_checker=FormatChecker())

    for name in ("ratings_rater_a.json", "ratings_rater_b.json"):
        payload = SCORER.load_rating_export(FIXTURES / name)
        validator.validate(payload)

    invalid = json.loads((FIXTURES / "ratings_rater_a.json").read_text(encoding="utf-8"))
    invalid["ratings"][0]["anchor"]["end_s"] = None
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(invalid), encoding="utf-8")
    assert list(validator.iter_errors(invalid))
    with pytest.raises(ValueError, match="schema validation failed"):
        SCORER.load_rating_export(path)


def test_partial_progress_export_is_valid_and_lossless(tmp_path: Path) -> None:
    partial = json.loads((FIXTURES / "ratings_rater_a.json").read_text(encoding="utf-8"))
    partial["ratings"] = [copy.deepcopy(partial["ratings"][0])]
    partial["item_order"] = [partial["ratings"][0]["blind_id"]]
    rating = partial["ratings"][0]
    rating["completed"] = False
    rating["anchor"] = {
        "status": "unrated",
        "start_s": 1.25,
        "end_s": None,
        "event_description": "partial mark",
    }
    rating["presence"] = {"verdict": None, "unrelated_background": None, "note": ""}
    path = tmp_path / "partial.json"
    path.write_text(json.dumps(partial), encoding="utf-8")

    schema = json.loads((PACK / "ratings.schema.json").read_text(encoding="utf-8"))
    Draft202012Validator(schema, format_checker=FormatChecker()).validate(partial)
    assert SCORER.load_rating_export(path) == partial


def test_blinded_manifest_contract_and_inline_copy_are_exact() -> None:
    schema = json.loads((PACK / "blinded_items.schema.json").read_text(encoding="utf-8"))
    manifest_path = PACK / "blinded_items.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(manifest)
    assert schema["properties"]["manifest_id"]["const"] == manifest["manifest_id"]
    assert manifest["status"] == "INCOMPLETE_ARTIFACTS_PRE_FREEZE_DO_NOT_RATE"
    assert manifest["counts"] == {
        "anchor_presence": 30,
        "two_event": 60,
        "total_tasks": 90,
        "unique_videos": 82,
    }
    assert len(manifest["items"]) == 82
    assert len({item["blind_id"] for item in manifest["items"]}) == 82
    assert sum("anchor_presence" in item["tasks"] for item in manifest["items"]) == 30
    assert sum("two_event" in item["tasks"] for item in manifest["items"]) == 60
    for item in manifest["items"]:
        assert set(item) == {"blind_id", "tasks", "media_path", "fps", "duration_s", "target_prompt"}
        assert re.fullmatch(r"HEV2-[0-9A-F]{12}", item["blind_id"])
        assert item["media_path"] == f"media/{item['blind_id']}.mp4"
        assert item["fps"] > 0 and item["duration_s"] > 0

    html = (PACK / "rate.html").read_text(encoding="utf-8")
    match = re.search(
        r'<script id="manifest-data" type="application/json" data-sha256="([0-9a-f]{64})">(.*?)</script>',
        html,
        re.DOTALL,
    )
    assert match is not None
    assert match.group(1) == _sha256(manifest_path)
    assert json.loads(match.group(2)) == manifest
    assert "__BLINDED_MANIFEST_" not in html


def test_unblinding_map_is_sealed_and_registered() -> None:
    sealed_path = PACK / "unblinding_map.sealed.json"
    sealed = json.loads(sealed_path.read_text(encoding="utf-8"))
    sums = json.loads((PACK / "SHA256SUMS.json").read_text(encoding="utf-8"))
    assert sealed["format"] == "sounddecisions-sealed-map-v1"
    assert sealed["cipher"] == "AES-256-CBC"
    assert "items" not in sealed
    assert "unblinding_map.sealed.json" in sums
    for relative_path, expected in sums.items():
        assert expected == _sha256(PACK / relative_path), relative_path
