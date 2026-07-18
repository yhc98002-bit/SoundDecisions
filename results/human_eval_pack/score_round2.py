#!/usr/bin/env python3
"""Produce descriptive counts from exactly one Round-2 Presence export."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from jsonschema import Draft202012Validator, FormatChecker


HERE = Path(__file__).resolve().parent
MANIFEST_SCHEMA_PATH = HERE / "release_src" / "round2_manifest.schema.json"
RATINGS_SCHEMA_PATH = HERE / "release_src" / "round2_ratings.schema.json"
SUMMARY_SCHEMA_PATH = HERE / "release_src" / "round2_summary.schema.json"


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: cannot read JSON: {exc}") from exc


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rater_hash(value: str) -> str:
    return hashlib.sha256(value.strip().lower().encode("utf-8")).hexdigest()


def _validate(payload: Any, schema_path: Path, context: str) -> None:
    schema = _load_json(schema_path)
    Draft202012Validator.check_schema(schema)
    errors = sorted(
        Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(payload),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if errors:
        error = errors[0]
        location = "$" + "".join(
            f"[{part}]" if isinstance(part, int) else f".{part}"
            for part in error.absolute_path
        )
        raise ValueError(f"{context}: schema validation failed at {location}: {error.message}")


def validate_export(
    manifest: Mapping[str, Any], ratings: Mapping[str, Any], manifest_sha256: str
) -> None:
    if ratings["manifest_id"] != manifest["manifest_id"]:
        raise ValueError("manifest_id mismatch")
    if ratings["manifest_sha256"] != manifest_sha256:
        raise ValueError("manifest_sha256 mismatch")
    if _rater_hash(ratings["rater_id"]) == manifest["curator_rater_id_sha256"]:
        raise ValueError("Round-2 evaluator must be different from the Round-1 curator")
    expected = {item["event_id"] for item in manifest["items"]}
    if len(expected) != len(manifest["items"]):
        raise ValueError("manifest has duplicate event IDs")
    if set(ratings["item_order"]) != expected or len(ratings["item_order"]) != len(expected):
        raise ValueError("item_order does not match manifest")
    actual = [rating["event_id"] for rating in ratings["ratings"]]
    if set(actual) != expected or len(actual) != len(expected):
        raise ValueError("ratings item set does not match manifest")


def summarize(
    manifest: Mapping[str, Any],
    ratings: Mapping[str, Any],
    *,
    manifest_sha256: str,
    export_sha256: str,
) -> dict[str, Any]:
    validate_export(manifest, ratings, manifest_sha256)
    total = len(manifest["items"])
    completed = [rating for rating in ratings["ratings"] if rating["completed"]]
    verdicts = Counter(rating["presence"]["verdict"] for rating in completed)
    backgrounds = Counter(rating["presence"]["unrelated_background"] for rating in completed)
    return {
        "schema_version": "sounddecisions-human-presence-summary-v1.0",
        "analysis_scope": "single_rater_descriptive_only",
        "source_manifest_id": manifest["manifest_id"],
        "source_manifest_sha256": manifest_sha256,
        "source_export_sha256": export_sha256,
        "rater_id": ratings["rater_id"],
        "counts": {
            "events_total": total,
            "completed": len(completed),
            "incomplete": total - len(completed),
        },
        "coverage_rate": len(completed) / total,
        "verdict_counts": {
            "target_present": verdicts["target_present"],
            "absent": verdicts["absent"],
            "uncertain": verdicts["uncertain"],
        },
        "unrelated_background_counts": {
            "true": backgrounds[True],
            "false": backgrounds[False],
        },
        "limitation": (
            "Single-rater descriptive summary only; inter-rater agreement and AC1 are "
            "not computed from one export."
        ),
    }


def score(manifest_path: Path, ratings_path: Path, output_path: Path) -> dict[str, Any]:
    manifest = _load_json(manifest_path)
    ratings = _load_json(ratings_path)
    _validate(manifest, MANIFEST_SCHEMA_PATH, str(manifest_path))
    _validate(ratings, RATINGS_SCHEMA_PATH, str(ratings_path))
    summary = summarize(
        manifest,
        ratings,
        manifest_sha256=_sha256(manifest_path),
        export_sha256=_sha256(ratings_path),
    )
    _validate(summary, SUMMARY_SCHEMA_PATH, "generated summary")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ratings", type=Path, required=True, help="exactly one Round-2 export")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = score(args.manifest, args.ratings, args.output)
    print(json.dumps({"counts": summary["counts"], "coverage_rate": summary["coverage_rate"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
