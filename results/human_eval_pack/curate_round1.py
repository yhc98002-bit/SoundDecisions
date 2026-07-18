#!/usr/bin/env python3
"""Convert one complete Round-1 curation export into a fixed event catalog.

This is a single-curator definition step. It deliberately computes no agreement
statistic. The source manifest, exact item set, and export must all validate
before any catalog is written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from jsonschema import Draft202012Validator, FormatChecker


HERE = Path(__file__).resolve().parent
ROUND1_MANIFEST_SCHEMA_PATH = HERE / "release_src" / "round1_manifest.schema.json"
ROUND1_RATINGS_SCHEMA_PATH = HERE / "release_src" / "round1_ratings.schema.json"
EVENT_CATALOG_SCHEMA_PATH = HERE / "event_catalog.schema.json"


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: cannot read JSON: {exc}") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validator(schema_path: Path) -> Draft202012Validator:
    schema = _load_json(schema_path)
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema, format_checker=FormatChecker())


def _validate_schema(payload: Any, schema_path: Path, context: str) -> None:
    errors = sorted(
        _validator(schema_path).iter_errors(payload),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if errors:
        error = errors[0]
        location = "$" + "".join(
            f"[{part}]" if isinstance(part, int) else f".{part}"
            for part in error.absolute_path
        )
        raise ValueError(f"{context}: schema validation failed at {location}: {error.message}")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validated_interval(
    value: Mapping[str, Any], *, duration_s: float, context: str
) -> dict[str, float]:
    start = value.get("start_s")
    end = value.get("end_s")
    _require(
        isinstance(start, (int, float))
        and not isinstance(start, bool)
        and math.isfinite(float(start))
        and isinstance(end, (int, float))
        and not isinstance(end, bool)
        and math.isfinite(float(end)),
        f"{context}: interval must contain finite numeric bounds",
    )
    start_f, end_f = float(start), float(end)
    _require(0.0 <= start_f <= end_f <= duration_s, f"{context}: interval is outside the video")
    return {"start_s": start_f, "end_s": end_f}


def _event(
    blind_id: str,
    index: int,
    description: str,
    interval: Mapping[str, Any],
    duration_s: float,
    source: str,
) -> dict[str, Any]:
    clean_description = description.strip()
    _require(bool(clean_description), f"{blind_id}-E{index}: event description is empty")
    return {
        "event_id": f"{blind_id}-E{index}",
        "description": clean_description,
        "anchor": _validated_interval(
            interval, duration_s=duration_s, context=f"{blind_id}-E{index}"
        ),
        "event_source": source,
    }


def build_event_catalog(
    manifest: Mapping[str, Any],
    ratings: Mapping[str, Any],
    *,
    manifest_sha256: str,
    export_sha256: str,
) -> dict[str, Any]:
    """Validate exact joins and return a deterministic, single-curator catalog."""

    _require(ratings["manifest_id"] == manifest["manifest_id"], "manifest_id mismatch")
    _require(ratings["manifest_sha256"] == manifest_sha256, "manifest_sha256 mismatch")

    manifest_items = {item["blind_id"]: item for item in manifest["items"]}
    _require(len(manifest_items) == len(manifest["items"]), "manifest has duplicate blind IDs")
    expected_ids = set(manifest_items)
    _require(set(ratings["item_order"]) == expected_ids, "item_order does not match manifest")
    _require(len(ratings["item_order"]) == len(expected_ids), "item_order has duplicates")

    rating_items = {rating["blind_id"]: rating for rating in ratings["ratings"]}
    _require(len(rating_items) == len(ratings["ratings"]), "ratings have duplicate blind IDs")
    _require(set(rating_items) == expected_ids, "ratings item set does not match manifest")

    counts = {
        "items_total": len(expected_ids),
        "anchor_eligible": 0,
        "anchor_unresolved": 0,
        "pair_eligible": 0,
        "pair_rejected": 0,
        "events_eligible": 0,
    }
    output_items: list[dict[str, Any]] = []
    for source_item in manifest["items"]:
        blind_id = source_item["blind_id"]
        rating = rating_items[blind_id]
        _require(rating["completed"] is True, f"{blind_id}: curation is incomplete")
        _require(
            set(rating["tasks"]) == set(source_item["tasks"])
            and len(rating["tasks"]) == len(source_item["tasks"]),
            f"{blind_id}: task list does not match manifest",
        )
        duration_s = float(source_item["duration_s"])
        anchor_result: dict[str, Any] | None = None
        pair_result: dict[str, Any] | None = None
        anchor_events: list[dict[str, Any]] = []
        pair_events: list[dict[str, Any]] = []

        if "anchor_curation" in source_item["tasks"]:
            response = rating["anchor_curation"]
            description = source_item["candidate_caption"].strip()
            _require(bool(description), f"{blind_id}: candidate caption is empty")
            if response["status"] == "marked":
                interval = _validated_interval(
                    response, duration_s=duration_s, context=f"{blind_id}.anchor_curation"
                )
                anchor_result = {
                    "status": "eligible",
                    "description": description,
                    "anchor": interval,
                    "curator_note": response["note"],
                }
                anchor_events = [
                    _event(blind_id, 1, description, interval, duration_s, "anchor_curation")
                ]
                counts["anchor_eligible"] += 1
            elif response["status"] == "too_uncertain":
                anchor_result = {
                    "status": "unresolved",
                    "description": description,
                    "anchor": None,
                    "curator_note": response["note"],
                }
                counts["anchor_unresolved"] += 1
            else:
                raise ValueError(f"{blind_id}: completed anchor has invalid status")

        if "two_event_curation" in source_item["tasks"]:
            response = rating["two_event_curation"]
            if response["verdict"] == "confirm":
                _require(
                    isinstance(response["event_1"], dict)
                    and isinstance(response["event_2"], dict),
                    f"{blind_id}: confirmed pair is missing an event",
                )
                pair_events = [
                    _event(
                        blind_id,
                        index,
                        response[f"event_{index}"]["description"],
                        response[f"event_{index}"],
                        duration_s,
                        "two_event_curation",
                    )
                    for index in (1, 2)
                ]
                _require(
                    pair_events[0]["anchor"]["start_s"]
                    <= pair_events[1]["anchor"]["start_s"],
                    f"{blind_id}: event 1 must not start after event 2",
                )
                pair_result = {
                    "status": "eligible",
                    "events": pair_events,
                    "curator_note": response["note"],
                }
                counts["pair_eligible"] += 1
            elif response["verdict"] == "reject":
                pair_result = {
                    "status": "rejected",
                    "events": [],
                    "curator_note": response["note"],
                }
                counts["pair_rejected"] += 1
            else:
                raise ValueError(f"{blind_id}: completed two-event curation has invalid verdict")

        # A confirmed pair is the more specific event definition on overlap.
        selected_events = pair_events if pair_events else anchor_events
        counts["events_eligible"] += len(selected_events)
        output_items.append(
            {
                "blind_id": blind_id,
                "media_path": source_item["media_path"],
                "fps": float(source_item["fps"]),
                "duration_s": duration_s,
                "anchor_curation": anchor_result,
                "two_event_curation": pair_result,
                "events": selected_events,
            }
        )

    return {
        "schema_version": "sounddecisions-human-event-catalog-v1.0",
        "catalog_id": f"{manifest['manifest_id']}-event-catalog-v1",
        "source_manifest_id": manifest["manifest_id"],
        "source_manifest_sha256": manifest_sha256,
        "source_export_sha256": export_sha256,
        "curator_id": ratings["rater_id"],
        "curator_exported_at": ratings["exported_at"],
        "analysis_scope": "single_curator_event_definition_not_agreement_evidence",
        "event_selection_rule": "confirmed_two_event_precedes_marked_anchor_on_overlap",
        "counts": counts,
        "items": output_items,
    }


def curate(manifest_path: Path, ratings_path: Path, output_path: Path) -> dict[str, Any]:
    manifest = _load_json(manifest_path)
    ratings = _load_json(ratings_path)
    _validate_schema(manifest, ROUND1_MANIFEST_SCHEMA_PATH, str(manifest_path))
    _validate_schema(ratings, ROUND1_RATINGS_SCHEMA_PATH, str(ratings_path))
    catalog = build_event_catalog(
        manifest,
        ratings,
        manifest_sha256=_sha256(manifest_path),
        export_sha256=_sha256(ratings_path),
    )
    _validate_schema(catalog, EVENT_CATALOG_SCHEMA_PATH, "generated event catalog")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(catalog, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return catalog


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Round-1 manifest JSON")
    parser.add_argument("--ratings", type=Path, required=True, help="exactly one curator export")
    parser.add_argument("--output", type=Path, required=True, help="new event catalog JSON")
    args = parser.parse_args()
    catalog = curate(args.manifest, args.ratings, args.output)
    print(json.dumps(catalog["counts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
