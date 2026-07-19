#!/usr/bin/env python3
"""Compute fixed-scale Gwet AC1 from two or more independent Round-2 exports."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from score_ac1 import gwet_ac1  # noqa: E402
from score_round2 import (  # noqa: E402
    MANIFEST_SCHEMA_PATH,
    RATINGS_SCHEMA_PATH,
    _load_json_snapshot,
    _validate,
    _write_new_json,
    validate_export,
)


OUTPUT_SCHEMA_PATH = HERE / "release_src" / "round2_ac1.schema.json"
VERDICT_CATEGORIES = ("target_present", "absent", "uncertain")
BACKGROUND_CATEGORIES = (False, True)


def compute_report(
    manifest_path: Path, ratings_paths: list[Path]
) -> dict[str, Any]:
    if len(ratings_paths) < 2:
        raise ValueError("Round-2 AC1 requires at least two rating exports")
    manifest, manifest_sha256 = _load_json_snapshot(manifest_path)
    _validate(manifest, MANIFEST_SCHEMA_PATH, str(manifest_path))

    exports: list[tuple[Path, dict[str, Any], str]] = []
    normalized_raters: set[str] = set()
    for path in ratings_paths:
        ratings, ratings_sha256 = _load_json_snapshot(path)
        _validate(ratings, RATINGS_SCHEMA_PATH, str(path))
        validate_export(manifest, ratings, manifest_sha256)
        normalized = ratings["rater_id"].strip().lower()
        if normalized in normalized_raters:
            raise ValueError("Round-2 AC1 requires distinct unique rater IDs")
        normalized_raters.add(normalized)
        exports.append((path, ratings, ratings_sha256))

    verdict_by_item: dict[str, list[str]] = defaultdict(list)
    background_by_item: dict[str, list[bool]] = defaultdict(list)
    for _, ratings, _ in exports:
        for rating in ratings["ratings"]:
            if not rating["completed"]:
                continue
            verdict_by_item[rating["event_id"]].append(rating["presence"]["verdict"])
            background_by_item[rating["event_id"]].append(
                rating["presence"]["unrelated_background"]
            )

    source_exports = sorted(
        (
            {"rater_id": ratings["rater_id"], "sha256": ratings_sha256}
            for _, ratings, ratings_sha256 in exports
        ),
        key=lambda row: row["rater_id"].lower(),
    )
    report = {
        "schema_version": "sounddecisions-human-presence-ac1-v1.0",
        "analysis_scope": "multi_rater_interrater_agreement",
        "source_manifest_id": manifest["manifest_id"],
        "source_manifest_sha256": manifest_sha256,
        "n_raters": len(exports),
        "source_exports": source_exports,
        "questions": {
            "presence_verdict": gwet_ac1(
                verdict_by_item, categories=VERDICT_CATEGORIES
            ),
            "unrelated_background": gwet_ac1(
                background_by_item, categories=BACKGROUND_CATEGORIES
            ),
        },
        "interpretation": (
            "Inter-rater reliability only; this report does not establish semantic "
            "correctness."
        ),
    }
    _validate(report, OUTPUT_SCHEMA_PATH, "generated Round-2 AC1 report")
    return report


def score(manifest_path: Path, ratings_paths: list[Path], output_path: Path) -> dict[str, Any]:
    report = compute_report(manifest_path, ratings_paths)
    _write_new_json(output_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ratings", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = score(args.manifest, args.ratings, args.output)
    print(json.dumps(report["questions"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
