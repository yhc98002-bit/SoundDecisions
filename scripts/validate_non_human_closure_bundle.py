#!/usr/bin/env python3
"""Fail-closed validation for the committed non-human closure bundle."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable


EXPECTED_PREDICTIONS = 113_212
EXPECTED_FEATURE_UNITS = 6_528
REQUIRED = (
    "NON_HUMAN_TRACK_REPORT.md",
    "EXECUTION_STATUS.json",
    "CLASS_POSTERIOR_MEASUREMENT_REPORT.json",
    "CLASS_POSTERIOR_MEASUREMENT_REPORT.md",
    "CLASS_MULTISEED_COMMITMENT.json",
    "CLASS_MULTISEED_COMMITMENT.md",
    "CLASS_MULTISEED_COMMITMENT.csv",
    "CLASS_VARIANCE_DECOMPOSITION.json",
    "CLASS_VARIANCE_DECOMPOSITION.md",
    "CLASS_VIDEO_DETERMINED_SENSITIVITY.json",
    "FEATURE_LINEAGE_REPORT.json",
    "FEATURE_LINEAGE_REPORT.md",
    "CLASS_INTERNAL_READOUT_REPORT.json",
    "CLASS_INTERNAL_READOUT_REPORT.md",
    "CLASS_INTERNAL_READOUT_OUTER_PREDICTIONS.jsonl.gz",
    "MATERIAL_CONTINUITY_2AFC_REPORT.json",
    "MATERIAL_CONTINUITY_2AFC_REPORT.md",
    "MATERIAL_REFERENCE_INSUFFICIENCY.json",
    "NUMBERS_INDEX.json",
    "REPRO.json",
    "COMMANDS.md",
    "BUGS_DEVIATIONS_UNRESOLVED.md",
    "MATERIALIZATION_MANIFEST.json",
    "feature_manifests/FEATURE_RECOLLECTION_MANIFEST.jsonl",
    "feature_manifests/FEATURE_CHECKSUMS.sha256",
    "test_logs/CHECKSUMS.sha256",
    "CHECKSUMS.sha256",
)


class BundleError(RuntimeError):
    """Raised when closure evidence is missing, corrupt, or inconsistent."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise BundleError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BundleError(f"invalid JSON {path}: {exc}") from exc
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def parse_checksums(path: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split("  ", 1)
        require(len(parts) == 2, f"invalid checksum row {path}:{line_number}")
        digest, name = parts
        require(len(digest) == 64, f"invalid digest {path}:{line_number}")
        require(name not in rows, f"duplicate checksum target: {name}")
        rows[name] = digest
    return rows


def validate_checksum_file(root: Path, checksum_path: Path, *, exact: bool) -> int:
    rows = parse_checksums(checksum_path)
    for name, expected in rows.items():
        target = root / name
        require(target.is_file(), f"checksum target missing: {target}")
        require(sha256_file(target) == expected, f"checksum mismatch: {target}")
    if exact:
        actual = {
            str(path.relative_to(root))
            for path in root.rglob("*")
            if path.is_file() and path != checksum_path
        }
        require(set(rows) == actual, "global checksum coverage is not exact")
    return len(rows)


def validate_feature_manifest(path: Path) -> dict[str, int]:
    keys: set[tuple[str, int, float]] = set()
    videos: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise BundleError(f"invalid feature JSONL line {line_number}") from exc
            key = (str(row["video_id"]), int(row["base_seed"]), float(row["progress"]))
            require(key not in keys, f"duplicate feature unit: {key}")
            keys.add(key)
            videos.add(key[0])
    require(len(keys) == EXPECTED_FEATURE_UNITS, "feature manifest cardinality mismatch")
    require(len(videos) == 48, "feature manifest video count mismatch")
    return {"units": len(keys), "videos": len(videos)}


def validate_predictions(path: Path) -> dict[str, int]:
    keys: set[tuple[str, float, str, str]] = set()
    video_folds: dict[str, int] = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise BundleError(f"invalid prediction JSONL line {line_number}") from exc
            key = (
                str(row["target"]),
                float(row["progress"]),
                str(row["family"]),
                str(row["candidate_id"]),
            )
            require(key not in keys, f"duplicate outer prediction: {key}")
            keys.add(key)
            video = str(row["video_id"])
            fold = int(row["outer_fold"])
            if video in video_folds:
                require(video_folds[video] == fold, f"outer-fold video leakage: {video}")
            video_folds[video] = fold
            probabilities = [float(value) for value in row["probabilities"]]
            require(probabilities, f"empty probability vector: {key}")
            require(all(math.isfinite(value) and value >= 0.0 for value in probabilities), f"invalid probabilities: {key}")
            require(abs(sum(probabilities) - 1.0) <= 1e-8, f"probabilities do not sum to one: {key}")
    require(len(keys) == EXPECTED_PREDICTIONS, "outer prediction cardinality mismatch")
    require(len(video_folds) == 48, "outer prediction video count mismatch")
    return {"predictions": len(keys), "videos": len(video_folds)}


def validate_bundle(result_dir: Path) -> dict[str, Any]:
    result_dir = Path(result_dir).resolve()
    missing = [name for name in REQUIRED if not (result_dir / name).is_file()]
    require(not missing, f"required closure files missing: {missing}")

    status = load_json(result_dir / "EXECUTION_STATUS.json")
    require(status.get("status") == "COMPLETE", "execution status is not complete")
    require(status.get("sealed_confirmatory_cohort_used") is False, "sealed cohort used")
    require(status.get("b6_used") is False, "B6 used")

    posterior = load_json(result_dir / "CLASS_POSTERIOR_MEASUREMENT_REPORT.json")
    require(posterior.get("record_count") == 79_152, "posterior cardinality mismatch")
    commitment = load_json(result_dir / "CLASS_MULTISEED_COMMITMENT.json")
    require(
        commitment.get("replication_label") == "not_reproduced"
        and commitment.get("scientific_status") == "NOT_SUPPORTED",
        "Class decision mismatch",
    )
    sensitivity = load_json(result_dir / "CLASS_VIDEO_DETERMINED_SENSITIVITY.json")
    require(sensitivity.get("analysis_type") == "post_hoc_audit_sensitivity", "sensitivity mislabeled")
    require(sensitivity.get("changes_registered_rule_or_status") is False, "sensitivity changed frozen rule")
    require(sensitivity.get("nondetermined_only_sustained_crossing") == 0.60, "sensitivity crossing mismatch")

    lineage = load_json(result_dir / "FEATURE_LINEAGE_REPORT.json")
    require(lineage["same_forward_gate"]["status"] == "PASS", "B-1 gate mismatch")
    feature_counts = validate_feature_manifest(
        result_dir / "feature_manifests" / "FEATURE_RECOLLECTION_MANIFEST.jsonl"
    )

    readout = load_json(result_dir / "CLASS_INTERNAL_READOUT_REPORT.json")
    require(readout.get("prediction_count") == EXPECTED_PREDICTIONS, "readout count mismatch")
    prediction_path = result_dir / "CLASS_INTERNAL_READOUT_OUTER_PREDICTIONS.jsonl.gz"
    require(sha256_file(prediction_path) == readout.get("predictions_sha256"), "prediction hash mismatch")
    prediction_counts = validate_predictions(prediction_path)

    material = load_json(result_dir / "MATERIAL_CONTINUITY_2AFC_REPORT.json")
    require(material.get("scientific_status") == "INCOMPLETE_ARTIFACTS", "Material status mismatch")
    require(material["measurement"]["candidate_previews_replayed"] == 0, "invalid Material replay")
    require(material["measurement"]["two_afc_accuracy"] is None, "invented Material metric")

    numbers = load_json(result_dir / "NUMBERS_INDEX.json")
    require(numbers.get("status") == "COMPLETE" and len(numbers.get("numbers", [])) >= 30, "numbers index incomplete")
    repro = load_json(result_dir / "REPRO.json")
    require(repro.get("status") == "COMPLETE", "REPRO incomplete")

    feature_checksum_count = validate_checksum_file(
        result_dir / "feature_manifests",
        result_dir / "feature_manifests" / "FEATURE_CHECKSUMS.sha256",
        exact=False,
    )
    test_checksum_count = validate_checksum_file(
        result_dir / "test_logs",
        result_dir / "test_logs" / "CHECKSUMS.sha256",
        exact=False,
    )
    global_checksum_count = validate_checksum_file(
        result_dir,
        result_dir / "CHECKSUMS.sha256",
        exact=True,
    )
    return {
        "status": "PASS",
        "feature_manifest": feature_counts,
        "predictions": prediction_counts,
        "feature_checksums": feature_checksum_count,
        "test_log_checksums": test_checksum_count,
        "global_checksums": global_checksum_count,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(json.dumps(validate_bundle(args.result_dir), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
