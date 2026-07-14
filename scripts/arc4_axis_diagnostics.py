#!/usr/bin/env python3
"""Build the WP-A2 cross-axis validity and readout diagnostic table."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
AXES = ("presence", "timing", "class", "material")
TOLERANCE = 1e-9


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read required JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"required JSON is not an object: {path}")
    return value


def _load_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="") as handle:
            return list(csv.DictReader(handle))
    except OSError as exc:
        raise RuntimeError(f"cannot read required CSV {path}: {exc}") from exc


def _cell_key(row: dict[str, str]) -> tuple[str, str, str, float]:
    try:
        return (
            row["axis_id"],
            row["probe"],
            row["target"],
            float(row["s"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"invalid readout row: {row}") from exc


def _index_cells(
    rows: list[dict[str, str]], path: Path
) -> dict[tuple[str, str, str, float], dict[str, str]]:
    indexed: dict[tuple[str, str, str, float], dict[str, str]] = {}
    for row in rows:
        key = _cell_key(row)
        if key in indexed:
            raise RuntimeError(f"duplicate readout cell {key} in {path}")
        indexed[key] = row
    return indexed


def _validate_legacy(
    current: dict[tuple[str, str, str, float], dict[str, str]],
    reference_path: Path,
) -> float:
    reference = _index_cells(_load_csv(reference_path), reference_path)
    missing = sorted(set(reference) - set(current))
    extra = sorted(set(current) - set(reference))
    if missing or extra:
        raise RuntimeError(
            f"readout join failed against {reference_path}: "
            f"missing={missing[:3]}, extra={extra[:3]}"
        )

    max_delta = 0.0
    for key, old_row in reference.items():
        try:
            delta = abs(float(current[key]["accuracy"]) - float(old_row["accuracy"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"non-numeric pooled metric for {key} against {reference_path}"
            ) from exc
        if not math.isfinite(delta):
            raise RuntimeError(f"non-finite pooled-metric delta for {key}")
        max_delta = max(max_delta, delta)

    if max_delta > TOLERANCE:
        raise RuntimeError(
            f"readout pooled metric does not reproduce {reference_path}: "
            f"max_abs_delta={max_delta:.17g} > {TOLERANCE}"
        )
    return max_delta


def _required_axes(per_axis: Any, source: str) -> dict[str, dict[str, Any]]:
    if not isinstance(per_axis, dict):
        raise RuntimeError(f"{source}.per_axis is not an object")
    missing = [axis for axis in AXES if axis not in per_axis]
    extra = [axis for axis in per_axis if axis not in AXES]
    if missing or extra:
        raise RuntimeError(f"{source} axis mismatch: missing={missing}, extra={extra}")
    for axis, row in per_axis.items():
        if not isinstance(row, dict):
            raise RuntimeError(f"{source}.per_axis.{axis} is not an object")
    return per_axis


def _number(value: Any, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(numeric):
        return "n/a"
    return f"{numeric:.{digits}f}"


def _integer(value: Any) -> str:
    if not isinstance(value, int):
        raise RuntimeError(f"expected integer diagnostic count, got {value!r}")
    return str(value)


def build_report(
    axis_validity_path: Path,
    windows_path: Path,
    class_path: Path,
    readout_path: Path,
    wp_a_v2_path: Path,
    legacy_path: Path,
) -> str:
    validity = _load_json(axis_validity_path)
    windows = _load_json(windows_path)
    class_result = _load_json(class_path)
    validity_axes = _required_axes(validity.get("per_axis"), "axis_validity")
    window_axes = _required_axes(windows.get("per_axis"), "window_partitioned")

    readout_rows = _load_csv(readout_path)
    readout = _index_cells(readout_rows, readout_path)
    max_delta_v2 = _validate_legacy(readout, wp_a_v2_path)
    max_delta_legacy = _validate_legacy(readout, legacy_path)

    readout_axes = {key[0] for key in readout}
    if readout_axes != set(AXES):
        raise RuntimeError(
            f"readout axis mismatch: expected={list(AXES)}, actual={sorted(readout_axes)}"
        )
    s_grids = {
        axis: sorted(key[3] for key in readout if key[0] == axis)
        for axis in AXES
    }
    expected_grid = s_grids[AXES[0]]
    if not expected_grid or any(s_grids[axis] != expected_grid for axis in AXES):
        raise RuntimeError(f"readout s-grid mismatch: {s_grids}")
    for axis in AXES:
        for s in expected_grid:
            key = (axis, "audio_tagger", "ode", s)
            if key not in readout:
                raise RuntimeError(f"missing required readout cell {key}")

    class_curve_source = str(windows.get("class_curve_source", ""))
    if "confident" not in class_curve_source.lower():
        raise RuntimeError(
            "window_partitioned.class_curve_source must identify the confident "
            f"curve, got {class_curve_source!r}"
        )
    if class_result.get("reproduces_committed_csv") is not True:
        raise RuntimeError("class reconstruction does not reproduce the committed CSV")
    if class_result.get("naive_confident_differ") is not True:
        raise RuntimeError("class naive/confident comparison lacks the required power check")
    class_delta = float(class_result.get("max_abs_delta", math.inf))
    if not math.isfinite(class_delta) or class_delta > TOLERANCE:
        raise RuntimeError(
            f"class reconstruction max_abs_delta={class_delta} exceeds {TOLERANCE}"
        )

    lines = [
        "# Arc-4 WP-A2 axis diagnostics",
        "",
        (
            "This table combines Tier-0 axis validity, the determination-status "
            "partition, and the reconstructed Phase-2 readout map. The readout "
            "columns in the first table are the frozen earliest grid point, s=0.05; "
            "the complete trajectory follows."
        ),
        "",
        "## Cross-axis diagnostic",
        "",
        (
            "| axis | majority_share | k_eff | a_between_video | a_ind_mean | "
            "abstain_rate | video_determined | crossing | censored | "
            "readout_metric_s0.05 | readout_margin_s0.05 | "
            "balanced_accuracy_s0.05 | Tier-0 verdict |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for axis in AXES:
        tier = validity_axes[axis]
        partition = window_axes[axis]
        early = readout[(axis, "audio_tagger", "ode", 0.05)]
        counts = [
            partition.get("n_video_determined"),
            partition.get("n_crossing"),
            partition.get("n_censored"),
        ]
        if any(not isinstance(value, int) for value in counts) or sum(counts) != 200:
            raise RuntimeError(f"invalid determination partition for {axis}: {counts}")
        lines.append(
            "| "
            + " | ".join(
                [
                    axis,
                    _number(tier.get("majority_share")),
                    _number(tier.get("k_eff")),
                    _number(tier.get("a_between_video")),
                    _number(tier.get("a_ind_mean")),
                    _number(tier.get("abstain_rate")),
                    _integer(counts[0]),
                    _integer(counts[1]),
                    _integer(counts[2]),
                    _number(early.get("accuracy")),
                    _number(early.get("margin_over_majority")),
                    _number(early.get("balanced_accuracy")),
                    str(tier.get("verdict")),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "Categorical majority, k_eff, abstain, readout margin, and balanced "
            "accuracy are undefined for the continuous material axis and are shown "
            "as n/a. Material's readout metric remains legacy mean cosine here; it "
            "is not called accuracy.",
            "",
            "## Readout trajectory",
            "",
            (
                "| axis | s | metric | pooled value | ci_lo | ci_hi | "
                "majority baseline | margin over majority | balanced accuracy | "
                "bal_ci_lo | bal_ci_hi |"
            ),
            "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for axis in AXES:
        for s in expected_grid:
            row = readout[(axis, "audio_tagger", "ode", s)]
            lines.append(
                "| "
                + " | ".join(
                    [
                        axis,
                        _number(s, 2),
                        str(row.get("metric")),
                        _number(row.get("accuracy")),
                        _number(row.get("ci_lo")),
                        _number(row.get("ci_hi")),
                        _number(row.get("majority_baseline")),
                        _number(row.get("margin_over_majority")),
                        _number(row.get("balanced_accuracy")),
                        _number(row.get("bal_ci_lo")),
                        _number(row.get("bal_ci_hi")),
                    ]
                )
                + " |"
            )

    lines.extend(
        [
            "",
            "## Reconstruction checks",
            "",
            (
                f"- Phase-2 v3 pooled values reproduce WP-A v2 at all "
                f"{len(readout)} cells: max_abs_delta = {max_delta_v2:.17g} "
                f"(tolerance {TOLERANCE})."
            ),
            (
                f"- Phase-2 v3 pooled values reproduce the committed legacy source "
                f"at all {len(readout)} cells: max_abs_delta = "
                f"{max_delta_legacy:.17g} (tolerance {TOLERANCE})."
            ),
            (
                "- Class confident-subset reconstruction reproduces the committed "
                f"Phase-1 CSV: max_abs_delta = {class_delta:.17g}; "
                f"s_commit confident = {_number(class_result.get('s_commit_confident'))}, "
                f"naive = {_number(class_result.get('s_commit_naive'))}; "
                f"gap confident = {_number(class_result.get('gap_confident'))}, "
                f"naive = {_number(class_result.get('gap_naive'))}."
            ),
            "- Track-P internal probes remain outside this table because persisted "
            "per-example predictions are unavailable; retraining is required.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/arc4_wpA2/axis_diagnostics.md",
    )
    args = parser.parse_args()
    report = build_report(
        ROOT / "results/arc4_wpA2/axis_validity.json",
        ROOT / "results/arc4_wpA2/window_partitioned.json",
        ROOT / "results/arc4_wpA2/class_reconstruction.json",
        ROOT / "results/arc4_wpA2/readout_map_v3.csv",
        ROOT / "results/arc4_wpA/readout_map_p2cfg1_v2.csv",
        ROOT / "results/stage0/phase1/readout_map_p2cfg1.csv",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report)
    print(args.output)


if __name__ == "__main__":
    main()
