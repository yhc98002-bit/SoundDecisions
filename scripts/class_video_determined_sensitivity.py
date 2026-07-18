#!/usr/bin/env python3
"""Reproduce the post-hoc Class sensitivity excluding video-determined cases.

This diagnostic does not modify the frozen replication rule or status.  It
exists to make the effect of the registered A_ind >= 0.90 cases visible and
auditable from the immutable merged posterior artifact.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from foley_cw.b2_class_closure import _finite_mean, _first_crossing, sha256_file  # noqa: E402


class SensitivityError(RuntimeError):
    """Raised when the registered evidence binding is incomplete."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SensitivityError(message)


def optional_float(value: str) -> float:
    """Parse a scientific CSV scalar while preserving explicit missingness."""
    return float(value) if str(value).strip() else float("nan")


def nondetermined_sensitivity(
    cells: Sequence[Mapping[str, Any]],
    baselines: Sequence[Mapping[str, Any]],
    *,
    theta: float,
    reproduction_cutoff: float,
) -> dict[str, Any]:
    determined = {
        str(row["video_id"])
        for row in baselines
        if bool(row["video_determined"])
    }
    progress_grid = sorted({float(row["progress"]) for row in cells})
    all_curve = {
        progress: _finite_mean(
            float(row["commitment_gain"])
            for row in cells
            if math.isclose(float(row["progress"]), progress)
        )
        for progress in progress_grid
    }
    nondetermined_curve = {
        progress: _finite_mean(
            float(row["commitment_gain"])
            for row in cells
            if str(row["video_id"]) not in determined
            and math.isclose(float(row["progress"]), progress)
        )
        for progress in progress_grid
    }
    all_crossing = _first_crossing(all_curve, theta, sustained=True)
    nondetermined_crossing = _first_crossing(
        nondetermined_curve, theta, sustained=True
    )
    return {
        "theta": float(theta),
        "reproduction_cutoff": float(reproduction_cutoff),
        "video_count": len({str(row["video_id"]) for row in baselines}),
        "video_determined_count": len(determined),
        "video_determined_ids": sorted(determined),
        "registered_all_cell_curve": [
            {"progress": progress, "mean_commitment_gain": all_curve[progress]}
            for progress in progress_grid
        ],
        "registered_all_cell_sustained_crossing": all_crossing,
        "nondetermined_only_curve": [
            {
                "progress": progress,
                "mean_commitment_gain": nondetermined_curve[progress],
            }
            for progress in progress_grid
        ],
        "nondetermined_only_sustained_crossing": nondetermined_crossing,
        "nondetermined_only_reproduces_by_cutoff": bool(
            nondetermined_crossing is not None
            and nondetermined_crossing <= reproduction_cutoff
        ),
    }


def build_report(
    completion_path: Path,
    registered_analysis_path: Path,
    commitment_csv_path: Path,
) -> dict[str, Any]:
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    require(isinstance(completion, dict), "merged completion is not an object")
    require(
        completion.get("status") == "COMPLETE"
        and int(completion.get("record_count", -1)) == 79_152,
        "unexpected merged posterior completion",
    )
    posterior_path = completion_path.parent / str(completion.get("data_file", ""))
    require(posterior_path.is_file(), "merged posterior data is missing")
    require(
        sha256_file(posterior_path) == completion.get("data_sha256"),
        "merged posterior data hash mismatch",
    )
    registered = json.loads(registered_analysis_path.read_text(encoding="utf-8"))
    require(isinstance(registered, dict), "registered analysis is not an object")
    require(
        registered.get("replication_label") == "not_reproduced"
        and registered.get("scientific_status") == "NOT_SUPPORTED",
        "unexpected frozen replication decision",
    )
    cells: list[dict[str, Any]] = []
    baseline_by_video: dict[str, float] = {}
    keys: set[tuple[str, int, float]] = set()
    with commitment_csv_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            video = str(row["video_id"])
            base_seed = int(row["base_seed"])
            progress = float(row["progress"])
            baseline = optional_float(row["a_independent"])
            key = (video, base_seed, progress)
            require(key not in keys, f"duplicate commitment cell: {key}")
            keys.add(key)
            if video in baseline_by_video:
                require(
                    (
                        not math.isfinite(baseline_by_video[video])
                        and not math.isfinite(baseline)
                    )
                    or math.isclose(baseline_by_video[video], baseline),
                    f"inconsistent baseline for video {video}",
                )
            baseline_by_video[video] = baseline
            cells.append(
                {
                    "video_id": video,
                    "base_seed": base_seed,
                    "progress": progress,
                    "commitment_gain": optional_float(row["commitment_gain"]),
                }
            )
    require(len(cells) == 6_528, "commitment CSV cardinality mismatch")
    require(len(baseline_by_video) == 48, "commitment CSV video count mismatch")
    require(
        len({int(row["base_seed"]) for row in cells}) == 17,
        "commitment CSV seed count mismatch",
    )
    require(
        len({float(row["progress"]) for row in cells}) == 8,
        "commitment CSV progress count mismatch",
    )
    baselines = [
        {
            "video_id": video,
            "a_independent": baseline,
            "video_determined": math.isfinite(baseline) and baseline >= 0.90,
        }
        for video, baseline in sorted(baseline_by_video.items())
    ]
    sensitivity = nondetermined_sensitivity(
        cells,
        baselines,
        theta=0.70,
        reproduction_cutoff=0.45,
    )
    frozen_crossing = registered["replication_classification"][
        "pooled_sustained_crossing_theta_0.70"
    ]
    require(
        sensitivity["registered_all_cell_sustained_crossing"] == frozen_crossing,
        "recomputed all-cell crossing differs from frozen analysis",
    )
    require(
        sensitivity["video_determined_count"]
        == registered["baseline_summary"]["n_video_determined"],
        "video-determined count differs from frozen analysis",
    )
    return {
        "schema": "sounddecisions.class_video_determined_sensitivity.v1",
        "analysis_type": "post_hoc_audit_sensitivity",
        "changes_registered_rule_or_status": False,
        "registered_scientific_status": "NOT_SUPPORTED",
        "interpretation": (
            "Excluding registered video-determined cases changes the pooled "
            "sustained point estimate but still does not meet the frozen "
            "s<=0.45 reproduction cutoff."
        ),
        "source": {
            "merged_completion": str(completion_path.resolve()),
            "merged_completion_sha256": sha256_file(completion_path),
            "posterior_data_sha256": completion["data_sha256"],
            "registered_analysis": str(registered_analysis_path.resolve()),
            "registered_analysis_sha256": sha256_file(registered_analysis_path),
            "commitment_cells": str(commitment_csv_path.resolve()),
            "commitment_cells_sha256": sha256_file(commitment_csv_path),
            "script_sha256": sha256_file(Path(__file__)),
        },
        **sensitivity,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merged-completion", type=Path, required=True)
    parser.add_argument("--registered-analysis", type=Path, required=True)
    parser.add_argument("--commitment-csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_report(
        args.merged_completion, args.registered_analysis, args.commitment_csv
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": "COMPLETE", "out": str(args.out)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
