#!/usr/bin/env python3
"""Partition cached Phase-1 clips into video-determined, crossing, and censored sets."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from foley_cw.determination import clip_shares

INPUT_CSV = ROOT / "results/stage0/phase1/commitment_map_p1cfg1.csv"
LEGACY_CSV = ROOT / "results/stage0/phase1/determination_budget_p1cfg1.csv"
CLASS_PROOF = ROOT / "results/arc4_wpA2/class_reconstruction.json"
OUT_CSV = ROOT / "results/arc4_wpA2/determination_partition.csv"
OUT_JSON = ROOT / "results/arc4_wpA2/window_partitioned.json"

AXES = ("presence", "timing", "class", "material")
S_GRID = (0.05, 0.15, 0.25, 0.35, 0.45, 0.60, 0.75, 0.90)
THETA_COMMIT = 0.70
VIDEO_DETERMINED_MIN = 1.0 - 1e-9
CENSOR_TIME = 1.0
N_BOOT = 1000
BOOTSTRAP_SEED = 0


def _float(value: str | float | None) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def load_curves(path: Path) -> dict[str, dict[str, dict[str, object]]]:
    grouped: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if (float(row["cfg"]), float(row["alpha"]), row["schedule"]) != (
                1.0, 0.8, "sqrt_down"
            ):
                raise ValueError(f"unexpected Phase-1 provenance: {row}")
            axis = row["axis_id"]
            if axis not in AXES:
                continue
            clip = row["clip"]
            cell = grouped[axis].setdefault(
                clip, {"a_ind": _float(row["a_independent"]), "a_fork": {}}
            )
            a_ind = _float(row["a_independent"])
            if not math.isclose(float(cell["a_ind"]), a_ind, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"inconsistent a_independent for {axis}/{clip}")
            s = _float(row["s"])
            a_fork = cell["a_fork"]
            if s in a_fork:
                raise ValueError(f"duplicate curve cell for {axis}/{clip}/s={s}")
            a_fork[s] = _float(row.get("a_fork"))

    expected_grid = set(S_GRID)
    for axis in AXES:
        if len(grouped[axis]) != 200:
            raise ValueError(f"{axis}: expected 200 clips, found {len(grouped[axis])}")
        for clip, cell in grouped[axis].items():
            if set(cell["a_fork"]) != expected_grid:
                raise ValueError(f"{axis}/{clip}: incomplete s grid")
            cell["curve"] = clip_shares(
                float(cell["a_ind"]),
                cell.pop("a_fork"),
                is_embedding=(axis == "material"),
            )["commit"]
    return grouped


def first_crossing(curve: dict[float, float], theta: float = THETA_COMMIT) -> float | None:
    for s in S_GRID:
        value = float(curve[s])
        if math.isfinite(value) and value >= theta:
            return s
    return None


def partition(curves: dict[str, dict[str, dict[str, object]]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for axis in AXES:
        for clip in sorted(curves[axis]):
            cell = curves[axis][clip]
            a_ind = float(cell["a_ind"])
            crossing = first_crossing(cell["curve"])
            if a_ind >= VIDEO_DETERMINED_MIN:
                status, s_commit = "VIDEO_DETERMINED", None
            elif crossing is not None:
                status, s_commit = "CROSSING", crossing
            else:
                status, s_commit = "CENSORED", None
            rows.append({
                "axis_id": axis,
                "clip_id": clip,
                "a_ind": a_ind,
                "status": status,
                "s_commit": s_commit,
            })
    return rows


def km_median(rows: list[dict[str, object]]) -> float | None:
    observations = [
        (float(row["s_commit"]), True)
        if row["status"] == "CROSSING" else (CENSOR_TIME, False)
        for row in rows
        if row["status"] != "VIDEO_DETERMINED"
    ]
    if not observations:
        return None
    survival = 1.0
    at_risk = len(observations)
    for time in sorted({time for time, _ in observations}):
        events = sum(t == time and event for t, event in observations)
        censored = sum(t == time and not event for t, event in observations)
        if events:
            survival *= 1.0 - events / at_risk
            if survival <= 0.5:
                return float(time)
        at_risk -= events + censored
    return None


def bootstrap_km_ci(
    rows: list[dict[str, object]], axis: str, n_boot: int = N_BOOT
) -> tuple[float | None, float | None]:
    eligible = [row for row in rows if row["status"] != "VIDEO_DETERMINED"]
    if not eligible:
        return None, None
    seed = BOOTSTRAP_SEED + zlib.crc32(axis.encode("utf-8")) % 1000
    rng = np.random.default_rng(seed)
    medians: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(eligible), size=len(eligible))
        median = km_median([eligible[int(i)] for i in idx])
        if median is not None:
            medians.append(median)
    if not medians:
        return None, None
    lo, hi = np.quantile(np.asarray(medians), [0.025, 0.975])
    return float(lo), float(hi)


def load_legacy(path: Path) -> dict[str, float]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["axis"]: float(row["s_commit"]) for row in csv.DictReader(handle)}


def summarize(rows: list[dict[str, object]], legacy: dict[str, float]) -> dict[str, object]:
    per_axis: dict[str, dict[str, object]] = {}
    for axis in AXES:
        selected = [row for row in rows if row["axis_id"] == axis]
        counts = {status: sum(row["status"] == status for row in selected) for status in (
            "VIDEO_DETERMINED", "CROSSING", "CENSORED"
        )}
        cross_times = [float(row["s_commit"]) for row in selected if row["status"] == "CROSSING"]
        legacy_mean = float(np.mean(cross_times)) if cross_times else None
        if legacy_mean is None or not math.isclose(
            legacy_mean, legacy[axis], rel_tol=0.0, abs_tol=1e-9
        ):
            raise ValueError(
                f"{axis}: reconstructed legacy mean {legacy_mean} != committed {legacy[axis]}"
            )
        median = km_median(selected)
        ci_lo, ci_hi = bootstrap_km_ci(selected, axis)
        per_axis[axis] = {
            "n_video_determined": counts["VIDEO_DETERMINED"],
            "n_crossing": counts["CROSSING"],
            "n_censored": counts["CENSORED"],
            "km_median": median,
            "km_ci_lo": ci_lo,
            "km_ci_hi": ci_hi,
            "legacy_mean_crossers": legacy_mean,
        }
    return {
        "analysis": "Arc-4 WP-A2 determination-status partition",
        "source": str(INPUT_CSV.relative_to(ROOT)),
        "theta_commit": THETA_COMMIT,
        "video_determined_rule": "a_ind >= 1 - 1e-9",
        "censor_time": CENSOR_TIME,
        "km_population": "CROSSING plus CENSORED only; VIDEO_DETERMINED excluded",
        "km_ci_method": "1000-draw clip bootstrap, percentile 95% CI",
        "class_curve_source": "confident raw-journal reconstruction verified against committed CSV",
        "per_axis": per_axis,
    }


def write_outputs(rows: list[dict[str, object]], summary: dict[str, object]) -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["axis_id", "clip_id", "a_ind", "status", "s_commit"]
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({
                **row,
                "a_ind": f"{float(row['a_ind']):.17g}",
                "s_commit": "" if row["s_commit"] is None else f"{float(row['s_commit']):.2f}",
            })
    OUT_JSON.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    proof = json.loads(CLASS_PROOF.read_text(encoding="utf-8"))
    if proof.get("reproduces_committed_csv") is not True:
        raise SystemExit("class reconstruction has not verified the committed CSV")
    curves = load_curves(INPUT_CSV)
    rows = partition(curves)
    summary = summarize(rows, load_legacy(LEGACY_CSV))
    write_outputs(rows, summary)
    print(f"wrote {OUT_CSV.relative_to(ROOT)} and {OUT_JSON.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
