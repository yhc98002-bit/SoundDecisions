#!/usr/bin/env python
"""Arc-4 WP-A robustness lens for Phase-1 per-video commitment windows.

This is a CPU-only re-analysis of the cached headline (cfg=1, alpha=0.8)
Phase-1 commitment map.  It reports the legacy mean over videos that cross a
threshold alongside a median that treats a video that never crosses as
right-censored at s=1.  No generation, inference, or cached artifact mutation
is performed.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from foley_cw.determination import clip_shares, s_commit  # noqa: E402


AXES = ("timing", "presence", "material")
THETAS = (0.60, 0.65, 0.70, 0.75, 0.80)
CENSOR_S = 1.0


def _finite(value: str | None) -> float:
    if value in (None, "", "None", "nan"):
        return float("nan")
    return float(value)


def load_curves(path: Path) -> tuple[dict[str, dict[str, dict[float, float]]], dict[str, str]]:
    """Reconstruct normalized per-clip curves using the Phase-1 aggregation semantics."""
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    if not rows:
        raise ValueError(f"no rows in commitment map: {path}")

    forks: dict[str, dict[str, dict[float, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    a_independent: dict[str, dict[str, float]] = defaultdict(dict)
    provenance = {
        "cfg": rows[0]["cfg"],
        "alpha": rows[0]["alpha"],
        "schedule": rows[0]["schedule"],
    }
    for row in rows:
        axis = row["axis_id"]
        if axis not in AXES:
            continue
        if any(row[key] != provenance[key] for key in provenance):
            raise ValueError("commitment map mixes cfg, alpha, or schedule values")
        clip = row["clip"]
        progress = float(row["s"])
        if progress in forks[axis][clip]:
            raise ValueError(f"duplicate row for axis={axis}, clip={clip}, s={progress}")
        forks[axis][clip][progress] = _finite(row["a_fork"])
        a_independent[axis][clip] = _finite(row["a_independent"])

    clip_sets = {axis: set(forks[axis]) for axis in AXES}
    if not all(clip_sets.values()) or len({frozenset(v) for v in clip_sets.values()}) != 1:
        raise ValueError(f"axes do not share one complete clip cohort: {clip_sets}")
    grids = {
        tuple(sorted(forks[axis][clip]))
        for axis in AXES
        for clip in sorted(forks[axis])
    }
    if len(grids) != 1:
        raise ValueError(f"per-clip commitment grids are inconsistent: {sorted(grids)}")

    curves: dict[str, dict[str, dict[float, float]]] = defaultdict(dict)
    for axis in AXES:
        for clip in sorted(forks[axis]):
            shares = clip_shares(
                a_independent[axis][clip],
                forks[axis][clip],
                is_embedding=(axis == "material"),
            )
            curves[axis][clip] = shares["commit"]
    return dict(curves), provenance


def summarize_axis(curves: dict[str, dict[float, float]], theta: float) -> dict[str, float | int]:
    crossings = np.asarray(
        [s_commit(curves[clip], theta) for clip in sorted(curves)], dtype=float
    )
    finite = crossings[np.isfinite(crossings)]
    censored = np.where(np.isfinite(crossings), crossings, CENSOR_S)
    return {
        "n_clips": int(crossings.size),
        "n_cross": int(finite.size),
        "crossing_fraction": float(finite.size / crossings.size),
        "legacy_mean_of_crossers": (
            float(np.mean(finite)) if finite.size else float("nan")
        ),
        "censored_median": float(np.median(censored)),
    }


def _ordering(values: dict[str, float], atol: float = 1e-12) -> tuple[str, dict[str, int]]:
    """Return an ordering with exact-value ties and one-based competition ranks."""
    ordered = sorted(values, key=lambda axis: (values[axis], axis))
    groups: list[list[str]] = []
    for axis in ordered:
        if groups and math.isclose(
            values[axis], values[groups[-1][0]], rel_tol=0.0, abs_tol=atol
        ):
            groups[-1].append(axis)
        else:
            groups.append([axis])
    text = " < ".join(" = ".join(group) for group in groups)
    ranks: dict[str, int] = {}
    rank = 1
    for group in groups:
        for axis in group:
            ranks[axis] = rank
        rank += len(group)
    return text, ranks


def _load_legacy(path: Path) -> dict[str, float]:
    return {
        row["axis"]: _finite(row["s_commit"])
        for row in csv.DictReader(path.open(newline="", encoding="utf-8"))
    }


def analyze(curves: dict[str, dict[str, dict[float, float]]]) -> list[dict]:
    records: list[dict] = []
    by_theta: dict[float, dict[str, dict]] = {}
    for theta in THETAS:
        by_theta[theta] = {
            axis: summarize_axis(curves[axis], theta) for axis in AXES
        }

    reference_orders: dict[str, str] = {}
    for estimator in ("legacy_mean_of_crossers", "censored_median"):
        reference_orders[estimator] = _ordering(
            {axis: float(by_theta[0.70][axis][estimator]) for axis in AXES}
        )[0]

    for theta in THETAS:
        legacy_order, legacy_ranks = _ordering(
            {
                axis: float(by_theta[theta][axis]["legacy_mean_of_crossers"])
                for axis in AXES
            }
        )
        censored_order, censored_ranks = _ordering(
            {
                axis: float(by_theta[theta][axis]["censored_median"])
                for axis in AXES
            }
        )
        for axis in AXES:
            records.append(
                {
                    "theta_commit": theta,
                    "axis": axis,
                    **by_theta[theta][axis],
                    "legacy_rank": legacy_ranks[axis],
                    "censored_rank": censored_ranks[axis],
                    "legacy_order": legacy_order,
                    "legacy_order_stable_vs_theta_0.70": (
                        legacy_order == reference_orders["legacy_mean_of_crossers"]
                    ),
                    "censored_order": censored_order,
                    "censored_order_stable_vs_theta_0.70": (
                        censored_order == reference_orders["censored_median"]
                    ),
                }
            )
    return records


def _fmt(value: float) -> str:
    return "nan" if not math.isfinite(value) else f"{value:.6f}"


def write_csv(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "theta_commit",
        "axis",
        "n_clips",
        "n_cross",
        "crossing_fraction",
        "legacy_mean_of_crossers",
        "censored_median",
        "legacy_rank",
        "censored_rank",
        "legacy_order",
        "legacy_order_stable_vs_theta_0.70",
        "censored_order",
        "censored_order_stable_vs_theta_0.70",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def write_markdown(
    records: list[dict], path: Path, source: Path, provenance: dict[str, str]
) -> None:
    by_key = {(float(r["theta_commit"]), r["axis"]): r for r in records}
    lines = [
        "# Window-estimator robustness (Arc-4 WP-A)",
        "",
        "Diagnostic re-analysis of cached per-video Phase-1 commitment curves. "
        "The legacy estimator averages only clips that cross the threshold; the "
        "censored estimator assigns never-crossers s=1 before taking the cohort median.",
        "",
        f"Source: `{source}`; cfg={provenance['cfg']}, alpha={provenance['alpha']}, "
        f"schedule={provenance['schedule']}. Crossing uses the existing discrete "
        "earliest-grid-point rule (no interpolation).",
        "",
        "## Theta_commit = 0.70",
        "",
        "| axis | crossing fraction | legacy mean of crossers | censored median |",
        "|---|---:|---:|---:|",
    ]
    for axis in AXES:
        row = by_key[(0.70, axis)]
        lines.append(
            f"| {axis} | {row['n_cross']}/{row['n_clips']} "
            f"({_fmt(float(row['crossing_fraction']))}) | "
            f"{_fmt(float(row['legacy_mean_of_crossers']))} | "
            f"{_fmt(float(row['censored_median']))} |"
        )

    lines += [
        "",
        "## Threshold sweep and ordering stability",
        "",
        "Stability is an exact comparison to the estimator's ordering at theta=0.70; "
        "`=` denotes a tied estimator value.",
        "",
        "| theta | legacy ordering | stable | censored ordering | stable |",
        "|---:|---|:---:|---|:---:|",
    ]
    for theta in THETAS:
        row = by_key[(theta, AXES[0])]
        lines.append(
            f"| {theta:.2f} | {row['legacy_order']} | "
            f"{row['legacy_order_stable_vs_theta_0.70']} | "
            f"{row['censored_order']} | "
            f"{row['censored_order_stable_vs_theta_0.70']} |"
        )

    lines += [
        "",
        "## Sweep values",
        "",
        "| theta | axis | crossing fraction | legacy mean | censored median |",
        "|---:|---|---:|---:|---:|",
    ]
    for theta in THETAS:
        for axis in AXES:
            row = by_key[(theta, axis)]
            lines.append(
                f"| {theta:.2f} | {axis} | "
                f"{_fmt(float(row['crossing_fraction']))} | "
                f"{_fmt(float(row['legacy_mean_of_crossers']))} | "
                f"{_fmt(float(row['censored_median']))} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--commitment-map",
        type=Path,
        default=Path("results/stage0/phase1/commitment_map_p1cfg1.csv"),
    )
    parser.add_argument(
        "--legacy-budget",
        type=Path,
        default=Path("results/stage0/phase1/determination_budget_p1cfg1.csv"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/arc4_wpA")
    )
    args = parser.parse_args()

    curves, provenance = load_curves(args.commitment_map)
    records = analyze(curves)

    # Refuse to emit if the reconstructed theta=0.70 legacy estimator diverges
    # from the committed Phase-1 aggregate.
    legacy = _load_legacy(args.legacy_budget)
    for row in records:
        if float(row["theta_commit"]) != 0.70:
            continue
        axis = row["axis"]
        actual = float(row["legacy_mean_of_crossers"])
        expected = legacy[axis]
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                f"legacy mismatch for {axis}: reconstructed={actual}, committed={expected}"
            )

    csv_path = args.output_dir / "window_robustness.csv"
    md_path = args.output_dir / "window_robustness.md"
    write_csv(records, csv_path)
    write_markdown(records, md_path, args.commitment_map, provenance)
    print(f"wrote {csv_path} and {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
