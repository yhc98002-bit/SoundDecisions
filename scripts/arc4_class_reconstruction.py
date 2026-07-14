#!/usr/bin/env python3
"""Reconstruct the Phase-1 class map from raw measurement records.

This is a CPU-only audit. It streams the RunStore JSONL and joins records using
the structured ``extra`` fields written by ``phase1_commitment.py``. Generation
ids are never parsed. The committed Phase-1 artifacts are read-only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from foley_cw.agreement import categorical_agreement, confident_agreement  # noqa: E402
from foley_cw.determination import s_commit  # noqa: E402
from foley_cw.types import AgreementMetric  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MEASUREMENTS = ROOT / "results/stage0/measurements/measurements.jsonl"
COMMITTED_CSV = ROOT / "results/stage0/phase1/commitment_map_p1cfg1.csv"
MANIFEST = ROOT / "data/manifests/phase1_manifest_frozen.json"
THRESHOLDS = ROOT / "configs/thresholds.json"
OUT_JSON = ROOT / "results/arc4_wpA2/class_reconstruction.json"
OUT_MD = ROOT / "results/arc4_wpA2/class_reconstruction.md"

AXIS = "class"
TAG = "p1cfg1"
INDEPENDENT_ROLE = f"{TAG}_independent"
FORK_ROLE = f"{TAG}_fork"
S_GRID = (0.05, 0.15, 0.25, 0.35, 0.45, 0.60, 0.75, 0.90)
N_CLIPS = 200
N_INDEPENDENT = 16
K_FORKS = 12
S_READ = 0.75
ABSTAIN = "abstain"
TOL = 1e-9


class ReconstructionError(RuntimeError):
    """Raised when raw or committed artifacts violate the audit contract."""


def _skey(s: float) -> str:
    return f"{s:.2f}"


def _canonical_s(value) -> float:
    s = float(value)
    for expected in S_GRID:
        if math.isclose(s, expected, rel_tol=0.0, abs_tol=1e-12):
            return expected
    raise ReconstructionError(f"unexpected fork s={s!r}")


def _exact_int(value, field: str) -> int:
    parsed = int(value)
    if float(value) != parsed:
        raise ReconstructionError(f"{field} must be integral, got {value!r}")
    return parsed


def _insert_unique(dest: dict, key: tuple, label, line_no: int) -> None:
    if key in dest:
        raise ReconstructionError(f"duplicate structured join key {key!r} at line {line_no}")
    dest[key] = label


def stream_class_measurements(path: Path) -> tuple[dict, dict, int]:
    """Return independent/fork labels keyed only by structured ``extra`` data."""
    if not path.is_file():
        raise ReconstructionError(f"missing raw measurement journal: {path}")

    independent: dict[tuple[str, int], str] = {}
    forks: dict[tuple[str, float, int], str] = {}
    n_lines = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            n_lines += 1
            # Avoid decoding large embedding rows. Selection and joins below use
            # parsed structured fields; this prefilter is only an I/O optimization.
            if TAG not in raw or f'"{AXIS}"' not in raw:
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ReconstructionError(f"invalid JSON at line {line_no}: {exc}") from exc
            if record.get("axis_id") != AXIS:
                continue
            extra = record.get("extra")
            if not isinstance(extra, dict):
                continue
            role = extra.get("role")
            if role not in (INDEPENDENT_ROLE, FORK_ROLE):
                continue
            required = {"clip", "cfg", "role"}
            required |= {"j"} if role == INDEPENDENT_ROLE else {"s", "k"}
            missing = sorted(required - set(extra))
            if missing:
                raise ReconstructionError(
                    f"line {line_no} role={role!r} missing structured fields {missing}"
                )
            if not math.isclose(float(extra["cfg"]), 1.0, rel_tol=0.0, abs_tol=1e-12):
                raise ReconstructionError(f"line {line_no} has cfg={extra['cfg']!r}, expected 1.0")
            target = record.get("target")
            if not isinstance(target, dict) or target.get("kind") != "categorical":
                raise ReconstructionError(f"line {line_no} has a non-categorical class target")
            label = target.get("label")
            if not isinstance(label, str):
                raise ReconstructionError(f"line {line_no} has invalid class label {label!r}")
            clip = str(extra["clip"])
            if role == INDEPENDENT_ROLE:
                key = (clip, _exact_int(extra["j"], "j"))
                _insert_unique(independent, key, label, line_no)
            else:
                key = (clip, _canonical_s(extra["s"]), _exact_int(extra["k"], "k"))
                _insert_unique(forks, key, label, line_no)
    return independent, forks, n_lines


def manifest_clips(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    clips = [str(clip) for clip in data["clips"]["single_event"]]
    if len(clips) != N_CLIPS or len(set(clips)) != N_CLIPS:
        raise ReconstructionError(
            f"manifest must contain {N_CLIPS} unique single-event clips, got {len(clips)}"
        )
    return sorted(clips)


def validate_raw_cardinality(
    independent: dict, forks: dict, clips: list[str]
) -> dict:
    expected_independent = {
        (clip, j) for clip in clips for j in range(N_INDEPENDENT)
    }
    expected_forks = {
        (clip, s, k) for clip in clips for s in S_GRID for k in range(K_FORKS)
    }
    actual_independent = set(independent)
    actual_forks = set(forks)
    if actual_independent != expected_independent:
        missing = sorted(expected_independent - actual_independent)[:3]
        extra = sorted(actual_independent - expected_independent)[:3]
        raise ReconstructionError(
            f"independent join mismatch: missing={missing}, extra={extra}"
        )
    if actual_forks != expected_forks:
        missing = sorted(expected_forks - actual_forks)[:3]
        extra = sorted(actual_forks - expected_forks)[:3]
        raise ReconstructionError(f"fork join mismatch: missing={missing}, extra={extra}")
    return {
        "independent": {
            "expected": N_CLIPS * N_INDEPENDENT,
            "observed": len(independent),
            "unique_join_keys": len(actual_independent),
            "shape": [N_CLIPS, N_INDEPENDENT],
        },
        "fork": {
            "expected": N_CLIPS * len(S_GRID) * K_FORKS,
            "observed": len(forks),
            "unique_join_keys": len(actual_forks),
            "shape": [N_CLIPS, len(S_GRID), K_FORKS],
        },
    }


def _agreement_pair(labels: list[str]) -> tuple[float, float, int, int]:
    confident, n_confident = confident_agreement(
        labels, AgreementMetric.EXACT_MATCH, abstain=ABSTAIN
    )
    naive = categorical_agreement(labels)
    n_abstain = sum(label == ABSTAIN for label in labels)
    return float(confident), float(naive), int(n_confident), int(n_abstain)


def _commit_gain(a_fork: float, a_independent: float) -> float:
    if not (np.isfinite(a_fork) and np.isfinite(a_independent)):
        return float("nan")
    denominator = 1.0 - a_independent
    if denominator <= 1e-9:
        return 0.0
    return float(np.clip((a_fork - a_independent) / denominator, 0.0, 1.0))


def reconstruct(
    independent: dict, forks: dict, clips: list[str]
) -> dict:
    a_ind_confident: dict[str, float] = {}
    a_ind_naive: dict[str, float] = {}
    a_fork_confident: dict[str, dict[float, float]] = defaultdict(dict)
    a_fork_naive: dict[str, dict[float, float]] = defaultdict(dict)
    commit_confident: dict[str, dict[float, float]] = defaultdict(dict)
    commit_naive: dict[str, dict[float, float]] = defaultdict(dict)
    abstain_counts = {"independent": 0, **{_skey(s): 0 for s in S_GRID}}
    total_counts = {
        "independent": N_CLIPS * N_INDEPENDENT,
        **{_skey(s): N_CLIPS * K_FORKS for s in S_GRID},
    }
    unscorable = {"independent": 0, **{_skey(s): 0 for s in S_GRID}}

    for clip in clips:
        labels = [independent[(clip, j)] for j in range(N_INDEPENDENT)]
        confident, naive, _n_confident, n_abstain = _agreement_pair(labels)
        a_ind_confident[clip] = confident
        a_ind_naive[clip] = naive
        abstain_counts["independent"] += n_abstain
        unscorable["independent"] += int(not np.isfinite(confident))

        for s in S_GRID:
            labels = [forks[(clip, s, k)] for k in range(K_FORKS)]
            confident, naive, _n_confident, n_abstain = _agreement_pair(labels)
            a_fork_confident[clip][s] = confident
            a_fork_naive[clip][s] = naive
            commit_confident[clip][s] = _commit_gain(confident, a_ind_confident[clip])
            commit_naive[clip][s] = _commit_gain(naive, a_ind_naive[clip])
            key = _skey(s)
            abstain_counts[key] += n_abstain
            unscorable[key] += int(not np.isfinite(confident))

    return {
        "a_ind_confident_by_clip": a_ind_confident,
        "a_ind_naive_by_clip": a_ind_naive,
        "a_fork_confident_by_clip": a_fork_confident,
        "a_fork_naive_by_clip": a_fork_naive,
        "commit_confident_by_clip": commit_confident,
        "commit_naive_by_clip": commit_naive,
        "abstain_counts": abstain_counts,
        "total_counts": total_counts,
        "unscorable": unscorable,
    }


def _csv_float(value) -> float:
    if value in (None, "", "None", "nan", "NaN"):
        return float("nan")
    return float(value)


def validate_committed_csv(path: Path, clips: list[str], reconstruction: dict) -> dict:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("axis_id") == AXIS:
                rows.append(row)
    expected_keys = {(clip, s) for clip in clips for s in S_GRID}
    seen_keys: set[tuple[str, float]] = set()
    max_abs_delta = 0.0
    n_numeric_cells = 0
    n_nonfinite_matches = 0
    mismatches = []
    for row in rows:
        clip = str(row["clip"])
        s = _canonical_s(row["s"])
        key = (clip, s)
        if key in seen_keys:
            raise ReconstructionError(f"duplicate committed class row {key!r}")
        seen_keys.add(key)
        if not math.isclose(float(row["cfg"]), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ReconstructionError(f"committed row {key!r} has cfg={row['cfg']!r}")
        if not math.isclose(float(row["alpha"]), 0.8, rel_tol=0.0, abs_tol=1e-12):
            raise ReconstructionError(f"committed row {key!r} has alpha={row['alpha']!r}")
        if row["schedule"] != "sqrt_down":
            raise ReconstructionError(f"committed row {key!r} has schedule={row['schedule']!r}")
        expected = {
            "a_independent": reconstruction["a_ind_confident_by_clip"][clip],
            "a_fork": reconstruction["a_fork_confident_by_clip"][clip][s],
            "commit_gain": reconstruction["commit_confident_by_clip"][clip][s],
        }
        for field, got in expected.items():
            committed = _csv_float(row[field])
            n_numeric_cells += 1
            if np.isfinite(committed) and np.isfinite(got):
                delta = abs(committed - got)
                max_abs_delta = max(max_abs_delta, float(delta))
                if delta > TOL:
                    mismatches.append((clip, s, field, committed, got))
            elif not np.isfinite(committed) and not np.isfinite(got):
                n_nonfinite_matches += 1
            else:
                mismatches.append((clip, s, field, committed, got))
    if seen_keys != expected_keys:
        missing = sorted(expected_keys - seen_keys)[:3]
        extra = sorted(seen_keys - expected_keys)[:3]
        raise ReconstructionError(f"committed class row join mismatch: missing={missing}, extra={extra}")
    if mismatches:
        raise ReconstructionError(
            f"committed class reconstruction differs in {len(mismatches)} cells; "
            f"first={mismatches[0]!r}"
        )
    return {
        "reproduces_committed_csv": True,
        "max_abs_delta": float(max_abs_delta),
        "n_class_rows": len(rows),
        "n_numeric_cells": n_numeric_cells,
        "n_nonfinite_matches": n_nonfinite_matches,
        "tolerance": TOL,
    }


def _mean_finite(values) -> float:
    finite = [float(value) for value in values if np.isfinite(value)]
    if not finite:
        raise ReconstructionError("expected at least one finite value")
    return float(np.mean(finite))


def _mean_s_commit(curves: dict[str, dict[float, float]], theta: float) -> tuple[float, int]:
    crossings = [s_commit(curve, theta) for curve in curves.values()]
    finite = [value for value in crossings if np.isfinite(value)]
    if not finite:
        raise ReconstructionError("no class commitment curve crosses the threshold")
    return float(np.mean(finite)), len(finite)


def build_output(
    reconstruction: dict,
    validation: dict,
    cardinality: dict,
    n_measurement_lines: int,
    theta_commit: float,
) -> dict:
    a_ind_confident = _mean_finite(reconstruction["a_ind_confident_by_clip"].values())
    a_ind_naive = _mean_finite(reconstruction["a_ind_naive_by_clip"].values())
    s_commit_confident, n_cross_confident = _mean_s_commit(
        reconstruction["commit_confident_by_clip"], theta_commit
    )
    s_commit_naive, n_cross_naive = _mean_s_commit(
        reconstruction["commit_naive_by_clip"], theta_commit
    )
    curves = []
    for s in S_GRID:
        confident_fork = _mean_finite(
            reconstruction["a_fork_confident_by_clip"][clip][s]
            for clip in sorted(reconstruction["a_fork_confident_by_clip"])
        )
        naive_fork = _mean_finite(
            reconstruction["a_fork_naive_by_clip"][clip][s]
            for clip in sorted(reconstruction["a_fork_naive_by_clip"])
        )
        confident_commit = _mean_finite(
            reconstruction["commit_confident_by_clip"][clip][s]
            for clip in sorted(reconstruction["commit_confident_by_clip"])
        )
        naive_commit = _mean_finite(
            reconstruction["commit_naive_by_clip"][clip][s]
            for clip in sorted(reconstruction["commit_naive_by_clip"])
        )
        key = _skey(s)
        curves.append({
            "s": s,
            "a_fork_confident": confident_fork,
            "a_fork_naive": naive_fork,
            "commit_gain_confident": confident_commit,
            "commit_gain_naive": naive_commit,
            "abstain_rate": (
                reconstruction["abstain_counts"][key]
                / reconstruction["total_counts"][key]
            ),
            "n_unscorable_confident": reconstruction["unscorable"][key],
        })

    aggregate_deltas = [
        abs(a_ind_confident - a_ind_naive),
        abs(s_commit_confident - s_commit_naive),
    ]
    for row in curves:
        aggregate_deltas.extend((
            abs(row["a_fork_confident"] - row["a_fork_naive"]),
            abs(row["commit_gain_confident"] - row["commit_gain_naive"]),
        ))
    max_lens_delta = float(max(aggregate_deltas))
    naive_confident_differ = bool(max_lens_delta > 1e-12)
    if not naive_confident_differ:
        raise ReconstructionError("naive/confident power check failed: reconstructed lenses are identical")

    abstain_rate_by_s = {
        key: reconstruction["abstain_counts"][key] / reconstruction["total_counts"][key]
        for key in sorted(reconstruction["total_counts"])
    }
    n_unscorable = int(sum(reconstruction["unscorable"].values()))
    return {
        "_doc": (
            "Arc-4 WP-A2 class reconstruction from raw RunStore measurements. "
            "Joins use structured extra.clip/extra.j/extra.s/extra.k fields, never gen_id parsing."
        ),
        "a_ind_confident": a_ind_confident,
        "a_ind_naive": a_ind_naive,
        "abstain_counts_by_s": reconstruction["abstain_counts"],
        "abstain_rate_by_s": abstain_rate_by_s,
        "cardinality": cardinality,
        "curve_by_s": curves,
        "gap_confident": float(S_READ - s_commit_confident),
        "gap_naive": float(S_READ - s_commit_naive),
        "max_abs_confident_naive_delta": max_lens_delta,
        "max_abs_delta": validation["max_abs_delta"],
        "n_class_csv_rows": validation["n_class_rows"],
        "n_class_csv_numeric_cells": validation["n_numeric_cells"],
        "n_class_csv_nonfinite_matches": validation["n_nonfinite_matches"],
        "n_crossing_confident": n_cross_confident,
        "n_crossing_naive": n_cross_naive,
        "n_measurement_lines_streamed": n_measurement_lines,
        "n_unscorable_cells": n_unscorable,
        "naive_confident_differ": naive_confident_differ,
        "reproduces_committed_csv": validation["reproduces_committed_csv"],
        "s_commit_confident": s_commit_confident,
        "s_commit_naive": s_commit_naive,
        "s_read": S_READ,
        "theta_commit": theta_commit,
        "tolerance": validation["tolerance"],
        "unscorable_cells_by_s": reconstruction["unscorable"],
    }


def render_markdown(data: dict) -> str:
    card = data["cardinality"]
    lines = [
        "# Arc-4 WP-A2 class reconstruction",
        "",
        "Raw class measurements were streamed from the RunStore JSONL and joined using "
        "structured `extra` keys; `gen_id` was not parsed.",
        "",
        f"Cardinality: independent `{card['independent']['observed']}` = "
        f"`{N_CLIPS} x {N_INDEPENDENT}`; forks `{card['fork']['observed']}` = "
        f"`{N_CLIPS} x {len(S_GRID)} x {K_FORKS}`. All join keys are unique.",
        "",
        f"Committed class CSV reproduction: **PASS**, {data['n_class_csv_numeric_cells']} "
        f"numeric/NaN cells, maximum absolute delta `{data['max_abs_delta']:.3g}` "
        f"(tolerance `{data['tolerance']:.0e}`).",
        "",
        f"Naive-vs-confident power check: **{'PASS' if data['naive_confident_differ'] else 'FAIL'}**.",
        "",
        "| lens | A_ind | crossing clips | s_commit | s_read | gap |",
        "|---|---:|---:|---:|---:|---:|",
        f"| confident (abstains excluded) | {data['a_ind_confident']:.6f} | "
        f"{data['n_crossing_confident']} | {data['s_commit_confident']:.6f} | "
        f"{data['s_read']:.2f} | {data['gap_confident']:.6f} |",
        f"| naive (abstain is a label) | {data['a_ind_naive']:.6f} | "
        f"{data['n_crossing_naive']} | {data['s_commit_naive']:.6f} | "
        f"{data['s_read']:.2f} | {data['gap_naive']:.6f} |",
        "",
        f"Confident-subset unscorable cells: **{data['n_unscorable_cells']}**.",
        "",
        "| s | abstain rate | unscorable | A_fork confident | A_fork naive | "
        "commit confident | commit naive |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in data["curve_by_s"]:
        lines.append(
            f"| {row['s']:.2f} | {row['abstain_rate']:.6f} | "
            f"{row['n_unscorable_confident']} | {row['a_fork_confident']:.6f} | "
            f"{row['a_fork_naive']:.6f} | {row['commit_gain_confident']:.6f} | "
            f"{row['commit_gain_naive']:.6f} |"
        )
    lines.extend((
        "",
        f"Independent-label abstain rate: `{data['abstain_rate_by_s']['independent']:.6f}`; "
        f"independent unscorable cells: `{data['unscorable_cells_by_s']['independent']}`.",
    ))
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurements", type=Path, default=MEASUREMENTS)
    parser.add_argument("--committed-csv", type=Path, default=COMMITTED_CSV)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--thresholds", type=Path, default=THRESHOLDS)
    parser.add_argument("--out-json", type=Path, default=OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=OUT_MD)
    args = parser.parse_args(argv)

    clips = manifest_clips(args.manifest)
    independent, forks, n_lines = stream_class_measurements(args.measurements)
    cardinality = validate_raw_cardinality(independent, forks, clips)
    reconstruction = reconstruct(independent, forks, clips)
    validation = validate_committed_csv(args.committed_csv, clips, reconstruction)
    thresholds = json.loads(args.thresholds.read_text(encoding="utf-8"))
    theta_commit = float(thresholds["theta_commit"])
    output = build_output(reconstruction, validation, cardinality, n_lines, theta_commit)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    args.out_md.write_text(render_markdown(output), encoding="utf-8")
    print(
        f"wrote {args.out_json} and {args.out_md}; "
        f"raw={len(independent)}+{len(forks)}, max_delta={validation['max_abs_delta']:.3g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
