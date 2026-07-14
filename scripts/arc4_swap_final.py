#!/usr/bin/env python3
"""Collision-corrected Stage-R swap analysis with two explicit estimands."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from foley_cw import condition_swap as CS
from foley_cw.types import AxisKind

JOURNAL_GLOB = "cswap__*.json"
LEGACY_CSV = ROOT / "results/stage0/stage_r/cond_swap_map_cswap.csv"
OUT_JSON = ROOT / "results/arc4_wpA2/swap_final.json"
AXES = ("presence", "timing", "class", "material")
KINDS = {axis: AxisKind.CATEGORICAL for axis in AXES}
KINDS["material"] = AxisKind.EMBEDDING


def cosine(a, b) -> float:
    av = np.asarray(a, dtype=float).ravel()
    bv = np.asarray(b, dtype=float).ravel()
    na, nb = np.linalg.norm(av), np.linalg.norm(bv)
    if na == 0.0 or nb == 0.0:
        raise ValueError("zero-norm material reference")
    return float(np.dot(av, bv) / (na * nb))


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n <= 0 or not 0 <= k <= n:
        raise ValueError(f"invalid binomial counts k={k}, n={n}")
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2.0, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return lo, hi


def load_records(journal_dir: Path) -> tuple[list[dict], dict[str, list[object]]]:
    paths = sorted(journal_dir.glob(JOURNAL_GLOB))
    if len(paths) != 40:
        raise FileNotFoundError(f"expected 40 swap journals, found {len(paths)}")
    records: list[dict] = []
    cohort: dict[str, list[object]] = defaultdict(list)
    seen_pairs = set()
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        pair = (str(data["source"]), str(data["donor"]))
        if pair in seen_pairs:
            raise ValueError(f"duplicate swap pair {pair}")
        seen_pairs.add(pair)
        for axis in AXES:
            source = data["source_val"][axis]
            donor = data["donor_val"][axis]
            cohort[axis].extend([source, donor])
            for s_key, swapped in sorted(data["swap_val"][axis].items(), key=lambda item: float(item[0])):
                records.append({
                    "axis_id": axis,
                    "s": float(s_key),
                    "source_clip": pair[0],
                    "donor_clip": pair[1],
                    "source": source,
                    "donor": donor,
                    "swapped": swapped,
                })
    return records, cohort


def validate_legacy(records: list[dict], legacy_csv: Path) -> dict[str, float | int]:
    grouped: dict[tuple[str, float], list[dict]] = defaultdict(list)
    for record in records:
        grouped[(record["axis_id"], record["s"])].append(record)
    legacy = list(csv.DictReader(legacy_csv.open(newline="", encoding="utf-8")))
    if set(grouped) != {(row["axis_id"], float(row["s"])) for row in legacy}:
        raise ValueError("journal/legacy swap cells do not match")
    max_delta = 0.0
    for row in legacy:
        key = (row["axis_id"], float(row["s"]))
        cell = grouped[key]
        rates = CS.follow_retention_rates(
            [r["swapped"] for r in cell],
            [r["donor"] for r in cell],
            [r["source"] for r in cell],
            KINDS[key[0]],
        )
        if rates["n"] != int(row["n"]):
            raise ValueError(f"legacy n mismatch for {key}")
        for old_key, new_key in (
            ("follow_rate", "follow"),
            ("retention_rate", "retention"),
            ("neither_rate", "neither"),
        ):
            delta = abs(float(row[old_key]) - float(rates[new_key]))
            max_delta = max(max_delta, delta)
            if delta > 1e-9:
                raise ValueError(f"legacy rate mismatch for {key}/{old_key}: {delta}")
    return {"reproduced_cells": len(legacy), "max_abs_delta": max_delta}


def pair_outcome(record: dict) -> tuple[bool, bool, bool]:
    if record["axis_id"] != "material":
        follow = record["swapped"] == record["donor"] and record["swapped"] != record["source"]
        retain = record["swapped"] == record["source"] and record["swapped"] != record["donor"]
    else:
        donor_sim = cosine(record["swapped"], record["donor"])
        source_sim = cosine(record["swapped"], record["source"])
        follow = donor_sim > source_sim
        retain = source_sim > donor_sim
    return bool(follow), bool(retain), bool(not follow and not retain)


def references_differ(record: dict) -> bool:
    if record["axis_id"] != "material":
        return record["donor"] != record["source"]
    return not np.array_equal(
        np.asarray(record["donor"], dtype=float), np.asarray(record["source"], dtype=float)
    )


def estimand(records: list[dict]) -> dict[str, float | int]:
    n = len(records)
    outcomes = [pair_outcome(record) for record in records]
    k_follow = sum(outcome[0] for outcome in outcomes)
    k_retain = sum(outcome[1] for outcome in outcomes)
    k_neither = sum(outcome[2] for outcome in outcomes)
    lo, hi = clopper_pearson(k_follow, n)
    return {
        "follow_only": k_follow / n,
        "follow_count": k_follow,
        "retention_only": k_retain / n,
        "retention_count": k_retain,
        "neither": k_neither / n,
        "neither_count": k_neither,
        "n": n,
        "ci_lo": lo,
        "ci_hi": hi,
    }


def categorical_floor(values: list[object]) -> float:
    labels = [str(value) for value in values]
    counts = Counter(labels)
    n = len(labels)
    return float(sum((count / n) ** 2 for count in counts.values()))


def analyze(records: list[dict], cohort: dict[str, list[object]]) -> dict:
    per_axis: dict[str, dict[str, dict]] = {}
    for axis in AXES:
        per_axis[axis] = {}
        axis_records = [record for record in records if record["axis_id"] == axis]
        for s in sorted({record["s"] for record in axis_records}):
            cell = [record for record in axis_records if record["s"] == s]
            distinct = [record for record in cell if references_differ(record)]
            per_axis[axis][f"{s:.2f}"] = {
                "n_collision": len(cell) - len(distinct),
                "unconditional": estimand(cell),
                "donor_ne_source": estimand(distinct),
            }

    class_cell = per_axis["class"]["0.05"]
    if class_cell["unconditional"]["follow_count"] != 8 or class_cell["unconditional"]["n"] != 20:
        raise ValueError(f"class s=.05 unconditional contract changed: {class_cell}")
    if class_cell["donor_ne_source"]["follow_count"] != 8 or class_cell["donor_ne_source"]["n"] != 17:
        raise ValueError(f"class s=.05 primary contract changed: {class_cell}")

    floors = {
        axis: (categorical_floor(cohort[axis]) if axis != "material" else 0.5)
        for axis in AXES
    }
    return {
        "analysis": "Arc-4 WP-A2 collision-corrected condition swap",
        "source": "results/stage0/journal/cswap__*.json",
        "primary_estimand": "donor_ne_source",
        "estimands": {
            "unconditional": {
                "axis_id": "class", "s": 0.05, **class_cell["unconditional"]
            },
            "donor_ne_source": {
                "axis_id": "class", "s": 0.05, **class_cell["donor_ne_source"]
            },
        },
        "material_rule": "nearest_reference",
        "material_chance_level": 0.5,
        "no_effect_floor": floors,
        "floor_basis": {
            "categorical": "sum of squared pooled source/donor cohort label marginals",
            "material": "symmetric two-alternative nearest-reference null",
        },
        "scale_note": "No binary-rate comparison is made against a mean-cosine ceiling; those scales are incompatible.",
        "per_axis": per_axis,
        "decision_token": None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--journal-dir", type=Path, default=ROOT / "results/stage0/journal")
    parser.add_argument("--legacy-csv", type=Path, default=LEGACY_CSV)
    parser.add_argument("--out", type=Path, default=OUT_JSON)
    args = parser.parse_args()
    records, cohort = load_records(args.journal_dir)
    output = analyze(records, cohort)
    output["legacy_validation"] = validate_legacy(records, args.legacy_csv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
