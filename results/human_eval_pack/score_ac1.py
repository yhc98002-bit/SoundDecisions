#!/usr/bin/env python3
"""Score blinded v2 human ratings without unblinding them.

The scorer accepts two or more exports produced by ``rate.html``. It reports
Gwet's AC1 for each categorical question, pairwise interval-overlap summaries,
and a deterministic, task-stratified 20% audit selection. No source clip IDs,
conditions, or model information are required or read.

Only non-null answers contribute, so a partially completed export cannot turn
missing fields into agreement. A question needs at least two ratings on an item
to enter its calculation. For unequal/missing rater counts, observed agreement
is the mean within-item pairwise agreement and AC1 marginals are the mean of the
eligible items' response distributions.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys
from typing import Any, Hashable, Iterable, Mapping, Sequence

from jsonschema import Draft202012Validator, FormatChecker


RATING_SCHEMA_VERSION = "sounddecisions-human-eval-v2-1.0"
SCORE_SCHEMA_VERSION = "sounddecisions-human-eval-score-v1.0"
AUDIT_FRACTION = 0.20
AUDIT_NAMESPACE = "sounddecisions-human-audit-v1"
VALID_TASKS = {"anchor_presence", "two_event"}
QUESTION_CATEGORIES: dict[str, tuple[Hashable, ...]] = {
    "anchor_status": ("marked", "too_uncertain"),
    "presence_verdict": ("target_present", "absent", "uncertain"),
    "unrelated_background": (False, True),
    "pair_curation_verdict": ("confirm", "reject"),
}
INSTRUMENT_VERSION = "human-eval-pack-1.0"
BLIND_ID_RE = re.compile(r"HEV2-[0-9A-F]{12}")
RATER_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
RFC3339_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}[Tt]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:[Zz]|[+-]\d{2}:\d{2})"
)
RATING_SCHEMA_PATH = Path(__file__).with_name("ratings.schema.json")
RATING_SCHEMA = json.loads(RATING_SCHEMA_PATH.read_text(encoding="utf-8"))
Draft202012Validator.check_schema(RATING_SCHEMA)
FORMAT_CHECKER = FormatChecker()


@FORMAT_CHECKER.checks("date-time", raises=ValueError)
def _is_rfc3339(value: object) -> bool:
    if not isinstance(value, str) or RFC3339_RE.fullmatch(value) is None:
        return False
    datetime.fromisoformat(value.replace("z", "+00:00").replace("Z", "+00:00"))
    return True


RATING_VALIDATOR = Draft202012Validator(
    RATING_SCHEMA,
    format_checker=FORMAT_CHECKER,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate_partial_interval(value: Any, context: str) -> None:
    _require(isinstance(value, dict), f"{context}: interval must be an object")
    _require(set(value) == {"start_s", "end_s"}, f"{context}: interval fields are invalid")
    for name in ("start_s", "end_s"):
        bound = value[name]
        _require(
            bound is None or (_is_number(bound) and math.isfinite(float(bound)) and float(bound) >= 0),
            f"{context}.{name}: expected a finite non-negative number or null",
        )
    if value["start_s"] is not None and value["end_s"] is not None:
        _require(value["start_s"] <= value["end_s"], f"{context}: start_s exceeds end_s")


def _validate_rating(rating: Mapping[str, Any], context: str) -> None:
    blind_id = rating.get("blind_id")
    _require(isinstance(blind_id, str) and BLIND_ID_RE.fullmatch(blind_id) is not None,
             f"{context}.blind_id is invalid")
    tasks = rating.get("tasks")
    _require(
        isinstance(tasks, list)
        and 1 <= len(tasks) <= len(VALID_TASKS)
        and len(tasks) == len(set(tasks))
        and set(tasks) <= VALID_TASKS,
        f"{context}.tasks is invalid",
    )
    completed = rating.get("completed")
    _require(isinstance(completed, bool), f"{context}.completed must be boolean")

    complete = True
    if "anchor_presence" in tasks:
        anchor = rating.get("anchor")
        presence = rating.get("presence")
        _require(isinstance(anchor, dict), f"{context}.anchor must be an object")
        _require(
            set(anchor) == {"status", "start_s", "end_s", "event_description"},
            f"{context}.anchor fields are invalid",
        )
        _require(anchor["status"] in {"marked", "too_uncertain", "unrated"},
                 f"{context}.anchor.status is invalid")
        _require(isinstance(anchor["event_description"], str)
                 and len(anchor["event_description"]) <= 500,
                 f"{context}.anchor.event_description is invalid")
        _validate_partial_interval(
            {"start_s": anchor["start_s"], "end_s": anchor["end_s"]},
            f"{context}.anchor",
        )
        if anchor["status"] == "marked":
            _require(anchor["start_s"] is not None and anchor["end_s"] is not None,
                     f"{context}: marked anchor is incomplete")
        if anchor["status"] == "too_uncertain":
            pass

        _require(isinstance(presence, dict), f"{context}.presence must be an object")
        _require(set(presence) == {"verdict", "unrelated_background", "note"},
                 f"{context}.presence fields are invalid")
        _require(presence["verdict"] in {"target_present", "absent", "uncertain", None},
                 f"{context}.presence.verdict is invalid")
        _require(presence["unrelated_background"] is None
                 or isinstance(presence["unrelated_background"], bool),
                 f"{context}.presence.unrelated_background is invalid")
        _require(isinstance(presence["note"], str) and len(presence["note"]) <= 2000,
                 f"{context}.presence.note is invalid")
        anchor_complete = (
            bool(anchor["event_description"].strip())
            and (
                anchor["status"] == "too_uncertain"
                or (
                    anchor["status"] == "marked"
                    and anchor["start_s"] is not None
                    and anchor["end_s"] is not None
                )
            )
        )
        complete = complete and anchor_complete and presence["verdict"] is not None \
            and isinstance(presence["unrelated_background"], bool)

    if "two_event" in tasks:
        pair = rating.get("pair_curation")
        _require(isinstance(pair, dict), f"{context}.pair_curation must be an object")
        _require(
            set(pair) == {
                "verdict", "event_1", "event_2", "event_1_description",
                "event_2_description", "note",
            },
            f"{context}.pair_curation fields are invalid",
        )
        _require(pair["verdict"] in {"confirm", "reject", None},
                 f"{context}.pair_curation.verdict is invalid")
        for field in ("event_1_description", "event_2_description"):
            _require(isinstance(pair[field], str) and len(pair[field]) <= 500,
                     f"{context}.pair_curation.{field} is invalid")
        _require(isinstance(pair["note"], str) and len(pair["note"]) <= 2000,
                 f"{context}.pair_curation.note is invalid")
        for field in ("event_1", "event_2"):
            if pair[field] is not None:
                _validate_partial_interval(pair[field], f"{context}.pair_curation.{field}")
        if pair["verdict"] is None:
            _require(pair["event_1"] is None and pair["event_2"] is None,
                     f"{context}: unrated pair must have null intervals")
        pair_complete = pair["verdict"] == "reject"
        if pair["verdict"] == "confirm" and pair["event_1"] and pair["event_2"]:
            event_1, event_2 = pair["event_1"], pair["event_2"]
            pair_complete = (
                event_1["start_s"] is not None
                and event_1["end_s"] is not None
                and event_2["start_s"] is not None
                and event_2["end_s"] is not None
                and event_1["start_s"] <= event_2["start_s"]
                and bool(pair["event_1_description"].strip())
                and bool(pair["event_2_description"].strip())
            )
        complete = complete and pair_complete

    _require(not completed or complete, f"{context}: completed rating is incomplete")


def load_rating_export(path: str | Path) -> dict[str, Any]:
    """Load one export and validate both its schema and scoring semantics."""

    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: cannot read rating JSON: {exc}") from exc

    schema_errors = sorted(
        RATING_VALIDATOR.iter_errors(payload),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if schema_errors:
        error = schema_errors[0]
        location = "$" + "".join(
            f"[{part}]" if isinstance(part, int) else f".{part}"
            for part in error.absolute_path
        )
        raise ValueError(f"{path}: schema validation failed at {location}: {error.message}")

    _require(isinstance(payload, dict), f"{path}: top level must be an object")
    _require(
        payload.get("schema_version") == RATING_SCHEMA_VERSION,
        f"{path}: expected schema_version {RATING_SCHEMA_VERSION!r}",
    )
    for field in (
        "instrument_version",
        "manifest_id",
        "manifest_sha256",
        "rater_id",
        "started_at",
        "exported_at",
        "item_order",
        "ratings",
    ):
        _require(field in payload, f"{path}: missing required field {field!r}")

    _require(payload["instrument_version"] == INSTRUMENT_VERSION,
             f"{path}: instrument_version is invalid")
    _require(isinstance(payload["manifest_id"], str) and bool(payload["manifest_id"]),
             f"{path}: manifest_id must be non-empty")
    _require(isinstance(payload["manifest_sha256"], str)
             and re.fullmatch(r"[0-9a-f]{64}", payload["manifest_sha256"]) is not None,
             f"{path}: manifest_sha256 is invalid")
    _require(
        isinstance(payload["rater_id"], str)
        and RATER_ID_RE.fullmatch(payload["rater_id"]) is not None,
        f"{path}: rater_id is invalid",
    )
    _require(isinstance(payload["item_order"], list), f"{path}: item_order must be a list")
    _require(isinstance(payload["ratings"], list), f"{path}: ratings must be a list")

    order = payload["item_order"]
    _require(
        all(isinstance(item, str) and BLIND_ID_RE.fullmatch(item) is not None for item in order),
        f"{path}: item_order entries must be blinded IDs",
    )
    _require(len(order) == len(set(order)), f"{path}: item_order contains duplicate blind IDs")
    order_set = set(order)

    seen: set[str] = set()
    for index, rating in enumerate(payload["ratings"]):
        prefix = f"{path}: ratings[{index}]"
        _require(isinstance(rating, dict), f"{prefix} must be an object")
        _validate_rating(rating, prefix)
        blind_id = rating["blind_id"]
        _require(blind_id not in seen, f"{path}: duplicate rating for {blind_id!r}")
        _require(blind_id in order_set, f"{path}: rating {blind_id!r} is absent from item_order")
        seen.add(blind_id)
    _require(seen == order_set, f"{path}: item_order and ratings contain different IDs")

    return payload


def _eligible_item_ratings(
    item_ratings: Mapping[str, Sequence[Hashable]],
) -> dict[str, list[Hashable]]:
    return {
        item_id: list(values)
        for item_id, values in item_ratings.items()
        if len(values) >= 2
    }


def gwet_ac1(
    item_ratings: Mapping[str, Sequence[Hashable]],
    *,
    categories: Sequence[Hashable] | None = None,
) -> dict[str, Any]:
    """Compute nominal multi-rater Gwet AC1 from ratings grouped by item.

    The chance term is ``sum(p_k * (1 - p_k)) / (q - 1)`` over the question's
    fixed response scale. Marginals are averaged per item so an item with an
    extra non-missing rater cannot receive extra weight. The output includes
    enough components to audit the coefficient.
    """

    eligible = _eligible_item_ratings(item_ratings)
    if not eligible:
        return {
            "ac1": None,
            "observed_agreement": None,
            "expected_agreement": None,
            "n_items": 0,
            "n_ratings": 0,
            "categories": [],
        }

    realised = {value for values in eligible.values() for value in values}
    category_values = list(categories) if categories is not None else sorted(realised, key=repr)
    _require(len(category_values) == len(set(category_values)), "AC1 categories must be unique")
    _require(realised <= set(category_values), "rating outside the fixed AC1 response scale")
    _require(len(category_values) >= 2, "AC1 requires a response scale with at least two categories")

    per_item_agreement: list[float] = []
    per_item_marginals: list[dict[Hashable, float]] = []
    pooled_counts: Counter = Counter()
    for item_id in sorted(eligible):
        values = eligible[item_id]
        counts = Counter(values)
        n = len(values)
        agreeing_ordered_pairs = sum(count * (count - 1) for count in counts.values())
        per_item_agreement.append(agreeing_ordered_pairs / (n * (n - 1)))
        per_item_marginals.append({value: counts[value] / n for value in category_values})
        pooled_counts.update(values)

    observed = statistics.fmean(per_item_agreement)
    marginals = {
        value: statistics.fmean(row[value] for row in per_item_marginals)
        for value in category_values
    }
    expected = sum(probability * (1.0 - probability) for probability in marginals.values()) \
        / (len(category_values) - 1)
    denominator = 1.0 - expected
    coefficient = 0.0 if denominator <= 1e-12 else (observed - expected) / denominator

    category_rows = [
        {
            "value": value,
            "count": pooled_counts[value],
            "item_balanced_marginal": float(marginals[value]),
        }
        for value in category_values
    ]
    return {
        "ac1": float(coefficient),
        "observed_agreement": float(observed),
        "expected_agreement": float(expected),
        "n_items": len(eligible),
        "n_ratings": sum(pooled_counts.values()),
        "categories": category_rows,
    }


def interval_iou(left: tuple[float, float], right: tuple[float, float]) -> float:
    """Return temporal intersection-over-union for two closed intervals."""

    left_start, left_end = left
    right_start, right_end = right
    _require(left_start <= left_end, "left interval has start after end")
    _require(right_start <= right_end, "right interval has start after end")
    intersection = max(0.0, min(left_end, right_end) - max(left_start, right_start))
    left_length = left_end - left_start
    right_length = right_end - right_start
    union = left_length + right_length - intersection
    if union == 0.0:
        return 1.0 if left_start == right_start else 0.0
    return intersection / union


def _interval(value: Any, context: str) -> tuple[float, float] | None:
    if value is None:
        return None
    _require(isinstance(value, dict), f"{context}: interval must be an object or null")
    start = value.get("start_s")
    end = value.get("end_s")
    if start is None or end is None:
        for name, bound in (("start_s", start), ("end_s", end)):
            _require(bound is None or (_is_number(bound) and math.isfinite(float(bound))),
                     f"{context}.{name}: bound must be a finite number or null")
        return None
    _require(_is_number(start) and _is_number(end), f"{context}: interval bounds must be numbers")
    start_f, end_f = float(start), float(end)
    _require(math.isfinite(start_f) and math.isfinite(end_f), f"{context}: bounds must be finite")
    _require(0.0 <= start_f <= end_f, f"{context}: expected 0 <= start_s <= end_s")
    return start_f, end_f


def _summarise_intervals(
    item_intervals: Mapping[str, Sequence[tuple[float, float]]],
) -> dict[str, Any]:
    ious: list[float] = []
    starts: list[float] = []
    ends: list[float] = []
    midpoints: list[float] = []
    n_items = 0

    for item_id in sorted(item_intervals):
        intervals = list(item_intervals[item_id])
        if len(intervals) < 2:
            continue
        n_items += 1
        for left_index, left in enumerate(intervals):
            for right in intervals[left_index + 1 :]:
                ious.append(interval_iou(left, right))
                starts.append(abs(left[0] - right[0]))
                ends.append(abs(left[1] - right[1]))
                left_mid = (left[0] + left[1]) / 2.0
                right_mid = (right[0] + right[1]) / 2.0
                midpoints.append(abs(left_mid - right_mid))

    def mean(values: Sequence[float]) -> float | None:
        return float(statistics.fmean(values)) if values else None

    def median(values: Sequence[float]) -> float | None:
        return float(statistics.median(values)) if values else None

    return {
        "n_items": n_items,
        "n_pairwise_comparisons": len(ious),
        "mean_iou": mean(ious),
        "median_iou": median(ious),
        "overlap_fraction": mean([float(value > 0.0) for value in ious]),
        "mean_start_abs_seconds": mean(starts),
        "median_start_abs_seconds": median(starts),
        "mean_end_abs_seconds": mean(ends),
        "median_end_abs_seconds": median(ends),
        "mean_midpoint_abs_seconds": mean(midpoints),
        "median_midpoint_abs_seconds": median(midpoints),
    }


def select_audit_items(
    items_by_task: Mapping[str, Iterable[str]],
    *,
    fraction: float = AUDIT_FRACTION,
) -> dict[str, Any]:
    """Select an outcome-independent audit subset by hashing blinded IDs.

    Selection is performed separately within each instrument task. Ranking uses
    only a public namespace, task name, and blinded ID; rating values cannot
    affect selection. ``ceil(0.20 * n)`` is used in each non-empty task stratum.
    """

    _require(0.0 < fraction <= 1.0, "audit fraction must be in (0, 1]")
    by_task: dict[str, dict[str, Any]] = {}
    selected_all: set[str] = set()
    selected_tasks_by_item: dict[str, list[str]] = defaultdict(list)

    for task in sorted(items_by_task):
        item_ids = sorted(set(items_by_task[task]))
        n_selected = math.ceil(fraction * len(item_ids)) if item_ids else 0

        def rank(blind_id: str) -> tuple[str, str]:
            material = f"{AUDIT_NAMESPACE}\0{task}\0{blind_id}".encode("utf-8")
            return hashlib.sha256(material).hexdigest(), blind_id

        selected = sorted(sorted(item_ids, key=rank)[:n_selected])
        selected_all.update(selected)
        for blind_id in selected:
            selected_tasks_by_item[blind_id].append(task)
        by_task[task] = {
            "n_items": len(item_ids),
            "n_selected": len(selected),
            "blind_ids": selected,
        }

    all_items = sorted({item for items in items_by_task.values() for item in items})
    return {
        "flow": "MLLM_PRIMARY_HUMAN_AUDIT",
        "fraction_requested": fraction,
        "rounding": "ceil_per_task_stratum",
        "selection_namespace": AUDIT_NAMESPACE,
        "uses_rating_outcomes": False,
        "n_items": len(all_items),
        "n_selected": len(selected_all),
        "blind_ids": sorted(selected_all),
        "by_task": by_task,
        "flags": [
            {
                "blind_id": blind_id,
                "selected": blind_id in selected_all,
                "selected_tasks": selected_tasks_by_item.get(blind_id, []),
            }
            for blind_id in all_items
        ],
    }


def score_exports(exports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Score two or more already-loaded rating exports."""

    _require(len(exports) >= 2, "at least two rating exports are required")
    rater_ids = [str(payload["rater_id"]) for payload in exports]
    _require(len(rater_ids) == len(set(rater_ids)), "rater_id values must be unique")

    first = exports[0]
    manifest_id = first["manifest_id"]
    manifest_sha256 = first["manifest_sha256"]
    instrument_version = first["instrument_version"]
    item_set = set(first["item_order"])
    for payload in exports[1:]:
        _require(payload["manifest_id"] == manifest_id, "rating exports use different manifest IDs")
        _require(
            payload["manifest_sha256"] == manifest_sha256,
            "rating exports use different manifest hashes",
        )
        _require(
            payload["instrument_version"] == instrument_version,
            "rating exports use different instrument versions",
        )
        _require(set(payload["item_order"]) == item_set, "rating exports cover different item sets")

    categorical: dict[str, dict[str, list[Hashable]]] = {
        "anchor_status": defaultdict(list),
        "presence_verdict": defaultdict(list),
        "unrelated_background": defaultdict(list),
        "pair_curation_verdict": defaultdict(list),
    }
    intervals: dict[str, dict[str, list[tuple[float, float]]]] = {
        "anchor": defaultdict(list),
        "pair_event_1": defaultdict(list),
        "pair_event_2": defaultdict(list),
    }
    tasks_by_item: dict[str, tuple[str, ...]] = {}

    for payload in exports:
        for rating in payload["ratings"]:
            blind_id = rating["blind_id"]
            tasks = tuple(sorted(rating["tasks"]))
            previous_tasks = tasks_by_item.setdefault(blind_id, tasks)
            _require(previous_tasks == tasks, f"task mismatch for blinded item {blind_id!r}")

            if "anchor_presence" in tasks:
                anchor = rating.get("anchor")
                presence = rating.get("presence")
                _require(isinstance(anchor, dict), f"{blind_id}: anchor response is missing")
                _require(isinstance(presence, dict), f"{blind_id}: presence response is missing")

                status = anchor.get("status")
                _require(
                    status in {"marked", "too_uncertain", "unrated"},
                    f"{blind_id}: invalid anchor status",
                )
                if status != "unrated":
                    categorical["anchor_status"][blind_id].append(status)
                if status == "marked":
                    marked = _interval(anchor, f"{blind_id}.anchor")
                    _require(marked is not None, f"{blind_id}: marked anchor has no interval")
                    intervals["anchor"][blind_id].append(marked)

                verdict = presence.get("verdict")
                if verdict is not None:
                    _require(
                        verdict in {"target_present", "absent", "uncertain"},
                        f"{blind_id}: invalid presence verdict",
                    )
                    categorical["presence_verdict"][blind_id].append(verdict)
                background = presence.get("unrelated_background")
                if background is not None:
                    _require(isinstance(background, bool), f"{blind_id}: background flag must be boolean")
                    categorical["unrelated_background"][blind_id].append(background)

            if "two_event" in tasks:
                curation = rating.get("pair_curation")
                _require(isinstance(curation, dict), f"{blind_id}: pair response is missing")
                verdict = curation.get("verdict")
                if verdict is not None:
                    _require(verdict in {"confirm", "reject"}, f"{blind_id}: invalid pair verdict")
                    categorical["pair_curation_verdict"][blind_id].append(verdict)
                if verdict == "confirm":
                    event_1 = _interval(curation.get("event_1"), f"{blind_id}.pair_curation.event_1")
                    event_2 = _interval(curation.get("event_2"), f"{blind_id}.pair_curation.event_2")
                    if event_1 is not None:
                        intervals["pair_event_1"][blind_id].append(event_1)
                    if event_2 is not None:
                        intervals["pair_event_2"][blind_id].append(event_2)

    missing_tasks = sorted(item_set - set(tasks_by_item))
    _require(not missing_tasks, f"no task metadata for blinded items: {missing_tasks}")
    items_by_task: dict[str, list[str]] = defaultdict(list)
    for blind_id in sorted(item_set):
        for task in tasks_by_item[blind_id]:
            items_by_task[task].append(blind_id)

    return {
        "schema_version": SCORE_SCHEMA_VERSION,
        "source_rating_schema_version": RATING_SCHEMA_VERSION,
        "instrument_version": instrument_version,
        "manifest_id": manifest_id,
        "manifest_sha256": manifest_sha256,
        "n_raters": len(exports),
        "rater_ids": sorted(rater_ids),
        "agreement": {
            question: gwet_ac1(values, categories=QUESTION_CATEGORIES[question])
            for question, values in sorted(categorical.items())
        },
        "interval_overlap": {
            name: _summarise_intervals(values)
            for name, values in sorted(intervals.items())
        },
        "audit_20_percent": select_audit_items(items_by_task),
    }


def score_paths(paths: Sequence[str | Path]) -> dict[str, Any]:
    return score_exports([load_rating_export(path) for path in paths])


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score 2+ blinded SoundDecisions v2 human-rating exports."
    )
    parser.add_argument("ratings", nargs="+", type=Path, help="ratings_<raterID>.json files")
    parser.add_argument("-o", "--output", type=Path, help="write JSON here instead of stdout")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if len(args.ratings) < 2:
        parser.error("at least two rating exports are required")
    try:
        result = score_paths(args.ratings)
    except ValueError as exc:
        parser.error(str(exc))

    rendered = json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
