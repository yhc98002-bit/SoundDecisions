#!/usr/bin/env python
"""Arc-4 WP-A2 axis-informativeness audit from cached Phase-1 finals.

This is a CPU-only re-analysis of ``p1cfg1_independent`` measurement rows.  It
compares within-video self-agreement with agreement between distinct videos and
reports pooled categorical marginals.  Categorical abstains are missing values:
they are excluded from agreement and marginal calculations and reported
separately.

The verdict thresholds below are fixed by the WP-A2 prompt.  They are module
constants, copied into the output before the journal is loaded, and never
estimated from the data.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JOURNAL = ROOT / "results/stage0/measurements/measurements.jsonl"
DEFAULT_MANIFEST = ROOT / "data/manifests/phase1_manifest_frozen.json"
DEFAULT_OUTPUT = ROOT / "results/arc4_wpA2/axis_validity.json"

AXES = ("presence", "timing", "class", "material")
CATEGORICAL_AXES = ("presence", "timing", "class")
MATERIAL_AXIS = "material"
ROLE = "p1cfg1_independent"
ABSTAIN = "abstain"
EXPECTED_CLIPS = 200
EXPECTED_INDEPENDENT_PER_CLIP = 16
PAIR_SAMPLE_SIZE = 10_000
PAIR_SAMPLE_SEED = 0
BOOTSTRAP_DRAWS = 1_000
BOOTSTRAP_SEED = 0

# Frozen A2-1 verdict rules.  Keep these independent of the observed cohort.
CATEGORICAL_MAJORITY_DEGENERATE_GTE = 0.85
CATEGORICAL_K_EFF_DEGENERATE_LT = 1.5
MATERIAL_RELATIVE_AGREEMENT_DEGENERATE_LT = 0.10

DECLARED_THRESHOLDS = {
    "categorical": {
        "degenerate_if_majority_share_gte": CATEGORICAL_MAJORITY_DEGENERATE_GTE,
        "degenerate_if_k_eff_lt": CATEGORICAL_K_EFF_DEGENERATE_LT,
        "combination": "OR",
        "k_eff_definition": "inverse_simpson_1_over_sum_p_squared",
    },
    "material": {
        "degenerate_if_relative_agreement_lt": (
            MATERIAL_RELATIVE_AGREEMENT_DEGENERATE_LT
        ),
        "relative_agreement_definition": (
            "(a_ind_mean - a_between_video) / (1 - a_between_video)"
        ),
    },
    "frozen_from": "WP-A2 A2-1 prompt",
}


def declared_thresholds() -> dict[str, Any]:
    """Return a detached, JSON-serializable copy of the frozen rules."""
    return copy.deepcopy(DECLARED_THRESHOLDS)


def load_manifest_clips(path: Path, expected_n: int = EXPECTED_CLIPS) -> list[str]:
    """Load and validate the exact frozen Phase-1 single-event cohort."""
    try:
        payload = json.loads(path.read_text())
        raw_clips = payload["clips"]["single_event"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(f"invalid Phase-1 manifest {path}: {exc}") from exc

    if not isinstance(raw_clips, list):
        raise ValueError("manifest clips.single_event must be a list")
    clips = [str(clip) for clip in raw_clips]
    if len(clips) != expected_n:
        raise ValueError(
            f"manifest has {len(clips)} single-event clips; expected {expected_n}"
        )
    if len(set(clips)) != len(clips):
        raise ValueError("manifest clips.single_event contains duplicate clip ids")
    return sorted(clips)


def _validate_record_target(record: Mapping[str, Any], axis: str, line_no: int) -> Any:
    target = record.get("target")
    if not isinstance(target, dict):
        raise ValueError(f"line {line_no}: target must be an object")
    if target.get("axis_id") != axis:
        raise ValueError(
            f"line {line_no}: target.axis_id={target.get('axis_id')!r} "
            f"does not match axis_id={axis!r}"
        )

    if axis in CATEGORICAL_AXES:
        if target.get("kind") != "categorical":
            raise ValueError(f"line {line_no}: {axis} target is not categorical")
        label = target.get("label")
        if not isinstance(label, (str, int, float, bool)) or label == "":
            raise ValueError(
                f"line {line_no}: {axis} label must be a non-empty JSON scalar"
            )
        if isinstance(label, float) and not math.isfinite(label):
            raise ValueError(f"line {line_no}: {axis} label must be finite")
        if target.get("embedding") is not None:
            raise ValueError(f"line {line_no}: categorical target carries an embedding")
        return label

    if target.get("kind") != "embedding":
        raise ValueError(f"line {line_no}: material target is not an embedding")
    if target.get("label") is not None:
        raise ValueError(f"line {line_no}: material target carries a label")
    embedding = np.asarray(target.get("embedding"), dtype=float)
    if embedding.ndim != 1 or embedding.size == 0:
        raise ValueError(f"line {line_no}: material embedding must be a non-empty vector")
    if not np.all(np.isfinite(embedding)):
        raise ValueError(f"line {line_no}: material embedding contains non-finite values")
    if float(np.linalg.norm(embedding)) <= 0.0:
        raise ValueError(f"line {line_no}: material embedding has zero norm")
    return embedding


def stream_independent_cohort(
    journal_path: Path,
    clips: Sequence[str],
    expected_per_clip: int = EXPECTED_INDEPENDENT_PER_CLIP,
) -> dict[str, dict[str, tuple[Any, ...]]]:
    """Stream and strictly join all cached cfg=1 independent-final targets.

    The join key is ``(axis_id, extra.clip, extra.j)``.  Every expected key must
    occur exactly once, and no role-matching row outside the manifest is allowed.
    """
    if expected_per_clip <= 0:
        raise ValueError("expected_per_clip must be positive")
    clip_set = set(clips)
    if len(clip_set) != len(clips):
        raise ValueError("clips contains duplicates")

    slots: dict[str, dict[str, list[Any | None]]] = {
        axis: {clip: [None] * expected_per_clip for clip in clips} for axis in AXES
    }
    selected = 0
    embedding_dim: int | None = None

    try:
        journal = journal_path.open()
    except OSError as exc:
        raise ValueError(f"cannot open measurement journal {journal_path}: {exc}") from exc

    with journal:
        for line_no, line in enumerate(journal, start=1):
            # The journal is large.  This exact role marker is only a prefilter;
            # every selected row is parsed and its structured role is rechecked.
            if ROLE not in line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {line_no}: invalid JSON: {exc}") from exc
            extra = record.get("extra")
            if not isinstance(extra, dict) or extra.get("role") != ROLE:
                continue

            axis = record.get("axis_id")
            if axis not in AXES:
                raise ValueError(f"line {line_no}: unexpected axis_id {axis!r}")
            clip = str(extra.get("clip"))
            if clip not in clip_set:
                raise ValueError(f"line {line_no}: clip {clip!r} is outside the manifest")
            j = extra.get("j")
            if isinstance(j, bool) or not isinstance(j, int):
                raise ValueError(f"line {line_no}: extra.j must be an integer")
            if not 0 <= j < expected_per_clip:
                raise ValueError(
                    f"line {line_no}: extra.j={j} outside [0, {expected_per_clip})"
                )
            try:
                cfg = float(extra.get("cfg"))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"line {line_no}: invalid cfg={extra.get('cfg')!r}") from exc
            if cfg != 1.0:
                raise ValueError(f"line {line_no}: {ROLE} row has cfg={cfg}, expected 1.0")

            expected_gen_id = f"{clip}__p1cfg1_ind{j}"
            if record.get("gen_id") != expected_gen_id:
                raise ValueError(
                    f"line {line_no}: gen_id={record.get('gen_id')!r}; "
                    f"expected {expected_gen_id!r}"
                )
            if slots[axis][clip][j] is not None:
                raise ValueError(
                    f"line {line_no}: duplicate join key ({axis!r}, {clip!r}, {j})"
                )

            value = _validate_record_target(record, axis, line_no)
            if axis == MATERIAL_AXIS:
                if embedding_dim is None:
                    embedding_dim = int(value.size)
                elif value.size != embedding_dim:
                    raise ValueError(
                        f"line {line_no}: material embedding dimension {value.size}; "
                        f"expected {embedding_dim}"
                    )
            slots[axis][clip][j] = value
            selected += 1

    expected_total = len(AXES) * len(clips) * expected_per_clip
    if selected != expected_total:
        raise ValueError(
            f"joined {selected} {ROLE} rows; expected exactly {expected_total} "
            f"({len(AXES)} axes x {len(clips)} clips x {expected_per_clip})"
        )

    missing = [
        (axis, clip, j)
        for axis in AXES
        for clip in clips
        for j, value in enumerate(slots[axis][clip])
        if value is None
    ]
    if missing:
        raise ValueError(f"missing {len(missing)} join keys; first={missing[0]!r}")

    return {
        axis: {clip: tuple(slots[axis][clip]) for clip in sorted(clips)}
        for axis in AXES
    }


def select_video_pairs(n_videos: int, n_pairs: int, seed: int) -> np.ndarray:
    """Select unique unordered pairs by a seeded permutation of all video pairs."""
    if n_videos < 2:
        raise ValueError("at least two videos are required")
    left, right = np.triu_indices(n_videos, k=1)
    total = int(left.size)
    if n_pairs <= 0 or n_pairs > total:
        raise ValueError(f"n_pairs must be in [1, {total}], got {n_pairs}")
    order = np.random.default_rng(seed).permutation(total)[:n_pairs]
    return np.column_stack((left[order], right[order])).astype(np.int64, copy=False)


def _label_sort_key(label: Any) -> tuple[str, str]:
    """Stable ordering that also distinguishes values such as ``0`` and ``"0"``."""
    return type(label).__name__, json.dumps(label, ensure_ascii=True, sort_keys=True)


def _categorical_within(labels: Sequence[Any]) -> float:
    n = len(labels)
    if n < 2:
        raise ValueError("within-video categorical agreement needs >=2 confident labels")
    counts = np.asarray(list(Counter(labels).values()), dtype=float)
    return float(np.sum(counts * (counts - 1.0)) / (n * (n - 1.0)))


def _bootstrap_heterogeneity(
    within: np.ndarray,
    pairs: np.ndarray,
    between: np.ndarray,
    n_boot: int,
    seed: int,
) -> tuple[float, float, float]:
    """Video bootstrap of mean(within) - mean(frozen cross-video pair sample).

    A resample is represented by video multiplicities.  Within-video values are
    weighted by those multiplicities; each frozen distinct-video pair is weighted
    by the product of its endpoint multiplicities.
    """
    within = np.asarray(within, dtype=float)
    between = np.asarray(between, dtype=float)
    if within.ndim != 1 or not np.all(np.isfinite(within)):
        raise ValueError("within values must be a finite vector")
    if between.shape != (len(pairs),) or not np.all(np.isfinite(between)):
        raise ValueError("between values must be finite and aligned with pairs")
    if n_boot <= 0:
        raise ValueError("n_boot must be positive")

    point = float(np.mean(within) - np.mean(between))
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=float)
    n_videos = within.size
    for b in range(n_boot):
        sampled = rng.integers(0, n_videos, size=n_videos)
        multiplicity = np.bincount(sampled, minlength=n_videos).astype(float)
        within_mean = float(np.dot(multiplicity, within) / np.sum(multiplicity))
        pair_weights = multiplicity[pairs[:, 0]] * multiplicity[pairs[:, 1]]
        pair_weight_sum = float(np.sum(pair_weights))
        if pair_weight_sum <= 0.0:
            raise RuntimeError("bootstrap draw has no represented frozen video pair")
        between_mean = float(np.dot(pair_weights, between) / pair_weight_sum)
        draws[b] = within_mean - between_mean

    return (
        point,
        float(np.percentile(draws, 2.5)),
        float(np.percentile(draws, 97.5)),
    )


def categorical_verdict(
    majority_share: float, k_eff: float, thresholds: Mapping[str, Any]
) -> tuple[str, str]:
    rule = thresholds["categorical"]
    majority_bad = majority_share >= float(rule["degenerate_if_majority_share_gte"])
    k_eff_bad = k_eff < float(rule["degenerate_if_k_eff_lt"])
    verdict = "DEGENERATE" if majority_bad or k_eff_bad else "INFORMATIVE"
    basis = (
        f"majority_share >= {rule['degenerate_if_majority_share_gte']} OR "
        f"k_eff < {rule['degenerate_if_k_eff_lt']}"
    )
    return verdict, basis


def material_verdict(
    relative_agreement: float, thresholds: Mapping[str, Any]
) -> tuple[str, str]:
    cutoff = float(thresholds["material"]["degenerate_if_relative_agreement_lt"])
    verdict = "DEGENERATE" if relative_agreement < cutoff else "INFORMATIVE"
    return verdict, f"relative_agreement < {cutoff}"


def _common_axis_result(
    within: np.ndarray,
    pairs: np.ndarray,
    between: np.ndarray,
    n_boot: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    difference, ci_lo, ci_hi = _bootstrap_heterogeneity(
        within, pairs, between, n_boot=n_boot, seed=bootstrap_seed
    )
    a_ind = float(np.mean(within))
    a_between = float(np.mean(between))
    denom = 1.0 - a_between
    relative = float((a_ind - a_between) / denom) if denom > 1e-12 else None
    return {
        "a_ind_mean": a_ind,
        "a_between_video": a_between,
        "relative_agreement": relative,
        "heterogeneity_difference": difference,
        "heterogeneity_difference_ci95": [ci_lo, ci_hi],
        "heterogeneity_significant": bool(ci_lo > 0.0),
        "heterogeneity_significance_definition": (
            "positive within-minus-between difference; 95% video-bootstrap CI lower bound > 0"
        ),
        "n_between_video_pairs": int(len(pairs)),
    }


def analyze_categorical_axis(
    per_clip: Mapping[str, Sequence[Any]],
    thresholds: Mapping[str, Any],
    n_pairs: int,
    pair_seed: int,
    n_boot: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    clips = sorted(per_clip)
    confident_by_clip = {
        clip: [label for label in per_clip[clip] if label != ABSTAIN] for clip in clips
    }
    too_small = [clip for clip, labels in confident_by_clip.items() if len(labels) < 2]
    if too_small:
        raise ValueError(
            f"categorical axis has <2 confident labels for {len(too_small)} clips; "
            f"first={too_small[0]!r}"
        )

    pooled = [label for clip in clips for label in confident_by_clip[clip]]
    label_types = {type(label) for label in pooled}
    if len(label_types) != 1:
        raise ValueError(
            "categorical axis mixes label scalar types: "
            + ", ".join(sorted(label_type.__name__ for label_type in label_types))
        )
    raw_n = sum(len(per_clip[clip]) for clip in clips)
    abstain_n = raw_n - len(pooled)
    counts = Counter(pooled)
    label_order = sorted(counts, key=_label_sort_key)
    count_vec = np.asarray([counts[label] for label in label_order], dtype=float)
    probabilities = count_vec / np.sum(count_vec)
    majority_share = float(np.max(probabilities))
    entropy = float(-np.sum(probabilities * np.log(probabilities)))
    k_eff = float(1.0 / np.sum(probabilities**2))

    profiles = np.zeros((len(clips), len(label_order)), dtype=float)
    within = np.empty(len(clips), dtype=float)
    for index, clip in enumerate(clips):
        labels = confident_by_clip[clip]
        local = Counter(labels)
        profiles[index] = np.asarray([local[label] for label in label_order]) / len(labels)
        within[index] = _categorical_within(labels)

    pairs = select_video_pairs(len(clips), n_pairs, pair_seed)
    between = np.einsum(
        "ij,ij->i", profiles[pairs[:, 0]], profiles[pairs[:, 1]], optimize=True
    )
    result = _common_axis_result(
        within, pairs, between, n_boot=n_boot, bootstrap_seed=bootstrap_seed
    )
    verdict, basis = categorical_verdict(majority_share, k_eff, thresholds)
    result.update(
        {
            "kind": "categorical",
            "n_videos": len(clips),
            "n_observations": raw_n,
            "n_confident": len(pooled),
            "n_abstain": abstain_n,
            "abstain_rate": float(abstain_n / raw_n),
            "majority_share": majority_share,
            "entropy": entropy,
            "entropy_log_base": "natural",
            "k_eff": k_eff,
            "k_eff_definition": "inverse_simpson_1_over_sum_p_squared",
            "label_marginals": [
                {"label": label, "share": float(counts[label] / len(pooled))}
                for label in label_order
            ],
            "verdict": verdict,
            "verdict_basis": basis,
        }
    )
    return result


def analyze_material_axis(
    per_clip: Mapping[str, Sequence[np.ndarray]],
    thresholds: Mapping[str, Any],
    n_pairs: int,
    pair_seed: int,
    n_boot: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    clips = sorted(per_clip)
    normalized_by_clip: list[np.ndarray] = []
    expected_n: int | None = None
    expected_dim: int | None = None
    for clip in clips:
        embeddings = np.asarray(per_clip[clip], dtype=float)
        if embeddings.ndim != 2 or not np.all(np.isfinite(embeddings)):
            raise ValueError(f"material clip {clip!r} is not a finite embedding matrix")
        if expected_n is None:
            expected_n, expected_dim = embeddings.shape
        elif embeddings.shape != (expected_n, expected_dim):
            raise ValueError(
                f"material clip {clip!r} shape={embeddings.shape}; "
                f"expected {(expected_n, expected_dim)}"
            )
        norms = np.linalg.norm(embeddings, axis=1)
        if np.any(norms <= 0.0):
            raise ValueError(f"material clip {clip!r} contains a zero-norm embedding")
        normalized_by_clip.append(embeddings / norms[:, None])

    within = np.empty(len(clips), dtype=float)
    profiles = np.empty((len(clips), int(expected_dim)), dtype=float)
    for index, normalized in enumerate(normalized_by_clip):
        n = normalized.shape[0]
        if n < 2:
            raise ValueError("within-video material agreement needs >=2 embeddings")
        summed = np.sum(normalized, axis=0)
        within[index] = float((np.dot(summed, summed) - n) / (n * (n - 1)))
        profiles[index] = np.mean(normalized, axis=0)

    pairs = select_video_pairs(len(clips), n_pairs, pair_seed)
    between = np.einsum(
        "ij,ij->i", profiles[pairs[:, 0]], profiles[pairs[:, 1]], optimize=True
    )
    result = _common_axis_result(
        within, pairs, between, n_boot=n_boot, bootstrap_seed=bootstrap_seed
    )
    relative = result["relative_agreement"]
    if relative is None or not math.isfinite(relative):
        raise ValueError("material relative agreement is undefined")
    verdict, basis = material_verdict(relative, thresholds)
    result.update(
        {
            "kind": "embedding",
            "n_videos": len(clips),
            "n_observations": int(sum(len(per_clip[clip]) for clip in clips)),
            "embedding_dimension": int(expected_dim),
            # These quantities only have categorical definitions.
            "n_confident": None,
            "n_abstain": None,
            "abstain_rate": None,
            "majority_share": None,
            "entropy": None,
            "entropy_log_base": None,
            "k_eff": None,
            "k_eff_definition": None,
            "label_marginals": None,
            "verdict": verdict,
            "verdict_basis": basis,
        }
    )
    return result


def build_analysis(
    cohort: Mapping[str, Mapping[str, Sequence[Any]]],
    thresholds: Mapping[str, Any],
    n_pairs: int = PAIR_SAMPLE_SIZE,
    pair_seed: int = PAIR_SAMPLE_SEED,
    n_boot: int = BOOTSTRAP_DRAWS,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    if set(cohort) != set(AXES):
        raise ValueError(f"cohort axes={sorted(cohort)}; expected {list(AXES)}")
    clip_sets = {axis: set(cohort[axis]) for axis in AXES}
    reference = clip_sets[AXES[0]]
    if any(clips != reference for clips in clip_sets.values()):
        raise ValueError("axis cohorts do not contain identical clip ids")
    if not reference:
        raise ValueError("axis cohorts are empty")
    cardinalities = {
        axis: {len(values) for values in cohort[axis].values()} for axis in AXES
    }
    if any(len(sizes) != 1 for sizes in cardinalities.values()):
        raise ValueError(f"per-clip cardinality is inconsistent: {cardinalities}")
    common_sizes = {next(iter(cardinalities[axis])) for axis in AXES}
    if len(common_sizes) != 1:
        raise ValueError(f"per-axis cardinality is inconsistent: {cardinalities}")

    per_axis = {
        axis: analyze_categorical_axis(
            cohort[axis], thresholds, n_pairs, pair_seed, n_boot, bootstrap_seed
        )
        for axis in CATEGORICAL_AXES
    }
    per_axis[MATERIAL_AXIS] = analyze_material_axis(
        cohort[MATERIAL_AXIS], thresholds, n_pairs, pair_seed, n_boot, bootstrap_seed
    )
    n_per_clip = next(iter(common_sizes))
    return {
        "analysis": "Arc-4 WP-A2 axis informativeness",
        "evidence_tier": "diagnostic/exploratory",
        "decision_token": None,
        "thresholds": copy.deepcopy(dict(thresholds)),
        "source": {
            "role": ROLE,
            "n_videos": len(reference),
            "n_axes": len(AXES),
            "n_independent_per_video_axis": n_per_clip,
            "join_key": ["axis_id", "extra.clip", "extra.j"],
            "cardinality_validated": True,
        },
        "between_video_sampling": {
            "unit": "unique unordered pair of distinct videos",
            "selection": "seeded permutation without replacement",
            "n_pairs": n_pairs,
            "seed": pair_seed,
            "pair_score": (
                "mean agreement over all cross-target pairs for the selected video pair"
            ),
        },
        "heterogeneity_bootstrap": {
            "unit": "video",
            "n_draws": n_boot,
            "seed": bootstrap_seed,
            "ci": 0.95,
            "pair_reweighting": (
                "frozen selected video pairs weighted by endpoint resample multiplicities"
            ),
        },
        "per_axis": {axis: per_axis[axis] for axis in AXES},
    }


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.write_text(text)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--journal", type=Path, default=DEFAULT_JOURNAL)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-pairs", type=int, default=PAIR_SAMPLE_SIZE)
    parser.add_argument("--n-boot", type=int, default=BOOTSTRAP_DRAWS)
    args = parser.parse_args(argv)

    # Freeze and retain the prompt rules before any data are loaded or scored.
    thresholds = declared_thresholds()
    clips = load_manifest_clips(args.manifest)
    cohort = stream_independent_cohort(args.journal, clips)
    result = build_analysis(
        cohort,
        thresholds,
        n_pairs=args.n_pairs,
        pair_seed=PAIR_SAMPLE_SEED,
        n_boot=args.n_boot,
        bootstrap_seed=BOOTSTRAP_SEED,
    )
    write_json(args.output, result)
    print(f"wrote {args.output}")
    for axis in AXES:
        row = result["per_axis"][axis]
        print(
            f"{axis}: A_ind={row['a_ind_mean']:.6f}, "
            f"A_between={row['a_between_video']:.6f}, verdict={row['verdict']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
