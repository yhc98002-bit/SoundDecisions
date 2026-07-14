"""Focused synthetic coverage for the Arc-4 WP-A2 axis-validity audit."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from scripts import arc4_axis_validity as A


def _record(axis: str, clip: str, j: int) -> dict:
    categorical = axis != "material"
    label = j % 2 if axis == "timing" else "present"
    target = {
        "axis_id": axis,
        "kind": "categorical" if categorical else "embedding",
        "label": (label if categorical else None),
        "embedding": (None if categorical else [1.0, float(j + 1)]),
    }
    return {
        "gen_id": f"{clip}__p1cfg1_ind{j}",
        "axis_id": axis,
        "target": target,
        "extra": {"role": A.ROLE, "clip": clip, "j": j, "cfg": 1.0},
    }


def _write_journal(path, clips=("a", "b"), n_per_clip=2):
    records = [
        _record(axis, clip, j)
        for axis in A.AXES
        for clip in clips
        for j in range(n_per_clip)
    ]
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    return records


def test_stream_join_requires_every_axis_clip_generation_key(tmp_path):
    journal = tmp_path / "measurements.jsonl"
    records = _write_journal(journal)
    cohort = A.stream_independent_cohort(
        journal, ["a", "b"], expected_per_clip=2
    )
    assert set(cohort) == set(A.AXES)
    assert all(
        len(cohort[axis][clip]) == 2 for axis in A.AXES for clip in ("a", "b")
    )

    journal.write_text("".join(json.dumps(record) + "\n" for record in records[:-1]))
    with pytest.raises(ValueError, match="expected exactly"):
        A.stream_independent_cohort(journal, ["a", "b"], expected_per_clip=2)


def test_stream_join_rejects_duplicate_keys(tmp_path):
    journal = tmp_path / "measurements.jsonl"
    records = _write_journal(journal)
    records.append(records[0])
    journal.write_text("".join(json.dumps(record) + "\n" for record in records))
    with pytest.raises(ValueError, match="duplicate join key"):
        A.stream_independent_cohort(journal, ["a", "b"], expected_per_clip=2)


def test_seeded_video_pair_sample_is_unique_and_deterministic():
    first = A.select_video_pairs(200, 10_000, seed=0)
    second = A.select_video_pairs(200, 10_000, seed=0)
    assert np.array_equal(first, second)
    assert first.shape == (10_000, 2)
    assert np.all(first[:, 0] < first[:, 1])
    assert len({tuple(pair) for pair in first.tolist()}) == 10_000


def test_categorical_marginals_drop_abstain_and_use_inverse_simpson():
    per_clip = {
        f"clip-{i:02d}": ("a", "a", "b", "abstain") for i in range(20)
    }
    result = A.analyze_categorical_axis(
        per_clip,
        A.declared_thresholds(),
        n_pairs=100,
        pair_seed=0,
        n_boot=100,
        bootstrap_seed=0,
    )
    probabilities = np.asarray([2 / 3, 1 / 3])
    expected_entropy = -float(np.sum(probabilities * np.log(probabilities)))
    assert result["n_observations"] == 80
    assert result["n_confident"] == 60
    assert result["n_abstain"] == 20
    assert result["abstain_rate"] == pytest.approx(0.25)
    assert result["majority_share"] == pytest.approx(2 / 3)
    assert result["entropy"] == pytest.approx(expected_entropy)
    assert result["k_eff"] == pytest.approx(1 / ((2 / 3) ** 2 + (1 / 3) ** 2))
    assert result["label_marginals"] == [
        {"label": "a", "share": pytest.approx(2 / 3)},
        {"label": "b", "share": pytest.approx(1 / 3)},
    ]
    assert result["verdict"] == "INFORMATIVE"


def test_verdict_boundaries_are_frozen_prompt_rules():
    thresholds = A.declared_thresholds()
    assert A.categorical_verdict(0.85, 2.0, thresholds)[0] == "DEGENERATE"
    assert A.categorical_verdict(0.84, 1.49, thresholds)[0] == "DEGENERATE"
    assert A.categorical_verdict(0.84, 1.50, thresholds)[0] == "INFORMATIVE"
    assert A.material_verdict(0.099, thresholds)[0] == "DEGENERATE"
    assert A.material_verdict(0.10, thresholds)[0] == "INFORMATIVE"
    assert thresholds["categorical"]["k_eff_definition"].startswith("inverse_simpson")


def test_material_embeddings_are_row_normalized_and_categorical_fields_null():
    base = {}
    scaled = {}
    for i in range(20):
        angle = (i + 1) / 10
        rows = np.asarray(
            [
                [math.cos(angle), math.sin(angle)],
                [math.cos(angle + 0.1), math.sin(angle + 0.1)],
                [math.cos(angle - 0.1), math.sin(angle - 0.1)],
            ]
        )
        base[f"clip-{i:02d}"] = rows
        scaled[f"clip-{i:02d}"] = rows * np.asarray([[2.0], [5.0], [11.0]])

    kwargs = dict(
        thresholds=A.declared_thresholds(),
        n_pairs=100,
        pair_seed=0,
        n_boot=100,
        bootstrap_seed=0,
    )
    result = A.analyze_material_axis(base, **kwargs)
    scaled_result = A.analyze_material_axis(scaled, **kwargs)
    for key in ("a_ind_mean", "a_between_video", "relative_agreement"):
        assert result[key] == pytest.approx(scaled_result[key], abs=1e-12)
    for key in (
        "majority_share",
        "k_eff",
        "entropy",
        "abstain_rate",
        "label_marginals",
    ):
        assert result[key] is None


def test_video_bootstrap_difference_is_deterministic_and_significant():
    within = np.full(20, 0.9)
    pairs = A.select_video_pairs(20, 100, seed=0)
    between = np.full(len(pairs), 0.1)
    first = A._bootstrap_heterogeneity(within, pairs, between, n_boot=200, seed=0)
    second = A._bootstrap_heterogeneity(within, pairs, between, n_boot=200, seed=0)
    assert first == pytest.approx(second)
    assert first == pytest.approx((0.8, 0.8, 0.8))


def test_json_writer_is_byte_deterministic_and_rejects_nan(tmp_path):
    payload = {"thresholds": A.declared_thresholds(), "per_axis": {"z": 1, "a": 2}}
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    A.write_json(first, payload)
    A.write_json(second, payload)
    assert first.read_bytes() == second.read_bytes()
    with pytest.raises(ValueError):
        A.write_json(tmp_path / "bad.json", {"bad": float("nan")})


def test_build_rejects_unequal_axis_cardinality():
    clips = [f"clip-{i:02d}" for i in range(20)]
    cohort = {
        axis: {clip: ("a", "b") for clip in clips} for axis in A.CATEGORICAL_AXES
    }
    cohort["material"] = {
        clip: (np.asarray([1.0, 0.0]), np.asarray([0.9, 0.1])) for clip in clips
    }
    cohort["timing"][clips[0]] = (0, 1, 1)
    with pytest.raises(ValueError, match="per-clip cardinality is inconsistent"):
        A.build_analysis(
            cohort,
            A.declared_thresholds(),
            n_pairs=100,
            n_boot=10,
        )
