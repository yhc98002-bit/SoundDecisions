"""CPU-only integrity and leakage tests for the exploratory Class readout."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from foley_cw import class_internal_readout as readout
from foley_cw.b2_class_closure import atomic_json_create, atomic_jsonl_create


REPO = Path(__file__).resolve().parents[1]
PROTOCOL = REPO / "experiment" / "non_human_closure" / "PROTOCOL.json"
IMPLEMENTATION = (
    REPO / "experiment" / "non_human_closure" / "CLASS_READOUT_IMPLEMENTATION.json"
)


def test_implementation_matches_parent_protocol():
    _, implementation, digest = readout.validate_protocols(PROTOCOL, IMPLEMENTATION)
    assert implementation["status"] == "FROZEN_BEFORE_B2_FEATURE_RECOLLECTION"
    assert len(digest) == 64


def test_group_assignment_is_deterministic_and_leakage_fails():
    videos = [f"v{index:02d}" for index in range(48)]
    first = readout.grouped_folds(videos, 6)
    second = readout.grouped_folds(reversed(videos), 6)
    assert first == second
    assert sorted(first.values()).count(0) == 8
    groups = ["a", "a", "b", "b"]
    readout.validate_no_group_leakage(groups, [0, 1], [2, 3])
    with pytest.raises(readout.InternalReadoutError, match="video leakage"):
        readout.validate_no_group_leakage(groups, [0, 2], [1, 3])


def test_fork_majority_never_invents_ties_or_support():
    names = ["impact", "water", "vehicle"]
    assert readout.fork_majority_target(["impact"], names)[:2] == (
        None,
        "confident_forks_lt_2",
    )
    assert readout.fork_majority_target(["impact", "water"], names)[:2] == (
        None,
        "fork_majority_tie",
    )
    label, reason, counts = readout.fork_majority_target(
        ["impact", "impact", "water"], names
    )
    assert label == "impact" and reason is None
    assert counts == {"impact": 2, "water": 1, "vehicle": 0}
    with pytest.raises(readout.InternalReadoutError):
        readout.fork_majority_target(["invented", "invented"], names)


def test_fixed_projections_are_reproducible_and_outcome_independent():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(7, 300)).astype(np.float32)
    first, first_meta = readout.project_vectors(X, "latent")
    second, second_meta = readout.project_vectors(X, "latent")
    other, other_meta = readout.project_vectors(X, "velocity")
    assert first.shape == (7, 256)
    assert np.array_equal(first, second)
    assert first_meta == second_meta
    assert first_meta["matrix_sha256"] != other_meta["matrix_sha256"]
    assert not np.array_equal(first, other)
    tokens = rng.normal(size=(7, 5, 80)).astype(np.float32)
    projected, metadata = readout.project_attention(tokens)
    assert projected.shape == (7, 5, 32)
    assert metadata["output_width"] == 32


def test_ridge_predictions_are_probabilities_and_grouped_outer_complete():
    videos = [f"v{index:02d}" for index in range(12)]
    groups, labels, ids, values = [], [], [], []
    for index, video in enumerate(videos):
        for seed in range(2):
            label = "a" if index % 2 == 0 else "b"
            groups.append(video)
            labels.append(label)
            ids.append(f"{video}-{seed}")
            values.append([1.0 if label == "a" else -1.0, float(seed)])
    X = np.asarray(values, dtype=np.float32)
    predictions, selections = readout.nested_outer_predictions(
        "conditioning_only",
        X,
        labels,
        groups,
        ids,
        videos,
        ["a", "b", "unused"],
        target="fork_majority",
        progress=0.35,
        device="cpu",
    )
    assert len(predictions) == len(ids)
    assert len({row["candidate_id"] for row in predictions}) == len(ids)
    assert len(selections) == 6
    for row in predictions:
        assert sum(row["probabilities"]) == pytest.approx(1.0)
        assert row["predicted_label"] in {"a", "b"}
    for selection in selections:
        assert not (
            set(selection["train_video_ids"]) & set(selection["test_video_ids"])
        )


def test_majority_nested_cv_has_complete_test_counts():
    videos = [f"v{index:02d}" for index in range(12)]
    groups = [video for video in videos for _ in range(2)]
    labels = ["a" if index % 3 else "b" for index in range(len(groups))]
    ids = [f"id-{index}" for index in range(len(groups))]
    predictions, _ = readout.nested_outer_predictions(
        "majority_null",
        None,
        labels,
        groups,
        ids,
        videos,
        ["a", "b"],
        target="ode_final",
        progress=0.05,
        device="cpu",
    )
    assert len(predictions) == len(ids)
    assert all(len(row["probabilities"]) == 2 for row in predictions)


def test_metrics_and_paired_video_bootstrap_are_deterministic():
    families = {}
    for family in readout.FAMILY_IDS:
        rows = []
        for video in ("a", "b", "c", "d"):
            for seed in range(2):
                truth = "x" if video in {"a", "c"} else "y"
                correct = family != "majority_null" or truth == "x"
                pred = truth if correct else "x"
                rows.append(
                    {
                        "candidate_id": f"{video}-{seed}",
                        "video_id": video,
                        "true_label": truth,
                        "predicted_label": pred,
                        "confidence": 0.8,
                        "abstain": False,
                    }
                )
        families[family] = rows
    first, _ = readout._metric_bootstrap(families, draws=100, seed=7)
    second, _ = readout._metric_bootstrap(families, draws=100, seed=7)
    assert first == second
    assert first["conditioning_only"]["balanced_accuracy"] == pytest.approx(1.0)
    assert first["majority_null"]["balanced_accuracy"] == pytest.approx(0.5)


def test_external_without_internal_is_reported_not_earlier():
    rows = []
    for target in readout.TARGETS:
        for progress in readout.S_POINTS:
            for family in readout.FAMILY_IDS:
                external_supported = family == "external_preview" and progress >= 0.45
                rows.append(
                    {
                        "target": target,
                        "progress": progress,
                        "family": family,
                        "balanced_accuracy": 0.2,
                        "expected_calibration_error": 0.2,
                        "coverage": 0.5,
                        "paired_balanced_accuracy_differences": {
                            "minus_majority_null": {
                                "point": 0.1 if external_supported else 0.0,
                                "ci": [0.01, 0.2] if external_supported else [-0.1, 0.1],
                            },
                            "minus_conditioning_only": {
                                "point": 0.1 if external_supported else 0.0,
                                "ci": [0.01, 0.2] if external_supported else [-0.1, 0.1],
                            },
                            "minus_external_preview": {
                                "point": 0.0,
                                "ci": [-0.1, 0.1],
                            },
                        },
                    }
                )
    decisions = readout._readout_decisions(rows)
    for target in readout.TARGETS:
        assert decisions[target]["external_preview_information_progress"] == 0.45
        assert decisions[target]["earliest_supported_internal_progress"] is None
        assert decisions[target]["conclusion"] == (
            "internal readout not earlier than external preview"
        )
        assert decisions[target]["scientific_status"] == "NOT_SUPPORTED"


def test_target_validator_rejects_corruption_before_science(tmp_path):
    rows_path = tmp_path / "targets.jsonl"
    atomic_jsonl_create(rows_path, [{"candidate_id": "one"}])
    completion = {
        "schema": readout.TARGET_SCHEMA,
        "status": "COMPLETE",
        "candidate_count": readout.EXPECTED_UNITS,
        "protocol_sha256": readout.PINNED_PROTOCOL_SHA256,
        "targets_file": rows_path.name,
        "targets_sha256": "0" * 64,
        "targets_bytes": rows_path.stat().st_size,
    }
    completion_path = tmp_path / "TARGETS_COMPLETION.json"
    atomic_json_create(completion_path, completion)
    with pytest.raises(readout.InternalReadoutError, match="hash/size"):
        readout.validate_targets(completion_path)


def test_mlp_and_single_query_capacity_contracts():
    pytest.importorskip("sklearn")
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    rng = np.random.default_rng(13)
    y = ["a"] * 12 + ["b"] * 12
    X = np.concatenate(
        (rng.normal(1, 0.1, (12, 8)), rng.normal(-1, 0.1, (12, 8))), axis=0
    ).astype(np.float32)
    pred, probability = readout.fit_mlp_predict(X, y, X[:4], ["a", "b"], 0.001, 3)
    pred_repeat, probability_repeat = readout.fit_mlp_predict(
        X, y, X[:4], ["a", "b"], 0.001, 3
    )
    assert len(pred) == 4 and probability.shape == (4, 2)
    assert pred == pred_repeat
    assert hashlib.sha256(probability.tobytes()).digest() == hashlib.sha256(
        probability_repeat.tobytes()
    ).digest()
    tokens = np.repeat(X[:, None, :4], 3, axis=1)
    tokens = np.pad(tokens, ((0, 0), (0, 0), (0, 28)))
    pred, probability = readout.fit_attention_predict(
        tokens, y, tokens[:4], ["a", "b"], 0.001, 3, "cpu"
    )
    pred_repeat, probability_repeat = readout.fit_attention_predict(
        tokens, y, tokens[:4], ["a", "b"], 0.001, 3, "cpu"
    )
    assert len(pred) == 4 and probability.shape == (4, 2)
    assert np.allclose(probability.sum(axis=1), 1.0)
    assert pred == pred_repeat
    assert hashlib.sha256(probability.tobytes()).digest() == hashlib.sha256(
        probability_repeat.tobytes()
    ).digest()
