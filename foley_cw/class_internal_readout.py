"""Nested video-grouped Class probes for the lineage-valid B2 feature bank.

This module is deliberately downstream of both immutable inputs: the full B2
Class posterior bank and the B-1-gated feature recollection.  It never fits on
old feature bundles.  Every reported score is reconstructed from outer-fold
candidate predictions, with video IDs (and therefore all seeds/progress) kept
together.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .b1_lineage import describe_array, sha256_file
from .b2_class_closure import (
    PINNED_PROTOCOL_SHA256,
    atomic_json_create,
    atomic_jsonl_create,
    canonical_json_bytes,
    sha256_bytes,
)
from .b2_feature_recollection import EXPECTED_UNITS, S_POINTS


IMPLEMENTATION_SCHEMA = "sounddecisions.class_internal_readout_implementation.v1"
TARGET_SCHEMA = "sounddecisions.class_readout_targets.v1"
SHARD_SCHEMA = "sounddecisions.class_internal_readout_shard.v1"
MERGE_SCHEMA = "sounddecisions.class_internal_readout_merge.v1"
FOLD_PREFIX = "non-human-readout-v1:"
PROJECTION_PREFIX = "non-human-readout-v1:projection:"
EXPECTED_CANDIDATES = 816
OUTER_FOLDS = 6
INNER_FOLDS = 4
BOOTSTRAP_DRAWS = 5000
BOOTSTRAP_SEED = 20260717
MAX_VECTOR_WIDTH = 256
ATTENTION_WIDTH = 32
LINEAR_GRID = (0.01, 0.1, 1.0)
NONLINEAR_GRID = (0.0001, 0.001)
SELECTIVE_THRESHOLD = 0.5

FAMILY_SPECS: tuple[tuple[str, str, str], ...] = (
    ("majority_null", "majority/null", "null"),
    ("conditioning_only", "video-conditioning-only", "vector"),
    ("external_preview", "external-preview PANNs representation", "vector"),
    ("latent_only", "latent-only", "vector"),
    ("velocity_only", "velocity-only", "vector"),
    ("tweedie_latent", "Tweedie-latent", "vector"),
    ("pooled_linear", "pooled linear", "vector"),
    ("pooled_mlp", "fixed-capacity MLP", "mlp"),
    ("token_statistics", "token statistics", "vector"),
    ("single_query_attention", "single-query token attention", "attention"),
    ("selected_cross_attention", "selected cross-attention representation", "vector"),
)
FAMILY_IDS = tuple(row[0] for row in FAMILY_SPECS)
FAMILY_KIND = {row[0]: row[2] for row in FAMILY_SPECS}
VIEW_FOR_FAMILY = {
    "conditioning_only": "conditioning",
    "external_preview": "external_preview",
    "latent_only": "latent",
    "velocity_only": "velocity",
    "tweedie_latent": "tweedie",
    "pooled_linear": "pooled",
    "pooled_mlp": "pooled",
    "token_statistics": "token_statistics",
    "single_query_attention": "attention_tokens",
    "selected_cross_attention": "selected_cross_attention",
}
INTERNAL_FAMILIES = (
    "latent_only",
    "velocity_only",
    "tweedie_latent",
    "pooled_linear",
    "pooled_mlp",
    "token_statistics",
    "single_query_attention",
    "selected_cross_attention",
)
TARGETS = ("fork_majority", "ode_final")


class InternalReadoutError(RuntimeError):
    """Fail-closed readout artifact, grouping, or model-contract error."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InternalReadoutError(f"invalid JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise InternalReadoutError(f"expected JSON object: {path}")
    return payload


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            for lineno, line in enumerate(handle, 1):
                if not line.strip():
                    raise InternalReadoutError(f"blank JSONL row {path}:{lineno}")
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise InternalReadoutError(f"non-object JSONL row {path}:{lineno}")
                rows.append(value)
    except InternalReadoutError:
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise InternalReadoutError(f"invalid JSONL {path}: {exc}") from exc
    return rows


def validate_protocols(
    protocol_path: Path,
    implementation_path: Path,
    *,
    protocol_sha256: str = PINNED_PROTOCOL_SHA256,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    if sha256_file(protocol_path) != protocol_sha256:
        raise InternalReadoutError("Class readout parent protocol hash mismatch")
    protocol = _load_json(protocol_path)
    readout = protocol.get("class_readout", {})
    if readout.get("outer_folds") != OUTER_FOLDS or readout.get("inner_folds") != INNER_FOLDS:
        raise InternalReadoutError("parent protocol fold contract mismatch")
    if readout.get("families") != [row[1] for row in FAMILY_SPECS]:
        raise InternalReadoutError("implemented probe family does not match parent protocol")
    if tuple(float(v) for v in readout.get("linear_regularization_grid", [])) != LINEAR_GRID:
        raise InternalReadoutError("linear regularization grid mismatch")
    if tuple(float(v) for v in readout.get("mlp_alpha_grid", [])) != NONLINEAR_GRID:
        raise InternalReadoutError("MLP regularization grid mismatch")
    if tuple(float(v) for v in readout.get("single_query_weight_decay_grid", [])) != NONLINEAR_GRID:
        raise InternalReadoutError("attention regularization grid mismatch")
    if int(readout.get("mlp_hidden_units", -1)) != 64:
        raise InternalReadoutError("MLP capacity mismatch")
    if int(readout.get("single_query_width", -1)) != ATTENTION_WIDTH:
        raise InternalReadoutError("attention capacity mismatch")
    implementation_sha256 = sha256_file(implementation_path)
    implementation = _load_json(implementation_path)
    if (
        implementation.get("schema") != IMPLEMENTATION_SCHEMA
        or implementation.get("status") != "FROZEN_BEFORE_B2_FEATURE_RECOLLECTION"
        or implementation.get("parent_protocol_sha256") != protocol_sha256
    ):
        raise InternalReadoutError("invalid Class readout implementation freeze")
    return protocol, implementation, implementation_sha256


def ordered_videos(video_ids: Iterable[str]) -> list[str]:
    values = sorted({str(value) for value in video_ids})
    return sorted(
        values,
        key=lambda value: hashlib.sha256((FOLD_PREFIX + value).encode("utf-8")).hexdigest(),
    )


def grouped_folds(video_ids: Iterable[str], n_folds: int) -> dict[str, int]:
    ordered = ordered_videos(video_ids)
    if len(ordered) < n_folds:
        raise InternalReadoutError(f"{len(ordered)} video groups cannot populate {n_folds} folds")
    return {video_id: index % n_folds for index, video_id in enumerate(ordered)}


def validate_no_group_leakage(
    groups: Sequence[str], train_indices: Sequence[int], test_indices: Sequence[int]
) -> None:
    train_groups = {str(groups[index]) for index in train_indices}
    test_groups = {str(groups[index]) for index in test_indices}
    overlap = train_groups & test_groups
    if overlap:
        raise InternalReadoutError(f"video leakage across fold: {sorted(overlap)}")


def candidate_id(video_id: str, base_seed: int, progress: float) -> str:
    return f"{str(video_id)}__seed{int(base_seed)}__s{float(progress):.2f}"


def fork_majority_target(
    labels: Sequence[str], class_names: Sequence[str]
) -> tuple[str | None, str | None, dict[str, int]]:
    if any(str(label) not in class_names for label in labels):
        raise InternalReadoutError("fork confident label outside frozen taxonomy")
    counts = Counter(str(label) for label in labels)
    primary: str | None = None
    reason: str | None = None
    if len(labels) < 2:
        reason = "confident_forks_lt_2"
    else:
        best = max(counts.values())
        winners = sorted(label for label, count in counts.items() if count == best)
        if len(winners) == 1:
            primary = winners[0]
        else:
            reason = "fork_majority_tie"
    return primary, reason, {name: int(counts.get(name, 0)) for name in class_names}


def _validate_class_completion(completion_path: Path) -> tuple[dict[str, Any], Path]:
    completion = _load_json(completion_path)
    if (
        completion.get("schema_version") != "sounddecisions.b2_class_posterior_merge.v1"
        or completion.get("status") != "COMPLETE"
        or completion.get("canonical_b2") is not True
        or int(completion.get("record_count", -1)) != 79152
        or completion.get("protocol_sha256") != PINNED_PROTOCOL_SHA256
    ):
        raise InternalReadoutError("invalid canonical Class posterior completion")
    data_path = Path(completion_path).parent / str(completion.get("data_file", ""))
    if (
        not data_path.is_file()
        or sha256_file(data_path) != completion.get("data_sha256")
        or data_path.stat().st_size != int(completion.get("data_bytes", -1))
    ):
        raise InternalReadoutError("Class posterior data hash/size mismatch")
    return completion, data_path


def _feature_rows(feature_completion_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    completion = _load_json(feature_completion_path)
    if (
        completion.get("schema") != "sounddecisions.b2_feature_recollection_merge.v1"
        or completion.get("status") != "COMPLETE"
        or completion.get("canonical_b2") is not True
        or int(completion.get("unit_count", -1)) != EXPECTED_UNITS
        or int(completion.get("base_trajectory_count", -1)) != EXPECTED_CANDIDATES
        or completion.get("protocol_sha256") != PINNED_PROTOCOL_SHA256
    ):
        raise InternalReadoutError("invalid canonical feature recollection completion")
    manifest_path = Path(feature_completion_path).parent / str(completion.get("manifest", ""))
    if (
        sha256_file(manifest_path) != completion.get("manifest_sha256")
        or manifest_path.stat().st_size != int(completion.get("manifest_bytes", -1))
    ):
        raise InternalReadoutError("feature recollection manifest hash/size mismatch")
    rows = _load_jsonl(manifest_path)
    identities = {
        (str(row["video_id"]), int(row["base_seed"]), float(row["progress"])) for row in rows
    }
    expected = {
        (video_id, seed, progress)
        for video_id in {str(row["video_id"]) for row in rows}
        for seed in range(17)
        for progress in S_POINTS
    }
    if len(rows) != EXPECTED_UNITS or identities != expected or len(expected) != EXPECTED_UNITS:
        raise InternalReadoutError("feature recollection scientific grid mismatch")
    return completion, rows


def construct_targets(class_completion_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Reconstruct both targets without reading any feature values."""
    completion, data_path = _validate_class_completion(class_completion_path)
    try:
        with np.load(data_path, allow_pickle=False) as archive:
            required = {
                "video_id", "base_seed", "progress", "role", "fork_index",
                "abstain", "confident_label", "coarse_class_names", "record_id",
            }
            if not required.issubset(archive.files):
                raise InternalReadoutError("Class posterior target fields are incomplete")
            data = {key: np.asarray(archive[key]) for key in required}
    except InternalReadoutError:
        raise
    except Exception as exc:
        raise InternalReadoutError(f"cannot load Class posteriors: {exc}") from exc
    class_names = [str(value) for value in data["coarse_class_names"].reshape(-1)]
    if len(class_names) != 15 or len(set(class_names)) != 15:
        raise InternalReadoutError("coarse Class universe is not the frozen 15-way map")
    n = int(data["video_id"].shape[0])
    for key in ("base_seed", "progress", "role", "fork_index", "abstain", "confident_label", "record_id"):
        if data[key].shape[0] != n:
            raise InternalReadoutError(f"Class posterior target array length mismatch: {key}")
    base_labels: dict[tuple[str, int], str | None] = {}
    fork_groups: dict[tuple[str, int, float], list[int]] = defaultdict(list)
    for index in range(n):
        video_id = str(data["video_id"][index])
        seed = int(data["base_seed"][index])
        role = str(data["role"][index])
        if role == "base":
            key = (video_id, seed)
            if key in base_labels:
                raise InternalReadoutError(f"duplicate B2 base target {key}")
            base_labels[key] = (
                None if bool(data["abstain"][index]) else str(data["confident_label"][index])
            )
        elif role == "fork":
            progress = float(np.round(float(data["progress"][index]), 2))
            fork_groups[(video_id, seed, progress)].append(index)
        else:
            raise InternalReadoutError(f"unexpected Class posterior role {role!r}")
    videos = sorted({key[0] for key in base_labels})
    if len(base_labels) != EXPECTED_CANDIDATES or len(videos) != 48:
        raise InternalReadoutError("base-final target population mismatch")
    rows: list[dict[str, Any]] = []
    primary_observed = Counter()
    secondary_observed = Counter()
    for video_id in videos:
        for seed in range(17):
            secondary = base_labels.get((video_id, seed))
            if secondary is not None and secondary not in class_names:
                raise InternalReadoutError(f"unknown base-final Class {secondary!r}")
            for progress in S_POINTS:
                indices = sorted(
                    fork_groups.get((video_id, seed, float(np.round(progress, 2))), []),
                    key=lambda idx: int(data["fork_index"][idx]),
                )
                if len(indices) != 12 or [int(data["fork_index"][idx]) for idx in indices] != list(range(12)):
                    raise InternalReadoutError(
                        f"fork target cardinality/index mismatch: {video_id}/{seed}/{progress}"
                    )
                labels = [
                    str(data["confident_label"][idx])
                    for idx in indices
                    if not bool(data["abstain"][idx])
                ]
                primary, reason, class_counts = fork_majority_target(labels, class_names)
                if primary is not None:
                    primary_observed[float(progress)] += 1
                if secondary is not None:
                    secondary_observed[float(progress)] += 1
                rows.append(
                    {
                        "candidate_id": candidate_id(video_id, seed, progress),
                        "video_id": video_id,
                        "base_seed": seed,
                        "progress": float(progress),
                        "fork_majority_label": primary,
                        "fork_majority_missing_reason": reason,
                        "fork_confident_count": len(labels),
                        "fork_class_counts": class_counts,
                        "ode_final_label": secondary,
                        "ode_final_missing_reason": None if secondary is not None else "base_final_abstained",
                    }
                )
    rows.sort(key=lambda row: (row["video_id"], row["base_seed"], row["progress"]))
    if len(rows) != EXPECTED_UNITS or len({row["candidate_id"] for row in rows}) != EXPECTED_UNITS:
        raise InternalReadoutError("target candidate grid mismatch")
    summary = {
        "schema": TARGET_SCHEMA,
        "status": "COMPLETE",
        "candidate_count": len(rows),
        "video_count": len(videos),
        "base_seed_count": 17,
        "progress_points": list(S_POINTS),
        "class_names": class_names,
        "observed_by_progress": {
            "fork_majority": {str(s): int(primary_observed[s]) for s in S_POINTS},
            "ode_final": {str(s): int(secondary_observed[s]) for s in S_POINTS},
        },
        "class_posterior_completion": str(Path(class_completion_path).resolve()),
        "class_posterior_completion_sha256": sha256_file(class_completion_path),
        "class_posterior_data_sha256": completion["data_sha256"],
        "protocol_sha256": PINNED_PROTOCOL_SHA256,
    }
    return rows, summary


def prepare_targets(
    class_completion_path: Path,
    feature_completion_path: Path,
    protocol_path: Path,
    implementation_path: Path,
    out_dir: Path,
) -> Path:
    _, _, implementation_sha256 = validate_protocols(protocol_path, implementation_path)
    feature_completion, feature_rows = _feature_rows(feature_completion_path)
    rows, summary = construct_targets(class_completion_path)
    feature_ids = {
        candidate_id(str(row["video_id"]), int(row["base_seed"]), float(row["progress"]))
        for row in feature_rows
    }
    if feature_ids != {str(row["candidate_id"]) for row in rows}:
        raise InternalReadoutError("target and feature candidate IDs do not match")
    out_dir = Path(out_dir)
    try:
        out_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise InternalReadoutError(f"refusing to overwrite targets {out_dir}") from exc
    targets_path = out_dir / "CLASS_READOUT_TARGETS.jsonl"
    atomic_jsonl_create(targets_path, rows)
    summary.update(
        {
            "targets_file": targets_path.name,
            "targets_sha256": sha256_file(targets_path),
            "targets_bytes": targets_path.stat().st_size,
            "feature_completion": str(Path(feature_completion_path).resolve()),
            "feature_completion_sha256": sha256_file(feature_completion_path),
            "feature_manifest_sha256": feature_completion["manifest_sha256"],
            "implementation": str(Path(implementation_path).resolve()),
            "implementation_sha256": implementation_sha256,
        }
    )
    atomic_json_create(out_dir / "TARGETS_COMPLETION.json", summary)
    return out_dir


def validate_targets(completion_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    completion = _load_json(completion_path)
    if (
        completion.get("schema") != TARGET_SCHEMA
        or completion.get("status") != "COMPLETE"
        or int(completion.get("candidate_count", -1)) != EXPECTED_UNITS
        or completion.get("protocol_sha256") != PINNED_PROTOCOL_SHA256
    ):
        raise InternalReadoutError("invalid target completion")
    path = Path(completion_path).parent / str(completion.get("targets_file", ""))
    if (
        sha256_file(path) != completion.get("targets_sha256")
        or path.stat().st_size != int(completion.get("targets_bytes", -1))
    ):
        raise InternalReadoutError("target file hash/size mismatch")
    rows = _load_jsonl(path)
    if len(rows) != EXPECTED_UNITS or len({row.get("candidate_id") for row in rows}) != EXPECTED_UNITS:
        raise InternalReadoutError("target rows are partial or duplicated")
    return completion, rows


def _seed_from_text(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "little")


def fixed_projection(view_name: str, input_width: int, output_width: int) -> np.ndarray:
    if input_width < 1 or output_width < 1:
        raise InternalReadoutError("invalid projection dimensions")
    rng = np.random.default_rng(_seed_from_text(PROJECTION_PREFIX + view_name))
    matrix = rng.standard_normal((input_width, output_width)).astype(np.float32)
    matrix /= np.float32(math.sqrt(output_width))
    return matrix


def project_vectors(X: np.ndarray, view_name: str) -> tuple[np.ndarray, dict[str, Any]]:
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2 or not np.all(np.isfinite(X)):
        raise InternalReadoutError(f"invalid vector matrix for {view_name}")
    if X.shape[1] <= MAX_VECTOR_WIDTH:
        return X, {
            "method": "identity",
            "input_width": int(X.shape[1]),
            "output_width": int(X.shape[1]),
            "matrix_sha256": None,
        }
    matrix = fixed_projection(view_name, X.shape[1], MAX_VECTOR_WIDTH)
    projected = np.asarray(X @ matrix, dtype=np.float32)
    return projected, {
        "method": "fixed_gaussian",
        "input_width": int(X.shape[1]),
        "output_width": MAX_VECTOR_WIDTH,
        "matrix_sha256": sha256_bytes(matrix.tobytes(order="C")),
    }


def project_attention(tokens: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    tokens = np.asarray(tokens, dtype=np.float32)
    if tokens.ndim != 3 or not np.all(np.isfinite(tokens)):
        raise InternalReadoutError("attention token matrix must be finite (n,tokens,width)")
    matrix = fixed_projection("attention_tokens", tokens.shape[2], ATTENTION_WIDTH)
    projected = np.asarray(tokens @ matrix, dtype=np.float32)
    return projected, {
        "method": "fixed_gaussian_channel_projection",
        "input_width": int(tokens.shape[2]),
        "output_width": ATTENTION_WIDTH,
        "matrix_sha256": sha256_bytes(matrix.tobytes(order="C")),
    }


def _standardize_fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = np.asarray(X.mean(axis=0), dtype=np.float64)
    sd = np.asarray(X.std(axis=0), dtype=np.float64)
    sd[sd == 0.0] = 1.0
    return mu, sd


def _softmax(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    values = values - values.max(axis=1, keepdims=True)
    exp = np.exp(values)
    return exp / exp.sum(axis=1, keepdims=True)


def _class_weights(y: Sequence[str]) -> np.ndarray:
    counts = Counter(str(value) for value in y)
    weights = np.asarray([len(y) / (len(counts) * counts[str(value)]) for value in y], dtype=np.float64)
    return weights / weights.mean()


def fit_ridge_predict(
    X_train: np.ndarray,
    y_train: Sequence[str],
    X_test: np.ndarray,
    class_names: Sequence[str],
    regularization: float,
) -> tuple[list[str], np.ndarray]:
    classes = list(class_names)
    if not y_train or X_test.shape[0] == 0:
        raise InternalReadoutError("ridge fit received empty train/test")
    present = sorted(set(str(value) for value in y_train))
    if len(present) == 1:
        probabilities = np.zeros((X_test.shape[0], len(classes)), dtype=np.float64)
        probabilities[:, classes.index(present[0])] = 1.0
        return [present[0]] * X_test.shape[0], probabilities
    mu, sd = _standardize_fit(np.asarray(X_train, dtype=np.float64))
    train = (np.asarray(X_train, dtype=np.float64) - mu) / sd
    test = (np.asarray(X_test, dtype=np.float64) - mu) / sd
    train_b = np.concatenate((train, np.ones((train.shape[0], 1))), axis=1)
    test_b = np.concatenate((test, np.ones((test.shape[0], 1))), axis=1)
    fitted_classes = present
    class_index = {name: index for index, name in enumerate(fitted_classes)}
    Y = np.zeros((len(y_train), len(fitted_classes)), dtype=np.float64)
    for row, label in enumerate(y_train):
        Y[row, class_index[str(label)]] = 1.0
    sample_weight = _class_weights(y_train)
    weighted = train_b * sample_weight[:, None]
    penalty = np.eye(train_b.shape[1], dtype=np.float64) * float(regularization)
    penalty[-1, -1] = 0.0
    W = np.linalg.solve(train_b.T @ weighted + penalty, train_b.T @ (Y * sample_weight[:, None]))
    fitted_probabilities = _softmax(test_b @ W)
    probabilities = np.zeros((X_test.shape[0], len(classes)), dtype=np.float64)
    for index, label in enumerate(fitted_classes):
        probabilities[:, classes.index(label)] = fitted_probabilities[:, index]
    indices = probabilities.argmax(axis=1)
    return [classes[index] for index in indices], probabilities


def fit_majority_predict(
    y_train: Sequence[str], n_test: int, class_names: Sequence[str]
) -> tuple[list[str], np.ndarray]:
    counts = Counter(str(value) for value in y_train)
    if not counts or n_test < 1:
        raise InternalReadoutError("majority fit received empty train/test")
    classes = list(class_names)
    majority = sorted(counts, key=lambda label: (-counts[label], label))[0]
    prevalence = np.asarray([counts.get(label, 0) / len(y_train) for label in classes], dtype=np.float64)
    probabilities = np.repeat(prevalence[None, :], n_test, axis=0)
    return [majority] * n_test, probabilities


def balanced_accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> tuple[float, dict[str, float]]:
    if len(y_true) != len(y_pred) or not y_true:
        return float("nan"), {}
    recalls: dict[str, float] = {}
    for label in sorted(set(str(value) for value in y_true)):
        indices = [index for index, value in enumerate(y_true) if str(value) == label]
        recalls[label] = float(np.mean([str(y_pred[index]) == label for index in indices]))
    return float(np.mean(list(recalls.values()))), recalls


def expected_calibration_error(
    y_true: Sequence[str], y_pred: Sequence[str], confidence: Sequence[float], bins: int = 10
) -> float:
    if not y_true:
        return float("nan")
    conf = np.asarray(confidence, dtype=np.float64)
    correct = np.asarray([str(a) == str(b) for a, b in zip(y_true, y_pred)], dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y_true)
    ece = 0.0
    for index in range(bins):
        mask = (conf >= edges[index]) & (conf < edges[index + 1] if index + 1 < bins else conf <= 1.0)
        if np.any(mask):
            ece += float(mask.sum() / total) * abs(float(correct[mask].mean() - conf[mask].mean()))
    return float(ece)


def _model_seed(*parts: Any) -> int:
    return _seed_from_text(":".join(str(part) for part in parts)) % (2**31 - 1)


def fit_mlp_predict(
    X_train: np.ndarray,
    y_train: Sequence[str],
    X_test: np.ndarray,
    class_names: Sequence[str],
    weight_decay: float,
    seed: int,
) -> tuple[list[str], np.ndarray]:
    present = sorted(set(str(value) for value in y_train))
    if len(present) == 1:
        probabilities = np.zeros((X_test.shape[0], len(class_names)), dtype=np.float64)
        probabilities[:, list(class_names).index(present[0])] = 1.0
        return [present[0]] * X_test.shape[0], probabilities
    try:
        from sklearn.neural_network import MLPClassifier
    except Exception as exc:
        raise InternalReadoutError(f"scikit-learn MLP is unavailable: {exc}") from exc
    mu, sd = _standardize_fit(np.asarray(X_train, dtype=np.float64))
    train = (np.asarray(X_train, dtype=np.float64) - mu) / sd
    test = (np.asarray(X_test, dtype=np.float64) - mu) / sd
    model = MLPClassifier(
        hidden_layer_sizes=(64,),
        activation="relu",
        solver="adam",
        alpha=float(weight_decay),
        batch_size=min(64, len(y_train)),
        learning_rate_init=0.001,
        max_iter=120,
        shuffle=True,
        random_state=int(seed),
        tol=0.0,
        n_iter_no_change=121,
        early_stopping=False,
    )
    try:
        model.fit(train, np.asarray(y_train), sample_weight=_class_weights(y_train))
    except TypeError as exc:
        raise InternalReadoutError(
            "installed MLPClassifier lacks the frozen sample-weight contract"
        ) from exc
    fitted = [str(value) for value in model.classes_]
    fitted_probabilities = np.asarray(model.predict_proba(test), dtype=np.float64)
    probabilities = np.zeros((X_test.shape[0], len(class_names)), dtype=np.float64)
    for index, label in enumerate(fitted):
        probabilities[:, list(class_names).index(label)] = fitted_probabilities[:, index]
    indices = probabilities.argmax(axis=1)
    return [str(class_names[index]) for index in indices], probabilities


def fit_attention_predict(
    X_train: np.ndarray,
    y_train: Sequence[str],
    X_test: np.ndarray,
    class_names: Sequence[str],
    weight_decay: float,
    seed: int,
    device: str,
) -> tuple[list[str], np.ndarray]:
    present = sorted(set(str(value) for value in y_train))
    if len(present) == 1:
        probabilities = np.zeros((X_test.shape[0], len(class_names)), dtype=np.float64)
        probabilities[:, list(class_names).index(present[0])] = 1.0
        return [present[0]] * X_test.shape[0], probabilities
    try:
        import torch
    except Exception as exc:
        raise InternalReadoutError(f"PyTorch attention probe is unavailable: {exc}") from exc
    train_np = np.asarray(X_train, dtype=np.float32)
    test_np = np.asarray(X_test, dtype=np.float32)
    if train_np.ndim != 3 or train_np.shape[2] != ATTENTION_WIDTH:
        raise InternalReadoutError("projected attention input violates frozen width")
    mu = train_np.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
    sd = train_np.std(axis=(0, 1), dtype=np.float64).astype(np.float32)
    sd[sd == 0.0] = 1.0
    train_np = (train_np - mu[None, None, :]) / sd[None, None, :]
    test_np = (test_np - mu[None, None, :]) / sd[None, None, :]
    torch.manual_seed(int(seed))
    if str(device).startswith("cuda"):
        if not torch.cuda.is_available():
            raise InternalReadoutError(f"requested attention device unavailable: {device}")
        torch.cuda.manual_seed_all(int(seed))
    torch.use_deterministic_algorithms(True)
    fitted_index = {label: index for index, label in enumerate(present)}
    y_codes = np.asarray([fitted_index[str(label)] for label in y_train], dtype=np.int64)

    class SingleQuery(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.query = torch.nn.Parameter(torch.zeros(ATTENTION_WIDTH))
            self.head = torch.nn.Linear(ATTENTION_WIDTH, len(present))

        def forward(self, values: Any) -> Any:
            logits = torch.einsum("ntd,d->nt", values, self.query) / math.sqrt(ATTENTION_WIDTH)
            pooled = torch.einsum("nt,ntd->nd", torch.softmax(logits, dim=1), values)
            return self.head(pooled)

    model = SingleQuery().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=float(weight_decay))
    counts = Counter(str(label) for label in y_train)
    loss_weight = torch.tensor(
        [len(y_train) / (len(present) * counts[label]) for label in present],
        dtype=torch.float32,
        device=device,
    )
    criterion = torch.nn.CrossEntropyLoss(weight=loss_weight)
    train_tensor = torch.from_numpy(train_np).to(device=device, dtype=torch.float32)
    target_tensor = torch.from_numpy(y_codes).to(device=device)
    model.train()
    for _ in range(40):
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(train_tensor), target_tensor)
        if not torch.isfinite(loss):
            raise InternalReadoutError("nonfinite single-query attention loss")
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.inference_mode():
        fitted_probabilities = torch.softmax(
            model(torch.from_numpy(test_np).to(device=device, dtype=torch.float32)), dim=1
        ).cpu().numpy().astype(np.float64)
    probabilities = np.zeros((X_test.shape[0], len(class_names)), dtype=np.float64)
    for index, label in enumerate(present):
        probabilities[:, list(class_names).index(label)] = fitted_probabilities[:, index]
    indices = probabilities.argmax(axis=1)
    return [str(class_names[index]) for index in indices], probabilities


def _fit_predict(
    family: str,
    X_train: np.ndarray | None,
    y_train: Sequence[str],
    X_test: np.ndarray | None,
    class_names: Sequence[str],
    parameter: float | None,
    seed: int,
    device: str,
) -> tuple[list[str], np.ndarray]:
    kind = FAMILY_KIND[family]
    n_test = 0 if X_test is None else int(X_test.shape[0])
    if family == "majority_null":
        return fit_majority_predict(y_train, n_test, class_names)
    if X_train is None or X_test is None or parameter is None:
        raise InternalReadoutError(f"model input/parameter absent for {family}")
    if kind == "vector":
        return fit_ridge_predict(X_train, y_train, X_test, class_names, parameter)
    if kind == "mlp":
        return fit_mlp_predict(X_train, y_train, X_test, class_names, parameter, seed)
    if kind == "attention":
        return fit_attention_predict(
            X_train, y_train, X_test, class_names, parameter, seed, device
        )
    raise InternalReadoutError(f"unknown model kind {kind}")


def _parameter_grid(family: str) -> tuple[float | None, ...]:
    if family == "majority_null":
        return (None,)
    if FAMILY_KIND[family] == "vector":
        return tuple(float(value) for value in LINEAR_GRID)
    return tuple(float(value) for value in NONLINEAR_GRID)


def _select_parameter(
    family: str,
    X: np.ndarray | None,
    labels: Sequence[str],
    groups: Sequence[str],
    outer_train: np.ndarray,
    all_videos: Sequence[str],
    class_names: Sequence[str],
    *,
    target: str,
    progress: float,
    outer_fold: int,
    device: str,
) -> tuple[float | None, list[dict[str, Any]]]:
    grid = _parameter_grid(family)
    if family == "majority_null":
        return None, [{"parameter": None, "mean_inner_balanced_accuracy": None, "fold_scores": []}]
    outer_video_set = {str(groups[index]) for index in outer_train}
    inner_map = grouped_folds([video for video in all_videos if video in outer_video_set], INNER_FOLDS)
    evaluations: list[dict[str, Any]] = []
    for parameter in grid:
        fold_scores: list[float] = []
        for inner_fold in range(INNER_FOLDS):
            inner_test = np.asarray(
                [index for index in outer_train if inner_map[str(groups[index])] == inner_fold],
                dtype=np.int64,
            )
            inner_train = np.asarray(
                [index for index in outer_train if inner_map[str(groups[index])] != inner_fold],
                dtype=np.int64,
            )
            validate_no_group_leakage(groups, inner_train, inner_test)
            if inner_train.size == 0 or inner_test.size == 0:
                raise InternalReadoutError("empty inner video fold")
            pred, _ = _fit_predict(
                family,
                None if X is None else X[inner_train],
                [labels[index] for index in inner_train],
                None if X is None else X[inner_test],
                class_names,
                parameter,
                _model_seed(target, progress, family, outer_fold, inner_fold, parameter),
                device,
            )
            score, _ = balanced_accuracy([labels[index] for index in inner_test], pred)
            fold_scores.append(score)
        evaluations.append(
            {
                "parameter": parameter,
                "mean_inner_balanced_accuracy": float(np.mean(fold_scores)),
                "fold_scores": [float(value) for value in fold_scores],
            }
        )
    best_score = max(float(row["mean_inner_balanced_accuracy"]) for row in evaluations)
    tied = [
        float(row["parameter"])
        for row in evaluations
        if abs(float(row["mean_inner_balanced_accuracy"]) - best_score) <= 1e-12
    ]
    return max(tied), evaluations


def nested_outer_predictions(
    family: str,
    X: np.ndarray | None,
    labels: Sequence[str],
    groups: Sequence[str],
    candidate_ids: Sequence[str],
    all_videos: Sequence[str],
    class_names: Sequence[str],
    *,
    target: str,
    progress: float,
    device: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if family not in FAMILY_IDS:
        raise InternalReadoutError(f"unknown probe family {family}")
    n = len(labels)
    if family == "majority_null" and X is None:
        X = np.zeros((n, 0), dtype=np.float32)
    if n == 0 or len(groups) != n or len(candidate_ids) != n or (X is not None and X.shape[0] != n):
        raise InternalReadoutError("nested-CV candidate arrays are inconsistent")
    outer_map = grouped_folds(all_videos, OUTER_FOLDS)
    predictions: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    seen: set[int] = set()
    for outer_fold in range(OUTER_FOLDS):
        test = np.asarray(
            [index for index, group in enumerate(groups) if outer_map[str(group)] == outer_fold],
            dtype=np.int64,
        )
        train = np.asarray(
            [index for index, group in enumerate(groups) if outer_map[str(group)] != outer_fold],
            dtype=np.int64,
        )
        validate_no_group_leakage(groups, train, test)
        if train.size == 0 or test.size == 0:
            raise InternalReadoutError("empty outer video fold")
        selected, evaluations = _select_parameter(
            family,
            X,
            labels,
            groups,
            train,
            all_videos,
            class_names,
            target=target,
            progress=progress,
            outer_fold=outer_fold,
            device=device,
        )
        predicted, probabilities = _fit_predict(
            family,
            None if X is None else X[train],
            [labels[index] for index in train],
            None if X is None else X[test],
            class_names,
            selected,
            _model_seed(target, progress, family, outer_fold, "outer", selected),
            device,
        )
        if probabilities.shape != (test.size, len(class_names)):
            raise InternalReadoutError("outer prediction probability shape mismatch")
        if not np.all(np.isfinite(probabilities)) or not np.allclose(
            probabilities.sum(axis=1), 1.0, atol=1e-6, rtol=0.0
        ):
            raise InternalReadoutError("outer prediction probabilities are invalid")
        selections.append(
            {
                "target": target,
                "progress": float(progress),
                "family": family,
                "outer_fold": outer_fold,
                "selected_parameter": selected,
                "inner_evaluations": evaluations,
                "train_video_ids": sorted({str(groups[index]) for index in train}),
                "test_video_ids": sorted({str(groups[index]) for index in test}),
                "train_candidate_count": int(train.size),
                "test_candidate_count": int(test.size),
            }
        )
        for local, index in enumerate(test):
            if int(index) in seen:
                raise InternalReadoutError("candidate received duplicate outer prediction")
            seen.add(int(index))
            order = np.argsort(probabilities[local])[::-1]
            confidence = float(probabilities[local, order[0]])
            margin = float(confidence - probabilities[local, order[1]]) if len(order) > 1 else confidence
            predictions.append(
                {
                    "candidate_id": str(candidate_ids[index]),
                    "video_id": str(groups[index]),
                    "target": target,
                    "progress": float(progress),
                    "family": family,
                    "outer_fold": outer_fold,
                    "true_label": str(labels[index]),
                    "predicted_label": str(predicted[local]),
                    "confidence": confidence,
                    "probability_margin": margin,
                    "abstain": bool(confidence < SELECTIVE_THRESHOLD),
                    "correct": bool(str(predicted[local]) == str(labels[index])),
                    "selected_parameter": selected,
                    "probabilities": [float(value) for value in probabilities[local]],
                }
            )
    if seen != set(range(n)):
        raise InternalReadoutError("outer predictions are incomplete")
    predictions.sort(key=lambda row: row["candidate_id"])
    return predictions, selections


def _load_progress_views(
    feature_completion_path: Path, progress: float
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray], dict[str, Any]]:
    _, rows = _feature_rows(feature_completion_path)
    selected = sorted(
        [row for row in rows if abs(float(row["progress"]) - float(progress)) <= 1e-8],
        key=lambda row: (str(row["video_id"]), int(row["base_seed"])),
    )
    if len(selected) != EXPECTED_CANDIDATES:
        raise InternalReadoutError(f"progress {progress} feature cardinality != 816")
    raw: dict[str, list[np.ndarray]] = {view: [] for view in set(VIEW_FOR_FAMILY.values())}
    expected_keys = {
        "conditioning", "external_preview", "latent", "velocity", "tweedie", "pooled",
        "token_statistics", "attention_tokens", "selected_cross_attention",
    }
    for row in selected:
        path = Path(str(row["probe_views_path"]))
        if sha256_file(path) != row.get("probe_views_sha256"):
            raise InternalReadoutError(f"probe-view hash mismatch: {path}")
        try:
            with np.load(path, allow_pickle=False) as archive:
                if set(archive.files) != expected_keys:
                    raise InternalReadoutError(f"probe-view key mismatch: {path}")
                for view in raw:
                    value = np.asarray(archive[view], dtype=np.float32)
                    if not np.all(np.isfinite(value)):
                        raise InternalReadoutError(f"nonfinite probe view: {path}:{view}")
                    raw[view].append(value)
        except InternalReadoutError:
            raise
        except Exception as exc:
            raise InternalReadoutError(f"cannot load probe views {path}: {exc}") from exc
    matrices: dict[str, np.ndarray] = {}
    projection: dict[str, Any] = {}
    for view, values in raw.items():
        try:
            stacked = np.stack(values)
        except ValueError as exc:
            raise InternalReadoutError(f"inconsistent probe view shapes for {view}") from exc
        if view == "attention_tokens":
            matrices[view], projection[view] = project_attention(stacked)
        else:
            matrices[view], projection[view] = project_vectors(
                stacked.reshape(stacked.shape[0], -1), view
            )
    return selected, matrices, projection


def fit_progress_shard(
    feature_completion_path: Path,
    target_completion_path: Path,
    protocol_path: Path,
    implementation_path: Path,
    out_dir: Path,
    *,
    progress: float,
    device: str,
) -> Path:
    if not any(abs(float(progress) - value) <= 1e-8 for value in S_POINTS):
        raise InternalReadoutError(f"unregistered readout progress {progress}")
    _, _, implementation_sha256 = validate_protocols(protocol_path, implementation_path)
    target_completion, target_rows = validate_targets(target_completion_path)
    feature_rows, matrices, projection = _load_progress_views(feature_completion_path, progress)
    feature_ids = [
        candidate_id(str(row["video_id"]), int(row["base_seed"]), float(progress))
        for row in feature_rows
    ]
    target_by_id = {str(row["candidate_id"]): row for row in target_rows}
    if any(value not in target_by_id for value in feature_ids):
        raise InternalReadoutError("feature candidate absent from target manifest")
    all_videos = ordered_videos(str(row["video_id"]) for row in feature_rows)
    if len(all_videos) != 48:
        raise InternalReadoutError("readout progress does not contain 48 videos")
    predictions: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    target_coverage: dict[str, Any] = {}
    for target in TARGETS:
        label_field = "fork_majority_label" if target == "fork_majority" else "ode_final_label"
        observed_indices = [
            index for index, cid in enumerate(feature_ids) if target_by_id[cid].get(label_field) is not None
        ]
        labels = [str(target_by_id[feature_ids[index]][label_field]) for index in observed_indices]
        groups = [str(feature_rows[index]["video_id"]) for index in observed_indices]
        ids = [feature_ids[index] for index in observed_indices]
        target_coverage[target] = {
            "observed_candidates": len(observed_indices),
            "total_candidates": EXPECTED_CANDIDATES,
            "coverage": len(observed_indices) / EXPECTED_CANDIDATES,
            "observed_classes": sorted(set(labels)),
        }
        if len(observed_indices) < OUTER_FOLDS or len(set(groups)) < OUTER_FOLDS:
            raise InternalReadoutError(f"insufficient grouped target coverage for {target}")
        for family in FAMILY_IDS:
            view = VIEW_FOR_FAMILY.get(family)
            X = None if view is None else matrices[view][observed_indices]
            family_predictions, family_selections = nested_outer_predictions(
                family,
                X,
                labels,
                groups,
                ids,
                all_videos,
                target_completion["class_names"],
                target=target,
                progress=float(progress),
                device=device,
            )
            seed_by_id = {feature_ids[index]: int(feature_rows[index]["base_seed"]) for index in observed_indices}
            for row in family_predictions:
                row["base_seed"] = seed_by_id[row["candidate_id"]]
                row["class_names"] = list(target_completion["class_names"])
            predictions.extend(family_predictions)
            selections.extend(family_selections)
    expected_predictions = sum(
        int(target_coverage[target]["observed_candidates"]) * len(FAMILY_IDS) for target in TARGETS
    )
    if len(predictions) != expected_predictions:
        raise InternalReadoutError("readout shard prediction cardinality mismatch")
    out_dir = Path(out_dir)
    try:
        out_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise InternalReadoutError(f"refusing to overwrite readout shard {out_dir}") from exc
    predictions.sort(key=lambda row: (row["target"], row["family"], row["candidate_id"]))
    predictions_path = out_dir / "OUTER_FOLD_PREDICTIONS.jsonl"
    atomic_jsonl_create(predictions_path, predictions)
    selections_path = out_dir / "INNER_SELECTIONS.json"
    atomic_json_create(selections_path, {"selections": selections})
    report = {
        "schema": SHARD_SCHEMA,
        "status": "COMPLETE",
        "progress": float(progress),
        "targets": list(TARGETS),
        "families": list(FAMILY_IDS),
        "target_coverage": target_coverage,
        "prediction_count": len(predictions),
        "outer_folds": OUTER_FOLDS,
        "inner_folds": INNER_FOLDS,
        "group": "video_id",
        "projection": projection,
        "device": str(device),
        "protocol_sha256": PINNED_PROTOCOL_SHA256,
        "implementation_sha256": implementation_sha256,
        "feature_completion": str(Path(feature_completion_path).resolve()),
        "feature_completion_sha256": sha256_file(feature_completion_path),
        "target_completion": str(Path(target_completion_path).resolve()),
        "target_completion_sha256": sha256_file(target_completion_path),
        "predictions_file": predictions_path.name,
        "predictions_sha256": sha256_file(predictions_path),
        "predictions_bytes": predictions_path.stat().st_size,
        "selections_file": selections_path.name,
        "selections_sha256": sha256_file(selections_path),
        "selections_bytes": selections_path.stat().st_size,
    }
    atomic_json_create(out_dir / "READOUT_SHARD_COMPLETION.json", report)
    return out_dir


def validate_readout_shard(completion_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    completion = _load_json(completion_path)
    if (
        completion.get("schema") != SHARD_SCHEMA
        or completion.get("status") != "COMPLETE"
        or completion.get("protocol_sha256") != PINNED_PROTOCOL_SHA256
        or completion.get("families") != list(FAMILY_IDS)
        or completion.get("targets") != list(TARGETS)
    ):
        raise InternalReadoutError("invalid readout shard completion")
    root = Path(completion_path).parent
    for prefix in ("predictions", "selections"):
        path = root / str(completion.get(f"{prefix}_file", ""))
        if (
            sha256_file(path) != completion.get(f"{prefix}_sha256")
            or path.stat().st_size != int(completion.get(f"{prefix}_bytes", -1))
        ):
            raise InternalReadoutError(f"readout shard {prefix} hash/size mismatch")
    rows = _load_jsonl(root / str(completion["predictions_file"]))
    if len(rows) != int(completion.get("prediction_count", -1)):
        raise InternalReadoutError("partial readout shard predictions")
    identities = {(row["target"], row["family"], row["candidate_id"]) for row in rows}
    if len(identities) != len(rows):
        raise InternalReadoutError("duplicate readout shard prediction identity")
    return completion, rows


def _quantile(values: np.ndarray, q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), q, method="linear"))


def _metric_bootstrap(
    family_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    draws: int,
    seed: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, np.ndarray]]:
    if set(family_rows) != set(FAMILY_IDS):
        raise InternalReadoutError("bootstrap family set is incomplete")
    reference = sorted(family_rows["majority_null"], key=lambda row: row["candidate_id"])
    if not reference:
        raise InternalReadoutError("bootstrap received no candidate predictions")
    candidate_ids = [str(row["candidate_id"]) for row in reference]
    y_true = [str(row["true_label"]) for row in reference]
    videos = ordered_videos(str(row["video_id"]) for row in reference)
    video_index = {video: index for index, video in enumerate(videos)}
    classes = sorted(set(y_true))
    class_index = {label: index for index, label in enumerate(classes)}
    rng = np.random.default_rng(int(seed))
    weights = rng.multinomial(len(videos), np.full(len(videos), 1.0 / len(videos)), size=draws)
    summaries: dict[str, dict[str, Any]] = {}
    distributions: dict[str, np.ndarray] = {}
    for family in FAMILY_IDS:
        rows = sorted(family_rows[family], key=lambda row: row["candidate_id"])
        if (
            [str(row["candidate_id"]) for row in rows] != candidate_ids
            or [str(row["true_label"]) for row in rows] != y_true
            or [str(row["video_id"]) for row in rows]
            != [str(row["video_id"]) for row in reference]
        ):
            raise InternalReadoutError("paired bootstrap prediction rows do not align")
        pred = [str(row["predicted_label"]) for row in rows]
        confidence = np.asarray([float(row["confidence"]) for row in rows], dtype=np.float64)
        correct = np.asarray([a == b for a, b in zip(y_true, pred)], dtype=np.float64)
        abstain = np.asarray([bool(row["abstain"]) for row in rows], dtype=bool)
        totals = np.zeros((len(videos), len(classes)), dtype=np.int64)
        hits = np.zeros_like(totals)
        bin_count = np.zeros((len(videos), 10), dtype=np.int64)
        bin_correct = np.zeros((len(videos), 10), dtype=np.float64)
        bin_confidence = np.zeros((len(videos), 10), dtype=np.float64)
        selected = np.zeros(len(videos), dtype=np.int64)
        video_total = np.zeros(len(videos), dtype=np.int64)
        for index, row in enumerate(rows):
            vi = video_index[str(row["video_id"])]
            ci = class_index[y_true[index]]
            totals[vi, ci] += 1
            hits[vi, ci] += int(correct[index])
            bi = min(9, int(math.floor(confidence[index] * 10.0)))
            bin_count[vi, bi] += 1
            bin_correct[vi, bi] += correct[index]
            bin_confidence[vi, bi] += confidence[index]
            selected[vi] += int(not abstain[index])
            video_total[vi] += 1
        bootstrap_total = weights @ totals
        bootstrap_hits = weights @ hits
        with np.errstate(invalid="ignore", divide="ignore"):
            recalls = bootstrap_hits / bootstrap_total
        recalls[bootstrap_total == 0] = np.nan
        ba_draws = np.nanmean(recalls, axis=1)
        bc = weights @ bin_count
        bh = weights @ bin_correct
        bf = weights @ bin_confidence
        n_draw = bc.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            bin_accuracy = bh / bc
            bin_mean_confidence = bf / bc
        bin_accuracy[bc == 0] = 0.0
        bin_mean_confidence[bc == 0] = 0.0
        ece_draws = (
            (bc / n_draw[:, None]) * np.abs(bin_accuracy - bin_mean_confidence)
        ).sum(axis=1)
        coverage_draws = (weights @ selected) / (weights @ video_total)
        point_ba, per_class = balanced_accuracy(y_true, pred)
        point_ece = expected_calibration_error(y_true, pred, confidence)
        point_coverage = float(np.mean(~abstain))
        summaries[family] = {
            "candidate_count": len(rows),
            "video_count": len(videos),
            "class_count": len(classes),
            "balanced_accuracy": point_ba,
            "balanced_accuracy_ci": [_quantile(ba_draws, 0.025), _quantile(ba_draws, 0.975)],
            "per_class_recall": per_class,
            "expected_calibration_error": point_ece,
            "expected_calibration_error_ci": [
                _quantile(ece_draws, 0.025),
                _quantile(ece_draws, 0.975),
            ],
            "coverage": point_coverage,
            "coverage_ci": [
                _quantile(coverage_draws, 0.025),
                _quantile(coverage_draws, 0.975),
            ],
            "abstention": 1.0 - point_coverage,
        }
        distributions[family] = ba_draws
    majority_ba = float(summaries["majority_null"]["balanced_accuracy"])
    for family in FAMILY_IDS:
        summaries[family]["margin_over_majority"] = (
            float(summaries[family]["balanced_accuracy"]) - majority_ba
        )
        paired: dict[str, Any] = {}
        for baseline in ("majority_null", "conditioning_only", "external_preview"):
            difference = distributions[family] - distributions[baseline]
            paired[f"minus_{baseline}"] = {
                "point": float(summaries[family]["balanced_accuracy"])
                - float(summaries[baseline]["balanced_accuracy"]),
                "ci": [_quantile(difference, 0.025), _quantile(difference, 0.975)],
            }
        summaries[family]["paired_balanced_accuracy_differences"] = paired
    return summaries, distributions


def _earliest_progress(rows: Sequence[Mapping[str, Any]], predicate: Any) -> float | None:
    for progress in S_POINTS:
        candidates = [row for row in rows if abs(float(row["progress"]) - progress) <= 1e-8]
        if len(candidates) == 1 and predicate(candidates[0]):
            return float(progress)
    return None


def _readout_decisions(metric_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for target in TARGETS:
        target_rows = [row for row in metric_rows if row["target"] == target]
        by_family: dict[str, Any] = {}
        for family in FAMILY_IDS:
            rows = [row for row in target_rows if row["family"] == family]

            def information(row: Mapping[str, Any]) -> bool:
                paired = row["paired_balanced_accuracy_differences"]
                return (
                    float(paired["minus_majority_null"]["ci"][0]) > 0.0
                    and float(paired["minus_conditioning_only"]["ci"][0]) > 0.0
                )

            def action(row: Mapping[str, Any]) -> bool:
                return (
                    float(row["balanced_accuracy"]) >= 0.7
                    and float(row["expected_calibration_error"]) <= 0.1
                    and float(row["coverage"]) >= 0.8
                )

            by_family[family] = {
                "information_readout_progress": _earliest_progress(rows, information),
                "action_readout_progress": _earliest_progress(rows, action),
            }
        supported_internal = {
            family: by_family[family]["information_readout_progress"]
            for family in INTERNAL_FAMILIES
            if by_family[family]["information_readout_progress"] is not None
        }
        earliest_internal = min(supported_internal.values()) if supported_internal else None
        earliest_families = sorted(
            family for family, progress in supported_internal.items() if progress == earliest_internal
        )
        external = by_family["external_preview"]["information_readout_progress"]
        conditioning_dominates = not supported_internal
        conclusion = "unresolved due to uncertainty"
        status = "UNRESOLVED"
        external_advantage_required = False
        if earliest_internal is not None:
            robust_over_external = []
            for family in earliest_families:
                row = next(
                    item for item in target_rows
                    if item["family"] == family
                    and abs(float(item["progress"]) - float(earliest_internal)) <= 1e-8
                )
                robust_over_external.append(
                    float(
                        row["paired_balanced_accuracy_differences"][
                            "minus_external_preview"
                        ]["ci"][0]
                    )
                    > 0.0
                )
            external_advantage_required = any(robust_over_external)
            if external is not None and earliest_internal < external and external_advantage_required:
                conclusion = "internal readout earlier than external preview"
                status = "SUPPORTED_EXPLORATORILY"
            elif external is not None and earliest_internal >= external:
                conclusion = "internal readout not earlier than external preview"
                status = "NOT_SUPPORTED"
            elif external is None and external_advantage_required:
                conclusion = "internal readout earlier than external preview"
                status = "SUPPORTED_EXPLORATORILY"
        elif external is not None:
            conclusion = "internal readout not earlier than external preview"
            status = "NOT_SUPPORTED"
        # If external preview already clears conditioning while no internal
        # family does, the directional question is resolved: internal readout
        # is not earlier.  Reserve the conditioning-dominates label for cases
        # where neither internal state nor external preview clears it.
        if conditioning_dominates and external is None:
            conclusion = "conditioning explains most predictive information"
            status = "UNRESOLVED"
        result[target] = {
            "families": by_family,
            "earliest_supported_internal_progress": earliest_internal,
            "earliest_supported_internal_families": earliest_families,
            "external_preview_information_progress": external,
            "earliest_internal_robustly_exceeds_external_at_same_progress": (
                external_advantage_required
            ),
            "conclusion": conclusion,
            "scientific_status": status,
        }
    return result


def _atomic_gzip_jsonl_create(path: Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_tmp = tempfile.mkstemp(prefix=f".{path.name}.tmp.", dir=path.parent)
    os.close(fd)
    tmp = Path(raw_tmp)
    try:
        with tmp.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
                for row in rows:
                    zipped.write(canonical_json_bytes(dict(row), indent=None))
            raw.flush()
            os.fsync(raw.fileno())
        os.link(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


def _report_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Class internal readout",
        "",
        "Exploratory multi-seed continuity analysis. This is not event-centered Axis Specification v2 confirmation.",
        "",
        "All scores below use outer-fold predictions from nested video-grouped cross-validation.",
        "",
        "| target | family | s | balanced accuracy (95% video CI) | ECE | coverage |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in report["metrics"]:
        ci = row["balanced_accuracy_ci"]
        lines.append(
            f"| {row['target']} | {row['family']} | {float(row['progress']):.2f} | "
            f"{float(row['balanced_accuracy']):.3f} [{float(ci[0]):.3f}, {float(ci[1]):.3f}] | "
            f"{float(row['expected_calibration_error']):.3f} | {float(row['coverage']):.3f} |"
        )
    lines.extend(["", "## Readout decisions", ""])
    for target, decision in report["decisions"].items():
        lines.extend(
            [
                f"- `{target}`: **{decision['scientific_status']}** — {decision['conclusion']}.",
                f"  Earliest internal: {decision['earliest_supported_internal_progress']}; "
                f"external preview: {decision['external_preview_information_progress']}.",
            ]
        )
    lines.extend(
        [
            "",
            "Missing fork-majority targets are excluded under the frozen >=2-confident, unique-majority rule; they are never imputed.",
            "",
        ]
    )
    return "\n".join(lines)


def merge_readout_shards(
    completion_paths: Sequence[Path],
    out_dir: Path,
    *,
    bootstrap_draws: int = BOOTSTRAP_DRAWS,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> Path:
    validated = [validate_readout_shard(path) for path in completion_paths]
    progresses = [float(item[0]["progress"]) for item in validated]
    if len(validated) != len(S_POINTS) or sorted(progresses) != list(S_POINTS):
        raise InternalReadoutError(f"partial/duplicate readout progress shards: {sorted(progresses)}")
    invariants = (
        "protocol_sha256",
        "implementation_sha256",
        "feature_completion_sha256",
        "target_completion_sha256",
        "families",
        "targets",
        "outer_folds",
        "inner_folds",
        "group",
    )
    first = validated[0][0]
    for completion, _ in validated[1:]:
        for key in invariants:
            if completion.get(key) != first.get(key):
                raise InternalReadoutError(f"readout shard provenance mismatch: {key}")
    predictions = [row for _, rows in validated for row in rows]
    identities = {(row["target"], row["progress"], row["family"], row["candidate_id"]) for row in predictions}
    if len(identities) != len(predictions):
        raise InternalReadoutError("duplicate merged outer prediction")
    metric_rows: list[dict[str, Any]] = []
    for target in TARGETS:
        for progress_index, progress in enumerate(S_POINTS):
            family_rows = {
                family: [
                    row for row in predictions
                    if row["target"] == target
                    and row["family"] == family
                    and abs(float(row["progress"]) - progress) <= 1e-8
                ]
                for family in FAMILY_IDS
            }
            summaries, _ = _metric_bootstrap(
                family_rows,
                draws=int(bootstrap_draws),
                seed=int(bootstrap_seed + progress_index + (1000 if target == "ode_final" else 0)),
            )
            coverage = next(
                item[0]["target_coverage"][target]
                for item in validated
                if abs(float(item[0]["progress"]) - progress) <= 1e-8
            )
            for family in FAMILY_IDS:
                metric_rows.append(
                    {
                        "target": target,
                        "progress": float(progress),
                        "family": family,
                        "target_coverage": coverage,
                        **summaries[family],
                    }
                )
    decisions = _readout_decisions(metric_rows)
    predictions.sort(
        key=lambda row: (row["target"], float(row["progress"]), row["family"], row["candidate_id"])
    )
    out_dir = Path(out_dir)
    try:
        out_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise InternalReadoutError(f"refusing to overwrite readout merge {out_dir}") from exc
    predictions_path = out_dir / "CLASS_INTERNAL_READOUT_OUTER_PREDICTIONS.jsonl.gz"
    _atomic_gzip_jsonl_create(predictions_path, predictions)
    report = {
        "schema": MERGE_SCHEMA,
        "status": "COMPLETE",
        "scope": "exploratory Class continuity; not event-centered v2 confirmation",
        "targets_separate": True,
        "prediction_count": len(predictions),
        "progress_points": list(S_POINTS),
        "families": list(FAMILY_IDS),
        "outer_folds": OUTER_FOLDS,
        "inner_folds": INNER_FOLDS,
        "bootstrap": {
            "unit": "video_id",
            "draws": int(bootstrap_draws),
            "seed": int(bootstrap_seed),
            "interval": [0.025, 0.975],
        },
        "metrics": metric_rows,
        "decisions": decisions,
        "protocol_sha256": first["protocol_sha256"],
        "implementation_sha256": first["implementation_sha256"],
        "feature_completion_sha256": first["feature_completion_sha256"],
        "target_completion_sha256": first["target_completion_sha256"],
        "predictions_file": predictions_path.name,
        "predictions_sha256": sha256_file(predictions_path),
        "predictions_bytes": predictions_path.stat().st_size,
        "input_shards": [
            {
                "progress": float(completion["progress"]),
                "completion": str(Path(path).resolve()),
                "completion_sha256": sha256_file(path),
                "prediction_count": int(completion["prediction_count"]),
            }
            for path, (completion, _) in sorted(
                zip(completion_paths, validated), key=lambda pair: float(pair[1][0]["progress"])
            )
        ],
    }
    report_path = out_dir / "CLASS_INTERNAL_READOUT_REPORT.json"
    atomic_json_create(report_path, report)
    markdown = _report_markdown(report)
    md_path = out_dir / "CLASS_INTERNAL_READOUT_REPORT.md"
    fd, raw_tmp = tempfile.mkstemp(prefix=f".{md_path.name}.tmp.", dir=out_dir)
    os.close(fd)
    tmp = Path(raw_tmp)
    try:
        tmp.write_text(markdown, encoding="utf-8")
        os.link(tmp, md_path)
    finally:
        tmp.unlink(missing_ok=True)
    completion = {
        "schema": MERGE_SCHEMA,
        "status": "COMPLETE",
        "report": report_path.name,
        "report_sha256": sha256_file(report_path),
        "report_bytes": report_path.stat().st_size,
        "markdown": md_path.name,
        "markdown_sha256": sha256_file(md_path),
        "predictions": predictions_path.name,
        "predictions_sha256": sha256_file(predictions_path),
        "prediction_count": len(predictions),
        "protocol_sha256": first["protocol_sha256"],
        "implementation_sha256": first["implementation_sha256"],
    }
    atomic_json_create(out_dir / "CLASS_INTERNAL_READOUT_COMPLETION.json", completion)
    return out_dir


__all__ = [
    "ATTENTION_WIDTH",
    "FAMILY_IDS",
    "FAMILY_SPECS",
    "InternalReadoutError",
    "TARGETS",
    "balanced_accuracy",
    "construct_targets",
    "expected_calibration_error",
    "fit_progress_shard",
    "fork_majority_target",
    "grouped_folds",
    "merge_readout_shards",
    "nested_outer_predictions",
    "prepare_targets",
    "project_attention",
    "project_vectors",
    "validate_no_group_leakage",
    "validate_readout_shard",
    "validate_targets",
]
