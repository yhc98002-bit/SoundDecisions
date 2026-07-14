#!/usr/bin/env python
"""Frozen Arc-4 B-1 V2 completeness gate and nested probe."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import multiprocessing
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from foley_cw.arc4_b1 import (  # noqa: E402
    accuracy_and_video_ci,
    classification_metrics,
    inner_clip_split,
    mlp_predict,
    mlp_predict_fixed_epochs,
    ridge_predict,
    select_spec,
)
from foley_cw.arc4_gpu import sha256_file  # noqa: E402

S_GRID = (0.05, 0.15, 0.25, 0.35, 0.45, 0.60, 0.75, 0.90)
N_INDEPENDENT = 16
EXPECTED_CLIPS = 200
EXPECTED_LABELS = EXPECTED_CLIPS * N_INDEPENDENT
EXPECTED_BUNDLES = EXPECTED_LABELS * len(S_GRID)
PROTOCOL_SHA256 = "1386287a34802bd8bdca7b1f390e22631f792dd3c650eb08341f8bc4bd03b56d"
AMENDMENT_SHA256 = "92f2e8382e39c5395d11228d70512760558946b852d5c5fd180c83fd6e40b808"
JOIN_AMENDMENT_SHA256 = "4040bc321de45ce5f2aa6fbfc75d38e780ee31be0c56248fcd6a78c8848bbabd"
JOIN_AMENDMENT_2_SHA256 = "08200aaa356b2c177aa6bdc10c8d4e641b4127e28dc1d74b78acacd4c1d07654"
JOIN_AMENDMENT_3_SHA256 = "47c2a4c6f4fdde160c093709b53a9e55cf589fed7dc3cd3d233a220d1ca2a968"
JOIN_AMENDMENT_4_SHA256 = "2283ae829b34a18970a8a7c8b46963ba20db72a4bd748b31dd23564651299b64"
MANIFEST_SHA256 = "64a7a3d1a194edffc69506bf7baddc85e03a3ab102298f61782d3be0fe4a595b"
COARSE_MAP_SHA256 = "55b5a1d4116caa4503a6b4b17192425da487a9c4385a287e343d850795be4fe7"
COLLECTION_PROTOCOL_SHA256 = "b85eeece6f18ff7ce3ab254411d06f97cf2446d393f74eb81ad34048131cc03f"
COLLECTION_CONFIG_SHA256 = "c16633ea96e66c9d502f6b9827a8c37f09197e82733fcd0c27aab7506505cb41"
COLLECTION_SCHEMA = "arc4_b1_collection_completion_v1"


def _expected(manifest: dict):
    clips = [str(clip) for clip in manifest["clips"]["single_event"]]
    if len(clips) != EXPECTED_CLIPS or len(set(clips)) != EXPECTED_CLIPS:
        raise ValueError(f"expected {EXPECTED_CLIPS} unique single-event clips")
    rows = [
        (clip, j, f"{clip}__p1cfg1_ind{j}")
        for clip in clips
        for j in range(N_INDEPENDENT)
    ]
    return clips, rows


def _single_event_splits(manifest: dict, clips: list[str]) -> tuple[set[str], set[str]]:
    single = set(map(str, clips))
    train = set(map(str, manifest["split_60_40_by_clip"]["probe_train"])) & single
    evaluation = set(map(str, manifest["split_60_40_by_clip"]["eval"])) & single
    if (
        train & evaluation
        or train | evaluation != single
        or len(train) != 126
        or len(evaluation) != 74
    ):
        raise ValueError("frozen single-event train/eval split failed integrity checks")
    return train, evaluation


def _load_labels(path: Path) -> tuple[dict[str, str], int]:
    labels = {}
    duplicates = 0
    with path.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            extra = row.get("extra") or {}
            if (
                row.get("axis_id") != "class"
                or extra.get("role") != "p1cfg1_independent"
            ):
                continue
            label = (row.get("target") or {}).get("label")
            gid = row.get("gen_id")
            if gid is None or label is None:
                continue
            gid = str(gid)
            if gid in labels:
                duplicates += 1
            labels[gid] = str(label)
    return labels, duplicates


def _write_once_or_same(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text() != text:
            raise FileExistsError(f"refusing to replace existing B1 output {path}")
        return
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.link(tmp, path)
    tmp.unlink()


def _read_hash_sidecar(path: Path) -> str:
    digest = sha256_file(path)
    sidecar = path.with_suffix(".sha256")
    fields = sidecar.read_text().strip().split()
    if len(fields) != 2 or fields[0] != digest or fields[1] != path.name:
        raise ValueError(f"invalid SHA256 sidecar for {path}")
    return digest


def _validate_collection_completion(path: Path) -> tuple[dict, str]:
    digest = _read_hash_sidecar(path)
    row = json.loads(path.read_text())
    expected_config = {
        "cfg": 1.0,
        "schedule": "sqrt_down",
        "seed": 0,
        "variant": "small_16k",
        "num_steps": 20,
        "duration_sec": 8.0,
        "n_independent": 16,
        "s_grid": list(S_GRID),
    }
    expected_counts = {
        "journals": EXPECTED_CLIPS,
        "per_token_bundles": EXPECTED_BUNDLES,
        "pooled_bundles": EXPECTED_BUNDLES,
        "labels": EXPECTED_LABELS,
    }
    if row.get("schema") != COLLECTION_SCHEMA or row.get("status") != "COMPLETE":
        raise ValueError("B1 collection completion manifest is not COMPLETE")
    if row.get("collection_config") != expected_config:
        raise ValueError("B1 collection completion config differs from V2")
    if row.get("counts") != expected_counts:
        raise ValueError("B1 collection completion counts differ from V2")

    launch_path = Path(row["launch_manifest_path"])
    if not launch_path.is_absolute():
        launch_path = Path.cwd() / launch_path
    if sha256_file(launch_path) != row.get("launch_manifest_sha256"):
        raise ValueError("B1 launch-manifest hash mismatch")
    launch = json.loads(launch_path.read_text())
    if (
        launch.get("node") != "an12"
        or launch.get("seed") != 0
        or launch.get("protocol_sha256") != COLLECTION_PROTOCOL_SHA256
        or launch.get("config_sha256") != COLLECTION_CONFIG_SHA256
        or launch.get("weights") != {"source": "hf", "offline": True}
        or launch.get("placement")
        != {
            "physical_gpu_ids": [4, 5, 6, 7],
            "cuda_visible_devices": "4,5,6,7",
            "tp_width": 1,
            "replica_count": 4,
            "rationale": "small_16k fits TP1; four independent replicas maximize throughput",
        }
    ):
        raise ValueError("B1 launch ledger placement/config contract failed")
    command = str(launch.get("command", ""))
    required_command_fragments = (
        "--cfg 1.0",
        "--schedule sqrt_down",
        "--variant small_16k",
        "--duration 8",
        "--num-steps 20",
        "--seed 0",
        "--n-independent 16",
    )
    if not all(fragment in command for fragment in required_command_fragments):
        raise ValueError("B1 launch command does not pin the V2 collection config")

    aggregate_path = Path(row["aggregate_manifest_path"])
    if not aggregate_path.is_absolute():
        aggregate_path = Path.cwd() / aggregate_path
    if sha256_file(aggregate_path) != row.get("aggregate_manifest_sha256"):
        raise ValueError("B1 aggregate-manifest hash mismatch")
    aggregate = json.loads(aggregate_path.read_text())
    if (
        aggregate.get("tag") != "p1cfg1"
        or aggregate.get("subdir") != "arc3/pertoken"
        or aggregate.get("n_done") != EXPECTED_CLIPS
        or aggregate.get("n_missing") != 0
        or aggregate.get("missing") != []
    ):
        raise ValueError("B1 aggregate manifest is incomplete")

    session_path = Path(row["session_log_path"])
    exit_path = Path(row["exit_code_path"])
    if not session_path.is_absolute():
        session_path = Path.cwd() / session_path
    if not exit_path.is_absolute():
        exit_path = Path.cwd() / exit_path
    if sha256_file(session_path) != row.get("session_log_sha256"):
        raise ValueError("B1 collection session-log hash mismatch")
    if sha256_file(exit_path) != row.get("exit_code_sha256"):
        raise ValueError("B1 collection exit-code hash mismatch")
    if exit_path.read_text().strip() != "0":
        raise ValueError("B1 collection did not exit zero")

    worker_logs = row.get("worker_logs") or []
    if len(worker_logs) != 4:
        raise ValueError("B1 completion manifest must hash four worker logs")
    for item in worker_logs:
        log_path = Path(item["path"])
        if not log_path.is_absolute():
            log_path = Path.cwd() / log_path
        if sha256_file(log_path) != item.get("sha256"):
            raise ValueError(f"B1 worker-log hash mismatch: {log_path}")
    return row, digest


def _hash_array(hasher, key: str, array: np.ndarray) -> None:
    if hasher is None:
        return
    hasher.update(key.encode("ascii"))
    hasher.update(array.dtype.str.encode("ascii"))
    hasher.update(json.dumps(list(array.shape)).encode("ascii"))
    hasher.update(np.ascontiguousarray(array).tobytes())


def _inspect_pertoken_v2(path: Path, hasher=None) -> np.ndarray:
    with np.load(path, allow_pickle=False) as z:
        expected = {
            "token_mean": ((12, 448), np.dtype("float16")),
            "token_mean_max": ((12, 896), np.dtype("float16")),
            "tokens_sub": ((12, 64, 448), np.dtype("float16")),
            "xattn_clip": ((4, 64), np.dtype("float16")),
            "xattn_frac": ((4,), np.dtype("float32")),
        }
        if set(z.files) != set(expected):
            raise ValueError("wrong per-token array set")
        arrays = {}
        for key, (shape, dtype) in expected.items():
            array = z[key]
            if array.shape != shape or array.dtype != dtype or not np.isfinite(array).all():
                raise ValueError(f"invalid per-token array {key}")
            arrays[key] = array
            _hash_array(hasher, key, array)
        return np.asarray(arrays["token_mean"], dtype=np.float32)


def _valid_pertoken_v2(path: Path) -> bool:
    try:
        _inspect_pertoken_v2(path)
        return True
    except (OSError, ValueError, KeyError):
        return False


def _inspect_pooled_v2(path: Path, hasher=None) -> np.ndarray:
    with np.load(path, allow_pickle=False) as z:
        if set(z.files) != {"pooled"}:
            raise ValueError("wrong pooled array set")
        pooled = z["pooled"]
        if (
            pooled.shape != (12, 448)
            or pooled.dtype != np.float16
            or not np.isfinite(pooled).all()
        ):
            raise ValueError("invalid pooled array")
        _hash_array(hasher, "pooled", pooled)
        return np.asarray(pooled, dtype=np.float32)


def _valid_pooled_v2(path: Path) -> bool:
    try:
        _inspect_pooled_v2(path)
        return True
    except (OSError, ValueError, KeyError):
        return False


def _validate_manifest_hash(path: Path) -> str:
    digest = sha256_file(path)
    if digest != MANIFEST_SHA256:
        raise ValueError("B1 frozen population manifest hash mismatch")
    return digest


def _join_relative_l2(token_mean: np.ndarray, pooled: np.ndarray) -> float:
    if token_mean.shape != pooled.shape:
        raise ValueError("retap/cache join shape mismatch")
    denominator = max(float(np.linalg.norm(pooled)), 1e-12)
    return float(np.linalg.norm(token_mean - pooled)) / denominator


def _join_failure_record(
    *,
    name: str,
    token_mean: np.ndarray,
    pooled: np.ndarray,
    checked_before_failure: int,
) -> dict:
    token64 = np.asarray(token_mean, dtype=np.float64)
    pooled64 = np.asarray(pooled, dtype=np.float64)
    delta = token64 - pooled64
    denominator = max(float(np.linalg.norm(pooled64)), 1e-12)
    cosine_denominator = max(
        float(np.linalg.norm(token64)) * float(np.linalg.norm(pooled64)), 1e-12
    )
    return {
        "schema": "arc4_b1_incomplete_v2",
        "status": "B1_INCOMPLETE",
        "reason": "retap_cache_join_failed",
        "scientific_token": None,
        "evaluation_started": False,
        "probe_metrics_computed": False,
        "prediction_ledgers_produced": False,
        "labels_loaded_for_set_and_universe_validation": True,
        "label_summary_or_outcome_metric_computed": False,
        "protocol_sha256": PROTOCOL_SHA256,
        "protocol_amendment_sha256": AMENDMENT_SHA256,
        "join_amendment_sha256": JOIN_AMENDMENT_SHA256,
        "join_amendment_2_sha256": JOIN_AMENDMENT_2_SHA256,
        "join_amendment_3_sha256": JOIN_AMENDMENT_3_SHA256,
        "join_amendment_4_sha256": JOIN_AMENDMENT_4_SHA256,
        "binding_join_rule": {
            "method": "bundle_relative_l2",
            "threshold": 0.0002,
        },
        "first_failing_bundle_in_sorted_scan": name,
        "bundles_checked_before_failure": checked_before_failure,
        "observed": {
            "relative_l2": _join_relative_l2(token_mean, pooled),
            "relative_l2_float64_diagnostic": float(np.linalg.norm(delta))
            / denominator,
            "l2": float(np.linalg.norm(delta)),
            "pooled_l2": float(np.linalg.norm(pooled64)),
            "max_abs": float(np.max(np.abs(delta))),
            "cosine_float64": float(np.sum(token64 * pooled64))
            / cosine_denominator,
        },
        "resolution": (
            "Run a newly frozen recollection that persists a pre-quantization "
            "GPU token mean using the same operation and reduction order as the "
            "original pooled tap, then rerun the identity gate before fitting "
            "any probe."
        ),
    }


def run_gate(args, manifest: dict, clips: list[str], rows: list[tuple]) -> dict:
    if sha256_file(args.protocol) != PROTOCOL_SHA256:
        raise ValueError("B1 V2 protocol hash mismatch")
    if sha256_file(args.protocol_amendment) != AMENDMENT_SHA256:
        raise ValueError("B1 V2 amendment hash mismatch")
    if sha256_file(args.join_amendment) != JOIN_AMENDMENT_SHA256:
        raise ValueError("B1 V2 join amendment hash mismatch")
    if sha256_file(args.join_amendment_2) != JOIN_AMENDMENT_2_SHA256:
        raise ValueError("B1 V2 second join amendment hash mismatch")
    if sha256_file(args.join_amendment_3) != JOIN_AMENDMENT_3_SHA256:
        raise ValueError("B1 V2 binding join amendment hash mismatch")
    if sha256_file(args.join_amendment_4) != JOIN_AMENDMENT_4_SHA256:
        raise ValueError("B1 V2 final join amendment hash mismatch")
    manifest_sha = _validate_manifest_hash(args.manifest)
    if sha256_file(args.coarse_class_map) != COARSE_MAP_SHA256:
        raise ValueError("B1 frozen coarse-class map hash mismatch")
    coarse_map = json.loads(args.coarse_class_map.read_text())
    allowed_labels = set(map(str, coarse_map["coarse_classes"])) | {"abstain"}
    completion, completion_sha = _validate_collection_completion(
        args.collection_completion
    )

    single = set(clips)
    train, evaluation = _single_event_splits(manifest, clips)

    labels, duplicates = _load_labels(args.measurements)
    measurements_sha = sha256_file(args.measurements)
    expected_gids = {gid for _, _, gid in rows}
    if duplicates or set(labels) != expected_gids:
        missing = len(expected_gids - set(labels))
        extra = len(set(labels) - expected_gids)
        raise RuntimeError(
            f"B1 label completeness failed: labels={len(labels)} "
            f"duplicates={duplicates} missing={missing} extra={extra}"
        )
    unknown_labels = sorted(set(labels.values()) - allowed_labels)
    if unknown_labels:
        raise RuntimeError(f"B1 labels outside frozen class universe: {unknown_labels}")

    pertoken_dir = args.b1_root / "arc3" / "pertoken"
    expected_pertoken = {
        f"{gid}__s{s:.2f}.npz" for _, _, gid in rows for s in S_GRID
    }
    actual_pertoken = {path.name for path in pertoken_dir.glob("*.npz")}
    if actual_pertoken != expected_pertoken:
        raise RuntimeError(
            f"B1 pertoken file-set failed: actual={len(actual_pertoken)} "
            f"missing={len(expected_pertoken - actual_pertoken)} "
            f"extra={len(actual_pertoken - expected_pertoken)}"
        )

    pooled_dir = args.pooled_root / "features"
    actual_pooled = {
        path.name for path in pooled_dir.glob("*__p1cfg1_ind*__s*.npz")
    }
    if actual_pooled != expected_pertoken:
        raise RuntimeError(
            f"B1 pooled file-set failed: actual={len(actual_pooled)} "
            f"missing={len(expected_pertoken - actual_pooled)} "
            f"extra={len(actual_pooled - expected_pertoken)}"
        )

    join_max_relative_l2 = 0.0
    feature_hasher = hashlib.sha256()
    for index, name in enumerate(sorted(expected_pertoken), start=1):
        feature_hasher.update(name.encode("ascii"))
        try:
            token_mean = _inspect_pertoken_v2(
                pertoken_dir / name, feature_hasher
            )
        except (OSError, ValueError, KeyError) as exc:
            raise RuntimeError(f"invalid B1 pertoken bundle: {name}") from exc
        try:
            pooled_value = _inspect_pooled_v2(pooled_dir / name, feature_hasher)
        except (OSError, ValueError, KeyError) as exc:
            raise RuntimeError(f"invalid B1 pooled bundle: {name}") from exc
        relative_l2 = _join_relative_l2(token_mean, pooled_value)
        if relative_l2 > 2e-4:
            failure = _join_failure_record(
                name=name,
                token_mean=token_mean,
                pooled=pooled_value,
                checked_before_failure=index - 1,
            )
            failure.update(
                {
                    "manifest_sha256": manifest_sha,
                    "coarse_class_map_sha256": COARSE_MAP_SHA256,
                    "measurements_sha256": measurements_sha,
                    "collection_completion_sha256": completion_sha,
                    "implementation_sha256": {
                        "runner": sha256_file(Path(__file__)),
                        "library": sha256_file(Path("foley_cw/arc4_b1.py")),
                    },
                    "failing_inputs": {
                        "retap": {
                            "path": str(pertoken_dir / name),
                            "sha256": sha256_file(pertoken_dir / name),
                        },
                        "pooled": {
                            "path": str(pooled_dir / name),
                            "sha256": sha256_file(pooled_dir / name),
                        },
                    },
                }
            )
            _write_once_or_same(
                args.b1_root / "b1_incomplete_v2.json",
                json.dumps(failure, indent=2, sort_keys=True) + "\n",
            )
            raise RuntimeError(
                "B1 retap/cache trajectory join failed: "
                f"{name} (relative_l2={failure['observed']['relative_l2']:.17g} "
                "> 0.0002); wrote B1_INCOMPLETE, no evaluation was run"
            )
        join_max_relative_l2 = max(join_max_relative_l2, relative_l2)
        if index % 1000 == 0:
            print(
                f"[b1 v2 gate] validated {index}/{EXPECTED_BUNDLES} bundles",
                flush=True,
            )

    journals = list((args.b1_root / "journal").glob("p1cfg1_pertoken__*.json"))
    if len(journals) != EXPECTED_CLIPS:
        raise RuntimeError(f"B1 journal gate expected 200, got {len(journals)}")
    journal_clips = set()
    for path in journals:
        row = json.loads(path.read_text())
        clip = str(row.get("clip"))
        if (
            row.get("tag") != "p1cfg1"
            or not math.isclose(float(row.get("cfg")), 1.0)
            or row.get("bundle_count") != 128
        ):
            raise RuntimeError(f"invalid B1 journal metadata: {path}")
        journal_clips.add(clip)
    if journal_clips != single:
        raise RuntimeError("B1 journal clip set does not match the frozen population")

    abstain = sum(label == "abstain" for label in labels.values())
    gate = {
        "status": "PASS",
        "protocol_sha256": PROTOCOL_SHA256,
        "protocol_amendment_sha256": AMENDMENT_SHA256,
        "join_amendment_sha256": JOIN_AMENDMENT_SHA256,
        "join_amendment_2_sha256": JOIN_AMENDMENT_2_SHA256,
        "join_amendment_3_sha256": JOIN_AMENDMENT_3_SHA256,
        "join_amendment_4_sha256": JOIN_AMENDMENT_4_SHA256,
        "manifest_sha256": manifest_sha,
        "coarse_class_map_sha256": COARSE_MAP_SHA256,
        "measurements_sha256": measurements_sha,
        "feature_corpus_sha256": feature_hasher.hexdigest(),
        "collection_completion_sha256": completion_sha,
        "collection_completion": completion,
        "n_clips": EXPECTED_CLIPS,
        "n_train_clips": 126,
        "n_eval_clips": 74,
        "n_labels_before_filter": EXPECTED_LABELS,
        "n_confident_labels": EXPECTED_LABELS - abstain,
        "n_abstain_labels": abstain,
        "n_pertoken_bundles": EXPECTED_BUNDLES,
        "n_pooled_bundles": EXPECTED_BUNDLES,
        "n_journals": EXPECTED_CLIPS,
        "retap_cache_join": {
            "method": "bundle_relative_l2",
            "threshold": 0.0002,
            "n_pairs": EXPECTED_BUNDLES,
            "max_relative_l2": join_max_relative_l2,
        },
        "schemas": {
            "pooled": [12, 448],
            "token_mean_max": [12, 896],
            "xattn_clip": [4, 64],
        },
    }
    _write_once_or_same(
        args.b1_root / "bundle_gate_v2.json",
        json.dumps(gate, indent=2, sort_keys=True) + "\n",
    )
    print("[b1 v2 gate] PASS", flush=True)
    return gate


def _load_family(args, rows: list[tuple], s: float, family: str) -> np.ndarray:
    arrays = []
    if family == "pooled":
        directory = args.pooled_root / "features"
        key = "pooled"
    else:
        directory = args.b1_root / "arc3" / "pertoken"
        key = family
    for _, _, gid in rows:
        with np.load(directory / f"{gid}__s{s:.2f}.npz", allow_pickle=False) as z:
            arrays.append(np.asarray(z[key], dtype=np.float32))
    result = np.stack(arrays)
    if result.dtype != np.float32 or not np.isfinite(result).all():
        raise RuntimeError(f"non-float32 or non-finite feature block: {family} s={s}")
    return result


def _prediction_records(
    rows: list[tuple],
    mask: np.ndarray,
    targets: np.ndarray,
    predictions: np.ndarray,
    *,
    split: str,
    s: float,
    family: str,
    probe: str,
    layer: int,
) -> list[dict]:
    selected_rows = [row for row, keep in zip(rows, mask) if keep]
    selected_targets = targets[mask]
    if len(selected_rows) != len(predictions):
        raise RuntimeError("prediction ledger row alignment failed")
    return [
        {
            "split": split,
            "s": s,
            "family": family,
            "probe": probe,
            "layer": layer,
            "clip": clip,
            "gen_id": gid,
            "true_label": str(target),
            "predicted_label": str(prediction),
        }
        for (clip, _j, gid), target, prediction in zip(
            selected_rows, selected_targets, predictions
        )
    ]


def _class_counts(values: np.ndarray) -> dict[str, int]:
    return dict(sorted(Counter(map(str, values.tolist())).items()))


def _filter_confident_rows(
    rows: list[tuple], labels: dict[str, str]
) -> list[tuple]:
    confident = [row for row in rows if labels[row[2]] != "abstain"]
    if any(labels[row[2]] == "abstain" for row in confident):
        raise RuntimeError("abstain reached the confident evaluation table")
    return confident


def _implementation_hashes() -> dict[str, str]:
    return {
        "library": sha256_file(Path(__file__).parents[1] / "foley_cw" / "arc4_b1.py"),
        "runner": sha256_file(Path(__file__)),
    }


def _s_token(s: float) -> str:
    return f"{s:.2f}".replace(".", "p")


def _inner_journal_path(args, s: float, family: str, probe: str, layer: int) -> Path:
    return (
        args.b1_root
        / "eval_v2"
        / "inner"
        / f"s{_s_token(s)}"
        / family
        / f"{probe}_layer{layer:02d}.json"
    )


def _outer_journal_path(args, s: float) -> Path:
    return args.b1_root / "eval_v2" / "outer" / f"s{_s_token(s)}.json"


def _inner_summary(spec: dict, metrics: dict, early: dict | None) -> dict:
    result = {
        **spec,
        "inner_validation_accuracy": metrics["accuracy"],
        "inner_validation_balanced_accuracy": metrics["balanced_accuracy"],
        "inner_validation_majority_baseline": metrics["majority_baseline"],
        "inner_validation_margin_over_majority": metrics["margin_over_majority"],
        "inner_validation_class_counts": metrics["class_counts"],
    }
    if early is not None:
        result["early_stopping"] = early
    return result


def _validate_prediction_rows(
    predictions: list[dict], expected_rows: list[dict], *, context: str
) -> None:
    keys = ("split", "s", "family", "probe", "layer", "clip", "gen_id", "true_label")
    if len(predictions) != len(expected_rows):
        raise RuntimeError(f"{context}: prediction-row count mismatch")
    for actual, expected in zip(predictions, expected_rows):
        if any(actual.get(key) != expected.get(key) for key in keys):
            raise RuntimeError(f"{context}: prediction-row identity mismatch")
        if actual.get("true_label") == "abstain" or not actual.get("predicted_label"):
            raise RuntimeError(f"{context}: invalid confident prediction row")


def _load_inner_journal(
    path: Path,
    *,
    spec: dict,
    expected_rows: list[dict],
    implementation: dict,
    input_hashes: dict,
) -> dict:
    row = json.loads(path.read_text())
    if (
        row.get("schema") != "arc4_b1_inner_candidate_v2"
        or row.get("protocol_sha256") != PROTOCOL_SHA256
        or row.get("protocol_amendment_sha256") != AMENDMENT_SHA256
        or row.get("join_amendment_sha256") != JOIN_AMENDMENT_SHA256
        or row.get("join_amendment_2_sha256") != JOIN_AMENDMENT_2_SHA256
        or row.get("join_amendment_3_sha256") != JOIN_AMENDMENT_3_SHA256
        or row.get("join_amendment_4_sha256") != JOIN_AMENDMENT_4_SHA256
        or row.get("implementation_sha256") != implementation
        or row.get("input_sha256") != input_hashes
        or row.get("spec") != spec
    ):
        raise RuntimeError(f"invalid or stale inner candidate journal: {path}")
    predictions = row.get("predictions") or []
    _validate_prediction_rows(predictions, expected_rows, context=str(path))
    metrics = classification_metrics(
        np.asarray([item["predicted_label"] for item in predictions]),
        [item["true_label"] for item in predictions],
    )
    summary = row.get("summary") or {}
    if any(summary.get(key) != value for key, value in spec.items()):
        raise RuntimeError(f"{path}: inner summary identity differs from spec")
    for key, metric_key in (
        ("inner_validation_accuracy", "accuracy"),
        ("inner_validation_balanced_accuracy", "balanced_accuracy"),
        ("inner_validation_majority_baseline", "majority_baseline"),
        ("inner_validation_margin_over_majority", "margin_over_majority"),
    ):
        if not math.isclose(float(summary.get(key, math.nan)), metrics[metric_key]):
            raise RuntimeError(f"{path}: stored inner metric mismatch for {key}")
    if summary.get("inner_validation_class_counts") != metrics["class_counts"]:
        raise RuntimeError(f"{path}: stored inner class counts mismatch")
    early = summary.get("early_stopping")
    if spec["probe"] == "ridge":
        if early is not None:
            raise RuntimeError(f"{path}: ridge journal carries MLP early stopping")
    else:
        if not isinstance(early, dict):
            raise RuntimeError(f"{path}: MLP journal lacks early stopping")
        best_epoch = early.get("best_epoch")
        epochs_run = early.get("epochs_run")
        if (
            not isinstance(best_epoch, int)
            or not isinstance(epochs_run, int)
            or not 1 <= best_epoch <= epochs_run <= 300
            or not math.isclose(
                float(early.get("inner_validation_accuracy", math.nan)),
                metrics["accuracy"],
            )
            or not math.isclose(
                float(early.get("inner_validation_balanced_accuracy", math.nan)),
                metrics["balanced_accuracy"],
            )
        ):
            raise RuntimeError(f"{path}: invalid MLP early-stopping metadata")
    return row


def _inner_family_worker(payload: dict) -> dict:
    args = payload["args"]
    s = payload["s"]
    family = payload["family"]
    rows = payload["rows"]
    targets = np.asarray(payload["targets"], dtype=object)
    is_fit = np.asarray(payload["is_fit"], dtype=bool)
    is_validation = np.asarray(payload["is_validation"], dtype=bool)
    implementation = payload["implementation"]
    input_hashes = payload["input_hashes"]
    fit_classes = sorted(set(targets[is_fit].tolist()))
    features = _load_family(args, rows, s, family)
    completed = 0
    try:
        from threadpoolctl import threadpool_limits

        limiter = threadpool_limits(limits=1)
    except ImportError:
        limiter = None
    try:
        for layer in range(features.shape[1]):
            X = features[:, layer, :]
            for probe in ("ridge", "mlp"):
                spec = {
                    "s": s,
                    "family": family,
                    "probe": probe,
                    "layer": layer,
                }
                expected_rows = _prediction_records(
                    rows,
                    is_validation,
                    targets,
                    np.asarray(["PENDING"] * int(np.sum(is_validation))),
                    split="inner_validation",
                    **spec,
                )
                path = _inner_journal_path(args, s, family, probe, layer)
                if path.exists():
                    _load_inner_journal(
                        path,
                        spec=spec,
                        expected_rows=expected_rows,
                        implementation=implementation,
                        input_hashes=input_hashes,
                    )
                    completed += 1
                    continue
                if probe == "ridge":
                    predicted = ridge_predict(
                        X[is_fit],
                        targets[is_fit].tolist(),
                        X[is_validation],
                        lam=1.0,
                    )
                    early = None
                else:
                    predicted, early = mlp_predict(
                        X[is_fit],
                        targets[is_fit].tolist(),
                        X[is_validation],
                        targets[is_validation].tolist(),
                        X[is_validation],
                        outer_classes=fit_classes,
                        seed=0,
                    )
                metrics = classification_metrics(predicted, targets[is_validation])
                summary = _inner_summary(spec, metrics, early)
                predictions = _prediction_records(
                    rows,
                    is_validation,
                    targets,
                    predicted,
                    split="inner_validation",
                    **spec,
                )
                journal = {
                    "schema": "arc4_b1_inner_candidate_v2",
                    "protocol_sha256": PROTOCOL_SHA256,
                    "protocol_amendment_sha256": AMENDMENT_SHA256,
                    "join_amendment_sha256": JOIN_AMENDMENT_SHA256,
                    "join_amendment_2_sha256": JOIN_AMENDMENT_2_SHA256,
                    "join_amendment_3_sha256": JOIN_AMENDMENT_3_SHA256,
                    "join_amendment_4_sha256": JOIN_AMENDMENT_4_SHA256,
                    "implementation_sha256": implementation,
                    "input_sha256": input_hashes,
                    "spec": spec,
                    "summary": summary,
                    "predictions": predictions,
                }
                _write_once_or_same(
                    path, json.dumps(journal, indent=2, sort_keys=True) + "\n"
                )
                completed += 1
                print(
                    f"[b1 v2] journaled inner s={s:.2f} family={family} "
                    f"probe={probe} layer={layer}",
                    flush=True,
                )
    finally:
        if limiter is not None:
            limiter.restore_original_limits()
    return {"s": s, "family": family, "completed": completed}


def _selected_identity(selected: dict) -> dict:
    return {
        "s": selected["s"],
        "family": selected["family"],
        "probe": selected["probe"],
        "layer": selected["layer"],
    }


def _load_outer_journal(
    path: Path,
    *,
    selected: dict,
    expected_rows: list[dict],
    implementation: dict,
    input_hashes: dict,
) -> dict:
    row = json.loads(path.read_text())
    if (
        row.get("schema") != "arc4_b1_outer_selected_v2"
        or row.get("protocol_sha256") != PROTOCOL_SHA256
        or row.get("protocol_amendment_sha256") != AMENDMENT_SHA256
        or row.get("join_amendment_sha256") != JOIN_AMENDMENT_SHA256
        or row.get("join_amendment_2_sha256") != JOIN_AMENDMENT_2_SHA256
        or row.get("join_amendment_3_sha256") != JOIN_AMENDMENT_3_SHA256
        or row.get("join_amendment_4_sha256") != JOIN_AMENDMENT_4_SHA256
        or row.get("implementation_sha256") != implementation
        or row.get("input_sha256") != input_hashes
        or row.get("selected_identity") != _selected_identity(selected)
        or row.get("inner_selected_spec") != selected
    ):
        raise RuntimeError(f"invalid or stale outer journal: {path}")
    predictions = row.get("predictions") or []
    _validate_prediction_rows(predictions, expected_rows, context=str(path))
    recomputed = accuracy_and_video_ci(
        np.asarray([item["predicted_label"] for item in predictions]),
        [item["true_label"] for item in predictions],
        [item["clip"] for item in predictions],
        n_boot=1000,
        seed=0,
    )
    stored = row.get("outer") or {}
    if stored != recomputed:
        raise RuntimeError(f"{path}: stored outer metric object mismatch")
    expected_epochs = (
        None
        if selected["probe"] == "ridge"
        else int(selected["early_stopping"]["best_epoch"])
    )
    if row.get("refit") != {"epochs": expected_epochs}:
        raise RuntimeError(f"{path}: outer refit metadata mismatch")
    return row


def _outer_worker(payload: dict) -> dict:
    args = payload["args"]
    s = payload["s"]
    selected = payload["selected"]
    rows = payload["rows"]
    targets = np.asarray(payload["targets"], dtype=object)
    row_clips = np.asarray([row[0] for row in rows], dtype=object)
    is_train = np.asarray(payload["is_train"], dtype=bool)
    is_eval = np.asarray(payload["is_eval"], dtype=bool)
    implementation = payload["implementation"]
    input_hashes = payload["input_hashes"]
    identity = _selected_identity(selected)
    expected_rows = _prediction_records(
        rows,
        is_eval,
        targets,
        np.asarray(["PENDING"] * int(np.sum(is_eval))),
        split="outer_eval",
        **identity,
    )
    path = _outer_journal_path(args, s)
    if path.exists():
        _load_outer_journal(
            path,
            selected=selected,
            expected_rows=expected_rows,
            implementation=implementation,
            input_hashes=input_hashes,
        )
        return {"s": s, "resumed": True}

    features = _load_family(args, rows, s, selected["family"])
    X = features[:, int(selected["layer"]), :]
    train_classes = sorted(set(targets[is_train].tolist()))
    try:
        from threadpoolctl import threadpool_limits

        limiter = threadpool_limits(limits=1)
    except ImportError:
        limiter = None
    try:
        if selected["probe"] == "ridge":
            predicted = ridge_predict(
                X[is_train], targets[is_train].tolist(), X[is_eval], lam=1.0
            )
            refit = {"epochs": None}
        else:
            epochs = int(selected["early_stopping"]["best_epoch"])
            predicted = mlp_predict_fixed_epochs(
                X[is_train],
                targets[is_train].tolist(),
                X[is_eval],
                outer_classes=train_classes,
                epochs=epochs,
                seed=0,
            )
            refit = {"epochs": epochs}
    finally:
        if limiter is not None:
            limiter.restore_original_limits()
    outer = accuracy_and_video_ci(
        predicted,
        targets[is_eval].tolist(),
        row_clips[is_eval].tolist(),
        n_boot=1000,
        seed=0,
    )
    predictions = _prediction_records(
        rows,
        is_eval,
        targets,
        predicted,
        split="outer_eval",
        **identity,
    )
    journal = {
        "schema": "arc4_b1_outer_selected_v2",
        "protocol_sha256": PROTOCOL_SHA256,
        "protocol_amendment_sha256": AMENDMENT_SHA256,
        "join_amendment_sha256": JOIN_AMENDMENT_SHA256,
        "join_amendment_2_sha256": JOIN_AMENDMENT_2_SHA256,
        "join_amendment_3_sha256": JOIN_AMENDMENT_3_SHA256,
        "join_amendment_4_sha256": JOIN_AMENDMENT_4_SHA256,
        "implementation_sha256": implementation,
        "input_sha256": input_hashes,
        "selected_identity": identity,
        "inner_selected_spec": selected,
        "refit": refit,
        "outer": outer,
        "predictions": predictions,
    }
    _write_once_or_same(path, json.dumps(journal, indent=2, sort_keys=True) + "\n")
    print(f"[b1 v2] journaled outer s={s:.2f}", flush=True)
    return {"s": s, "resumed": False}


def _run_parallel(function, payloads: list[dict], workers: int) -> None:
    context = multiprocessing.get_context("fork")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=min(workers, len(payloads)), mp_context=context
    ) as executor:
        futures = [executor.submit(function, payload) for payload in payloads]
        for future in concurrent.futures.as_completed(futures):
            print(f"[b1 v2 worker] {future.result()}", flush=True)


def run_evaluation(
    args,
    manifest: dict,
    rows: list[tuple],
    labels: dict[str, str],
    gate: dict,
) -> tuple[dict, list[dict], list[dict]]:
    population_clips = sorted({row[0] for row in rows})
    train_clips, eval_clips = _single_event_splits(manifest, population_clips)
    fit_clips, validation_clips = inner_clip_split(train_clips)
    confident_rows = _filter_confident_rows(rows, labels)
    if len(confident_rows) == len(rows):
        raise RuntimeError("B1 V2 power check failed: no abstaining rows")
    row_clips = np.asarray([clip for clip, _, _ in confident_rows], dtype=object)
    targets = np.asarray([labels[gid] for _, _, gid in confident_rows], dtype=object)
    is_train = np.asarray([clip in train_clips for clip in row_clips])
    is_eval = np.asarray([clip in eval_clips for clip in row_clips])
    is_fit = np.asarray([clip in fit_clips for clip in row_clips])
    is_validation = np.asarray([clip in validation_clips for clip in row_clips])
    if np.any(targets == "abstain") or not all(
        np.any(mask) for mask in (is_fit, is_validation, is_eval)
    ):
        raise RuntimeError("invalid B1 V2 confident split")
    if np.any(is_train & is_eval) or not np.all(is_train | is_eval):
        raise RuntimeError("B1 V2 confident rows violate the outer clip split")
    implementation = _implementation_hashes()
    input_hashes = {
        "protocol": PROTOCOL_SHA256,
        "bootstrap_amendment": AMENDMENT_SHA256,
        "join_amendment": JOIN_AMENDMENT_SHA256,
        "join_amendment_2": JOIN_AMENDMENT_2_SHA256,
        "join_amendment_3": JOIN_AMENDMENT_3_SHA256,
        "join_amendment_4": JOIN_AMENDMENT_4_SHA256,
        "population_manifest": gate["manifest_sha256"],
        "coarse_class_map": gate["coarse_class_map_sha256"],
        "measurements": gate["measurements_sha256"],
        "feature_corpus": gate["feature_corpus_sha256"],
        "collection_completion": gate["collection_completion_sha256"],
    }
    common = {
        "args": args,
        "rows": confident_rows,
        "targets": targets.tolist(),
        "is_train": is_train.tolist(),
        "is_eval": is_eval.tolist(),
        "is_fit": is_fit.tolist(),
        "is_validation": is_validation.tolist(),
        "implementation": implementation,
        "input_hashes": input_hashes,
    }
    inner_payloads = [
        {**common, "s": s, "family": family}
        for s in S_GRID
        for family in ("pooled", "token_mean_max", "xattn_clip")
    ]
    _run_parallel(_inner_family_worker, inner_payloads, args.workers)

    validation_targets = targets[is_validation]
    validation_pending = np.asarray(["PENDING"] * int(np.sum(is_validation)))
    inner_candidates_by_s = {}
    inner_predictions = []
    selected_by_s = {}
    for s in S_GRID:
        candidates = []
        for family, layers in (("pooled", 12), ("token_mean_max", 12), ("xattn_clip", 4)):
            for layer in range(layers):
                for probe in ("ridge", "mlp"):
                    spec = {"s": s, "family": family, "probe": probe, "layer": layer}
                    expected_rows = _prediction_records(
                        confident_rows,
                        is_validation,
                        targets,
                        validation_pending,
                        split="inner_validation",
                        **spec,
                    )
                    journal = _load_inner_journal(
                        _inner_journal_path(args, s, family, probe, layer),
                        spec=spec,
                        expected_rows=expected_rows,
                        implementation=implementation,
                        input_hashes=input_hashes,
                    )
                    candidates.append(journal["summary"])
                    inner_predictions.extend(journal["predictions"])
        inner_candidates_by_s[f"{s:.2f}"] = candidates
        selected_by_s[f"{s:.2f}"] = {
            "inner_selected_spec": select_spec(candidates)
        }

    outer_payloads = [
        {
            **common,
            "s": s,
            "selected": selected_by_s[f"{s:.2f}"]["inner_selected_spec"],
        }
        for s in S_GRID
    ]
    _run_parallel(_outer_worker, outer_payloads, args.workers)
    outer_predictions = []
    for s in S_GRID:
        selected = selected_by_s[f"{s:.2f}"]["inner_selected_spec"]
        identity = _selected_identity(selected)
        expected_rows = _prediction_records(
            confident_rows,
            is_eval,
            targets,
            np.asarray(["PENDING"] * int(np.sum(is_eval))),
            split="outer_eval",
            **identity,
        )
        journal = _load_outer_journal(
            _outer_journal_path(args, s),
            selected=selected,
            expected_rows=expected_rows,
            implementation=implementation,
            input_hashes=input_hashes,
        )
        selected_by_s[f"{s:.2f}"].update(
            {"outer": journal["outer"], "refit": journal["refit"]}
        )
        outer_predictions.extend(journal["predictions"])

    s_read = next(
        (
            s
            for s in S_GRID
            if selected_by_s[f"{s:.2f}"]["outer"]["balanced_accuracy"] >= 0.70
            and selected_by_s[f"{s:.2f}"]["outer"]["margin_over_majority"] >= 0.15
        ),
        None,
    )
    token = (
        "CLASS_INTERNAL_READOUT_FOUND"
        if s_read is not None and s_read <= 0.45
        else "R2_CLASS_CONFIRMED"
    )
    counts = {
        "collection_labels": len(rows),
        "abstain_labels": len(rows) - len(confident_rows),
        "confident_labels": len(confident_rows),
        "train_confident": int(np.sum(is_train)),
        "fit_confident": int(np.sum(is_fit)),
        "validation_confident": int(np.sum(is_validation)),
        "eval_confident": int(np.sum(is_eval)),
        "train_confident_clips": len(set(row_clips[is_train].tolist())),
        "fit_confident_clips": len(set(row_clips[is_fit].tolist())),
        "validation_confident_clips": len(set(row_clips[is_validation].tolist())),
        "eval_confident_clips": len(set(row_clips[is_eval].tolist())),
        "inner_candidate_journals": len(inner_candidates_by_s) * 56,
        "outer_selected_journals": len(selected_by_s),
    }
    result = {
        "_doc": "Arc-4 B-1 nested confirmatory result under B1_PROTOCOL_V2.md.",
        "protocol_sha256": PROTOCOL_SHA256,
        "protocol_amendment_sha256": AMENDMENT_SHA256,
        "join_amendment_sha256": JOIN_AMENDMENT_SHA256,
        "join_amendment_2_sha256": JOIN_AMENDMENT_2_SHA256,
        "join_amendment_3_sha256": JOIN_AMENDMENT_3_SHA256,
        "join_amendment_4_sha256": JOIN_AMENDMENT_4_SHA256,
        "collection_completion_sha256": gate["collection_completion_sha256"],
        "implementation_sha256": implementation,
        "input_sha256": input_hashes,
        "selection_source": "inner_validation_only",
        "outer_unselected_specs_evaluated": 0,
        "abstain_policy": "dropped_before_all_fitting_and_metrics",
        "numeric_dtype": "float32",
        "resume_contract": "write_once_candidate_and_selected_outer_journals",
        "workers": args.workers,
        "theta_balanced_accuracy": 0.70,
        "margin_over_majority_threshold": 0.15,
        "s_grid": list(S_GRID),
        "counts": counts,
        "class_counts": {
            "fit": _class_counts(targets[is_fit]),
            "validation": _class_counts(targets[is_validation]),
            "train": _class_counts(targets[is_train]),
            "eval": _class_counts(targets[is_eval]),
        },
        "inner_candidates_by_s": inner_candidates_by_s,
        "selected_by_s": selected_by_s,
        "s_read_internal_class": s_read,
        "decision": {"token": token, "complete": True},
    }
    return result, inner_predictions, outer_predictions


def _jsonl(rows: list[dict]) -> str:
    return "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    )


def _report(result: dict) -> str:
    lines = [
        "# Arc-4 B-1 confirmatory probe V2",
        "",
        f"Protocol SHA256: `{result['protocol_sha256']}`",
        f"Pre-evaluation amendment SHA256: `{result['protocol_amendment_sha256']}`",
        f"Join amendment 1 SHA256: `{result['join_amendment_sha256']}`",
        f"Join amendment 2 SHA256: `{result['join_amendment_2_sha256']}`",
        f"Join amendment 3 SHA256: `{result['join_amendment_3_sha256']}`",
        f"Binding join amendment 4 SHA256: `{result['join_amendment_4_sha256']}`",
        "",
        "Only confident labels enter fitting or evaluation. Family/probe/layer "
        "selection is performed on the fixed inner-validation clips; outer-eval "
        "labels never select an architecture. Selected specifications are refit "
        "on the outer-training clips and evaluated once.",
        "",
        "| s | balanced accuracy (95% CI) | raw accuracy (95% CI) | majority | "
        "margin | family | probe | layer |",
        "|---:|---:|---:|---:|---:|---|---|---:|",
    ]
    for s in S_GRID:
        row = result["selected_by_s"][f"{s:.2f}"]
        spec = row["inner_selected_spec"]
        outer = row["outer"]
        lines.append(
            f"| {s:.2f} | {outer['balanced_accuracy']:.4f} "
            f"[{outer['bal_ci_lo']:.4f}, {outer['bal_ci_hi']:.4f}] | "
            f"{outer['accuracy']:.4f} [{outer['ci_lo']:.4f}, "
            f"{outer['ci_hi']:.4f}] | {outer['majority_baseline']:.4f} | "
            f"{outer['margin_over_majority']:.4f} | {spec['family']} | "
            f"{spec['probe']} | {spec['layer']} |"
        )
    counts = result["counts"]
    lines += [
        "",
        f"Confident trajectories: train `{counts['train_confident']}`, outer eval "
        f"`{counts['eval_confident']}`; dropped abstentions: "
        f"`{counts['abstain_labels']}`.",
        "",
        f"s_read_internal_class: `{result['s_read_internal_class']}`",
        f"Decision token: **{result['decision']['token']}**",
        "",
        "CIs use 1,000 seed-0 video-bootstrap draws and fixed true-class "
        "universes; missing-class draws are discarded and replaced. The searched "
        "specification family is multiplicity-bearing, while the outer result is "
        "selected without outer-label access.",
        "",
        "Scope: one MMAudio checkpoint and confident self-target labels; this is "
        "not a human-perceptual class claim.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--b1-root", type=Path, default=Path("results/arc4_b1"))
    parser.add_argument("--pooled-root", type=Path, required=True)
    parser.add_argument("--measurements", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/phase1_manifest_frozen.json"),
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("experiment/preregistered/B1_PROTOCOL_V2.md"),
    )
    parser.add_argument(
        "--protocol-amendment",
        type=Path,
        default=Path("experiment/preregistered/B1_PROTOCOL_V2_AMENDMENT.md"),
    )
    parser.add_argument(
        "--join-amendment",
        type=Path,
        default=Path("experiment/preregistered/B1_PROTOCOL_V2_JOIN_AMENDMENT.md"),
    )
    parser.add_argument(
        "--join-amendment-2",
        type=Path,
        default=Path("experiment/preregistered/B1_PROTOCOL_V2_JOIN_AMENDMENT_2.md"),
    )
    parser.add_argument(
        "--join-amendment-3",
        type=Path,
        default=Path("experiment/preregistered/B1_PROTOCOL_V2_JOIN_AMENDMENT_3.md"),
    )
    parser.add_argument(
        "--join-amendment-4",
        type=Path,
        default=Path("experiment/preregistered/B1_PROTOCOL_V2_JOIN_AMENDMENT_4.md"),
    )
    parser.add_argument(
        "--coarse-class-map",
        type=Path,
        default=Path("configs/coarse_class_map.json"),
    )
    parser.add_argument(
        "--collection-completion",
        type=Path,
        default=Path("results/arc4_b1/collection_completion_manifest.json"),
    )
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()
    if not 1 <= args.workers <= 24:
        parser.error("--workers must be between 1 and 24")

    manifest = json.loads(args.manifest.read_text())
    clips, rows = _expected(manifest)
    gate = run_gate(args, manifest, clips, rows)
    if not args.evaluate:
        print("[b1 v2] gate-only mode; no evaluation run", flush=True)
        return 0

    labels, _ = _load_labels(args.measurements)
    result, inner_predictions, outer_predictions = run_evaluation(
        args, manifest, rows, labels, gate
    )
    inner_path = args.b1_root / "b1_inner_predictions_v2.jsonl"
    outer_path = args.b1_root / "b1_outer_predictions_v2.jsonl"
    _write_once_or_same(inner_path, _jsonl(inner_predictions))
    _write_once_or_same(outer_path, _jsonl(outer_predictions))
    result["prediction_ledgers"] = {
        "inner": {
            "path": str(inner_path),
            "sha256": sha256_file(inner_path),
            "rows": len(inner_predictions),
        },
        "outer": {
            "path": str(outer_path),
            "sha256": sha256_file(outer_path),
            "rows": len(outer_predictions),
        },
    }
    _write_once_or_same(
        args.b1_root / "b1_probe_v2.json",
        json.dumps(result, indent=2, sort_keys=True) + "\n",
    )
    _write_once_or_same(args.b1_root / "b1_probe_v2.md", _report(result))
    print(f"[b1 v2] decision={result['decision']['token']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
