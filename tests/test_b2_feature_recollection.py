"""Fail-closed tests for B2 full feature recollection artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import foley_cw.b2_feature_recollection as recollect
from foley_cw.b1_lineage import (
    CONDITION_FIELDS,
    PASS_ROLES,
    _required_replay_keys,
    create_bound_attempt,
    load_protocol,
    sha256_file,
)
from foley_cw.b2_class_closure import atomic_json_create


REPO = Path(__file__).resolve().parents[1]
PROTOCOL = REPO / "experiment" / "non_human_closure" / "PROTOCOL.json"
PROTOCOL_SHA = sha256_file(PROTOCOL)


def _capture_and_arrays() -> tuple[dict, dict[str, np.ndarray]]:
    passes = []
    tokens = []
    attention = []
    arrays: dict[str, np.ndarray] = {
        "packet_x_s_fp32": np.zeros((250, 20), np.float32),
        "packet_model_time_fp32": np.asarray(0.35, np.float32),
        "device_latent_fp32": np.zeros((1, 250, 20), np.float32),
        "device_time_fp32": np.asarray(0.35, np.float32),
        "device_latent_native": np.zeros((1, 250, 20), np.float32),
        "device_time_native": np.asarray(0.35, np.float32),
        "returned_velocity_native": np.ones((1, 250, 20), np.float32),
        "returned_velocity_fp32": np.ones((1, 250, 20), np.float32),
        "tweedie_latent_normalized_native": np.ones((1, 250, 20), np.float32),
        "tweedie_latent_fp32": np.ones((1, 250, 20), np.float32),
        "tweedie_latent_unnormalized_fp32": np.ones((1, 250, 20), np.float32),
        "panns_clipwise_output_fp32": np.zeros((1, 527), np.float32),
        "panns_embedding_fp32": np.zeros((1, 8), np.float32),
        "external_preview_waveform_fp32": np.zeros(128000, np.float32),
    }
    for prefix in ("conditions", "empty_conditions"):
        for name in CONDITION_FIELDS:
            arrays[f"{prefix}__{name}"] = np.zeros((1, 2, 3), np.float32)
    for pass_index, role in enumerate(PASS_ROLES):
        time_key = f"time_{role}"
        latent_key = f"latent_{role}"
        arrays[time_key] = np.zeros((1,), np.float32)
        arrays[latent_key] = np.zeros((1, 250, 20), np.float32)
        consumed = {}
        for name in CONDITION_FIELDS:
            key = f"consumed_{role}_{name}"
            arrays[key] = np.zeros((1, 2, 3), np.float32)
            consumed[name] = key
        network = {}
        for field in (
            "joint_input_latent",
            "joint_input_clip_tokens",
            "joint_input_text_tokens",
            "global_condition",
            "extended_condition",
        ):
            key = f"{field}_{role}"
            arrays[key] = np.zeros((1, 2, 3), np.float32)
            network[field] = key
        passes.append(
            {
                "pass_index": pass_index,
                "pass_role": role,
                "actual_broadcast_time": time_key,
                "actual_latent": latent_key,
                "conditions_as_consumed": consumed,
                "network_consumed": network,
            }
        )
        for layer in range(12):
            names = {}
            for field in (
                "persisted_tokens",
                "quantized_tokens",
                "dequantized_tokens",
                "mean_after_quantization",
                "pooled_original",
                "pooled_repaired",
                "token_mean",
                "token_stats",
            ):
                key = f"{field}_{role}_{layer}"
                names[field] = key
                if field == "persisted_tokens" or field == "dequantized_tokens":
                    arrays[key] = np.zeros((1, 2, 3), np.float32)
                elif field == "quantized_tokens":
                    arrays[key] = np.zeros((1, 2, 3), np.float16)
                elif field == "token_stats":
                    arrays[key] = np.zeros((5,), np.float32)
                else:
                    arrays[key] = np.zeros((3,), np.float32)
            tokens.append({"pass_role": role, "site": f"layer.{layer}", **names})
        for site in range(3):
            names = {}
            for field in (
                "q",
                "k",
                "v",
                "actual_attention_output",
                "actual_latent_query_output",
                "actual_latent_query_summary",
            ):
                key = f"{field}_{role}_{site}"
                names[field] = key
                arrays[key] = np.zeros((1, 1, 2, 3), np.float32)
            if site < 2:
                names["probability_map"] = f"probability_map_{role}_{site}"
                names["probability_summary"] = f"probability_summary_{role}_{site}"
                arrays[names["probability_map"]] = np.zeros((1, 1, 2, 2), np.float32)
                arrays[names["probability_summary"]] = np.zeros((1, 2), np.float32)
            else:
                names["probability_map"] = None
                names["probability_summary"] = None
            attention.append({"pass_role": role, "site": f"site.{site}", **names})
    capture = {
        "capture_nonce": "n",
        "one_ode_wrapper_evaluation": True,
        "pass_roles": list(PASS_ROLES),
        "conditioning_complete": True,
        "passes": passes,
        "tokens": tokens,
        "attention": attention,
    }
    assert set(arrays) == _required_replay_keys(capture) | {
        "external_preview_waveform_fp32"
    }
    return capture, arrays


def _root(tmp_path: Path, name: str) -> Path:
    protocol = load_protocol(PROTOCOL, PROTOCOL_SHA)
    return create_bound_attempt(tmp_path, "feature_shards", name, protocol)


def _write_one(tmp_path: Path, name: str = "s") -> Path:
    capture, arrays = _capture_and_arrays()
    root = _root(tmp_path, name)
    return recollect.write_feature_unit(
        root,
        identity={"video_id": "v0", "base_seed": 0, "progress": 0.35},
        arrays=arrays,
        capture=capture,
        base_final={"exact_sample_match": True, "bank_audio_sha256": "a" * 64},
        lineage_gate={"status": "PASS", "completion_sha256": "b" * 64},
        provenance={"test": True},
    )


def test_feature_unit_schema_and_deep_readback(tmp_path):
    unit = _write_one(tmp_path)
    manifest = recollect.validate_feature_unit(unit, deep=True)
    assert manifest["identity"] == {
        "video_id": "v0",
        "base_seed": 0,
        "progress": 0.35,
        "cfg": 4.5,
    }
    assert len(manifest["capture"]["tokens"]) == 24
    assert len(manifest["capture"]["attention"]) == 6


def test_feature_npz_is_byte_deterministic(tmp_path):
    left = _write_one(tmp_path / "left")
    right = _write_one(tmp_path / "right")
    assert sha256_file(left / "arrays.npz") == sha256_file(right / "arrays.npz")


def test_feature_corruption_is_rejected(tmp_path):
    unit = _write_one(tmp_path)
    path = unit / "arrays.npz"
    damaged = bytearray(path.read_bytes())
    damaged[len(damaged) // 2] ^= 1
    path.write_bytes(damaged)
    with pytest.raises(recollect.FeatureRecollectionError, match="hash/size"):
        recollect.validate_feature_unit(unit)


def test_partial_shard_is_rejected(monkeypatch, tmp_path):
    unit = _write_one(tmp_path)
    root = unit.parent.parent
    inventory = tmp_path / "inventory.json"
    inventory.write_text("{}\n")
    bases = [
        {
            "video_id": "v0",
            "base_seed": 0,
            "role": "base",
        }
    ]
    monkeypatch.setattr(
        recollect,
        "base_records",
        lambda _path: (bases, {"records_sha256": "r" * 64}),
    )
    row = {
        "unit_id": unit.name,
        "completion_sha256": sha256_file(unit / "COMPLETED.json"),
        "manifest_sha256": sha256_file(unit / "manifest.json"),
        "arrays_sha256": sha256_file(unit / "arrays.npz"),
    }
    completion = {
        "schema": recollect.SHARD_SCHEMA,
        "status": "COMPLETE",
        "protocol_sha256": PROTOCOL_SHA,
        "inventory_manifest": str(inventory),
        "inventory_manifest_sha256": sha256_file(inventory),
        "inventory_records_sha256": "r" * 64,
        "shard_index": 0,
        "shard_count": 1,
        "unit_count": 1,
        "units": [row],
    }
    path = root / "FEATURE_SHARD_COMPLETION.json"
    atomic_json_create(path, completion)
    with pytest.raises(recollect.FeatureRecollectionError, match="partial"):
        recollect.validate_feature_shard(path)


def test_assignment_is_deterministic_and_disjoint():
    bases = [
        {"video_id": f"v{i // 2}", "base_seed": i % 2} for i in range(12)
    ]
    shards = [recollect.assigned_bases(bases, index, 4) for index in range(4)]
    flattened = [(row["video_id"], row["base_seed"]) for shard in shards for row in shard]
    assert len(flattened) == len(set(flattened)) == len(bases)
    assert recollect.assigned_bases(bases, 2, 4) == shards[2]
