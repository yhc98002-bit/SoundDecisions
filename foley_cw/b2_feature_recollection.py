"""Lineage-valid B2 base-trajectory feature recollection.

The five-clip B-1 gate lives in :mod:`foley_cw.b1_lineage`.  This module is
deliberately downstream of that gate: a collector cannot create an output root
unless an immutable held-out attempt reports ``PASS``.  Every worker owns one
create-only shard root, and the reducer records a canonical manifest without
copying the large tensors into Git.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import time
import zlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from . import score_sde
from .b1_lineage import (
    CONDITION_FIELDS,
    PASS_ROLES,
    SameForwardCapture,
    _condition_arrays,
    _required_replay_keys,
    _require_asset_roots,
    describe_array,
    environment_provenance,
    load_protocol,
    sha256_file,
    validate_attempt,
    validate_protocol_binding,
    create_bound_attempt,
)
from .b2_class_closure import (
    atomic_json_create,
    atomic_jsonl_create,
    canonical_json_bytes,
    deterministic_npz_create,
    load_inventory,
    sha256_bytes,
)
from .types import ScheduleSpec


SHARD_SCHEMA = "sounddecisions.b2_feature_shard.v1"
UNIT_SCHEMA = "sounddecisions.b2_feature_unit.v1"
UNIT_COMPLETION_SCHEMA = "sounddecisions.b2_feature_unit_completion.v1"
MERGE_SCHEMA = "sounddecisions.b2_feature_recollection_merge.v1"
S_POINTS = (0.05, 0.15, 0.25, 0.35, 0.45, 0.60, 0.75, 0.90)
EXPECTED_BASES = 816
EXPECTED_UNITS = EXPECTED_BASES * len(S_POINTS)
CFG = 4.5
UNIT_RE = re.compile(r"^[A-Za-z0-9_-]+__seed(?:0|[1-9][0-9]*)__s[0-9]+\.[0-9]{2}$")


class FeatureRecollectionError(RuntimeError):
    """Fail-closed feature collection or reduction error."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FeatureRecollectionError(f"invalid JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise FeatureRecollectionError(f"expected JSON object: {path}")
    return payload


def _git_head(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()


def _git_clean(repo: Path) -> bool:
    return not subprocess.check_output(
        ["git", "-C", str(repo), "status", "--porcelain"], text=True
    ).strip()


def rng_for(base_seed: int, *parts: Any) -> np.random.Generator:
    """Exact Arc-4 B2 RNG contract."""
    entropy = [int(base_seed)] + [zlib.crc32(str(part).encode("utf-8")) for part in parts]
    return np.random.default_rng(np.random.SeedSequence(entropy))


def feature_unit_id(video_id: str, base_seed: int, progress: float) -> str:
    value = f"{video_id}__seed{int(base_seed)}__s{float(progress):.2f}"
    if not UNIT_RE.fullmatch(value):
        raise FeatureRecollectionError(f"unsafe feature unit id {value!r}")
    return value


def base_records(inventory_manifest: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records, manifest = load_inventory(inventory_manifest)
    bases = sorted(
        (row for row in records if row["role"] == "base"),
        key=lambda row: (str(row["video_id"]), int(row["base_seed"])),
    )
    if manifest.get("canonical_b2") is True:
        if len(bases) != EXPECTED_BASES:
            raise FeatureRecollectionError(f"canonical base cardinality {len(bases)} != 816")
        if len({str(row["video_id"]) for row in bases}) != 48:
            raise FeatureRecollectionError("canonical inventory does not contain 48 videos")
        if sorted({int(row["base_seed"]) for row in bases}) != list(range(17)):
            raise FeatureRecollectionError("canonical inventory does not contain seeds 0..16")
    return bases, manifest


def assigned_bases(
    bases: Sequence[Mapping[str, Any]], shard_index: int, shard_count: int
) -> list[Mapping[str, Any]]:
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise FeatureRecollectionError("invalid feature shard assignment")
    return [row for index, row in enumerate(bases) if index % shard_count == shard_index]


def _validate_full_protocol(protocol: Mapping[str, Any]) -> None:
    full = protocol["payload"].get("lineage", {}).get("full_recollection", {})
    required = {
        "population": "48 B2 videos x 17 base seeds x 8 progress points",
        "base_trajectory_cfg": 4.5,
        "base_trajectory_seed_contract": (
            "SeedSequence([base_seed,crc32(clip),crc32('base')])"
        ),
        "require_banked_base_final_identity": True,
        "store_fp32_post_block_tokens": True,
        "store_large_tensors_outside_git": True,
    }
    for key, expected in required.items():
        if full.get(key) != expected:
            raise FeatureRecollectionError(
                f"frozen lineage.full_recollection.{key}={full.get(key)!r}; "
                f"expected {expected!r}"
            )


def validate_lineage_gate(
    heldout_attempt: Path, *, protocol_sha256: str
) -> dict[str, Any]:
    summary = validate_attempt(
        heldout_attempt,
        expected_stage="heldout",
        expected_protocol_sha256=protocol_sha256,
    )
    report_path = Path(heldout_attempt) / "HELDOUT_REPORT.json"
    report = _load_json(report_path)
    if (
        report.get("status") != "PASS"
        or int(report.get("failure_count", -1)) != 0
        or report.get("heldout_clip") != "1002"
        or report.get("progress_points") != list(S_POINTS)
        or report.get("protocol_sha256") != protocol_sha256
        or report.get("tolerance_unchanged") is not True
    ):
        raise FeatureRecollectionError("B-1 held-out lineage gate did not pass")
    return {
        "status": "PASS",
        "attempt": str(Path(heldout_attempt).resolve()),
        "completion_sha256": summary["completion_sha256"],
        "report_sha256": sha256_file(report_path),
        "tolerance_sha256": report["tolerance_sha256"],
    }


def _capture_state(
    backend: Any,
    panns_model: Any,
    condition_bundle: Any,
    x_s: np.ndarray,
    progress: float,
    device: str,
    nonce: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    import torch
    from mmaudio.model import transformer_layers

    x_np = np.ascontiguousarray(x_s, dtype=np.float32)
    model_time = np.asarray(backend.s_to_t.s_to_t(float(progress)), dtype=np.float32)
    x_device = torch.from_numpy(x_np).to(device=device, dtype=backend.dtype).unsqueeze(0)
    t_device = torch.tensor(float(model_time), device=device, dtype=backend.dtype)
    collector = SameForwardCapture(backend.net, transformer_layers)
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(torch.device(device))
    started = time.time()
    with torch.inference_mode():
        with collector.armed(nonce):
            velocity = backend.net.ode_wrapper(
                t_device,
                x_device,
                condition_bundle.conditions,
                condition_bundle.empty_conditions,
                CFG,
            )
        capture_arrays, capture = collector.finish()
        tweedie = (x_device + (1.0 - t_device) * velocity).clone()
        unnormalized = backend.net.unnormalize(tweedie.clone())
        spectrogram = backend.feature_utils.decode(unnormalized)
        preview_tensor = backend.feature_utils.vocode(spectrogram)
        preview = preview_tensor.float().cpu().numpy().reshape(-1).astype(np.float32)
        panns = panns_model(
            torch.from_numpy(preview[None, :]).to(device=device, dtype=torch.float32)
        )
    arrays: dict[str, np.ndarray] = {
        "packet_x_s_fp32": x_np,
        "packet_model_time_fp32": model_time,
        "device_latent_fp32": x_device.detach().float().cpu().numpy(),
        "device_time_fp32": t_device.detach().float().cpu().numpy(),
        "device_latent_native": x_device.detach().cpu().numpy(),
        "device_time_native": t_device.detach().cpu().numpy(),
        "returned_velocity_native": velocity.detach().cpu().numpy(),
        "returned_velocity_fp32": velocity.detach().float().cpu().numpy(),
        "tweedie_latent_normalized_native": tweedie.detach().cpu().numpy(),
        "tweedie_latent_fp32": tweedie.detach().float().cpu().numpy(),
        "tweedie_latent_unnormalized_fp32": unnormalized.detach().float().cpu().numpy(),
        "panns_clipwise_output_fp32": panns["clipwise_output"].detach().float().cpu().numpy(),
        "panns_embedding_fp32": panns["embedding"].detach().float().cpu().numpy(),
        "external_preview_waveform_fp32": preview,
        **capture_arrays,
        **_condition_arrays(condition_bundle.conditions, "conditions"),
        **_condition_arrays(condition_bundle.empty_conditions, "empty_conditions"),
    }
    capture = dict(capture)
    capture.update(
        {
            "conditioning_fields": list(CONDITION_FIELDS),
            "conditioning_complete": True,
            "actual_inputs": {
                "device_latent": describe_array(arrays["device_latent_fp32"]),
                "device_time": describe_array(arrays["device_time_fp32"]),
            },
            "returned_velocity": describe_array(arrays["returned_velocity_fp32"]),
            "tweedie_latent": describe_array(arrays["tweedie_latent_fp32"]),
            "tweedie_contract": {
                "operation": "x_device + (1.0 - t_device) * returned_velocity",
                "normalized_clone_before_unnormalize": True,
                "unnormalize_is_in_place": True,
            },
            "runtime": {
                "elapsed_seconds": time.time() - started,
                "one_ode_wrapper_evaluation": True,
                "predict_flow_nfe": len(PASS_ROLES),
                "peak_allocated_bytes": (
                    int(torch.cuda.max_memory_allocated(torch.device(device)))
                    if str(device).startswith("cuda") and torch.cuda.is_available()
                    else 0
                ),
            },
        }
    )
    expected = _required_replay_keys(capture) | {"external_preview_waveform_fp32"}
    if set(arrays) != expected:
        raise FeatureRecollectionError(
            f"same-forward tensor contract mismatch: missing={sorted(expected-set(arrays))}, "
            f"extra={sorted(set(arrays)-expected)}"
        )
    return arrays, capture


def write_feature_unit(
    shard_root: Path,
    *,
    identity: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    capture: Mapping[str, Any],
    base_final: Mapping[str, Any],
    lineage_gate: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> Path:
    protocol = validate_protocol_binding(shard_root)
    uid = feature_unit_id(
        str(identity["video_id"]), int(identity["base_seed"]), float(identity["progress"])
    )
    root = Path(shard_root) / "units" / uid
    try:
        root.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise FeatureRecollectionError(f"refusing to overwrite feature unit {root}") from exc
    normalized = {key: np.asarray(value) for key, value in arrays.items()}
    required = _required_replay_keys(capture) | {"external_preview_waveform_fp32"}
    if set(normalized) != required:
        raise FeatureRecollectionError("feature unit array inventory is incomplete")
    descriptors = {key: describe_array(value) for key, value in normalized.items()}
    arrays_path = root / "arrays.npz"
    deterministic_npz_create(arrays_path, normalized)
    manifest = {
        "schema": UNIT_SCHEMA,
        "protocol_sha256": protocol["sha256"],
        "unit_id": uid,
        "identity": {
            "video_id": str(identity["video_id"]),
            "base_seed": int(identity["base_seed"]),
            "progress": float(identity["progress"]),
            "cfg": CFG,
        },
        "base_final_identity": dict(base_final),
        "lineage_gate": dict(lineage_gate),
        "arrays_file": arrays_path.name,
        "arrays_file_sha256": sha256_file(arrays_path),
        "arrays_file_bytes": arrays_path.stat().st_size,
        "required_array_keys": sorted(required),
        "arrays": descriptors,
        "capture": dict(capture),
        "provenance": dict(provenance),
    }
    manifest_path = root / "manifest.json"
    atomic_json_create(manifest_path, manifest)
    completion = {
        "schema": UNIT_COMPLETION_SCHEMA,
        "status": "COMPLETE",
        "unit_id": uid,
        "manifest_sha256": sha256_file(manifest_path),
        "arrays_sha256": manifest["arrays_file_sha256"],
        "arrays_bytes": manifest["arrays_file_bytes"],
    }
    atomic_json_create(root / "COMPLETED.json", completion)
    return root


def validate_feature_unit(root: Path, *, deep: bool = False) -> dict[str, Any]:
    root = Path(root)
    completion = _load_json(root / "COMPLETED.json")
    manifest = _load_json(root / "manifest.json")
    if completion.get("schema") != UNIT_COMPLETION_SCHEMA or completion.get("status") != "COMPLETE":
        raise FeatureRecollectionError(f"invalid unit completion: {root}")
    if manifest.get("schema") != UNIT_SCHEMA or manifest.get("unit_id") != root.name:
        raise FeatureRecollectionError(f"invalid unit manifest: {root}")
    if completion.get("unit_id") != root.name:
        raise FeatureRecollectionError(f"unit completion ID mismatch: {root}")
    if sha256_file(root / "manifest.json") != completion.get("manifest_sha256"):
        raise FeatureRecollectionError(f"unit manifest hash mismatch: {root}")
    arrays_path = root / str(manifest.get("arrays_file", ""))
    if (
        sha256_file(arrays_path) != completion.get("arrays_sha256")
        or completion.get("arrays_sha256") != manifest.get("arrays_file_sha256")
        or arrays_path.stat().st_size != int(completion.get("arrays_bytes", -1))
    ):
        raise FeatureRecollectionError(f"unit arrays hash/size mismatch: {root}")
    identity = manifest.get("identity", {})
    if feature_unit_id(
        str(identity.get("video_id", "")),
        int(identity.get("base_seed", -1)),
        float(identity.get("progress", -1)),
    ) != root.name:
        raise FeatureRecollectionError(f"unit scientific identity mismatch: {root}")
    base = manifest.get("base_final_identity", {})
    if base.get("exact_sample_match") is not True or not base.get("bank_audio_sha256"):
        raise FeatureRecollectionError(f"banked base identity absent: {root}")
    capture = manifest.get("capture", {})
    required = _required_replay_keys(capture) | {"external_preview_waveform_fp32"}
    if set(manifest.get("required_array_keys", [])) != required:
        raise FeatureRecollectionError(f"required tensor inventory mismatch: {root}")
    if set(manifest.get("arrays", {})) != required:
        raise FeatureRecollectionError(f"described tensor inventory mismatch: {root}")
    passes = capture.get("passes", [])
    tokens = capture.get("tokens", [])
    attention = capture.get("attention", [])
    if (
        capture.get("one_ode_wrapper_evaluation") is not True
        or capture.get("pass_roles") != list(PASS_ROLES)
        or capture.get("conditioning_complete") is not True
        or len(passes) != len(PASS_ROLES)
        or [row.get("pass_role") for row in passes] != list(PASS_ROLES)
        or len(tokens) != 24
        or {row.get("pass_role") for row in tokens} != set(PASS_ROLES)
        or len(attention) != 6
        or {row.get("pass_role") for row in attention} != set(PASS_ROLES)
    ):
        raise FeatureRecollectionError(f"same-forward capture contract absent: {root}")
    for row in tokens:
        token = manifest["arrays"].get(str(row.get("persisted_tokens")), {})
        pooled_original = manifest["arrays"].get(str(row.get("pooled_original")), {})
        pooled_repaired = manifest["arrays"].get(str(row.get("pooled_repaired")), {})
        if (
            token.get("dtype") != "<f4"
            or len(token.get("shape", [])) != 3
            or pooled_original.get("dtype") != "<f4"
            or len(pooled_original.get("shape", [])) != 1
            or pooled_repaired.get("shape") != pooled_original.get("shape")
        ):
            raise FeatureRecollectionError(f"post-block fp32 token/pool schema mismatch: {root}")
    preview = manifest["arrays"]["external_preview_waveform_fp32"]
    if preview.get("dtype") != "<f4" or preview.get("shape") != [128000]:
        raise FeatureRecollectionError(f"external preview schema mismatch: {root}")
    if deep:
        try:
            with np.load(arrays_path, allow_pickle=False) as archive:
                if set(archive.files) != required:
                    raise FeatureRecollectionError(f"NPZ key mismatch: {root}")
                for key in archive.files:
                    if describe_array(np.asarray(archive[key])) != manifest["arrays"][key]:
                        raise FeatureRecollectionError(f"array descriptor mismatch: {root}:{key}")
        except FeatureRecollectionError:
            raise
        except Exception as exc:
            raise FeatureRecollectionError(f"cannot read feature unit {root}: {exc}") from exc
    return manifest


def collect_feature_shard(
    inventory_manifest: Path,
    heldout_attempt: Path,
    output_root: Path,
    attempt_id: str,
    *,
    shard_index: int,
    shard_count: int,
    mmaudio_root: Path,
    weights_dir: Path,
    clips_root: Path,
    device: str,
    protocol_path: Path,
    protocol_sha256: str,
) -> Path:
    protocol = load_protocol(protocol_path, protocol_sha256)
    _validate_full_protocol(protocol)
    gate = validate_lineage_gate(heldout_attempt, protocol_sha256=protocol["sha256"])
    assets = _require_asset_roots(mmaudio_root, weights_dir, clips_root, protocol["pilot"])
    bases, inventory = base_records(inventory_manifest)
    selected = assigned_bases(bases, shard_index, shard_count)
    if not selected:
        raise FeatureRecollectionError("feature shard has no assigned base trajectories")
    repo = Path(__file__).resolve().parents[1]
    collector_hash = sha256_file(Path(__file__).resolve())
    start_commit = _git_head(repo)
    if not _git_clean(repo):
        raise FeatureRecollectionError("feature collection requires a clean worktree")
    provenance = environment_provenance(
        device,
        Path(mmaudio_root),
        protocol_sha256=protocol["sha256"],
        asset_observation=assets,
    )
    if provenance.get("project_git_dirty"):
        raise FeatureRecollectionError("feature provenance observed a dirty worktree")
    provenance.update(
        {
            "feature_collector_sha256": collector_hash,
            "inventory_manifest": str(Path(inventory_manifest).resolve()),
            "inventory_manifest_sha256": sha256_file(inventory_manifest),
            "lineage_gate": gate,
            "shard_index": int(shard_index),
            "shard_count": int(shard_count),
            "network_downloads": "forbidden; pinned local assets only",
        }
    )
    root = create_bound_attempt(output_root, "feature_shards", attempt_id, protocol)

    mmaudio_root = Path(mmaudio_root).resolve()
    if str(mmaudio_root) not in sys.path:
        sys.path.insert(0, str(mmaudio_root))
    from .mmaudio_backend import MMAudioBackend
    from .measurers_panns_cnn14 import load_cnn14_16k
    import soundfile as sf

    backend = MMAudioBackend(
        variant="small_16k",
        device=device,
        full_precision=True,
        cfg_strength=CFG,
        num_steps=20,
        duration_sec=8.0,
        weights_root=str(mmaudio_root),
        enable_conditions=True,
    )
    panns_path = Path(weights_dir) / "Cnn14_16k_mAP=0.438.pth"
    panns = load_cnn14_16k(panns_path, device=device)
    schedule = ScheduleSpec(
        n_steps=20,
        scan_points=S_POINTS,
        K_forks=12,
        N_independent=16,
        g_kind="sqrt_down",
        g_value=1.0,
    )
    grid = schedule.integration_s_grid()
    if any(not np.any(np.isclose(grid, point, atol=1e-12, rtol=0.0)) for point in S_POINTS):
        raise FeatureRecollectionError("registered progress grid is off the B2 integration grid")
    conditions: dict[str, Any] = {}
    written: list[dict[str, Any]] = []
    for base in selected:
        video_id = str(base["video_id"])
        base_seed = int(base["base_seed"])
        if video_id not in conditions:
            video_path = Path(clips_root) / f"{video_id}.mp4"
            conditions[video_id] = backend.make_video_cond(str(video_path), video_id=video_id)
        condition = conditions[video_id]
        trajectory = score_sde.generate_trajectory(
            backend,
            condition,
            schedule,
            rng_for(base_seed, video_id, "base"),
            alpha=0.0,
            record_points=S_POINTS,
        )
        generated = np.asarray(trajectory["audio"], dtype=np.float32).reshape(-1)
        banked, sample_rate = sf.read(base["audio_path"], dtype="float32", always_2d=False)
        banked = np.asarray(banked, dtype=np.float32).reshape(-1)
        exact = generated.shape == banked.shape and np.array_equal(generated, banked)
        if not exact:
            raise FeatureRecollectionError(
                f"banked base-final identity failed for {video_id}/seed{base_seed}"
            )
        base_identity = {
            "bank_audio_path": str(Path(base["audio_path"]).resolve()),
            "bank_audio_sha256": str(base["audio_sha256"]),
            "bank_source_unit_journal_sha256": str(base["source_unit_journal_sha256"]),
            "decoded_samples_sha256": sha256_bytes(banked.tobytes(order="C")),
            "generated_samples_sha256": sha256_bytes(generated.tobytes(order="C")),
            "sample_rate": int(sample_rate),
            "frames": int(generated.size),
            "exact_sample_match": True,
        }
        for progress in S_POINTS:
            identity = {
                "video_id": video_id,
                "base_seed": base_seed,
                "progress": float(progress),
            }
            nonce = hashlib.sha256(
                canonical_json_bytes(
                    {
                        **identity,
                        "attempt_id": attempt_id,
                        "collector_sha256": collector_hash,
                    },
                    indent=None,
                )
            ).hexdigest()[:32]
            arrays, capture = _capture_state(
                backend,
                panns,
                condition,
                trajectory["states"][progress],
                progress,
                device,
                nonce,
            )
            unit = write_feature_unit(
                root,
                identity=identity,
                arrays=arrays,
                capture=capture,
                base_final=base_identity,
                lineage_gate=gate,
                provenance=provenance,
            )
            written.append(
                {
                    "unit_id": unit.name,
                    "completion_sha256": sha256_file(unit / "COMPLETED.json"),
                    "manifest_sha256": sha256_file(unit / "manifest.json"),
                    "arrays_sha256": sha256_file(unit / "arrays.npz"),
                }
            )
        print(
            f"[feature {shard_index}/{shard_count}] {video_id}/seed{base_seed} "
            f"complete ({len(written)} state units)",
            flush=True,
        )
    if (
        sha256_file(Path(__file__).resolve()) != collector_hash
        or _git_head(repo) != start_commit
        or not _git_clean(repo)
    ):
        raise FeatureRecollectionError("source/worktree changed during feature collection")
    completion = {
        "schema": SHARD_SCHEMA,
        "status": "COMPLETE",
        "immutable": True,
        "protocol_sha256": protocol["sha256"],
        "inventory_manifest": str(Path(inventory_manifest).resolve()),
        "inventory_manifest_sha256": sha256_file(inventory_manifest),
        "inventory_records_sha256": inventory["records_sha256"],
        "lineage_gate": gate,
        "shard_index": int(shard_index),
        "shard_count": int(shard_count),
        "base_trajectory_count": len(selected),
        "unit_count": len(written),
        "expected_unit_count": len(selected) * len(S_POINTS),
        "feature_collector_sha256": collector_hash,
        "project_git_commit": start_commit,
        "provenance": provenance,
        "units": written,
    }
    atomic_json_create(root / "FEATURE_SHARD_COMPLETION.json", completion)
    return root


def validate_feature_shard(
    completion_path: Path, *, deep: bool = False
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    completion_path = Path(completion_path)
    completion = _load_json(completion_path)
    root = completion_path.parent
    if completion.get("schema") != SHARD_SCHEMA or completion.get("status") != "COMPLETE":
        raise FeatureRecollectionError(f"invalid feature shard completion: {completion_path}")
    protocol = validate_protocol_binding(root, completion.get("protocol_sha256"))
    if protocol["sha256"] != completion.get("protocol_sha256"):
        raise FeatureRecollectionError("feature shard protocol mismatch")
    inventory_path = Path(str(completion.get("inventory_manifest", "")))
    if sha256_file(inventory_path) != completion.get("inventory_manifest_sha256"):
        raise FeatureRecollectionError("feature shard inventory hash mismatch")
    bases, inventory = base_records(inventory_path)
    if inventory.get("records_sha256") != completion.get("inventory_records_sha256"):
        raise FeatureRecollectionError("feature shard inventory-record hash mismatch")
    index, count = int(completion["shard_index"]), int(completion["shard_count"])
    expected = {
        feature_unit_id(str(row["video_id"]), int(row["base_seed"]), progress)
        for row in assigned_bases(bases, index, count)
        for progress in S_POINTS
    }
    listed = completion.get("units", [])
    if len(listed) != len(expected) or int(completion.get("unit_count", -1)) != len(expected):
        raise FeatureRecollectionError("partial feature shard cardinality")
    if {str(row.get("unit_id")) for row in listed} != expected:
        raise FeatureRecollectionError("feature shard assignment/identity mismatch")
    manifests: list[dict[str, Any]] = []
    for row in listed:
        unit_root = root / "units" / str(row["unit_id"])
        if sha256_file(unit_root / "COMPLETED.json") != row.get("completion_sha256"):
            raise FeatureRecollectionError(f"feature unit completion hash mismatch: {unit_root}")
        manifest = validate_feature_unit(unit_root, deep=deep)
        if sha256_file(unit_root / "manifest.json") != row.get("manifest_sha256"):
            raise FeatureRecollectionError(f"feature manifest list hash mismatch: {unit_root}")
        if sha256_file(unit_root / "arrays.npz") != row.get("arrays_sha256"):
            raise FeatureRecollectionError(f"feature arrays list hash mismatch: {unit_root}")
        if manifest.get("protocol_sha256") != protocol["sha256"]:
            raise FeatureRecollectionError(f"feature unit protocol mismatch: {unit_root}")
        manifests.append(manifest)
    return completion, manifests


def merge_feature_shards(completion_paths: Sequence[Path], out_dir: Path) -> Path:
    if not completion_paths:
        raise FeatureRecollectionError("no feature shard completions supplied")
    validated = [validate_feature_shard(path, deep=False) for path in completion_paths]
    counts = {int(item[0]["shard_count"]) for item in validated}
    if len(counts) != 1:
        raise FeatureRecollectionError("feature shards use inconsistent shard counts")
    shard_count = counts.pop()
    indices = [int(item[0]["shard_index"]) for item in validated]
    if sorted(indices) != list(range(shard_count)):
        raise FeatureRecollectionError(f"partial/duplicate feature shard set: {sorted(indices)}")
    invariant = (
        "protocol_sha256",
        "inventory_manifest_sha256",
        "inventory_records_sha256",
        "feature_collector_sha256",
        "project_git_commit",
        "lineage_gate",
    )
    first = validated[0][0]
    for completion, _ in validated[1:]:
        for key in invariant:
            if completion.get(key) != first.get(key):
                raise FeatureRecollectionError(f"feature shard provenance mismatch: {key}")
    rows: list[dict[str, Any]] = []
    identities: set[tuple[str, int, float]] = set()
    for path, (completion, manifests) in zip(completion_paths, validated):
        root = Path(path).parent
        for manifest in manifests:
            identity = manifest["identity"]
            key = (
                str(identity["video_id"]),
                int(identity["base_seed"]),
                float(identity["progress"]),
            )
            if key in identities:
                raise FeatureRecollectionError(f"duplicate feature identity: {key}")
            identities.add(key)
            unit_root = root / "units" / manifest["unit_id"]
            rows.append(
                {
                    **identity,
                    "unit_id": manifest["unit_id"],
                    "unit_root": str(unit_root.resolve()),
                    "manifest_sha256": sha256_file(unit_root / "manifest.json"),
                    "arrays_path": str((unit_root / "arrays.npz").resolve()),
                    "arrays_sha256": manifest["arrays_file_sha256"],
                    "arrays_bytes": manifest["arrays_file_bytes"],
                    "base_final_audio_sha256": manifest["base_final_identity"][
                        "bank_audio_sha256"
                    ],
                    "shard_index": int(completion["shard_index"]),
                }
            )
    rows.sort(key=lambda row: (row["video_id"], row["base_seed"], row["progress"]))
    canonical = bool(_load_json(Path(str(first["inventory_manifest"]))).get("canonical_b2"))
    expected_count = EXPECTED_UNITS if canonical else sum(
        int(item[0]["expected_unit_count"]) for item in validated
    )
    if len(rows) != expected_count:
        raise FeatureRecollectionError(
            f"feature merge cardinality {len(rows)} != {expected_count}"
        )
    out_dir = Path(out_dir)
    try:
        out_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise FeatureRecollectionError(f"refusing to overwrite feature merge {out_dir}") from exc
    manifest_path = out_dir / "FEATURE_RECOLLECTION_MANIFEST.jsonl"
    atomic_jsonl_create(manifest_path, rows)
    completion = {
        "schema": MERGE_SCHEMA,
        "status": "COMPLETE",
        "canonical_b2": canonical,
        "unit_count": len(rows),
        "base_trajectory_count": len({(row["video_id"], row["base_seed"]) for row in rows}),
        "progress_points": sorted({float(row["progress"]) for row in rows}),
        "video_count": len({str(row["video_id"]) for row in rows}),
        "manifest": manifest_path.name,
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_bytes": manifest_path.stat().st_size,
        "input_shards": [
            {
                "shard_index": int(item[0]["shard_index"]),
                "completion": str(Path(path).resolve()),
                "completion_sha256": sha256_file(path),
                "unit_count": int(item[0]["unit_count"]),
            }
            for path, item in sorted(
                zip(completion_paths, validated), key=lambda pair: int(pair[1][0]["shard_index"])
            )
        ],
        **{key: first[key] for key in invariant},
    }
    atomic_json_create(out_dir / "FEATURE_RECOLLECTION_COMPLETION.json", completion)
    return out_dir


__all__ = [
    "CFG",
    "EXPECTED_UNITS",
    "FeatureRecollectionError",
    "MERGE_SCHEMA",
    "SHARD_SCHEMA",
    "S_POINTS",
    "UNIT_SCHEMA",
    "assigned_bases",
    "base_records",
    "collect_feature_shard",
    "feature_unit_id",
    "merge_feature_shards",
    "rng_for",
    "validate_feature_shard",
    "validate_feature_unit",
    "validate_lineage_gate",
    "write_feature_unit",
]
