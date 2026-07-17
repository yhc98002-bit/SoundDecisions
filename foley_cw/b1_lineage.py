"""Fail-closed same-forward lineage pilot for the B-1 internal-readout repair.

This module deliberately stops at the five-clip engineering gate.  It creates
canonical replay packets, captures one native ``net.ode_wrapper`` evaluation,
and freezes/applies a calibration-only numerical tolerance.  It does *not*
collect the 816-trajectory B2 feature bank and does not fit a probe.

The load-bearing rule is that both pooled paths operate on the same native
fp32 post-block tensor in the same capture.  Quantize-then-reduce versus
reduce-then-quantize is recorded as a forbidden gating comparison and is never
used as identity evidence.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import platform
import re
import socket
import subprocess
import sys
import uuid
import zlib
from contextlib import contextmanager
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


CALIBRATION_CLIPS: tuple[str, ...] = ("3780", "1813", "3112", "1048")
HELDOUT_CLIP = "1002"
S_POINTS: tuple[float, ...] = (0.05, 0.15, 0.25, 0.35, 0.45, 0.60, 0.75, 0.90)
CONDITION_FIELDS: tuple[str, ...] = (
    "clip_f", "sync_f", "text_f", "clip_f_c", "text_f_c",
)
PASS_ROLES: tuple[str, ...] = ("conditional", "empty")
PACKET_SCHEMA = "sounddecisions_b1_canonical_replay_packet_v1"
REPLAY_SCHEMA = "sounddecisions_b1_same_forward_replay_v1"
TOLERANCE_SCHEMA = "sounddecisions_b1_same_forward_tolerance_v1"
HELDOUT_SCHEMA = "sounddecisions_b1_same_forward_heldout_v1"
ATTEMPT_SCHEMA = "sounddecisions_b1_immutable_attempt_v1"
SELECTION_SCHEMA = "sounddecisions_b1_pilot_selection_v1"
EQUIVALENT_COMPARISON = "same_native_fp32_mean_equivalent"
INPUT_COMPARISON = "packet_to_device_fp32_identity"
TIME_COMPARISON = "packet_to_device_time_identity"
FORBIDDEN_GATING_COMPARISON = "mean_fp16_tokens_vs_fp16_mean_fp32_tokens"
ATOMIC_READBACK_COMPARISON = "in_memory_vs_atomic_readback"
TWEEDIE_RECOMPUTE_COMPARISON = "stored_tweedie_vs_same_order_recompute"
ATTENTION_SUMMARY_COMPARISON = "stored_attention_summary_vs_contract_recompute"
CROSS_REPLAY_COMPARISON = "exact_packet_cross_replay"
PROTOCOL_SCHEMA = "sounddecisions_non_human_closure_protocol_v1"
PROTOCOL_BINDING_SCHEMA = "sounddecisions_b1_protocol_binding_v1"

_PROTOCOL_GATE_TEXT: tuple[str, ...] = (
    "same-native-tensor original fp32 pooling vs repaired fp32 pooling",
    "in-memory tensor vs atomic readback",
    "stored Tweedie vs same-order reducer recomputation",
    "stored derived attention summary vs same-contract reducer recomputation",
    "corresponding exact-packet replay outputs across registered devices/repeats",
)

_ATTEMPT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_UNIT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,191}$")


class LineageError(RuntimeError):
    """Base class for fail-closed B-1 lineage errors."""


class ImmutableArtifactError(LineageError):
    """An immutable attempt or unit already exists."""


class ArtifactValidationError(LineageError):
    """An artifact is partial, corrupt, conflicting, or schema-invalid."""


class HeldoutLeakageError(LineageError):
    """Held-out clip 1002 entered calibration evidence."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(array))
    header = canonical_json_bytes({"dtype": arr.dtype.str, "shape": list(arr.shape)})
    return sha256_bytes(header + b"\0" + arr.tobytes(order="C"))


def describe_array(array: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(array)
    try:
        finite_mask = np.isfinite(arr)
    except TypeError as exc:
        raise ArtifactValidationError(
            f"scientific arrays must have a finite-checkable numeric dtype, got {arr.dtype}"
        ) from exc
    finite_count = int(np.count_nonzero(finite_mask))
    if finite_count != int(arr.size):
        raise ArtifactValidationError(
            f"nonfinite scientific tensor: dtype={arr.dtype}, shape={arr.shape}, "
            f"finite={finite_count}/{arr.size}"
        )
    return {
        "dtype": arr.dtype.str,
        "shape": list(arr.shape),
        "sha256": array_sha256(arr),
        "nbytes": int(arr.nbytes),
        "finite": True,
        "finite_count": finite_count,
        "element_count": int(arr.size),
    }


def _protocol_contract(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    if payload.get("schema") != PROTOCOL_SCHEMA:
        raise ArtifactValidationError(
            f"protocol schema {payload.get('schema')!r} != {PROTOCOL_SCHEMA!r}"
        )
    pilot = payload.get("lineage", {}).get("pilot", {})
    required = {
        "calibration_clips": list(CALIBRATION_CLIPS),
        "heldout_clip": HELDOUT_CLIP,
        "independent_index": 12,
        "seed": 0,
        "cfg": 1.0,
        "schedule": "sqrt_down",
        "steps": 20,
        "duration_seconds": 8.0,
        "progress": list(S_POINTS),
        "tolerance_quantile": 0.999,
        "quantile_method": "higher",
        "safety_factor": 2.0,
        "epsilon_floor": None,
        "gating_comparisons": list(_PROTOCOL_GATE_TEXT),
    }
    for key, expected in required.items():
        if pilot.get(key) != expected:
            raise ArtifactValidationError(
                f"frozen protocol lineage.pilot.{key}={pilot.get(key)!r}; expected {expected!r}"
            )
    assets = pilot.get("asset_contract")
    if not isinstance(assets, dict):
        raise ArtifactValidationError("frozen protocol lacks lineage.pilot.asset_contract")
    for key in (
        "mmaudio_git_commit", "mmaudio_backend_sha256", "mmaudio_weights",
        "panns_checkpoint_sha256", "panns_labels_sha256", "panns_loader_sha256",
        "clip_hf_model", "clip_hf_revision", "clip_snapshot_files", "offline_environment",
    ):
        if not assets.get(key):
            raise ArtifactValidationError(f"frozen protocol asset contract lacks {key}")
    return pilot


def load_protocol(protocol_path: Path, expected_sha256: str) -> dict[str, Any]:
    """Load and enforce the immutable exploratory protocol by caller-supplied hash."""
    protocol_path = Path(protocol_path)
    if not expected_sha256 or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ArtifactValidationError("a lowercase 64-hex --protocol-sha256 is mandatory")
    raw = protocol_path.read_bytes()
    observed = sha256_bytes(raw)
    if observed != expected_sha256:
        raise ArtifactValidationError(
            f"protocol hash mismatch: observed {observed}, required {expected_sha256}"
        )
    try:
        payload = json.loads(raw)
    except Exception as exc:
        raise ArtifactValidationError(f"invalid frozen protocol JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ArtifactValidationError("frozen protocol must be a JSON object")
    pilot = _protocol_contract(payload)
    return {
        "path": str(protocol_path.resolve()),
        "sha256": observed,
        "raw": raw,
        "payload": payload,
        "pilot": pilot,
        "pilot_sha256": sha256_bytes(canonical_json_bytes(pilot)),
    }


def bind_protocol(root: Path, binding: Mapping[str, Any]) -> None:
    """Copy exact protocol bytes into an attempt so validation is path-independent."""
    raw = bytes(binding["raw"])
    _write_new_bytes(Path(root) / "PROTOCOL.json", raw)
    _write_new_json(Path(root) / "PROTOCOL_BINDING.json", {
        "schema": PROTOCOL_BINDING_SCHEMA,
        "protocol_schema": PROTOCOL_SCHEMA,
        "protocol_sha256": binding["sha256"],
        "lineage_pilot_sha256": binding["pilot_sha256"],
        "source_path": binding["path"],
    })


def validate_protocol_binding(root: Path, expected_sha256: str | None = None) -> dict[str, Any]:
    root = Path(root)
    raw_path = root / "PROTOCOL.json"
    binding_path = root / "PROTOCOL_BINDING.json"
    if not raw_path.is_file() or not binding_path.is_file():
        raise ArtifactValidationError(f"attempt lacks frozen protocol binding: {root}")
    raw = raw_path.read_bytes()
    observed = sha256_bytes(raw)
    binding = _load_json(binding_path)
    if binding.get("schema") != PROTOCOL_BINDING_SCHEMA:
        raise ArtifactValidationError(f"protocol binding schema mismatch: {root}")
    if binding.get("protocol_sha256") != observed:
        raise ArtifactValidationError(f"bound protocol hash mismatch: {root}")
    if expected_sha256 is not None and observed != expected_sha256:
        raise ArtifactValidationError(
            f"attempt protocol {observed} != required protocol {expected_sha256}"
        )
    try:
        payload = json.loads(raw)
    except Exception as exc:
        raise ArtifactValidationError(f"bound protocol is invalid JSON: {exc}") from exc
    pilot = _protocol_contract(payload)
    if binding.get("lineage_pilot_sha256") != sha256_bytes(canonical_json_bytes(pilot)):
        raise ArtifactValidationError(f"bound lineage pilot contract hash mismatch: {root}")
    return {
        "sha256": observed, "payload": payload, "pilot": pilot, "raw": raw,
        "path": str(raw_path.resolve()),
        "pilot_sha256": sha256_bytes(canonical_json_bytes(pilot)),
    }


def create_bound_attempt(output_root: Path, stage: str, attempt_id: str,
                         protocol: Mapping[str, Any]) -> Path:
    root = create_attempt(output_root, stage, attempt_id)
    bind_protocol(root, protocol)
    return root


def _safe_component(value: str, pattern: re.Pattern[str], kind: str) -> str:
    value = str(value)
    if not pattern.fullmatch(value):
        raise ValueError(f"unsafe {kind} {value!r}")
    return value


def unit_id(clip_id: str, s: float, repeat_index: int | None = None) -> str:
    base = f"clip{clip_id}__s{s:.2f}"
    if repeat_index is not None:
        base += f"__r{int(repeat_index):03d}"
    return _safe_component(base, _UNIT_RE, "unit id")


def phase1_rng(seed: int, *parts: Any) -> np.random.Generator:
    """Exact Phase-1 RNG contract used by ``phase1_commitment.py``."""
    entropy = [int(seed)] + [zlib.crc32(str(part).encode()) for part in parts]
    return np.random.default_rng(np.random.SeedSequence(entropy))


def phase1_identity(clip_id: str) -> dict[str, Any]:
    return {
        "cfg": 1.0,
        "seed": 0,
        "role": "independent",
        "independent_j": 12,
        "rng_parts": [str(clip_id), "ind", 12],
        "schedule": "sqrt_down",
        "num_steps": 20,
        "duration_seconds": 8.0,
        "progress_points": list(S_POINTS),
        "variant": "small_16k",
        "full_precision": True,
    }


def _write_new_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise ImmutableArtifactError(f"refusing to overwrite {path}") from exc


def _write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    _write_new_bytes(path, json.dumps(payload, indent=2, sort_keys=True,
                                      allow_nan=False).encode("utf-8") + b"\n")


def _write_new_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            np.savez_compressed(handle, **{k: np.asarray(v) for k, v in arrays.items()})
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise ImmutableArtifactError(f"refusing to overwrite {path}") from exc


def _file_inventory(root: Path, *, exclude_completion: bool = True) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        # Exclude only *this root's* completion journal.  Nested unit completion
        # journals are parents of the attempt and must themselves be hashed.
        if exclude_completion and rel == "COMPLETED.json":
            continue
        if path.name.endswith(".tmp"):
            raise ArtifactValidationError(f"temporary/partial file present: {path}")
        items.append({"path": rel, "size": path.stat().st_size, "sha256": sha256_file(path)})
    return items


def create_attempt(output_root: Path, stage: str, attempt_id: str) -> Path:
    attempt = _safe_component(attempt_id, _ATTEMPT_RE, "attempt id")
    stage_name = _safe_component(stage, _ATTEMPT_RE, "stage")
    root = Path(output_root) / stage_name / attempt
    try:
        root.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise ImmutableArtifactError(f"attempt already exists: {root}") from exc
    return root


def finish_attempt(root: Path, stage: str, expected_units: int) -> dict[str, Any]:
    if (root / "COMPLETED.json").exists():
        raise ImmutableArtifactError(f"attempt already completed: {root}")
    inventory = _file_inventory(root)
    unit_completions = sorted(root.glob("units/*/COMPLETED.json"))
    if len(unit_completions) != int(expected_units):
        raise ArtifactValidationError(
            f"cannot complete {stage}: {len(unit_completions)} completed units, "
            f"expected {expected_units}"
        )
    payload = {
        "schema": ATTEMPT_SCHEMA,
        "stage": stage,
        "status": "COMPLETE",
        "completed_utc": utc_now(),
        "expected_units": int(expected_units),
        "completed_units": len(unit_completions),
        "inventory": inventory,
        "inventory_sha256": sha256_bytes(canonical_json_bytes(inventory)),
    }
    _write_new_json(root / "COMPLETED.json", payload)
    return payload


def _finish_unit(unit_root: Path, stage: str, identity: Mapping[str, Any]) -> dict[str, Any]:
    inventory = _file_inventory(unit_root)
    payload = {
        "schema": ATTEMPT_SCHEMA,
        "stage": stage,
        "status": "COMPLETE",
        "identity": dict(identity),
        "completed_utc": utc_now(),
        "inventory": inventory,
        "inventory_sha256": sha256_bytes(canonical_json_bytes(inventory)),
    }
    _write_new_json(unit_root / "COMPLETED.json", payload)
    return payload


def _git_head(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL, timeout=10,
        ).strip()
    except Exception:
        return None


def _git_dirty(path: Path) -> tuple[bool | None, list[str]]:
    try:
        output = subprocess.check_output(
            ["git", "-C", str(path), "status", "--porcelain"], text=True,
            stderr=subprocess.DEVNULL, timeout=30,
        )
        rows = [line for line in output.splitlines() if line]
        return bool(rows), rows
    except Exception:
        return None, []


def _dependency_hashes(repo_root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for name in ("pyproject.toml", "requirements.txt", "environment.yml"):
        path = repo_root / name
        if path.is_file():
            result[name] = sha256_file(path)
    return result


def require_offline_environment(asset_contract: Mapping[str, Any]) -> dict[str, Any]:
    required = asset_contract.get("offline_environment", {})
    observed = {name: os.environ.get(name) for name in sorted(required) if name != "HF_HOME_required"}
    for name, expected in required.items():
        if name == "HF_HOME_required":
            continue
        if observed.get(name) != str(expected):
            raise ArtifactValidationError(
                f"offline guard requires {name}={expected!r}; observed {observed.get(name)!r}"
            )
    hf_home_raw = os.environ.get("HF_HOME")
    if required.get("HF_HOME_required") and not hf_home_raw:
        raise ArtifactValidationError("offline guard requires explicit HF_HOME")
    if not hf_home_raw:
        raise ArtifactValidationError("HF_HOME must name the pinned local cache")
    hf_home = Path(hf_home_raw).resolve()
    if not hf_home.is_dir():
        raise ArtifactValidationError(f"pinned HF_HOME does not exist: {hf_home}")
    observed["HF_HOME"] = str(hf_home)
    return observed


def _snapshot_observation(hf_home: Path, asset_contract: Mapping[str, Any]) -> dict[str, Any]:
    model = str(asset_contract["clip_hf_model"])
    revision = str(asset_contract["clip_hf_revision"])
    model_dir = "models--" + model.replace("/", "--")
    snapshot = hf_home / "hub" / model_dir / "snapshots" / revision
    if not snapshot.is_dir():
        raise ArtifactValidationError(f"pinned offline CLIP snapshot missing: {snapshot}")
    files: dict[str, Any] = {}
    expected_files = asset_contract["clip_snapshot_files"]
    for rel, expected_blob in sorted(expected_files.items()):
        path = snapshot / rel
        if not path.is_file():
            raise ArtifactValidationError(f"pinned CLIP snapshot file missing: {path}")
        resolved = path.resolve()
        if resolved.name != expected_blob:
            raise ArtifactValidationError(
                f"CLIP blob mismatch for {rel}: {resolved.name} != {expected_blob}"
            )
        content_sha = sha256_file(resolved)
        if len(str(expected_blob)) == 64 and content_sha != expected_blob:
            raise ArtifactValidationError(
                f"CLIP content hash mismatch for {rel}: {content_sha} != {expected_blob}"
            )
        files[rel] = {
            "blob_id": resolved.name,
            "content_sha256": content_sha,
            "size": resolved.stat().st_size,
        }
    return {
        "model": model,
        "revision": revision,
        "snapshot": str(snapshot),
        "files": files,
        "manifest_sha256": sha256_bytes(canonical_json_bytes(files)),
    }


def verify_asset_contract(mmaudio_root: Path, weights_dir: Path,
                          protocol_pilot: Mapping[str, Any]) -> dict[str, Any]:
    """Hash every locally consumed model/code asset before any model construction."""
    mmaudio_root = Path(mmaudio_root).resolve()
    weights_dir = Path(weights_dir).resolve()
    contract = protocol_pilot["asset_contract"]
    offline = require_offline_environment(contract)
    observed: dict[str, Any] = {
        "mmaudio_root": str(mmaudio_root),
        "weights_dir": str(weights_dir),
        "offline_environment": offline,
    }
    commit = _git_head(mmaudio_root)
    dirty, dirty_rows = _git_dirty(mmaudio_root)
    if commit != contract["mmaudio_git_commit"] or dirty:
        raise ArtifactValidationError(
            f"MMAudio code mismatch: commit={commit}, dirty={dirty}, "
            f"required={contract['mmaudio_git_commit']}"
        )
    observed["mmaudio_git_commit"] = commit
    observed["mmaudio_dirty"] = dirty
    observed["mmaudio_dirty_rows"] = dirty_rows
    model_hashes: dict[str, str] = {}
    for rel, expected in sorted(contract["mmaudio_weights"].items()):
        path = mmaudio_root / rel
        if not path.is_file():
            raise ArtifactValidationError(f"required MMAudio asset missing: {path}")
        got = sha256_file(path)
        if got != expected:
            raise ArtifactValidationError(f"MMAudio asset hash mismatch: {rel}: {got} != {expected}")
        model_hashes[rel] = got
    observed["mmaudio_weights"] = model_hashes
    local_contract = {
        "panns_checkpoint_sha256": weights_dir / "Cnn14_16k_mAP=0.438.pth",
        "panns_labels_sha256": weights_dir / "class_labels_indices.csv",
        "panns_loader_sha256": Path(__file__).resolve().parent / "measurers_panns_cnn14.py",
        "mmaudio_backend_sha256": Path(__file__).resolve().parent / "mmaudio_backend.py",
    }
    local_hashes: dict[str, str] = {}
    for key, path in local_contract.items():
        if not path.is_file():
            raise ArtifactValidationError(f"required local code/asset missing: {path}")
        got = sha256_file(path)
        if got != contract[key]:
            raise ArtifactValidationError(f"local asset hash mismatch: {key}: {got} != {contract[key]}")
        local_hashes[key] = got
    observed["local_hashes"] = local_hashes
    observed["clip_snapshot"] = _snapshot_observation(Path(offline["HF_HOME"]), contract)
    observed["contract_sha256"] = sha256_bytes(canonical_json_bytes(contract))
    return observed


def environment_provenance(device: str, mmaudio_root: Path | None = None,
                           *, protocol_sha256: str | None = None,
                           asset_observation: Mapping[str, Any] | None = None) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    dirty, dirty_rows = _git_dirty(repo_root)
    info: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "executable": sys.executable,
        "device_argument": str(device),
        "numpy": np.__version__,
        "project_git_commit": _git_head(repo_root),
        "project_git_dirty": dirty,
        "project_git_dirty_rows": dirty_rows,
        "collector_sha256": sha256_file(Path(__file__).resolve()),
        "collector_schema_sha256": sha256_bytes(canonical_json_bytes({
            "packet": PACKET_SCHEMA,
            "replay": REPLAY_SCHEMA,
            "tolerance": TOLERANCE_SCHEMA,
            "heldout": HELDOUT_SCHEMA,
            "condition_fields": list(CONDITION_FIELDS),
            "pass_roles": list(PASS_ROLES),
            "gate_comparisons": list(_PROTOCOL_GATE_TEXT),
        })),
        "protocol_sha256": protocol_sha256,
        "dependency_hashes": _dependency_hashes(repo_root),
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    if asset_observation is not None:
        info["asset_observation"] = dict(asset_observation)
    try:
        import torch
        info.update({
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "cudnn": torch.backends.cudnn.version(),
            "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
            "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
            "autocast_enabled": bool(torch.is_autocast_enabled()),
        })
        if str(device).startswith("cuda") and torch.cuda.is_available():
            index = torch.device(device).index
            index = torch.cuda.current_device() if index is None else index
            info["cuda_device_index"] = int(index)
            info["cuda_device_name"] = torch.cuda.get_device_name(index)
            props = torch.cuda.get_device_properties(index)
            # PyTorch exposes this as a private ``_CUuuid`` object on some
            # CUDA builds.  Preserve its canonical printable identity while
            # keeping the provenance document strictly JSON serializable.
            device_uuid = getattr(props, "uuid", None)
            info["cuda_device_uuid"] = (
                None if device_uuid is None else str(device_uuid)
            )
            info["cuda_total_memory"] = int(props.total_memory)
            try:
                smi = subprocess.check_output([
                    "nvidia-smi", "--query-gpu=index,uuid,name,driver_version,memory.total,memory.free",
                    "--format=csv,noheader,nounits",
                ], text=True, stderr=subprocess.STDOUT, timeout=30)
                info["nvidia_smi_inventory"] = [row.strip() for row in smi.splitlines() if row.strip()]
            except Exception as exc:
                info["nvidia_smi_error"] = f"{type(exc).__name__}: {exc}"
    except Exception as exc:  # pragma: no cover - provenance must not hide the core error
        info["torch_probe_error"] = f"{type(exc).__name__}: {exc}"
    if mmaudio_root is not None:
        info["mmaudio_root"] = str(Path(mmaudio_root).resolve())
        info["mmaudio_git_commit"] = _git_head(Path(mmaudio_root))
    return info


def _require_asset_roots(mmaudio_root: Path, weights_dir: Path, clips_root: Path,
                         protocol_pilot: Mapping[str, Any] | None = None) -> dict[str, Any] | None:
    mmaudio_root = Path(mmaudio_root)
    weights_dir = Path(weights_dir)
    clips_root = Path(clips_root)
    required = [
        mmaudio_root / "mmaudio" / "model" / "networks.py",
        mmaudio_root / "weights" / "mmaudio_small_16k.pth",
        weights_dir / "Cnn14_16k_mAP=0.438.pth",
        weights_dir / "class_labels_indices.csv",
        clips_root,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("required local/offline assets missing: " + ", ".join(missing))
    if protocol_pilot is not None:
        return verify_asset_contract(mmaudio_root, weights_dir, protocol_pilot)
    return None


def create_selection_attempt(
    output_root: Path,
    attempt_id: str,
    *,
    mmaudio_root: Path,
    weights_dir: Path,
    clips_root: Path,
    protocol_path: Path,
    protocol_sha256: str,
) -> Path:
    """Freeze the outcome-independent four-calibration/one-held-out pilot selection."""
    protocol = load_protocol(protocol_path, protocol_sha256)
    asset_observation = _require_asset_roots(
        mmaudio_root, weights_dir, clips_root, protocol["pilot"]
    )
    root = create_bound_attempt(output_root, "selection", attempt_id, protocol)
    selected: list[dict[str, Any]] = []
    for role, clips in (("calibration", CALIBRATION_CLIPS), ("heldout", (HELDOUT_CLIP,))):
        for clip_id in clips:
            video = Path(clips_root) / f"{clip_id}.mp4"
            if not video.is_file():
                raise FileNotFoundError(video)
            selected.append({
                "clip_id": clip_id,
                "role": role,
                "video_path": str(video.resolve()),
                "video_sha256": sha256_file(video),
            })
    payload = {
        "schema": SELECTION_SCHEMA,
        "created_utc": utc_now(),
        "selection_rule": "fixed_outcome_independent_b1_identity_pilot_v1",
        "clips": selected,
        "progress_points": list(S_POINTS),
        "counts": {
            "calibration_clips": len(CALIBRATION_CLIPS),
            "heldout_clips": 1,
            "calibration_packets": len(CALIBRATION_CLIPS) * len(S_POINTS),
            "heldout_packets": len(S_POINTS),
            "total_packets": (len(CALIBRATION_CLIPS) + 1) * len(S_POINTS),
        },
        "phase1_identity": phase1_identity("<clip_id>"),
        "protocol_sha256": protocol["sha256"],
        "asset_observation": asset_observation,
        "asset_roots": {
            "mmaudio_root": str(Path(mmaudio_root).resolve()),
            "weights_dir": str(Path(weights_dir).resolve()),
            "clips_root": str(Path(clips_root).resolve()),
        },
    }
    _write_new_json(root / "selection.json", payload)
    finish_attempt(root, "selection", expected_units=0)
    return root


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        raise ArtifactValidationError(f"invalid JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ArtifactValidationError(f"expected JSON object in {path}")
    return value


def _validate_inventory(root: Path, completion: Mapping[str, Any], *, unit: bool = False) -> None:
    expected = completion.get("inventory")
    if not isinstance(expected, list):
        raise ArtifactValidationError(f"completion inventory missing in {root}")
    actual = _file_inventory(root)
    if unit:
        actual = [item for item in actual if item["path"] != "COMPLETED.json"]
    if actual != expected:
        raise ArtifactValidationError(f"inventory/hash mismatch in {root}")
    digest = sha256_bytes(canonical_json_bytes(expected))
    if completion.get("inventory_sha256") != digest:
        raise ArtifactValidationError(f"inventory digest mismatch in {root}")


def _load_npz_verified(path: Path, descriptors: Mapping[str, Any]) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != set(descriptors):
                raise ArtifactValidationError(
                    f"array key mismatch in {path}: {sorted(archive.files)} != "
                    f"{sorted(descriptors)}"
                )
            arrays = {key: np.array(archive[key], copy=True) for key in archive.files}
    except ArtifactValidationError:
        raise
    except Exception as exc:
        raise ArtifactValidationError(f"cannot load {path}: {exc}") from exc
    for key, array in arrays.items():
        if describe_array(array) != descriptors[key]:
            raise ArtifactValidationError(f"array descriptor/hash mismatch: {path}:{key}")
    return arrays


def write_packet_unit(
    attempt_root: Path,
    *,
    clip_id: str,
    role: str,
    s: float,
    arrays: Mapping[str, np.ndarray],
    parent_hashes: Mapping[str, str],
    provenance: Mapping[str, Any],
    video_sha256: str,
) -> Path:
    protocol = validate_protocol_binding(attempt_root)
    if role not in {"calibration", "heldout"}:
        raise ValueError(f"invalid pilot role {role!r}")
    expected_keys = {"x_s", "model_time"} | {
        f"conditions__{name}" for name in CONDITION_FIELDS
    } | {f"empty_conditions__{name}" for name in CONDITION_FIELDS}
    if set(arrays) != expected_keys:
        raise ArtifactValidationError(
            f"packet arrays are not complete: missing={sorted(expected_keys - set(arrays))}, "
            f"extra={sorted(set(arrays) - expected_keys)}"
        )
    x_s = np.asarray(arrays["x_s"])
    if x_s.dtype != np.float32 or x_s.ndim != 2:
        raise ArtifactValidationError("canonical x_s must be fp32 with shape (tokens, channels)")
    model_time = np.asarray(arrays["model_time"])
    if model_time.dtype != np.float32 or model_time.shape != ():
        raise ArtifactValidationError("model_time must be a scalar fp32 array")
    uid = unit_id(clip_id, s)
    root = Path(attempt_root) / "units" / uid
    try:
        root.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise ImmutableArtifactError(f"packet unit already exists: {root}") from exc
    normalized = {key: np.asarray(value) for key, value in arrays.items()}
    for value in normalized.values():
        describe_array(value)
    _write_new_npz(root / "arrays.npz", normalized)
    identity = {"clip_id": str(clip_id), "role": role, "s": float(s),
                "phase1": phase1_identity(str(clip_id))}
    manifest = {
        "schema": PACKET_SCHEMA,
        "protocol_sha256": protocol["sha256"],
        "unit_id": uid,
        "identity": identity,
        "video_sha256": video_sha256,
        "parent_hashes": dict(parent_hashes),
        "arrays_file": "arrays.npz",
        "arrays_file_sha256": sha256_file(root / "arrays.npz"),
        "arrays": {key: describe_array(value) for key, value in normalized.items()},
        "conditioning_contract": {
            "fields": list(CONDITION_FIELDS),
            "complete_conditions": True,
            "complete_empty_conditions": True,
        },
        "provenance": dict(provenance),
    }
    _write_new_json(root / "manifest.json", manifest)
    _finish_unit(root, "packets", identity)
    return root


def validate_packet_unit(root: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    root = Path(root)
    completion_path = root / "COMPLETED.json"
    if not completion_path.is_file():
        raise ArtifactValidationError(f"partial packet unit (no completion): {root}")
    completion = _load_json(completion_path)
    if completion.get("status") != "COMPLETE" or completion.get("stage") != "packets":
        raise ArtifactValidationError(f"invalid packet completion: {root}")
    _validate_inventory(root, completion, unit=True)
    manifest = _load_json(root / "manifest.json")
    attempt_root = root.parent.parent
    protocol = validate_protocol_binding(attempt_root)
    if manifest.get("schema") != PACKET_SCHEMA:
        raise ArtifactValidationError(f"wrong packet schema: {root}")
    if manifest.get("unit_id") != root.name:
        raise ArtifactValidationError(f"packet unit id conflicts with directory: {root}")
    if manifest.get("protocol_sha256") != protocol["sha256"]:
        raise ArtifactValidationError(f"packet protocol parent mismatch: {root}")
    identity = manifest.get("identity", {})
    if identity.get("role") not in {"calibration", "heldout"}:
        raise ArtifactValidationError(f"invalid packet role: {root}")
    if identity.get("clip_id") == HELDOUT_CLIP and identity.get("role") != "heldout":
        raise ArtifactValidationError("clip 1002 must be heldout")
    if manifest.get("arrays_file_sha256") != sha256_file(root / "arrays.npz"):
        raise ArtifactValidationError(f"packet npz file hash mismatch: {root}")
    arrays = _load_npz_verified(root / "arrays.npz", manifest.get("arrays", {}))
    expected = {"x_s", "model_time"} | {
        f"conditions__{name}" for name in CONDITION_FIELDS
    } | {f"empty_conditions__{name}" for name in CONDITION_FIELDS}
    if set(arrays) != expected:
        raise ArtifactValidationError(f"incomplete conditioning packet: {root}")
    if arrays["x_s"].dtype != np.float32 or arrays["model_time"].dtype != np.float32:
        raise ArtifactValidationError(f"non-fp32 replay state/time: {root}")
    return manifest, arrays


def _condition_arrays(cond: Any, prefix: str) -> dict[str, np.ndarray]:
    if is_dataclass(cond):
        available = {field.name for field in fields(cond)}
    else:
        available = set(CONDITION_FIELDS)
    if available != set(CONDITION_FIELDS):
        raise ArtifactValidationError(
            f"PreprocessedConditions fields changed: {sorted(available)}"
        )
    arrays: dict[str, np.ndarray] = {}
    for name in CONDITION_FIELDS:
        value = getattr(cond, name)
        try:
            arr = value.detach().cpu().numpy()
        except Exception as exc:
            raise ArtifactValidationError(f"conditioning tensor {name} is not persistable: {exc}")
        arrays[f"{prefix}__{name}"] = np.ascontiguousarray(arr)
    return arrays


def make_packet_attempt(
    selection_attempt: Path,
    output_root: Path,
    attempt_id: str,
    *,
    mmaudio_root: Path,
    weights_dir: Path,
    clips_root: Path,
    device: str,
    protocol_path: Path,
    protocol_sha256: str,
) -> Path:
    """Generate the five canonical j=12 trajectories and persist forty replay packets."""
    protocol = load_protocol(protocol_path, protocol_sha256)
    asset_observation = _require_asset_roots(
        mmaudio_root, weights_dir, clips_root, protocol["pilot"]
    )
    validate_attempt(
        selection_attempt, expected_stage="selection",
        expected_protocol_sha256=protocol["sha256"],
    )
    selection = _load_json(Path(selection_attempt) / "selection.json")
    expected_roots = selection.get("asset_roots", {})
    current_roots = {
        "mmaudio_root": str(Path(mmaudio_root).resolve()),
        "weights_dir": str(Path(weights_dir).resolve()),
        "clips_root": str(Path(clips_root).resolve()),
    }
    if expected_roots != current_roots:
        raise ArtifactValidationError("asset roots conflict with frozen selection")
    root = create_bound_attempt(output_root, "packets", attempt_id, protocol)

    mmaudio_root = Path(mmaudio_root).resolve()
    if str(mmaudio_root) not in sys.path:
        sys.path.insert(0, str(mmaudio_root))
    from . import score_sde
    from .mmaudio_backend import MMAudioBackend
    from .types import ScheduleSpec

    schedule = ScheduleSpec(n_steps=20, scan_points=S_POINTS, K_forks=12,
                            N_independent=16, g_kind="sqrt_down", g_value=1.0)
    integration_grid = schedule.integration_s_grid()
    for s in S_POINTS:
        if not np.any(np.isclose(integration_grid, s, rtol=0.0, atol=1e-12)):
            raise ArtifactValidationError(f"registered progress point {s} is off schedule grid")
    backend = MMAudioBackend(
        variant="small_16k", device=device, full_precision=True, cfg_strength=1.0,
        num_steps=20, duration_sec=8.0, weights_root=str(mmaudio_root),
        enable_conditions=True,
    )
    selection_hash = sha256_file(Path(selection_attempt) / "selection.json")
    selection_completion_hash = sha256_file(Path(selection_attempt) / "COMPLETED.json")
    model_weight = mmaudio_root / "weights" / "mmaudio_small_16k.pth"
    provenance = environment_provenance(
        device, mmaudio_root, protocol_sha256=protocol["sha256"],
        asset_observation=asset_observation,
    )
    provenance["model_weight_sha256"] = sha256_file(model_weight)
    count = 0
    for record in selection["clips"]:
        clip_id = str(record["clip_id"])
        video = Path(clips_root) / f"{clip_id}.mp4"
        if sha256_file(video) != record["video_sha256"]:
            raise ArtifactValidationError(f"video changed after selection: {video}")
        cond = backend.make_video_cond(str(video), video_id=clip_id)
        trajectory = score_sde.generate_trajectory(
            backend, cond, schedule, phase1_rng(0, clip_id, "ind", 12),
            alpha=0.0, record_points=S_POINTS,
        )
        cond_arrays = _condition_arrays(cond.conditions, "conditions")
        empty_arrays = _condition_arrays(cond.empty_conditions, "empty_conditions")
        for s in S_POINTS:
            arrays: dict[str, np.ndarray] = {
                "x_s": np.ascontiguousarray(trajectory["states"][s], dtype=np.float32),
                "model_time": np.asarray(backend.s_to_t.s_to_t(s), dtype=np.float32),
                **cond_arrays,
                **empty_arrays,
            }
            write_packet_unit(
                root, clip_id=clip_id, role=record["role"], s=s, arrays=arrays,
                parent_hashes={
                    "selection_manifest_sha256": selection_hash,
                    "selection_completion_sha256": selection_completion_hash,
                },
                provenance=provenance,
                video_sha256=record["video_sha256"],
            )
            count += 1
    expected = (len(CALIBRATION_CLIPS) + 1) * len(S_POINTS)
    if count != expected:
        raise ArtifactValidationError(f"packet cardinality {count} != {expected}")
    finish_attempt(root, "packets", expected_units=expected)
    return root


def _relative_l2(lhs: np.ndarray, rhs: np.ndarray) -> float:
    a = np.asarray(lhs, dtype=np.float64)
    b = np.asarray(rhs, dtype=np.float64)
    denom = max(float(np.linalg.norm(a.ravel())), np.finfo(np.float64).tiny)
    return float(np.linalg.norm((a - b).ravel()) / denom)


def _max_abs(lhs: np.ndarray, rhs: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(lhs, dtype=np.float64) -
                               np.asarray(rhs, dtype=np.float64))))


def _comparison_metrics(lhs: np.ndarray, rhs: np.ndarray) -> dict[str, float]:
    a = np.asarray(lhs, dtype=np.float64).ravel()
    b = np.asarray(rhs, dtype=np.float64).ravel()
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ArtifactValidationError("comparison operands must be shape-equal and finite")
    delta = a - b
    denom = max(float(np.linalg.norm(a)), 1e-12)
    norm_a, norm_b = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    cosine = 1.0 if norm_a == 0.0 and norm_b == 0.0 else (
        float(np.dot(a, b) / (norm_a * norm_b)) if norm_a and norm_b else 0.0
    )
    return {
        "relative_l2": float(np.linalg.norm(delta) / denom),
        "absolute_l2": float(np.linalg.norm(delta)),
        "max_abs": float(np.max(np.abs(delta))) if delta.size else 0.0,
        "cosine": cosine,
        "exact_fraction": float(np.mean(a == b)) if a.size else 1.0,
    }


def _repaired_pool_same_contract(native: Any) -> Any:
    """Independent implementation of the historical reduce-then-fp32 contract."""
    import torch
    detached = native.detach()
    reduced_native_dtype = torch.mean(detached, dim=1, dtype=detached.dtype)
    return reduced_native_dtype[0].to(dtype=torch.float32)


def _attention_summary(array: np.ndarray) -> np.ndarray:
    """Registered reducer: float64 head/query mean, cast once to fp32."""
    value = np.asarray(array, dtype=np.float32)
    if value.ndim != 4:
        raise ArtifactValidationError(
            f"attention clip map must be (batch,heads,latent_query,clip_key), got {value.shape}"
        )
    return np.mean(value, axis=(1, 2), dtype=np.float64).astype(np.float32)


class SameForwardCapture:
    """Capture native block/attention tensors from one ``ode_wrapper`` evaluation.

    ``attention_module.attention`` is temporarily wrapped.  The wrapper records
    the exact Q/K/V passed to the actual attention implementation and its actual
    returned output.  A softmax map recomputed from Q/K is explicitly labelled as
    derived rather than an output exposed by the fused SDPA kernel.
    """

    def __init__(self, net: Any, attention_module: Any) -> None:
        self.net = net
        self.attention_module = attention_module
        self.joint = list(net.joint_blocks)
        self.fused = list(net.fused_blocks)
        if not self.joint or not self.fused:
            raise ArtifactValidationError("MMAudio block topology is empty")
        self.selected_sites = {
            "joint.0", f"joint.{len(self.joint) - 1}", f"fused.{len(self.fused) - 1}"
        }
        self._handles: list[Any] = []
        self._original_attention: Any = None
        self._original_predict_flow: Any = None
        self._active_site: str | None = None
        self._pass_index = -1
        self._active_lengths: dict[str, int] = {}
        self._nonce = ""
        self._arrays: dict[str, np.ndarray] = {}
        self._tokens: list[dict[str, Any]] = []
        self._attention: list[dict[str, Any]] = []
        self._passes: list[dict[str, Any]] = []

    @staticmethod
    def _key(kind: str, role: str, site: str) -> str:
        return f"{kind}__{role}__{site.replace('.', '_')}"

    def _pre_hook(self, site: str):
        def hook(_module: Any, args: Any) -> None:
            import torch
            if self._pass_index < 0 or self._pass_index >= len(PASS_ROLES):
                raise ArtifactValidationError(
                    f"unexpected predict_flow pass index {self._pass_index} at {site}"
                )
            self._active_site = site
            if site == "joint.0":
                if len(args) < 5:
                    raise ArtifactValidationError(
                        "joint.0 did not expose latent, clip, text, global, extended inputs"
                    )
                role = PASS_ROLES[self._pass_index]
                latent, clip_tokens, text_tokens, global_c, extended_c = args[:5]
                self._active_lengths = {
                    "latent": int(latent.shape[1]),
                    "clip": int(clip_tokens.shape[1]),
                    "text": int(text_tokens.shape[1]),
                }
                for semantic, tensor in (
                    ("joint_input_latent", latent),
                    ("joint_input_clip_tokens", clip_tokens),
                    ("joint_input_text_tokens", text_tokens),
                    ("global_condition", global_c),
                    ("extended_condition", extended_c),
                ):
                    key = self._key(semantic + "_fp32", role, "predict_flow")
                    self._arrays[key] = tensor.detach().to(dtype=torch.float32).cpu().numpy()
                self._passes[self._pass_index]["network_consumed"] = {
                    "joint_input_latent": self._key(
                        "joint_input_latent_fp32", role, "predict_flow"
                    ),
                    "joint_input_clip_tokens": self._key(
                        "joint_input_clip_tokens_fp32", role, "predict_flow"
                    ),
                    "joint_input_text_tokens": self._key(
                        "joint_input_text_tokens_fp32", role, "predict_flow"
                    ),
                    "global_condition": self._key(
                        "global_condition_fp32", role, "predict_flow"
                    ),
                    "extended_condition": self._key(
                        "extended_condition_fp32", role, "predict_flow"
                    ),
                    "lengths": dict(self._active_lengths),
                }
        return hook

    def _wrapped_predict_flow(self, latent: Any, t: Any, conditions: Any) -> Any:
        import torch
        self._pass_index += 1
        if self._pass_index >= len(PASS_ROLES):
            raise ArtifactValidationError(
                f"ode_wrapper executed more than {len(PASS_ROLES)} predict_flow calls"
            )
        role = PASS_ROLES[self._pass_index]
        time_key = self._key("actual_broadcast_time", role, "predict_flow")
        latent_key = self._key("actual_predict_flow_latent", role, "predict_flow")
        self._arrays[time_key] = t.detach().to(dtype=torch.float32).cpu().numpy()
        self._arrays[latent_key] = latent.detach().to(dtype=torch.float32).cpu().numpy()
        condition_keys: dict[str, str] = {}
        for name in CONDITION_FIELDS:
            value = getattr(conditions, name)
            key = self._key(f"consumed_condition_{name}_fp32", role, "predict_flow")
            self._arrays[key] = value.detach().to(dtype=torch.float32).cpu().numpy()
            condition_keys[name] = key
        self._passes.append({
            "capture_nonce": self._nonce,
            "pass_index": self._pass_index,
            "pass_role": role,
            "actual_broadcast_time": time_key,
            "actual_latent": latent_key,
            "conditions_as_consumed": condition_keys,
        })
        return self._original_predict_flow(latent, t, conditions)

    def _post_hook(self, site: str, is_joint: bool):
        def hook(_module: Any, _args: Any, output: Any) -> None:
            import torch
            role = PASS_ROLES[self._pass_index]
            latent = output[0] if is_joint else output
            if latent.ndim != 3 or latent.shape[0] != 1:
                raise ArtifactValidationError(
                    f"post-block {site} expected (1,tokens,dim), got {tuple(latent.shape)}"
                )
            native = latent.detach()
            native_fp32 = native.to(dtype=torch.float32)
            # Historical contract: reduce the native activation first, select batch
            # zero, then convert the pooled vector to fp32.  The repaired path is a
            # separate implementation with the same reduction/quantization order.
            pooled_original = native.detach().mean(dim=1)[0].float()
            pooled_repaired = _repaired_pool_same_contract(native)
            token_array = native_fp32.cpu().numpy()
            original_array = pooled_original.cpu().numpy()
            repaired_array = pooled_repaired.cpu().numpy()
            quantized = native.to(dtype=torch.float16)
            dequantized = quantized.to(dtype=torch.float32)
            token_key = self._key("post_block_tokens_fp32", role, site)
            quantized_key = self._key("post_block_tokens_quantized_fp16", role, site)
            dequantized_key = self._key("post_block_tokens_dequantized_fp32", role, site)
            mean_after_quant_key = self._key("mean_after_quantization_fp32", role, site)
            original_key = self._key("pooled_original_prequant_fp32", role, site)
            repaired_key = self._key("pooled_repaired_prequant_fp32", role, site)
            token_mean_key = self._key("token_mean_fp32", role, site)
            stats_key = self._key("token_stats_fp32", role, site)
            stats = torch.stack((
                native_fp32.mean(), native_fp32.std(unbiased=False), native_fp32.min(),
                native_fp32.max(), torch.linalg.vector_norm(native_fp32),
            )).cpu().numpy()
            self._arrays[token_key] = token_array
            self._arrays[quantized_key] = quantized.cpu().numpy()
            self._arrays[dequantized_key] = dequantized.cpu().numpy()
            self._arrays[mean_after_quant_key] = dequantized.mean(dim=1)[0].cpu().numpy()
            self._arrays[original_key] = original_array
            self._arrays[repaired_key] = repaired_array
            self._arrays[token_mean_key] = native_fp32.mean(dim=1)[0].cpu().numpy()
            self._arrays[stats_key] = stats
            native_parent = describe_array(native.cpu().numpy())
            self._tokens.append({
                "capture_nonce": self._nonce,
                "pass_index": self._pass_index,
                "pass_role": role,
                "site": site,
                "hook_site": f"{type(_module).__module__}.{type(_module).__qualname__}",
                "native_dtype": str(latent.dtype),
                "native_shape": list(latent.shape),
                "native_parent_sha256": native_parent["sha256"],
                "persisted_tokens": token_key,
                "quantized_tokens": quantized_key,
                "dequantized_tokens": dequantized_key,
                "mean_after_quantization": mean_after_quant_key,
                "pooled_original": original_key,
                "pooled_repaired": repaired_key,
                "token_mean": token_mean_key,
                "token_stats": stats_key,
                "token_stats_order": ["mean", "std_population", "min", "max", "l2"],
                "pool_operation_original": "latent.detach().mean(dim=1)[0].float()",
                "pool_operation_repaired": (
                    "torch.mean(latent.detach(),dim=1,dtype=latent.dtype)[0].to(float32)"
                ),
                "quantizer": {
                    "source_dtype": str(native.dtype),
                    "stored_dtype": "torch.float16",
                    "rounding": "torch_cast_round_to_nearest_platform_contract",
                    "scale": None,
                    "zero_point": None,
                    "saturation_count": int(torch.count_nonzero(
                        torch.isfinite(native) & ~torch.isfinite(quantized)
                    ).item()),
                },
                "comparison_class": EQUIVALENT_COMPARISON,
            })
            self._active_site = None
        return hook

    def _wrapped_attention(self, q: Any, k: Any, v: Any) -> Any:
        import torch
        site = self._active_site
        role = PASS_ROLES[self._pass_index]
        actual_output = self._original_attention(q, k, v)
        if site in self.selected_sites:
            prefix = self._key("attention", role, str(site))
            q_key, k_key, v_key = prefix + "__q", prefix + "__k", prefix + "__v"
            out_key = prefix + "__actual_output"
            latent_out_key = prefix + "__actual_latent_query_output_fp32"
            latent_summary_key = prefix + "__actual_latent_query_summary_fp32"
            prob_key = prefix + "__derived_latent_to_clip_probability_fp32"
            summary_key = prefix + "__derived_clip_summary_fp32"
            q_fp32 = q.detach().to(torch.float32)
            k_fp32 = k.detach().to(torch.float32)
            latent_len = self._active_lengths.get("latent")
            clip_len = self._active_lengths.get("clip")
            if latent_len is None or clip_len is None:
                raise ArtifactValidationError(f"attention lengths unavailable at {site}")
            latent_output = actual_output[:, :latent_len]
            self._arrays[q_key] = q_fp32.cpu().numpy()
            self._arrays[k_key] = k_fp32.cpu().numpy()
            self._arrays[v_key] = v.detach().to(torch.float32).cpu().numpy()
            self._arrays[out_key] = actual_output.detach().to(torch.float32).cpu().numpy()
            self._arrays[latent_out_key] = latent_output.detach().to(torch.float32).cpu().numpy()
            self._arrays[latent_summary_key] = latent_output.detach().to(
                torch.float32
            ).mean(dim=1).cpu().numpy()
            record = {
                "capture_nonce": self._nonce,
                "pass_index": self._pass_index,
                "pass_role": role,
                "site": site,
                "q": q_key,
                "k": k_key,
                "v": v_key,
                "actual_attention_output": out_key,
                "actual_latent_query_output": latent_out_key,
                "actual_latent_query_summary": latent_summary_key,
                "latent_query_slice": [0, latent_len],
            }
            if str(site).startswith("joint."):
                probability_all = torch.softmax(
                    torch.matmul(q_fp32, k_fp32.transpose(-2, -1)) /
                    float(q.shape[-1]) ** 0.5,
                    dim=-1,
                )
                clip_start, clip_stop = latent_len, latent_len + clip_len
                clip_probability = probability_all[:, :, :latent_len, clip_start:clip_stop]
                clip_probability_array = clip_probability.cpu().numpy().astype(np.float32)
                if clip_probability_array.shape[-1] != clip_len or clip_len < 1:
                    raise ArtifactValidationError(f"joint attention clip slice invalid at {site}")
                self._arrays[prob_key] = clip_probability_array
                self._arrays[summary_key] = _attention_summary(clip_probability_array)
                record.update({
                    "probability_map": prob_key,
                    "probability_summary": summary_key,
                    "clip_key_slice": [clip_start, clip_stop],
                    "attention_scale": f"1/sqrt({int(q.shape[-1])})",
                    "mask": None,
                    "softmax_dtype": "torch.float32",
                    "summary_contract": (
                        "numpy.mean(float32_map,axis=(heads,latent_query),dtype=float64)->float32"
                    ),
                    "probability_map_provenance": (
                        "RECOMPUTED_DERIVED latent-query/clip-key slice of "
                        "softmax(fp32(Q)@fp32(K)^T/sqrt(d)); "
                        "not exposed by scaled_dot_product_attention"
                    ),
                })
            else:
                record.update({
                    "probability_map": None,
                    "probability_summary": None,
                    "clip_key_slice": None,
                    "probability_map_provenance": (
                        "NOT_APPLICABLE fused block is latent self-attention with no clip-key slice"
                    ),
                })
            self._attention.append(record)
        return actual_output

    @contextmanager
    def armed(self, capture_nonce: str):
        if (self._handles or self._original_attention is not None or
                self._original_predict_flow is not None):
            raise ArtifactValidationError("capture is already armed")
        self._nonce = str(capture_nonce)
        self._pass_index = -1
        self._arrays = {}
        self._tokens = []
        self._attention = []
        self._passes = []
        self._active_lengths = {}
        for index, block in enumerate(self.joint):
            site = f"joint.{index}"
            self._handles.append(block.register_forward_pre_hook(self._pre_hook(site)))
            self._handles.append(block.register_forward_hook(self._post_hook(site, True)))
        for index, block in enumerate(self.fused):
            site = f"fused.{index}"
            self._handles.append(block.register_forward_pre_hook(self._pre_hook(site)))
            self._handles.append(block.register_forward_hook(self._post_hook(site, False)))
        self._original_attention = self.attention_module.attention
        self._original_predict_flow = self.net.predict_flow
        self.attention_module.attention = self._wrapped_attention
        self.net.predict_flow = self._wrapped_predict_flow
        try:
            yield self
        finally:
            self.net.predict_flow = self._original_predict_flow
            self._original_predict_flow = None
            self.attention_module.attention = self._original_attention
            self._original_attention = None
            for handle in self._handles:
                handle.remove()
            self._handles = []
            self._active_site = None
            self._active_lengths = {}

    def finish(self) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        if len(self._passes) != len(PASS_ROLES):
            raise ArtifactValidationError(
                f"captured {len(self._passes)} predict_flow calls, expected {len(PASS_ROLES)}"
            )
        for index, record in enumerate(self._passes):
            if record.get("pass_index") != index or record.get("pass_role") != PASS_ROLES[index]:
                raise ArtifactValidationError("predict_flow pass order/role is ambiguous")
            if set(record.get("conditions_as_consumed", {})) != set(CONDITION_FIELDS):
                raise ArtifactValidationError("network-consumed conditioning capture is incomplete")
            if not record.get("network_consumed"):
                raise ArtifactValidationError("global/extended conditioning capture is incomplete")
        expected_tokens = len(PASS_ROLES) * (len(self.joint) + len(self.fused))
        if len(self._tokens) != expected_tokens:
            raise ArtifactValidationError(
                f"captured {len(self._tokens)} post-block tensors, expected {expected_tokens}"
            )
        expected_attention = len(PASS_ROLES) * len(self.selected_sites)
        if len(self._attention) != expected_attention:
            raise ArtifactValidationError(
                f"captured {len(self._attention)} selected attentions, expected "
                f"{expected_attention}"
            )
        per_role = {role: sum(row["pass_role"] == role for row in self._tokens)
                    for role in PASS_ROLES}
        metadata = {
            "capture_nonce": self._nonce,
            "one_ode_wrapper_evaluation": True,
            "expected_passes": len(PASS_ROLES),
            "observed_passes": self._pass_index + 1,
            "pass_roles": list(PASS_ROLES),
            "post_block_count": len(self._tokens),
            "post_block_count_by_role": per_role,
            "joint_blocks": len(self.joint),
            "fused_blocks": len(self.fused),
            "selected_attention_sites": sorted(self.selected_sites),
            "attention_capture_count": len(self._attention),
            "tokens": self._tokens,
            "attention": self._attention,
            "passes": self._passes,
        }
        if metadata["observed_passes"] != len(PASS_ROLES):
            raise ArtifactValidationError(
                f"observed {metadata['observed_passes']} passes, expected two"
            )
        return dict(self._arrays), metadata


def _torch_dtype_from_numpy(torch: Any, array: np.ndarray) -> Any:
    probe = torch.from_numpy(np.empty((), dtype=np.asarray(array).dtype))
    return probe.dtype


def _rehydrate_conditions(arrays: Mapping[str, np.ndarray], prefix: str,
                          condition_type: Any, device: str) -> Any:
    import torch
    values: dict[str, Any] = {}
    for name in CONDITION_FIELDS:
        array = np.asarray(arrays[f"{prefix}__{name}"])
        tensor = torch.from_numpy(np.ascontiguousarray(array)).to(
            device=device, dtype=_torch_dtype_from_numpy(torch, array)
        )
        values[name] = tensor
    return condition_type(**values)


def _capture_nonce(packet_completion_sha256: str, repeat_index: int,
                   replay_instance_id: str) -> str:
    return sha256_bytes(
        f"{packet_completion_sha256}|repeat={int(repeat_index)}|instance={replay_instance_id}".encode(
            "utf-8"
        )
    )[:32]


def _write_float_wav(path: Path, wav: np.ndarray, sample_rate: int = 16000) -> None:
    import soundfile as sf
    if path.exists():
        raise ImmutableArtifactError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = np.asarray(wav, dtype=np.float32)
    describe_array(normalized)
    sf.write(str(path), normalized, sample_rate, subtype="FLOAT")


def _same_order_tweedie_recompute(arrays: Mapping[str, np.ndarray]) -> np.ndarray:
    import torch
    x = torch.from_numpy(np.ascontiguousarray(arrays["device_latent_native"]))
    t = torch.from_numpy(np.ascontiguousarray(arrays["device_time_native"]))
    velocity = torch.from_numpy(np.ascontiguousarray(arrays["returned_velocity_native"]))
    return (x + (1.0 - t) * velocity).cpu().numpy()


def _exact_comparison(comparison_type: str, lhs_key: str, rhs_key: str,
                      lhs: np.ndarray, rhs: np.ndarray, **metadata: Any) -> dict[str, Any]:
    return {
        "eligible_for_tolerance": False,
        "required_exact": True,
        "comparison_type": comparison_type,
        "lhs": lhs_key,
        "rhs": rhs_key,
        "metrics": _comparison_metrics(lhs, rhs),
        **metadata,
    }


def _comparison_records(arrays: Mapping[str, np.ndarray], capture: Mapping[str, Any],
                        readback: Mapping[str, np.ndarray]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in capture["tokens"]:
        lhs_key = record["pooled_original"]
        rhs_key = record["pooled_repaired"]
        lhs, rhs = arrays[lhs_key], arrays[rhs_key]
        rows.append(_exact_comparison(
            EQUIVALENT_COMPARISON, lhs_key, rhs_key, lhs, rhs,
            pass_role=record["pass_role"], site=record["site"],
            capture_nonce=record["capture_nonce"],
            lhs_operation=record["pool_operation_original"],
            rhs_operation=record["pool_operation_repaired"],
        ))
    packet_x = arrays["packet_x_s_fp32"]
    device_x = arrays["device_latent_fp32"][0]
    rows.append(_exact_comparison(
        INPUT_COMPARISON, "packet_x_s_fp32", "device_latent_fp32[0]", packet_x, device_x,
        pass_role="input", site="ode_wrapper_input_latent",
        capture_nonce=capture["capture_nonce"],
    ))
    for pass_record in capture["passes"]:
        role = pass_record["pass_role"]
        actual_key = pass_record["actual_broadcast_time"]
        actual = arrays[actual_key]
        expected = np.full(actual.shape, arrays["packet_model_time_fp32"], dtype=np.float32)
        rows.append(_exact_comparison(
            TIME_COMPARISON, "broadcast(packet_model_time_fp32)", actual_key,
            expected, actual, pass_role=role, site="predict_flow_broadcast_time",
            capture_nonce=capture["capture_nonce"],
        ))
    recomputed_tweedie = _same_order_tweedie_recompute(readback)
    rows.append({
        "eligible_for_tolerance": True,
        "required_exact": False,
        "comparison_type": TWEEDIE_RECOMPUTE_COMPARISON,
        "pass_role": "combined_velocity",
        "site": "normalized_tweedie_reducer",
        "capture_nonce": capture["capture_nonce"],
        "lhs": "tweedie_latent_normalized_native",
        "rhs": "same_order_recompute_from_atomic_readback",
        "operation": "x + (1.0 - t) * velocity in stored native dtype/order",
        "metrics": _comparison_metrics(
            readback["tweedie_latent_normalized_native"], recomputed_tweedie
        ),
    })
    for attention in capture["attention"]:
        map_key = attention["probability_map"]
        summary_key = attention["probability_summary"]
        if map_key is None or summary_key is None:
            continue
        recomputed = _attention_summary(readback[map_key])
        rows.append(_exact_comparison(
            ATTENTION_SUMMARY_COMPARISON, summary_key,
            f"same_contract_recompute({map_key})", readback[summary_key], recomputed,
            pass_role=attention["pass_role"], site=attention["site"],
            capture_nonce=capture["capture_nonce"],
        ))
    for key in sorted(arrays):
        rows.append(_exact_comparison(
            ATOMIC_READBACK_COMPARISON, key, f"atomic_readback:{key}",
            arrays[key], readback[key], pass_role="storage", site=key,
            capture_nonce=capture["capture_nonce"],
        ))
    return rows


def _capture_array_keys(capture: Mapping[str, Any]) -> set[str]:
    keys: set[str] = set()
    for record in capture.get("tokens", []):
        for field in (
            "persisted_tokens", "quantized_tokens", "dequantized_tokens",
            "mean_after_quantization", "pooled_original", "pooled_repaired",
            "token_mean", "token_stats",
        ):
            if isinstance(record.get(field), str):
                keys.add(record[field])
    for record in capture.get("attention", []):
        for field in (
            "q", "k", "v", "actual_attention_output", "actual_latent_query_output",
            "actual_latent_query_summary", "probability_map", "probability_summary",
        ):
            if isinstance(record.get(field), str):
                keys.add(record[field])
    for record in capture.get("passes", []):
        for field in ("actual_broadcast_time", "actual_latent"):
            if isinstance(record.get(field), str):
                keys.add(record[field])
        keys.update(str(value) for value in record.get("conditions_as_consumed", {}).values())
        consumed = record.get("network_consumed", {})
        for field in (
            "joint_input_latent", "joint_input_clip_tokens", "joint_input_text_tokens",
            "global_condition", "extended_condition",
        ):
            if isinstance(consumed.get(field), str):
                keys.add(consumed[field])
    return keys


def _required_replay_keys(capture: Mapping[str, Any]) -> set[str]:
    base = {
        "packet_x_s_fp32", "packet_model_time_fp32", "device_latent_fp32",
        "device_time_fp32", "device_latent_native", "device_time_native",
        "returned_velocity_native", "returned_velocity_fp32",
        "tweedie_latent_normalized_native", "tweedie_latent_fp32",
        "tweedie_latent_unnormalized_fp32", "panns_clipwise_output_fp32",
        "panns_embedding_fp32",
    }
    base.update(
        f"{prefix}__{field}"
        for prefix in ("conditions", "empty_conditions")
        for field in CONDITION_FIELDS
    )
    return base | _capture_array_keys(capture)


def write_replay_unit(
    attempt_root: Path,
    *,
    packet_manifest: Mapping[str, Any],
    packet_arrays: Mapping[str, np.ndarray],
    packet_unit_root: Path,
    repeat_index: int,
    arrays: Mapping[str, np.ndarray],
    capture_metadata: Mapping[str, Any],
    preview_wav: np.ndarray,
    provenance: Mapping[str, Any],
) -> Path:
    attempt_protocol = validate_protocol_binding(attempt_root)
    packet_attempt_protocol = validate_protocol_binding(Path(packet_unit_root).parent.parent)
    if attempt_protocol["sha256"] != packet_attempt_protocol["sha256"]:
        raise ArtifactValidationError("replay attempt protocol differs from packet protocol")
    if packet_manifest.get("protocol_sha256") != attempt_protocol["sha256"]:
        raise ArtifactValidationError("packet manifest protocol differs from replay protocol")
    identity = packet_manifest["identity"]
    uid = unit_id(identity["clip_id"], identity["s"], repeat_index)
    root = Path(attempt_root) / "units" / uid
    try:
        root.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise ImmutableArtifactError(f"replay unit already exists: {root}") from exc
    normalized = {key: np.asarray(value) for key, value in arrays.items()}
    required_keys = _required_replay_keys(capture_metadata)
    if set(normalized) != required_keys:
        raise ArtifactValidationError(
            f"replay tensor contract mismatch: missing={sorted(required_keys - set(normalized))}, "
            f"extra={sorted(set(normalized) - required_keys)}"
        )
    descriptors = {key: describe_array(value) for key, value in normalized.items()}
    _write_new_npz(root / "arrays.npz", normalized)
    with np.load(root / "arrays.npz", allow_pickle=False) as archive:
        readback = {key: np.array(archive[key], copy=True) for key in archive.files}
    for key in sorted(normalized):
        if describe_array(readback[key]) != descriptors[key] or not np.array_equal(
            normalized[key], readback[key]
        ):
            raise ArtifactValidationError(f"atomic NPZ readback mismatch for {key}")
    _write_float_wav(root / "external_preview.wav", preview_wav)
    replay_identity = {
        "clip_id": identity["clip_id"],
        "role": identity["role"],
        "s": identity["s"],
        "repeat_index": int(repeat_index),
        "phase1": identity["phase1"],
    }
    manifest = {
        "schema": REPLAY_SCHEMA,
        "protocol_sha256": attempt_protocol["sha256"],
        "unit_id": uid,
        "identity": replay_identity,
        "packet_parent": {
            "unit_id": packet_manifest["unit_id"],
            "packet_manifest_sha256": sha256_file(Path(packet_unit_root) / "manifest.json"),
            "packet_completion_sha256": sha256_file(Path(packet_unit_root) / "COMPLETED.json"),
            "packet_arrays_sha256": sha256_file(Path(packet_unit_root) / "arrays.npz"),
            "packet_x_s_sha256": array_sha256(packet_arrays["x_s"]),
        },
        "arrays_file": "arrays.npz",
        "arrays_file_sha256": sha256_file(root / "arrays.npz"),
        "arrays": descriptors,
        "required_array_keys": sorted(required_keys),
        "external_preview": {
            "path": "external_preview.wav",
            "sha256": sha256_file(root / "external_preview.wav"),
            "sample_rate": 16000,
            "subtype": "FLOAT",
            "samples": int(np.asarray(preview_wav).size),
            "parent": "tweedie_latent_from_same_ode_wrapper_evaluation",
        },
        "capture": dict(capture_metadata),
        "comparisons": _comparison_records(normalized, capture_metadata, readback),
        "gating_policy": {
            "protocol_gating_comparisons": list(_PROTOCOL_GATE_TEXT),
            "exact_comparison_classes": [
                EQUIVALENT_COMPARISON, INPUT_COMPARISON, TIME_COMPARISON,
                ATOMIC_READBACK_COMPARISON, ATTENTION_SUMMARY_COMPARISON,
            ],
            "tolerance_comparison_classes": [TWEEDIE_RECOMPUTE_COMPARISON,
                                               CROSS_REPLAY_COMPARISON],
            "forbidden_comparison": FORBIDDEN_GATING_COMPARISON,
            "forbidden_comparison_present": False,
        },
        "provenance": dict(provenance),
    }
    _write_new_json(root / "manifest.json", manifest)
    _finish_unit(root, "replay", {
        **replay_identity, "capture_nonce": capture_metadata["capture_nonce"]
    })
    return root


def validate_replay_unit(root: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    root = Path(root)
    completion_path = root / "COMPLETED.json"
    if not completion_path.is_file():
        raise ArtifactValidationError(f"partial replay unit (no completion): {root}")
    completion = _load_json(completion_path)
    if completion.get("status") != "COMPLETE" or completion.get("stage") != "replay":
        raise ArtifactValidationError(f"invalid replay completion: {root}")
    _validate_inventory(root, completion, unit=True)
    manifest = _load_json(root / "manifest.json")
    protocol = validate_protocol_binding(root.parent.parent)
    if manifest.get("schema") != REPLAY_SCHEMA or manifest.get("unit_id") != root.name:
        raise ArtifactValidationError(f"replay schema/unit conflict: {root}")
    if manifest.get("protocol_sha256") != protocol["sha256"]:
        raise ArtifactValidationError(f"replay protocol parent mismatch: {root}")
    if manifest.get("arrays_file_sha256") != sha256_file(root / "arrays.npz"):
        raise ArtifactValidationError(f"replay npz hash mismatch: {root}")
    arrays = _load_npz_verified(root / "arrays.npz", manifest.get("arrays", {}))
    if set(arrays) != set(manifest.get("required_array_keys", [])):
        raise ArtifactValidationError(f"replay required-array contract mismatch: {root}")
    preview = manifest.get("external_preview", {})
    if preview.get("sha256") != sha256_file(root / "external_preview.wav"):
        raise ArtifactValidationError(f"preview hash mismatch: {root}")
    try:
        import soundfile as sf
        waveform, sample_rate = sf.read(root / "external_preview.wav", dtype="float32")
    except Exception as exc:
        raise ArtifactValidationError(f"cannot read preview waveform {root}: {exc}") from exc
    describe_array(np.asarray(waveform))
    if int(sample_rate) != 16000 or int(np.asarray(waveform).size) != preview.get("samples"):
        raise ArtifactValidationError(f"preview waveform schema mismatch: {root}")
    capture = manifest.get("capture", {})
    if not capture.get("one_ode_wrapper_evaluation") or capture.get("pass_roles") != list(PASS_ROLES):
        raise ArtifactValidationError(f"same-forward/pass-role contract absent: {root}")
    if capture.get("observed_passes") != 2:
        raise ArtifactValidationError(f"replay did not capture exactly two passes: {root}")
    if capture.get("capture_nonce") != completion.get("identity", {}).get(
        "capture_nonce", capture.get("capture_nonce")
    ):
        raise ArtifactValidationError(f"capture nonce conflicts with completion: {root}")
    passes = capture.get("passes", [])
    if [row.get("pass_role") for row in passes] != list(PASS_ROLES):
        raise ArtifactValidationError(f"predict_flow pass order/roles invalid: {root}")
    for index, row in enumerate(passes):
        if row.get("pass_index") != index or row.get("capture_nonce") != capture.get("capture_nonce"):
            raise ArtifactValidationError(f"predict_flow pass identity invalid: {root}")
        if set(row.get("conditions_as_consumed", {})) != set(CONDITION_FIELDS):
            raise ArtifactValidationError(f"consumed conditioning is incomplete: {root}")
        consumed = row.get("network_consumed", {})
        if not all(consumed.get(key) for key in (
            "joint_input_latent", "joint_input_clip_tokens", "joint_input_text_tokens",
            "global_condition", "extended_condition",
        )):
            raise ArtifactValidationError(f"global/extended conditioning is incomplete: {root}")
    joint_count = int(capture.get("joint_blocks", -1))
    fused_count = int(capture.get("fused_blocks", -1))
    if joint_count < 1 or fused_count < 1:
        raise ArtifactValidationError(f"invalid block topology: {root}")
    expected_sites = ([f"joint.{i}" for i in range(joint_count)] +
                      [f"fused.{i}" for i in range(fused_count)])
    tokens = capture.get("tokens", [])
    expected_token_pairs = [(role, site) for role in PASS_ROLES for site in expected_sites]
    if [(row.get("pass_role"), row.get("site")) for row in tokens] != expected_token_pairs:
        raise ArtifactValidationError(f"post-block hook count/order invalid: {root}")
    for row in tokens:
        if row.get("capture_nonce") != capture.get("capture_nonce"):
            raise ArtifactValidationError(f"post-block capture parent mismatch: {root}")
        if row.get("pool_operation_original") != "latent.detach().mean(dim=1)[0].float()":
            raise ArtifactValidationError(f"historical original pooling contract missing: {root}")
        token_key = row.get("persisted_tokens")
        if array_sha256(arrays[token_key]) != row.get("native_parent_sha256"):
            raise ArtifactValidationError(f"post-block native parent hash mismatch: {root}:{token_key}")
    selected_sites = set(capture.get("selected_attention_sites", []))
    required_selected = {"joint.0", f"joint.{joint_count - 1}", f"fused.{fused_count - 1}"}
    if selected_sites != required_selected:
        raise ArtifactValidationError(f"selected attention sites changed: {root}")
    expected_attention_pairs = [
        (role, site) for role in PASS_ROLES for site in expected_sites if site in required_selected
    ]
    attention = capture.get("attention", [])
    if [(row.get("pass_role"), row.get("site")) for row in attention] != expected_attention_pairs:
        raise ArtifactValidationError(f"selected attention count/order invalid: {root}")
    for row in attention:
        if row.get("capture_nonce") != capture.get("capture_nonce"):
            raise ArtifactValidationError(f"attention parent mismatch: {root}")
        if str(row.get("site", "")).startswith("joint."):
            if "RECOMPUTED_DERIVED" not in row.get("probability_map_provenance", ""):
                raise ArtifactValidationError(f"derived attention map mislabeled: {root}")
            if not row.get("probability_map") or not row.get("probability_summary"):
                raise ArtifactValidationError(f"joint attention summary absent: {root}")
        elif row.get("probability_map") is not None or row.get("probability_summary") is not None:
            raise ArtifactValidationError(f"fused self-attention falsely exposes clip map: {root}")
    required_keys = _required_replay_keys(capture)
    if set(arrays) != required_keys:
        raise ArtifactValidationError(
            f"replay scientific tensor set mismatch: missing={sorted(required_keys - set(arrays))}, "
            f"extra={sorted(set(arrays) - required_keys)}"
        )
    policy = manifest.get("gating_policy", {})
    if policy.get("forbidden_comparison_present"):
        raise ArtifactValidationError(f"forbidden reduction-order comparison gates {root}")
    if policy.get("protocol_gating_comparisons") != list(_PROTOCOL_GATE_TEXT):
        raise ArtifactValidationError(f"protocol gate comparisons are incomplete: {root}")
    recomputed = _comparison_records(arrays, capture, arrays)
    if canonical_json_bytes(recomputed) != canonical_json_bytes(manifest.get("comparisons", [])):
        raise ArtifactValidationError(f"stored comparison metrics do not recompute: {root}")
    for comparison in recomputed:
        for metric_name, value in comparison.get("metrics", {}).items():
            if not np.isfinite(float(value)) or (
                metric_name != "cosine" and float(value) < 0
            ):
                raise ArtifactValidationError(f"invalid comparison metric: {root}")
        if comparison.get("required_exact"):
            metrics = comparison["metrics"]
            if (metrics["relative_l2"] != 0.0 or metrics["absolute_l2"] != 0.0 or
                    metrics["max_abs"] != 0.0 or metrics["exact_fraction"] != 1.0):
                raise ArtifactValidationError(
                    f"exact identity comparison failed: {root}:"
                    f"{comparison['comparison_type']}:{comparison.get('site')}"
                )
    return manifest, arrays


def _real_replay_one(
    backend: Any,
    panns_model: Any,
    packet_root: Path,
    repeat_index: int,
    device: str,
    *,
    replay_instance_id: str,
    condition_type: Any | None = None,
    attention_module: Any | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any], np.ndarray]:
    import torch
    if condition_type is None or attention_module is None:
        from mmaudio.model.networks import PreprocessedConditions
        from mmaudio.model import transformer_layers
        condition_type = PreprocessedConditions
        attention_module = transformer_layers

    packet_manifest, packet = validate_packet_unit(packet_root)
    conditions = _rehydrate_conditions(packet, "conditions", condition_type, device)
    empty_conditions = _rehydrate_conditions(
        packet, "empty_conditions", condition_type, device
    )
    x_device = torch.from_numpy(np.ascontiguousarray(packet["x_s"])).to(
        device=device, dtype=backend.dtype
    ).unsqueeze(0)
    t_device = torch.tensor(float(packet["model_time"]), device=device, dtype=backend.dtype)
    packet_completion_hash = sha256_file(packet_root / "COMPLETED.json")
    nonce = _capture_nonce(packet_completion_hash, repeat_index, replay_instance_id)
    collector = SameForwardCapture(backend.net, attention_module)
    started = utc_now()
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(torch.device(device))
    with torch.inference_mode():
        with collector.armed(nonce):
            velocity = backend.net.ode_wrapper(
                t_device, x_device, conditions, empty_conditions, 1.0
            )
        capture_arrays, capture_metadata = collector.finish()
        # MMAudio.unnormalize is in-place.  Preserve the normalized Tweedie
        # evidence before giving a clone to the decoder path.
        tweedie_normalized = (x_device + (1.0 - t_device) * velocity).clone()
        unnormalized = backend.net.unnormalize(tweedie_normalized.clone())
        spectrogram = backend.feature_utils.decode(unnormalized)
        preview_tensor = backend.feature_utils.vocode(spectrogram)
        preview = preview_tensor.float().cpu().numpy().reshape(-1).astype(np.float32)
        panns_out = panns_model(
            torch.from_numpy(preview[None, :]).to(device=device, dtype=torch.float32)
        )
    arrays: dict[str, np.ndarray] = {
        "packet_x_s_fp32": np.asarray(packet["x_s"], dtype=np.float32),
        "packet_model_time_fp32": np.asarray(packet["model_time"], dtype=np.float32),
        "device_latent_fp32": x_device.detach().float().cpu().numpy(),
        "device_time_fp32": t_device.detach().float().cpu().numpy(),
        "device_latent_native": x_device.detach().cpu().numpy(),
        "device_time_native": t_device.detach().cpu().numpy(),
        "returned_velocity_native": velocity.detach().cpu().numpy(),
        "returned_velocity_fp32": velocity.detach().float().cpu().numpy(),
        "tweedie_latent_normalized_native": tweedie_normalized.detach().cpu().numpy(),
        "tweedie_latent_fp32": tweedie_normalized.detach().float().cpu().numpy(),
        "tweedie_latent_unnormalized_fp32": unnormalized.detach().float().cpu().numpy(),
        "panns_clipwise_output_fp32": panns_out["clipwise_output"].detach().float().cpu().numpy(),
        "panns_embedding_fp32": panns_out["embedding"].detach().float().cpu().numpy(),
        **capture_arrays,
    }
    # Complete conditioning is duplicated into replay evidence, not merely linked.
    for key, value in packet.items():
        if key.startswith("conditions__") or key.startswith("empty_conditions__"):
            arrays[key] = np.asarray(value)
    capture_metadata = dict(capture_metadata)
    capture_metadata["actual_inputs"] = {
        "device_latent": describe_array(arrays["device_latent_fp32"]),
        "device_time": describe_array(arrays["device_time_fp32"]),
    }
    capture_metadata["returned_velocity"] = describe_array(arrays["returned_velocity_fp32"])
    capture_metadata["tweedie_latent"] = describe_array(arrays["tweedie_latent_fp32"])
    capture_metadata["tweedie_contract"] = {
        "operation": "x_device + (1.0 - t_device) * returned_velocity",
        "normalized_clone_before_unnormalize": True,
        "unnormalize_is_in_place": True,
        "decoder_input": "clone(tweedie_latent_normalized_native)",
    }
    capture_metadata["conditioning_fields"] = list(CONDITION_FIELDS)
    capture_metadata["conditioning_complete"] = True
    capture_metadata["runtime"] = {
        "started_utc": started,
        "ended_utc": utc_now(),
        "ode_wrapper_evaluations": 1,
        "predict_flow_nfe": len(PASS_ROLES),
        "peak_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(torch.device(device)))
            if str(device).startswith("cuda") and torch.cuda.is_available() else 0
        ),
    }
    return arrays, capture_metadata, preview


def replay_attempt(
    packet_attempt: Path,
    output_root: Path,
    attempt_id: str,
    *,
    role: str,
    mmaudio_root: Path,
    weights_dir: Path,
    clips_root: Path,
    device: str,
    repeats: int = 1,
    repeat_offset: int = 0,
    protocol_path: Path,
    protocol_sha256: str,
    calibration_attempt: Path | None = None,
    tolerance_sha256: str | None = None,
) -> Path:
    if role not in {"calibration", "heldout"}:
        raise ValueError("replay role must be calibration or heldout")
    if repeats < 1 or repeat_offset < 0:
        raise ValueError("repeats must be >=1 and repeat_offset >=0")
    protocol = load_protocol(protocol_path, protocol_sha256)
    asset_observation = _require_asset_roots(
        mmaudio_root, weights_dir, clips_root, protocol["pilot"]
    )
    validate_attempt(
        packet_attempt, expected_stage="packets",
        expected_protocol_sha256=protocol["sha256"],
    )
    calibration_authority: dict[str, Any] | None = None
    if role == "heldout":
        if calibration_attempt is None or tolerance_sha256 is None:
            raise ArtifactValidationError(
                "heldout replay is prohibited before an immutable calibration attempt and "
                "mandatory tolerance SHA are supplied"
            )
        calibration_authority = validate_calibration_authority(
            calibration_attempt, tolerance_sha256,
            expected_protocol_sha256=protocol["sha256"],
        )
    elif calibration_attempt is not None or tolerance_sha256 is not None:
        raise ArtifactValidationError("calibration replay cannot accept heldout authority inputs")
    packet_units: list[tuple[Path, dict[str, Any]]] = []
    for packet_root in sorted((Path(packet_attempt) / "units").iterdir()):
        if not packet_root.is_dir():
            continue
        manifest, _ = validate_packet_unit(packet_root)
        if manifest["identity"]["role"] == role:
            packet_assets = manifest.get("provenance", {}).get("asset_observation", {})
            if packet_assets.get("contract_sha256") != asset_observation.get("contract_sha256"):
                raise ArtifactValidationError(
                    f"packet asset contract does not match replay assets: {packet_root}"
                )
            if packet_assets.get("mmaudio_weights") != asset_observation.get("mmaudio_weights"):
                raise ArtifactValidationError(
                    f"packet MMAudio weight hashes do not match replay: {packet_root}"
                )
            if packet_assets.get("local_hashes") != asset_observation.get("local_hashes"):
                raise ArtifactValidationError(
                    f"packet PANNs/backend hashes do not match replay: {packet_root}"
                )
            packet_units.append((packet_root, manifest))
    expected_clips = set(CALIBRATION_CLIPS if role == "calibration" else (HELDOUT_CLIP,))
    if {manifest["identity"]["clip_id"] for _, manifest in packet_units} != expected_clips:
        raise ArtifactValidationError(f"packet attempt does not contain exact {role} clip set")
    if len(packet_units) != len(expected_clips) * len(S_POINTS):
        raise ArtifactValidationError(f"packet role {role} has incomplete progress grid")

    root = create_bound_attempt(output_root, "replay", attempt_id, protocol)
    mmaudio_root = Path(mmaudio_root).resolve()
    if str(mmaudio_root) not in sys.path:
        sys.path.insert(0, str(mmaudio_root))
    from .mmaudio_backend import MMAudioBackend
    from .measurers_panns_cnn14 import load_cnn14_16k

    backend = MMAudioBackend(
        variant="small_16k", device=device, full_precision=True, cfg_strength=1.0,
        num_steps=20, duration_sec=8.0, weights_root=str(mmaudio_root),
        enable_conditions=True,
    )
    panns_path = Path(weights_dir) / "Cnn14_16k_mAP=0.438.pth"
    panns_model = load_cnn14_16k(panns_path, device=device)
    provenance = environment_provenance(
        device, mmaudio_root, protocol_sha256=protocol["sha256"],
        asset_observation=asset_observation,
    )
    replay_instance_id = sha256_bytes(canonical_json_bytes({
        "attempt_id": attempt_id,
        "hostname": socket.gethostname(),
        "device": device,
        "pid": os.getpid(),
        "started_utc": utc_now(),
    }))[:32]
    provenance.update({
        "panns_checkpoint": str(panns_path.resolve()),
        "panns_checkpoint_sha256": sha256_file(panns_path),
        "packet_attempt_completion_sha256": sha256_file(Path(packet_attempt) / "COMPLETED.json"),
        "repeat_count": int(repeats),
        "repeat_offset": int(repeat_offset),
        "role": role,
        "replay_instance_id": replay_instance_id,
        "process_id": os.getpid(),
        "calibration_authority": calibration_authority,
    })
    count = 0
    for packet_root, packet_manifest in packet_units:
        packet_manifest_checked, packet_arrays = validate_packet_unit(packet_root)
        if packet_manifest_checked != packet_manifest:
            raise ArtifactValidationError(f"packet changed during replay: {packet_root}")
        for repeat_index in range(repeat_offset, repeat_offset + repeats):
            arrays, capture, preview = _real_replay_one(
                backend, panns_model, packet_root, repeat_index, device,
                replay_instance_id=replay_instance_id,
            )
            write_replay_unit(
                root, packet_manifest=packet_manifest, packet_arrays=packet_arrays,
                packet_unit_root=packet_root, repeat_index=repeat_index, arrays=arrays,
                capture_metadata=capture, preview_wav=preview, provenance=provenance,
            )
            count += 1
    finish_attempt(root, "replay", expected_units=len(packet_units) * repeats)
    return root


def validate_attempt(root: Path, expected_stage: str | None = None,
                     expected_protocol_sha256: str | None = None) -> dict[str, Any]:
    root = Path(root)
    completion_path = root / "COMPLETED.json"
    if not completion_path.is_file():
        raise ArtifactValidationError(f"partial attempt (no completion journal): {root}")
    completion = _load_json(completion_path)
    if completion.get("schema") != ATTEMPT_SCHEMA or completion.get("status") != "COMPLETE":
        raise ArtifactValidationError(f"invalid attempt completion: {root}")
    protocol = validate_protocol_binding(root, expected_protocol_sha256)
    stage = completion.get("stage")
    if expected_stage is not None and stage != expected_stage:
        raise ArtifactValidationError(f"attempt stage {stage!r} != {expected_stage!r}")
    _validate_inventory(root, completion)
    units_dir = root / "units"
    identities: dict[bytes, str] = {}
    completed_units = 0
    if units_dir.exists():
        for unit_root in sorted(path for path in units_dir.iterdir() if path.is_dir()):
            if stage == "packets":
                manifest, _ = validate_packet_unit(unit_root)
            elif stage == "replay":
                manifest, _ = validate_replay_unit(unit_root)
            else:
                raise ArtifactValidationError(f"unexpected units for stage {stage}: {unit_root}")
            key = canonical_json_bytes(manifest["identity"])
            if key in identities:
                raise ArtifactValidationError(
                    f"conflicting/duplicate scientific identity in {unit_root} and "
                    f"{identities[key]}"
                )
            identities[key] = str(unit_root)
            completed_units += 1
    if completed_units != int(completion.get("expected_units", -1)):
        raise ArtifactValidationError(
            f"unit cardinality {completed_units} != completion ledger "
            f"{completion.get('expected_units')}"
        )
    if stage == "selection":
        selection = _load_json(root / "selection.json")
        if selection.get("schema") != SELECTION_SCHEMA:
            raise ArtifactValidationError("selection schema mismatch")
        clips = selection.get("clips", [])
        roles = {(row.get("clip_id"), row.get("role")) for row in clips}
        expected = {(clip, "calibration") for clip in CALIBRATION_CLIPS} | {
            (HELDOUT_CLIP, "heldout")
        }
        if roles != expected or selection.get("progress_points") != list(S_POINTS):
            raise ArtifactValidationError("selection role/grid contract mismatch")
    elif stage == "calibration":
        tolerance = _load_json(root / "TOLERANCE.json")
        if tolerance.get("schema") != TOLERANCE_SCHEMA or tolerance.get("status") != "FROZEN":
            raise ArtifactValidationError("calibration tolerance is not frozen")
    elif stage == "heldout":
        report = _load_json(root / "HELDOUT_REPORT.json")
        if report.get("schema") != HELDOUT_SCHEMA:
            raise ArtifactValidationError("heldout report schema mismatch")
    return {
        "root": str(root),
        "stage": stage,
        "status": "PASS",
        "units": completed_units,
        "inventory_sha256": completion["inventory_sha256"],
        "completion_sha256": sha256_file(completion_path),
        "protocol_sha256": protocol["sha256"],
    }


def _replay_records(replay_attempt_roots: Sequence[Path], *,
                    expected_protocol_sha256: str | None = None) -> list[dict[str, Any]]:
    roots = [Path(path) for path in replay_attempt_roots]
    if len(roots) < 2:
        raise ArtifactValidationError(
            "identity reduction requires at least two fresh immutable replay attempts"
        )
    if len({str(path.resolve()) for path in roots}) != len(roots):
        raise ArtifactValidationError("duplicate replay attempt path supplied to reducer")
    records: list[dict[str, Any]] = []
    completion_hashes: set[str] = set()
    replay_instances: set[str] = set()
    for attempt_root in roots:
        summary = validate_attempt(
            attempt_root, expected_stage="replay",
            expected_protocol_sha256=expected_protocol_sha256,
        )
        if summary["completion_sha256"] in completion_hashes:
            raise ArtifactValidationError("fresh replay attempts share a completion hash")
        completion_hashes.add(summary["completion_sha256"])
        attempt_records: list[dict[str, Any]] = []
        for unit_root in sorted((attempt_root / "units").iterdir()):
            if not unit_root.is_dir():
                continue
            manifest, arrays = validate_replay_unit(unit_root)
            instance = manifest.get("provenance", {}).get("replay_instance_id")
            if not instance:
                raise ArtifactValidationError(f"replay instance identity absent: {unit_root}")
            replay_instances.add(str(instance))
            attempt_records.append({
                "attempt_root": attempt_root,
                "attempt_completion_sha256": summary["completion_sha256"],
                "unit_root": unit_root,
                "manifest": manifest,
                "arrays": arrays,
            })
        records.extend(attempt_records)
    if len(replay_instances) < 2:
        raise ArtifactValidationError(
            "cross-replay calibration is vacuous: fewer than two replay instance IDs"
        )
    nonces = [row["manifest"]["capture"]["capture_nonce"] for row in records]
    if len(set(nonces)) != len(nonces):
        raise ArtifactValidationError("capture nonces are not unique across replay replicas")
    return records


def _cross_tensor_keys(record: Mapping[str, Any]) -> list[str]:
    arrays = record["arrays"]
    excluded_prefixes = (
        "packet_", "device_", "conditions__", "empty_conditions__",
        "consumed_condition_", "actual_broadcast_time", "actual_predict_flow_latent",
        "joint_input_", "global_condition_", "extended_condition_",
    )
    return sorted(key for key in arrays if not key.startswith(excluded_prefixes))


def _placement(manifest: Mapping[str, Any]) -> dict[str, Any]:
    provenance = manifest.get("provenance", {})
    return {
        "hostname": provenance.get("hostname"),
        "device_argument": provenance.get("device_argument"),
        "cuda_device_uuid": provenance.get("cuda_device_uuid"),
        "replay_instance_id": provenance.get("replay_instance_id"),
        "PYTHONHASHSEED": provenance.get("PYTHONHASHSEED"),
    }


def _placement_stratum(lhs: Mapping[str, Any], rhs: Mapping[str, Any]) -> str:
    if lhs.get("hostname") != rhs.get("hostname"):
        return "different_node"
    if lhs.get("cuda_device_uuid") and lhs.get("cuda_device_uuid") == rhs.get("cuda_device_uuid"):
        return "same_device_fresh_process"
    if lhs.get("device_argument") == rhs.get("device_argument"):
        return "same_visible_device_fresh_process"
    return "different_device_same_node"


def cross_replay_comparisons(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, float], list[Mapping[str, Any]]] = {}
    for record in records:
        manifest = record["manifest"]
        identity = manifest["identity"]
        key = (
            manifest["packet_parent"]["packet_completion_sha256"],
            str(identity["clip_id"]), float(identity["s"]),
        )
        groups.setdefault(key, []).append(record)
    rows: list[dict[str, Any]] = []
    for (packet_hash, clip_id, s), replicas in sorted(groups.items()):
        attempt_hashes = {row["attempt_completion_sha256"] for row in replicas}
        if len(attempt_hashes) < 2:
            raise ArtifactValidationError(
                f"packet {clip_id}/s={s} lacks two fresh replay attempts"
            )
        key_sets = {tuple(_cross_tensor_keys(row)) for row in replicas}
        if len(key_sets) != 1:
            raise ArtifactValidationError(f"cross-replay tensor inventory conflict: {clip_id}/s={s}")
        for lhs, rhs in itertools.combinations(replicas, 2):
            if lhs["attempt_completion_sha256"] == rhs["attempt_completion_sha256"]:
                continue
            left_placement = _placement(lhs["manifest"])
            right_placement = _placement(rhs["manifest"])
            for tensor_key in next(iter(key_sets)):
                a, b = lhs["arrays"][tensor_key], rhs["arrays"][tensor_key]
                if a.shape != b.shape:
                    raise ArtifactValidationError(
                        f"cross-replay tensor shape conflict: {clip_id}/s={s}/{tensor_key}"
                    )
                rows.append({
                    "comparison_type": CROSS_REPLAY_COMPARISON,
                    "clip_id": clip_id,
                    "s": s,
                    "packet_completion_sha256": packet_hash,
                    "tensor_key": tensor_key,
                    "lhs_attempt_completion_sha256": lhs["attempt_completion_sha256"],
                    "rhs_attempt_completion_sha256": rhs["attempt_completion_sha256"],
                    "lhs_capture_nonce": lhs["manifest"]["capture"]["capture_nonce"],
                    "rhs_capture_nonce": rhs["manifest"]["capture"]["capture_nonce"],
                    "lhs_placement": left_placement,
                    "rhs_placement": right_placement,
                    "stratum": _placement_stratum(left_placement, right_placement),
                    "metrics": _comparison_metrics(a, b),
                })
    if not rows:
        raise ArtifactValidationError("cross-replay comparison reducer produced no rows")
    return rows


def _validate_replay_role_grid(records: Sequence[Mapping[str, Any]],
                               expected_clips: set[str], role: str) -> None:
    by_attempt: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        by_attempt.setdefault(record["attempt_completion_sha256"], []).append(record)
    for attempt_hash, attempt_rows in by_attempt.items():
        identities = [row["manifest"]["identity"] for row in attempt_rows]
        if {str(row["clip_id"]) for row in identities} != expected_clips:
            raise ArtifactValidationError(
                f"replay attempt {attempt_hash} does not contain the exact {role} clip set"
            )
        if {str(row["role"]) for row in identities} != {role}:
            raise ArtifactValidationError(f"replay attempt {attempt_hash} mixes pilot roles")
        repeats = {int(row["repeat_index"]) for row in identities}
        expected = {
            (clip, float(s), repeat)
            for clip in expected_clips for s in S_POINTS for repeat in repeats
        }
        observed = {
            (str(row["clip_id"]), float(row["s"]), int(row["repeat_index"]))
            for row in identities
        }
        if not repeats or observed != expected:
            raise ArtifactValidationError(
                f"replay attempt {attempt_hash} is not a complete clip/progress/repeat grid"
            )


def higher_quantile_times_two(values: Sequence[float], quantile: float = 0.999) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all() or (array < 0).any():
        raise ArtifactValidationError("tolerance values must be nonempty, finite, nonnegative")
    raw = float(np.quantile(array, quantile, method="higher"))
    return raw, 2.0 * raw


def calibrate_attempt(
    replay_attempt_roots: Sequence[Path],
    output_root: Path,
    attempt_id: str,
    *,
    protocol_path: Path,
    protocol_sha256: str,
) -> Path:
    protocol = load_protocol(protocol_path, protocol_sha256)
    records = _replay_records(
        replay_attempt_roots, expected_protocol_sha256=protocol["sha256"]
    )
    manifests = [row["manifest"] for row in records]
    clips = {str(row["identity"]["clip_id"]) for row in manifests}
    if HELDOUT_CLIP in clips:
        raise HeldoutLeakageError("clip 1002 is physically rejected from calibration input")
    _validate_replay_role_grid(records, set(CALIBRATION_CLIPS), "calibration")
    if clips != set(CALIBRATION_CLIPS):
        raise ArtifactValidationError(
            f"calibration requires exactly {CALIBRATION_CLIPS}; observed {sorted(clips)}"
        )
    points_by_clip = {
        clip: {float(row["identity"]["s"]) for row in manifests
               if row["identity"]["clip_id"] == clip}
        for clip in CALIBRATION_CLIPS
    }
    if any(points != set(S_POINTS) for points in points_by_clip.values()):
        raise ArtifactValidationError("calibration replay is missing a registered progress point")
    grouped: dict[str, list[float]] = {}
    for manifest in manifests:
        for row in manifest.get("comparisons", []):
            if not row.get("eligible_for_tolerance"):
                continue
            comparison_type = row.get("comparison_type")
            if comparison_type != TWEEDIE_RECOMPUTE_COMPARISON:
                raise ArtifactValidationError(
                    f"non-equivalent comparison attempted to enter calibration: {comparison_type}"
                )
            grouped.setdefault(TWEEDIE_RECOMPUTE_COMPARISON, []).append(
                float(row["metrics"]["relative_l2"])
            )
    cross_rows = cross_replay_comparisons(records)
    for row in cross_rows:
        grouped.setdefault(f"{CROSS_REPLAY_COMPARISON}:{row['tensor_key']}", []).append(
            float(row["metrics"]["relative_l2"])
        )
    if not any(key.startswith(CROSS_REPLAY_COMPARISON + ":") for key in grouped):
        raise ArtifactValidationError(
            "refusing vacuous tolerance calibration without cross-replay deltas"
        )
    thresholds: dict[str, Any] = {}
    for family, values in sorted(grouped.items()):
        raw, tolerance = higher_quantile_times_two(values)
        thresholds[family] = {
            "metric": "relative_l2", "n": len(values), "quantile": 0.999,
            "method": "higher", "raw_quantile": raw, "multiplier": 2.0,
            "epsilon_floor": None, "tolerance": tolerance,
        }
    root = create_bound_attempt(output_root, "calibration", attempt_id, protocol)
    _write_new_json(root / "CALIBRATION_DELTAS.json", {
        "schema": "sounddecisions_b1_cross_replay_deltas_v1",
        "protocol_sha256": protocol["sha256"],
        "rows": cross_rows,
    })
    payload = {
        "schema": TOLERANCE_SCHEMA,
        "status": "FROZEN",
        "created_utc": utc_now(),
        "calibration_clips": list(CALIBRATION_CLIPS),
        "heldout_clip": HELDOUT_CLIP,
        "heldout_rejected": True,
        "progress_points": list(S_POINTS),
        "protocol_sha256": protocol["sha256"],
        "source_replay_attempts": [
            {
                "path": str(Path(path).resolve()),
                "completion_sha256": sha256_file(Path(path) / "COMPLETED.json"),
            }
            for path in replay_attempt_roots
        ],
        "source_replay_units": len(manifests),
        "selection_method": "q0.999(method=higher) * 2",
        "thresholds": thresholds,
        "eligible_comparison_classes": [TWEEDIE_RECOMPUTE_COMPARISON,
                                         CROSS_REPLAY_COMPARISON],
        "exact_comparison_classes": [
            EQUIVALENT_COMPARISON, INPUT_COMPARISON, TIME_COMPARISON,
            ATOMIC_READBACK_COMPARISON, ATTENTION_SUMMARY_COMPARISON,
        ],
        "forbidden_comparison": FORBIDDEN_GATING_COMPARISON,
        "cross_replay_rows": len(cross_rows),
        "strata": sorted({row["stratum"] for row in cross_rows}),
    }
    _write_new_json(root / "TOLERANCE.json", payload)
    finish_attempt(root, "calibration", expected_units=0)
    return root


def validate_calibration_authority(
    calibration_attempt: Path,
    tolerance_sha256: str,
    *,
    expected_protocol_sha256: str,
) -> dict[str, Any]:
    summary = validate_attempt(
        calibration_attempt, expected_stage="calibration",
        expected_protocol_sha256=expected_protocol_sha256,
    )
    tolerance_file = Path(calibration_attempt) / "TOLERANCE.json"
    observed = sha256_file(tolerance_file)
    if not tolerance_sha256 or observed != tolerance_sha256:
        raise ArtifactValidationError(
            f"mandatory tolerance SHA mismatch: observed {observed}, required {tolerance_sha256}"
        )
    tolerance = _load_json(tolerance_file)
    if (tolerance.get("schema") != TOLERANCE_SCHEMA or tolerance.get("status") != "FROZEN" or
            tolerance.get("protocol_sha256") != expected_protocol_sha256 or
            tolerance.get("calibration_clips") != list(CALIBRATION_CLIPS) or
            not tolerance.get("heldout_rejected")):
        raise ArtifactValidationError("immutable calibration attempt is not valid heldout authority")
    if not any(key.startswith(CROSS_REPLAY_COMPARISON + ":")
               for key in tolerance.get("thresholds", {})):
        raise ArtifactValidationError("calibration authority lacks cross-replay thresholds")
    return {
        "calibration_attempt": str(Path(calibration_attempt).resolve()),
        "calibration_completion_sha256": summary["completion_sha256"],
        "tolerance_sha256": observed,
        "protocol_sha256": expected_protocol_sha256,
    }


def heldout_attempt(
    replay_attempt_roots: Sequence[Path],
    calibration_attempt: Path,
    output_root: Path,
    attempt_id: str,
    *,
    tolerance_sha256: str,
    protocol_path: Path,
    protocol_sha256: str,
) -> Path:
    protocol = load_protocol(protocol_path, protocol_sha256)
    authority = validate_calibration_authority(
        calibration_attempt, tolerance_sha256,
        expected_protocol_sha256=protocol["sha256"],
    )
    tolerance_file = Path(calibration_attempt) / "TOLERANCE.json"
    before_bytes = tolerance_file.read_bytes()
    tolerance_hash = sha256_bytes(before_bytes)
    tolerance = json.loads(before_bytes)
    records = _replay_records(
        replay_attempt_roots, expected_protocol_sha256=protocol["sha256"]
    )
    manifests = [row["manifest"] for row in records]
    _validate_replay_role_grid(records, {HELDOUT_CLIP}, "heldout")
    clips = {str(row["identity"]["clip_id"]) for row in manifests}
    if clips != {HELDOUT_CLIP}:
        raise ArtifactValidationError(f"heldout reducer accepts only clip 1002, got {clips}")
    if {float(row["identity"]["s"]) for row in manifests} != set(S_POINTS):
        raise ArtifactValidationError("heldout replay is missing a registered progress point")
    failures: list[dict[str, Any]] = []
    evaluated = 0
    thresholds = tolerance.get("thresholds", {})
    for manifest in manifests:
        replay_authority = manifest.get("provenance", {}).get("calibration_authority")
        if replay_authority != authority:
            raise ArtifactValidationError(
                "heldout replay was not launched under this exact frozen calibration authority"
            )
    for manifest in manifests:
        for row in manifest.get("comparisons", []):
            if not row.get("eligible_for_tolerance"):
                continue
            family = row["comparison_type"]
            if family != TWEEDIE_RECOMPUTE_COMPARISON or family not in thresholds:
                raise ArtifactValidationError(f"tolerance lacks within-unit family {family}")
            evaluated += 1
            value = float(row["metrics"]["relative_l2"])
            limit = float(thresholds[family]["tolerance"])
            if value > limit:
                failures.append({
                        "unit_id": manifest["unit_id"],
                        "clip_id": manifest["identity"]["clip_id"],
                        "s": manifest["identity"]["s"],
                        "repeat_index": manifest["identity"]["repeat_index"],
                        "capture_nonce": row.get("capture_nonce"),
                        "pass_role": row.get("pass_role"),
                        "site": row.get("site"),
                        "comparison_type": family,
                        "metric": "relative_l2",
                        "value": value,
                        "tolerance": limit,
                })
    cross_rows = cross_replay_comparisons(records)
    for row in cross_rows:
        family = f"{CROSS_REPLAY_COMPARISON}:{row['tensor_key']}"
        if family not in thresholds:
            raise ArtifactValidationError(f"frozen tolerance lacks heldout family {family}")
        evaluated += 1
        value = float(row["metrics"]["relative_l2"])
        limit = float(thresholds[family]["tolerance"])
        if value > limit:
            failures.append({
                "unit_id": f"clip{row['clip_id']}__s{row['s']:.2f}",
                "clip_id": row["clip_id"], "s": row["s"],
                "comparison_type": CROSS_REPLAY_COMPARISON,
                "tensor_key": row["tensor_key"], "stratum": row["stratum"],
                "metric": "relative_l2", "value": value, "tolerance": limit,
                "lhs_capture_nonce": row["lhs_capture_nonce"],
                "rhs_capture_nonce": row["rhs_capture_nonce"],
            })
    after_bytes = tolerance_file.read_bytes()
    if after_bytes != before_bytes:
        raise ArtifactValidationError("tolerance file mutated while applying heldout gate")
    root = create_bound_attempt(output_root, "heldout", attempt_id, protocol)
    _write_new_json(root / "HELDOUT_DELTAS.json", {
        "schema": "sounddecisions_b1_cross_replay_deltas_v1",
        "protocol_sha256": protocol["sha256"], "rows": cross_rows,
    })
    payload = {
        "schema": HELDOUT_SCHEMA,
        "status": "PASS" if not failures else "ENGINEERING_FAILURE",
        "created_utc": utc_now(),
        "heldout_clip": HELDOUT_CLIP,
        "progress_points": list(S_POINTS),
        "protocol_sha256": protocol["sha256"],
        "source_replay_attempts": [str(Path(path).resolve()) for path in replay_attempt_roots],
        "source_replay_units": len(manifests),
        "tolerance_path": str(tolerance_file.resolve()),
        "tolerance_sha256": tolerance_hash,
        "calibration_authority": authority,
        "tolerance_unchanged": True,
        "evaluated_metrics": evaluated,
        "failure_count": len(failures),
        "failures": failures,
    }
    _write_new_json(root / "HELDOUT_REPORT.json", payload)
    finish_attempt(root, "heldout", expected_units=0)
    return root


__all__ = [
    "ATOMIC_READBACK_COMPARISON", "ATTENTION_SUMMARY_COMPARISON",
    "ArtifactValidationError", "CALIBRATION_CLIPS", "CONDITION_FIELDS",
    "CROSS_REPLAY_COMPARISON", "EQUIVALENT_COMPARISON",
    "FORBIDDEN_GATING_COMPARISON", "HELDOUT_CLIP",
    "HeldoutLeakageError", "ImmutableArtifactError", "PASS_ROLES", "S_POINTS",
    "SameForwardCapture", "TWEEDIE_RECOMPUTE_COMPARISON", "array_sha256",
    "bind_protocol", "calibrate_attempt", "create_attempt", "create_bound_attempt",
    "create_selection_attempt", "describe_array", "finish_attempt",
    "heldout_attempt", "higher_quantile_times_two", "make_packet_attempt",
    "cross_replay_comparisons", "load_protocol", "phase1_identity", "phase1_rng",
    "replay_attempt", "require_offline_environment", "sha256_file",
    "unit_id", "validate_attempt", "validate_calibration_authority",
    "validate_packet_unit", "validate_protocol_binding", "validate_replay_unit",
    "verify_asset_contract",
    "write_packet_unit", "write_replay_unit",
]
