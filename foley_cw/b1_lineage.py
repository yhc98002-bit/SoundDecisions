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
    return {
        "dtype": arr.dtype.str,
        "shape": list(arr.shape),
        "sha256": array_sha256(arr),
        "nbytes": int(arr.nbytes),
    }


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


def environment_provenance(device: str, mmaudio_root: Path | None = None) -> dict[str, Any]:
    info: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "executable": sys.executable,
        "device_argument": str(device),
    }
    try:
        import torch
        info.update({
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
        })
        if str(device).startswith("cuda") and torch.cuda.is_available():
            index = torch.device(device).index
            index = torch.cuda.current_device() if index is None else index
            info["cuda_device_index"] = int(index)
            info["cuda_device_name"] = torch.cuda.get_device_name(index)
    except Exception as exc:  # pragma: no cover - provenance must not hide the core error
        info["torch_probe_error"] = f"{type(exc).__name__}: {exc}"
    if mmaudio_root is not None:
        info["mmaudio_root"] = str(Path(mmaudio_root).resolve())
        info["mmaudio_git_commit"] = _git_head(Path(mmaudio_root))
    return info


def _require_asset_roots(mmaudio_root: Path, weights_dir: Path, clips_root: Path) -> None:
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


def create_selection_attempt(
    output_root: Path,
    attempt_id: str,
    *,
    mmaudio_root: Path,
    weights_dir: Path,
    clips_root: Path,
) -> Path:
    """Freeze the outcome-independent four-calibration/one-held-out pilot selection."""
    _require_asset_roots(mmaudio_root, weights_dir, clips_root)
    root = create_attempt(output_root, "selection", attempt_id)
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
    _write_new_npz(root / "arrays.npz", normalized)
    identity = {"clip_id": str(clip_id), "role": role, "s": float(s),
                "phase1": phase1_identity(str(clip_id))}
    manifest = {
        "schema": PACKET_SCHEMA,
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
    if manifest.get("schema") != PACKET_SCHEMA:
        raise ArtifactValidationError(f"wrong packet schema: {root}")
    if manifest.get("unit_id") != root.name:
        raise ArtifactValidationError(f"packet unit id conflicts with directory: {root}")
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
) -> Path:
    """Generate the five canonical j=12 trajectories and persist forty replay packets."""
    _require_asset_roots(mmaudio_root, weights_dir, clips_root)
    validate_attempt(selection_attempt, expected_stage="selection")
    selection = _load_json(Path(selection_attempt) / "selection.json")
    expected_roots = selection.get("asset_roots", {})
    current_roots = {
        "mmaudio_root": str(Path(mmaudio_root).resolve()),
        "weights_dir": str(Path(weights_dir).resolve()),
        "clips_root": str(Path(clips_root).resolve()),
    }
    if expected_roots != current_roots:
        raise ArtifactValidationError("asset roots conflict with frozen selection")
    root = create_attempt(output_root, "packets", attempt_id)

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
    provenance = environment_provenance(device, mmaudio_root)
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
        self._active_site: str | None = None
        self._pass_index = -1
        self._nonce = ""
        self._arrays: dict[str, np.ndarray] = {}
        self._tokens: list[dict[str, Any]] = []
        self._attention: list[dict[str, Any]] = []

    @staticmethod
    def _key(kind: str, role: str, site: str) -> str:
        return f"{kind}__{role}__{site.replace('.', '_')}"

    def _pre_hook(self, site: str):
        def hook(_module: Any, _args: Any) -> None:
            if site == "joint.0":
                self._pass_index += 1
            if self._pass_index < 0 or self._pass_index >= len(PASS_ROLES):
                raise ArtifactValidationError(
                    f"unexpected predict_flow pass index {self._pass_index} at {site}"
                )
            self._active_site = site
        return hook

    def _post_hook(self, site: str, is_joint: bool):
        def hook(_module: Any, _args: Any, output: Any) -> None:
            import torch
            role = PASS_ROLES[self._pass_index]
            latent = output[0] if is_joint else output
            if latent.ndim != 3 or latent.shape[0] != 1:
                raise ArtifactValidationError(
                    f"post-block {site} expected (1,tokens,dim), got {tuple(latent.shape)}"
                )
            native_fp32 = latent.detach().to(dtype=torch.float32)
            # These are intentionally two spellings of the identical operation,
            # on the same native fp32 tensor and in the same hook invocation.
            pooled_original = native_fp32.mean(dim=1)
            pooled_repaired = torch.mean(native_fp32, dim=1)
            token_array = native_fp32.cpu().numpy()
            original_array = pooled_original.cpu().numpy()
            repaired_array = pooled_repaired.cpu().numpy()
            token_key = self._key("post_block_tokens_fp32", role, site)
            original_key = self._key("pooled_original_fp32", role, site)
            repaired_key = self._key("pooled_repaired_fp32", role, site)
            stats_key = self._key("token_stats_fp32", role, site)
            stats = torch.stack((
                native_fp32.mean(), native_fp32.std(unbiased=False), native_fp32.min(),
                native_fp32.max(), torch.linalg.vector_norm(native_fp32),
            )).cpu().numpy()
            self._arrays[token_key] = token_array
            self._arrays[original_key] = original_array
            self._arrays[repaired_key] = repaired_array
            self._arrays[stats_key] = stats
            self._tokens.append({
                "capture_nonce": self._nonce,
                "pass_index": self._pass_index,
                "pass_role": role,
                "site": site,
                "hook_site": f"{type(_module).__module__}.{type(_module).__qualname__}",
                "native_dtype": str(latent.dtype),
                "native_shape": list(latent.shape),
                "persisted_tokens": token_key,
                "pooled_original": original_key,
                "pooled_repaired": repaired_key,
                "token_stats": stats_key,
                "token_stats_order": ["mean", "std_population", "min", "max", "l2"],
                "pool_operation_original": "native_tensor.float().mean(dim=1)",
                "pool_operation_repaired": "torch.mean(native_tensor.float(), dim=1)",
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
            prob_key = prefix + "__derived_probability_map"
            q_fp32 = q.detach().to(torch.float32)
            k_fp32 = k.detach().to(torch.float32)
            probability = torch.softmax(
                torch.matmul(q_fp32, k_fp32.transpose(-2, -1)) /
                float(q.shape[-1]) ** 0.5,
                dim=-1,
            )
            self._arrays[q_key] = q.detach().cpu().numpy()
            self._arrays[k_key] = k.detach().cpu().numpy()
            self._arrays[v_key] = v.detach().cpu().numpy()
            self._arrays[out_key] = actual_output.detach().cpu().numpy()
            self._arrays[prob_key] = probability.cpu().numpy()
            self._attention.append({
                "capture_nonce": self._nonce,
                "pass_index": self._pass_index,
                "pass_role": role,
                "site": site,
                "q": q_key,
                "k": k_key,
                "v": v_key,
                "actual_attention_output": out_key,
                "probability_map": prob_key,
                "probability_map_provenance": (
                    "RECOMPUTED_DERIVED softmax(fp32(Q)@fp32(K)^T/sqrt(d)); "
                    "not exposed by scaled_dot_product_attention"
                ),
            })
        return actual_output

    @contextmanager
    def armed(self, capture_nonce: str):
        if self._handles or self._original_attention is not None:
            raise ArtifactValidationError("capture is already armed")
        self._nonce = str(capture_nonce)
        self._pass_index = -1
        self._arrays = {}
        self._tokens = []
        self._attention = []
        for index, block in enumerate(self.joint):
            site = f"joint.{index}"
            self._handles.append(block.register_forward_pre_hook(self._pre_hook(site)))
            self._handles.append(block.register_forward_hook(self._post_hook(site, True)))
        for index, block in enumerate(self.fused):
            site = f"fused.{index}"
            self._handles.append(block.register_forward_pre_hook(self._pre_hook(site)))
            self._handles.append(block.register_forward_hook(self._post_hook(site, False)))
        self._original_attention = self.attention_module.attention
        self.attention_module.attention = self._wrapped_attention
        try:
            yield self
        finally:
            self.attention_module.attention = self._original_attention
            self._original_attention = None
            for handle in self._handles:
                handle.remove()
            self._handles = []
            self._active_site = None

    def finish(self) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
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


def _capture_nonce(packet_completion_sha256: str, repeat_index: int) -> str:
    return sha256_bytes(
        f"{packet_completion_sha256}|repeat={int(repeat_index)}".encode("utf-8")
    )[:32]


def _write_float_wav(path: Path, wav: np.ndarray, sample_rate: int = 16000) -> None:
    import soundfile as sf
    if path.exists():
        raise ImmutableArtifactError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(wav, dtype=np.float32), sample_rate, subtype="FLOAT")


def _comparison_records(arrays: Mapping[str, np.ndarray], capture: Mapping[str, Any]
                        ) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in capture["tokens"]:
        lhs_key = record["pooled_original"]
        rhs_key = record["pooled_repaired"]
        lhs, rhs = arrays[lhs_key], arrays[rhs_key]
        rows.append({
            "eligible_for_tolerance": True,
            "comparison_type": EQUIVALENT_COMPARISON,
            "pass_role": record["pass_role"],
            "site": record["site"],
            "capture_nonce": record["capture_nonce"],
            "lhs": lhs_key,
            "rhs": rhs_key,
            "lhs_operation": record["pool_operation_original"],
            "rhs_operation": record["pool_operation_repaired"],
            "metrics": {"relative_l2": _relative_l2(lhs, rhs),
                        "max_abs": _max_abs(lhs, rhs)},
        })
    packet_x = arrays["packet_x_s_fp32"]
    device_x = arrays["device_latent_fp32"][0]
    rows.append({
        "eligible_for_tolerance": True,
        "comparison_type": INPUT_COMPARISON,
        "pass_role": "input",
        "site": "ode_wrapper_input_latent",
        "capture_nonce": capture["capture_nonce"],
        "lhs": "packet_x_s_fp32",
        "rhs": "device_latent_fp32[0]",
        "metrics": {"relative_l2": _relative_l2(packet_x, device_x),
                    "max_abs": _max_abs(packet_x, device_x)},
    })
    packet_t = arrays["packet_model_time_fp32"]
    device_t = arrays["device_time_fp32"]
    rows.append({
        "eligible_for_tolerance": True,
        "comparison_type": TIME_COMPARISON,
        "pass_role": "input",
        "site": "ode_wrapper_input_time",
        "capture_nonce": capture["capture_nonce"],
        "lhs": "packet_model_time_fp32",
        "rhs": "device_time_fp32",
        "metrics": {"relative_l2": _relative_l2(packet_t, device_t),
                    "max_abs": _max_abs(packet_t, device_t)},
    })
    return rows


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
    identity = packet_manifest["identity"]
    uid = unit_id(identity["clip_id"], identity["s"], repeat_index)
    root = Path(attempt_root) / "units" / uid
    try:
        root.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise ImmutableArtifactError(f"replay unit already exists: {root}") from exc
    normalized = {key: np.asarray(value) for key, value in arrays.items()}
    _write_new_npz(root / "arrays.npz", normalized)
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
        "arrays": {key: describe_array(value) for key, value in normalized.items()},
        "external_preview": {
            "path": "external_preview.wav",
            "sha256": sha256_file(root / "external_preview.wav"),
            "sample_rate": 16000,
            "subtype": "FLOAT",
            "samples": int(np.asarray(preview_wav).size),
            "parent": "tweedie_latent_from_same_ode_wrapper_evaluation",
        },
        "capture": dict(capture_metadata),
        "comparisons": _comparison_records(normalized, capture_metadata),
        "gating_policy": {
            "eligible_comparison_classes": [EQUIVALENT_COMPARISON, INPUT_COMPARISON,
                                             TIME_COMPARISON],
            "forbidden_comparison": FORBIDDEN_GATING_COMPARISON,
            "forbidden_comparison_present": False,
        },
        "provenance": dict(provenance),
    }
    _write_new_json(root / "manifest.json", manifest)
    _finish_unit(root, "replay", replay_identity)
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
    if manifest.get("schema") != REPLAY_SCHEMA or manifest.get("unit_id") != root.name:
        raise ArtifactValidationError(f"replay schema/unit conflict: {root}")
    if manifest.get("arrays_file_sha256") != sha256_file(root / "arrays.npz"):
        raise ArtifactValidationError(f"replay npz hash mismatch: {root}")
    arrays = _load_npz_verified(root / "arrays.npz", manifest.get("arrays", {}))
    preview = manifest.get("external_preview", {})
    if preview.get("sha256") != sha256_file(root / "external_preview.wav"):
        raise ArtifactValidationError(f"preview hash mismatch: {root}")
    capture = manifest.get("capture", {})
    if not capture.get("one_ode_wrapper_evaluation") or capture.get("pass_roles") != list(PASS_ROLES):
        raise ArtifactValidationError(f"same-forward/pass-role contract absent: {root}")
    if capture.get("observed_passes") != 2:
        raise ArtifactValidationError(f"replay did not capture exactly two passes: {root}")
    policy = manifest.get("gating_policy", {})
    if policy.get("forbidden_comparison_present"):
        raise ArtifactValidationError(f"forbidden reduction-order comparison gates {root}")
    eligible = set(policy.get("eligible_comparison_classes", []))
    for comparison in manifest.get("comparisons", []):
        if comparison.get("eligible_for_tolerance") and comparison.get("comparison_type") not in eligible:
            raise ArtifactValidationError(f"ineligible comparison marked gating: {root}")
        for value in comparison.get("metrics", {}).values():
            if not np.isfinite(float(value)) or float(value) < 0:
                raise ArtifactValidationError(f"invalid comparison metric: {root}")
    return manifest, arrays


def _real_replay_one(
    backend: Any,
    panns_model: Any,
    packet_root: Path,
    repeat_index: int,
    device: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any], np.ndarray]:
    import torch
    from mmaudio.model.networks import PreprocessedConditions
    from mmaudio.model import transformer_layers

    packet_manifest, packet = validate_packet_unit(packet_root)
    conditions = _rehydrate_conditions(packet, "conditions", PreprocessedConditions, device)
    empty_conditions = _rehydrate_conditions(
        packet, "empty_conditions", PreprocessedConditions, device
    )
    x_device = torch.from_numpy(np.ascontiguousarray(packet["x_s"])).to(
        device=device, dtype=backend.dtype
    ).unsqueeze(0)
    t_device = torch.tensor(float(packet["model_time"]), device=device, dtype=backend.dtype)
    packet_completion_hash = sha256_file(packet_root / "COMPLETED.json")
    nonce = _capture_nonce(packet_completion_hash, repeat_index)
    collector = SameForwardCapture(backend.net, transformer_layers)
    with torch.inference_mode():
        with collector.armed(nonce):
            velocity = backend.net.ode_wrapper(
                t_device, x_device, conditions, empty_conditions, 1.0
            )
        capture_arrays, capture_metadata = collector.finish()
        tweedie = x_device + (1.0 - t_device) * velocity
        unnormalized = backend.net.unnormalize(tweedie)
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
        "returned_velocity_fp32": velocity.detach().float().cpu().numpy(),
        "tweedie_latent_fp32": tweedie.detach().float().cpu().numpy(),
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
    capture_metadata["conditioning_fields"] = list(CONDITION_FIELDS)
    capture_metadata["conditioning_complete"] = True
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
) -> Path:
    if role not in {"calibration", "heldout"}:
        raise ValueError("replay role must be calibration or heldout")
    if repeats < 1 or repeat_offset < 0:
        raise ValueError("repeats must be >=1 and repeat_offset >=0")
    _require_asset_roots(mmaudio_root, weights_dir, clips_root)
    validate_attempt(packet_attempt, expected_stage="packets")
    packet_units: list[tuple[Path, dict[str, Any]]] = []
    for packet_root in sorted((Path(packet_attempt) / "units").iterdir()):
        if not packet_root.is_dir():
            continue
        manifest, _ = validate_packet_unit(packet_root)
        if manifest["identity"]["role"] == role:
            packet_units.append((packet_root, manifest))
    expected_clips = set(CALIBRATION_CLIPS if role == "calibration" else (HELDOUT_CLIP,))
    if {manifest["identity"]["clip_id"] for _, manifest in packet_units} != expected_clips:
        raise ArtifactValidationError(f"packet attempt does not contain exact {role} clip set")
    if len(packet_units) != len(expected_clips) * len(S_POINTS):
        raise ArtifactValidationError(f"packet role {role} has incomplete progress grid")

    root = create_attempt(output_root, "replay", attempt_id)
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
    provenance = environment_provenance(device, mmaudio_root)
    provenance.update({
        "panns_checkpoint": str(panns_path.resolve()),
        "panns_checkpoint_sha256": sha256_file(panns_path),
        "packet_attempt_completion_sha256": sha256_file(Path(packet_attempt) / "COMPLETED.json"),
        "repeat_count": int(repeats),
        "repeat_offset": int(repeat_offset),
        "role": role,
    })
    count = 0
    for packet_root, packet_manifest in packet_units:
        packet_manifest_checked, packet_arrays = validate_packet_unit(packet_root)
        if packet_manifest_checked != packet_manifest:
            raise ArtifactValidationError(f"packet changed during replay: {packet_root}")
        for repeat_index in range(repeat_offset, repeat_offset + repeats):
            arrays, capture, preview = _real_replay_one(
                backend, panns_model, packet_root, repeat_index, device
            )
            write_replay_unit(
                root, packet_manifest=packet_manifest, packet_arrays=packet_arrays,
                packet_unit_root=packet_root, repeat_index=repeat_index, arrays=arrays,
                capture_metadata=capture, preview_wav=preview, provenance=provenance,
            )
            count += 1
    finish_attempt(root, "replay", expected_units=len(packet_units) * repeats)
    return root


def validate_attempt(root: Path, expected_stage: str | None = None) -> dict[str, Any]:
    root = Path(root)
    completion_path = root / "COMPLETED.json"
    if not completion_path.is_file():
        raise ArtifactValidationError(f"partial attempt (no completion journal): {root}")
    completion = _load_json(completion_path)
    if completion.get("schema") != ATTEMPT_SCHEMA or completion.get("status") != "COMPLETE":
        raise ArtifactValidationError(f"invalid attempt completion: {root}")
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
    }


def _replay_manifests(replay_attempt_root: Path) -> list[dict[str, Any]]:
    validate_attempt(replay_attempt_root, expected_stage="replay")
    manifests: list[dict[str, Any]] = []
    for unit_root in sorted((Path(replay_attempt_root) / "units").iterdir()):
        if unit_root.is_dir():
            manifest, _ = validate_replay_unit(unit_root)
            manifests.append(manifest)
    return manifests


def higher_quantile_times_two(values: Sequence[float], quantile: float = 0.999) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all() or (array < 0).any():
        raise ArtifactValidationError("tolerance values must be nonempty, finite, nonnegative")
    raw = float(np.quantile(array, quantile, method="higher"))
    return raw, 2.0 * raw


def calibrate_attempt(
    replay_attempt_root: Path,
    output_root: Path,
    attempt_id: str,
) -> Path:
    manifests = _replay_manifests(replay_attempt_root)
    clips = {str(row["identity"]["clip_id"]) for row in manifests}
    if HELDOUT_CLIP in clips:
        raise HeldoutLeakageError("clip 1002 is physically rejected from calibration input")
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
    grouped: dict[str, dict[str, list[float]]] = {}
    for manifest in manifests:
        for row in manifest.get("comparisons", []):
            if not row.get("eligible_for_tolerance"):
                continue
            comparison_type = row.get("comparison_type")
            if comparison_type not in {EQUIVALENT_COMPARISON, INPUT_COMPARISON, TIME_COMPARISON}:
                raise ArtifactValidationError(
                    f"non-equivalent comparison attempted to enter calibration: {comparison_type}"
                )
            for metric, value in row["metrics"].items():
                grouped.setdefault(comparison_type, {}).setdefault(metric, []).append(float(value))
    thresholds: dict[str, Any] = {}
    for comparison_type, metrics in sorted(grouped.items()):
        thresholds[comparison_type] = {}
        for metric, values in sorted(metrics.items()):
            raw, tolerance = higher_quantile_times_two(values)
            thresholds[comparison_type][metric] = {
                "n": len(values),
                "quantile": 0.999,
                "method": "higher",
                "raw_quantile": raw,
                "multiplier": 2.0,
                "tolerance": tolerance,
            }
    root = create_attempt(output_root, "calibration", attempt_id)
    payload = {
        "schema": TOLERANCE_SCHEMA,
        "status": "FROZEN",
        "created_utc": utc_now(),
        "calibration_clips": list(CALIBRATION_CLIPS),
        "heldout_clip": HELDOUT_CLIP,
        "heldout_rejected": True,
        "progress_points": list(S_POINTS),
        "source_replay_completion_sha256": sha256_file(
            Path(replay_attempt_root) / "COMPLETED.json"
        ),
        "source_replay_units": len(manifests),
        "selection_method": "q0.999(method=higher) * 2",
        "thresholds": thresholds,
        "eligible_comparison_classes": [EQUIVALENT_COMPARISON, INPUT_COMPARISON,
                                         TIME_COMPARISON],
        "forbidden_comparison": FORBIDDEN_GATING_COMPARISON,
    }
    _write_new_json(root / "TOLERANCE.json", payload)
    finish_attempt(root, "calibration", expected_units=0)
    return root


def heldout_attempt(
    replay_attempt_root: Path,
    tolerance_file: Path,
    output_root: Path,
    attempt_id: str,
    *,
    expected_tolerance_sha256: str | None = None,
) -> Path:
    tolerance_file = Path(tolerance_file)
    if not tolerance_file.is_file():
        raise FileNotFoundError(tolerance_file)
    before_bytes = tolerance_file.read_bytes()
    tolerance_hash = sha256_bytes(before_bytes)
    if expected_tolerance_sha256 is not None and tolerance_hash != expected_tolerance_sha256:
        raise ArtifactValidationError("tolerance hash does not match required hash")
    tolerance = json.loads(before_bytes)
    if tolerance.get("schema") != TOLERANCE_SCHEMA or tolerance.get("status") != "FROZEN":
        raise ArtifactValidationError("heldout reducer requires a frozen tolerance JSON")
    if tolerance.get("calibration_clips") != list(CALIBRATION_CLIPS) or not tolerance.get(
        "heldout_rejected"
    ):
        raise ArtifactValidationError("tolerance calibration/heldout provenance is invalid")
    manifests = _replay_manifests(replay_attempt_root)
    clips = {str(row["identity"]["clip_id"]) for row in manifests}
    if clips != {HELDOUT_CLIP}:
        raise ArtifactValidationError(f"heldout reducer accepts only clip 1002, got {clips}")
    if {float(row["identity"]["s"]) for row in manifests} != set(S_POINTS):
        raise ArtifactValidationError("heldout replay is missing a registered progress point")
    failures: list[dict[str, Any]] = []
    evaluated = 0
    thresholds = tolerance.get("thresholds", {})
    for manifest in manifests:
        for row in manifest.get("comparisons", []):
            if not row.get("eligible_for_tolerance"):
                continue
            comparison_type = row["comparison_type"]
            if comparison_type not in thresholds:
                raise ArtifactValidationError(f"tolerance lacks {comparison_type}")
            for metric, value in row["metrics"].items():
                if metric not in thresholds[comparison_type]:
                    raise ArtifactValidationError(
                        f"tolerance lacks {comparison_type}/{metric}"
                    )
                evaluated += 1
                limit = float(thresholds[comparison_type][metric]["tolerance"])
                if float(value) > limit:
                    failures.append({
                        "unit_id": manifest["unit_id"],
                        "clip_id": manifest["identity"]["clip_id"],
                        "s": manifest["identity"]["s"],
                        "repeat_index": manifest["identity"]["repeat_index"],
                        "capture_nonce": row.get("capture_nonce"),
                        "pass_role": row.get("pass_role"),
                        "site": row.get("site"),
                        "comparison_type": comparison_type,
                        "metric": metric,
                        "value": float(value),
                        "tolerance": limit,
                    })
    after_bytes = tolerance_file.read_bytes()
    if after_bytes != before_bytes:
        raise ArtifactValidationError("tolerance file mutated while applying heldout gate")
    root = create_attempt(output_root, "heldout", attempt_id)
    payload = {
        "schema": HELDOUT_SCHEMA,
        "status": "PASS" if not failures else "ENGINEERING_FAILURE",
        "created_utc": utc_now(),
        "heldout_clip": HELDOUT_CLIP,
        "progress_points": list(S_POINTS),
        "source_replay_completion_sha256": sha256_file(
            Path(replay_attempt_root) / "COMPLETED.json"
        ),
        "source_replay_units": len(manifests),
        "tolerance_path": str(tolerance_file.resolve()),
        "tolerance_sha256": tolerance_hash,
        "tolerance_unchanged": True,
        "evaluated_metrics": evaluated,
        "failure_count": len(failures),
        "failures": failures,
    }
    _write_new_json(root / "HELDOUT_REPORT.json", payload)
    finish_attempt(root, "heldout", expected_units=0)
    return root


__all__ = [
    "ArtifactValidationError", "CALIBRATION_CLIPS", "CONDITION_FIELDS",
    "EQUIVALENT_COMPARISON", "FORBIDDEN_GATING_COMPARISON", "HELDOUT_CLIP",
    "HeldoutLeakageError", "ImmutableArtifactError", "PASS_ROLES", "S_POINTS",
    "SameForwardCapture", "array_sha256", "calibrate_attempt", "create_attempt",
    "create_selection_attempt", "describe_array", "finish_attempt",
    "heldout_attempt", "higher_quantile_times_two", "make_packet_attempt",
    "phase1_identity", "phase1_rng", "replay_attempt", "sha256_file",
    "unit_id", "validate_attempt", "validate_packet_unit", "validate_replay_unit",
    "write_packet_unit", "write_replay_unit",
]
