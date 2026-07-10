#!/usr/bin/env python3
"""Small helpers for ORBIT_STATE.json.

This module intentionally avoids third-party dependencies. The JSON Schema in
`schemas/orbit_state.schema.json` is the external contract; this file provides a
lightweight runtime shape check for tools that need to read or construct the
state object.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional


SCHEMA_VERSION = "0.1"
ORBIT_STATE_REL_PATH = "orbit-research/ORBIT_STATE.json"

VALID_STOPS = {"NONE", "STOP_A", "STOP_B", "STOP_C", "STOP_D", "COMPLETED"}
VALID_STATUSES = {"in_progress", "paused", "blocked", "completed"}
VALID_PAUSE_REASONS = {
    "stop_review",
    "missing_prereq",
    "gate_failed",
    "codex_review_needed",
    "codex_review_imported",
    "ambiguous_resume",
    "external_dependency",
    None,
}
VALID_BLOCKER_KINDS = {
    "missing_artifact",
    "bad_verdict",
    "codex_unavailable",
    "stale_state",
    "legacy_conflict",
}

DEFAULT_CANONICAL_PACKS = {
    "proposal_pack": "proposal/proposal_pack.json",
    "experiment_pack": "experiment/experiment_pack.json",
    "claim_ledger": "claims/claim_ledger.json",
    "paper_package": "paper/paper_package.json",
}


class OrbitStateError(ValueError):
    """Raised when an ORBIT_STATE object is malformed."""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def orbit_state_path(repo: Path) -> Path:
    return repo / ORBIT_STATE_REL_PATH


def make_blocker(
    blocker_id: str,
    kind: str,
    artifact: str,
    message: str,
    safe_next_command: Optional[str],
) -> Dict[str, Any]:
    return {
        "id": blocker_id,
        "kind": kind,
        "artifact": artifact,
        "message": message,
        "safe_next_command": safe_next_command,
    }


def make_state(
    current_stop: str = "NONE",
    current_skill: Optional[str] = None,
    current_phase: str = "not_started",
    status: str = "paused",
    pause_reason: Optional[str] = "ambiguous_resume",
    blockers: Optional[List[Mapping[str, Any]]] = None,
    canonical_packs: Optional[Mapping[str, str]] = None,
    legacy_artifacts_detected: Optional[List[str]] = None,
    safe_next_command: Optional[str] = None,
    updated_at: Optional[str] = None,
) -> Dict[str, Any]:
    state = {
        "schema_version": SCHEMA_VERSION,
        "current_stop": current_stop,
        "current_skill": current_skill,
        "current_phase": current_phase,
        "status": status,
        "pause_reason": pause_reason,
        "blockers": [dict(blocker) for blocker in (blockers or [])],
        "canonical_packs": dict(canonical_packs or DEFAULT_CANONICAL_PACKS),
        "legacy_artifacts_detected": sorted(legacy_artifacts_detected or []),
        "safe_next_command": safe_next_command,
        "updated_at": updated_at or utc_now_iso(),
    }
    validate_state_shape(state)
    return state


def read_state(repo: Path) -> Optional[Dict[str, Any]]:
    path = orbit_state_path(repo)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    validate_state_shape(data)
    return data


def write_state(repo: Path, state: Mapping[str, Any]) -> Path:
    validate_state_shape(state)
    path = orbit_state_path(repo)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(dict(state), handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def validate_state_shape(state: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "current_stop",
        "current_skill",
        "current_phase",
        "status",
        "pause_reason",
        "blockers",
        "canonical_packs",
        "legacy_artifacts_detected",
        "safe_next_command",
        "updated_at",
    }
    missing = sorted(required - set(state))
    if missing:
        raise OrbitStateError("ORBIT_STATE missing required keys: %s" % ", ".join(missing))

    if state.get("schema_version") != SCHEMA_VERSION:
        raise OrbitStateError("unsupported ORBIT_STATE schema_version: %r" % state.get("schema_version"))
    if state.get("current_stop") not in VALID_STOPS:
        raise OrbitStateError("invalid current_stop: %r" % state.get("current_stop"))
    if state.get("status") not in VALID_STATUSES:
        raise OrbitStateError("invalid status: %r" % state.get("status"))
    if state.get("pause_reason") not in VALID_PAUSE_REASONS:
        raise OrbitStateError("invalid pause_reason: %r" % state.get("pause_reason"))
    if not isinstance(state.get("blockers"), list):
        raise OrbitStateError("blockers must be a list")
    if not isinstance(state.get("canonical_packs"), dict):
        raise OrbitStateError("canonical_packs must be an object")
    if not isinstance(state.get("legacy_artifacts_detected"), list):
        raise OrbitStateError("legacy_artifacts_detected must be a list")

    for key in DEFAULT_CANONICAL_PACKS:
        if key not in state["canonical_packs"]:
            raise OrbitStateError("canonical_packs missing %s" % key)

    for index, blocker in enumerate(state["blockers"]):
        validate_blocker_shape(blocker, index)


def validate_blocker_shape(blocker: Mapping[str, Any], index: int = 0) -> None:
    required = {"id", "kind", "artifact", "message", "safe_next_command"}
    missing = sorted(required - set(blocker))
    if missing:
        raise OrbitStateError("blocker %d missing keys: %s" % (index, ", ".join(missing)))
    if blocker.get("kind") not in VALID_BLOCKER_KINDS:
        raise OrbitStateError("blocker %d invalid kind: %r" % (index, blocker.get("kind")))
    for key in ("id", "artifact", "message"):
        if not isinstance(blocker.get(key), str) or not blocker.get(key):
            raise OrbitStateError("blocker %d %s must be a non-empty string" % (index, key))
