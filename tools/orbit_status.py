#!/usr/bin/env python3
"""Read-only ORBIT status doctor.

The command prefers `orbit-research/ORBIT_STATE.json` when present. If it is
missing, it infers a conservative status from legacy ORBIT artifacts and emits
the same state shape without writing anything.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from types import SimpleNamespace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

try:
    from check_stop_c_approval import (
        HUMAN_PROCEED,
        HUMAN_VERDICTS,
        evaluate_stop_c_approval,
        parse_final_verdict,
    )
    from orbit_state import (
        DEFAULT_CANONICAL_PACKS,
        ORBIT_STATE_REL_PATH,
        OrbitStateError,
        make_blocker,
        make_state,
        read_state,
    )
    from validate_orbit_pack import validate_selection
except ImportError:  # pragma: no cover - used when imported as tools.orbit_status
    from tools.check_stop_c_approval import (
        HUMAN_PROCEED,
        HUMAN_VERDICTS,
        evaluate_stop_c_approval,
        parse_final_verdict,
    )
    from tools.orbit_state import (
        DEFAULT_CANONICAL_PACKS,
        ORBIT_STATE_REL_PATH,
        OrbitStateError,
        make_blocker,
        make_state,
        read_state,
    )
    from tools.validate_orbit_pack import validate_selection


COMMON_LEGACY_ARTIFACTS = (
    "orbit-research/IDEA_TO_PROPOSAL_STATE.json",
    "orbit-research/DIAGNOSTIC_TO_REVIEW_STATE.json",
    "refine-logs/REFINE_STATE.json",
    "orbit-research/PLAN_CODE_AUDIT.md",
    "orbit-research/DIAGNOSTIC_RUN_AUDIT.md",
    "orbit-research/CLAIM_CONSTRUCTION.md",
    "orbit-research/RED_TEAM_REVIEW.md",
    "orbit-research/HUMAN_DECISION_NOTE.md",
)

INFERENCE_ARTIFACTS = COMMON_LEGACY_ARTIFACTS + (
    "proposal/proposal_pack.json",
    "experiment/experiment_pack.json",
    "experiment/PROBE_REPORT.md",
    "experiment/PROBE_AUDIT.md",
    "experiment/HEADROOM_NOTE.md",
    "claims/claim_ledger.json",
    "claims/CLAIM_LEDGER.md",
    "refine-logs/FINAL_PROPOSAL.md",
    "refine-logs/FINAL_PROPOSAL_SHORT.md",
    "orbit-research/METHOD_SPEC.md",
    "orbit-research/RESULT_INTERPRETATION.md",
    "paper/paper_package.json",
)

PLAN_CODE_GOOD = {"MATCHES_PLAN", "PARTIAL_MISMATCH", "PASS"}
PLAN_CODE_BAD = {
    "CRITICAL_MISMATCH",
    "MISMATCH",
    "FAIL",
    "FAILED",
    "BLOCKED",
    "ERROR",
    "REDESIGN_EXPERIMENT",
    "FIX_BEFORE_GPU",
}
DIAGNOSTIC_GOOD = {"PASS"}
DIAGNOSTIC_BAD = {
    "FAIL",
    "FAILED",
    "BLOCKED",
    "ERROR",
    "REDESIGN_EXPERIMENT",
    "FIX_BEFORE_GPU",
    "CRITICAL_MISMATCH",
}
CLAIM_GOOD = {"SUPPORTED", "PARTIAL", "PARTIALLY_SUPPORTED", "PASS", "READY_FOR_REVIEW"}
CLAIM_BAD = {"UNSUPPORTED", "FAIL", "FAILED", "BLOCKED", "ERROR", "OVERCLAIMED"}
REVIEW_GOOD = {"PASS", "APPROVE", "APPROVED", "READY", "READY_FOR_PAPER", "POSITIVE"}
REVIEW_BAD = {
    "FAIL",
    "FAILED",
    "BLOCKED",
    "ERROR",
    "REQUEST_CHANGES",
    "REJECT",
    "NEGATIVE",
    "REQUIRES_FIXES",
    "REDESIGN_REQUIRED",
    "HUMAN_DECISION_REQUIRED",
}
PROBE_GOOD = {"PASS", "PASSED", "OK", "COMPLETE", "COMPLETED", "READY"}
PROBE_BAD = {"FAIL", "FAILED", "BLOCKED", "ERROR", "CRITICAL_MISMATCH"}
PROBE_RUNNING = {"RUNNING", "IN_PROGRESS", "STARTED"}
VALID_DIAGNOSTIC_KINDS = {
    "implementation_smoke",
    "headroom_probe",
    "local_mechanism_probe",
    "paper_bearing_main",
    "paper_bearing_ablation",
    "scaleup_candidate",
    "unknown",
}
VALID_CLAIM_RELEVANCE = {
    "none",
    "local",
    "paper_scope_affecting",
    "primary_evidence",
    "unknown",
}


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report current ORBIT stop, blockers, and safe next command.")
    parser.add_argument("--repo", default=".", help="Repository root to inspect.")
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--pretty", action="store_true", help="Print short human-readable status.")
    output.add_argument("--json", action="store_true", help="Emit ORBIT_STATE-shaped JSON.")
    return parser.parse_args(argv)


def rel_path(path: Path, repo: Path) -> str:
    return path.relative_to(repo).as_posix()


def existing_artifacts(repo: Path, candidates: Iterable[str]) -> List[str]:
    return sorted(path for path in candidates if (repo / path).exists())


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def parse_json_file(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def normalize_legacy_status(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    lowered = value.strip().lower()
    if lowered in {"in_progress", "running", "active"}:
        return "in_progress"
    if lowered in {"paused", "awaiting_human_continue", "awaiting_user_action", "waiting"}:
        return "paused"
    if lowered in {"blocked", "failed", "error", "aborted"}:
        return "blocked"
    if lowered in {"completed", "complete", "done", "success"}:
        return "completed"
    return None


def legacy_state_hint(path: Path) -> Dict[str, Optional[str]]:
    data = parse_json_file(path)
    if not data:
        return {"status": None, "phase": None, "safe_next_command": None}

    status = None
    for key in ("status", "state", "pipeline_status"):
        status = normalize_legacy_status(data.get(key))
        if status:
            break

    phase = None
    for key in ("current_phase", "phase", "stage", "current_stage"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            phase = value.strip()
            break

    safe_next_command = None
    for key in ("safe_next_command", "next_command", "next_action"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            safe_next_command = value.strip()
            break

    return {"status": status, "phase": phase, "safe_next_command": safe_next_command}


def strip_markdown_markup(text: str) -> str:
    return text.replace("`", "").replace("*", "").strip()


def parse_verdict(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None

    candidates: List[Tuple[int, int, str, str]] = []
    for line_no, line in enumerate(read_text(path).splitlines(), start=1):
        lowered = line.lower()
        if "verdict" not in lowered:
            continue
        if "!=" in line or "not equal" in lowered or "valid verdict" in lowered:
            continue

        clean = strip_markdown_markup(line)
        match = re.search(
            r"\b(?:final\s+)?(?:audit\s+)?verdict\b\s*(?:[:=]|is|->)?\s*([A-Z][A-Z0-9_-]{2,})",
            clean,
            flags=re.IGNORECASE,
        )
        if not match:
            continue
        verdict = match.group(1).upper().replace("-", "_")
        score = 0
        if "final" in lowered:
            score += 3
        if re.search(r"\bverdict\b\s*[:=]", clean, flags=re.IGNORECASE):
            score += 2
        if line.lstrip().startswith(("#", "-")):
            score += 1
        candidates.append((score, -line_no, verdict, clean.strip()))

    if not candidates:
        return None
    candidates.sort(reverse=True)
    score, negative_line_no, verdict, line = candidates[0]
    return {"verdict": verdict, "line": line, "line_number": -negative_line_no}


def verdict_blocker(
    blocker_id: str,
    artifact: str,
    message: str,
    safe_next_command: Optional[str],
) -> Dict[str, Any]:
    return make_blocker(blocker_id, "bad_verdict", artifact, message, safe_next_command)


def missing_blocker(
    blocker_id: str,
    artifact: str,
    message: str,
    safe_next_command: Optional[str],
) -> Dict[str, Any]:
    return make_blocker(blocker_id, "missing_artifact", artifact, message, safe_next_command)


def proposal_path(repo: Path) -> str:
    for candidate in (
        "proposal/proposal_pack.json",
        "refine-logs/FINAL_PROPOSAL.md",
        "refine-logs/FINAL_PROPOSAL_SHORT.md",
    ):
        if (repo / candidate).exists():
            return candidate
    return "proposal/proposal_pack.json"


def experiment_pack_path(repo: Path) -> Path:
    return repo / "experiment/experiment_pack.json"


def quote_skill_arg(value: str) -> str:
    return json.dumps(value)


def formal_diagnostic_input(entry: Mapping[str, Any]) -> Optional[str]:
    for key in ("command", "manifest", "manifest_path", "grid_spec"):
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def formal_diagnostic_entries(pack: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    diagnostics = pack.get("formal_diagnostics")
    if not isinstance(diagnostics, list):
        return []
    entries: List[Mapping[str, Any]] = []
    for item in diagnostics:
        if not isinstance(item, Mapping):
            continue
        if item.get("kind") not in VALID_DIAGNOSTIC_KINDS:
            continue
        if item.get("claim_relevance") not in VALID_CLAIM_RELEVANCE:
            continue
        if formal_diagnostic_input(item):
            entries.append(item)
    return entries


def formal_diagnostic_next_command(pack: Mapping[str, Any]) -> Optional[str]:
    entries = formal_diagnostic_entries(pack)
    if not entries:
        return None
    if len(entries) == 1:
        formal_input = formal_diagnostic_input(entries[0])
        if formal_input:
            return "/diagnostic-to-review %s" % quote_skill_arg(formal_input)
    return '/diagnostic-to-review "experiment/experiment_pack.json"'


def legacy_experiment_plan_path(repo: Path) -> Optional[str]:
    for candidate in (
        "refine-logs/EXPERIMENT_PLAN_EXEC.md",
        "experiment/EXPERIMENT_PLAN_EXEC.md",
        "refine-logs/EXPERIMENT_PLAN.md",
        "experiment/EXPERIMENT_PLAN.md",
    ):
        if (repo / candidate).exists():
            return candidate
    return None


def summarize_probe_status(probes: Any) -> str:
    if not isinstance(probes, list) or not probes:
        return "not_run"
    statuses = {
        str(item.get("status", "")).upper().replace("-", "_")
        for item in probes
        if isinstance(item, dict)
    }
    if not statuses:
        return "recorded"
    if statuses & PROBE_BAD:
        return "blocked"
    if statuses & PROBE_RUNNING:
        return "in_progress"
    if statuses and statuses <= PROBE_GOOD:
        return "complete"
    return "recorded"


def summarize_plan_status(pack: Mapping[str, Any]) -> str:
    if pack.get("status") == "ready":
        return "ready"
    if pack.get("status") == "blocked":
        return "blocked"
    if pack.get("decision_tree") or pack.get("controls") or pack.get("formal_diagnostics"):
        return "draft"
    return "missing"


def summarize_audit_status(repo: Path, pack: Mapping[str, Any]) -> Tuple[str, Optional[str]]:
    audit = pack.get("plan_code_audit")
    if isinstance(audit, dict):
        verdict = audit.get("verdict")
        if isinstance(verdict, str) and verdict.strip():
            normalized = verdict.strip().upper().replace("-", "_")
            if normalized in PLAN_CODE_BAD:
                return "blocked", normalized
            if normalized in PLAN_CODE_GOOD:
                return "pass", normalized
            return "recorded", normalized

    parsed = parse_verdict(repo / "orbit-research/PLAN_CODE_AUDIT.md")
    if parsed:
        verdict = parsed["verdict"]
        if verdict in PLAN_CODE_BAD:
            return "blocked", verdict
        if verdict in PLAN_CODE_GOOD:
            return "pass", verdict
        return "recorded", verdict
    return "missing", None


def summarize_claim_ledger(repo: Path) -> Optional[Dict[str, Any]]:
    ledger = parse_json_file(repo / "claims/claim_ledger.json")
    if not ledger:
        return None

    claim_statuses = []
    for claim in ledger.get("claims", []):
        if isinstance(claim, dict):
            status = claim.get("status")
            if isinstance(status, str) and status.strip():
                claim_statuses.append(status.strip().lower())

    top_status = ledger.get("status")
    if not isinstance(top_status, str) or not top_status.strip():
        top_status = "draft"
    top_status = top_status.strip().lower()

    return {
        "pack_status": top_status,
        "claim_statuses": sorted(set(claim_statuses)),
        "claim_count": len(claim_statuses),
        "has_unsupported": any(status == "unsupported" for status in claim_statuses),
        "has_supported": any(status == "supported" for status in claim_statuses),
    }


def paper_package_validation_errors(repo: Path) -> List[str]:
    args = SimpleNamespace(all=False, pack="paper_package", kind=None, path=None, json=False)
    report = validate_selection(repo, args)
    errors: List[str] = []
    for result in report.get("results", []):
        if result.get("name") != "paper_package":
            continue
        errors.extend(str(error) for error in result.get("errors", []))
    return errors


def package_blockers_from_errors(errors: Iterable[str], safe_next: str) -> List[Mapping[str, Any]]:
    return [
        make_blocker(
            "STOP_D",
            "bad_verdict",
            "paper/paper_package.json",
            error,
            safe_next,
        )
        for error in errors
    ]


def state_from_paper_package(repo: Path, legacy_artifacts: List[str]) -> Optional[Dict[str, Any]]:
    paper_path = repo / "paper/paper_package.json"
    if not paper_path.exists():
        return None

    safe_next = '/submission-package "paper/"'
    package = parse_json_file(paper_path)
    if not isinstance(package, dict):
        return state_with_legacy(
            repo,
            legacy_artifacts,
            "STOP_D",
            "submission-package",
            "paper_package_invalid",
            "blocked",
            "ambiguous_resume",
            [
                make_blocker(
                    "STOP_D",
                    "bad_verdict",
                    "paper/paper_package.json",
                    "paper_package.json is missing, invalid, or not a JSON object",
                    safe_next,
                )
            ],
            safe_next,
        )

    package_status = package.get("status")
    normalized_status = package_status.strip().lower() if isinstance(package_status, str) else None
    errors = paper_package_validation_errors(repo)
    blockers = package_blockers_from_errors(errors, safe_next)

    if normalized_status == "ready" and not errors:
        state = state_with_legacy(
            repo,
            legacy_artifacts,
            "STOP_D",
            "submission-package",
            "paper_package_ready",
            "completed",
            None,
            [],
            None,
        )
        state["stop_d"] = {"paper_package_status": "ready"}
        return state

    if normalized_status == "blocked":
        package_blockers = package.get("blockers")
        if isinstance(package_blockers, list):
            for index, blocker in enumerate(package_blockers):
                if not isinstance(blocker, Mapping):
                    continue
                message = blocker.get("message")
                if not isinstance(message, str) or not message.strip():
                    message = "paper_package blocker %d" % index
                artifact = blocker.get("artifact")
                if not isinstance(artifact, str) or not artifact.strip():
                    artifact = "paper/paper_package.json"
                blockers.append(
                    make_blocker(
                        str(blocker.get("id") or "STOP_D"),
                        str(blocker.get("kind") or "bad_verdict"),
                        artifact,
                        message,
                        safe_next,
                    )
                )
        if not blockers:
            blockers = package_blockers_from_errors(
                ["paper_package status is blocked"],
                safe_next,
            )
        state = state_with_legacy(
            repo,
            legacy_artifacts,
            "STOP_D",
            "submission-package",
            "paper_package_blocked",
            "blocked",
            "gate_failed",
            blockers,
            safe_next,
        )
        state["stop_d"] = {"paper_package_status": "blocked"}
        return state

    if normalized_status in {"draft", "in_progress"}:
        state = state_with_legacy(
            repo,
            legacy_artifacts,
            "STOP_D",
            "submission-package",
            "paper_package_%s" % normalized_status,
            "in_progress" if normalized_status == "in_progress" else "paused",
            "stop_review" if normalized_status == "draft" else None,
            blockers,
            safe_next,
        )
        state["stop_d"] = {"paper_package_status": normalized_status}
        return state

    if errors:
        state = state_with_legacy(
            repo,
            legacy_artifacts,
            "STOP_D",
            "submission-package",
            "paper_package_not_ready",
            "blocked",
            "gate_failed",
            blockers,
            safe_next,
        )
        state["stop_d"] = {"paper_package_status": normalized_status or "missing"}
        return state

    state = state_with_legacy(
        repo,
        legacy_artifacts,
        "STOP_D",
        "submission-package",
        "paper_package_status_ambiguous",
        "blocked",
        "ambiguous_resume",
        [
            make_blocker(
                "STOP_D",
                "bad_verdict",
                "paper/paper_package.json",
                "paper_package.status is missing or not recognized: %r" % package_status,
                safe_next,
            )
        ],
        safe_next,
    )
    state["stop_d"] = {"paper_package_status": normalized_status or "missing"}
    return state


def human_decision_verdict(repo: Path) -> Optional[str]:
    path = repo / "orbit-research/HUMAN_DECISION_NOTE.md"
    if not path.exists():
        return None
    return parse_final_verdict(read_text(path), HUMAN_VERDICTS)


def state_from_claim_ledger(repo: Path, legacy_artifacts: List[str]) -> Optional[Dict[str, Any]]:
    if not (repo / "claims/claim_ledger.json").exists():
        return None

    summary = summarize_claim_ledger(repo) or {}
    approval = evaluate_stop_c_approval(repo, "claims/claim_ledger.json")
    if approval.get("status") == "approved":
        state = state_with_legacy(
            repo,
            legacy_artifacts,
            "STOP_D",
            "paper-from-claims",
            "ready_for_paper_from_claims",
            "paused",
            "stop_review",
            [],
            '/paper-from-claims "claims/claim_ledger.json"',
        )
        state["stop_c"] = {"claim_ledger": summary, "approval": approval}
        return state

    errors = [str(error) for error in approval.get("errors", [])]
    semantic_errors = [str(error) for error in approval.get("claim_ledger_semantic_errors", [])]
    joined_errors = "\n".join(errors)
    red_team_path = approval.get("red_team_review") or "orbit-research/RED_TEAM_REVIEW.md"
    if approval.get("diagnostic_id"):
        red_team_path = (
            "orbit-research/diagnostics/%s/RED_TEAM_REVIEW.md"
            % approval["diagnostic_id"]
        )

    if (
        "claim ledger must be 'ready'" in joined_errors
        or "codex_review" in joined_errors
        or "non-gating" in joined_errors
        or bool(semantic_errors)
    ):
        command = '/result-to-claim "claims/claim_ledger.json"'
        routed_errors = semantic_errors or errors
        state = state_with_legacy(
            repo,
            legacy_artifacts,
            "STOP_C",
            "result-to-claim",
            "claim_ledger_invalid",
            "blocked",
            "gate_failed",
            [
                verdict_blocker(
                    "claim_ledger_invalid",
                    "claims/claim_ledger.json",
                    error,
                    command,
                )
                for error in routed_errors
            ],
            command,
        )
        state["stop_c"] = {"claim_ledger": summary, "approval": approval}
        return state

    if any("missing RED_TEAM_REVIEW.md" in error for error in errors):
        command = '/auto-review-loop "claims/claim_ledger.json" -- difficulty: hard -- orbit-red-team: true'
        state = state_with_legacy(
            repo,
            legacy_artifacts,
            "STOP_C",
            "auto-review-loop",
            "red_team_review_missing",
            "paused",
            "missing_prereq",
            [
                missing_blocker(
                    "G23",
                    red_team_path,
                    "RED_TEAM_REVIEW is missing after claim_ledger",
                    command,
                )
            ],
            command,
        )
        state["stop_c"] = {"claim_ledger": summary, "approval": approval}
        return state

    verdict = human_decision_verdict(repo)
    if verdict and verdict != HUMAN_PROCEED:
        command = "Review STOP_C_REVIEW.md and update orbit-research/HUMAN_DECISION_NOTE.md only if the human decision changes."
        state = state_with_legacy(
            repo,
            legacy_artifacts,
            "STOP_C",
            "diagnostic-to-review",
            "human_decision_%s" % verdict.lower(),
            "blocked",
            "gate_failed",
            [
                verdict_blocker(
                    "G19",
                    "orbit-research/HUMAN_DECISION_NOTE.md",
                    "HUMAN_DECISION_NOTE verdict %s does not permit paper handoff" % verdict,
                    command,
                )
            ],
            command,
        )
        state["stop_c"] = {"claim_ledger": summary, "approval": approval}
        return state

    if any("HUMAN_DECISION_NOTE" in error for error in errors):
        command = "review orbit-research/RED_TEAM_REVIEW.md and write orbit-research/HUMAN_DECISION_NOTE.md ending PROCEED"
        if approval.get("red_team_review"):
            command = "review %s and write orbit-research/HUMAN_DECISION_NOTE.md ending PROCEED" % approval["red_team_review"]
        state = state_with_legacy(
            repo,
            legacy_artifacts,
            "STOP_C",
            "diagnostic-to-review",
            "awaiting_human_decision",
            "paused",
            "stop_review",
            [
                missing_blocker(
                    "G19",
                    "orbit-research/HUMAN_DECISION_NOTE.md",
                    "HUMAN_DECISION_NOTE is missing or does not end PROCEED",
                    command,
                )
            ],
            command,
        )
        state["stop_c"] = {"claim_ledger": summary, "approval": approval}
        return state

    command = '/auto-review-loop "claims/claim_ledger.json" -- difficulty: hard -- orbit-red-team: true'
    state = state_with_legacy(
        repo,
        legacy_artifacts,
        "STOP_C",
        "auto-review-loop",
        "red_team_review",
        "blocked",
        "gate_failed",
        [
            verdict_blocker(
                "G23",
                red_team_path,
                "; ".join(errors) if errors else "STOP C approval is blocked",
                command,
            )
        ],
        command,
    )
    state["stop_c"] = {"claim_ledger": summary, "approval": approval}
    return state


def state_from_experiment_pack(repo: Path, legacy_artifacts: List[str]) -> Optional[Dict[str, Any]]:
    pack = parse_json_file(experiment_pack_path(repo))
    if not pack:
        return None

    plan_status = summarize_plan_status(pack)
    audit_status, audit_verdict = summarize_audit_status(repo, pack)
    probe_status = summarize_probe_status(pack.get("probes"))
    summary = {
        "plan_status": plan_status,
        "audit_status": audit_status if audit_verdict is None else "%s (%s)" % (audit_status, audit_verdict),
        "probe_status": probe_status,
        "formal_diagnostics": len(formal_diagnostic_entries(pack)),
    }

    blockers: List[Mapping[str, Any]] = []
    status = "paused"
    pause_reason: Optional[str] = "stop_review"
    safe_next = formal_diagnostic_next_command(pack)
    current_phase = "plan_code_probe_review"

    if plan_status == "blocked" or audit_status == "blocked" or probe_status == "blocked":
        status = "blocked"
        pause_reason = "gate_failed"
        current_phase = "blocked_stop_b_review"
        safe_next = '/experiment-bridge "experiment/experiment_pack.json" -- mode: audit-only'
        artifact = "experiment/experiment_pack.json"
        message = "experiment_pack reports blocked status"
        if audit_status == "blocked":
            artifact = "orbit-research/PLAN_CODE_AUDIT.md"
            message = "PLAN_CODE_AUDIT verdict %s" % audit_verdict
        elif probe_status == "blocked":
            artifact = "experiment/PROBE_AUDIT.md"
            message = "STOP B probe status blocked"
        blockers.append(verdict_blocker("G11", artifact, message, safe_next))
    elif audit_status == "missing":
        status = "paused"
        pause_reason = "missing_prereq"
        current_phase = "plan_code_audit_missing"
        safe_next = '/experiment-bridge "experiment/experiment_pack.json" -- mode: audit-only'
        blockers.append(
            missing_blocker(
                "G11",
                "orbit-research/PLAN_CODE_AUDIT.md",
                "PLAN_CODE_AUDIT is missing after experiment_pack",
                safe_next,
            )
        )
    elif safe_next is None:
        status = "blocked"
        pause_reason = "missing_prereq"
        current_phase = "formal_diagnostics_missing"
        safe_next = '/experiment-bridge "experiment/experiment_pack.json" — mode: plan-only'
        blockers.append(
            missing_blocker(
                "G12",
                "experiment_pack.formal_diagnostics",
                "experiment_pack.formal_diagnostics is missing or lacks a command/manifest",
                safe_next,
            )
        )

    state = state_with_legacy(
        repo,
        legacy_artifacts,
        "STOP_B",
        "experiment-bridge",
        current_phase,
        status,
        pause_reason,
        blockers,
        safe_next,
    )
    state["stop_b"] = summary
    return state


def state_with_legacy(
    repo: Path,
    legacy_artifacts: List[str],
    current_stop: str,
    current_skill: Optional[str],
    current_phase: str,
    status: str,
    pause_reason: Optional[str],
    blockers: Optional[List[Mapping[str, Any]]],
    safe_next_command: Optional[str],
) -> Dict[str, Any]:
    return make_state(
        current_stop=current_stop,
        current_skill=current_skill,
        current_phase=current_phase,
        status=status,
        pause_reason=pause_reason,
        blockers=blockers or [],
        canonical_packs=DEFAULT_CANONICAL_PACKS,
        legacy_artifacts_detected=legacy_artifacts,
        safe_next_command=safe_next_command,
    )


def infer_from_legacy(repo: Path) -> Dict[str, Any]:
    legacy_artifacts = existing_artifacts(
        repo,
        (
            path
            for path in INFERENCE_ARTIFACTS
            if path
            not in {
                "proposal/proposal_pack.json",
                "experiment/experiment_pack.json",
                "experiment/PROBE_REPORT.md",
                "experiment/PROBE_AUDIT.md",
                "experiment/HEADROOM_NOTE.md",
                "claims/claim_ledger.json",
                "claims/CLAIM_LEDGER.md",
            }
        ),
    )
    plan_audit = repo / "orbit-research/PLAN_CODE_AUDIT.md"
    diagnostic_audit = repo / "orbit-research/DIAGNOSTIC_RUN_AUDIT.md"
    claim_ledger = repo / "claims/claim_ledger.json"
    claim_construction = repo / "orbit-research/CLAIM_CONSTRUCTION.md"
    red_team_review = repo / "orbit-research/RED_TEAM_REVIEW.md"
    human_note = repo / "orbit-research/HUMAN_DECISION_NOTE.md"

    paper_state = state_from_paper_package(repo, legacy_artifacts)
    if paper_state is not None:
        return paper_state

    claim_state = state_from_claim_ledger(repo, legacy_artifacts)
    if claim_state is not None:
        return claim_state

    if red_team_review.exists():
        claim_source = "claims/claim_ledger.json" if claim_ledger.exists() else "orbit-research/CLAIM_CONSTRUCTION.md"
        parsed = parse_verdict(red_team_review)
        if parsed and parsed["verdict"] in REVIEW_BAD:
            command = '/auto-review-loop "%s" -- difficulty: hard -- orbit-red-team: true' % claim_source
            return state_with_legacy(
                repo,
                legacy_artifacts,
                "STOP_C",
                "auto-review-loop",
                "red_team_review",
                "blocked",
                "gate_failed",
                [
                    verdict_blocker(
                        "G14",
                        "orbit-research/RED_TEAM_REVIEW.md",
                        "RED_TEAM_REVIEW verdict %s" % parsed["verdict"],
                        command,
                    )
                ],
                command,
            )
        if parsed and parsed["verdict"] in REVIEW_GOOD:
            if human_note.exists():
                command = (
                    '/paper-from-claims "claims/claim_ledger.json"'
                    if claim_ledger.exists()
                    else '/paper-writing "orbit-research/CLAIM_CONSTRUCTION.md"'
                )
                return state_with_legacy(
                    repo,
                    legacy_artifacts,
                    "STOP_D",
                    "paper-writing" if not claim_ledger.exists() else "paper-from-claims",
                    "ready_for_paper_writing",
                    "paused",
                    "stop_review",
                    [],
                    command,
                )
            command = "review orbit-research/RED_TEAM_REVIEW.md and write orbit-research/HUMAN_DECISION_NOTE.md"
            return state_with_legacy(
                repo,
                legacy_artifacts,
                "STOP_C",
                "diagnostic-to-review",
                "awaiting_human_decision",
                "paused",
                "stop_review",
                [],
                command,
            )

        command = '/auto-review-loop "%s" -- difficulty: hard -- orbit-red-team: true' % claim_source
        return state_with_legacy(
            repo,
            legacy_artifacts,
            "STOP_C",
            "auto-review-loop",
            "red_team_review",
            "blocked",
            "ambiguous_resume",
            [
                verdict_blocker(
                    "G14",
                    "orbit-research/RED_TEAM_REVIEW.md",
                    "RED_TEAM_REVIEW verdict could not be parsed",
                    command,
                )
            ],
            command,
        )

    if claim_construction.exists():
        parsed = parse_verdict(claim_construction)
        if parsed and parsed["verdict"] in CLAIM_BAD:
            command = '/result-to-claim "orbit-research/RESULT_INTERPRETATION.md"'
            return state_with_legacy(
                repo,
                legacy_artifacts,
                "STOP_C",
                "result-to-claim",
                "claim_construction",
                "blocked",
                "gate_failed",
                [
                    verdict_blocker(
                        "G21",
                        "orbit-research/CLAIM_CONSTRUCTION.md",
                        "CLAIM_CONSTRUCTION verdict %s" % parsed["verdict"],
                        command,
                    )
                ],
                command,
            )
        if parsed and parsed["verdict"] in CLAIM_GOOD:
            command = '/auto-review-loop "orbit-research/CLAIM_CONSTRUCTION.md"'
            return state_with_legacy(
                repo,
                legacy_artifacts,
                "STOP_C",
                "auto-review-loop",
                "red_team_review_missing",
                "paused",
                "missing_prereq",
                [
                    missing_blocker(
                        "G23",
                        "orbit-research/RED_TEAM_REVIEW.md",
                        "RED_TEAM_REVIEW is missing after claim construction",
                        command,
                    )
                ],
                command,
            )

        command = '/result-to-claim "orbit-research/RESULT_INTERPRETATION.md"'
        return state_with_legacy(
            repo,
            legacy_artifacts,
            "STOP_C",
            "result-to-claim",
            "claim_construction",
            "blocked",
            "ambiguous_resume",
            [
                verdict_blocker(
                    "G21",
                    "orbit-research/CLAIM_CONSTRUCTION.md",
                    "CLAIM_CONSTRUCTION verdict could not be parsed",
                    command,
                )
            ],
            command,
        )

    if diagnostic_audit.exists():
        parsed = parse_verdict(diagnostic_audit)
        if parsed and parsed["verdict"] in DIAGNOSTIC_BAD:
            command = '/diagnostic-to-review "orbit-research/DIAGNOSTIC_RUN_AUDIT.md"'
            return state_with_legacy(
                repo,
                legacy_artifacts,
                "STOP_C",
                "diagnostic-to-review",
                "diagnostic_run_audit",
                "blocked",
                "gate_failed",
                [
                    verdict_blocker(
                        "G12",
                        "orbit-research/DIAGNOSTIC_RUN_AUDIT.md",
                        "DIAGNOSTIC_RUN_AUDIT verdict %s" % parsed["verdict"],
                        command,
                    )
                ],
                command,
            )
        if parsed and parsed["verdict"] in DIAGNOSTIC_GOOD:
            command = '/diagnostic-to-review "orbit-research/DIAGNOSTIC_RUN_AUDIT.md"'
            return state_with_legacy(
                repo,
                legacy_artifacts,
                "STOP_C",
                "result-to-claim",
                "claim_ledger_missing",
                "paused",
                "missing_prereq",
                [
                    missing_blocker(
                        "G21",
                        "claims/claim_ledger.json",
                        "claim_ledger is missing after diagnostic PASS",
                        command,
                    )
                ],
                command,
            )

        command = '/diagnostic-to-review "orbit-research/DIAGNOSTIC_RUN_AUDIT.md"'
        return state_with_legacy(
            repo,
            legacy_artifacts,
            "STOP_C",
            "diagnostic-to-review",
            "diagnostic_run_audit",
            "blocked",
            "ambiguous_resume",
            [
                verdict_blocker(
                    "G12",
                    "orbit-research/DIAGNOSTIC_RUN_AUDIT.md",
                    "DIAGNOSTIC_RUN_AUDIT verdict could not be parsed",
                    command,
                )
            ],
            command,
        )

    experiment_state = state_from_experiment_pack(repo, legacy_artifacts)
    if experiment_state is not None:
        return experiment_state

    if plan_audit.exists():
        parsed = parse_verdict(plan_audit)
        if parsed and parsed["verdict"] in PLAN_CODE_BAD:
            command = '/experiment-bridge "%s" -- mode: audit-only' % proposal_path(repo)
            return state_with_legacy(
                repo,
                legacy_artifacts,
                "STOP_B",
                "experiment-bridge",
                "plan_code_audit",
                "blocked",
                "gate_failed",
                [
                    verdict_blocker(
                        "G11",
                        "orbit-research/PLAN_CODE_AUDIT.md",
                        "PLAN_CODE_AUDIT verdict %s" % parsed["verdict"],
                        command,
                    )
                ],
                command,
            )
        if parsed and parsed["verdict"] in PLAN_CODE_GOOD:
            legacy_plan = legacy_experiment_plan_path(repo)
            command = (
                '/experiment-bridge "%s" — mode: plan-only' % legacy_plan
                if legacy_plan
                else '/experiment-bridge "%s" — mode: plan-only' % proposal_path(repo)
            )
            return state_with_legacy(
                repo,
                legacy_artifacts,
                "STOP_B",
                "experiment-bridge",
                "formal_diagnostics_missing",
                "blocked",
                "missing_prereq",
                [
                    missing_blocker(
                        "G12",
                        "experiment/experiment_pack.json",
                        (
                            "experiment_pack.json with formal_diagnostics is missing after "
                            "PLAN_CODE_AUDIT %s; PLAN_CODE_AUDIT.md is not a diagnostic input"
                        )
                        % parsed["verdict"],
                        command,
                    )
                ],
                command,
            )

        command = '/experiment-bridge "%s" -- mode: audit-only' % proposal_path(repo)
        return state_with_legacy(
            repo,
            legacy_artifacts,
            "STOP_B",
            "experiment-bridge",
            "plan_code_audit",
            "blocked",
            "ambiguous_resume",
            [
                verdict_blocker(
                    "G11",
                    "orbit-research/PLAN_CODE_AUDIT.md",
                    "PLAN_CODE_AUDIT verdict could not be parsed",
                    command,
                )
            ],
            command,
        )

    final_proposal_detected = any(
        (repo / path).exists()
        for path in (
            "proposal/proposal_pack.json",
            "refine-logs/FINAL_PROPOSAL.md",
            "refine-logs/FINAL_PROPOSAL_SHORT.md",
            "orbit-research/METHOD_SPEC.md",
        )
    )
    if (repo / "proposal/proposal_pack.json").exists():
        command = '/experiment-bridge "proposal/proposal_pack.json"'
        return state_with_legacy(
            repo,
            legacy_artifacts,
            "STOP_A",
            "idea-to-proposal",
            "proposal_review",
            "paused",
            "stop_review",
            [],
            command,
        )

    if final_proposal_detected:
        command = '/experiment-bridge "%s"' % proposal_path(repo)
        return state_with_legacy(
            repo,
            legacy_artifacts,
            "STOP_B",
            "experiment-bridge",
            "plan_code_audit_missing",
            "paused",
            "missing_prereq",
            [
                missing_blocker(
                    "G11",
                    "orbit-research/PLAN_CODE_AUDIT.md",
                    "PLAN_CODE_AUDIT is missing after proposal artifacts",
                    command,
                )
            ],
            command,
        )

    for rel_state, stop, skill, default_command in (
        (
            "orbit-research/DIAGNOSTIC_TO_REVIEW_STATE.json",
            "STOP_C",
            "diagnostic-to-review",
            '/diagnostic-to-review "<explicit diagnostic command>"',
        ),
        (
            "orbit-research/IDEA_TO_PROPOSAL_STATE.json",
            "STOP_A",
            "idea-to-proposal",
            '/idea-to-proposal "<research direction>"',
        ),
        (
            "refine-logs/REFINE_STATE.json",
            "STOP_A",
            "research-refine",
            '/research-refine "<research direction>"',
        ),
    ):
        path = repo / rel_state
        if not path.exists():
            continue
        hint = legacy_state_hint(path)
        status = hint["status"] or "paused"
        command = hint["safe_next_command"] or default_command
        pause_reason = "ambiguous_resume" if status != "completed" else None
        return state_with_legacy(
            repo,
            legacy_artifacts,
            stop,
            skill,
            hint["phase"] or "legacy_state_detected",
            status,
            pause_reason,
            [],
            command,
        )

    return state_with_legacy(
        repo,
        legacy_artifacts,
        "NONE",
        None,
        "not_started",
        "paused",
        "ambiguous_resume",
        [],
        '/idea-to-proposal "<research direction>"',
    )


def stale_state(repo: Path, error: Exception) -> Dict[str, Any]:
    command = "/orbit-status"
    return make_state(
        current_stop="NONE",
        current_skill="orbit-status",
        current_phase="read_orbit_state",
        status="blocked",
        pause_reason="ambiguous_resume",
        blockers=[
            make_blocker(
                "ORBIT_STATE",
                "stale_state",
                ORBIT_STATE_REL_PATH,
                "ORBIT_STATE.json could not be read: %s" % error,
                command,
            )
        ],
        canonical_packs=DEFAULT_CANONICAL_PACKS,
        legacy_artifacts_detected=existing_artifacts(repo, INFERENCE_ARTIFACTS),
        safe_next_command=command,
    )


def normalize_existing_state(repo: Path, state: Dict[str, Any]) -> Dict[str, Any]:
    """Keep explicit ORBIT_STATE, but correct unsafe v1/v2 paper inferences."""
    normalized = dict(state)
    legacy_artifacts = existing_artifacts(repo, INFERENCE_ARTIFACTS)
    safe_next = normalized.get("safe_next_command")
    current_skill = normalized.get("current_skill")
    current_stop = normalized.get("current_stop")

    paper_command = is_paper_handoff_command(safe_next)
    paper_context = (
        paper_command
        or current_stop in {"STOP_D", "COMPLETED"}
        or current_skill in {"paper-writing", "paper-from-claims", "submission-package", "paper-draft"}
    )

    if current_skill == "paper-writing":
        if isinstance(safe_next, str) and "/paper-draft" in safe_next:
            normalized["current_skill"] = "paper-draft"
        elif (repo / "paper/paper_package.json").exists() or (
            isinstance(safe_next, str) and "/submission-package" in safe_next
        ):
            normalized["current_skill"] = "submission-package"
        elif (repo / "claims/claim_ledger.json").exists() or (
            isinstance(safe_next, str) and "/paper-from-claims" in safe_next
        ):
            normalized["current_skill"] = "paper-from-claims"

    paper_state = state_from_paper_package(repo, legacy_artifacts)
    if paper_state is not None:
        if paper_state.get("status") == "blocked" or paper_context:
            return paper_state

    if paper_command:
        claim_state = state_from_claim_ledger(repo, legacy_artifacts)
        if claim_state is not None:
            return claim_state

    return normalized


def is_paper_handoff_command(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    return any(
        command in value
        for command in (
            "/paper-from-claims",
            "/submission-package",
            "/paper-writing",
        )
    )


def get_status(repo: Path) -> Dict[str, Any]:
    repo = repo.resolve()
    if (repo / ORBIT_STATE_REL_PATH).exists():
        try:
            return normalize_existing_state(repo, read_state(repo))  # type: ignore[arg-type]
        except (OSError, json.JSONDecodeError, OrbitStateError) as exc:
            return stale_state(repo, exc)
    return infer_from_legacy(repo)


def format_pretty(state: Mapping[str, Any]) -> str:
    lines = [
        "Current stop: %s" % state.get("current_stop"),
        "Status: %s" % state.get("status"),
        "Pause reason: %s" % (state.get("pause_reason") if state.get("pause_reason") is not None else "none"),
    ]
    if state.get("current_skill"):
        lines.append("Current skill: %s" % state.get("current_skill"))
    if state.get("current_phase"):
        lines.append("Current phase: %s" % state.get("current_phase"))

    legacy = state.get("legacy_artifacts_detected") or []
    if legacy:
        lines.append("Legacy artifacts detected: %d" % len(legacy))

    stop_b = state.get("stop_b")
    if isinstance(stop_b, dict):
        lines.append("STOP B:")
        lines.append("  Plan: %s" % stop_b.get("plan_status", "unknown"))
        lines.append("  Audit: %s" % stop_b.get("audit_status", "unknown"))
        lines.append("  Probe: %s" % stop_b.get("probe_status", "unknown"))
        if "formal_diagnostics" in stop_b:
            lines.append("  Formal diagnostics: %s" % stop_b.get("formal_diagnostics", 0))

    stop_c = state.get("stop_c")
    if isinstance(stop_c, dict):
        ledger = stop_c.get("claim_ledger")
        if isinstance(ledger, dict):
            lines.append("STOP C:")
            lines.append("  Claim ledger: %s" % ledger.get("pack_status", "unknown"))
            lines.append("  Claims: %s" % ledger.get("claim_count", 0))
            statuses = ledger.get("claim_statuses") or []
            if statuses:
                lines.append("  Claim statuses: %s" % ", ".join(str(value) for value in statuses))

    lines.append("Blocked by:")
    blockers = state.get("blockers") or []
    if blockers:
        for blocker in blockers:
            lines.append("  - %s: %s" % (blocker.get("id"), blocker.get("message")))
    else:
        lines.append("  none")

    lines.append("Safe next command:")
    safe_next = state.get("safe_next_command")
    lines.append("  %s" % (safe_next if safe_next else "none"))
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    repo = Path(args.repo)
    state = get_status(repo)
    if args.json:
        print(json.dumps(state, indent=2, sort_keys=True))
    else:
        print(format_pretty(state))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
