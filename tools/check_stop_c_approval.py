#!/usr/bin/env python3
"""Check STOP C approval before evidence-bound paper or submission work."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

try:
    from orbit_verdicts import parse_final_token
except ImportError:  # pragma: no cover - used when imported as tools.check_stop_c_approval
    from tools.orbit_verdicts import parse_final_token


RED_TEAM_READY = "READY_FOR_PAPER"
HUMAN_PROCEED = "PROCEED"

RED_TEAM_VERDICTS = {
    "READY_FOR_PAPER",
    "REQUIRES_FIXES",
    "REDESIGN_REQUIRED",
    "HUMAN_DECISION_REQUIRED",
}

HUMAN_VERDICTS = {
    "PROCEED",
    "NARROW",
    "REDESIGN",
    "REVISE",
    "STOP",
    "HOLD",
    "BLOCKED",
}

READY_CODEX_REVIEWS = {"passed", "imported"}


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".", help="Project or repository root.")
    parser.add_argument(
        "--claim-ledger",
        default="claims/claim_ledger.json",
        help="Claim ledger path relative to --repo.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON report.")
    parser.add_argument(
        "--allow-legacy-missing-codex-review",
        action="store_true",
        help="Compatibility mode: allow a ready ledger with no codex_review field.",
    )
    parser.add_argument(
        "--allow-unmatched-legacy-approval",
        action="store_true",
        help="Compatibility mode: warn instead of blocking when legacy approval notes omit ledger identity.",
    )
    return parser.parse_args(argv)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def relpath(path: Path, repo: Path) -> str:
    try:
        return path.relative_to(repo).as_posix()
    except ValueError:
        return path.as_posix()


def extract_metadata(ledger: Mapping[str, Any]) -> Dict[str, Optional[str]]:
    diagnostic_id = ledger.get("diagnostic_id")
    if not isinstance(diagnostic_id, str):
        diagnostic = ledger.get("diagnostic")
        if isinstance(diagnostic, Mapping):
            value = diagnostic.get("id")
            diagnostic_id = value if isinstance(value, str) else None

    ledger_hash = ledger.get("ledger_hash")
    if not isinstance(ledger_hash, str):
        value = ledger.get("hash")
        ledger_hash = value if isinstance(value, str) else None

    return {
        "diagnostic_id": diagnostic_id,
        "ledger_hash": ledger_hash,
    }


def parse_final_verdict(text: str, allowed: List[str] | set[str] | Tuple[str, ...]) -> Optional[str]:
    return parse_final_token(text, allowed)


def claim_ledger_readiness_errors(
    ledger: Mapping[str, Any],
    location: str = "$",
    allow_legacy_missing_codex_review: bool = False,
) -> List[str]:
    errors: List[str] = []

    status = ledger.get("status")
    if status != "ready":
        errors.append("%s.status: claim ledger must be 'ready', got %r" % (location, status))

    if ledger.get("gating") is False:
        errors.append("%s.gating: non-gating claim ledger cannot satisfy STOP C approval" % location)

    metadata = extract_metadata(ledger)
    if not metadata["diagnostic_id"] and not metadata["ledger_hash"]:
        errors.append(
            "%s: ready claim ledger must include diagnostic_id or ledger_hash for STOP C approval identity checks"
            % location
        )

    codex_review = ledger.get("codex_review")
    if codex_review in READY_CODEX_REVIEWS:
        return errors
    if codex_review is None and allow_legacy_missing_codex_review:
        return errors

    if codex_review is None:
        errors.append(
            "%s.codex_review: ready claim ledger must record Codex review as 'passed' or 'imported'"
            % location
        )
    else:
        errors.append(
            "%s.codex_review: %r cannot satisfy STOP C approval; expected 'passed' or 'imported'"
            % (location, codex_review)
        )
    return errors


def non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def non_empty_string_list(value: Any) -> bool:
    return isinstance(value, list) and any(non_empty_string(item) for item in value)


def has_clear_limitations(claim: Mapping[str, Any]) -> bool:
    limitations = claim.get("limitations")
    if non_empty_string(limitations):
        return True
    return non_empty_string_list(limitations)


def duplicate_claim_id_errors(claims: Any, location: str) -> List[str]:
    errors: List[str] = []
    if not isinstance(claims, list):
        return errors

    seen: Dict[str, int] = {}
    for index, claim in enumerate(claims):
        if not isinstance(claim, Mapping):
            continue
        claim_id = claim.get("id")
        if not non_empty_string(claim_id):
            continue
        if str(claim_id) in seen:
            errors.append(
                "%s[%d].id: duplicate claim id %r also used at %s[%d]"
                % (location, index, claim_id, location, seen[str(claim_id)])
            )
        else:
            seen[str(claim_id)] = index
    return errors


def errors_for_claim(errors: List[str], claim_location: str) -> bool:
    return any(error.startswith(claim_location + ".") for error in errors)


def claim_ledger_semantic_errors(
    ledger: Mapping[str, Any],
    location: str = "$",
    allow_legacy_missing_codex_review: bool = False,
    require_ready: bool = False,
) -> List[str]:
    """Return approval-relevant claim ledger semantic errors."""
    errors: List[str] = []
    claims = ledger.get("claims")
    if not isinstance(claims, list):
        return errors

    if require_ready or ledger.get("status") == "ready":
        errors.extend(
            claim_ledger_readiness_errors(
                ledger,
                location,
                allow_legacy_missing_codex_review=allow_legacy_missing_codex_review,
            )
        )

    errors.extend(duplicate_claim_id_errors(claims, "%s.claims" % location))

    for index, claim in enumerate(claims):
        if not isinstance(claim, Mapping):
            continue

        claim_id = claim.get("id") or index
        status = claim.get("status")
        role = claim.get("claim_role")
        paper_use = claim.get("paper_use")
        claim_loc = "%s.claims[%d]" % (location, index)

        for key in ("id", "statement", "status"):
            if not non_empty_string(claim.get(key)):
                errors.append("%s.%s: claim must have a non-empty %s" % (claim_loc, key, key))

        if paper_use == "allowed" and status not in {"supported", "partial"}:
            errors.append(
                "%s.paper_use: claim %r with paper_use='allowed' must be supported or partial, got %r"
                % (claim_loc, claim_id, status)
            )
        if paper_use == "allowed" and status == "partial" and not has_clear_limitations(claim):
            errors.append(
                "%s.limitations: partial allowed claim %r must include clear limitations"
                % (claim_loc, claim_id)
            )

        if status != "unsupported":
            continue

        if paper_use == "allowed":
            errors.append(
                "%s.paper_use: unsupported claim %r cannot have paper_use='allowed'"
                % (claim_loc, claim_id)
            )
        if role == "main_claim":
            errors.append(
                "%s.claim_role: unsupported claim %r cannot be a main_claim"
                % (claim_loc, claim_id)
            )
        if role == "negative_result_claim":
            errors.append(
                "%s.claim_role: negative_result_claim %r must be supported or partial, not unsupported"
                % (claim_loc, claim_id)
            )

        allowed_unsupported = (
            role == "original_hypothesis"
            and paper_use in {"do_not_claim", "limitations_only"}
        )
        if not allowed_unsupported and not errors_for_claim(errors, claim_loc):
            errors.append(
                "%s.status: unsupported claim %r is allowed in a ready ledger only as "
                "claim_role='original_hypothesis' with paper_use='do_not_claim' or 'limitations_only'"
                % (claim_loc, claim_id)
            )

    return errors


def candidate_red_team_paths(repo: Path, diagnostic_id: Optional[str]) -> List[Path]:
    paths: List[Path] = []
    if diagnostic_id:
        paths.append(
            repo
            / "orbit-research"
            / "diagnostics"
            / diagnostic_id
            / "RED_TEAM_REVIEW.md"
        )
    else:
        diagnostics_root = repo / "orbit-research" / "diagnostics"
        if diagnostics_root.is_dir():
            paths.extend(sorted(diagnostics_root.glob("*/RED_TEAM_REVIEW.md")))
    paths.append(repo / "orbit-research" / "RED_TEAM_REVIEW.md")

    unique: List[Path] = []
    seen = set()
    for path in paths:
        key = path.as_posix()
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def file_metadata_reference_messages(
    path: Path,
    repo: Path,
    diagnostic_id: Optional[str],
    ledger_hash: Optional[str],
) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []
    text = path.read_text(encoding="utf-8", errors="replace")
    display_path = relpath(path, repo)
    if diagnostic_id and diagnostic_id not in text and diagnostic_id not in display_path:
        errors.append(
            "%s does not reference diagnostic_id %s" % (display_path, diagnostic_id)
        )
    if ledger_hash and ledger_hash not in text:
        errors.append("%s does not reference ledger_hash %s" % (display_path, ledger_hash))
    return errors, warnings


def evaluate_stop_c_approval(
    repo: Path,
    claim_ledger: str = "claims/claim_ledger.json",
    allow_legacy_missing_codex_review: bool = False,
    allow_unmatched_legacy_approval: bool = False,
) -> Dict[str, Any]:
    repo = repo.resolve()
    requested_ledger_path = Path(claim_ledger)
    if requested_ledger_path.is_absolute():
        ledger_path = requested_ledger_path
    else:
        ledger_path = repo / requested_ledger_path
        if not ledger_path.exists() and requested_ledger_path.exists():
            ledger_path = requested_ledger_path.resolve()
    claim_ledger_display = relpath(ledger_path, repo)
    report: Dict[str, Any] = {
        "status": "approved",
        "claim_ledger": claim_ledger_display,
        "diagnostic_id": None,
        "ledger_hash": None,
        "red_team_review": None,
        "red_team_verdict": None,
        "human_decision_note": "orbit-research/HUMAN_DECISION_NOTE.md",
        "human_decision_verdict": None,
        "claim_ledger_semantic_errors": [],
        "errors": [],
        "warnings": [],
    }

    if not ledger_path.exists():
        report["errors"].append("missing %s" % claim_ledger_display)
        report["status"] = "blocked"
        return report

    try:
        ledger = load_json(ledger_path)
    except (OSError, json.JSONDecodeError) as exc:
        report["errors"].append("could not parse %s: %s" % (claim_ledger_display, exc))
        report["status"] = "blocked"
        return report

    if not isinstance(ledger, Mapping):
        report["errors"].append("%s must be a JSON object" % claim_ledger_display)
        report["status"] = "blocked"
        return report

    metadata = extract_metadata(ledger)
    diagnostic_id = metadata["diagnostic_id"]
    ledger_hash = metadata["ledger_hash"]
    report["diagnostic_id"] = diagnostic_id
    report["ledger_hash"] = ledger_hash
    report["claim_ledger_status"] = ledger.get("status")
    report["claim_ledger_gating"] = ledger.get("gating")
    report["claim_ledger_codex_review"] = ledger.get("codex_review")

    semantic_errors = claim_ledger_semantic_errors(
        ledger,
        "$",
        allow_legacy_missing_codex_review=allow_legacy_missing_codex_review,
        require_ready=True,
    )
    report["claim_ledger_semantic_errors"] = semantic_errors
    report["errors"].extend(semantic_errors)
    if ledger.get("codex_review") is None and allow_legacy_missing_codex_review:
        report["warnings"].append(
            "claim ledger has no codex_review field; accepted only because "
            "--allow-legacy-missing-codex-review was set"
        )

    per_diagnostic_red_team = None
    if diagnostic_id:
        per_diagnostic_red_team = (
            repo
            / "orbit-research"
            / "diagnostics"
            / diagnostic_id
            / "RED_TEAM_REVIEW.md"
        )
    red_team_paths = candidate_red_team_paths(repo, diagnostic_id)
    if per_diagnostic_red_team is not None and per_diagnostic_red_team.exists():
        red_team_paths = [per_diagnostic_red_team]
    existing_red_team_paths = [path for path in red_team_paths if path.exists()]
    if not existing_red_team_paths:
        report["errors"].append(
            "missing RED_TEAM_REVIEW.md with final verdict %s" % RED_TEAM_READY
        )
    else:
        found_verdicts = []
        ready_identity_errors: List[str] = []
        for path in existing_red_team_paths:
            text = path.read_text(encoding="utf-8", errors="replace")
            verdict = parse_final_verdict(text, RED_TEAM_VERDICTS)
            if report["red_team_review"] is None:
                report["red_team_review"] = relpath(path, repo)
                report["red_team_verdict"] = verdict
            found_verdicts.append("%s=%s" % (relpath(path, repo), verdict or "UNKNOWN"))
            if verdict == RED_TEAM_READY:
                identity_errors, identity_warnings = file_metadata_reference_messages(
                    path,
                    repo,
                    diagnostic_id,
                    ledger_hash,
                )
                if identity_errors and not allow_unmatched_legacy_approval:
                    ready_identity_errors.extend(identity_errors)
                    continue
                report["red_team_review"] = relpath(path, repo)
                report["red_team_verdict"] = verdict
                if identity_errors and allow_unmatched_legacy_approval:
                    report["warnings"].extend(
                        "legacy unmatched approval accepted: %s" % error
                        for error in identity_errors
                    )
                report["warnings"].extend(identity_warnings)
                break
        if report["red_team_verdict"] != RED_TEAM_READY:
            if ready_identity_errors:
                report["errors"].extend(ready_identity_errors)
            else:
                report["errors"].append(
                    "RED_TEAM_REVIEW final verdict must be %s; found %s"
                    % (RED_TEAM_READY, ", ".join(found_verdicts))
                )

    human_path = repo / "orbit-research" / "HUMAN_DECISION_NOTE.md"
    if not human_path.exists():
        report["errors"].append(
            "missing orbit-research/HUMAN_DECISION_NOTE.md ending %s" % HUMAN_PROCEED
        )
    else:
        text = human_path.read_text(encoding="utf-8", errors="replace")
        verdict = parse_final_verdict(text, HUMAN_VERDICTS)
        report["human_decision_verdict"] = verdict
        if verdict != HUMAN_PROCEED:
            report["errors"].append(
                "HUMAN_DECISION_NOTE final verdict must be %s; found %s"
                % (HUMAN_PROCEED, verdict or "UNKNOWN")
            )
        else:
            identity_errors, identity_warnings = file_metadata_reference_messages(
                human_path,
                repo,
                diagnostic_id,
                ledger_hash,
            )
            if identity_errors and allow_unmatched_legacy_approval:
                report["warnings"].extend(
                    "legacy unmatched approval accepted: %s" % error
                    for error in identity_errors
                )
            else:
                report["errors"].extend(identity_errors)
            report["warnings"].extend(identity_warnings)

    if report["errors"]:
        report["status"] = "blocked"
    return report


def print_pretty(report: Mapping[str, Any]) -> None:
    print("STOP C approval: %s" % report["status"])
    print("Claim ledger: %s" % report["claim_ledger"])
    print(
        "Red-team review: %s (%s)"
        % (report.get("red_team_review") or "missing", report.get("red_team_verdict") or "UNKNOWN")
    )
    print(
        "Human decision: %s (%s)"
        % (report.get("human_decision_note") or "missing", report.get("human_decision_verdict") or "UNKNOWN")
    )
    if report.get("errors"):
        print("Blocked by:")
        for error in report["errors"]:
            print("  - %s" % error)
    if report.get("warnings"):
        print("Warnings:")
        for warning in report["warnings"]:
            print("  - %s" % warning)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    repo = Path(args.repo)
    report = evaluate_stop_c_approval(
        repo,
        args.claim_ledger,
        allow_legacy_missing_codex_review=args.allow_legacy_missing_codex_review,
        allow_unmatched_legacy_approval=args.allow_unmatched_legacy_approval,
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_pretty(report)
    return 0 if report["status"] == "approved" else 1


if __name__ == "__main__":
    raise SystemExit(main())
