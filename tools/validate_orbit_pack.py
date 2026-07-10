#!/usr/bin/env python3
"""Validate ORBIT machine-readable packs with a small stdlib schema checker."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

try:
    from orbit_pack import PACK_SPECS, get_pack_spec, pack_names, pack_path, schema_path
    from check_stop_c_approval import claim_ledger_semantic_errors, evaluate_stop_c_approval
except ImportError:  # pragma: no cover - used when imported as tools.validate_orbit_pack
    from tools.orbit_pack import PACK_SPECS, get_pack_spec, pack_names, pack_path, schema_path
    from tools.check_stop_c_approval import claim_ledger_semantic_errors, evaluate_stop_c_approval


TOOL_REPO_ROOT = Path(__file__).resolve().parents[1]

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

JSON_TYPE_NAMES = {
    "object": dict,
    "array": list,
    "string": str,
    "boolean": bool,
}


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate ORBIT pack JSON files.")
    parser.add_argument("--repo", default=".", help="Repository root.")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--all", action="store_true", help="Validate every canonical pack path.")
    selection.add_argument("--pack", choices=pack_names(), help="Validate one named pack.")
    selection.add_argument(
        "--kind",
        help="Compatibility alias for --pack; accepts names like experiment or experiment_pack.",
    )
    selection.add_argument("--path", help="Validate a specific pack file path.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable validation report.")
    return parser.parse_args(argv)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def type_matches(expected: str, instance: Any) -> bool:
    if expected == "null":
        return instance is None
    if expected == "integer":
        return isinstance(instance, int) and not isinstance(instance, bool)
    if expected == "number":
        return (isinstance(instance, int) or isinstance(instance, float)) and not isinstance(instance, bool)
    py_type = JSON_TYPE_NAMES.get(expected)
    if py_type is None:
        return True
    return isinstance(instance, py_type)


def validate_schema_subset(schema: Mapping[str, Any], instance: Any, location: str = "$") -> List[str]:
    errors: List[str] = []

    if "const" in schema and instance != schema["const"]:
        errors.append("%s: expected constant %r, got %r" % (location, schema["const"], instance))

    if "enum" in schema and instance not in schema["enum"]:
        errors.append("%s: expected one of %r, got %r" % (location, schema["enum"], instance))

    expected_type = schema.get("type")
    if expected_type is not None:
        expected_types = expected_type if isinstance(expected_type, list) else [expected_type]
        if not any(type_matches(value, instance) for value in expected_types):
            errors.append("%s: expected type %s, got %s" % (location, expected_types, type(instance).__name__))
            return errors

    if isinstance(instance, str) and "minLength" in schema:
        if len(instance) < int(schema["minLength"]):
            errors.append("%s: expected minLength %s" % (location, schema["minLength"]))

    if isinstance(instance, dict):
        for key in schema.get("required", []):
            if key not in instance:
                errors.append("%s: missing required key %s" % (location, key))
        properties = schema.get("properties", {})
        if isinstance(properties, dict):
            for key, child_schema in properties.items():
                if key in instance and isinstance(child_schema, dict):
                    errors.extend(validate_schema_subset(child_schema, instance[key], "%s.%s" % (location, key)))

    if isinstance(instance, list):
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(instance):
                errors.extend(validate_schema_subset(item_schema, item, "%s[%d]" % (location, index)))

    return errors


def validate_updated_at(instance: Mapping[str, Any]) -> List[str]:
    updated_at = instance.get("updated_at")
    if not isinstance(updated_at, str):
        return ["$.updated_at: expected ISO-8601 string"]
    value = updated_at.replace("Z", "+00:00")
    try:
        datetime.fromisoformat(value)
    except ValueError:
        return ["$.updated_at: invalid ISO-8601 datetime %r" % updated_at]
    return []


def validate_pack_semantics(name: str, instance: Mapping[str, Any]) -> List[str]:
    errors: List[str] = []

    if name == "experiment_pack":
        errors.extend(experiment_pack_usage_errors(instance))
    elif name == "claim_ledger":
        errors.extend(claim_ledger_usage_errors(instance))
    elif name == "figure_manifest":
        errors.extend(duplicate_id_errors(instance.get("figures"), "id", "$.figures", "figure id"))
    elif name == "citation_cache":
        errors.extend(citation_cache_usage_errors(instance))
    elif name == "paper_package":
        errors.extend(paper_package_semantic_errors(instance))

    return errors


def validate_pack_warnings(repo: Path, name: str, instance: Mapping[str, Any]) -> List[str]:
    warnings: List[str] = []

    if name == "claim_ledger":
        warnings.extend(claim_ledger_usage_warnings(instance))
    elif name == "figure_manifest":
        warnings.extend(figure_manifest_usage_warnings(repo, instance))

    return warnings


def non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def non_empty_string_list(value: Any) -> bool:
    return isinstance(value, list) and any(non_empty_string(item) for item in value)


def has_clear_limitations(claim: Mapping[str, Any]) -> bool:
    limitations = claim.get("limitations")
    if non_empty_string(limitations):
        return True
    return non_empty_string_list(limitations)


def duplicate_id_errors(items: Any, key: str, location: str, label: str) -> List[str]:
    errors: List[str] = []
    if not isinstance(items, list):
        return errors

    seen: Dict[str, int] = {}
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        value = item.get(key)
        if not non_empty_string(value):
            continue
        if value in seen:
            errors.append(
                "%s[%d].%s: duplicate %s %r also used at %s[%d]"
                % (location, index, key, label, value, location, seen[value])
            )
        else:
            seen[value] = index
    return errors


def experiment_pack_usage_errors(instance: Mapping[str, Any], location: str = "$") -> List[str]:
    errors: List[str] = []
    diagnostics = instance.get("formal_diagnostics")
    if not isinstance(diagnostics, list):
        return errors

    errors.extend(
        duplicate_id_errors(
            diagnostics,
            "id",
            "%s.formal_diagnostics" % location,
            "formal diagnostic id",
        )
    )

    probe_ids = ids_from_items(instance.get("probes"), "id")
    probe_artifacts = {
        "experiment/PROBE_REPORT.md",
        "experiment/PROBE_AUDIT.md",
        "experiment/HEADROOM_NOTE.md",
    }

    for index, diagnostic in enumerate(list_items(diagnostics)):
        diag_id = diagnostic.get("id") or index
        diag_loc = "%s.formal_diagnostics[%d]" % (location, index)

        if diagnostic.get("id") in probe_ids:
            errors.append(
                "%s.id: formal diagnostic %r reuses a probe id; probes do not satisfy formal diagnostics"
                % (diag_loc, diag_id)
            )

        kind = diagnostic.get("kind")
        if kind not in VALID_DIAGNOSTIC_KINDS:
            errors.append("%s.kind: invalid formal diagnostic kind %r" % (diag_loc, kind))

        relevance = diagnostic.get("claim_relevance")
        if relevance not in VALID_CLAIM_RELEVANCE:
            errors.append(
                "%s.claim_relevance: invalid claim relevance %r"
                % (diag_loc, relevance)
            )

        if not formal_diagnostic_input(diagnostic):
            errors.append(
                "%s.command: formal diagnostic %r must include command, manifest, manifest_path, or grid_spec"
                % (diag_loc, diag_id)
            )

        for key in ("command", "manifest", "manifest_path", "grid_spec", "expected_report", "expected_audit"):
            value = diagnostic.get(key)
            if non_empty_string(value) and str(value).strip() in probe_artifacts:
                errors.append(
                    "%s.%s: formal diagnostic %r points at STOP B probe artifact %s"
                    % (diag_loc, key, diag_id, value)
                )
    return errors


def formal_diagnostic_input(entry: Mapping[str, Any]) -> Optional[str]:
    for key in ("command", "manifest", "manifest_path", "grid_spec"):
        value = entry.get(key)
        if non_empty_string(value):
            return str(value).strip()
    return None


def claim_ledger_usage_errors(instance: Mapping[str, Any], location: str = "$") -> List[str]:
    """Return semantic errors that would let unsupported claims leak into paper prose."""
    return claim_ledger_semantic_errors(instance, location)


def claim_ledger_usage_warnings(instance: Mapping[str, Any], location: str = "$") -> List[str]:
    warnings: List[str] = []
    claims = instance.get("claims")
    if not isinstance(claims, list):
        return warnings

    for index, claim in enumerate(claims):
        if not isinstance(claim, dict):
            continue
        if claim.get("paper_use") != "allowed":
            continue
        evidence_refs = claim.get("evidence_refs")
        if non_empty_string_list(evidence_refs):
            continue
        claim_id = claim.get("id") or index
        warnings.append(
            "%s.claims[%d].evidence_refs: allowed claim %r has no evidence refs"
            % (location, index, claim_id)
        )
    return warnings


def figure_manifest_usage_warnings(repo: Path, instance: Mapping[str, Any], location: str = "$") -> List[str]:
    warnings: List[str] = []
    figures = instance.get("figures")
    if not isinstance(figures, list):
        return warnings
    for index, figure in enumerate(figures):
        if not isinstance(figure, dict):
            continue
        if figure.get("status") != "verified":
            continue
        output = figure.get("output")
        if not non_empty_string(output):
            continue
        if (repo / output).exists():
            continue
        figure_id = figure.get("id") or index
        warnings.append(
            "%s.figures[%d].output: verified figure %r output path does not exist: %s"
            % (location, index, figure_id, output)
        )
    return warnings


def citation_cache_usage_errors(instance: Mapping[str, Any], location: str = "$") -> List[str]:
    errors = duplicate_id_errors(instance.get("citations"), "key", "%s.citations" % location, "citation key")
    citations = instance.get("citations")
    if not isinstance(citations, list):
        return errors

    for index, citation in enumerate(citations):
        if not isinstance(citation, dict):
            continue
        if citation.get("verified") is not True:
            continue
        if non_empty_string(citation.get("source")):
            continue
        key = citation.get("key") or index
        errors.append(
            "%s.citations[%d].source: verified citation %r must include a source"
            % (location, index, key)
        )
    return errors


def paper_package_semantic_errors(instance: Mapping[str, Any], location: str = "$") -> List[str]:
    errors: List[str] = []
    if instance.get("status") != "ready":
        return errors

    compile_status = instance.get("compile_status")
    if not status_passes(compile_status):
        errors.append("%s.compile_status: ready paper_package requires passing compile_status" % location)

    audits = instance.get("audits")
    if not isinstance(audits, list) or not audits:
        errors.append("%s.audits: ready paper_package requires at least one audit" % location)
    elif not all(audit_passes(audit) for audit in audits):
        errors.append("%s.audits: ready paper_package contains a non-passing audit" % location)

    return errors


def status_passes(value: Any) -> bool:
    passing = {"pass", "passed", "ok", "ready", "success"}
    if isinstance(value, str):
        return value.strip().lower() in passing
    if isinstance(value, dict):
        return status_passes(value.get("status"))
    return False


def audit_passes(value: Any) -> bool:
    if isinstance(value, dict):
        return status_passes(value.get("status"))
    if isinstance(value, str):
        return status_passes(value)
    return False


def errors_for_claim(errors: List[str], claim_location: str) -> bool:
    return any(error.startswith(claim_location + ".") for error in errors)


def append_cross_pack_errors(repo: Path, report: Dict[str, Any]) -> None:
    append_figure_claim_reference_errors(repo, report)
    append_paper_package_readiness_errors(repo, report)
    append_stop_c_approval_errors(repo, report)
    append_claim_ledger_usage_errors(repo, report)

    paper_path = pack_path(repo, "paper_package")
    citation_path = pack_path(repo, "citation_cache")
    if not paper_path.exists() or not citation_path.exists():
        return

    paper_package = parse_json_or_none(paper_path)
    citation_cache = parse_json_or_none(citation_path)
    if not isinstance(paper_package, dict) or not isinstance(citation_cache, dict):
        return
    if paper_package.get("status") != "ready":
        return
    if not paper_package.get("citation_cache_ref"):
        return

    citations = citation_cache.get("citations")
    if not isinstance(citations, list):
        return

    errors: List[str] = []
    for index, citation in enumerate(citations):
        if not isinstance(citation, dict):
            continue
        if citation.get("verified") is True:
            continue
        key = citation.get("key") or index
        errors.append(
            "$.citations[%d].verified: unverified citation %r cannot be used when paper_package status is ready"
            % (index, key)
        )
    if not errors:
        return

    for result in report["results"]:
        if result.get("name") == "citation_cache":
            result["status"] = "error"
            result.setdefault("errors", []).extend(errors)
            return


def result_for(report: Dict[str, Any], name: str) -> Optional[Dict[str, Any]]:
    for result in report["results"]:
        if result.get("name") == name:
            return result
    return None


def add_result_errors(report: Dict[str, Any], name: str, errors: List[str]) -> None:
    if not errors:
        return
    result = result_for(report, name)
    if result is None:
        return
    result["status"] = "error"
    result.setdefault("errors", []).extend(errors)


def add_result_warnings(report: Dict[str, Any], name: str, warnings: List[str]) -> None:
    if not warnings:
        return
    result = result_for(report, name)
    if result is None:
        return
    if result.get("status") == "ok":
        result["status"] = "warning"
    result.setdefault("warnings", []).extend(warnings)


def append_figure_claim_reference_errors(repo: Path, report: Dict[str, Any]) -> None:
    if result_for(report, "figure_manifest") is None:
        return
    manifest_path = pack_path(repo, "figure_manifest")
    ledger_path = pack_path(repo, "claim_ledger")
    if not manifest_path.exists() or not ledger_path.exists():
        return

    manifest = parse_json_or_none(manifest_path)
    ledger = parse_json_or_none(ledger_path)
    if not isinstance(manifest, dict) or not isinstance(ledger, dict):
        return

    claim_ids = ids_from_items(ledger.get("claims"), "id")
    errors: List[str] = []
    for index, figure in enumerate(list_items(manifest.get("figures"))):
        supports = figure.get("supports_claims")
        if not isinstance(supports, list):
            continue
        for claim_id in supports:
            if not non_empty_string(claim_id) or claim_id in claim_ids:
                continue
            figure_id = figure.get("id") or index
            errors.append(
                "$.figures[%d].supports_claims: figure %r references missing claim id %r"
                % (index, figure_id, claim_id)
            )
    add_result_errors(report, "figure_manifest", errors)


def append_paper_package_readiness_errors(repo: Path, report: Dict[str, Any]) -> None:
    if result_for(report, "paper_package") is None:
        return
    paper_path = pack_path(repo, "paper_package")
    if not paper_path.exists():
        return

    paper_package = parse_json_or_none(paper_path)
    if not isinstance(paper_package, dict) or paper_package.get("status") != "ready":
        return

    errors: List[str] = []
    errors.extend(validate_paper_package_pdf(repo, paper_package))
    errors.extend(validate_referenced_pack(repo, paper_package, "claim_ledger_ref", "claim_ledger"))
    errors.extend(validate_paper_package_figure_refs(repo, paper_package))
    errors.extend(validate_paper_package_citation_refs(repo, paper_package))
    add_result_errors(report, "paper_package", errors)


def validate_paper_package_pdf(repo: Path, paper_package: Mapping[str, Any]) -> List[str]:
    compile_status = paper_package.get("compile_status")
    pdf = None
    if isinstance(compile_status, Mapping):
        for key in ("pdf", "pdf_path", "output_pdf", "output"):
            value = compile_status.get(key)
            if non_empty_string(value):
                pdf = str(value).strip()
                break
    if not pdf:
        return ["$.compile_status.pdf: ready paper_package requires declared compiled PDF path"]
    if not (repo / pdf).exists():
        return ["$.compile_status.pdf: declared compiled PDF does not exist: %s" % pdf]
    return []


def validate_referenced_pack(repo: Path, paper_package: Mapping[str, Any], ref_key: str, pack_name: str) -> List[str]:
    ref = paper_package.get(ref_key)
    if not non_empty_string(ref):
        return ["$.%s: ready paper_package requires %s" % (ref_key, ref_key)]
    path = repo / ref
    if not path.exists():
        return ["$.%s: referenced %s does not exist: %s" % (ref_key, pack_name, ref)]
    result = validate_pack_file(repo, pack_name, path)
    return [
        "$.%s: referenced %s validation failed: %s" % (ref_key, pack_name, error)
        for error in result.get("errors", [])
    ]


def validate_paper_package_figure_refs(repo: Path, paper_package: Mapping[str, Any]) -> List[str]:
    refs = string_refs_from_fields(
        paper_package,
        ("figure_refs", "figure_ids", "referenced_figures", "figures"),
    )
    if not refs:
        return []

    ref = paper_package.get("figure_manifest_ref")
    if not non_empty_string(ref):
        return ["$.figure_manifest_ref: ready paper_package has figure refs but no figure_manifest_ref"]
    manifest_path = repo / ref
    if not manifest_path.exists():
        return ["$.figure_manifest_ref: referenced figure_manifest does not exist: %s" % ref]
    manifest = parse_json_or_none(manifest_path)
    if not isinstance(manifest, dict):
        return ["$.figure_manifest_ref: referenced figure_manifest is not valid JSON: %s" % ref]

    figures = {
        str(figure.get("id")): figure
        for figure in list_items(manifest.get("figures"))
        if non_empty_string(figure.get("id"))
    }
    errors = []
    for figure_id in refs:
        figure = figures.get(figure_id)
        if figure is None:
            errors.append("$.figure_refs: referenced figure id %r is missing from figure_manifest" % figure_id)
            continue
        if figure.get("status") != "verified":
            errors.append("$.figure_refs: referenced figure id %r is not verified" % figure_id)
            continue
        output = figure.get("output")
        if not non_empty_string(output):
            errors.append("$.figure_refs: verified figure id %r has no output path" % figure_id)
            continue
        if not (repo / str(output).strip()).exists():
            errors.append(
                "$.figure_refs: verified figure id %r output path does not exist: %s"
                % (figure_id, output)
            )
    return errors


def validate_paper_package_citation_refs(repo: Path, paper_package: Mapping[str, Any]) -> List[str]:
    refs = string_refs_from_fields(
        paper_package,
        ("citation_refs", "citation_keys", "referenced_citations", "citations"),
    )
    ref = paper_package.get("citation_cache_ref")
    if not non_empty_string(ref):
        return ["$.citation_cache_ref: ready paper_package requires citation_cache_ref"]
    cache_path = repo / ref
    if not cache_path.exists():
        return ["$.citation_cache_ref: referenced citation_cache does not exist: %s" % ref]
    cache = parse_json_or_none(cache_path)
    if not isinstance(cache, dict):
        return ["$.citation_cache_ref: referenced citation_cache is not valid JSON: %s" % ref]

    citations = {
        str(citation.get("key")): citation
        for citation in list_items(cache.get("citations"))
        if non_empty_string(citation.get("key"))
    }
    keys_to_check = refs or sorted(citations)
    errors: List[str] = []
    for key in keys_to_check:
        citation = citations.get(key)
        if citation is None:
            errors.append("$.citation_refs: referenced citation key %r is missing from citation_cache" % key)
            continue
        if citation.get("verified") is not True:
            errors.append("$.citation_refs: referenced citation key %r is not verified" % key)
    return errors


def list_items(value: Any) -> List[Mapping[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def ids_from_items(items: Any, key: str) -> set[str]:
    return {
        str(item.get(key))
        for item in list_items(items)
        if non_empty_string(item.get(key))
    }


def string_refs_from_fields(instance: Mapping[str, Any], keys: tuple[str, ...]) -> List[str]:
    refs: List[str] = []
    for key in keys:
        refs.extend(string_refs_from_value(instance.get(key)))
    return sorted(set(refs))


def string_refs_from_value(value: Any) -> List[str]:
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    if not isinstance(value, list):
        return []
    refs: List[str] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            refs.append(item.strip())
        elif isinstance(item, dict):
            for key in ("id", "key", "figure_id", "citation_key"):
                maybe = item.get(key)
                if isinstance(maybe, str) and maybe.strip():
                    refs.append(maybe.strip())
                    break
    return refs


def append_claim_ledger_usage_errors(repo: Path, report: Dict[str, Any]) -> None:
    paper_path = pack_path(repo, "paper_package")
    if not paper_path.exists():
        return

    paper_package = parse_json_or_none(paper_path)
    if not isinstance(paper_package, dict) or paper_package.get("status") != "ready":
        return
    claim_ledger_ref = paper_package.get("claim_ledger_ref")
    if not isinstance(claim_ledger_ref, str) or not claim_ledger_ref:
        return

    ledger_path = repo / claim_ledger_ref
    ledger = parse_json_or_none(ledger_path)
    if not isinstance(ledger, dict):
        return
    errors = [
        "claim ledger usage blocks ready paper_package: %s" % error
        for error in claim_ledger_usage_errors(ledger, "$")
    ]
    if not errors:
        return

    for result in report["results"]:
        if result.get("name") == "paper_package":
            result["status"] = "error"
            result.setdefault("errors", []).extend(errors)
            return


def append_stop_c_approval_errors(repo: Path, report: Dict[str, Any]) -> None:
    paper_path = pack_path(repo, "paper_package")
    if not paper_path.exists():
        return

    paper_package = parse_json_or_none(paper_path)
    if not isinstance(paper_package, dict):
        return
    if paper_package.get("status") != "ready":
        return

    claim_ledger_ref = paper_package.get("claim_ledger_ref")
    if not isinstance(claim_ledger_ref, str) or not claim_ledger_ref:
        return

    approval = evaluate_stop_c_approval(repo, claim_ledger_ref)
    if approval.get("status") == "approved":
        return

    errors = [
        "STOP C approval required for ready claim-bearing paper_package: %s" % error
        for error in approval.get("errors", [])
    ]
    if not errors:
        errors = ["STOP C approval required for ready claim-bearing paper_package"]

    for result in report["results"]:
        if result.get("name") == "paper_package":
            result["status"] = "error"
            result.setdefault("errors", []).extend(errors)
            result.setdefault("warnings", []).extend(
                "STOP C approval warning: %s" % warning
                for warning in approval.get("warnings", [])
            )
            return


def parse_json_or_none(path: Path) -> Optional[Any]:
    try:
        return load_json(path)
    except (OSError, json.JSONDecodeError):
        return None


def infer_pack_name(path: Path) -> Optional[str]:
    rel = path.as_posix()
    for name, spec in PACK_SPECS.items():
        if rel.endswith(spec.rel_path):
            return name
    basename = path.name
    for name, spec in PACK_SPECS.items():
        if basename == Path(spec.rel_path).name:
            return name
    return None


def normalize_pack_kind(kind: Optional[str]) -> Optional[str]:
    if kind is None:
        return None
    normalized = kind.strip().replace("-", "_")
    aliases = {
        "proposal": "proposal_pack",
        "experiment": "experiment_pack",
        "claims": "claim_ledger",
        "claim": "claim_ledger",
        "figures": "figure_manifest",
        "figure": "figure_manifest",
        "citations": "citation_cache",
        "citation": "citation_cache",
        "paper": "paper_package",
    }
    return aliases.get(normalized, normalized)


def relative_to_or_none(path: Path, root: Path) -> Optional[Path]:
    try:
        return path.relative_to(root)
    except ValueError:
        return None


def validate_pack_file(repo: Path, name: str, path: Path) -> Dict[str, Any]:
    spec = get_pack_spec(name)
    result = {
        "name": name,
        "path": spec.rel_path if path == repo / spec.rel_path else path.as_posix(),
        "schema": spec.schema_path,
        "status": "ok",
        "warnings": [],
        "errors": [],
    }

    project_schema_path = schema_path(repo, name)
    fallback_schema_path = schema_path(TOOL_REPO_ROOT, name)
    effective_schema_path = project_schema_path if project_schema_path.exists() else fallback_schema_path

    try:
        schema = load_json(effective_schema_path)
    except (OSError, json.JSONDecodeError) as exc:
        result["status"] = "error"
        result["errors"].append("could not load schema: %s" % exc)
        return result

    try:
        instance = load_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        result["status"] = "error"
        result["errors"].append("could not load pack JSON: %s" % exc)
        return result

    errors = validate_schema_subset(schema, instance)
    warnings: List[str] = []
    if isinstance(instance, dict):
        errors.extend(validate_updated_at(instance))
        errors.extend(validate_pack_semantics(name, instance))
        warnings.extend(validate_pack_warnings(repo, name, instance))
    else:
        errors.append("$: pack root must be a JSON object")

    if errors:
        result["status"] = "error"
        result["errors"].extend(errors)
    elif warnings:
        result["status"] = "warning"
        result["warnings"].extend(warnings)
    return result


def warning_for_missing(repo: Path, name: str) -> Dict[str, Any]:
    spec = get_pack_spec(name)
    return {
        "name": name,
        "path": spec.rel_path,
        "schema": spec.schema_path,
        "status": "warning",
        "warnings": ["missing pack; allowed during incremental migration"],
        "errors": [],
    }


def validate_selection(repo: Path, args: argparse.Namespace) -> Dict[str, Any]:
    selected: List[str]
    path_override: Optional[Path] = None

    if args.path:
        path_override = Path(args.path)
        if not path_override.is_absolute():
            path_override = repo / path_override
        inferred = infer_pack_name(relative_to_or_none(path_override, repo) or path_override)
        if not inferred:
            return {
                "results": [
                    {
                        "name": None,
                        "path": path_override.as_posix(),
                        "schema": None,
                        "status": "error",
                        "warnings": [],
                        "errors": ["could not infer pack type from path; use a canonical pack filename"],
                    }
                ]
            }
        selected = [inferred]
    elif args.pack:
        selected = [args.pack]
    elif args.kind:
        kind = normalize_pack_kind(args.kind)
        if kind not in PACK_SPECS:
            return {
                "results": [
                    {
                        "name": kind,
                        "path": None,
                        "schema": None,
                        "status": "error",
                        "warnings": [],
                        "errors": [
                            "unknown pack kind %r; valid packs: %s"
                            % (args.kind, ", ".join(pack_names()))
                        ],
                    }
                ]
            }
        selected = [kind]
    else:
        selected = pack_names()

    results = []
    for name in selected:
        path = path_override if path_override is not None else pack_path(repo, name)
        if not path.exists():
            results.append(warning_for_missing(repo, name))
        else:
            results.append(validate_pack_file(repo, name, path))
    report = {"results": results}
    append_cross_pack_errors(repo, report)
    return report


def print_pretty(report: Mapping[str, Any]) -> None:
    print("ORBIT pack validation")
    for result in report["results"]:
        status = result["status"]
        prefix = {"ok": "[ok]", "warning": "[warn]", "error": "[error]"}.get(status, "[?]")
        print("%s %s: %s" % (prefix, result.get("name") or "unknown", result["path"]))
        for warning in result.get("warnings", []):
            print("  warning: %s" % warning)
        for error in result.get("errors", []):
            print("  error: %s" % error)

    ok_count = sum(1 for result in report["results"] if result["status"] == "ok")
    warning_count = sum(1 for result in report["results"] if result["status"] == "warning")
    error_count = sum(1 for result in report["results"] if result["status"] == "error")
    print("Summary: %d ok, %d warning, %d error" % (ok_count, warning_count, error_count))


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    repo = Path(args.repo).resolve()
    report = validate_selection(repo, args)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True))
    else:
        print_pretty(report)
    return 1 if any(result["status"] == "error" for result in report["results"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
