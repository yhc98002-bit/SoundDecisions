#!/usr/bin/env python3
"""Helpers for ORBIT machine-readable packs.

The pack layer makes JSON the source of truth while old Markdown remains
readable as views during migration. Bootstrap helpers are intentionally
best-effort: they record source Markdown paths and light snippets, but do not
pretend to fully parse legacy prose.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


SCHEMA_VERSION = "0.1"
VALID_PACK_STATUSES = {"draft", "ready", "blocked", "deprecated"}
PROPOSAL_VIEW_PATH = "proposal/PROPOSAL.md"
PROPOSAL_METHOD_VIEW_PATH = "proposal/METHOD_SPEC.md"
EXPERIMENT_PLAN_VIEW_PATH = "experiment/EXPERIMENT_PLAN.md"
EXPERIMENT_EXEC_VIEW_PATH = "experiment/EXPERIMENT_PLAN_EXEC.md"
EXPERIMENT_PROBE_REPORT_PATH = "experiment/PROBE_REPORT.md"
EXPERIMENT_PROBE_AUDIT_PATH = "experiment/PROBE_AUDIT.md"
EXPERIMENT_HEADROOM_NOTE_PATH = "experiment/HEADROOM_NOTE.md"
CLAIM_LEDGER_VIEW_PATH = "claims/CLAIM_LEDGER.md"
LEGACY_CLAIM_CONSTRUCTION_PATH = "orbit-research/CLAIM_CONSTRUCTION.md"
FIGURE_MANIFEST_VIEW_PATH = "figures/FIGURE_MANIFEST.md"
CITATION_CACHE_VIEW_PATH = "references/CITATION_CACHE.md"
LEGACY_PROPOSAL_VIEW_PATHS = (
    "refine-logs/FINAL_PROPOSAL.md",
    "refine-logs/FINAL_PROPOSAL_SHORT.md",
    "refine-logs/METHOD_SPEC.md",
)


@dataclass(frozen=True)
class PackSpec:
    name: str
    rel_path: str
    schema_path: str
    stop: str
    source_candidates: tuple[str, ...]


PACK_SPECS: Dict[str, PackSpec] = {
    "proposal_pack": PackSpec(
        name="proposal_pack",
        rel_path="proposal/proposal_pack.json",
        schema_path="schemas/proposal_pack.schema.json",
        stop="STOP_A",
        source_candidates=(
            "refine-logs/FINAL_PROPOSAL.md",
            "refine-logs/FINAL_PROPOSAL_SHORT.md",
            "orbit-research/PIPELINE_INTAKE.md",
            "orbit-research/PROBLEM_SELECTION.md",
            "orbit-research/ASSUMPTION_LEDGER.md",
            "orbit-research/ABSTRACT_TASK_MECHANISM.md",
            "orbit-research/BASELINE_CEILING.md",
            "orbit-research/MECHANISM_IDEATION.md",
            "orbit-research/METHOD_SPEC.md",
        ),
    ),
    "experiment_pack": PackSpec(
        name="experiment_pack",
        rel_path="experiment/experiment_pack.json",
        schema_path="schemas/experiment_pack.schema.json",
        stop="STOP_B",
        source_candidates=(
            "refine-logs/EXPERIMENT_PLAN.md",
            "refine-logs/EXPERIMENT_PLAN_EXEC.md",
            "orbit-research/DIAGNOSTIC_EXPERIMENT_PLAN.md",
            "orbit-research/COMPONENT_BUNDLE_LADDER.md",
            "orbit-research/CONTROL_DESIGN.md",
            "orbit-research/NULL_RESULT_CONTRACT.md",
            "orbit-research/ALGORITHMIC_FORMALIZATION.md",
            "orbit-research/PLAN_CODE_AUDIT.md",
            "experiment/EXPERIMENT_PLAN.md",
            "experiment/EXPERIMENT_PLAN_EXEC.md",
            "experiment/PROBE_REPORT.md",
            "experiment/PROBE_AUDIT.md",
            "experiment/HEADROOM_NOTE.md",
            "orbit-research/DIAGNOSTIC_RUN_REPORT.md",
            "orbit-research/DIAGNOSTIC_RUN_AUDIT.md",
        ),
    ),
    "claim_ledger": PackSpec(
        name="claim_ledger",
        rel_path="claims/claim_ledger.json",
        schema_path="schemas/claim_ledger.schema.json",
        stop="STOP_C",
        source_candidates=(
            "orbit-research/CLAIM_CONSTRUCTION.md",
            "orbit-research/RESULT_INTERPRETATION.md",
            "orbit-research/NEGATIVE_RESULT_STRATEGY.md",
            "orbit-research/RED_TEAM_REVIEW.md",
        ),
    ),
    "figure_manifest": PackSpec(
        name="figure_manifest",
        rel_path="figures/figure_manifest.json",
        schema_path="schemas/figure_manifest.schema.json",
        stop="SUPPORT",
        source_candidates=(
            "figures/README.md",
            "PAPER_PLAN.md",
            "paper/PAPER_PLAN.md",
            "paper/FIGURE_PLAN.md",
        ),
    ),
    "citation_cache": PackSpec(
        name="citation_cache",
        rel_path="references/citation_cache.json",
        schema_path="schemas/citation_cache.schema.json",
        stop="SUPPORT",
        source_candidates=(
            "references/README.md",
            "paper/references.bib",
            "paper/main.bib",
            "orbit-research/CITATION_AUDIT.md",
        ),
    ),
    "paper_package": PackSpec(
        name="paper_package",
        rel_path="paper/paper_package.json",
        schema_path="schemas/paper_package.schema.json",
        stop="STOP_D",
        source_candidates=(
            "paper/PAPER_PLAN.md",
            "paper/PAPER_DRAFT.md",
            "paper/PAPER_CLAIM_AUDIT.md",
            "paper/CITATION_AUDIT.md",
            "paper/PROOF_AUDIT.md",
            "paper/compile.log",
        ),
    ),
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def pack_names() -> List[str]:
    return sorted(PACK_SPECS)


def get_pack_spec(name: str) -> PackSpec:
    try:
        return PACK_SPECS[name]
    except KeyError as exc:
        raise ValueError("unknown pack %r; valid packs: %s" % (name, ", ".join(pack_names()))) from exc


def pack_path(repo: Path, name: str) -> Path:
    return repo / get_pack_spec(name).rel_path


def schema_path(repo: Path, name: str) -> Path:
    return repo / get_pack_spec(name).schema_path


def existing_sources(repo: Path, spec: PackSpec) -> List[str]:
    return sorted(path for path in spec.source_candidates if (repo / path).exists())


def read_snippet(repo: Path, rel_path: str, max_chars: int = 2000) -> str:
    path = repo / rel_path
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()[:max_chars]
    except OSError:
        return ""


def legacy_snippets(repo: Path, sources: Iterable[str], max_chars: int = 2000) -> List[Dict[str, str]]:
    snippets = []
    for rel_path in sources:
        text = read_snippet(repo, rel_path, max_chars=max_chars)
        if text:
            snippets.append({"path": rel_path, "snippet": text})
    return snippets


def base_pack(status: str = "draft", sources: Optional[List[str]] = None) -> Dict[str, Any]:
    if status not in VALID_PACK_STATUSES:
        raise ValueError("invalid pack status %r" % status)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "updated_at": utc_now_iso(),
        "source_markdown": sorted(sources or []),
        "generated_views": [],
    }


def empty_pack(name: str, status: str = "draft", sources: Optional[List[str]] = None) -> Dict[str, Any]:
    pack = base_pack(status=status, sources=sources)
    if name == "proposal_pack":
        pack.update(
            {
                "problem_selection": None,
                "assumptions": [],
                "abstract_task": None,
                "baseline_headroom": None,
                "candidate_mechanisms": [],
                "selected_sketch": None,
                "open_risks": [],
            }
        )
    elif name == "experiment_pack":
        pack.update(
            {
                "proposal_ref": "proposal/proposal_pack.json",
                "decision_tree": [],
                "controls": [],
                "null_result_contract": None,
                "component_ladder": [],
                "algorithmic_formalization": None,
                "plan_code_audit": None,
                "probes": [],
                "formal_diagnostics": [],
            }
        )
    elif name == "claim_ledger":
        pack.update({"claims": [], "result_refs": []})
    elif name == "figure_manifest":
        pack.update({"figures": []})
    elif name == "citation_cache":
        pack.update({"citations": []})
    elif name == "paper_package":
        pack.update(
            {
                "claim_ledger_ref": "claims/claim_ledger.json",
                "figure_manifest_ref": "figures/figure_manifest.json",
                "citation_cache_ref": "references/citation_cache.json",
                "compile_status": None,
                "audits": [],
            }
        )
    else:
        get_pack_spec(name)
    return pack


def unique_sorted(values: Iterable[str]) -> List[str]:
    return sorted({value for value in values if value})


def create_or_update_proposal_pack(
    repo: Path,
    updates: Optional[Mapping[str, Any]] = None,
    status: str = "draft",
    source_markdown: Optional[Sequence[str]] = None,
    generated_views: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Create or update STOP A's canonical proposal pack.

    Existing unknown fields are preserved so migration metadata can accumulate
    without schema churn. `updates` may contain any proposal_pack field.
    """
    existing = read_pack(repo, "proposal_pack")
    pack = dict(existing) if existing else empty_pack("proposal_pack", status=status)

    pack["schema_version"] = SCHEMA_VERSION
    pack["status"] = status or pack.get("status", "draft")
    if pack["status"] not in VALID_PACK_STATUSES:
        raise ValueError("invalid proposal_pack status %r" % pack["status"])
    pack["updated_at"] = utc_now_iso()

    if updates:
        pack.update(dict(updates))

    pack["source_markdown"] = unique_sorted(
        list(pack.get("source_markdown") or []) + list(source_markdown or [])
    )
    pack["generated_views"] = unique_sorted(
        list(pack.get("generated_views") or [])
        + list(generated_views or [PROPOSAL_VIEW_PATH, PROPOSAL_METHOD_VIEW_PATH])
    )

    for key, default in empty_pack("proposal_pack").items():
        pack.setdefault(key, default)
    return pack


def create_or_update_experiment_pack(
    repo: Path,
    updates: Optional[Mapping[str, Any]] = None,
    status: str = "draft",
    source_markdown: Optional[Sequence[str]] = None,
    generated_views: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Create or update STOP B's canonical experiment pack."""
    existing = read_pack(repo, "experiment_pack")
    pack = dict(existing) if existing else empty_pack("experiment_pack", status=status)

    pack["schema_version"] = SCHEMA_VERSION
    pack["status"] = status or pack.get("status", "draft")
    if pack["status"] not in VALID_PACK_STATUSES:
        raise ValueError("invalid experiment_pack status %r" % pack["status"])
    pack["updated_at"] = utc_now_iso()

    if updates:
        pack.update(dict(updates))

    default_views = [EXPERIMENT_PLAN_VIEW_PATH, EXPERIMENT_EXEC_VIEW_PATH]
    pack["source_markdown"] = unique_sorted(
        list(pack.get("source_markdown") or []) + list(source_markdown or [])
    )
    pack["generated_views"] = unique_sorted(
        list(pack.get("generated_views") or []) + list(generated_views or default_views)
    )

    for key, default in empty_pack("experiment_pack").items():
        pack.setdefault(key, default)
    return pack


def create_or_update_claim_ledger(
    repo: Path,
    updates: Optional[Mapping[str, Any]] = None,
    status: str = "draft",
    source_markdown: Optional[Sequence[str]] = None,
    generated_views: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Create or update STOP C's canonical claim ledger."""
    existing = read_pack(repo, "claim_ledger")
    pack = dict(existing) if existing else empty_pack("claim_ledger", status=status)

    pack["schema_version"] = SCHEMA_VERSION
    pack["status"] = status or pack.get("status", "draft")
    if pack["status"] not in VALID_PACK_STATUSES:
        raise ValueError("invalid claim_ledger status %r" % pack["status"])
    pack["updated_at"] = utc_now_iso()

    if updates:
        pack.update(dict(updates))

    default_views = [CLAIM_LEDGER_VIEW_PATH]
    pack["source_markdown"] = unique_sorted(
        list(pack.get("source_markdown") or []) + list(source_markdown or [])
    )
    pack["generated_views"] = unique_sorted(
        list(pack.get("generated_views") or []) + list(generated_views or default_views)
    )

    for key, default in empty_pack("claim_ledger").items():
        pack.setdefault(key, default)
    return pack


def merge_records_by_id(
    existing: Iterable[Any],
    incoming: Iterable[Mapping[str, Any]],
    id_key: str,
) -> List[Any]:
    records: Dict[str, Dict[str, Any]] = {}
    passthrough: List[Any] = []
    for item in existing:
        if isinstance(item, Mapping) and isinstance(item.get(id_key), str):
            records[str(item[id_key])] = dict(item)
        else:
            passthrough.append(item)
    for item in incoming:
        record_id = item.get(id_key)
        if not isinstance(record_id, str) or not record_id:
            passthrough.append(dict(item))
            continue
        merged = dict(records.get(record_id, {}))
        merged.update(dict(item))
        records[record_id] = merged
    return passthrough + [records[key] for key in sorted(records)]


def create_or_update_figure_manifest(
    repo: Path,
    figures: Optional[Sequence[Mapping[str, Any]]] = None,
    status: str = "draft",
    source_markdown: Optional[Sequence[str]] = None,
    generated_views: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Create or update the shared figure manifest by figure id."""
    existing = read_pack(repo, "figure_manifest")
    pack = dict(existing) if existing else empty_pack("figure_manifest", status=status)

    pack["schema_version"] = SCHEMA_VERSION
    pack["status"] = status or pack.get("status", "draft")
    if pack["status"] not in VALID_PACK_STATUSES:
        raise ValueError("invalid figure_manifest status %r" % pack["status"])
    pack["updated_at"] = utc_now_iso()

    if figures:
        pack["figures"] = merge_records_by_id(pack.get("figures") or [], figures, "id")

    pack["source_markdown"] = unique_sorted(
        list(pack.get("source_markdown") or []) + list(source_markdown or [])
    )
    pack["generated_views"] = unique_sorted(
        list(pack.get("generated_views") or []) + list(generated_views or [FIGURE_MANIFEST_VIEW_PATH])
    )

    for key, default in empty_pack("figure_manifest").items():
        pack.setdefault(key, default)
    return pack


def create_or_update_citation_cache(
    repo: Path,
    citations: Optional[Sequence[Mapping[str, Any]]] = None,
    status: str = "draft",
    source_markdown: Optional[Sequence[str]] = None,
    generated_views: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Create or update the shared citation cache by citation key."""
    existing = read_pack(repo, "citation_cache")
    pack = dict(existing) if existing else empty_pack("citation_cache", status=status)

    pack["schema_version"] = SCHEMA_VERSION
    pack["status"] = status or pack.get("status", "draft")
    if pack["status"] not in VALID_PACK_STATUSES:
        raise ValueError("invalid citation_cache status %r" % pack["status"])
    pack["updated_at"] = utc_now_iso()

    if citations:
        pack["citations"] = merge_records_by_id(pack.get("citations") or [], citations, "key")

    pack["source_markdown"] = unique_sorted(
        list(pack.get("source_markdown") or []) + list(source_markdown or [])
    )
    pack["generated_views"] = unique_sorted(
        list(pack.get("generated_views") or []) + list(generated_views or [CITATION_CACHE_VIEW_PATH])
    )

    for key, default in empty_pack("citation_cache").items():
        pack.setdefault(key, default)
    return pack


def markdown_value(value: Any) -> str:
    if value is None:
        return "_Not recorded yet._"
    if isinstance(value, str):
        return value.strip() or "_Not recorded yet._"
    if isinstance(value, list):
        if not value:
            return "_None recorded._"
        lines = []
        for item in value:
            if isinstance(item, dict):
                label = item.get("id") or item.get("name") or item.get("title") or item.get("statement")
                detail = item.get("summary") or item.get("text") or item.get("description") or item
                if label and detail != label:
                    lines.append("- %s: %s" % (label, detail))
                else:
                    lines.append("- %s" % detail)
            else:
                lines.append("- %s" % item)
        return "\n".join(lines)
    if isinstance(value, dict):
        lines = []
        for key in sorted(value):
            lines.append("- %s: %s" % (key, value[key]))
        return "\n".join(lines) if lines else "_Not recorded yet._"
    return str(value)


def render_proposal_pack_to_markdown(pack: Mapping[str, Any]) -> str:
    """Render the canonical proposal pack as a human-readable proposal view."""
    return "\n".join(
        [
            "# Proposal",
            "",
            "> Generated view of `proposal/proposal_pack.json`. Do not treat this Markdown file as the source of truth.",
            "",
            "## Status",
            "",
            "- Pack status: `%s`" % pack.get("status", "draft"),
            "- Updated at: `%s`" % pack.get("updated_at", ""),
            "- Safe next command: `/experiment-bridge \"proposal/proposal_pack.json\"`",
            "",
            "## Problem Selection",
            "",
            markdown_value(pack.get("problem_selection")),
            "",
            "## Abstract Task",
            "",
            markdown_value(pack.get("abstract_task")),
            "",
            "## Baseline Headroom",
            "",
            markdown_value(pack.get("baseline_headroom")),
            "",
            "## Candidate Mechanisms",
            "",
            markdown_value(pack.get("candidate_mechanisms")),
            "",
            "## Selected Sketch",
            "",
            markdown_value(pack.get("selected_sketch")),
            "",
            "## Assumptions",
            "",
            markdown_value(pack.get("assumptions")),
            "",
            "## Open Risks",
            "",
            markdown_value(pack.get("open_risks")),
            "",
            "## Source Markdown",
            "",
            markdown_value(pack.get("source_markdown")),
            "",
        ]
    )


def render_proposal_method_spec(pack: Mapping[str, Any]) -> str:
    """Render implementation-relevant STOP A method details as a view."""
    return "\n".join(
        [
            "# Method Spec",
            "",
            "> Generated view of `proposal/proposal_pack.json`. Detailed experiment planning still belongs to STOP B.",
            "",
            "## Selected Sketch",
            "",
            markdown_value(pack.get("selected_sketch")),
            "",
            "## Candidate Mechanisms",
            "",
            markdown_value(pack.get("candidate_mechanisms")),
            "",
            "## Assumptions",
            "",
            markdown_value(pack.get("assumptions")),
            "",
            "## Baseline Headroom",
            "",
            markdown_value(pack.get("baseline_headroom")),
            "",
        ]
    )


def render_experiment_pack_to_markdown(pack: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# Experiment Plan",
            "",
            "> Generated view of `experiment/experiment_pack.json`. Do not treat this Markdown file as the source of truth.",
            "",
            "## Status",
            "",
            "- Pack status: `%s`" % pack.get("status", "draft"),
            "- Updated at: `%s`" % pack.get("updated_at", ""),
            "- Proposal ref: `%s`" % pack.get("proposal_ref", ""),
            "- Safe next command: `/diagnostic-to-review \"experiment/experiment_pack.json\"`",
            "",
            "## Decision Tree",
            "",
            markdown_value(pack.get("decision_tree")),
            "",
            "## Controls",
            "",
            markdown_value(pack.get("controls")),
            "",
            "## Null Result Contract",
            "",
            markdown_value(pack.get("null_result_contract")),
            "",
            "## Component Ladder",
            "",
            markdown_value(pack.get("component_ladder")),
            "",
            "## Algorithmic Formalization",
            "",
            markdown_value(pack.get("algorithmic_formalization")),
            "",
            "## Plan-Code Audit",
            "",
            markdown_value(pack.get("plan_code_audit")),
            "",
            "## Probes",
            "",
            markdown_value(pack.get("probes")),
            "",
            "## Formal Diagnostics",
            "",
            markdown_value(pack.get("formal_diagnostics")),
            "",
        ]
    )


def render_experiment_exec_markdown(pack: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# Experiment Plan Exec",
            "",
            "> Generated execution view of `experiment/experiment_pack.json`.",
            "",
            "## Decision Tree / Branch Table",
            "",
            markdown_value(pack.get("decision_tree")),
            "",
            "## Formal Diagnostics",
            "",
            markdown_value(pack.get("formal_diagnostics")),
            "",
            "## Probes",
            "",
            "These are implementation/headroom probes only; formal diagnostic reports are owned by `/diagnostic-to-review`.",
            "",
            markdown_value(pack.get("probes")),
            "",
        ]
    )


def render_claim_ledger_to_markdown(pack: Mapping[str, Any]) -> str:
    """Render the canonical claim ledger as a human-readable STOP C view."""
    lines = [
        "# Claim Ledger",
        "",
        "> Generated view of `claims/claim_ledger.json`. Do not treat this Markdown file as the source of truth.",
        "",
        "## Status",
        "",
        "- Pack status: `%s`" % pack.get("status", "draft"),
        "- Updated at: `%s`" % pack.get("updated_at", ""),
        "- Safe next command after STOP C human approval: `/paper-from-claims \"claims/claim_ledger.json\"`",
        "",
        "## Result References",
        "",
        markdown_value(pack.get("result_refs")),
        "",
        "## Claims",
        "",
    ]
    claims = pack.get("claims")
    if not isinstance(claims, list) or not claims:
        lines.append("_No claims recorded yet._")
    else:
        for claim in claims:
            if not isinstance(claim, Mapping):
                lines.extend(["- %s" % claim])
                continue
            lines.extend(
                [
                    "### %s" % claim.get("id", "claim"),
                    "",
                    "- Statement: %s" % markdown_value(claim.get("statement")),
                    "- Status: `%s`" % claim.get("status", "draft"),
                    "- Scope: %s" % markdown_value(claim.get("scope")),
                    "",
                    "Evidence refs:",
                    markdown_value(claim.get("evidence_refs")),
                    "",
                    "Controls:",
                    markdown_value(claim.get("controls")),
                    "",
                    "Limitations:",
                    markdown_value(claim.get("limitations")),
                    "",
                    "Forbidden overclaims:",
                    markdown_value(claim.get("forbidden_overclaims")),
                    "",
                    "Allowed paper sections:",
                    markdown_value(claim.get("allowed_paper_sections")),
                    "",
                ]
            )
    lines.extend(["", "## Source Markdown", "", markdown_value(pack.get("source_markdown")), ""])
    return "\n".join(lines)


def render_proposal_views(
    pack: Mapping[str, Any],
    include_method_spec: bool = True,
    include_legacy: bool = False,
) -> Dict[str, str]:
    views = {PROPOSAL_VIEW_PATH: render_proposal_pack_to_markdown(pack)}
    if include_method_spec:
        views[PROPOSAL_METHOD_VIEW_PATH] = render_proposal_method_spec(pack)
    if include_legacy:
        views["refine-logs/FINAL_PROPOSAL.md"] = views[PROPOSAL_VIEW_PATH]
        views["refine-logs/FINAL_PROPOSAL_SHORT.md"] = views[PROPOSAL_VIEW_PATH]
        if include_method_spec:
            views["refine-logs/METHOD_SPEC.md"] = views[PROPOSAL_METHOD_VIEW_PATH]
    return views


def write_proposal_views(
    repo: Path,
    pack: Mapping[str, Any],
    include_method_spec: bool = True,
    include_legacy: bool = False,
) -> List[str]:
    written: List[str] = []
    for rel_path, text in render_proposal_views(pack, include_method_spec, include_legacy).items():
        path = repo / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        written.append(rel_path)
    return sorted(written)


def render_experiment_views(pack: Mapping[str, Any], include_legacy: bool = False) -> Dict[str, str]:
    views = {
        EXPERIMENT_PLAN_VIEW_PATH: render_experiment_pack_to_markdown(pack),
        EXPERIMENT_EXEC_VIEW_PATH: render_experiment_exec_markdown(pack),
    }
    if include_legacy:
        views["refine-logs/EXPERIMENT_PLAN.md"] = views[EXPERIMENT_PLAN_VIEW_PATH]
        views["refine-logs/EXPERIMENT_PLAN_EXEC.md"] = views[EXPERIMENT_EXEC_VIEW_PATH]
    return views


def write_experiment_views(
    repo: Path,
    pack: Mapping[str, Any],
    include_legacy: bool = False,
) -> List[str]:
    written: List[str] = []
    for rel_path, text in render_experiment_views(pack, include_legacy).items():
        path = repo / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        written.append(rel_path)
    return sorted(written)


def render_claim_views(pack: Mapping[str, Any], include_legacy: bool = False) -> Dict[str, str]:
    view = render_claim_ledger_to_markdown(pack)
    views = {CLAIM_LEDGER_VIEW_PATH: view}
    if include_legacy:
        views[LEGACY_CLAIM_CONSTRUCTION_PATH] = view
    return views


def write_claim_views(
    repo: Path,
    pack: Mapping[str, Any],
    include_legacy: bool = False,
) -> List[str]:
    written: List[str] = []
    for rel_path, text in render_claim_views(pack, include_legacy).items():
        path = repo / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        written.append(rel_path)
    return sorted(written)


def render_figure_manifest_to_markdown(pack: Mapping[str, Any]) -> str:
    lines = [
        "# Figure Manifest",
        "",
        "> Generated view of `figures/figure_manifest.json`. The JSON file is the source of truth.",
        "",
        "## Status",
        "",
        "- Pack status: `%s`" % pack.get("status", "draft"),
        "- Updated at: `%s`" % pack.get("updated_at", ""),
        "",
        "## Figures",
        "",
    ]
    figures = pack.get("figures")
    if not isinstance(figures, list) or not figures:
        lines.append("_No figures recorded yet._")
    else:
        for figure in figures:
            if not isinstance(figure, Mapping):
                lines.append("- %s" % figure)
                continue
            lines.extend(
                [
                    "### %s" % figure.get("id", "figure"),
                    "",
                    "- Type: %s" % markdown_value(figure.get("type")),
                    "- Data source: %s" % markdown_value(figure.get("data_source")),
                    "- Generator: %s" % markdown_value(figure.get("generator")),
                    "- Output: %s" % markdown_value(figure.get("output")),
                    "- LaTeX label: %s" % markdown_value(figure.get("latex_label")),
                    "- Status: `%s`" % figure.get("status", "draft"),
                    "",
                    "Supports claims:",
                    markdown_value(figure.get("supports_claims")),
                    "",
                ]
            )
    return "\n".join(lines)


def render_citation_cache_to_markdown(pack: Mapping[str, Any]) -> str:
    lines = [
        "# Citation Cache",
        "",
        "> Generated view of `references/citation_cache.json`. The JSON file is the source of truth.",
        "",
        "## Status",
        "",
        "- Pack status: `%s`" % pack.get("status", "draft"),
        "- Updated at: `%s`" % pack.get("updated_at", ""),
        "",
        "## Citations",
        "",
    ]
    citations = pack.get("citations")
    if not isinstance(citations, list) or not citations:
        lines.append("_No citations recorded yet._")
    else:
        for citation in citations:
            if not isinstance(citation, Mapping):
                lines.append("- %s" % citation)
                continue
            lines.extend(
                [
                    "### %s" % citation.get("key", "citation"),
                    "",
                    "- Title: %s" % markdown_value(citation.get("title")),
                    "- Venue: %s" % markdown_value(citation.get("venue")),
                    "- Year: %s" % markdown_value(citation.get("year")),
                    "- Verified source: %s" % markdown_value(citation.get("source")),
                    "- Verified: `%s`" % citation.get("verified", False),
                    "",
                    "Authors:",
                    markdown_value(citation.get("authors")),
                    "",
                    "Used for:",
                    markdown_value(citation.get("used_for")),
                    "",
                    "Contexts:",
                    markdown_value(citation.get("contexts")),
                    "",
                ]
            )
    return "\n".join(lines)


def render_figure_views(pack: Mapping[str, Any]) -> Dict[str, str]:
    return {FIGURE_MANIFEST_VIEW_PATH: render_figure_manifest_to_markdown(pack)}


def render_citation_views(pack: Mapping[str, Any]) -> Dict[str, str]:
    return {CITATION_CACHE_VIEW_PATH: render_citation_cache_to_markdown(pack)}


def write_figure_views(repo: Path, pack: Mapping[str, Any]) -> List[str]:
    written: List[str] = []
    for rel_path, text in render_figure_views(pack).items():
        path = repo / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        written.append(rel_path)
    return sorted(written)


def write_citation_views(repo: Path, pack: Mapping[str, Any]) -> List[str]:
    written: List[str] = []
    for rel_path, text in render_citation_views(pack).items():
        path = repo / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        written.append(rel_path)
    return sorted(written)


def bootstrap_pack_from_markdown(repo: Path, name: str, status: str = "draft") -> Dict[str, Any]:
    spec = get_pack_spec(name)
    sources = existing_sources(repo, spec)
    pack = empty_pack(name, status=status, sources=sources)
    pack["legacy_bootstrap"] = {
        "mode": "best_effort_markdown_inventory",
        "source_count": len(sources),
        "snippets": legacy_snippets(repo, sources),
    }
    return pack


def read_pack(repo: Path, name: str) -> Optional[Dict[str, Any]]:
    path = pack_path(repo, name)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("%s must contain a JSON object" % get_pack_spec(name).rel_path)
    return data


def write_pack(repo: Path, name: str, pack: Mapping[str, Any]) -> Path:
    path = pack_path(repo, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(dict(pack), handle, indent=2, sort_keys=True, ensure_ascii=True)
        handle.write("\n")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create or inspect ORBIT machine-readable packs.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List known pack names and paths.")
    list_parser.add_argument("--repo", default=".", help="Repository root.")

    bootstrap_parser = subparsers.add_parser(
        "bootstrap",
        help="Bootstrap a pack from existing Markdown artifacts without perfect parsing.",
    )
    bootstrap_parser.add_argument("--repo", default=".", help="Repository root.")
    bootstrap_parser.add_argument("--pack", choices=pack_names(), required=True)
    bootstrap_parser.add_argument("--status", choices=sorted(VALID_PACK_STATUSES), default="draft")
    bootstrap_parser.add_argument("--write", action="store_true", help="Write the pack to its canonical path.")

    render_parser = subparsers.add_parser(
        "render-proposal",
        help="Render proposal/proposal_pack.json into Markdown views.",
    )
    render_parser.add_argument("--repo", default=".", help="Repository root.")
    render_parser.add_argument("--write", action="store_true", help="Write views instead of printing JSON.")
    render_parser.add_argument("--legacy", action="store_true", help="Also write refine-logs compatibility views.")

    render_experiment_parser = subparsers.add_parser(
        "render-experiment",
        help="Render experiment/experiment_pack.json into Markdown views.",
    )
    render_experiment_parser.add_argument("--repo", default=".", help="Repository root.")
    render_experiment_parser.add_argument("--write", action="store_true", help="Write views instead of printing JSON.")
    render_experiment_parser.add_argument("--legacy", action="store_true", help="Also write refine-logs compatibility views.")

    render_claim_parser = subparsers.add_parser(
        "render-claim",
        help="Render claims/claim_ledger.json into Markdown views.",
    )
    render_claim_parser.add_argument("--repo", default=".", help="Repository root.")
    render_claim_parser.add_argument("--write", action="store_true", help="Write views instead of printing JSON.")
    render_claim_parser.add_argument(
        "--legacy",
        action="store_true",
        help="Also write orbit-research/CLAIM_CONSTRUCTION.md compatibility view.",
    )

    render_figure_parser = subparsers.add_parser(
        "render-figure",
        help="Render figures/figure_manifest.json into a Markdown view.",
    )
    render_figure_parser.add_argument("--repo", default=".", help="Repository root.")
    render_figure_parser.add_argument("--write", action="store_true", help="Write views instead of printing JSON.")

    render_citation_parser = subparsers.add_parser(
        "render-citation",
        help="Render references/citation_cache.json into a Markdown view.",
    )
    render_citation_parser.add_argument("--repo", default=".", help="Repository root.")
    render_citation_parser.add_argument("--write", action="store_true", help="Write views instead of printing JSON.")

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = Path(getattr(args, "repo", ".")).resolve()

    if args.command == "list":
        for name in pack_names():
            spec = get_pack_spec(name)
            marker = "exists" if (repo / spec.rel_path).exists() else "missing"
            print("%s\t%s\t%s" % (name, spec.rel_path, marker))
        return 0

    if args.command == "bootstrap":
        pack = bootstrap_pack_from_markdown(repo, args.pack, status=args.status)
        if args.write:
            path = write_pack(repo, args.pack, pack)
            print("Wrote %s" % path.relative_to(repo).as_posix())
        else:
            print(json.dumps(pack, indent=2, sort_keys=True, ensure_ascii=True))
        return 0

    if args.command == "render-proposal":
        pack = read_pack(repo, "proposal_pack")
        if pack is None:
            print("Missing proposal/proposal_pack.json", file=sys.stderr)
            return 1
        views = render_proposal_views(pack, include_legacy=args.legacy)
        if args.write:
            for rel_path in write_proposal_views(repo, pack, include_legacy=args.legacy):
                print("Wrote %s" % rel_path)
        else:
            print(json.dumps(views, indent=2, sort_keys=True, ensure_ascii=True))
        return 0

    if args.command == "render-experiment":
        pack = read_pack(repo, "experiment_pack")
        if pack is None:
            print("Missing experiment/experiment_pack.json", file=sys.stderr)
            return 1
        views = render_experiment_views(pack, include_legacy=args.legacy)
        if args.write:
            for rel_path in write_experiment_views(repo, pack, include_legacy=args.legacy):
                print("Wrote %s" % rel_path)
        else:
            print(json.dumps(views, indent=2, sort_keys=True, ensure_ascii=True))
        return 0

    if args.command == "render-claim":
        pack = read_pack(repo, "claim_ledger")
        if pack is None:
            print("Missing claims/claim_ledger.json", file=sys.stderr)
            return 1
        views = render_claim_views(pack, include_legacy=args.legacy)
        if args.write:
            for rel_path in write_claim_views(repo, pack, include_legacy=args.legacy):
                print("Wrote %s" % rel_path)
        else:
            print(json.dumps(views, indent=2, sort_keys=True, ensure_ascii=True))
        return 0

    if args.command == "render-figure":
        pack = read_pack(repo, "figure_manifest")
        if pack is None:
            print("Missing figures/figure_manifest.json", file=sys.stderr)
            return 1
        views = render_figure_views(pack)
        if args.write:
            for rel_path in write_figure_views(repo, pack):
                print("Wrote %s" % rel_path)
        else:
            print(json.dumps(views, indent=2, sort_keys=True, ensure_ascii=True))
        return 0

    if args.command == "render-citation":
        pack = read_pack(repo, "citation_cache")
        if pack is None:
            print("Missing references/citation_cache.json", file=sys.stderr)
            return 1
        views = render_citation_views(pack)
        if args.write:
            for rel_path in write_citation_views(repo, pack):
                print("Wrote %s" % rel_path)
        else:
            print(json.dumps(views, indent=2, sort_keys=True, ensure_ascii=True))
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
