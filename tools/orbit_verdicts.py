#!/usr/bin/env python3
"""Shared conservative parsing for ORBIT final verdict tokens."""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Sequence


DEFAULT_LABELS = ("Final verdict", "Final decision", "Decision")
LIST_MARKER_RE = re.compile(r"^\s*(?:[-*]\s+|\d+\.\s+|>\s*)")


def normalize_token(value: str) -> str:
    value = value.strip().rstrip(".").strip()
    previous = None
    while previous != value:
        previous = value
        value = value.strip().strip("*_`").strip()
    return value.upper()


def normalize_allowed_tokens(tokens: Iterable[str]) -> List[str]:
    normalized: List[str] = []
    seen = set()
    for token in tokens:
        value = normalize_token(str(token))
        if not value or value in seen:
            continue
        seen.add(value)
        normalized.append(value)
    return normalized


def allowed_tokens_in_line(line: str, allowed_set: set[str]) -> List[str]:
    upper = line.upper()
    return sorted(
        {
            token
            for token in allowed_set
            if re.search(r"(?<![A-Z0-9_])%s(?![A-Z0-9_])" % re.escape(token), upper)
        }
    )


def is_list_item(line: str) -> bool:
    return bool(LIST_MARKER_RE.match(line))


def strip_heading_marker(line: str) -> str:
    return re.sub(r"^\s{0,3}#{1,6}\s+", "", line).strip()


def label_regex(labels: Sequence[str]) -> re.Pattern[str]:
    alternatives = []
    for label in labels:
        parts = [re.escape(part) for part in label.strip().split()]
        alternatives.append(r"\s+".join(parts))
    return re.compile(r"^(?:%s)\s*[:=\-]\s*(.+)$" % "|".join(alternatives), re.IGNORECASE)


def invalid_template_or_candidate(line: str, token_matches: Sequence[str]) -> bool:
    return "|" in line or "<" in line or ">" in line or len(token_matches) > 1


def extract_final_token(
    text: str,
    allowed_tokens: Iterable[str],
    labels: Sequence[str] = DEFAULT_LABELS,
    allow_final_bare: bool = True,
    require_exactly_one: bool = False,
) -> Dict[str, Any]:
    """Extract explicit final verdict/decision tokens.

    List items and blockquotes are never accepted as final approvals. A bare token is
    accepted only when it is the final non-empty line and is not a list item.
    """

    allowed = normalize_allowed_tokens(allowed_tokens)
    allowed_set = set(allowed)
    errors: List[str] = []
    occurrences: List[Dict[str, Any]] = []
    if not allowed_set:
        return {"verdict": None, "errors": ["expected verdict tokens are empty"], "occurrences": []}

    lines = [(line_no, raw_line.strip()) for line_no, raw_line in enumerate(text.splitlines(), start=1)]
    non_empty = [(line_no, line) for line_no, line in lines if line]
    final_non_empty_line = non_empty[-1][0] if non_empty else None
    final_label_re = label_regex(labels)

    for line_no, line in non_empty:
        token_matches = allowed_tokens_in_line(line, allowed_set)
        candidate_or_template = invalid_template_or_candidate(line, token_matches)
        list_item = is_list_item(line)
        clean = strip_heading_marker(line)
        match = final_label_re.search(clean)

        if match:
            if list_item:
                continue
            if candidate_or_template:
                if token_matches or "<" in line or ">" in line:
                    errors.append(
                        "line %d is a template or candidate list, not a final verdict" % line_no
                    )
                continue
            value = normalize_token(match.group(1))
            if value in allowed_set:
                occurrences.append({"line": line_no, "verdict": value})
            elif token_matches:
                errors.append(
                    "line %d mentions a verdict token but is not exactly one expected final verdict"
                    % line_no
                )
            continue

        if not allow_final_bare or line_no != final_non_empty_line:
            continue
        if list_item or candidate_or_template:
            continue
        value = normalize_token(clean)
        if value in allowed_set:
            occurrences.append({"line": line_no, "verdict": value})

    if require_exactly_one:
        if len(occurrences) == 0:
            errors.append("missing exactly one final verdict token from expected_verdict_tokens")
        elif len(occurrences) > 1:
            errors.append(
                "expected exactly one final verdict token, found %d at lines %s"
                % (len(occurrences), ", ".join(str(item["line"]) for item in occurrences))
            )

    verdict = None
    if occurrences and not errors:
        verdict = occurrences[-1]["verdict"]
    return {"verdict": verdict, "errors": errors, "occurrences": occurrences}


def parse_final_token(
    text: str,
    allowed_tokens: Iterable[str],
    labels: Sequence[str] = DEFAULT_LABELS,
    allow_final_bare: bool = True,
) -> str | None:
    return extract_final_token(
        text,
        allowed_tokens,
        labels=labels,
        allow_final_bare=allow_final_bare,
        require_exactly_one=False,
    )["verdict"]
