"""Reporting writers for foley_cw.

Writes CSV (stdlib csv) and Markdown output for Phase 0/1/2/3 artefacts.
No external dependencies beyond stdlib + numpy (which is already a hard dep for types.py).

Output files are placed under results/ by the renderer helpers; callers may
pass any path. Parent directories are created automatically.

CSV columns match the dataclass field order exactly so downstream tools can
parse them without a schema document.

Renderer helpers take structured inputs (cells, windows, decisions, etc.) and
write a named report file under results/.  Each renderer returns the Path it
wrote so callers can chain or log it.
"""

from __future__ import annotations

import csv
import dataclasses
import math
from pathlib import Path
from typing import Any, Optional

# Absolute path for results root — resolved relative to this file's package root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_RESULTS_DIR = _REPO_ROOT / "results"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_parent(path: Path) -> None:
    """Create all parent directories for *path* if they do not already exist."""
    path.parent.mkdir(parents=True, exist_ok=True)


def _float_fmt(v: float, precision: int = 6) -> str:
    """Stable, readable float representation (NaN/inf safe)."""
    if math.isnan(v):
        return "nan"
    if math.isinf(v):
        return "inf" if v > 0 else "-inf"
    return f"{v:.{precision}f}"


def _cell_to_row(cell: Any) -> dict[str, str]:
    """Convert a dataclass to an ordered {field: str} dict for csv.DictWriter."""
    row: dict[str, str] = {}
    for f in dataclasses.fields(cell):
        val = getattr(cell, f.name)
        if isinstance(val, float):
            row[f.name] = _float_fmt(val)
        else:
            row[f.name] = str(val)
    return row


# ---------------------------------------------------------------------------
# Core CSV writers
# ---------------------------------------------------------------------------

_COMMIT_WINDOW_COLS = ["s_commit", "s_commit_ci_low", "s_commit_ci_high", "s_commit_underpowered"]
_READ_WINDOW_COLS = ["s_read", "s_read_ci_low", "s_read_ci_high", "s_read_underpowered"]


def _window_cols(win: Any, cols: list[str]) -> dict[str, str]:
    """Render a WindowEstimate (s_hat + CI + underpowered flag) onto the given columns."""
    if win is None:
        return {c: "" for c in cols}
    return {
        cols[0]: _float_fmt(win.s_hat),
        cols[1]: _float_fmt(win.ci_low),
        cols[2]: _float_fmt(win.ci_high),
        cols[3]: str(bool(win.underpowered)),
    }


def write_commitment_map_csv(cells: list, path: "Path | str",
                             windows: "dict | None" = None) -> Path:
    """Write a list of CommitmentCell to a CSV file.

    Surface columns (field order): axis_id, s, alpha, a_fork, a_independent,
    commit_gain, n_videos. When *windows* (axis_id -> WindowEstimate) is supplied,
    the per-axis s_commit and its bootstrap-over-videos CI are appended as columns
    (s_commit, s_commit_ci_low, s_commit_ci_high, s_commit_underpowered), as the plan's
    commitment_map.csv requires ("A_fork, A_independent, normalized gain, s_commit with CIs").

    Creates parent directories as needed.
    """
    path = Path(path)
    _ensure_parent(path)
    base = ["axis_id", "s", "alpha", "a_fork", "a_independent", "commit_gain", "n_videos"]
    fieldnames = ([f.name for f in dataclasses.fields(cells[0])] if cells else base)
    if windows is not None:
        fieldnames = fieldnames + _COMMIT_WINDOW_COLS
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for cell in cells:
            row = _cell_to_row(cell)
            if windows is not None:
                row.update(_window_cols(windows.get(cell.axis_id), _COMMIT_WINDOW_COLS))
            writer.writerow(row)
    return path


def write_readout_map_csv(cells: list, path: "Path | str",
                          windows: "dict | None" = None) -> Path:
    """Write a list of ReadoutCell to a CSV file.

    Surface columns (field order): axis_id, probe, s, target, score, n_videos. When
    *windows* ((axis_id, probe, target) -> WindowEstimate) is supplied, the per-(axis,
    probe, target) s_read and its bootstrap-over-videos CI are appended as columns
    (s_read, s_read_ci_low, s_read_ci_high, s_read_underpowered), as the plan's
    readout_map.csv requires ("s_read with CIs").

    Creates parent directories as needed.
    """
    path = Path(path)
    _ensure_parent(path)
    base = ["axis_id", "probe", "s", "target", "score", "n_videos"]
    fieldnames = ([f.name for f in dataclasses.fields(cells[0])] if cells else base)
    if windows is not None:
        fieldnames = fieldnames + _READ_WINDOW_COLS
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for cell in cells:
            row = _cell_to_row(cell)
            if windows is not None:
                row.update(_window_cols(windows.get((cell.axis_id, cell.probe, cell.target)),
                                        _READ_WINDOW_COLS))
            writer.writerow(row)
    return path


# ---------------------------------------------------------------------------
# Generic Markdown writer
# ---------------------------------------------------------------------------

def write_markdown(path: "Path | str", title: str,
                   sections: list[tuple[str, str]]) -> Path:
    """Write a Markdown file with a title and a list of (heading, body) sections.

    Each section is rendered as:
        ## <heading>
        <body>

    Creates parent directories as needed.  Returns the written Path.
    """
    path = Path(path)
    _ensure_parent(path)
    lines: list[str] = [f"# {title}", ""]
    for heading, body in sections:
        lines.append(f"## {heading}")
        lines.append("")
        lines.append(body.rstrip())
        lines.append("")
    with path.open("w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
        if not lines[-1]:
            pass  # already ends with blank
        fh.write("\n")
    return path


# ---------------------------------------------------------------------------
# Per-report renderer helpers
# Each helper writes one named report file under results/ (or caller-supplied
# path) and returns the Path it wrote.
# ---------------------------------------------------------------------------

def feasibility_report(
    trajectory_ok: bool,
    x_s_shape: Optional[tuple],
    resume_ok: bool,
    x0_shape: Optional[tuple],
    s_to_t_name: str,
    s_to_t_verified: bool,
    notes: str = "",
    path: Optional["Path | str"] = None,
) -> Path:
    """Write results/feasibility_report.md.

    Covers Phase 0.1: trajectory access (extract x_s, resume from x_s, compute
    x0(s)) and the s<->t mapping audit.
    """
    out = Path(path) if path is not None else _RESULTS_DIR / "feasibility_report.md"
    status = "PASS" if (trajectory_ok and resume_ok and x0_shape is not None) else "FAIL"
    sections = [
        ("Status", status),
        (
            "Trajectory Access",
            "\n".join([
                f"- trajectory_ok: {trajectory_ok}",
                f"- x_s shape: {x_s_shape}",
                f"- resume_ok: {resume_ok}",
                f"- x0(s) shape: {x0_shape}",
            ]),
        ),
        (
            "s<->t Mapping",
            "\n".join([
                f"- mapping name: {s_to_t_name}",
                f"- verified: {s_to_t_verified}",
            ]),
        ),
    ]
    if notes:
        sections.append(("Notes", notes))
    return write_markdown(out, "Feasibility Report (Phase 0.1)", sections)


def score_sde_validation_report(
    validation_results: list,
    token: str,
    alpha_tested: float,
    notes: str = "",
    path: Optional["Path | str"] = None,
) -> Path:
    """Write results/score_sde_validation_report.md.

    Covers Phase 0.2: velocity->score SDE validation (alpha=0 + nonzero-alpha
    checks).  validation_results is a list of ValidationResult dataclass instances.
    """
    out = Path(path) if path is not None else _RESULTS_DIR / "score_sde_validation_report.md"

    rows: list[str] = [
        "| check | passed | value | threshold | detail |",
        "| --- | --- | --- | --- | --- |",
    ]
    for vr in validation_results:
        rows.append(
            f"| {vr.name} | {vr.passed} | {_float_fmt(vr.value)} "
            f"| {_float_fmt(vr.threshold)} | {vr.detail} |"
        )
    table = "\n".join(rows)

    sections = [
        ("Token", token),
        (f"alpha tested: {_float_fmt(alpha_tested)}", ""),
        ("Validation Checks", table),
    ]
    if notes:
        sections.append(("Notes", notes))
    return write_markdown(out, "Score-SDE Validation Report (Phase 0.2)", sections)


def dataset_subset_manifest(
    manifest_dict: dict[str, Any],
    path: Optional["Path | str"] = None,
) -> Path:
    """Write results/dataset_subset_manifest.md.

    manifest_dict should contain the keys produced by dataset.build_manifest /
    manifest_to_markdown; this renderer just formats them.
    """
    out = Path(path) if path is not None else _RESULTS_DIR / "dataset_subset_manifest.md"
    rows: list[str] = []
    for k, v in manifest_dict.items():
        rows.append(f"- **{k}**: {v}")
    body = "\n".join(rows) if rows else "_empty manifest_"
    sections = [("Dataset Manifest", body)]
    return write_markdown(out, "Dataset Subset Manifest (Phase 0.3)", sections)


def event_anchor_validation_report(
    anchor_rows: list[dict[str, Any]],
    coverage: float,
    notes: str = "",
    path: Optional["Path | str"] = None,
) -> Path:
    """Write results/event_anchor_validation_report.md.

    anchor_rows: list of dicts with keys such as video_id, source, uncertainty,
    check_error.
    """
    out = (
        Path(path)
        if path is not None
        else _RESULTS_DIR / "event_anchor_validation_report.md"
    )
    header_keys = list(anchor_rows[0].keys()) if anchor_rows else ["(no anchors)"]
    if anchor_rows:
        header = "| " + " | ".join(header_keys) + " |"
        sep = "| " + " | ".join("---" for _ in header_keys) + " |"
        body_rows = [
            "| " + " | ".join(str(r.get(k, "")) for k in header_keys) + " |"
            for r in anchor_rows
        ]
        table = "\n".join([header, sep] + body_rows)
    else:
        table = "_no anchor rows provided_"

    sections = [
        ("Coverage", f"{_float_fmt(coverage * 100, 1)}%"),
        ("Anchor Table", table),
    ]
    if notes:
        sections.append(("Notes", notes))
    return write_markdown(out, "Event Anchor Validation Report (Phase 0.4)", sections)


def axis_reliability_report(
    reliability_results: list,
    thresholds,
    path: Optional["Path | str"] = None,
) -> Path:
    """Write results/axis_reliability_report.md.

    reliability_results is a list of ReliabilityResult instances.
    thresholds is a Thresholds instance.
    """
    out = (
        Path(path)
        if path is not None
        else _RESULTS_DIR / "axis_reliability_report.md"
    )
    header = "| axis | determinism | robustness | validity | passed | demoted | reason |"
    sep = "| --- | --- | --- | --- | --- | --- | --- |"
    rows = [header, sep]
    for rr in reliability_results:
        rows.append(
            f"| {rr.axis_id} "
            f"| {_float_fmt(rr.determinism)} "
            f"| {_float_fmt(rr.robustness)} "
            f"| {_float_fmt(rr.validity)} "
            f"| {rr.passed} "
            f"| {rr.demoted} "
            f"| {rr.reason} |"
        )
    table = "\n".join(rows)

    thresh_lines = [
        f"- theta_rel (determinism): {thresholds.theta_rel}",
        f"- theta_robust: {thresholds.theta_robust}",
        f"- theta_cal (validity): {thresholds.theta_cal}",
        f"- frozen: {thresholds.frozen}",
        f"- frozen_from: {thresholds.frozen_from}",
    ]
    sections = [
        ("Pre-registered Thresholds", "\n".join(thresh_lines)),
        ("Per-Axis Reliability Gate", table),
    ]
    return write_markdown(out, "Axis Reliability Report (Phase 0.5)", sections)


def commitment_readout_gap_report(
    commit_windows: dict[str, Any],
    read_windows: dict,
    crosstab: Optional[dict[str, Any]] = None,
    separation_score_val: Optional[float] = None,
    ordered_non_overlapping: Optional[bool] = None,
    threshold_sweep: Optional[dict] = None,
    gap_cis: Optional[dict] = None,
    separation_sensitivity: Optional[dict] = None,
    notes: str = "",
    path: Optional["Path | str"] = None,
) -> Path:
    """Write results/commitment_readout_gap_report.md.

    commit_windows: dict axis_id -> WindowEstimate
    read_windows:   dict (axis_id, probe, target) -> WindowEstimate (keys as str or tuple)
    crosstab:       optional dict from gap.r1_r2_crosstab
    separation_score_val: optional float from stats.separation_score
    ordered_non_overlapping: optional bool from stats.ordered_non_overlapping
    threshold_sweep: optional dict theta -> WindowEstimate
    """
    out = (
        Path(path)
        if path is not None
        else _RESULTS_DIR / "commitment_readout_gap_report.md"
    )

    # Commitment windows table
    commit_header = "| axis | kind | s_hat | ci_low | ci_high | n_videos | underpowered |"
    commit_sep = "| --- | --- | --- | --- | --- | --- | --- |"
    commit_rows = [commit_header, commit_sep]
    for axis_id, w in commit_windows.items():
        commit_rows.append(
            f"| {w.axis_id} | {w.kind} "
            f"| {_float_fmt(w.s_hat)} "
            f"| {_float_fmt(w.ci_low)} "
            f"| {_float_fmt(w.ci_high)} "
            f"| {w.n_videos} "
            f"| {w.underpowered} |"
        )
    commit_table = "\n".join(commit_rows)

    # Readout windows table
    read_header = "| key | axis | kind | s_hat | ci_low | ci_high | n_videos | underpowered |"
    read_sep = "| --- | --- | --- | --- | --- | --- | --- | --- |"
    read_rows = [read_header, read_sep]
    for key, w in read_windows.items():
        read_rows.append(
            f"| {key} | {w.axis_id} | {w.kind} "
            f"| {_float_fmt(w.s_hat)} "
            f"| {_float_fmt(w.ci_low)} "
            f"| {_float_fmt(w.ci_high)} "
            f"| {w.n_videos} "
            f"| {w.underpowered} |"
        )
    read_table = "\n".join(read_rows)

    # Gap table (s_read - s_commit per (axis, probe, target) key), with bootstrap-over-video
    # CIs when provided (plan §3: bootstrap CIs on gaps, not just point estimates).
    gap_lines: list[str] = []
    excluded = 0
    for key, rw in read_windows.items():
        cw = commit_windows.get(rw.axis_id)
        if cw is None:
            gap_lines.append(f"- {key}: commit window missing for axis {rw.axis_id}")
            continue
        # Underpowered or undefined (no-crossing) windows are NOT results (plan §3); they are
        # shown in the window tables (with the underpowered flag) but excluded from the gap
        # evidence so a NaN/underpowered gap is never presented as a result.
        if rw.underpowered or cw.underpowered or math.isnan(rw.s_hat) or math.isnan(cw.s_hat):
            excluded += 1
            continue
        ci_entry = gap_cis.get(key) if gap_cis else None
        if ci_entry is not None:
            point, lo, hi, n_boot = ci_entry
            gap_lines.append(
                f"- {key}: gap = {_float_fmt(point)} "
                f"(95% CI [{_float_fmt(lo)}, {_float_fmt(hi)}], bootstrap-over-videos n={n_boot})"
            )
        else:
            gap_lines.append(f"- {key}: gap = {_float_fmt(rw.s_hat - cw.s_hat)} (point estimate)")
    if excluded:
        gap_lines.append(
            f"- _{excluded} (axis, probe, target) pair(s) excluded as underpowered / "
            "no crossing — not results_"
        )
    gap_body = "\n".join(gap_lines) if gap_lines else "_no result gap entries_"

    sections: list[tuple[str, str]] = [
        ("Commitment Windows", commit_table),
        ("Readout Windows", read_table),
        ("Gap (s_read - s_commit)", gap_body),
    ]

    if separation_score_val is not None:
        sep_body = (
            f"separation_score = {_float_fmt(separation_score_val)}\n"
            f"ordered_non_overlapping = {ordered_non_overlapping}"
        )
        sections.append(("Separation Analysis", sep_body))

    if separation_sensitivity is not None:
        ss_header = "| theta_commit | separation_score | ordered_non_overlapping |"
        ss_sep = "| --- | --- | --- |"
        ss_rows = [ss_header, ss_sep]
        for theta in sorted(separation_sensitivity.keys()):
            entry = separation_sensitivity[theta]
            ss_rows.append(
                f"| {_float_fmt(float(theta))} "
                f"| {_float_fmt(entry.get('separation', float('nan')))} "
                f"| {entry.get('ordered_non_overlapping')} |"
            )
        sections.append((
            "Threshold Sensitivity — Separation under theta_commit sweep",
            "\n".join(ss_rows),
        ))

    if crosstab is not None:
        ct_lines = [f"- {k}: {v}" for k, v in crosstab.items()]
        sections.append(("R1/R2 Cross-tab", "\n".join(ct_lines)))

    if threshold_sweep is not None:
        ts_lines = [f"- theta={k}: s_hat={_float_fmt(v.s_hat)}" for k, v in threshold_sweep.items()]
        sections.append(("Threshold Sensitivity — first-axis s_commit", "\n".join(ts_lines)))

    if notes:
        sections.append(("Notes", notes))

    return write_markdown(out, "Commitment/Readout Gap Report (Phase 3)", sections)


def go_no_go_decision(
    decision,
    path: Optional["Path | str"] = None,
) -> Path:
    """Write results/go_no_go_decision.md.

    decision is a GoNoGoDecision instance.
    """
    out = (
        Path(path)
        if path is not None
        else _RESULTS_DIR / "go_no_go_decision.md"
    )
    tokens_str = ", ".join(decision.tokens)
    sections: list[tuple[str, str]] = [
        ("Emitted Tokens", tokens_str),
        ("Justification", decision.justification),
    ]
    if decision.thresholds is not None:
        th = decision.thresholds
        thresh_lines = [
            f"- theta_commit: {th.theta_commit}",
            f"- theta_read: {th.theta_read}",
            f"- theta_rel: {th.theta_rel}",
            f"- theta_robust: {th.theta_robust}",
            f"- theta_cal: {th.theta_cal}",
            f"- frozen: {th.frozen}",
            f"- frozen_from: {th.frozen_from}",
        ]
        sections.append(("Pre-registered Thresholds", "\n".join(thresh_lines)))
    if decision.extra:
        extra_lines = [f"- {k}: {v}" for k, v in decision.extra.items()]
        sections.append(("Extra", "\n".join(extra_lines)))
    return write_markdown(out, "Go/No-Go Decision", sections)
