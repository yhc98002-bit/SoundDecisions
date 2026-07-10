"""Tests for foley_cw.reporting.

All tests are numpy-only and run entirely on CPU.  They exercise:
  - write_commitment_map_csv: round-trip via csv.DictReader
  - write_readout_map_csv:    round-trip via csv.DictReader
  - write_markdown:           generic title + sections
  - feasibility_report:       writes a .md file
  - score_sde_validation_report: writes a .md file
  - dataset_subset_manifest:  writes a .md file
  - event_anchor_validation_report: writes a .md file
  - axis_reliability_report:  writes a .md file
  - commitment_readout_gap_report: writes a .md file
  - go_no_go_decision:        writes a .md file

Parent directory creation is implicitly tested (all writes go into a tmp dir).
"""

from __future__ import annotations

import csv
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from foley_cw.types import (
    CommitmentCell,
    GoNoGoDecision,
    ReadoutCell,
    ReliabilityResult,
    Thresholds,
    ValidationResult,
    WindowEstimate,
)
from foley_cw.reporting import (
    axis_reliability_report,
    commitment_readout_gap_report,
    dataset_subset_manifest,
    event_anchor_validation_report,
    feasibility_report,
    go_no_go_decision,
    score_sde_validation_report,
    write_commitment_map_csv,
    write_markdown,
    write_readout_map_csv,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_commit_cells() -> list[CommitmentCell]:
    return [
        CommitmentCell(
            axis_id="presence",
            s=0.0,
            alpha=0.1,
            a_fork=0.5,
            a_independent=0.4,
            commit_gain=0.1667,
            n_videos=10,
        ),
        CommitmentCell(
            axis_id="timing",
            s=0.5,
            alpha=0.1,
            a_fork=0.8,
            a_independent=0.4,
            commit_gain=0.6667,
            n_videos=10,
        ),
    ]


def _make_readout_cells() -> list[ReadoutCell]:
    return [
        ReadoutCell(
            axis_id="presence",
            probe="EnergyOnsetProbe",
            s=0.3,
            target="ode",
            score=0.72,
            n_videos=10,
        ),
        ReadoutCell(
            axis_id="timing",
            probe="EnergyOnsetProbe",
            s=0.6,
            target="fork_majority",
            score=0.65,
            n_videos=10,
        ),
    ]


def _make_thresholds() -> Thresholds:
    return Thresholds(
        theta_commit=0.7,
        theta_read=0.7,
        theta_rel=0.9,
        theta_robust=0.8,
        theta_cal=0.7,
        frozen=False,
        frozen_from=None,
    )


# ---------------------------------------------------------------------------
# CSV round-trip tests
# ---------------------------------------------------------------------------

class TestWriteCommitmentMapCsv:
    def test_header_and_row_count(self, tmp_path: Path) -> None:
        cells = _make_commit_cells()
        out = tmp_path / "commitment_map.csv"
        write_commitment_map_csv(cells, out)

        assert out.exists(), "commitment_map.csv was not created"
        with out.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)

        assert len(rows) == len(cells), (
            f"Expected {len(cells)} data rows, got {len(rows)}"
        )

    def test_header_matches_fields(self, tmp_path: Path) -> None:
        cells = _make_commit_cells()
        out = tmp_path / "commitment_map.csv"
        write_commitment_map_csv(cells, out)

        with out.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            headers = reader.fieldnames

        expected = ["axis_id", "s", "alpha", "a_fork", "a_independent",
                    "commit_gain", "n_videos"]
        assert headers == expected, f"Header mismatch: {headers}"

    def test_values_round_trip(self, tmp_path: Path) -> None:
        cells = _make_commit_cells()
        out = tmp_path / "commitment_map.csv"
        write_commitment_map_csv(cells, out)

        with out.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)

        # Check axis_id strings survive
        assert rows[0]["axis_id"] == "presence"
        assert rows[1]["axis_id"] == "timing"

        # Check float values survive with reasonable precision
        assert abs(float(rows[0]["s"]) - 0.0) < 1e-9
        assert abs(float(rows[1]["s"]) - 0.5) < 1e-9
        assert int(rows[0]["n_videos"]) == 10

    def test_empty_cells_writes_header_only(self, tmp_path: Path) -> None:
        out = tmp_path / "empty_commit.csv"
        write_commitment_map_csv([], out)
        assert out.exists()
        with out.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        assert rows == []

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        out = tmp_path / "deep" / "nested" / "commitment_map.csv"
        write_commitment_map_csv(_make_commit_cells(), out)
        assert out.exists()

    def test_nan_value_encoded(self, tmp_path: Path) -> None:
        cells = [
            CommitmentCell(
                axis_id="presence", s=float("nan"), alpha=0.1,
                a_fork=float("nan"), a_independent=0.4,
                commit_gain=0.0, n_videos=0,
            )
        ]
        out = tmp_path / "nan_commit.csv"
        write_commitment_map_csv(cells, out)
        with out.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        assert rows[0]["s"] == "nan"
        assert math.isnan(float(rows[0]["a_fork"]))


class TestWriteReadoutMapCsv:
    def test_header_and_row_count(self, tmp_path: Path) -> None:
        cells = _make_readout_cells()
        out = tmp_path / "readout_map.csv"
        write_readout_map_csv(cells, out)

        assert out.exists()
        with out.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)

        assert len(rows) == len(cells)

    def test_header_matches_fields(self, tmp_path: Path) -> None:
        cells = _make_readout_cells()
        out = tmp_path / "readout_map.csv"
        write_readout_map_csv(cells, out)

        with out.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            headers = reader.fieldnames

        expected = ["axis_id", "probe", "s", "target", "score", "n_videos"]
        assert headers == expected, f"Header mismatch: {headers}"

    def test_values_round_trip(self, tmp_path: Path) -> None:
        cells = _make_readout_cells()
        out = tmp_path / "readout_map.csv"
        write_readout_map_csv(cells, out)

        with out.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)

        assert rows[0]["axis_id"] == "presence"
        assert rows[0]["probe"] == "EnergyOnsetProbe"
        assert rows[0]["target"] == "ode"
        assert abs(float(rows[0]["score"]) - 0.72) < 1e-5

    def test_empty_cells_writes_header_only(self, tmp_path: Path) -> None:
        out = tmp_path / "empty_readout.csv"
        write_readout_map_csv([], out)
        assert out.exists()
        with out.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        assert rows == []

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        out = tmp_path / "deep" / "nested" / "readout_map.csv"
        write_readout_map_csv(_make_readout_cells(), out)
        assert out.exists()


# ---------------------------------------------------------------------------
# write_markdown
# ---------------------------------------------------------------------------

class TestWriteMarkdown:
    def test_creates_file(self, tmp_path: Path) -> None:
        out = tmp_path / "test.md"
        write_markdown(out, "My Title", [("Intro", "Hello world.")])
        assert out.exists()

    def test_title_present(self, tmp_path: Path) -> None:
        out = tmp_path / "test.md"
        write_markdown(out, "My Title", [])
        content = out.read_text(encoding="utf-8")
        assert "# My Title" in content

    def test_sections_present(self, tmp_path: Path) -> None:
        out = tmp_path / "test.md"
        write_markdown(
            out, "T",
            [("Section A", "Body A"), ("Section B", "Body B")]
        )
        content = out.read_text(encoding="utf-8")
        assert "## Section A" in content
        assert "Body A" in content
        assert "## Section B" in content
        assert "Body B" in content

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        out = tmp_path / "sub" / "dir" / "report.md"
        write_markdown(out, "T", [])
        assert out.exists()

    def test_returns_path(self, tmp_path: Path) -> None:
        out = tmp_path / "ret.md"
        result = write_markdown(out, "T", [("A", "B")])
        assert result == out


# ---------------------------------------------------------------------------
# Per-report renderer tests
# ---------------------------------------------------------------------------

class TestFeasibilityReport:
    def test_writes_file(self, tmp_path: Path) -> None:
        out = tmp_path / "feasibility.md"
        feasibility_report(
            trajectory_ok=True,
            x_s_shape=(4,),
            resume_ok=True,
            x0_shape=(4,),
            s_to_t_name="identity_ascending",
            s_to_t_verified=True,
            path=out,
        )
        assert out.exists()

    def test_contains_status(self, tmp_path: Path) -> None:
        out = tmp_path / "feasibility.md"
        feasibility_report(
            trajectory_ok=True,
            x_s_shape=(4,),
            resume_ok=True,
            x0_shape=(4,),
            s_to_t_name="identity_ascending",
            s_to_t_verified=True,
            path=out,
        )
        content = out.read_text(encoding="utf-8")
        assert "PASS" in content

    def test_fail_status_on_false(self, tmp_path: Path) -> None:
        out = tmp_path / "feasibility_fail.md"
        feasibility_report(
            trajectory_ok=False,
            x_s_shape=None,
            resume_ok=False,
            x0_shape=None,
            s_to_t_name="unknown",
            s_to_t_verified=False,
            path=out,
        )
        content = out.read_text(encoding="utf-8")
        assert "FAIL" in content


class TestScoreSdeValidationReport:
    def test_writes_file(self, tmp_path: Path) -> None:
        vrs = [
            ValidationResult(
                name="alpha0_reproduces_ode",
                passed=True,
                value=1e-9,
                threshold=1e-6,
                detail="max diff",
            ),
        ]
        out = tmp_path / "sde_val.md"
        score_sde_validation_report(
            validation_results=vrs,
            token="OK",
            alpha_tested=0.1,
            path=out,
        )
        assert out.exists()
        content = out.read_text(encoding="utf-8")
        assert "alpha0_reproduces_ode" in content
        assert "OK" in content


class TestDatasetSubsetManifest:
    def test_writes_file(self, tmp_path: Path) -> None:
        out = tmp_path / "manifest.md"
        dataset_subset_manifest(
            {"n_videos": 200, "n_axes": 4, "source": "FoleyBench"},
            path=out,
        )
        assert out.exists()
        content = out.read_text(encoding="utf-8")
        assert "n_videos" in content


class TestEventAnchorValidationReport:
    def test_writes_file(self, tmp_path: Path) -> None:
        rows = [
            {"video_id": "vid001", "source": "metadata", "uncertainty": "50ms"},
            {"video_id": "vid002", "source": "metadata", "uncertainty": "30ms"},
        ]
        out = tmp_path / "anchors.md"
        event_anchor_validation_report(anchor_rows=rows, coverage=0.95, path=out)
        assert out.exists()
        content = out.read_text(encoding="utf-8")
        assert "vid001" in content
        assert "95" in content  # coverage %

    def test_empty_rows(self, tmp_path: Path) -> None:
        out = tmp_path / "anchors_empty.md"
        event_anchor_validation_report(anchor_rows=[], coverage=0.0, path=out)
        assert out.exists()
        content = out.read_text(encoding="utf-8")
        assert "no anchor rows" in content.lower() or "(no anchors)" in content


class TestAxisReliabilityReport:
    def test_writes_file(self, tmp_path: Path) -> None:
        rrs = [
            ReliabilityResult(
                axis_id="presence",
                determinism=0.99,
                robustness=0.92,
                validity=0.85,
                passed=True,
                demoted=False,
                reason="",
            ),
            ReliabilityResult(
                axis_id="material",
                determinism=0.85,
                robustness=0.70,
                validity=0.60,
                passed=False,
                demoted=True,
                reason="validity below theta_cal",
            ),
        ]
        out = tmp_path / "reliability.md"
        axis_reliability_report(
            reliability_results=rrs,
            thresholds=_make_thresholds(),
            path=out,
        )
        assert out.exists()
        content = out.read_text(encoding="utf-8")
        assert "presence" in content
        assert "material" in content
        assert "theta_rel" in content

    def test_demotion_reason_in_report(self, tmp_path: Path) -> None:
        rrs = [
            ReliabilityResult(
                axis_id="class",
                determinism=0.88,
                robustness=0.75,
                validity=0.60,
                passed=False,
                demoted=True,
                reason="validity below theta_cal",
            )
        ]
        out = tmp_path / "rel2.md"
        axis_reliability_report(rrs, _make_thresholds(), path=out)
        content = out.read_text(encoding="utf-8")
        assert "validity below theta_cal" in content


class TestCommitmentReadoutGapReport:
    def _windows(self) -> tuple[dict, dict]:
        commit = {
            "presence": WindowEstimate(
                axis_id="presence", kind="commit",
                s_hat=0.3, ci_low=0.2, ci_high=0.4,
                n_videos=10, underpowered=False,
            ),
        }
        read = {
            "presence_EnergyOnsetProbe_ode": WindowEstimate(
                axis_id="presence", kind="read",
                s_hat=0.5, ci_low=0.4, ci_high=0.6,
                n_videos=10, underpowered=False,
            ),
        }
        return commit, read

    def test_writes_file(self, tmp_path: Path) -> None:
        commit, read = self._windows()
        out = tmp_path / "gap.md"
        commitment_readout_gap_report(commit, read, path=out)
        assert out.exists()

    def test_gap_computed(self, tmp_path: Path) -> None:
        commit, read = self._windows()
        out = tmp_path / "gap2.md"
        commitment_readout_gap_report(commit, read, path=out)
        content = out.read_text(encoding="utf-8")
        # s_read 0.5 - s_commit 0.3 = 0.2
        assert "0.200000" in content

    def test_separation_score_included(self, tmp_path: Path) -> None:
        commit, read = self._windows()
        out = tmp_path / "gap3.md"
        commitment_readout_gap_report(
            commit, read,
            separation_score_val=1.5,
            ordered_non_overlapping=True,
            path=out,
        )
        content = out.read_text(encoding="utf-8")
        assert "1.500000" in content
        assert "True" in content


class TestGoNoGoDecision:
    def test_writes_file(self, tmp_path: Path) -> None:
        decision = GoNoGoDecision(
            tokens=["GO_MAPS_PHASE"],
            justification="All Phase 0 checks passed.",
            thresholds=_make_thresholds(),
            extra={"n_reliable_axes": 3},
        )
        out = tmp_path / "go_no_go.md"
        go_no_go_decision(decision, path=out)
        assert out.exists()

    def test_tokens_in_report(self, tmp_path: Path) -> None:
        decision = GoNoGoDecision(
            tokens=["GO_MAP", "GO_READOUT"],
            justification="Axes separate and probes readable.",
            thresholds=None,
        )
        out = tmp_path / "go_no_go2.md"
        go_no_go_decision(decision, path=out)
        content = out.read_text(encoding="utf-8")
        assert "GO_MAP" in content
        assert "GO_READOUT" in content
        assert "Axes separate" in content

    def test_thresholds_in_report(self, tmp_path: Path) -> None:
        decision = GoNoGoDecision(
            tokens=["GO_MAPS_PHASE"],
            justification="Passed.",
            thresholds=_make_thresholds(),
        )
        out = tmp_path / "go_no_go3.md"
        go_no_go_decision(decision, path=out)
        content = out.read_text(encoding="utf-8")
        assert "theta_commit" in content
        assert "0.7" in content

    def test_extra_included(self, tmp_path: Path) -> None:
        decision = GoNoGoDecision(
            tokens=["STOP_ADSR"],
            justification="Windows coincide near s=1.",
            thresholds=None,
            extra={"separation_score": 0.05, "note": "degenerate"},
        )
        out = tmp_path / "go_no_go4.md"
        go_no_go_decision(decision, path=out)
        content = out.read_text(encoding="utf-8")
        assert "separation_score" in content


# ---------------------------------------------------------------------------
# Integration: both CSVs in the same temp dir
# ---------------------------------------------------------------------------

class TestCsvIntegration:
    """Verify both CSVs can be written to the same directory and read back."""

    def test_commitment_and_readout_in_same_dir(self, tmp_path: Path) -> None:
        commit_cells = _make_commit_cells()
        readout_cells = _make_readout_cells()

        commit_path = tmp_path / "maps" / "commitment_map.csv"
        readout_path = tmp_path / "maps" / "readout_map.csv"

        write_commitment_map_csv(commit_cells, commit_path)
        write_readout_map_csv(readout_cells, readout_path)

        # Read back and verify
        with commit_path.open(newline="", encoding="utf-8") as fh:
            cr = list(csv.DictReader(fh))
        with readout_path.open(newline="", encoding="utf-8") as fh:
            rr = list(csv.DictReader(fh))

        assert len(cr) == 2
        assert len(rr) == 2

        # Spot-check values survive
        assert cr[0]["axis_id"] == "presence"
        assert cr[1]["axis_id"] == "timing"
        assert rr[0]["target"] == "ode"
        assert rr[1]["target"] == "fork_majority"

    def test_overwrite_preserves_last_write(self, tmp_path: Path) -> None:
        path = tmp_path / "commitment_map.csv"
        cells_a = _make_commit_cells()[:1]
        cells_b = _make_commit_cells()

        write_commitment_map_csv(cells_a, path)
        write_commitment_map_csv(cells_b, path)

        with path.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 2  # second write wins
