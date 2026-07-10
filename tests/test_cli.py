"""Tests for foley_cw/cli/phase0_feasibility.py and foley_cw/cli/phases123_maps.py.

Runs the full synthetic dry-runs end-to-end under /tmp so the repo results/ dir
is not touched.  All tests use numpy-only core (SyntheticGaussianFlow +
SyntheticMeasurer).

Run with:
    pytest tests/test_cli.py -q
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tmp_out() -> Path:
    """Return a fresh temporary directory for output files."""
    d = Path(tempfile.mkdtemp(prefix="fcw_cli_test_"))
    return d


# ---------------------------------------------------------------------------
# phase0_feasibility tests
# ---------------------------------------------------------------------------


class TestPhase0Feasibility:

    def test_imports(self):
        """Module must be importable with numpy only."""
        from foley_cw.cli import phase0_feasibility  # noqa: F401

    def test_main_synthetic_exits_zero(self):
        """main() with --synthetic should exit 0."""
        from foley_cw.cli.phase0_feasibility import main
        out = _tmp_out()
        rc = main(["--synthetic", "--out", str(out), "--n-videos", "3", "--seed", "42"])
        assert rc == 0, f"main returned {rc}"

    def test_main_no_synthetic_exits_nonzero(self):
        """main() with --no-synthetic should exit non-zero without touching out dir."""
        from foley_cw.cli.phase0_feasibility import main
        out = _tmp_out()
        rc = main(["--no-synthetic", "--out", str(out), "--n-videos", "3"])
        assert rc != 0, "Expected non-zero exit for --no-synthetic"

    def test_output_files_written(self):
        """All five Phase-0 report files must be written."""
        from foley_cw.cli.phase0_feasibility import main
        out = _tmp_out()
        rc = main(["--synthetic", "--out", str(out), "--n-videos", "3", "--seed", "7"])
        assert rc == 0
        expected = [
            "feasibility_report.md",
            "score_sde_validation_report.md",
            "dataset_subset_manifest.md",
            "event_anchor_validation_report.md",
            "axis_reliability_report.md",
        ]
        for fname in expected:
            p = out / fname
            assert p.exists(), f"Expected output file missing: {fname}"
            assert p.stat().st_size > 0, f"Output file is empty: {fname}"

    def test_tokens_are_valid(self):
        """run_phase0_synthetic must return a non-empty list of known token strings."""
        from foley_cw.cli.phase0_feasibility import run_phase0_synthetic
        out = _tmp_out()
        tokens = run_phase0_synthetic(
            out_dir=out, n_videos=3, seed=0, config_dir=None, fast=True
        )
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        valid_tokens = {
            "GO_MAPS_PHASE", "FIX_SCORE_CONVERSION",
            "NO_TRAJECTORY_ACCESS", "STOP_PROJECT",
        }
        for tok in tokens:
            assert isinstance(tok, str)
            assert tok in valid_tokens, f"Unexpected token: {tok!r}"

    def test_go_maps_phase_on_synthetic(self):
        """Synthetic backend (analytic) should emit GO_MAPS_PHASE (all checks pass)."""
        from foley_cw.cli.phase0_feasibility import run_phase0_synthetic
        out = _tmp_out()
        tokens = run_phase0_synthetic(
            out_dir=out, n_videos=4, seed=0, config_dir=None, fast=True
        )
        assert "GO_MAPS_PHASE" in tokens, (
            f"Expected GO_MAPS_PHASE for synthetic backend; got {tokens}"
        )

    def test_feasibility_report_content(self):
        """feasibility_report.md must mention PASS and the trajectory shape."""
        from foley_cw.cli.phase0_feasibility import run_phase0_synthetic
        out = _tmp_out()
        run_phase0_synthetic(out_dir=out, n_videos=3, seed=0, config_dir=None, fast=True)
        text = (out / "feasibility_report.md").read_text(encoding="utf-8")
        assert "PASS" in text
        assert "trajectory" in text.lower() or "s_to_t" in text.lower()

    def test_sde_validation_report_content(self):
        """score_sde_validation_report.md must mention OK token and check names."""
        from foley_cw.cli.phase0_feasibility import run_phase0_synthetic
        out = _tmp_out()
        run_phase0_synthetic(out_dir=out, n_videos=3, seed=0, config_dir=None, fast=True)
        text = (out / "score_sde_validation_report.md").read_text(encoding="utf-8")
        assert "OK" in text
        assert "alpha0_reproduces_ode" in text

    def test_axis_reliability_report_content(self):
        """axis_reliability_report.md must mention presence and timing axes."""
        from foley_cw.cli.phase0_feasibility import run_phase0_synthetic
        out = _tmp_out()
        run_phase0_synthetic(out_dir=out, n_videos=3, seed=0, config_dir=None, fast=True)
        text = (out / "axis_reliability_report.md").read_text(encoding="utf-8")
        assert "presence" in text
        assert "timing" in text

    def test_anchor_report_content(self):
        """event_anchor_validation_report.md must show synthetic source."""
        from foley_cw.cli.phase0_feasibility import run_phase0_synthetic
        out = _tmp_out()
        run_phase0_synthetic(out_dir=out, n_videos=3, seed=0, config_dir=None, fast=True)
        text = (out / "event_anchor_validation_report.md").read_text(encoding="utf-8")
        assert "synthetic" in text.lower()

    def test_deterministic_under_same_seed(self):
        """Two calls with the same seed produce the same tokens."""
        from foley_cw.cli.phase0_feasibility import run_phase0_synthetic
        tokens_a = run_phase0_synthetic(
            out_dir=_tmp_out(), n_videos=3, seed=42, config_dir=None, fast=True
        )
        tokens_b = run_phase0_synthetic(
            out_dir=_tmp_out(), n_videos=3, seed=42, config_dir=None, fast=True
        )
        assert tokens_a == tokens_b


# ---------------------------------------------------------------------------
# phases123_maps tests
# ---------------------------------------------------------------------------


class TestPhases123Maps:

    def test_imports(self):
        """Module must be importable with numpy only."""
        from foley_cw.cli import phases123_maps  # noqa: F401

    def test_main_synthetic_exits_zero(self):
        """main() with --synthetic should exit 0."""
        from foley_cw.cli.phases123_maps import main
        out = _tmp_out()
        rc = main(["--synthetic", "--out", str(out), "--n-videos", "3", "--seed", "42"])
        assert rc == 0, f"main returned {rc}"

    def test_main_no_synthetic_exits_nonzero(self):
        """main() with --no-synthetic should exit non-zero."""
        from foley_cw.cli.phases123_maps import main
        out = _tmp_out()
        rc = main(["--no-synthetic", "--out", str(out), "--n-videos", "3"])
        assert rc != 0, "Expected non-zero exit for --no-synthetic"

    def test_output_files_written(self):
        """All four result files must be written."""
        from foley_cw.cli.phases123_maps import main
        out = _tmp_out()
        rc = main(["--synthetic", "--out", str(out), "--n-videos", "3", "--seed", "7"])
        assert rc == 0
        expected = [
            "commitment_map.csv",
            "readout_map.csv",
            "commitment_readout_gap_report.md",
            "go_no_go_decision.md",
        ]
        for fname in expected:
            p = out / fname
            assert p.exists(), f"Expected output file missing: {fname}"
            assert p.stat().st_size > 0, f"Output file is empty: {fname}"

    def test_tokens_are_valid(self):
        """run_phases123_synthetic must return a non-empty list of known Phase-3 tokens."""
        from foley_cw.cli.phases123_maps import run_phases123_synthetic
        out = _tmp_out()
        tokens = run_phases123_synthetic(
            out_dir=out, n_videos=3, seed=0, config_dir=None, fast=True
        )
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        valid_tokens = {
            "GO_MAP", "GO_READOUT", "GO_RESTRICTED", "GO_DIAGNOSTIC",
            "STOP_ADSR", "STOP_PROJECT",
            "FORK_ALPHA_NO_VALID_OPERATING_POINT",
            "THRESHOLDS_REGISTERED_PRE_MAP",  # placeholder written before maps
        }
        for tok in tokens:
            assert isinstance(tok, str)
            assert tok in valid_tokens, f"Unexpected token: {tok!r}"

    def test_commitment_map_csv_structure(self):
        """commitment_map.csv must have the correct header columns."""
        from foley_cw.cli.phases123_maps import run_phases123_synthetic
        out = _tmp_out()
        run_phases123_synthetic(out_dir=out, n_videos=3, seed=0, config_dir=None, fast=True)
        csv_path = out / "commitment_map.csv"
        with open(csv_path, "r") as fh:
            header = fh.readline().strip()
        actual_cols = set(header.split(","))
        # Surface columns (the A(axis,s,alpha) surface) ...
        surface_cols = {"axis_id", "s", "alpha", "a_fork", "a_independent",
                        "commit_gain", "n_videos"}
        # ... plus s_commit with CIs (plan §5: commitment_map.csv carries s_commit with CIs).
        window_cols = {"s_commit", "s_commit_ci_low", "s_commit_ci_high", "s_commit_underpowered"}
        assert surface_cols <= actual_cols, f"missing surface cols: {surface_cols - actual_cols}"
        assert window_cols <= actual_cols, f"missing s_commit window cols: {window_cols - actual_cols}"

    def test_readout_map_csv_structure(self):
        """readout_map.csv must have the correct header columns."""
        from foley_cw.cli.phases123_maps import run_phases123_synthetic
        out = _tmp_out()
        run_phases123_synthetic(out_dir=out, n_videos=3, seed=0, config_dir=None, fast=True)
        csv_path = out / "readout_map.csv"
        with open(csv_path, "r") as fh:
            header = fh.readline().strip()
        actual_cols = set(header.split(","))
        surface_cols = {"axis_id", "probe", "s", "target", "score", "n_videos"}
        # plus s_read with CIs (plan §5: readout_map.csv carries s_read with CIs).
        window_cols = {"s_read", "s_read_ci_low", "s_read_ci_high", "s_read_underpowered"}
        assert surface_cols <= actual_cols, f"missing surface cols: {surface_cols - actual_cols}"
        assert window_cols <= actual_cols, f"missing s_read window cols: {window_cols - actual_cols}"

    def test_go_no_go_decision_has_thresholds(self):
        """go_no_go_decision.md must record the pre-registered thresholds."""
        from foley_cw.cli.phases123_maps import run_phases123_synthetic
        out = _tmp_out()
        run_phases123_synthetic(out_dir=out, n_videos=3, seed=0, config_dir=None, fast=True)
        text = (out / "go_no_go_decision.md").read_text(encoding="utf-8")
        assert "theta_commit" in text
        assert "theta_read" in text
        assert "frozen" in text.lower()

    def test_gap_report_has_commitment_and_readout_sections(self):
        """commitment_readout_gap_report.md must contain the commitment and readout tables."""
        from foley_cw.cli.phases123_maps import run_phases123_synthetic
        out = _tmp_out()
        run_phases123_synthetic(out_dir=out, n_videos=3, seed=0, config_dir=None, fast=True)
        text = (out / "commitment_readout_gap_report.md").read_text(encoding="utf-8")
        assert "Commitment Windows" in text
        assert "Readout Windows" in text
        assert "Gap" in text

    def test_commitment_map_csv_has_rows(self):
        """commitment_map.csv must have data rows (not just a header)."""
        from foley_cw.cli.phases123_maps import run_phases123_synthetic
        out = _tmp_out()
        run_phases123_synthetic(out_dir=out, n_videos=3, seed=0, config_dir=None, fast=True)
        csv_path = out / "commitment_map.csv"
        lines = csv_path.read_text(encoding="utf-8").strip().split("\n")
        # header + at least one data row
        assert len(lines) > 1, f"commitment_map.csv has no data rows; lines={lines}"

    def test_readout_map_csv_has_rows(self):
        """readout_map.csv must have data rows (not just a header)."""
        from foley_cw.cli.phases123_maps import run_phases123_synthetic
        out = _tmp_out()
        run_phases123_synthetic(out_dir=out, n_videos=3, seed=0, config_dir=None, fast=True)
        csv_path = out / "readout_map.csv"
        lines = csv_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) > 1, f"readout_map.csv has no data rows; lines={lines}"

    def test_no_mmaudio_fabrication(self, capsys):
        """--no-synthetic must print a clear message and not write any output files."""
        from foley_cw.cli.phases123_maps import main
        out = _tmp_out()
        rc = main(["--no-synthetic", "--out", str(out)])
        assert rc != 0
        captured = capsys.readouterr()
        assert "MMAudio" in captured.err or "mmaudio" in captured.err.lower()
        # No output files should have been fabricated
        for f in out.iterdir():
            assert False, f"Unexpected output file created: {f}"

    def test_deterministic_under_same_seed(self):
        """Two calls with the same seed produce the same Phase-3 tokens."""
        from foley_cw.cli.phases123_maps import run_phases123_synthetic
        tokens_a = run_phases123_synthetic(
            out_dir=_tmp_out(), n_videos=3, seed=99, config_dir=None, fast=True
        )
        tokens_b = run_phases123_synthetic(
            out_dir=_tmp_out(), n_videos=3, seed=99, config_dir=None, fast=True
        )
        assert tokens_a == tokens_b


# ---------------------------------------------------------------------------
# Cross-module integration: phase0 -> phases123
# ---------------------------------------------------------------------------


class TestEndToEnd:

    def test_phase0_then_phase123(self):
        """Full synthetic pipeline: phase0 emits GO_MAPS_PHASE, phase123 runs successfully."""
        from foley_cw.cli.phase0_feasibility import run_phase0_synthetic
        from foley_cw.cli.phases123_maps import run_phases123_synthetic

        p0_out = _tmp_out()
        p0_tokens = run_phase0_synthetic(
            out_dir=p0_out, n_videos=4, seed=0, config_dir=None, fast=True
        )
        assert "GO_MAPS_PHASE" in p0_tokens, (
            f"Phase-0 did not emit GO_MAPS_PHASE; got {p0_tokens}"
        )

        p123_out = _tmp_out()
        p3_tokens = run_phases123_synthetic(
            out_dir=p123_out, n_videos=4, seed=0, config_dir=None, fast=True
        )
        assert isinstance(p3_tokens, list) and len(p3_tokens) > 0

        # Verify the four required files exist and are non-empty
        for fname in ["commitment_map.csv", "readout_map.csv",
                      "commitment_readout_gap_report.md", "go_no_go_decision.md"]:
            p = p123_out / fname
            assert p.exists() and p.stat().st_size > 0, f"Missing or empty: {fname}"
