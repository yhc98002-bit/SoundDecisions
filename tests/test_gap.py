"""Tests for foley_cw/gap.py (numpy-only, CPU).

Scientific contracts verified:
  * gap: returns s_read.s_hat - s_commit.s_hat; NaN propagation.
  * r1_r2_crosstab: correct R1/R2/early_action labelling under known curves.
  * decide_phase0:
      - synthetic-style inputs (trajectory_ok, token=OK, manifest_ok, >=3 pass)
        -> GO_MAPS_PHASE only.
      - trajectory failure -> NO_TRAJECTORY_ACCESS.
      - FIX_SCORE_CONVERSION token -> FIX_SCORE_CONVERSION emitted.
      - manifest failure -> STOP_PROJECT emitted.
      - too few reliable axes -> STOP_PROJECT emitted.
      - multiple simultaneous failures -> multiple tokens.
  * decide_phase3:
      - coincident s_commit near s=1 -> STOP_ADSR.
      - alpha_ok=False -> FORK_ALPHA_NO_VALID_OPERATING_POINT.
      - well-spread windows + early readout -> GO_MAP + GO_READOUT.
      - commitment but lagging readout -> GO_DIAGNOSTIC.
      - all tokens can appear simultaneously.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from foley_cw.gap import (
    _EARLY_READ_CEILING,
    _NEAR_S1_THRESHOLD,
    _READOUT_LAG_THRESHOLD,
    _SEP_SCORE_MIN_GO_MAP,
    decide_phase0,
    decide_phase3,
    gap,
    r1_r2_crosstab,
)
from foley_cw.stats import separation_score
from foley_cw.types import (
    GoNoGoDecision,
    ReliabilityResult,
    ScheduleSpec,
    Thresholds,
    WindowEstimate,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reliability(axis_id: str, passed: bool, demoted: bool = False) -> ReliabilityResult:
    return ReliabilityResult(
        axis_id=axis_id,
        determinism=1.0 if passed else 0.3,
        robustness=1.0 if passed else 0.3,
        validity=1.0 if passed else 0.3,
        passed=passed,
        demoted=demoted,
        reason="" if passed else "failed",
    )


def _win(axis_id: str, kind: str, s_hat: float, ci_low: float = 0.0,
         ci_high: float = 0.0, n_videos: int = 10) -> WindowEstimate:
    return WindowEstimate(
        axis_id=axis_id,
        kind=kind,
        s_hat=s_hat,
        ci_low=ci_low,
        ci_high=ci_high,
        n_videos=n_videos,
    )


def _thresholds(theta_commit: float = 0.5, theta_read: float = 0.6) -> Thresholds:
    return Thresholds(
        theta_commit=theta_commit,
        theta_read=theta_read,
        theta_rel=0.8,
        theta_robust=0.7,
        theta_cal=0.6,
        frozen=False,
    )


# ---------------------------------------------------------------------------
# gap
# ---------------------------------------------------------------------------

class TestGap:
    def test_positive_gap(self):
        s_read = _win("ax1", "read", 0.7)
        s_commit = _win("ax1", "commit", 0.4)
        assert gap(s_read, s_commit) == pytest.approx(0.3)

    def test_zero_gap(self):
        s_read = _win("ax1", "read", 0.5)
        s_commit = _win("ax1", "commit", 0.5)
        assert gap(s_read, s_commit) == pytest.approx(0.0)

    def test_negative_gap(self):
        """Readout before commitment — rare but valid."""
        s_read = _win("ax1", "read", 0.3)
        s_commit = _win("ax1", "commit", 0.5)
        assert gap(s_read, s_commit) == pytest.approx(-0.2)

    def test_nan_propagation_read(self):
        s_read = _win("ax1", "read", float("nan"))
        s_commit = _win("ax1", "commit", 0.4)
        assert math.isnan(gap(s_read, s_commit))

    def test_nan_propagation_commit(self):
        s_read = _win("ax1", "read", 0.7)
        s_commit = _win("ax1", "commit", float("nan"))
        assert math.isnan(gap(s_read, s_commit))


# ---------------------------------------------------------------------------
# r1_r2_crosstab
# ---------------------------------------------------------------------------

class TestR1R2Crosstab:
    def _make_curves(self):
        """Single axis, 5 s-points: commitment rises, readout rises later."""
        s_grid = np.array([0.0, 0.2, 0.4, 0.6, 0.8])
        # commit crosses 0.5 between s=0.4 and s=0.6
        commit = {"ax1": np.array([0.0, 0.2, 0.4, 0.6, 0.8])}
        # readout crosses 0.6 between s=0.6 and s=0.8
        readout = {"ax1": np.array([0.0, 0.1, 0.3, 0.5, 0.7])}
        return s_grid, commit, readout

    def test_labels(self):
        s_grid, commit, readout = self._make_curves()
        thr = _thresholds(theta_commit=0.5, theta_read=0.6)
        result = r1_r2_crosstab(commit, readout, s_grid, thr)

        # s=0.0 and s=0.2: commit < 0.5 -> R1
        assert result[("ax1", 0.0)] == "R1"
        assert result[("ax1", 0.2)] == "R1"
        # s=0.4: commit=0.4 < 0.5 -> R1
        assert result[("ax1", 0.4)] == "R1"
        # s=0.6: commit=0.6 >= 0.5, readout=0.5 < 0.6 -> R2
        assert result[("ax1", 0.6)] == "R2"
        # s=0.8: commit=0.8 >= 0.5, readout=0.7 >= 0.6 -> early_action
        assert result[("ax1", 0.8)] == "early_action"

    def test_missing_readout_axis_treated_as_zero(self):
        """Axis in commit_curves but not in readout_curves -> R1 or R2."""
        s_grid = np.array([0.0, 0.5, 1.0])
        commit = {"ax1": np.array([0.0, 0.8, 1.0])}
        readout: dict[str, np.ndarray] = {}  # no readout for ax1
        thr = _thresholds()
        result = r1_r2_crosstab(commit, readout, s_grid, thr)
        # s=0.5: commit=0.8 >= 0.5, readout=0.0 < 0.6 -> R2
        assert result[("ax1", 0.5)] == "R2"
        # s=1.0: commit=1.0 >= 0.5, readout=0.0 < 0.6 -> R2
        assert result[("ax1", 1.0)] == "R2"
        # s=0.0: commit=0.0 < 0.5 -> R1
        assert result[("ax1", 0.0)] == "R1"

    def test_multiple_axes(self):
        s_grid = np.array([0.5])
        commit = {
            "ax1": np.array([0.8]),  # committed
            "ax2": np.array([0.2]),  # uncommitted
        }
        readout = {
            "ax1": np.array([0.9]),  # readable
            "ax2": np.array([0.9]),
        }
        thr = _thresholds()
        result = r1_r2_crosstab(commit, readout, s_grid, thr)
        assert result[("ax1", 0.5)] == "early_action"
        assert result[("ax2", 0.5)] == "R1"

    def test_all_early_action(self):
        s_grid = np.array([1.0])
        commit = {"ax1": np.array([1.0]), "ax2": np.array([1.0])}
        readout = {"ax1": np.array([1.0]), "ax2": np.array([1.0])}
        thr = _thresholds()
        result = r1_r2_crosstab(commit, readout, s_grid, thr)
        assert result[("ax1", 1.0)] == "early_action"
        assert result[("ax2", 1.0)] == "early_action"


# ---------------------------------------------------------------------------
# decide_phase0
# ---------------------------------------------------------------------------

class TestDecidePhase0:
    def _ok_reliability(self, n: int = 4) -> list[ReliabilityResult]:
        return [_reliability(f"ax{i}", passed=True) for i in range(n)]

    def test_all_ok_gives_go_maps_phase(self):
        """All four conditions met -> single token GO_MAPS_PHASE."""
        rel = self._ok_reliability(4)
        decision = decide_phase0(
            validation_token="OK",
            reliability=rel,
            trajectory_ok=True,
            manifest_ok=True,
            min_reliable_axes=3,
        )
        assert decision.tokens == ["GO_MAPS_PHASE"]
        assert "GO_MAPS_PHASE" in decision.justification or "trajectory" in decision.justification

    def test_trajectory_fail_gives_no_trajectory_access(self):
        rel = self._ok_reliability(4)
        decision = decide_phase0("OK", rel, trajectory_ok=False, manifest_ok=True)
        assert "NO_TRAJECTORY_ACCESS" in decision.tokens
        assert "GO_MAPS_PHASE" not in decision.tokens

    def test_fix_score_conversion(self):
        rel = self._ok_reliability(4)
        decision = decide_phase0("FIX_SCORE_CONVERSION", rel, trajectory_ok=True, manifest_ok=True)
        assert "FIX_SCORE_CONVERSION" in decision.tokens
        assert "GO_MAPS_PHASE" not in decision.tokens

    def test_manifest_fail_gives_stop_project(self):
        rel = self._ok_reliability(4)
        decision = decide_phase0("OK", rel, trajectory_ok=True, manifest_ok=False)
        assert "STOP_PROJECT" in decision.tokens
        assert "GO_MAPS_PHASE" not in decision.tokens

    def test_too_few_reliable_axes_gives_stop_project(self):
        rel = [
            _reliability("ax0", passed=True),
            _reliability("ax1", passed=True),
            _reliability("ax2", passed=False),
            _reliability("ax3", passed=False),
        ]
        decision = decide_phase0("OK", rel, trajectory_ok=True, manifest_ok=True, min_reliable_axes=3)
        assert "STOP_PROJECT" in decision.tokens
        assert "GO_MAPS_PHASE" not in decision.tokens

    def test_exactly_min_reliable_axes_passes(self):
        rel = [
            _reliability("ax0", passed=True),
            _reliability("ax1", passed=True),
            _reliability("ax2", passed=True),
            _reliability("ax3", passed=False),
        ]
        decision = decide_phase0("OK", rel, trajectory_ok=True, manifest_ok=True, min_reliable_axes=3)
        assert decision.tokens == ["GO_MAPS_PHASE"]

    def test_multiple_failures_give_multiple_tokens(self):
        rel = [_reliability("ax0", passed=False)]
        decision = decide_phase0(
            "FIX_SCORE_CONVERSION",
            rel,
            trajectory_ok=False,
            manifest_ok=False,
            min_reliable_axes=3,
        )
        assert "NO_TRAJECTORY_ACCESS" in decision.tokens
        assert "FIX_SCORE_CONVERSION" in decision.tokens
        assert "STOP_PROJECT" in decision.tokens
        # STOP_PROJECT should not be duplicated
        assert decision.tokens.count("STOP_PROJECT") == 1

    def test_extra_fields_populated(self):
        rel = self._ok_reliability(4)
        decision = decide_phase0("OK", rel, trajectory_ok=True, manifest_ok=True, min_reliable_axes=3)
        assert decision.extra["n_reliable_axes"] == 4
        assert decision.extra["trajectory_ok"] is True
        assert decision.extra["manifest_ok"] is True
        assert decision.extra["validation_token"] == "OK"

    def test_empty_reliability_list_fails_gate(self):
        decision = decide_phase0("OK", [], trajectory_ok=True, manifest_ok=True, min_reliable_axes=1)
        assert "STOP_PROJECT" in decision.tokens


# ---------------------------------------------------------------------------
# decide_phase3
# ---------------------------------------------------------------------------

class TestDecidePhase3:
    def _spread_windows(self) -> dict[str, WindowEstimate]:
        """Four well-separated, non-overlapping commitment windows."""
        return {
            "presence":   _win("presence",   "commit", 0.2, ci_low=0.15, ci_high=0.25),
            "timing":     _win("timing",     "commit", 0.4, ci_low=0.35, ci_high=0.45),
            "coarse_cls": _win("coarse_cls", "commit", 0.6, ci_low=0.55, ci_high=0.65),
            "material":   _win("material",   "commit", 0.8, ci_low=0.75, ci_high=0.85),
        }

    def _coincident_windows(self) -> dict[str, WindowEstimate]:
        """All windows at the same s — STOP_ADSR case."""
        return {
            "ax1": _win("ax1", "commit", 0.9, ci_low=0.88, ci_high=0.92),
            "ax2": _win("ax2", "commit", 0.91, ci_low=0.89, ci_high=0.93),
        }

    def _late_windows(self) -> dict[str, WindowEstimate]:
        """All s_commit near s=1 — also STOP_ADSR."""
        return {
            "ax1": _win("ax1", "commit", 0.88, ci_low=0.86, ci_high=0.90),
            "ax2": _win("ax2", "commit", 0.92, ci_low=0.90, ci_high=0.94),
            "ax3": _win("ax3", "commit", 0.95, ci_low=0.93, ci_high=0.97),
        }

    def _early_read_windows(
        self, commit_windows: dict[str, WindowEstimate]
    ) -> dict[tuple[str, str, str], WindowEstimate]:
        """Readout windows well before s=1 for each axis."""
        result: dict[tuple[str, str, str], WindowEstimate] = {}
        for axis_id in commit_windows:
            c_s = commit_windows[axis_id].s_hat
            # readout a bit after commitment, well below _EARLY_READ_CEILING
            r_s = min(c_s + 0.05, 0.70)
            result[(axis_id, "energy_onset", "ode")] = _win(
                axis_id, "read", r_s, ci_low=r_s - 0.05, ci_high=r_s + 0.05
            )
        return result

    def test_coincident_windows_gives_stop_adsr(self):
        cw = self._coincident_windows()
        rw: dict[tuple[str, str, str], WindowEstimate] = {}
        thr = _thresholds()
        sep = separation_score(cw)
        decision = decide_phase3(cw, rw, sep, thr, alpha_ok=True)
        assert "STOP_ADSR" in decision.tokens
        # STOP_ADSR (coincident) and GO_MAP (separated) are mutually exclusive.
        assert "GO_MAP" not in decision.tokens

    def test_all_near_s1_gives_stop_adsr(self):
        cw = self._late_windows()
        rw: dict[tuple[str, str, str], WindowEstimate] = {}
        thr = _thresholds()
        sep = separation_score(cw)
        decision = decide_phase3(cw, rw, sep, thr, alpha_ok=True)
        assert "STOP_ADSR" in decision.tokens
        # All windows near s=1 -> not an early-separated map; GO_MAP must not fire.
        assert "GO_MAP" not in decision.tokens

    def test_alpha_not_ok_gives_fork_alpha_token(self):
        cw = self._spread_windows()
        rw = self._early_read_windows(cw)
        thr = _thresholds()
        sep = separation_score(cw)
        decision = decide_phase3(cw, rw, sep, thr, alpha_ok=False)
        assert "FORK_ALPHA_NO_VALID_OPERATING_POINT" in decision.tokens

    def test_spread_windows_good_readout_gives_go_map_and_go_readout(self):
        cw = self._spread_windows()
        rw = self._early_read_windows(cw)
        thr = _thresholds()
        sep = separation_score(cw)
        # Ensure separation is large enough
        assert sep > _SEP_SCORE_MIN_GO_MAP, f"test setup: sep={sep} too low"
        decision = decide_phase3(cw, rw, sep, thr, alpha_ok=True)
        assert "GO_MAP" in decision.tokens
        assert "GO_READOUT" in decision.tokens

    def test_no_valid_windows_gives_stop_project(self):
        cw: dict[str, WindowEstimate] = {}
        rw: dict[tuple[str, str, str], WindowEstimate] = {}
        thr = _thresholds()
        decision = decide_phase3(cw, rw, float("nan"), thr, alpha_ok=True)
        assert "STOP_PROJECT" in decision.tokens

    def test_committed_no_readout_gives_go_diagnostic(self):
        """Commitment window exists but no readout probe for that axis."""
        cw = {
            "coarse_cls": _win("coarse_cls", "commit", 0.4, ci_low=0.35, ci_high=0.45),
        }
        rw: dict[tuple[str, str, str], WindowEstimate] = {}  # no readout
        thr = _thresholds()
        sep = float("nan")  # only 1 axis, no meaningful separation
        decision = decide_phase3(cw, rw, sep, thr, alpha_ok=True)
        assert "GO_DIAGNOSTIC" in decision.tokens

    def test_readout_lags_commitment_gives_go_diagnostic(self):
        """Readout exists but is far behind commitment."""
        cw = {
            "coarse_cls": _win("coarse_cls", "commit", 0.3, ci_low=0.25, ci_high=0.35),
        }
        # readout at 0.3 + 0.15 + epsilon => gap > threshold
        r_s = 0.3 + _READOUT_LAG_THRESHOLD + 0.05
        rw: dict[tuple[str, str, str], WindowEstimate] = {
            ("coarse_cls", "energy_onset", "ode"): _win(
                "coarse_cls", "read", r_s, ci_low=r_s - 0.05, ci_high=r_s + 0.05
            )
        }
        thr = _thresholds()
        sep = float("nan")
        decision = decide_phase3(cw, rw, sep, thr, alpha_ok=True)
        assert "GO_DIAGNOSTIC" in decision.tokens

    def test_multiple_tokens_can_coexist(self):
        """GO_MAP + GO_READOUT + FORK_ALPHA_NO_VALID_OPERATING_POINT can coexist on
        spread windows with early readout. (STOP_ADSR is NOT among them: it is mutually
        exclusive with GO_MAP — see the coincident/near-s=1 tests.)"""
        cw = self._spread_windows()
        rw = self._early_read_windows(cw)
        thr = _thresholds()
        sep = separation_score(cw)
        decision = decide_phase3(cw, rw, sep, thr, alpha_ok=False)
        # FORK_ALPHA must always appear when alpha_ok=False
        assert "FORK_ALPHA_NO_VALID_OPERATING_POINT" in decision.tokens
        # GO_MAP and GO_READOUT should still fire if the windows support it
        assert "GO_MAP" in decision.tokens
        assert "GO_READOUT" in decision.tokens

    def test_go_restricted_only_presence_timing(self):
        """Only presence and timing axes show early ACTIONABLE (readable) windows ->
        GO_RESTRICTED. The axes must be readable early (have a readout window), not merely
        committed early, for a restricted policy to be licensed (plan §3)."""
        cw = {
            "presence": _win("presence", "commit", 0.2, ci_low=0.15, ci_high=0.25),
            "timing":   _win("timing",   "commit", 0.35, ci_low=0.30, ci_high=0.40),
        }
        # presence/timing are readable early (small gap after commitment).
        rw: dict[tuple[str, str, str], WindowEstimate] = {
            ("presence", "energy_onset", "ode"): _win("presence", "read", 0.25, ci_low=0.20, ci_high=0.30),
            ("timing", "energy_onset", "ode"): _win("timing", "read", 0.40, ci_low=0.35, ci_high=0.45),
        }
        thr = _thresholds()
        sep = separation_score(cw)
        decision = decide_phase3(cw, rw, sep, thr, alpha_ok=True)
        assert "GO_RESTRICTED" in decision.tokens

    def test_go_restricted_not_emitted_without_readout(self):
        """Early commitment alone (no readable window) does NOT license GO_RESTRICTED."""
        cw = {
            "presence": _win("presence", "commit", 0.2, ci_low=0.15, ci_high=0.25),
            "timing":   _win("timing",   "commit", 0.35, ci_low=0.30, ci_high=0.40),
        }
        rw: dict[tuple[str, str, str], WindowEstimate] = {}
        thr = _thresholds()
        sep = separation_score(cw)
        decision = decide_phase3(cw, rw, sep, thr, alpha_ok=True)
        assert "GO_RESTRICTED" not in decision.tokens

    def test_tokens_not_duplicated(self):
        """Even with multiple failure paths, each token appears at most once."""
        cw = self._coincident_windows()
        rw: dict[tuple[str, str, str], WindowEstimate] = {}
        thr = _thresholds()
        sep = separation_score(cw)
        decision = decide_phase3(cw, rw, sep, thr, alpha_ok=False)
        token_counts = {t: decision.tokens.count(t) for t in decision.tokens}
        for t, cnt in token_counts.items():
            assert cnt == 1, f"token '{t}' appears {cnt} times"

    def test_nan_windows_are_excluded_from_separation(self):
        """NaN windows do not contaminate separation logic."""
        cw = {
            "ax1": _win("ax1", "commit", float("nan")),
            "ax2": _win("ax2", "commit", float("nan")),
        }
        rw: dict[tuple[str, str, str], WindowEstimate] = {}
        thr = _thresholds()
        sep = separation_score(cw)
        assert math.isnan(sep)
        decision = decide_phase3(cw, rw, sep, thr, alpha_ok=True)
        # All NaN windows -> no valid commit -> STOP_PROJECT
        assert "STOP_PROJECT" in decision.tokens

    def test_decision_returns_gonogodecision(self):
        """Return type is always GoNoGoDecision."""
        cw = self._spread_windows()
        rw = self._early_read_windows(cw)
        thr = _thresholds()
        sep = separation_score(cw)
        decision = decide_phase3(cw, rw, sep, thr, alpha_ok=True)
        assert isinstance(decision, GoNoGoDecision)
        assert isinstance(decision.tokens, list)
        assert isinstance(decision.justification, str)
        assert isinstance(decision.extra, dict)

    def test_phase0_returns_gonogodecision(self):
        rel = [_reliability(f"ax{i}", passed=True) for i in range(3)]
        decision = decide_phase0("OK", rel, trajectory_ok=True, manifest_ok=True)
        assert isinstance(decision, GoNoGoDecision)

    # Ensure module imports cleanly with numpy only
    def test_module_imports_with_numpy_only(self):
        """gap.py must be importable without scipy / torch / librosa."""
        import importlib
        import sys
        # Already imported; just verify no scipy/torch in gap's namespace
        import foley_cw.gap as gap_mod
        src_file = gap_mod.__file__
        assert src_file is not None
        with open(src_file) as fh:
            src = fh.read()
        # Must not have top-level scipy/torch imports
        import re
        top_level_imports = re.findall(r'^(?:import|from)\s+(\S+)', src, re.MULTILINE)
        forbidden = {"scipy", "torch", "librosa"}
        bad = set(top_level_imports) & forbidden
        assert not bad, f"gap.py has top-level forbidden imports: {bad}"
