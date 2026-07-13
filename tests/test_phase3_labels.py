"""Regression tests for the gap-aware Phase-3 R-class labels."""

import json
import math
from pathlib import Path

from scripts import phase3_decision
from scripts.phase3_decision import classify_r_window


def test_r_class_decision_order_covers_all_five_branches():
    assert classify_r_window(math.nan, 0.25) == "R1 (uncommitted — defer)"
    assert classify_r_window(0.25, math.nan) == "R2 (committed, unreadable — Track P)"
    assert classify_r_window(0.346, 0.75) == (
        "R2-in-window (committed at 0.35, readable from 0.75)"
    )
    assert classify_r_window(0.70, 0.80) == "early-action (committed & readable)"
    assert classify_r_window(0.80, 0.90) == "R2 (readable only near s=1)"


def test_trajectory_predictable_suffix_is_appended_for_negative_gap():
    assert classify_r_window(0.70, 0.50) == (
        "early-action (committed & readable) "
        "(trajectory-predictable before tail-stable)"
    )


def test_arc3_go_booleans_are_unchanged(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    phase1 = root / "results/stage0/phase1"
    original = json.loads((phase1 / "phase3_decision.json").read_text())
    monkeypatch.setattr(
        "sys.argv",
        [
            "phase3_decision.py",
            "--phase1-dir", str(phase1),
            "--output-dir", str(tmp_path),
            "--output-stem", "corrected",
        ],
    )

    assert phase3_decision.main() == 0
    corrected = json.loads((tmp_path / "corrected.json").read_text())
    for key in ("go_map", "go_readout"):
        assert json.dumps(corrected[key]).encode() == json.dumps(original[key]).encode()
    class_row = next(row for row in corrected["rows"] if row["axis"] == "class")
    assert class_row["class"] == (
        "R2-in-window (committed at 0.35, readable from 0.75)"
    )
