"""Tests for the post-hoc video-determined Class sensitivity."""

import math

from scripts.class_video_determined_sensitivity import (
    nondetermined_sensitivity,
    optional_float,
)


def test_optional_float_preserves_missingness():
    assert math.isnan(optional_float(""))
    assert optional_float("0.25") == 0.25


def test_determined_cases_are_excluded_without_changing_all_cell_curve():
    baselines = [
        {"video_id": "a", "video_determined": True},
        {"video_id": "b", "video_determined": False},
    ]
    cells = [
        {"video_id": "a", "progress": 0.1, "commitment_gain": 0.0},
        {"video_id": "a", "progress": 0.2, "commitment_gain": 0.0},
        {"video_id": "b", "progress": 0.1, "commitment_gain": 0.7},
        {"video_id": "b", "progress": 0.2, "commitment_gain": 0.8},
    ]
    result = nondetermined_sensitivity(
        cells, baselines, theta=0.7, reproduction_cutoff=0.15
    )
    assert result["registered_all_cell_sustained_crossing"] is None
    assert result["nondetermined_only_sustained_crossing"] == 0.1
    assert result["nondetermined_only_reproduces_by_cutoff"] is True
    assert result["video_determined_ids"] == ["a"]


def test_nondetermined_sensitivity_respects_sustained_rule():
    baselines = [{"video_id": "b", "video_determined": False}]
    cells = [
        {"video_id": "b", "progress": 0.1, "commitment_gain": 0.8},
        {"video_id": "b", "progress": 0.2, "commitment_gain": 0.6},
        {"video_id": "b", "progress": 0.3, "commitment_gain": 0.9},
    ]
    result = nondetermined_sensitivity(
        cells, baselines, theta=0.7, reproduction_cutoff=0.2
    )
    assert result["nondetermined_only_sustained_crossing"] == 0.3
    assert result["nondetermined_only_reproduces_by_cutoff"] is False
