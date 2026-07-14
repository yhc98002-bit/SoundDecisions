"""Synthetic checks for the WP-A2 determination partition."""

import pytest

from foley_cw.determination import clip_shares
from scripts.arc4_determination_partition import first_crossing, km_median


def test_first_crossing_uses_discrete_grid_and_ignores_nan():
    curve = {s: float("nan") for s in (0.05, 0.15, 0.25, 0.35, 0.45, 0.60, 0.75, 0.90)}
    curve[0.35] = 0.69
    curve[0.45] = 0.70
    assert first_crossing(curve) == pytest.approx(0.45)


def test_material_curve_uses_trajectory_floor_not_a_independent():
    forks = {0.05: 0.80, 0.15: 0.85, 0.25: 0.95}
    curve = clip_shares(0.60, forks, is_embedding=True)["commit"]
    assert curve[0.05] == pytest.approx(0.0)
    assert curve[0.25] == pytest.approx(0.75)


def test_km_excludes_video_determined_and_keeps_censored():
    rows = [
        {"status": "VIDEO_DETERMINED", "s_commit": None},
        {"status": "CROSSING", "s_commit": 0.15},
        {"status": "CROSSING", "s_commit": 0.35},
        {"status": "CENSORED", "s_commit": None},
    ]
    assert km_median(rows) == pytest.approx(0.35)
