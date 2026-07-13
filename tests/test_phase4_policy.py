"""Regression tests for the corrected Phase-4 headroom screen."""

from scripts.phase4_policy import headroom_supported


def test_headroom_requires_quality_gap_and_two_percent_compute_tolerance():
    # Corrected 200-clip replay values committed in policy_pareto_corrected.csv.
    assert headroom_supported(0.785, 49_151, 0.365, 51_525)
    assert not headroom_supported(0.374, 49_151, 0.365, 51_525)
    assert not headroom_supported(0.785, 52_556, 0.365, 51_525)
