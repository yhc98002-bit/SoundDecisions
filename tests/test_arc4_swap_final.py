"""Focused checks for the WP-A2 swap estimands."""

import pytest

from scripts.arc4_swap_final import clopper_pearson, estimand


def test_follow_only_excludes_categorical_collision():
    rows = [
        {"axis_id": "class", "source": "a", "donor": "a", "swapped": "a"},
        {"axis_id": "class", "source": "a", "donor": "b", "swapped": "b"},
        {"axis_id": "class", "source": "a", "donor": "b", "swapped": "a"},
    ]
    result = estimand(rows)
    assert result["follow_only"] == pytest.approx(1 / 3)
    assert result["retention_only"] == pytest.approx(1 / 3)
    assert result["neither"] == pytest.approx(1 / 3)


def test_clopper_pearson_handles_boundaries():
    lo0, hi0 = clopper_pearson(0, 10)
    lo1, hi1 = clopper_pearson(10, 10)
    assert lo0 == pytest.approx(0.0)
    assert 0.0 < hi0 < 1.0
    assert 0.0 < lo1 < 1.0
    assert hi1 == pytest.approx(1.0)
