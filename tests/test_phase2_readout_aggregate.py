import math

import pytest

from scripts.phase2_readout import bootstrap_clip_mean, summarize_rows


def test_bootstrap_uses_clips_as_the_sampling_unit():
    values = {"clip_a": [1.0, 1.0, 1.0], "clip_b": [0.0]}

    point, lo, hi, n_clips = bootstrap_clip_mean(values, n_boot=1000, seed=0)
    repeat = bootstrap_clip_mean(values, n_boot=1000, seed=0)

    assert point == 0.5
    assert point != pytest.approx(0.75)  # pooled-row mean
    assert (point, lo, hi, n_clips) == repeat
    assert n_clips == 2


def test_summarize_rows_adds_metric_baseline_and_margin_without_gen_id():
    rows = [
        {"clip": "a", "j": 0, "axis_id": "timing", "probe": "p",
         "target": "ode", "s": 0.05, "correct": 1.0},
        {"clip": "a", "j": 1, "axis_id": "timing", "probe": "p",
         "target": "ode", "s": 0.05, "correct": 0.0},
        {"clip": "b", "j": 0, "axis_id": "timing", "probe": "p",
         "target": "ode", "s": 0.05, "correct": 0.0},
    ]
    labels = {
        "a__p1cfg1_ind0": {"timing": "zero"},
        "a__p1cfg1_ind1": {"timing": "zero"},
        "b__p1cfg1_ind0": {"timing": "one"},
    }

    summary, has_gen_id = summarize_rows(rows, labels)

    assert not has_gen_id
    assert len(summary) == 1
    row = summary[0]
    assert row["metric"] == "exact_match"
    assert row["accuracy"] == 0.25  # mean([mean(1, 0), mean(0)])
    assert row["n_clips"] == 2
    assert row["majority_baseline"] == pytest.approx(2 / 3)
    assert row["margin_over_majority"] == pytest.approx(0.25 - 2 / 3)
    assert "balanced_accuracy" not in row


def test_balanced_accuracy_requires_persisted_gen_ids():
    rows = [
        {"clip": "a", "j": 0, "gen_id": "ga", "axis_id": "presence",
         "probe": "p", "target": "ode", "s": 0.05, "correct": 1.0},
        {"clip": "a", "j": 1, "gen_id": "gb", "axis_id": "presence",
         "probe": "p", "target": "ode", "s": 0.05, "correct": 0.0},
        {"clip": "b", "j": 0, "gen_id": "gc", "axis_id": "presence",
         "probe": "p", "target": "ode", "s": 0.05, "correct": 1.0},
    ]
    labels = {
        "ga": {"presence": "present"},
        "gb": {"presence": "present"},
        "gc": {"presence": "absent"},
    }

    summary, has_gen_id = summarize_rows(rows, labels)

    assert has_gen_id
    assert summary[0]["balanced_accuracy"] == pytest.approx(0.75)


def test_material_is_cosine_and_has_no_categorical_baseline():
    rows = [
        {"clip": "a", "j": 0, "axis_id": "material", "probe": "p",
         "target": "ode", "s": 0.05, "correct": 0.4},
    ]

    summary, _ = summarize_rows(rows, {})

    assert summary[0]["metric"] == "cosine"
    assert math.isnan(summary[0]["majority_baseline"])
    assert math.isnan(summary[0]["margin_over_majority"])


def test_missing_ode_target_label_fails_the_join():
    rows = [
        {"clip": "a", "j": 0, "axis_id": "class", "probe": "p",
         "target": "ode", "s": 0.05, "correct": 1.0},
    ]

    with pytest.raises(ValueError, match="missing ODE-target label"):
        summarize_rows(rows, {})
