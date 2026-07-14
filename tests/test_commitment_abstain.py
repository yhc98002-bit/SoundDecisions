"""Regression tests for categorical abstention in commitment statistics."""

from __future__ import annotations

import numpy as np
import pytest

from foley_cw import commitment
from foley_cw.config import load_config
from foley_cw.types import AxisKind, ScheduleSpec, SelfTarget


def _class_target(label: str) -> SelfTarget:
    return SelfTarget(axis_id="class", kind=AxisKind.CATEGORICAL, label=label)


def _patch_measurement(monkeypatch):
    monkeypatch.setattr(
        commitment,
        "measure_self_target",
        lambda audio, axis, measurer: _class_target(str(audio)),
    )


def _class_axis():
    return next(axis for axis in load_config().axes if axis.id == "class")


def test_a_independent_excludes_abstain_pairs(monkeypatch):
    labels = iter(("abstain", "abstain", "impact", "impact"))
    monkeypatch.setattr(
        commitment,
        "generate_trajectory",
        lambda *args, **kwargs: {"audio": next(labels)},
    )
    _patch_measurement(monkeypatch)
    schedule = ScheduleSpec(
        n_steps=2, scan_points=(0.0, 1.0), N_independent=4, K_forks=2
    )

    value = commitment.a_independent(
        object(), object(), _class_axis(), object(), schedule, np.random.default_rng(0)
    )

    assert value == pytest.approx(1.0)


def test_a_fork_excludes_abstain_pairs(monkeypatch):
    monkeypatch.setattr(
        commitment,
        "fork_tail",
        lambda *args, **kwargs: ["abstain", "abstain", "impact", "impact"],
    )
    _patch_measurement(monkeypatch)
    schedule = ScheduleSpec(
        n_steps=2, scan_points=(0.0, 1.0), N_independent=2, K_forks=4
    )

    value = commitment.a_fork(
        object(), np.zeros(1), 0.5, object(), _class_axis(), object(),
        alpha=0.8, schedule=schedule, rng=np.random.default_rng(0),
    )

    assert value == pytest.approx(1.0)


@pytest.mark.parametrize("labels", [
    ("abstain", "abstain", "abstain", "impact"),
    ("abstain", "abstain", "abstain", "abstain"),
])
def test_categorical_cell_with_fewer_than_two_confident_labels_is_unscorable(
    monkeypatch, labels
):
    values = iter(labels)
    monkeypatch.setattr(
        commitment,
        "generate_trajectory",
        lambda *args, **kwargs: {"audio": next(values)},
    )
    _patch_measurement(monkeypatch)
    schedule = ScheduleSpec(
        n_steps=2, scan_points=(0.0, 1.0), N_independent=4, K_forks=2
    )

    value = commitment.a_independent(
        object(), object(), _class_axis(), object(), schedule, np.random.default_rng(0)
    )

    assert np.isnan(value)
