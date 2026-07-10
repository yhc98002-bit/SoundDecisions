"""Tests for foley_cw.determination — three-share budget + Fig-1 taxonomy."""
from __future__ import annotations

import math

import numpy as np

from foley_cw.determination import (build_determination_budget, clip_shares,
                                     s_commit)


class TestClipShares:
    def test_label_axis_hand_values(self):
        sh = clip_shares(0.5, {0.05: 0.6, 0.90: 0.9}, is_embedding=False)
        assert sh["conditioning_share"] == 0.5
        assert sh["seed_share"] == pytest_approx(0.1)
        assert sh["trajectory_share"] == pytest_approx(0.3)
        assert sh["residual"] == pytest_approx(0.1)
        assert sh["commit"][0.05] == pytest_approx(0.2)   # (0.6-0.5)/(1-0.5)
        assert sh["commit"][0.90] == pytest_approx(0.8)   # (0.9-0.5)/0.5

    def test_embedding_axis_uses_trajectory_normalization(self):
        sh = clip_shares(0.5, {0.05: 0.6, 0.90: 0.9}, is_embedding=True)
        # embedding commit normalizes by (1 - A_fork(s_min)), NOT A_independent
        assert sh["commit"][0.05] == pytest_approx(0.0)   # (0.6-0.6)/(1-0.6)
        assert sh["commit"][0.90] == pytest_approx(0.75)  # (0.9-0.6)/0.4

    def test_shares_clip_at_zero(self):
        # anti-correlation: A_fork(s_min) < A_independent -> seed share clipped to 0
        sh = clip_shares(0.7, {0.05: 0.6, 0.90: 0.65}, is_embedding=False)
        assert sh["seed_share"] == 0.0
        assert sh["trajectory_share"] == pytest_approx(0.05)

    def test_nan_a_fork_propagates_to_commit_nan(self):
        sh = clip_shares(0.5, {0.05: float("nan"), 0.90: 0.9}, is_embedding=False)
        assert math.isnan(sh["commit"][0.05])


class TestSCommit:
    def test_first_crossing(self):
        assert s_commit({0.05: 0.2, 0.45: 0.5, 0.90: 0.8}, 0.7) == 0.90
        assert s_commit({0.05: 0.8, 0.90: 0.9}, 0.7) == 0.05

    def test_never_crosses_is_nan(self):
        assert math.isnan(s_commit({0.05: 0.1, 0.90: 0.5}, 0.7))


class TestBudgetAggregate:
    def test_aggregate_means_and_taxonomy(self):
        # two clips, one label axis; clip A commits early, clip B never
        per_clip = {"presence": {
            "A": {0.05: 0.9, 0.90: 0.95},   # high seed -> seed-determined, commits at 0.05
            "B": {0.05: 0.5, 0.90: 0.55},   # never commits (theta 0.7)
        }}
        a_ind = {"presence": {"A": 0.5, "B": 0.5}}
        out = build_determination_budget(per_clip, a_ind, embedding_axes=set(),
                                         theta_commit=0.7, n_boot=50, seed=0)
        p = out["presence"]
        assert p["n_clips"] == 2
        assert p["taxonomy"]["seed_determined"] == 1     # clip A
        assert p["taxonomy"]["never_commits"] == 1       # clip B
        assert p["taxonomy"]["trajectory_early"] == 1    # clip A commits at 0.05 <= 0.4
        assert 0.0 <= p["budget"]["seed_share"]["mean"] <= 1.0

    def test_video_determined_counted(self):
        per_clip = {"x": {"A": {0.05: 0.99, 0.90: 0.99}}}
        a_ind = {"x": {"A": 0.95}}     # >= 0.9 -> video-determined
        out = build_determination_budget(per_clip, a_ind, set(), 0.7, n_boot=10)
        assert out["x"]["taxonomy"]["video_determined"] == 1


def pytest_approx(x, tol=1e-9):
    import pytest
    return pytest.approx(x, abs=tol)
