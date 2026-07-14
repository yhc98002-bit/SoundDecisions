"""Regression coverage for the B4 joint-recovery bootstrap floor."""

import pytest

from foley_cw import bridge as B
from foley_cw.stats import bootstrap_over_videos
from scripts import b4_bridge


def _clip(no, oracle, full, same, reject, resample, rerank):
    return {"rows": {
        "non_oracle_axis_gated": {"final": no},
        "oracle_axis_gated": {"final": oracle},
        "full_bon": {"final": full},
        "same_compute_bon": {"final": same},
        "diffrs_scalar": {"final": reject},
        "smc_scalar": {"final": resample},
        "final_rerank": {"final": rerank},
    }}


def test_joint_bootstrap_recomputes_scalar_floor_within_each_draw():
    clips = [
        _clip(0.9, 1.0, 0.8, 0.1, 0.2, 0.1, 0.2),
        _clip(0.7, 1.0, 0.1, 0.8, 0.2, 0.1, 0.2),
        _clip(0.8, 1.0, 0.6, 0.4, 0.2, 0.1, 0.2),
        _clip(0.6, 1.0, 0.2, 0.7, 0.2, 0.1, 0.2),
    ]
    fixed_floor, _ = b4_bridge.scalar_floor_for_sample(clips, "final")

    def fixed_floor_stat(sample):
        no = sum(c["rows"]["non_oracle_axis_gated"]["final"] for c in sample) / len(sample)
        oracle = sum(c["rows"]["oracle_axis_gated"]["final"] for c in sample) / len(sample)
        return B.headroom_recovery(no, fixed_floor, oracle)

    corrected = bootstrap_over_videos(
        clips, b4_bridge.joint_recovery_for_sample, n_boot=500, seed=0,
    )
    control = bootstrap_over_videos(clips, fixed_floor_stat, n_boot=500, seed=0)

    assert corrected[0] == pytest.approx(control[0])
    corrected_width = corrected[2] - corrected[1]
    control_width = control[2] - control[1]
    assert corrected_width != pytest.approx(control_width)
