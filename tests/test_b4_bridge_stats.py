"""Regression tests for corrected B4 bootstrap statistics."""

import zlib

import pytest

from foley_cw import bridge as B
from foley_cw.stats import bootstrap_over_videos
from scripts import b4_bridge


def _clip(no, oracle, full, same, reject, resample, rerank):
    key = "correct_presence"
    return {"rows": {
        "non_oracle_axis_gated": {key: no},
        "oracle_axis_gated": {key: oracle},
        "full_bon": {key: full},
        "same_compute_bon": {key: same},
        "diffrs_scalar": {key: reject},
        "smc_scalar": {key: resample},
        "final_rerank": {key: rerank},
    }}


def test_axis_bootstrap_seed_uses_crc32():
    expected = 17 + zlib.crc32(b"material") % 1000
    assert b4_bridge.axis_bootstrap_seed(17, "material") == expected


def test_bootstrap_recomputes_scalar_floor_within_each_draw():
    clips = [
        _clip(0.9, 1.0, 0.8, 0.1, 0.2, 0.1, 0.2),
        _clip(0.7, 1.0, 0.1, 0.8, 0.2, 0.1, 0.2),
        _clip(0.8, 1.0, 0.6, 0.4, 0.2, 0.1, 0.2),
        _clip(0.6, 1.0, 0.2, 0.7, 0.2, 0.1, 0.2),
    ]
    key = "correct_presence"
    fixed_floor, _ = b4_bridge.scalar_floor_for_sample(clips, key)

    def fixed_stat(sample):
        no = sum(c["rows"]["non_oracle_axis_gated"][key] for c in sample) / len(sample)
        oracle = sum(c["rows"]["oracle_axis_gated"][key] for c in sample) / len(sample)
        return B.headroom_recovery(no, fixed_floor, oracle)

    corrected = bootstrap_over_videos(
        clips, lambda sample: b4_bridge.recovery_for_sample(sample, "presence"),
        n_boot=500, seed=0,
    )
    control = bootstrap_over_videos(clips, fixed_stat, n_boot=500, seed=0)
    assert corrected[0] == pytest.approx(control[0])
    corrected_width = corrected[2] - corrected[1]
    control_width = control[2] - control[1]
    assert corrected_width != pytest.approx(control_width)
