import json
from collections import Counter

import numpy as np
import pytest

from foley_cw.arc4_gpu import (
    B6_S_GRID,
    atomic_npz_create,
    load_confident_clip_labels,
    select_balanced_pairs,
    valid_b1_bundle,
    valid_b2_bundle,
    validate_pair_manifest,
)


def test_atomic_npz_create_never_overwrites(tmp_path):
    path = tmp_path / "bundle.npz"
    atomic_npz_create(path, value=np.array([1]))
    with pytest.raises(FileExistsError):
        atomic_npz_create(path, value=np.array([2]))
    assert np.load(path)["value"].tolist() == [1]


def test_b1_and_b2_schema_validation(tmp_path):
    b1 = tmp_path / "b1.npz"
    atomic_npz_create(
        b1,
        token_mean=np.zeros((12, 448), np.float16),
        token_mean_max=np.zeros((12, 896), np.float16),
        tokens_sub=np.zeros((12, 64, 448), np.float16),
        xattn_clip=np.zeros((4, 64), np.float16),
        xattn_frac=np.zeros(4, np.float32),
    )
    assert valid_b1_bundle(b1)

    b2 = tmp_path / "b2.npz"
    atomic_npz_create(
        b2,
        pooled=np.zeros(2688, np.float32),
        clip_f=np.zeros(896, np.float32),
        sync_f=np.zeros(896, np.float32),
        clip_f_c=np.zeros(896, np.float32),
        cond_keys=np.array(["clip_f", "sync_f", "clip_f_c"]),
        raw_shapes=np.array("{}"),
    )
    assert valid_b2_bundle(b2)


def test_confident_labels_require_complete_cached_rows(tmp_path):
    path = tmp_path / "measurements.jsonl"
    rows = []
    for clip, labels in {
        "complete": ["impact"] * 9 + ["abstain"] * 7,
        "tie": ["zeta"] * 8 + ["alpha"] * 8,
        "all_abstain": ["abstain"] * 16,
        "incomplete": ["impact"] * 15,
    }.items():
        for label in labels:
            rows.append({
                "axis_id": "class",
                "extra": {"role": "p1cfg1_independent", "clip": clip},
                "target": {"label": label},
            })
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    assert load_confident_clip_labels(path, "p1cfg1_independent") == {
        "complete": "impact",
        "tie": "alpha",
    }


def test_balanced_pair_selection_is_deterministic_and_cross_class():
    labels = {
        f"{label}_{index}": label
        for label in ("a", "b", "c", "d")
        for index in range(10)
    }
    first = select_balanced_pairs(labels, cfg=1.0, n_pairs=128, seed=0)
    second = select_balanced_pairs(labels, cfg=1.0, n_pairs=128, seed=0)
    assert first == second
    assert len({(pair["source"], pair["donor"]) for pair in first}) == 128
    assert all(pair["source_cached_label"] != pair["donor_cached_label"]
               for pair in first)
    src = Counter(pair["source_cached_label"] for pair in first)
    donor = Counter(pair["donor_cached_label"] for pair in first)
    assert max(src.values()) - min(src.values()) <= 1
    assert max(donor.values()) - min(donor.values()) <= 1

    manifest = {
        "seed": 0,
        "s_grid": list(B6_S_GRID),
        "pairs": first + select_balanced_pairs(labels, cfg=4.5, n_pairs=128, seed=0),
    }
    validate_pair_manifest(manifest)
