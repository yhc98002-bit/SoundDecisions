import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from foley_cw.arc4_gpu import (
    B2_BASE_SEEDS,
    B2_S_GRID,
    B6_S_GRID,
    atomic_json_create,
    atomic_npz_create,
    atomic_wav_create,
    load_confident_clip_labels,
    select_b2_clips,
    select_balanced_pairs,
    valid_b1_bundle,
    valid_b2_bundle,
    validate_b2_generation_manifest,
    validate_pair_manifest,
    wav_metadata,
)


def test_atomic_npz_create_never_overwrites(tmp_path):
    path = tmp_path / "bundle.npz"
    atomic_npz_create(path, value=np.array([1]))
    with pytest.raises(FileExistsError):
        atomic_npz_create(path, value=np.array([2]))
    assert np.load(path)["value"].tolist() == [1]


def test_atomic_json_and_wav_never_overwrite(tmp_path):
    pytest.importorskip("soundfile")
    json_path = tmp_path / "journal.json"
    atomic_json_create(json_path, {"value": 1})
    with pytest.raises(FileExistsError):
        atomic_json_create(json_path, {"value": 2})
    assert json.loads(json_path.read_text()) == {"value": 1}

    wav_path = tmp_path / "raw.wav"
    audio = np.linspace(-0.5, 0.5, 128, dtype=np.float32)
    atomic_wav_create(wav_path, audio, sample_rate=16000)
    with pytest.raises(FileExistsError):
        atomic_wav_create(wav_path, -audio, sample_rate=16000)
    meta = wav_metadata(wav_path)
    assert meta["sample_rate"] == 16000
    assert meta["frames"] == 128
    assert meta["channels"] == 1
    assert meta["subtype"] == "FLOAT"


def test_b2_generation_manifest_is_deterministic_and_pins_cardinality():
    pool = [str(index) for index in range(200)]
    selected = select_b2_clips(pool, seed=0)
    assert len(selected) == 48
    assert len(set(selected)) == 48
    assert selected == select_b2_clips(reversed(pool), seed=0)

    manifest = {
        "selection_seed": 0,
        "n_clips": 48,
        "clips": selected,
        "base_seeds": list(B2_BASE_SEEDS),
        "cfg": 4.5,
        "schedule": "sqrt_down",
        "alpha": 0.8,
        "s_grid": list(B2_S_GRID),
        "k_forks": 12,
        "variant": "small_16k",
        "duration_sec": 8.0,
        "num_steps": 20,
        "conditioning": "full_video_clip_synchformer_empty_text",
        "audio_format": "WAV",
        "audio_subtype": "FLOAT",
        "sample_rate": 16000,
        "expected_frames": 128000,
        "expected_artifacts": {
            "base_units": 240,
            "base_wavs": 240,
            "fork_cells": 1920,
            "fork_wavs": 23040,
        },
    }
    validate_b2_generation_manifest(manifest)
    manifest["k_forks"] = 11
    with pytest.raises(ValueError, match="k_forks"):
        validate_b2_generation_manifest(manifest)


def test_b6_generator_is_raw_only_and_resume_guarded():
    source = (Path(__file__).parents[1] / "scripts" / "arc4_b6_generate.py").read_text()
    assert "RealFoleyMeasurer" not in source
    assert "load_config" not in source
    assert "record_measurement" not in source
    assert "atomic_wav_create" in source
    assert "_journal_complete" in source


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
