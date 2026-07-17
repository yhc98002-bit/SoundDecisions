"""Synthetic integrity tests for the B2 Class closure pipeline.

These tests never read the scientific B2 bank and never load a GPU model.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from foley_cw.b2_class_closure import (
    ALLOWED_SCIENTIFIC_STATUSES,
    B2ClosureError,
    DEFAULT_BOOTSTRAP_DRAWS,
    DEFAULT_BOOTSTRAP_SEED,
    POSTERIOR_SCHEMA,
    SENSITIVITY_THRESHOLDS,
    VIDEO_DETERMINED_MIN,
    _first_crossing,
    _majority_label,
    analyze_multiseed,
    assigned_inventory_records,
    build_commitment_cells,
    build_posterior_arrays,
    deterministic_npz_create,
    derive_coarse_scores,
    inventory_b2_roots,
    load_inventory,
    measure_inventory_shard,
    merge_posterior_shards,
    classify_frozen_replication,
    pooled_and_seed_crossings,
    sha256_file,
    summarize_thresholds,
    validate_posterior_arrays,
    validate_class_protocol,
    validate_cuda_determinism_environment,
    validate_canonical_analysis_inputs,
    validate_shard_completion,
    write_inventory,
)
from foley_cw.real_measurer import load_coarse_map


PROTOCOL = Path("experiment/non_human_closure/PROTOCOL.json").resolve()


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _artifact(root: Path, relative: str, *, role: str, progress=None, fork=None) -> dict:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((relative + "\n").encode("utf-8"))
    row = {
        "path": relative,
        "role": role,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "sample_rate": 16000,
        "frames": 4,
        "channels": 1,
        "format": "WAV",
        "subtype": "FLOAT",
    }
    if progress is not None:
        row["s"] = progress
    if fork is not None:
        row["fork_index"] = fork
    return row


def _make_bank_root(
    parent: Path,
    name: str,
    seeds: list[int],
    *,
    clips=("v0", "v1"),
    grid=(0.25, 0.75),
    k_forks=2,
) -> Path:
    root = parent / name
    manifest = {
        "schema_version": 1,
        "clips": list(clips),
        "base_seeds": seeds,
        "s_grid": list(grid),
        "k_forks": k_forks,
        "cfg": 4.5,
        "alpha": 0.8,
        "schedule": "sqrt_down",
        "variant": "synthetic",
        "sample_rate": 16000,
        "expected_frames": 4,
        "audio_subtype": "FLOAT",
        "expected_artifacts": {
            "base_units": len(clips) * len(seeds),
            "base_wavs": len(clips) * len(seeds),
            "fork_cells": len(clips) * len(seeds) * len(grid),
            "fork_wavs": len(clips) * len(seeds) * len(grid) * k_forks,
        },
    }
    manifest_path = root / "generation_manifest.json"
    _write_json(manifest_path, manifest)
    digest = sha256_file(manifest_path)
    manifest_path.with_suffix(".sha256").write_text(
        f"{digest} {manifest_path.name}\n", encoding="utf-8"
    )
    for clip in clips:
        for seed in seeds:
            base = _artifact(
                root, f"raw/{clip}/seed{seed}/base.wav", role="base"
            )
            cell_refs = []
            for progress in grid:
                artifacts = [
                    _artifact(
                        root,
                        f"raw/{clip}/seed{seed}/s{progress:.2f}/fork{fork:02d}.wav",
                        role="fork",
                        progress=progress,
                        fork=fork,
                    )
                    for fork in range(k_forks)
                ]
                cell_path = (
                    root
                    / "journal"
                    / "cells"
                    / f"{clip}__seed{seed}__s{progress:.2f}.json"
                )
                _write_json(
                    cell_path,
                    {
                        "clip": clip,
                        "base_seed": seed,
                        "s": progress,
                        "k_forks": k_forks,
                        "generation_manifest_sha256": digest,
                        "artifacts": artifacts,
                    },
                )
                cell_refs.append(
                    {
                        "s": progress,
                        "journal": str(cell_path.relative_to(root)),
                        "sha256": sha256_file(cell_path),
                    }
                )
            unit_path = root / "journal" / "units" / f"{clip}__seed{seed}.json"
            _write_json(
                unit_path,
                {
                    "clip": clip,
                    "base_seed": seed,
                    "s_grid": list(grid),
                    "fork_wavs": len(grid) * k_forks,
                    "generation_manifest_sha256": digest,
                    "base_artifact": base,
                    "cell_journals": cell_refs,
                    "provenance": {"git_commit": "a" * 40},
                },
            )
    return root


def _make_coarse_map(path: Path) -> Path:
    mapping = {str(index): ("group_a" if index % 2 == 0 else "group_b") for index in range(527)}
    _write_json(
        path,
        {
            "version": "synthetic-v1",
            "coarse_classes": ["group_a", "group_b"],
            "index_to_coarse": mapping,
            "class_excluded_coarse": [],
            "non_event_indices": [],
        },
    )
    return path


@pytest.fixture()
def synthetic_inventory(tmp_path):
    roots = [
        _make_bank_root(tmp_path, "bank0", [0]),
        _make_bank_root(tmp_path, "bank1", [1]),
    ]
    out = tmp_path / "inventory"
    result = write_inventory(
        roots, out, canonical=False, verify_wav_headers=False
    )
    return roots, Path(result["manifest_path"])


def test_inventory_reconstructs_journals_without_generation(synthetic_inventory):
    roots, manifest_path = synthetic_inventory
    records, manifest = load_inventory(manifest_path)
    assert manifest["record_count"] == 20
    assert manifest["base_record_count"] == 4
    assert manifest["fork_record_count"] == 16
    assert {row["base_seed"] for row in records} == {0, 1}
    assert all(Path(row["audio_path"]).is_file() for row in records)
    before = {path: sha256_file(path) for root in roots for path in root.glob("**/*") if path.is_file()}
    inventory_b2_roots(roots, canonical=False, verify_wav_headers=False)
    after = {path: sha256_file(path) for root in roots for path in root.glob("**/*") if path.is_file()}
    assert before == after


def test_inventory_fails_on_corrupt_wav(synthetic_inventory):
    roots, _manifest_path = synthetic_inventory
    wav = next((roots[0] / "raw").glob("**/*.wav"))
    wav.write_bytes(wav.read_bytes() + b"corrupt")
    with pytest.raises(B2ClosureError, match="hash mismatch"):
        inventory_b2_roots(roots, canonical=False, verify_wav_headers=False)


def test_inventory_rejects_seed_leakage_between_roots(tmp_path):
    roots = [
        _make_bank_root(tmp_path, "left", [0]),
        _make_bank_root(tmp_path, "right", [0]),
    ]
    with pytest.raises(B2ClosureError, match="multiple roots"):
        inventory_b2_roots(roots, canonical=False, verify_wav_headers=False)


def _fake_audio_loader(records):
    return np.zeros((len(records), 4), dtype=np.float32)


def _fake_posterior(waveforms):
    output = np.full((len(waveforms), 527), 0.001, dtype=np.float32)
    output[:, 0] = 0.8
    output[:, 1] = 0.1
    return output


def _measure_two_shards(tmp_path, manifest_path, map_path, batch_sizes=(3, 3)):
    completions = []
    for index in range(2):
        result = measure_inventory_shard(
            manifest_path,
            tmp_path / f"shard{index}",
            protocol_path=PROTOCOL,
            canonical=False,
            shard_index=index,
            shard_count=2,
            coarse_map_path=map_path,
            posterior_fn=_fake_posterior,
            audio_loader=_fake_audio_loader,
            batch_size=batch_sizes[index],
            tagger_revision="synthetic-tagger",
            tagger_checkpoint_sha256="b" * 64,
            measurer_revision="c" * 40,
        )
        completions.append(Path(result["completion_path"]))
    return completions


def test_shard_schema_assignment_and_deterministic_npz(synthetic_inventory, tmp_path):
    _roots, manifest_path = synthetic_inventory
    map_path = _make_coarse_map(tmp_path / "coarse.json")
    completions = _measure_two_shards(tmp_path, manifest_path, map_path)
    records, _manifest = load_inventory(manifest_path)
    completion, arrays = validate_shard_completion(completions[0])
    expected = [row["record_id"] for row in assigned_inventory_records(records, 0, 2)]
    assert arrays["record_id"].tolist() == expected
    assert arrays["clipwise_output_527"].shape == (len(expected), 527)
    assert arrays["clipwise_output_527"].dtype == np.float32
    assert np.allclose(arrays["coarse_posterior"].sum(axis=1), 1.0)
    np.testing.assert_allclose(
        arrays["coarse_score_sums"].sum(axis=1),
        arrays["clipwise_output_527"].sum(axis=1),
        rtol=0.0,
        atol=1e-5,
    )
    assert arrays["confident_label"].tolist() == ["group_a"] * len(expected)
    assert completion["tagger_checkpoint_sha256"] == "b" * 64
    assert completion["protocol_sha256"] == sha256_file(PROTOCOL)
    assert str(arrays["protocol_sha256"].item()) == completion["protocol_sha256"]

    duplicate = measure_inventory_shard(
        manifest_path,
        tmp_path / "duplicate",
        protocol_path=PROTOCOL,
        canonical=False,
        shard_index=0,
        shard_count=2,
        coarse_map_path=map_path,
        posterior_fn=_fake_posterior,
        audio_loader=_fake_audio_loader,
        batch_size=4,
        tagger_revision="synthetic-tagger",
        tagger_checkpoint_sha256="b" * 64,
        measurer_revision="c" * 40,
    )
    first_data = completions[0].parent / completion["data_file"]
    duplicate_completion = json.loads(Path(duplicate["completion_path"]).read_text())
    duplicate_data = Path(duplicate["completion_path"]).parent / duplicate_completion["data_file"]
    assert sha256_file(first_data) == sha256_file(duplicate_data)


def test_deterministic_npz_is_create_only(tmp_path):
    path = tmp_path / "x.npz"
    deterministic_npz_create(path, {"x": np.arange(5, dtype=np.int16)})
    with pytest.raises(FileExistsError):
        deterministic_npz_create(path, {"x": np.arange(5, dtype=np.int16)})


def test_merge_is_fail_closed_for_partial_and_corrupt_shards(synthetic_inventory, tmp_path):
    _roots, manifest_path = synthetic_inventory
    map_path = _make_coarse_map(tmp_path / "coarse.json")
    completions = _measure_two_shards(tmp_path, manifest_path, map_path)
    partial_out = tmp_path / "partial"
    with pytest.raises(B2ClosureError, match="incomplete"):
        merge_posterior_shards(manifest_path, completions[:1], partial_out)
    assert not partial_out.exists()

    completion = json.loads(completions[1].read_text())
    data_path = completions[1].parent / completion["data_file"]
    damaged = bytearray(data_path.read_bytes())
    damaged[len(damaged) // 2] ^= 0x01
    data_path.write_bytes(damaged)
    with pytest.raises(B2ClosureError, match="hash mismatch"):
        merge_posterior_shards(manifest_path, completions, tmp_path / "corrupt")
    assert not (tmp_path / "corrupt").exists()


def test_complete_merge_has_exact_inventory_coverage(synthetic_inventory, tmp_path):
    _roots, manifest_path = synthetic_inventory
    map_path = _make_coarse_map(tmp_path / "coarse.json")
    completions = _measure_two_shards(tmp_path, manifest_path, map_path)
    result = merge_posterior_shards(manifest_path, completions, tmp_path / "merged")
    merged_completion = json.loads(Path(result["completion_path"]).read_text())
    arrays_path = Path(result["completion_path"]).parent / merged_completion["data_file"]
    with np.load(arrays_path, allow_pickle=False) as arrays:
        records, inventory = load_inventory(manifest_path)
        assert arrays["record_id"].tolist() == [row["record_id"] for row in records]
        assert result["record_ids_sha256"] == inventory["record_ids_sha256"]


def test_merge_retains_heterogeneous_batch_sizes_as_per_shard_provenance(
    synthetic_inventory, tmp_path
):
    _roots, manifest_path = synthetic_inventory
    map_path = _make_coarse_map(tmp_path / "coarse.json")
    completions = _measure_two_shards(
        tmp_path, manifest_path, map_path, batch_sizes=(3, 4)
    )
    result = merge_posterior_shards(
        manifest_path, completions, tmp_path / "merged_heterogeneous_batches"
    )
    assert [row["batch_size"] for row in result["input_shards"]] == [3, 4]
    assert "batch_size" not in result


def _posterior_for_label(label: str) -> np.ndarray:
    row = np.full(527, 0.001, dtype=np.float32)
    if label == "group_a":
        row[0], row[1] = 0.9, 0.1
    elif label == "group_b":
        row[0], row[1] = 0.1, 0.9
    elif label == "abstain":
        row[0], row[1] = 0.50, 0.48
    else:  # pragma: no cover
        raise ValueError(label)
    return row


def _analysis_arrays(tmp_path):
    map_path = _make_coarse_map(tmp_path / "analysis_map.json")
    coarse_map = load_coarse_map(map_path)
    records = []
    posterior = []
    videos = ["v0", "v1", "v2"]
    seeds = [0, 1, 2]
    base_labels = {
        "v0": ["group_a", "group_a", "group_a"],
        "v1": ["group_a", "group_b", "group_a"],
        "v2": ["group_b", "group_b", "group_a"],
    }

    def add(video, seed, role, progress, fork, label):
        record_id = (
            f"{video}__seed{seed}__base" if role == "base"
            else f"{video}__seed{seed}__s{progress:.2f}__fork{fork:02d}"
        )
        records.append(
            {
                "record_id": record_id,
                "video_id": video,
                "base_seed": seed,
                "role": role,
                "fork_index": fork,
                "progress": progress,
                "audio_path": f"/{record_id}.wav",
                "audio_sha256": hashlib.sha256(record_id.encode()).hexdigest(),
                "source_unit_journal_sha256": "1" * 64,
                "source_cell_journal_sha256": None if role == "base" else "2" * 64,
                "generation_manifest_sha256": "3" * 64,
                "generation_revision": "4" * 40,
                "generation_model_variant": "synthetic",
                "cfg": 4.5,
                "alpha": 0.8,
                "schedule": "sqrt_down",
                "sample_rate": 16000,
            }
        )
        posterior.append(_posterior_for_label(label))

    for video in videos:
        for seed, label in zip(seeds, base_labels[video]):
            add(video, seed, "base", None, None, label)
            low = ["group_a", "group_a", "group_b", "group_b"]
            high = (
                ["group_a"] * 4
                if not (video == "v2" and seed == 2)
                else ["group_a", "group_a", "group_b", "group_b"]
            )
            for progress, labels in ((0.25, low), (0.75, high)):
                for fork, fork_label in enumerate(labels):
                    add(video, seed, "fork", progress, fork, fork_label)
    order = np.argsort([row["record_id"] for row in records])
    ordered_records = [records[int(index)] for index in order]
    ordered_posterior = np.stack([posterior[int(index)] for index in order])
    arrays = build_posterior_arrays(
        ordered_records,
        ordered_posterior,
        coarse_map=coarse_map,
        coarse_map_sha256=sha256_file(map_path),
        tagger_revision="synthetic",
        tagger_checkpoint_sha256="5" * 64,
        measurer_revision="6" * 40,
        protocol_sha256=sha256_file(PROTOCOL),
    )
    return arrays


def test_video_conditioned_baseline_prevents_cross_video_leakage(tmp_path):
    arrays = _analysis_arrays(tmp_path)
    cells, baselines = build_commitment_cells(arrays)
    by_video = {row["video_id"]: row for row in baselines}
    assert by_video["v0"]["a_independent"] == pytest.approx(1.0)
    assert by_video["v0"]["video_determined"] is True
    assert by_video["v1"]["a_independent"] == pytest.approx(1.0 / 3.0)
    assert by_video["v2"]["a_independent"] == pytest.approx(1.0 / 3.0)
    assert all(
        row["a_independent"] == by_video[row["video_id"]]["a_independent"]
        for row in cells
    )


def test_analysis_is_deterministic_and_separates_crossing_cases(tmp_path):
    arrays = _analysis_arrays(tmp_path)
    first = analyze_multiseed(
        arrays,
        protocol_path=PROTOCOL,
        canonical=False,
        thresholds=(0.60, 0.70, 0.80),
        n_video_boot=30,
        n_fork_boot=20,
        seed=7,
    )
    second = analyze_multiseed(
        arrays,
        protocol_path=PROTOCOL,
        canonical=False,
        thresholds=(0.60, 0.70, 0.80),
        n_video_boot=30,
        n_fork_boot=20,
        seed=7,
    )
    assert first == second
    summary, cells, video_rows, _baselines, video_seed_rows = first
    assert summary["cardinality"] == {
        "videos": 3,
        "base_seeds": 3,
        "progress_points": 2,
        "video_seed_progress_cells": 18,
    }
    theta = summary["theta_0.70_summary"]
    assert theta["n_video_determined"] == 1
    assert theta["n_crossing"] == 5
    assert theta["n_noncrossing"] == 1
    assert {row["progress"] for row in cells} == {0.25, 0.75}
    assert any(row["n_noncrossing_seeds"] == 1 for row in video_rows)
    assert summary["variance_decomposition"]["measurer_repeatability_variance"] is None
    for progress_row in summary["variance_decomposition"]["by_progress"]:
        assert progress_row["video_cluster_bootstrap_draws"] == 30
        assert progress_row["component_video_bootstrap_ci"]
        assert all(
            {"ci_low", "ci_high", "valid_draws"}.issubset(interval)
            for interval in progress_row["component_video_bootstrap_ci"].values()
        )
    overall_ci = summary["variance_decomposition"][
        "overall_mean_component_video_bootstrap_ci"
    ]
    assert set(overall_ci) == {
        "video",
        "base_seed",
        "video_by_seed_interaction",
        "fork_monte_carlo_nonabstention",
        "abstention_within_fork_monte_carlo",
    }
    assert all(interval["valid_draws"] == 30 for interval in overall_ci.values())
    assert len(video_seed_rows) == 3 * 3 * 3
    assert len(summary["base_seed_crossings"]) == 3 * 3
    assert summary["scientific_status"] in ALLOWED_SCIENTIFIC_STATUSES


def test_schema_validator_detects_posterior_corruption(tmp_path):
    arrays = _analysis_arrays(tmp_path)
    assert str(arrays["schema_version"].item()) == POSTERIOR_SCHEMA
    validate_posterior_arrays(arrays)
    corrupt = dict(arrays)
    corrupt["coarse_posterior"] = arrays["coarse_posterior"].copy()
    corrupt["coarse_posterior"][0, 0] += 0.2
    with pytest.raises(B2ClosureError, match="mass conservation"):
        validate_posterior_arrays(corrupt)


def test_frozen_protocol_defaults_and_pinned_asset_gate(tmp_path):
    protocol = validate_class_protocol(PROTOCOL, canonical=True)["payload"]
    class_protocol = protocol["class_measurement"]
    assert tuple(class_protocol["sensitivity_thresholds"]) == SENSITIVITY_THRESHOLDS
    assert class_protocol["bootstrap"] == {
        "unit": "video",
        "draws": DEFAULT_BOOTSTRAP_DRAWS,
        "seed": DEFAULT_BOOTSTRAP_SEED,
        "interval": [0.025, 0.975],
    }
    assert class_protocol["video_determined_if_baseline_gte"] == VIDEO_DETERMINED_MIN
    with pytest.raises(B2ClosureError, match="abstain delta"):
        validate_class_protocol(PROTOCOL, abstain_delta=0.051, canonical=True)
    bad_checkpoint = tmp_path / "bad.pth"
    bad_checkpoint.write_bytes(b"not the pinned checkpoint")
    with pytest.raises(B2ClosureError, match="checkpoint"):
        validate_class_protocol(
            PROTOCOL,
            checkpoint_path=bad_checkpoint,
            coarse_map_path=Path("configs/coarse_class_map.json"),
            canonical=True,
        )
    with pytest.raises(B2ClosureError, match="coarse map"):
        validate_class_protocol(
            PROTOCOL,
            coarse_map_path=_make_coarse_map(tmp_path / "wrong_map.json"),
            canonical=True,
        )


def test_completion_scalar_provenance_tamper_is_detected(synthetic_inventory, tmp_path):
    _roots, manifest_path = synthetic_inventory
    map_path = _make_coarse_map(tmp_path / "coarse.json")
    completion_path = _measure_two_shards(tmp_path, manifest_path, map_path)[0]
    completion = json.loads(completion_path.read_text())
    completion["tagger_revision"] = "tampered"
    completion_path.write_text(json.dumps(completion, sort_keys=True) + "\n")
    with pytest.raises(B2ClosureError, match="completion/NPZ provenance"):
        validate_shard_completion(completion_path)


def test_sustained_crossing_ignores_later_unscorable_and_majority_requires_two():
    curve = {0.05: 0.8, 0.15: float("nan"), 0.25: 0.9}
    assert _first_crossing(curve, 0.7, sustained=True) == pytest.approx(0.05)
    label, share = _majority_label(["group_a", "abstain"])
    assert label is None and np.isnan(share)
    assert _majority_label(["group_a", "group_a", "abstain"]) == (
        "group_a",
        1.0,
    )


def test_unscorable_partition_is_not_noncrossing_or_numeric_censoring():
    cells = [
        {"video_id": "v", "base_seed": 0, "progress": progress, "commitment_gain": float("nan")}
        for progress in (0.05, 0.15)
    ]
    baselines = [
        {
            "video_id": "v",
            "a_independent": 0.5,
            "n_confident_base_finals": 2,
            "base_abstention_rate": 0.0,
            "video_determined": False,
        }
    ]
    summaries, _videos, units = summarize_thresholds(
        cells, baselines, (0.7,), n_boot=10, seed=3
    )
    assert units[0]["status"] == "UNSCORABLE"
    assert summaries[0]["n_unscorable"] == 1
    assert summaries[0]["n_noncrossing"] == 0
    assert "censored_median" not in summaries[0]
    assert summaries[0]["noncrossers_are_right_censored_without_numeric_imputation"]


def test_pooled_and_all_17_seed_crossings_are_explicit():
    cells = []
    for video in ("v0", "v1"):
        for base_seed in range(17):
            for progress, gain in ((0.25, 0.2), (0.35, 0.8), (0.45, 0.9)):
                cells.append(
                    {
                        "video_id": video,
                        "base_seed": base_seed,
                        "progress": progress,
                        "commitment_gain": gain,
                    }
                )
    pooled, per_seed = pooled_and_seed_crossings(
        cells, (0.7,), n_boot=10, seed=DEFAULT_BOOTSTRAP_SEED
    )
    assert pooled[0]["sustained_crossing"] == pytest.approx(0.35)
    assert len(per_seed) == 17
    assert {row["base_seed"] for row in per_seed} == set(range(17))
    assert all(row["sustained_crossing"] == pytest.approx(0.35) for row in per_seed)


def test_frozen_four_way_replication_classifier():
    def pooled(value):
        return [
            {
                "theta_commit": 0.7,
                "sustained_crossing": value,
                "sustained_crossing_bootstrap_ci_low": 0.25,
                "sustained_crossing_bootstrap_ci_high": 0.45,
            }
        ]

    def seeds(n_early):
        return [
            {
                "theta_commit": 0.7,
                "sustained_crossing": 0.35 if index < n_early else 0.75,
            }
            for index in range(17)
        ]

    stable_variance = {
        "overall_mean_components": {
            "video": 1.0,
            "base_seed": 0.1,
            "video_by_seed_interaction": 0.2,
            "fork_monte_carlo_nonabstention": 0.1,
            "abstention_within_fork_monte_carlo": 0.0,
        }
    }
    strong_variance = {
        "overall_mean_components": {
            "video": 0.1,
            "base_seed": 0.2,
            "video_by_seed_interaction": 0.2,
            "fork_monte_carlo_nonabstention": 0.1,
            "abstention_within_fork_monte_carlo": 0.0,
        }
    }
    grid = (0.05, 0.15, 0.25, 0.35, 0.45, 0.6, 0.75, 0.9)
    assert classify_frozen_replication(
        pooled(0.35), seeds(17), stable_variance, progress_grid=grid
    )["replication_label"] == "stable_across_seeds"
    assert classify_frozen_replication(
        pooled(0.35), seeds(11), stable_variance, progress_grid=grid
    )["replication_label"] == "heterogeneous_but_directionally_consistent"
    assert classify_frozen_replication(
        pooled(0.35), seeds(17), strong_variance, progress_grid=grid
    )["replication_label"] == "strongly_seed_dependent"
    assert classify_frozen_replication(
        pooled(0.6), seeds(17), stable_variance, progress_grid=grid
    )["replication_label"] == "not_reproduced"


def test_canonical_analysis_requires_pinned_historical_comparator(tmp_path):
    arrays = _analysis_arrays(tmp_path)
    with pytest.raises(B2ClosureError, match="pinned WP-A2 Class comparator"):
        analyze_multiseed(
            arrays,
            protocol_path=PROTOCOL,
            canonical=True,
            historical_jsons=(),
        )


def test_production_analysis_rejects_noncanonical_population_before_output(tmp_path):
    arrays = _analysis_arrays(tmp_path)
    with pytest.raises(B2ClosureError, match="canonical_b2=true"):
        validate_canonical_analysis_inputs(
            arrays,
            {"canonical_b2": False},
            protocol_path=PROTOCOL,
            historical_jsons=[Path("results/arc4_wpA2/class_reconstruction.json")],
        )


def test_cuda_determinism_workspace_gate(monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    with pytest.raises(B2ClosureError, match="CUBLAS_WORKSPACE_CONFIG"):
        validate_cuda_determinism_environment("cuda:0")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    assert validate_cuda_determinism_environment("cuda:0") == ":4096:8"
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    assert validate_cuda_determinism_environment("cuda") == ":16:8"
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", "bad")
    with pytest.raises(B2ClosureError, match="observed 'bad'"):
        validate_cuda_determinism_environment("cuda:0")
    assert validate_cuda_determinism_environment("cpu") is None


def test_normalized_coarse_vector_is_reconstructible_from_persisted_raw_sums():
    rng = np.random.default_rng(20260717)
    probabilities = rng.uniform(1e-7, 1.0, size=(31, 527)).astype(np.float32)
    coarse_map = load_coarse_map(Path("configs/coarse_class_map.json"))
    raw_sums, normalized, _names = derive_coarse_scores(probabilities, coarse_map)
    reconstructed = raw_sums / raw_sums.sum(
        axis=1, keepdims=True, dtype=np.float32
    )
    assert np.array_equal(normalized, reconstructed)
