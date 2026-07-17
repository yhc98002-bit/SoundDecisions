from __future__ import annotations

import json
from pathlib import Path

import pytest

import foley_cw.b2_inventory_merge as merger
from foley_cw.b2_class_closure import (
    B2ClosureError,
    INVENTORY_SCHEMA,
    canonical_json_bytes,
    load_inventory,
    sha256_bytes,
    sha256_file,
)


TEST_VIDEOS = ("1002", "2001")
TEST_GRID = (0.25, 0.75)
TEST_SERIES = ((0,), (1,), (2,), (3,))
TEST_K = 2


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(payload))


def _row(
    root: Path,
    video: str,
    seed: int,
    role: str,
    revision: str,
    manifest_sha: str,
    *,
    progress: float | None = None,
    fork: int | None = None,
) -> dict[str, object]:
    unit = root / "journal" / "units" / f"{video}__seed{seed}.json"
    unit_sha = sha256_bytes(f"unit:{video}:{seed}".encode())
    if role == "base":
        record_id = f"{video}__seed{seed}__base"
        audio = root / "raw" / video / f"seed{seed}" / "base.wav"
        cell = None
        cell_sha = None
    else:
        assert progress is not None and fork is not None
        record_id = f"{video}__seed{seed}__s{progress:.2f}__fork{fork:02d}"
        audio = root / "raw" / video / f"seed{seed}" / f"s{progress:.2f}" / f"fork{fork:02d}.wav"
        cell = root / "journal" / "cells" / f"{video}__seed{seed}__s{progress:.2f}.json"
        cell_sha = sha256_bytes(f"cell:{video}:{seed}:{progress:.2f}".encode())
    return {
        "record_id": record_id,
        "video_id": video,
        "base_seed": seed,
        "role": role,
        "fork_index": fork,
        "progress": progress,
        "audio_path": str(audio),
        "audio_sha256": sha256_bytes(f"audio:{record_id}".encode()),
        "audio_bytes": 512080,
        "sample_rate": 16000,
        "frames": 128000,
        "audio_subtype": "FLOAT",
        "source_root": str(root),
        "generation_manifest_sha256": manifest_sha,
        "source_unit_journal": str(unit),
        "source_unit_journal_sha256": unit_sha,
        "source_cell_journal": None if cell is None else str(cell),
        "source_cell_journal_sha256": cell_sha,
        "cfg": 4.5,
        "alpha": 0.8,
        "schedule": "sqrt_down",
        "generation_model_variant": "small_16k",
        "generation_revision": revision,
    }


def _make_parts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    manifests: list[Path] = []
    lineage: dict[str, dict[str, object]] = {}
    for part_index, seeds in enumerate(TEST_SERIES):
        root = (tmp_path / f"bank{part_index}").resolve()
        generation_manifest = root / "generation_manifest.json"
        _write_json(generation_manifest, {"test_bank": part_index, "base_seeds": list(seeds)})
        manifest_sha = sha256_file(generation_manifest)
        revision = f"{part_index + 1:040x}"
        lineage[manifest_sha] = {
            "base_seeds": seeds,
            "generation_revision": revision,
        }
        rows = []
        for video in TEST_VIDEOS:
            for seed in seeds:
                rows.append(_row(root, video, seed, "base", revision, manifest_sha))
                for progress in TEST_GRID:
                    for fork in range(TEST_K):
                        rows.append(
                            _row(
                                root,
                                video,
                                seed,
                                "fork",
                                revision,
                                manifest_sha,
                                progress=progress,
                                fork=fork,
                            )
                        )
        rows.sort(key=lambda row: row["record_id"])
        part_dir = tmp_path / f"part{part_index}"
        records_path = part_dir / "B2_WAV_INVENTORY.jsonl"
        records_path.parent.mkdir(parents=True)
        records_path.write_bytes(
            b"".join(canonical_json_bytes(row, indent=None) for row in rows)
        )
        ids = [str(row["record_id"]) for row in rows]
        expected_base = len(TEST_VIDEOS) * len(seeds)
        expected_fork = expected_base * len(TEST_GRID) * TEST_K
        root_row = {
            "root": str(root),
            "generation_manifest": str(generation_manifest),
            "generation_manifest_sha256": manifest_sha,
            "base_seeds": list(seeds),
            "records": len(rows),
            "unit_journals": expected_base,
            "cell_journals": expected_base * len(TEST_GRID),
            "base_wavs": expected_base,
            "fork_wavs": expected_fork,
            "journal_hash_set_sha256": sha256_bytes(f"journals:{part_index}".encode()),
        }
        manifest = {
            "schema_version": INVENTORY_SCHEMA,
            "status": "COMPLETE",
            "read_only_inventory": True,
            "canonical_b2": False,
            "verify_wav_headers": True,
            "roots": [root_row],
            "record_count": len(rows),
            "base_record_count": expected_base,
            "fork_record_count": expected_fork,
            "video_ids": list(TEST_VIDEOS),
            "base_seeds": list(seeds),
            "progress_grid": list(TEST_GRID),
            "k_forks": TEST_K,
            "record_ids_sha256": sha256_bytes(("\n".join(ids) + "\n").encode()),
            "records_file": records_path.name,
            "records_sha256": sha256_file(records_path),
            "records_bytes": records_path.stat().st_size,
            "inventory_code_sha256": "a" * 64,
            "inventory_git_commit": "b" * 40,
        }
        manifest_path = part_dir / "B2_WAV_INVENTORY_MANIFEST.json"
        _write_json(manifest_path, manifest)
        manifests.append(manifest_path)
    monkeypatch.setattr(merger, "PINNED_GENERATION_LINEAGE", lineage)
    return manifests


@pytest.fixture(autouse=True)
def _small_frozen_design(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(merger, "EXPECTED_N_CLIPS", len(TEST_VIDEOS))
    monkeypatch.setattr(merger, "EXPECTED_SEEDS", tuple(range(4)))
    monkeypatch.setattr(merger, "EXPECTED_SEED_SERIES", TEST_SERIES)
    monkeypatch.setattr(merger, "EXPECTED_S_GRID", TEST_GRID)
    monkeypatch.setattr(merger, "EXPECTED_K_FORKS", TEST_K)


def _refresh_records_manifest(manifest_path: Path, rows: list[dict[str, object]]) -> None:
    manifest = json.loads(manifest_path.read_text())
    records_path = manifest_path.parent / manifest["records_file"]
    rows.sort(key=lambda row: row["record_id"])
    records_path.write_bytes(
        b"".join(canonical_json_bytes(row, indent=None) for row in rows)
    )
    ids = [str(row["record_id"]) for row in rows]
    manifest["records_sha256"] = sha256_file(records_path)
    manifest["records_bytes"] = records_path.stat().st_size
    manifest["record_ids_sha256"] = sha256_bytes(("\n".join(ids) + "\n").encode())
    _write_json(manifest_path, manifest)


def test_merge_is_deterministic_create_only_and_consumer_compatible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifests = _make_parts(tmp_path, monkeypatch)
    first = merger.merge_partial_inventories(manifests[::-1], tmp_path / "merged-a")
    second = merger.merge_partial_inventories(manifests, tmp_path / "merged-b")

    first_path = Path(first["manifest_path"])
    second_path = Path(second["manifest_path"])
    records, manifest = load_inventory(first_path)
    assert manifest["canonical_b2"] is True
    assert manifest["wav_rehash_during_merge"] is False
    assert manifest["source_part_count"] == 4
    assert len(records) == len(TEST_VIDEOS) * 4 * (1 + len(TEST_GRID) * TEST_K)
    assert first["records_sha256"] == second["records_sha256"]
    assert sha256_file(first_path) == sha256_file(second_path)
    with pytest.raises(B2ClosureError, match="create-only"):
        merger.merge_partial_inventories(manifests, tmp_path / "merged-a")


def test_merge_rejects_partial_jsonl_hash_corruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifests = _make_parts(tmp_path, monkeypatch)
    payload = json.loads(manifests[0].read_text())
    records_path = manifests[0].parent / payload["records_file"]
    records_path.write_bytes(records_path.read_bytes() + b" ")
    with pytest.raises(B2ClosureError, match="byte-count mismatch|hash mismatch"):
        merger.merge_partial_inventories(manifests, tmp_path / "merged")


def test_merge_rejects_metadata_tamper_even_with_refreshed_file_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifests = _make_parts(tmp_path, monkeypatch)
    payload = json.loads(manifests[2].read_text())
    records_path = manifests[2].parent / payload["records_file"]
    rows = [json.loads(line) for line in records_path.read_text().splitlines()]
    rows[0]["sample_rate"] = 44100
    _refresh_records_manifest(manifests[2], rows)
    with pytest.raises(B2ClosureError, match="sample_rate mismatch"):
        merger.merge_partial_inventories(manifests, tmp_path / "merged")


def test_merge_rejects_seed_lineage_reassignment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifests = _make_parts(tmp_path, monkeypatch)
    payload = json.loads(manifests[1].read_text())
    payload["base_seeds"] = [0]
    payload["roots"][0]["base_seeds"] = [0]
    _write_json(manifests[1], payload)
    with pytest.raises(B2ClosureError, match="unexpected seed series"):
        merger.merge_partial_inventories(manifests, tmp_path / "merged")
