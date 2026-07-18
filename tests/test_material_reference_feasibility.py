"""CPU-only tests for the outcome-blind legacy Material reference gate."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from foley_cw.material_reference_feasibility import (
    EXPECTED_CELLS_PER_VIDEO,
    MATCHING_BANDS,
    MIN_VALID_CELLS,
    PROGRESS_POINTS,
    SCHEMA_VERSION,
    SUBJECT_INDICES,
    FeasibilityInputError,
    FinalInventory,
    LoudnessCollection,
    Phase2Inventory,
    build_reference_manifest,
    immutable_write,
    inventory_phase2_journals,
    scan_retained_finals,
    source_audio_loudness,
    stable_hash,
    validate_protocol,
    validate_reference_manifest,
    write_feasibility_outputs,
)


def _phase2_inventory() -> Phase2Inventory:
    cells = frozenset((j, s) for j in SUBJECT_INDICES for s in PROGRESS_POINTS)
    clips = tuple(str(index) for index in range(1, 201))
    return Phase2Inventory(
        clips=clips,
        cells_by_clip={clip: cells for clip in clips},
        journal_hashes={clip: stable_hash(f"inventory:{clip}") for clip in clips},
        aggregate_sha256="a" * 64,
        errors=(),
    )


def _valid_inputs():
    phase2 = _phase2_inventory()
    valid_clips = tuple(str(index) for index in range(1, 21))
    class_labels: dict[str, str] = {}
    material_ids: set[str] = set()
    metadata: dict[str, dict[str, str]] = {}
    timings: dict[str, dict[str, object]] = {}
    loudness_records: dict[str, dict[str, object]] = {}
    for index, clip in enumerate(phase2.clips, start=1):
        metadata[clip] = {
            "path": f"/retained/{clip}.mp4",
            "ucs_category": "METAL" if index <= 20 and index % 2 else (
                "WOOD" if index <= 20 else "FIGHT"
            ),
            "audioset_category": "Impact",
            "source_type": "Single-source",
            "discrete_vs_rest": "Discrete",
            "status": "ok",
        }
        if clip not in valid_clips:
            continue
        timings[clip] = {
            "timing_s": 1.0 if index % 2 else 1.1,
            "source": "foleybench_audio_onset",
        }
        loudness_records[clip] = {
            "rms_dbfs": -20.0 if index % 2 else -21.0,
            "source_video_sha256": stable_hash(f"video:{clip}"),
        }
        # ind0..3 are candidate subjects; ind4 makes every positive disjoint.
        for j in range(5):
            gen_id = f"{clip}__p1cfg1_ind{j}"
            class_labels[gen_id] = "impacts"
            material_ids.add(gen_id)
    finals = FinalInventory(
        class_labels=class_labels,
        material_final_ids=frozenset(material_ids),
        measurement_sha256="b" * 64,
        counters={"class_final_ids": len(class_labels), "material_final_ids": len(material_ids)},
        errors=(),
    )
    loudness = LoudnessCollection(
        records=loudness_records,
        failures={},
        ffmpeg={"resolved_path": "/usr/bin/ffmpeg", "executable_sha256": "c" * 64},
    )
    provenance = {
        "metadata_input_errors": [],
        "timing_input_errors": [],
        "metadata_csv_sha256": "d" * 64,
        "anchors_json_sha256": "e" * 64,
    }
    return phase2, finals, metadata, timings, loudness, provenance


def _build_valid():
    phase2, finals, metadata, timings, loudness, provenance = _valid_inputs()
    return build_reference_manifest(
        phase2=phase2,
        finals=finals,
        metadata=metadata,
        timings=timings,
        loudness=loudness,
        source_provenance=provenance,
        protocol_sha256="f" * 64,
    )


def _journal_payload(clip: str, correct: float) -> dict:
    rows = []
    for j in SUBJECT_INDICES:
        for s in PROGRESS_POINTS:
            # Insertion order intentionally matches the immutable historical
            # journal writer.  The feasibility scanner skips ``correct``.
            rows.append({
                "axis_id": "material",
                "clip": clip,
                "correct": correct,
                "j": j,
                "probe": "audio_tagger",
                "s": s,
                "target": "ode",
            })
    return {"clip": clip, "rows": rows}


def _write_journal_set(root: Path, correct: float, n: int = 200) -> None:
    root.mkdir(parents=True)
    for index in range(1, n + 1):
        clip = str(index)
        (root / f"p2cfg1__{clip}.json").write_text(
            json.dumps(_journal_payload(clip, correct)), encoding="utf-8"
        )


class TestOutcomeBlindInputs:
    def test_phase2_inventory_does_not_depend_on_legacy_cosine(self, tmp_path: Path):
        left = tmp_path / "left"
        right = tmp_path / "right"
        _write_journal_set(left, correct=-12345.0)
        _write_journal_set(right, correct=0.999999)

        inv_left = inventory_phase2_journals(left)
        inv_right = inventory_phase2_journals(right)

        assert not inv_left.errors
        assert not inv_right.errors
        assert inv_left.cells_by_clip == inv_right.cells_by_clip
        assert inv_left.journal_hashes == inv_right.journal_hashes
        assert inv_left.aggregate_sha256 == inv_right.aggregate_sha256
        assert sum(map(len, inv_left.cells_by_clip.values())) == 6400

    def test_partial_journal_set_is_evidenced_not_synthesized(self, tmp_path: Path):
        root = tmp_path / "partial"
        _write_journal_set(root, correct=0.5, n=199)
        inventory = inventory_phase2_journals(root)
        assert any("journal count 199" in error for error in inventory.errors)
        assert sum(map(len, inventory.cells_by_clip.values())) == 199 * EXPECTED_CELLS_PER_VIDEO

    def test_material_embeddings_are_never_returned_or_decoded(self, tmp_path: Path):
        def write(path: Path, sentinel: str) -> None:
            gen_id = "1__p1cfg1_ind0"
            class_row = {
                "gen_id": gen_id,
                "axis_id": "class",
                "target": {"axis_id": "class", "kind": "categorical", "label": "impacts", "embedding": None},
                "extra": {"role": "p1cfg1_independent", "j": 0, "clip": "1", "cfg": 1.0},
            }
            # A string sentinel makes accidental numerical embedding access fail
            # while preserving a syntactically valid historical Material row.
            material_row = {
                "gen_id": gen_id,
                "axis_id": "material",
                "target": {"axis_id": "material", "kind": "embedding", "label": None, "embedding": [sentinel]},
                "extra": {"role": "p1cfg1_independent", "j": 0, "clip": "1", "cfg": 1.0},
            }
            path.write_text(json.dumps(class_row) + "\n" + json.dumps(material_row) + "\n")

        left = tmp_path / "left.jsonl"
        right = tmp_path / "right.jsonl"
        write(left, "DO_NOT_INSPECT_A")
        write(right, "DO_NOT_INSPECT_B")
        a = scan_retained_finals(left)
        b = scan_retained_finals(right)
        assert a.class_labels == b.class_labels == {"1__p1cfg1_ind0": "impacts"}
        assert a.material_final_ids == b.material_final_ids == {"1__p1cfg1_ind0"}
        assert a.measurement_sha256 != b.measurement_sha256  # exact source provenance only
        assert not hasattr(a, "embeddings")


class TestSourceLoudness:
    def test_pinned_decode_command_and_rms(self, tmp_path: Path):
        video = tmp_path / "clip.mp4"
        video.write_bytes(b"retained-original-mp4")
        samples = np.asarray([0.5, -0.5, 0.5, -0.5], dtype="<f4")
        seen: list[list[str]] = []

        def runner(command, **kwargs):
            seen.append(command)
            return subprocess.CompletedProcess(command, 0, stdout=samples.tobytes(), stderr=b"")

        record = source_audio_loudness(
            video,
            {"resolved_path": "/pinned/ffmpeg"},
            runner=runner,
        )
        assert record["sample_rate_hz"] == 16000
        assert record["channels"] == 1
        assert record["n_samples"] == 4
        assert record["rms_dbfs"] == pytest.approx(20 * np.log10(0.5))
        assert seen[0][-1] == "pipe:1"
        assert seen[0][seen[0].index("-map") + 1] == "0:a:0"
        assert record["decode_command_sha256"] == stable_hash("\0".join(seen[0]))

    def test_empty_audio_is_a_blocker(self, tmp_path: Path):
        video = tmp_path / "clip.mp4"
        video.write_bytes(b"retained-original-mp4")

        def runner(command, **kwargs):
            return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")

        with pytest.raises(RuntimeError, match="byte count"):
            source_audio_loudness(video, {"resolved_path": "/pinned/ffmpeg"}, runner=runner)


class TestStrictMatching:
    def test_sufficient_strict_match_freezes_exactly_640_trials(self):
        report, manifest = _build_valid()
        assert report["status"] == "READY_TO_FREEZE"
        assert manifest is not None
        assert manifest["schema"] == SCHEMA_VERSION
        assert manifest["coverage"]["complete_candidate_videos"] == 20
        assert len(manifest["trials"]) == MIN_VALID_CELLS
        assert set(manifest["coverage"]["valid_videos_by_progress"].values()) == {20}
        for trial in manifest["trials"]:
            assert trial["candidate"]["video_id"] == trial["positive"]["video_id"]
            assert trial["candidate"]["video_id"] != trial["negative"]["video_id"]
            assert trial["candidate"]["retained_final_generation_id"] != trial["positive"]["generation_id"]
            assert trial["candidate"]["coarse_class"] == trial["negative"]["coarse_class"]
            assert trial["matching"]["candidate_material"] != trial["matching"]["negative_material"]
            assert trial["matching"]["difficulty"] == "hard"
        validate_reference_manifest(manifest)

    def test_deterministic_under_mapping_insertion_order(self):
        phase2, finals, metadata, timings, loudness, provenance = _valid_inputs()
        first_report, first_manifest = build_reference_manifest(
            phase2=phase2,
            finals=finals,
            metadata=metadata,
            timings=timings,
            loudness=loudness,
            source_provenance=provenance,
            protocol_sha256="f" * 64,
        )
        reversed_finals = FinalInventory(
            class_labels=dict(reversed(list(finals.class_labels.items()))),
            material_final_ids=frozenset(reversed(sorted(finals.material_final_ids))),
            measurement_sha256=finals.measurement_sha256,
            counters=finals.counters,
            errors=finals.errors,
        )
        reversed_loudness = LoudnessCollection(
            records=dict(reversed(list(loudness.records.items()))),
            failures={},
            ffmpeg=loudness.ffmpeg,
        )
        second_report, second_manifest = build_reference_manifest(
            phase2=phase2,
            finals=reversed_finals,
            metadata=dict(reversed(list(metadata.items()))),
            timings=dict(reversed(list(timings.items()))),
            loudness=reversed_loudness,
            source_provenance=dict(reversed(list(provenance.items()))),
            protocol_sha256="f" * 64,
        )
        assert first_report == second_report
        assert first_manifest == second_manifest

    def test_lowercase_ucs_and_caption_cannot_create_material_truth(self):
        phase2, finals, metadata, timings, loudness, provenance = _valid_inputs()
        metadata["1"]["ucs_category"] = "metal"  # exact/case-sensitive rule
        metadata["1"]["caption"] = "obviously a METAL impact"  # must be ignored
        report, manifest = build_reference_manifest(
            phase2=phase2,
            finals=finals,
            metadata=metadata,
            timings=timings,
            loudness=loudness,
            source_provenance=provenance,
            protocol_sha256="f" * 64,
        )
        assert manifest is None
        assert report["status"] == "INCOMPLETE_ARTIFACTS"
        assert report["coverage"]["complete_candidate_videos"] == 19
        # 180 intentionally ineligible legacy videos contribute 720 exclusions;
        # the lowercase row adds exactly four more and captions cannot rescue it.
        assert report["exclusion_reason_counts"]["UCS_MATERIAL_NOT_ADMISSIBLE"] == 724
        assert all(
            report["subject_exclusions"][f"1:j{j}"] == ["UCS_MATERIAL_NOT_ADMISSIBLE"]
            for j in SUBJECT_INDICES
        )

    def test_missing_loudness_is_reported_and_never_imputed(self):
        phase2, finals, metadata, timings, loudness, provenance = _valid_inputs()
        records = dict(loudness.records)
        records.pop("1")
        blocked = LoudnessCollection(
            records=records,
            failures={"1": "SOURCE_LOUDNESS_UNAVAILABLE: no audio stream"},
            ffmpeg=loudness.ffmpeg,
        )
        report, manifest = build_reference_manifest(
            phase2=phase2,
            finals=finals,
            metadata=metadata,
            timings=timings,
            loudness=blocked,
            source_provenance=provenance,
            protocol_sha256="f" * 64,
        )
        assert manifest is None
        assert report["coverage"]["complete_candidate_videos"] == 19
        assert report["exclusion_reason_counts"]["SOURCE_LOUDNESS_UNAVAILABLE"] == 4
        assert "minimum_valid_cells" in report["blocker_summary"]


class TestSchemaLeakageAndImmutability:
    def test_checked_in_protocol_matches_implementation(self):
        protocol_path = Path("experiment/non_human_closure/PROTOCOL.json")
        validate_protocol(json.loads(protocol_path.read_text(encoding="utf-8")))
        assert list(MATCHING_BANDS) == json.loads(protocol_path.read_text())[
            "material_continuity"
        ]["matching_bands"]

    def test_json_schema_and_semantic_validator(self):
        jsonschema = pytest.importorskip("jsonschema")
        _, manifest = _build_valid()
        assert manifest is not None
        schema = json.loads(Path(
            "experiment/non_human_closure/material_reference_manifest.schema.json"
        ).read_text(encoding="utf-8"))
        jsonschema.Draft7Validator.check_schema(schema)
        jsonschema.validate(manifest, schema)

    @pytest.mark.parametrize("leak_key", [
        "embedding", "positive_similarity", "negative_similarity",
        "similarity_margin", "cosine", "observed_choice", "correct",
    ])
    def test_outcome_or_embedding_field_is_rejected(self, leak_key: str):
        _, manifest = _build_valid()
        assert manifest is not None
        manifest["trials"][0]["matching"][leak_key] = 0.123
        with pytest.raises(FeasibilityInputError, match="forbidden"):
            validate_reference_manifest(manifest)

    def test_corrupt_existing_immutable_output_is_never_overwritten(self, tmp_path: Path):
        target = tmp_path / "frozen.json"
        first_hash = immutable_write(target, {"a": 1})
        assert immutable_write(target, {"a": 1}) == first_hash
        with pytest.raises(FileExistsError, match="refusing to overwrite"):
            immutable_write(target, {"a": 2})
        assert json.loads(target.read_text()) == {"a": 1}

    def test_insufficiency_output_contains_no_manifest(self, tmp_path: Path):
        report, _ = _build_valid()
        report = dict(report)
        report["status"] = "INCOMPLETE_ARTIFACTS"
        loudness = LoudnessCollection(records={}, failures={"1": "blocked"}, ffmpeg={})
        outputs = write_feasibility_outputs(tmp_path, report, None, loudness)
        assert "insufficiency" in outputs
        assert not (tmp_path / "MATERIAL_REFERENCE_MANIFEST.json").exists()
        payload = json.loads(Path(outputs["insufficiency"]).read_text())
        assert payload["status"] == "INCOMPLETE_ARTIFACTS"
