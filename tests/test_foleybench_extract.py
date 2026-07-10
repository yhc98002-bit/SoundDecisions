"""Tests for foley_cw/foleybench_extract.py — FoleyBench extraction + selection.

CPU-only, NO network, NO GPU, NO real FoleyBench data: a tiny synthetic parquet
dataset (two shards: one binary-typed, one base64-string-typed video_data
column) plus a matching metadata CSV are built in tmp_path.

Key contracts checked here:
  * decode_video_bytes: raw-bytes and base64-str paths, ftyp magic validation,
    corrupt payloads raise VideoDecodeError.
  * extract_clips: atomic writes, skip-if-exists on rerun (no duplicate index
    rows), only_discrete / keys filters, decode errors recorded with empty path.
  * validate_clips_av: PyAV second pass flips corrupt files to decode_error and
    reports ok_rate; the CLI halts with exit 2 when ok_rate < 0.95.
  * select_stage_m_clips / select_screening_pool: deterministic seeded
    stratified selection over the Single-source+Discrete+ok pool, exclusions
    respected.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from foley_cw.foleybench_extract import (  # noqa: E402
    ExtractionReport,
    INDEX_COLUMNS,
    VideoDecodeError,
    decode_video_bytes,
    extract_clips,
    iter_foleybench_rows,
    load_metadata_csv,
    select_screening_pool,
    select_stage_m_clips,
    validate_clips_av,
    write_manifest_json,
)

# --------------------------------------------------------------------------------------
# Synthetic dataset construction
# --------------------------------------------------------------------------------------

#: Minimal payload passing the MP4 magic check (b'ftyp' at offset 4) + filler.
FAKE_MP4: bytes = b"\x00\x00\x00\x18ftypisom" + b"\x00" * 64

#: Payload failing the magic check.
CORRUPT_PAYLOAD: bytes = b"this is definitely not an mp4 container"

_CATEGORIES = ["CAT_A", "CAT_B", "CAT_C", "CAT_D", "CAT_E", "CAT_F"]

_CSV_COLUMNS = [
    "key", "duration", "dataset", "width", "height", "caption",
    "discrete_vs_rest", "source_type", "sound_type", "ucs_category",
    "audioset_category", "metadata",
]


def _meta_row(key: str, discrete: str, source: str, ucs: str) -> dict:
    return {
        "key": key,
        "duration": "10",
        "dataset": "testset",
        "width": "640.0",
        "height": "360.0",
        "caption": f"caption for {key}, with a comma",
        "discrete_vs_rest": discrete,
        "source_type": source,
        "sound_type": "Action",
        "ucs_category": ucs,
        "audioset_category": "Sounds of things",
        "metadata": "{'yt-id': 'x', 'start_time': 0, 'end_time': 10}",
    }


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


@pytest.fixture
def dataset(tmp_path: Path) -> dict:
    """Build a 32-row two-shard parquet dataset + metadata CSV in tmp_path.

    Shard 0 (binary video_data): rawvalid, rawcorrupt.
    Shard 1 (base64-str video_data): b64valid, b64corrupt, badb64, multisrc,
    restrow, nometa, s001..s024 (selection pool, 6 ucs categories x 4).

    Expected extract_clips(only_discrete=True) outcome:
      extracted   27 (rawvalid, b64valid, multisrc, s001..s024)
      decode_err   3 (rawcorrupt, b64corrupt, badb64)
      filtered     2 (restrow: Rest; nometa: no metadata row)
    """
    parquet_dir = tmp_path / "parquet"
    parquet_dir.mkdir()

    shard0 = pa.table({
        "key": pa.array(["rawvalid", "rawcorrupt"], type=pa.string()),
        "video_data": pa.array([FAKE_MP4, CORRUPT_PAYLOAD], type=pa.binary()),
    })
    pq.write_table(shard0, parquet_dir / "train-0-of-2.parquet")

    sel_keys = [f"s{i:03d}" for i in range(1, 25)]
    str_keys = ["b64valid", "b64corrupt", "badb64", "multisrc", "restrow", "nometa"]
    str_payloads = [
        _b64(FAKE_MP4),
        _b64(CORRUPT_PAYLOAD),       # decodes as base64, fails magic
        "!!!not-base64-at-all!!!",   # fails base64 decoding
        _b64(FAKE_MP4),
        _b64(FAKE_MP4),
        _b64(FAKE_MP4),
    ]
    str_keys += sel_keys
    str_payloads += [_b64(FAKE_MP4)] * len(sel_keys)
    shard1 = pa.table({
        "key": pa.array(str_keys, type=pa.string()),
        "video_data": pa.array(str_payloads, type=pa.string()),
    })
    pq.write_table(shard1, parquet_dir / "train-1-of-2.parquet")

    csv_path = tmp_path / "foleybench.csv"
    meta_rows = [
        _meta_row("rawvalid", "Discrete", "Single-source", "CAT_A"),
        _meta_row("rawcorrupt", "Discrete", "Single-source", "CAT_B"),
        _meta_row("b64valid", "Discrete", "Single-source", "CAT_B"),
        _meta_row("b64corrupt", "Discrete", "Single-source", "CAT_C"),
        _meta_row("badb64", "Discrete", "Single-source", "CAT_D"),
        _meta_row("multisrc", "Discrete", "Multi-source", "CAT_A"),
        _meta_row("restrow", "Rest", "Single-source", "CAT_A"),
        # "nometa" intentionally has NO metadata row.
    ]
    for i, key in enumerate(sel_keys):
        meta_rows.append(
            _meta_row(key, "Discrete", "Single-source", _CATEGORIES[i % len(_CATEGORIES)])
        )
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(meta_rows)

    return {
        "parquet_dir": parquet_dir,
        "csv_path": csv_path,
        "out_dir": tmp_path / "clips",
        "index_csv": tmp_path / "clips_index.csv",
        "n_rows": 32,
        "sel_keys": sel_keys,
    }


@pytest.fixture
def extracted(dataset: dict) -> dict:
    """Dataset after one extract_clips pass (only_discrete=True)."""
    report = extract_clips(
        parquet_dir=dataset["parquet_dir"],
        csv_path=dataset["csv_path"],
        out_dir=dataset["out_dir"],
        index_csv=dataset["index_csv"],
        only_discrete=True,
    )
    return {**dataset, "report": report}


def _read_index(index_csv: Path) -> dict[str, dict]:
    with open(index_csv, "r", encoding="utf-8", newline="") as fh:
        return {row["key"]: row for row in csv.DictReader(fh)}


# --------------------------------------------------------------------------------------
# decode_video_bytes
# --------------------------------------------------------------------------------------

class TestDecodeVideoBytes:
    def test_raw_bytes_valid_magic(self):
        assert decode_video_bytes(FAKE_MP4) == FAKE_MP4

    def test_bytearray_accepted(self):
        assert decode_video_bytes(bytearray(FAKE_MP4)) == FAKE_MP4

    def test_base64_str_valid(self):
        assert decode_video_bytes(_b64(FAKE_MP4)) == FAKE_MP4

    def test_raw_bytes_bad_magic_raises(self):
        with pytest.raises(VideoDecodeError):
            decode_video_bytes(CORRUPT_PAYLOAD)

    def test_base64_of_bad_magic_raises(self):
        with pytest.raises(VideoDecodeError):
            decode_video_bytes(_b64(CORRUPT_PAYLOAD))

    def test_invalid_base64_str_raises(self):
        with pytest.raises(VideoDecodeError):
            decode_video_bytes("!!!not-base64-at-all!!!")

    def test_too_short_payload_raises(self):
        with pytest.raises(VideoDecodeError):
            decode_video_bytes(b"\x00\x00")

    def test_unsupported_type_raises(self):
        with pytest.raises(VideoDecodeError):
            decode_video_bytes(12345)


# --------------------------------------------------------------------------------------
# iter_foleybench_rows / load_metadata_csv
# --------------------------------------------------------------------------------------

class TestStreaming:
    def test_yields_all_rows_with_requested_columns(self, dataset):
        rows = list(iter_foleybench_rows(
            dataset["parquet_dir"], columns=["key", "video_data"], batch_size=2,
        ))
        assert len(rows) == dataset["n_rows"]
        assert all(set(r.keys()) == {"key", "video_data"} for r in rows)

    def test_binary_column_yields_bytes(self, dataset):
        rows = list(iter_foleybench_rows(
            dataset["parquet_dir"], columns=["key", "video_data"], batch_size=4,
        ))
        by_key = {r["key"]: r["video_data"] for r in rows}
        assert isinstance(by_key["rawvalid"], bytes)
        assert isinstance(by_key["b64valid"], str)

    def test_missing_shards_raise(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            next(iter_foleybench_rows(tmp_path, columns=["key"]))

    def test_load_metadata_csv_keyed_by_key(self, dataset):
        meta = load_metadata_csv(dataset["csv_path"])
        assert "rawvalid" in meta
        assert meta["rawvalid"]["discrete_vs_rest"] == "Discrete"
        assert meta["restrow"]["discrete_vs_rest"] == "Rest"
        assert "nometa" not in meta


# --------------------------------------------------------------------------------------
# extract_clips
# --------------------------------------------------------------------------------------

class TestExtractClips:
    def test_report_counts(self, extracted):
        report: ExtractionReport = extracted["report"]
        assert report.n_seen == 32
        assert report.n_extracted == 27
        assert report.n_decode_error == 3
        assert report.n_filtered_out == 2
        assert report.n_skipped_existing == 0

    def test_clip_files_written(self, extracted):
        out_dir = extracted["out_dir"]
        assert (out_dir / "rawvalid.mp4").exists()
        assert (out_dir / "b64valid.mp4").exists()
        assert (out_dir / "b64valid.mp4").read_bytes() == FAKE_MP4
        assert not (out_dir / "rawcorrupt.mp4").exists()
        assert not (out_dir / "restrow.mp4").exists()

    def test_no_tmp_files_left(self, extracted):
        assert list(extracted["out_dir"].glob("*.tmp")) == []

    def test_index_rows_and_columns(self, extracted):
        index = _read_index(extracted["index_csv"])
        assert len(index) == 30  # 27 ok + 3 decode_error
        assert set(index["b64valid"].keys()) == set(INDEX_COLUMNS)
        assert index["b64valid"]["status"] == "ok"
        assert index["b64valid"]["ucs_category"] == "CAT_B"
        assert index["b64valid"]["caption"] == "caption for b64valid, with a comma"

    def test_decode_error_row_has_empty_path(self, extracted):
        index = _read_index(extracted["index_csv"])
        for key in ("rawcorrupt", "b64corrupt", "badb64"):
            assert index[key]["status"] == "decode_error"
            assert index[key]["path"] == ""
            assert index[key]["sha256"] == ""

    def test_sha256_matches_payload(self, extracted):
        index = _read_index(extracted["index_csv"])
        expected = hashlib.sha256(FAKE_MP4).hexdigest()
        assert index["b64valid"]["sha256"] == expected
        assert index["rawvalid"]["sha256"] == expected

    def test_rerun_skips_existing_no_duplicate_rows(self, extracted):
        report2 = extract_clips(
            parquet_dir=extracted["parquet_dir"],
            csv_path=extracted["csv_path"],
            out_dir=extracted["out_dir"],
            index_csv=extracted["index_csv"],
            only_discrete=True,
        )
        assert report2.n_skipped_existing == 27
        assert report2.n_extracted == 0
        with open(extracted["index_csv"], "r", encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 30
        assert len({r["key"] for r in rows}) == 30  # no duplicates

    def test_only_discrete_false_extracts_rest(self, dataset):
        out_dir = dataset["out_dir"].parent / "clips_all"
        index_csv = dataset["index_csv"].parent / "index_all.csv"
        report = extract_clips(
            parquet_dir=dataset["parquet_dir"],
            csv_path=dataset["csv_path"],
            out_dir=out_dir,
            index_csv=index_csv,
            only_discrete=False,
        )
        assert (out_dir / "restrow.mp4").exists()
        assert report.n_extracted == 28
        assert report.n_filtered_out == 1  # nometa only

    def test_keys_subset_filter(self, dataset):
        out_dir = dataset["out_dir"].parent / "clips_subset"
        index_csv = dataset["index_csv"].parent / "index_subset.csv"
        report = extract_clips(
            parquet_dir=dataset["parquet_dir"],
            csv_path=dataset["csv_path"],
            out_dir=out_dir,
            index_csv=index_csv,
            only_discrete=True,
            keys={"b64valid"},
        )
        assert report.n_extracted == 1
        assert report.n_decode_error == 0
        assert report.n_filtered_out == 31
        assert sorted(p.name for p in out_dir.glob("*.mp4")) == ["b64valid.mp4"]


# --------------------------------------------------------------------------------------
# Stratified selection
# --------------------------------------------------------------------------------------

class TestSelection:
    # Pool = status ok AND Single-source AND Discrete:
    # rawvalid, b64valid, s001..s024 -> 26 keys over 6 categories.
    POOL_SIZE = 26

    def test_same_seed_same_result(self, extracted):
        a = select_stage_m_clips(extracted["index_csv"], n=16, seed=0)
        b = select_stage_m_clips(extracted["index_csv"], n=16, seed=0)
        assert a == b
        assert len(a) == 16

    def test_different_seed_different_result(self, extracted):
        a = select_stage_m_clips(extracted["index_csv"], n=16, seed=0)
        b = select_stage_m_clips(extracted["index_csv"], n=16, seed=1)
        assert a != b

    def test_pool_constraints(self, extracted):
        selected = select_stage_m_clips(extracted["index_csv"], n=16, seed=0)
        index = _read_index(extracted["index_csv"])
        assert len(selected) == len(set(selected))
        for key in selected:
            row = index[key]
            assert row["status"] == "ok"
            assert row["source_type"] == "Single-source"
            assert row["discrete_vs_rest"] == "Discrete"
        assert "multisrc" not in selected
        assert "restrow" not in selected

    def test_stratified_round_robin_covers_categories(self, extracted):
        # n = number of categories: round-robin must take exactly one per category.
        selected = select_stage_m_clips(extracted["index_csv"], n=6, seed=0)
        index = _read_index(extracted["index_csv"])
        cats = [index[k]["ucs_category"] for k in selected]
        assert sorted(cats) == sorted(_CATEGORIES)

    def test_n16_spans_at_least_4_categories(self, extracted):
        selected = select_stage_m_clips(extracted["index_csv"], n=16, seed=0)
        index = _read_index(extracted["index_csv"])
        cats = {index[k]["ucs_category"] for k in selected}
        assert len(cats) >= 4

    def test_n_larger_than_pool_returns_whole_pool(self, extracted):
        selected = select_stage_m_clips(extracted["index_csv"], n=400, seed=0)
        assert len(selected) == self.POOL_SIZE
        assert len(set(selected)) == self.POOL_SIZE

    def test_screening_exclusion_respected(self, extracted):
        stage_m = select_stage_m_clips(extracted["index_csv"], n=16, seed=0)
        screening = select_screening_pool(
            extracted["index_csv"], n=400, exclude=set(stage_m), seed=0,
        )
        assert set(stage_m) & set(screening) == set()
        assert len(screening) == self.POOL_SIZE - 16

    def test_screening_deterministic(self, extracted):
        stage_m = select_stage_m_clips(extracted["index_csv"], n=16, seed=0)
        a = select_screening_pool(extracted["index_csv"], n=8, exclude=set(stage_m), seed=3)
        b = select_screening_pool(extracted["index_csv"], n=8, exclude=set(stage_m), seed=3)
        assert a == b
        assert len(a) == 8


# --------------------------------------------------------------------------------------
# PyAV validation + ok_rate halt logic
# --------------------------------------------------------------------------------------

def _write_real_mp4(path: Path, n_frames: int = 8) -> None:
    """Encode a tiny real MP4 (mpeg4, 32x32) so PyAV validation passes."""
    av = pytest.importorskip("av")
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=8)
        stream.width = 32
        stream.height = 32
        stream.pix_fmt = "yuv420p"
        for i in range(n_frames):
            img = np.full((32, 32, 3), (i * 17) % 255, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            frame = frame.reformat(format="yuv420p")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


@pytest.fixture
def small_extracted(dataset: dict) -> dict:
    """Three-clip extraction (s001..s003) with REAL mp4 contents for validation."""
    pytest.importorskip("av")
    out_dir = dataset["out_dir"]
    index_csv = dataset["index_csv"]
    extract_clips(
        parquet_dir=dataset["parquet_dir"],
        csv_path=dataset["csv_path"],
        out_dir=out_dir,
        index_csv=index_csv,
        only_discrete=True,
        keys={"s001", "s002", "s003"},
    )
    for key in ("s001", "s002", "s003"):
        _write_real_mp4(out_dir / f"{key}.mp4")
    return dataset


class TestValidateClipsAv:
    def test_all_valid_clips_pass(self, small_extracted):
        stats = validate_clips_av(small_extracted["index_csv"])
        assert stats == {"n_ok": 3, "n_failed": 0, "ok_rate": 1.0}
        index = _read_index(small_extracted["index_csv"])
        assert all(index[k]["status"] == "ok" for k in ("s001", "s002", "s003"))

    def test_corrupt_file_flipped_and_ok_rate_below_threshold(self, small_extracted):
        # Corrupt one mp4 ON DISK after extraction.
        (small_extracted["out_dir"] / "s003.mp4").write_bytes(b"garbage" * 100)
        stats = validate_clips_av(small_extracted["index_csv"])
        assert stats["n_ok"] == 2
        assert stats["n_failed"] == 1
        assert stats["ok_rate"] == pytest.approx(2.0 / 3.0)
        # The CLI gate: ok_rate < 0.95 must halt.
        assert stats["ok_rate"] < 0.95
        index = _read_index(small_extracted["index_csv"])
        assert index["s003"]["status"] == "decode_error"
        assert index["s001"]["status"] == "ok"
        assert index["s002"]["status"] == "ok"

    def test_cli_validate_exits_2_on_low_ok_rate(self, small_extracted):
        (small_extracted["out_dir"] / "s003.mp4").write_bytes(b"garbage" * 100)
        script = Path(__file__).resolve().parent.parent / "scripts" / "extract_foleybench.py"
        proc = subprocess.run(
            [sys.executable, str(script), "validate",
             "--index-csv", str(small_extracted["index_csv"])],
            capture_output=True, text=True, timeout=120,
        )
        assert proc.returncode == 2, proc.stderr
        assert "HALT" in proc.stderr

    def test_cli_validate_exits_0_on_healthy_index(self, small_extracted):
        script = Path(__file__).resolve().parent.parent / "scripts" / "extract_foleybench.py"
        proc = subprocess.run(
            [sys.executable, str(script), "validate",
             "--index-csv", str(small_extracted["index_csv"])],
            capture_output=True, text=True, timeout=120,
        )
        assert proc.returncode == 0, proc.stderr


# --------------------------------------------------------------------------------------
# write_manifest_json
# --------------------------------------------------------------------------------------

class TestWriteManifestJson:
    def test_writes_sorted_json_with_generated_by(self, tmp_path):
        path = tmp_path / "manifests" / "stage_m_clips.json"
        write_manifest_json(path, {
            "clips": ["k1", "k2"], "seed": 0, "n": 2, "pool": "single_source_discrete",
        })
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["clips"] == ["k1", "k2"]
        assert payload["generated_by"] == "foley_cw.foleybench_extract"
        text = path.read_text(encoding="utf-8")
        assert text.index('"clips"') < text.index('"generated_by"') < text.index('"seed"')

    def test_atomic_no_tmp_left(self, tmp_path):
        path = tmp_path / "m.json"
        write_manifest_json(path, {"clips": []})
        assert not Path(str(path) + ".tmp").exists()

    def test_payload_not_mutated_and_override_kept(self, tmp_path):
        payload = {"clips": [], "generated_by": "custom"}
        write_manifest_json(tmp_path / "m.json", payload)
        assert payload == {"clips": [], "generated_by": "custom"}
        loaded = json.loads((tmp_path / "m.json").read_text(encoding="utf-8"))
        assert loaded["generated_by"] == "custom"


# --------------------------------------------------------------------------------------
# Module import safety
# --------------------------------------------------------------------------------------

def test_public_api_exists():
    import foley_cw.foleybench_extract as m

    for name in (
        "VideoDecodeError", "iter_foleybench_rows", "decode_video_bytes",
        "load_metadata_csv", "extract_clips", "ExtractionReport",
        "validate_clips_av", "select_stage_m_clips", "select_screening_pool",
        "write_manifest_json",
    ):
        assert hasattr(m, name)
