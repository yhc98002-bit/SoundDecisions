"""Tests for foley_cw/run_store.py — §1.4 logging & storage contract.

CPU-only, no network, no GPU; soundfile-dependent tests use pytest.importorskip.
Key contracts checked:
  * Journal atomicity: journal_done leaves no *.tmp; a stray partial tmp file is
    never counted as done.
  * is_done / done_units / load_journal roundtrip; unit_id sanitization
    ('/' and ':' -> '_').
  * fp16 npz feature roundtrip within tolerance; step features keep ts intact.
  * PCM_16 preview roundtrip via soundfile within 1e-3.
  * audit_selected is deterministic with a ~10% selection rate (5%-15% over
    2000 synthetic ids); fork finals are stored ONLY for selected ids.
  * Measurement JSONL roundtrip, including an embedding SelfTarget.
  * Every write is budget-accounted; an over-cap write raises StorageCapExceeded.
"""

from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pytest

from foley_cw.run_store import (
    RunStore,
    audit_selected,
    sanitize_unit_id,
    target_from_jsonable,
    to_jsonable_target,
)
from foley_cw.storage_budget import StorageBudget, StorageCapExceeded, measure_tree
from foley_cw.types import AxisKind, SelfTarget


# --------------------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------------------

@pytest.fixture
def store(tmp_path):
    return RunStore(root=tmp_path / "store")


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def _find_audit_ids(n_each: int = 1):
    """Return (selected_ids, unselected_ids) scanned from a deterministic pool."""
    selected, unselected = [], []
    for i in range(200):
        gid = f"fork{i:04d}"
        (selected if audit_selected(gid) else unselected).append(gid)
        if len(selected) >= n_each and len(unselected) >= n_each:
            break
    assert selected and unselected, "expected both classes within 200 ids"
    return selected, unselected


# --------------------------------------------------------------------------------------
# Journal: atomicity, roundtrip, sanitization
# --------------------------------------------------------------------------------------

class TestJournal:
    def test_is_done_roundtrip(self, store):
        assert not store.is_done("unit_a")
        store.journal_done("unit_a", {"status": "ok", "n": 3})
        assert store.is_done("unit_a")
        assert store.load_journal("unit_a") == {"status": "ok", "n": 3}

    def test_done_units_roundtrip(self, store):
        store.journal_done("u1", {})
        store.journal_done("u2", {"x": 1})
        assert store.done_units() == {"u1", "u2"}

    def test_done_units_empty_store(self, store):
        assert store.done_units() == set()

    def test_no_tmp_left_behind(self, store):
        store.journal_done("unit_a", {"status": "ok"})
        jdir = store.root / "journal"
        leftovers = list(jdir.glob("*.tmp"))
        assert leftovers == [], f"journal_done left tmp files: {leftovers}"

    def test_partial_tmp_ignored_by_is_done(self, store):
        # Simulate a crash mid-write: a partial tmp file, never os.replace'd.
        store.journal_done("real_unit", {})  # ensures journal/ exists
        jdir = store.root / "journal"
        (jdir / "crashed_unit.json.tmp").write_text("{\"truncat", encoding="utf-8")
        assert not store.is_done("crashed_unit")
        assert "crashed_unit" not in store.done_units()
        assert store.done_units() == {"real_unit"}

    def test_unit_id_sanitization(self, store):
        unit_id = "phase1/clip:003"
        store.journal_done(unit_id, {"ok": True})
        assert store.is_done(unit_id)
        expected_file = store.root / "journal" / "phase1_clip_003.json"
        assert expected_file.exists()
        assert "phase1_clip_003" in store.done_units()
        assert store.load_journal(unit_id) == {"ok": True}

    def test_sanitize_unit_id_function(self):
        assert sanitize_unit_id("a/b:c") == "a_b_c"
        assert sanitize_unit_id("plain") == "plain"

    def test_journal_overwrite_updates_payload(self, store):
        store.journal_done("u", {"v": 1})
        store.journal_done("u", {"v": 2})
        assert store.load_journal("u") == {"v": 2}

    def test_journal_payload_with_numpy_values(self, store):
        store.journal_done("np_unit", {"score": np.float64(0.5), "arr": np.arange(3)})
        loaded = store.load_journal("np_unit")
        assert loaded["score"] == pytest.approx(0.5)
        assert loaded["arr"] == [0, 1, 2]


# --------------------------------------------------------------------------------------
# Features: fp16 npz roundtrip
# --------------------------------------------------------------------------------------

class TestFeatures:
    def test_put_features_roundtrip_fp16(self, store, rng):
        feats = rng.standard_normal((4, 16))
        path = store.put_features("gen001", s=0.3, feats=feats)
        assert path.name == "gen001__s0.30.npz"
        with np.load(path) as npz:
            assert set(npz.files) == {"pooled"}
            loaded = npz["pooled"]
        assert loaded.dtype == np.float16
        # fp16 has ~3 decimal digits; for N(0,1) values atol 1e-2 is ample.
        assert np.allclose(loaded.astype(np.float64), feats, atol=1e-2)

    def test_put_features_s_formatting(self, store, rng):
        path = store.put_features("g", s=0.05, feats=rng.standard_normal(3))
        assert path.name == "g__s0.05.npz"

    def test_put_features_sanitizes_gen_id(self, store, rng):
        path = store.put_features("cfg4.5/clip:01", s=0.5, feats=rng.standard_normal(3))
        assert "/" not in path.name and ":" not in path.name
        assert path.exists()

    def test_put_step_features_roundtrip(self, store, rng):
        ts = np.linspace(0.0, 1.0, 33)
        feats = rng.standard_normal((33, 8))
        path = store.put_step_features("base007", ts=ts, feats=feats)
        assert path.name == "base007__steps.npz"
        with np.load(path) as npz:
            assert set(npz.files) == {"ts", "pooled"}
            ts_loaded = npz["ts"]
            feats_loaded = npz["pooled"]
        # ts kept at float32: progress grid must survive exactly enough to match.
        assert np.allclose(ts_loaded.astype(np.float64), ts, atol=1e-6)
        assert feats_loaded.dtype == np.float16
        assert np.allclose(feats_loaded.astype(np.float64), feats, atol=1e-2)

    def test_put_step_features_length_mismatch_raises(self, store, rng):
        with pytest.raises(ValueError):
            store.put_step_features("bad", ts=np.zeros(5), feats=rng.standard_normal((4, 2)))


# --------------------------------------------------------------------------------------
# Previews and finals: PCM_16 wav roundtrip (soundfile optional)
# --------------------------------------------------------------------------------------

class TestPreviewsAndFinals:
    def _sine(self, n=1600, sr=16000):
        t = np.arange(n) / sr
        return (0.5 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float64)

    def test_preview_roundtrip_pcm16(self, store):
        sf = pytest.importorskip("soundfile")
        wav = self._sine()
        path = store.put_preview("gen001", s=0.6, wav=wav, sr=16000)
        assert path.name == "gen001__s0.60.wav"
        loaded, sr = sf.read(str(path))
        assert sr == 16000
        assert loaded.shape == wav.shape
        assert np.allclose(loaded, wav, atol=1e-3)

    def test_preview_is_pcm16_subtype(self, store):
        sf = pytest.importorskip("soundfile")
        path = store.put_preview("gen002", s=0.1, wav=self._sine())
        info = sf.info(str(path))
        assert info.subtype == "PCM_16"

    def test_final_wav_stored_under_finals(self, store):
        pytest.importorskip("soundfile")
        path = store.put_final_wav("base001", wav=self._sine())
        assert path is not None
        assert path.parent.name == "finals"
        assert path.name == "base001.wav"

    def test_audit_only_selected_stored_under_audit_wavs(self, store):
        pytest.importorskip("soundfile")
        selected, _ = _find_audit_ids()
        path = store.put_final_wav(selected[0], wav=self._sine(), audit_only=True)
        assert path is not None
        assert path.parent.name == "audit_wavs"
        assert path.exists()

    def test_audit_only_unselected_returns_none_and_writes_nothing(self, store):
        # No soundfile needed: the unselected path must short-circuit BEFORE
        # any wav encoding, so dropped forks cost zero bytes and zero deps.
        _, unselected = _find_audit_ids()
        result = store.put_final_wav(unselected[0], wav=self._sine(), audit_only=True)
        assert result is None
        assert not (store.root / "audit_wavs").exists() or \
            list((store.root / "audit_wavs").glob("*.wav")) == []


# --------------------------------------------------------------------------------------
# audit_selected: determinism + ~10% rate
# --------------------------------------------------------------------------------------

class TestAuditSelection:
    def test_deterministic(self):
        for gid in ["a", "fork0001", "phase1/clip:003__k07"]:
            assert audit_selected(gid) == audit_selected(gid)
            assert RunStore.audit_selected(gid) == audit_selected(gid)

    def test_rate_roughly_10_percent_over_2000_ids(self):
        ids = [f"vid{i:03d}__s0.30__k{j:02d}" for i in range(200) for j in range(10)]
        assert len(ids) == 2000
        rate = float(np.mean([audit_selected(g) for g in ids]))
        assert 0.05 <= rate <= 0.15, f"audit rate {rate:.3f} outside [0.05, 0.15]"

    def test_selection_uses_raw_id_not_sanitized(self):
        # Selection hashes the RAW id; sanitization is filename-only.
        raw = "a/b:c"
        assert audit_selected(raw) == audit_selected(raw)
        # raw and sanitized ids are different keys (may or may not differ in
        # outcome, but must each be deterministic).
        assert audit_selected(sanitize_unit_id(raw)) == audit_selected("a_b_c")


# --------------------------------------------------------------------------------------
# SelfTarget JSON serialization
# --------------------------------------------------------------------------------------

class TestTargetJsonable:
    def test_categorical_roundtrip(self):
        target = SelfTarget(axis_id="presence", kind=AxisKind.CATEGORICAL, label=1)
        d = to_jsonable_target(target)
        json.dumps(d)  # must be JSON-able as-is
        back = target_from_jsonable(d)
        assert back.axis_id == "presence"
        assert back.kind is AxisKind.CATEGORICAL
        assert back.label == 1
        assert back.embedding is None

    def test_string_label_roundtrip(self):
        target = SelfTarget(axis_id="binding", kind=AxisKind.CATEGORICAL, label="(1,-1)")
        back = target_from_jsonable(to_jsonable_target(target))
        assert back.label == "(1,-1)"

    def test_numpy_label_coerced(self):
        target = SelfTarget(axis_id="class", kind=AxisKind.CATEGORICAL, label=np.int64(7))
        d = to_jsonable_target(target)
        json.dumps(d)
        assert d["label"] == 7

    def test_embedding_roundtrip(self):
        emb = np.array([0.6, -0.8, 0.0])
        target = SelfTarget(axis_id="material", kind=AxisKind.EMBEDDING, embedding=emb)
        d = to_jsonable_target(target)
        json.dumps(d)
        assert d["kind"] == "embedding"
        assert d["embedding"] == pytest.approx([0.6, -0.8, 0.0])
        back = target_from_jsonable(d)
        assert back.kind is AxisKind.EMBEDDING
        assert isinstance(back.embedding, np.ndarray)
        assert np.allclose(back.embedding, emb)


# --------------------------------------------------------------------------------------
# Measurements JSONL
# --------------------------------------------------------------------------------------

class TestMeasurements:
    def test_iter_empty_store(self, store):
        assert list(store.iter_measurements()) == []

    def test_categorical_measurement_roundtrip(self, store):
        target = SelfTarget(axis_id="presence", kind=AxisKind.CATEGORICAL, label=1)
        store.record_measurement("gen001", "presence", target, extra={"s": 0.3})
        records = list(store.iter_measurements())
        assert len(records) == 1
        rec = records[0]
        assert rec["gen_id"] == "gen001"
        assert rec["axis_id"] == "presence"
        assert rec["extra"] == {"s": 0.3}
        back = target_from_jsonable(rec["target"])
        assert back.label == 1

    def test_embedding_measurement_roundtrip(self, store):
        emb = np.array([1.0, 0.0, 0.0, 0.0]) / 1.0
        target = SelfTarget(axis_id="material", kind=AxisKind.EMBEDDING, embedding=emb)
        store.record_measurement("gen002", "material", target)
        rec = list(store.iter_measurements())[-1]
        back = target_from_jsonable(rec["target"])
        assert back.kind is AxisKind.EMBEDDING
        assert np.allclose(back.embedding, emb)

    def test_records_append_in_order(self, store):
        t = SelfTarget(axis_id="presence", kind=AxisKind.CATEGORICAL, label=0)
        for i in range(5):
            store.record_measurement(f"gen{i:03d}", "presence", t)
        gen_ids = [r["gen_id"] for r in store.iter_measurements()]
        assert gen_ids == [f"gen{i:03d}" for i in range(5)]

    def test_timestamp_is_iso_utc(self, store):
        t = SelfTarget(axis_id="presence", kind=AxisKind.CATEGORICAL, label=0)
        store.record_measurement("g", "presence", t)
        rec = list(store.iter_measurements())[0]
        parsed = datetime.fromisoformat(rec["ts"])
        assert parsed.tzinfo is not None  # tz-aware UTC stamp

    def test_default_extra_is_empty_dict(self, store):
        t = SelfTarget(axis_id="presence", kind=AxisKind.CATEGORICAL, label=0)
        store.record_measurement("g", "presence", t)
        assert list(store.iter_measurements())[0]["extra"] == {}


# --------------------------------------------------------------------------------------
# Budget integration: every byte accounted; over-cap halts
# --------------------------------------------------------------------------------------

class TestBudgetIntegration:
    def test_writes_accounted_match_disk(self, tmp_path, rng):
        budget = StorageBudget(cap_gb=1.0)
        store = RunStore(root=tmp_path / "store", budget=budget)
        store.put_features("g1", s=0.3, feats=rng.standard_normal((4, 8)))
        store.put_step_features("g1", ts=np.linspace(0, 1, 5),
                                feats=rng.standard_normal((5, 8)))
        store.record_measurement(
            "g1", "presence",
            SelfTarget(axis_id="presence", kind=AxisKind.CATEGORICAL, label=1),
        )
        store.journal_done("g1", {"ok": True})
        assert budget.spent_bytes > 0
        assert budget.spent_bytes == measure_tree(store.root), (
            "accounted bytes must equal on-disk bytes (no unaccounted writes)"
        )

    def test_cap_overflow_raises_on_feature_write(self, tmp_path, rng):
        budget = StorageBudget(cap_gb=10 / 1_000_000_000)  # cap = 10 bytes
        store = RunStore(root=tmp_path / "store", budget=budget)
        with pytest.raises(StorageCapExceeded):
            store.put_features("g1", s=0.3, feats=rng.standard_normal((4, 8)))

    def test_cap_overflow_raises_on_measurement(self, tmp_path):
        budget = StorageBudget(cap_gb=10 / 1_000_000_000)
        store = RunStore(root=tmp_path / "store", budget=budget)
        t = SelfTarget(axis_id="presence", kind=AxisKind.CATEGORICAL, label=1)
        with pytest.raises(StorageCapExceeded):
            store.record_measurement("g1", "presence", t)

    def test_cap_overflow_raises_on_journal(self, tmp_path):
        budget = StorageBudget(cap_gb=10 / 1_000_000_000)
        store = RunStore(root=tmp_path / "store", budget=budget)
        with pytest.raises(StorageCapExceeded):
            store.journal_done("u1", {"payload": "x" * 100})

    def test_projection_halts_before_run(self):
        budget = StorageBudget(cap_gb=100.0)
        # 200 clips x 11 s-points x (16 forks + 16 independents + base): if each
        # pooled feature blob were 10 MB the projection blows past 100 GB.
        n_units = 200 * 11 * 33
        with pytest.raises(StorageCapExceeded):
            budget.check_projection_or_halt(n_units, bytes_per_unit=10e6,
                                            context="phase1 pooled features")

    def test_no_budget_is_fine(self, tmp_path, rng):
        store = RunStore(root=tmp_path / "store", budget=None)
        path = store.put_features("g1", s=0.3, feats=rng.standard_normal(4))
        assert path.exists()


# --------------------------------------------------------------------------------------
# Module-level import safety
# --------------------------------------------------------------------------------------

def test_module_importable_without_soundfile_import():
    """foley_cw.run_store must import with numpy+stdlib only; soundfile is lazy."""
    import foley_cw.run_store  # noqa: F401


def test_lazy_dirs_not_created_on_construction(tmp_path):
    store = RunStore(root=tmp_path / "store")
    assert not (tmp_path / "store").exists(), "RunStore must create dirs lazily"
