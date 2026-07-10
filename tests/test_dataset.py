"""Tests for foley_cw/dataset.py.

All tests are numpy-only and use only the synthetic backend.  No GPU, no
MMAudio, no scipy, no librosa.

Coverage:
  1. build_synthetic_dataset — shapes, distinctness, anchor invariants.
  2. build_manifest — reads configs/dataset.json and captures min_usable_n.
  3. EventAnchor — constructor validation and properties.
  4. load_foleybench — raises FileNotFoundError when root missing.
  5. manifest_to_markdown — contains expected axis table lines.
  6. anchor_report_markdown — renders without error, contains summary.
  7. DatasetManifest.usable_n — correct lookup + default.
  8. SubsetSpec — single-event and two-event entries present.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

# Make sure the project root is on the path when running from anywhere.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from foley_cw.dataset import (
    DatasetManifest,
    EventAnchor,
    SubsetSpec,
    anchor_report_markdown,
    build_manifest,
    build_synthetic_dataset,
    load_foleybench,
    manifest_to_markdown,
)
from foley_cw.synthetic_backend import SyntheticVideoCond


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CONFIGS_DIR = ROOT / "configs"


def _load_dataset_cfg() -> dict:
    import json
    with open(CONFIGS_DIR / "dataset.json", "r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# 1. build_synthetic_dataset
# ---------------------------------------------------------------------------

class TestBuildSyntheticDataset:
    def test_length(self):
        records = build_synthetic_dataset(8)
        assert len(records) == 8

    def test_required_keys(self):
        records = build_synthetic_dataset(4)
        for rec in records:
            assert "video_id" in rec
            assert "cond" in rec
            assert "anchors" in rec

    def test_cond_type(self):
        records = build_synthetic_dataset(4)
        for rec in records:
            assert isinstance(rec["cond"], SyntheticVideoCond)

    def test_anchor_type(self):
        records = build_synthetic_dataset(4)
        for rec in records:
            assert isinstance(rec["anchors"], EventAnchor)

    def test_distinct_video_ids(self):
        n = 8
        records = build_synthetic_dataset(n)
        ids = [rec["video_id"] for rec in records]
        assert len(set(ids)) == n, "video_ids should all be distinct"

    def test_distinct_cond_mu(self):
        """All synthetic conds must have distinct mu vectors."""
        records = build_synthetic_dataset(8)
        mus = [tuple(rec["cond"].mu.tolist()) for rec in records]
        assert len(set(mus)) == len(mus), "All conds must have distinct mu vectors"

    def test_anchors_single_event(self):
        """Each synthetic record has exactly one event onset."""
        records = build_synthetic_dataset(4)
        for rec in records:
            a = rec["anchors"]
            assert a.is_single_event
            assert a.n_events == 1
            assert len(a.timestamps) == 1
            assert len(a.uncertainty) == 1

    def test_anchor_uncertainty_positive(self):
        records = build_synthetic_dataset(4)
        for rec in records:
            a = rec["anchors"]
            assert all(u > 0 for u in a.uncertainty)

    def test_anchor_source_synthetic(self):
        records = build_synthetic_dataset(4)
        for rec in records:
            assert rec["anchors"].source == "synthetic"

    def test_anchor_timestamps_in_range(self):
        """Synthetic onsets fall in [0.5, 2.5] seconds."""
        records = build_synthetic_dataset(8)
        for rec in records:
            t = rec["anchors"].timestamps[0]
            assert 0.5 <= t <= 2.5

    def test_reproducible_with_same_seed(self):
        r1 = build_synthetic_dataset(4, seed=7)
        r2 = build_synthetic_dataset(4, seed=7)
        for a, b in zip(r1, r2):
            np.testing.assert_array_equal(a["cond"].mu, b["cond"].mu)
            assert a["anchors"].timestamps == b["anchors"].timestamps

    def test_different_seeds_differ(self):
        r1 = build_synthetic_dataset(4, seed=0)
        r2 = build_synthetic_dataset(4, seed=42)
        mus_1 = [rec["cond"].mu for rec in r1]
        mus_2 = [rec["cond"].mu for rec in r2]
        any_diff = any(
            not np.array_equal(m1, m2) for m1, m2 in zip(mus_1, mus_2)
        )
        assert any_diff, "Different seeds should produce different mu vectors"

    def test_zero_videos(self):
        records = build_synthetic_dataset(0)
        assert records == []

    def test_large_n_unique(self):
        records = build_synthetic_dataset(100)
        ids = [rec["video_id"] for rec in records]
        assert len(set(ids)) == 100

    def test_dim_respected(self):
        records = build_synthetic_dataset(4, dim=8)
        for rec in records:
            assert rec["cond"].mu.shape == (8,)


# ---------------------------------------------------------------------------
# 2. build_manifest
# ---------------------------------------------------------------------------

class TestBuildManifest:
    def test_returns_datasetmanifest(self):
        cfg = _load_dataset_cfg()
        manifest = build_manifest(cfg)
        assert isinstance(manifest, DatasetManifest)

    def test_source(self):
        cfg = _load_dataset_cfg()
        manifest = build_manifest(cfg)
        assert manifest.source == "FoleyBench"

    def test_min_usable_n_presence(self):
        cfg = _load_dataset_cfg()
        manifest = build_manifest(cfg)
        assert manifest.min_usable_n_per_axis["presence"] == 40

    def test_min_usable_n_binding(self):
        cfg = _load_dataset_cfg()
        manifest = build_manifest(cfg)
        assert manifest.min_usable_n_per_axis["binding"] == 30

    def test_min_usable_n_all_axes(self):
        cfg = _load_dataset_cfg()
        manifest = build_manifest(cfg)
        # Verify all axis ids present
        expected = {"presence", "timing", "class", "material", "binding"}
        assert expected.issubset(set(manifest.min_usable_n_per_axis.keys()))

    def test_anchor_uncertainty_required(self):
        cfg = _load_dataset_cfg()
        manifest = build_manifest(cfg)
        assert manifest.anchor_uncertainty_required is True

    def test_clip_duration(self):
        cfg = _load_dataset_cfg()
        manifest = build_manifest(cfg)
        assert manifest.clip_duration_range_s == (1.0, 10.0)

    def test_event_anchor_priority_order(self):
        cfg = _load_dataset_cfg()
        manifest = build_manifest(cfg)
        assert manifest.event_anchor_priority[0] == "foleybench_metadata"
        assert "visual_onset_detector" in manifest.event_anchor_priority

    def test_single_event_subset_present(self):
        cfg = _load_dataset_cfg()
        manifest = build_manifest(cfg)
        names = [ss.name for ss in manifest.subsets]
        assert "single_event" in names

    def test_single_event_subset_enabled(self):
        cfg = _load_dataset_cfg()
        manifest = build_manifest(cfg)
        se = next(ss for ss in manifest.subsets if ss.name == "single_event")
        assert se.enabled is True

    def test_two_event_subset_present(self):
        cfg = _load_dataset_cfg()
        manifest = build_manifest(cfg)
        names = [ss.name for ss in manifest.subsets]
        assert "two_event" in names

    def test_two_event_min_n(self):
        cfg = _load_dataset_cfg()
        manifest = build_manifest(cfg)
        te = next(ss for ss in manifest.subsets if ss.name == "two_event")
        assert te.min_n_per_axis == 30

    def test_foleybench_root_none(self):
        """Configs ship with foleybench_root=null (not yet available)."""
        cfg = _load_dataset_cfg()
        manifest = build_manifest(cfg)
        assert manifest.foleybench_root is None

    def test_extra_doc_key_not_in_manifest_fields(self):
        """The _doc key is a documentation annotation — not a schema field and not in extra.
        build_manifest deliberately strips _doc (it is in known_keys for filtering).
        """
        cfg = _load_dataset_cfg()
        manifest = build_manifest(cfg)
        # _doc is a documentation key, not a data field; it should NOT appear in extra
        # (it is filtered alongside known schema keys to avoid polluting the manifest).
        assert "_doc" not in manifest.extra
        # The manifest fields themselves are schema-defined attributes, not JSON passthrough.
        assert not hasattr(manifest, "_doc")

    def test_usable_n_helper(self):
        cfg = _load_dataset_cfg()
        manifest = build_manifest(cfg)
        assert manifest.usable_n("presence") == 40
        assert manifest.usable_n("binding") == 30
        assert manifest.usable_n("nonexistent_axis") == 0

    def test_minimal_cfg(self):
        """build_manifest works with a bare-minimum dict."""
        cfg = {"source": "TestBench"}
        manifest = build_manifest(cfg)
        assert manifest.source == "TestBench"
        assert manifest.anchor_uncertainty_required is True  # default


# ---------------------------------------------------------------------------
# 3. EventAnchor
# ---------------------------------------------------------------------------

class TestEventAnchor:
    def test_n_events(self):
        a = EventAnchor(timestamps=[0.5, 1.2], uncertainty=[0.02, 0.03], source="foleybench_metadata")
        assert a.n_events == 2

    def test_is_single_event_true(self):
        a = EventAnchor(timestamps=[1.0], uncertainty=[0.02])
        assert a.is_single_event is True

    def test_is_single_event_false(self):
        a = EventAnchor(timestamps=[0.5, 1.5], uncertainty=[0.01, 0.01])
        assert a.is_single_event is False

    def test_max_uncertainty(self):
        a = EventAnchor(timestamps=[0.3, 1.1], uncertainty=[0.02, 0.10])
        assert abs(a.max_uncertainty - 0.10) < 1e-9

    def test_max_uncertainty_empty(self):
        a = EventAnchor(timestamps=[], uncertainty=[])
        assert a.max_uncertainty == float("inf")

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            EventAnchor(timestamps=[0.5, 1.0], uncertainty=[0.01])

    def test_default_source(self):
        a = EventAnchor(timestamps=[0.5], uncertainty=[0.02])
        assert a.source == "unknown"


# ---------------------------------------------------------------------------
# 4. load_foleybench — missing root raises FileNotFoundError
# ---------------------------------------------------------------------------

class TestLoadFoleybench:
    def test_raises_on_missing_root(self, tmp_path):
        missing = tmp_path / "nonexistent_foleybench_dir"
        with pytest.raises(FileNotFoundError):
            load_foleybench(missing)

    def test_raises_on_string_missing(self, tmp_path):
        missing = str(tmp_path / "also_missing")
        with pytest.raises(FileNotFoundError):
            load_foleybench(missing)

    def test_returns_manifest_on_empty_dir(self, tmp_path):
        """An existing-but-empty directory should return a manifest (no crash, no download)."""
        manifest = load_foleybench(tmp_path)
        assert isinstance(manifest, DatasetManifest)

    def test_manifest_root_set_on_empty_dir(self, tmp_path):
        manifest = load_foleybench(tmp_path)
        assert manifest.foleybench_root == str(tmp_path)

    def test_reads_metadata_json_if_present(self, tmp_path):
        import json as _json
        meta = {
            "source": "FoleyBench_custom",
            "min_usable_n_per_axis": {"presence": 99},
        }
        (tmp_path / "metadata.json").write_text(_json.dumps(meta))
        manifest = load_foleybench(tmp_path)
        assert manifest.source == "FoleyBench_custom"
        assert manifest.min_usable_n_per_axis["presence"] == 99


# ---------------------------------------------------------------------------
# 5. manifest_to_markdown
# ---------------------------------------------------------------------------

class TestManifestToMarkdown:
    def _make_manifest(self) -> DatasetManifest:
        return build_manifest(_load_dataset_cfg())

    def test_returns_str(self):
        md = manifest_to_markdown(self._make_manifest())
        assert isinstance(md, str)

    def test_contains_title(self):
        md = manifest_to_markdown(self._make_manifest())
        assert "# Dataset Subset Manifest" in md

    def test_contains_source(self):
        md = manifest_to_markdown(self._make_manifest())
        assert "FoleyBench" in md

    def test_contains_axis_presence(self):
        md = manifest_to_markdown(self._make_manifest())
        assert "presence" in md

    def test_contains_axis_binding(self):
        md = manifest_to_markdown(self._make_manifest())
        assert "binding" in md

    def test_contains_min_usable_n_header(self):
        md = manifest_to_markdown(self._make_manifest())
        assert "Minimum Usable n per Axis" in md

    def test_contains_anchor_priority(self):
        md = manifest_to_markdown(self._make_manifest())
        assert "foleybench_metadata" in md

    def test_contains_clip_duration(self):
        md = manifest_to_markdown(self._make_manifest())
        assert "1.0" in md
        assert "10.0" in md

    def test_contains_underpowered_note(self):
        md = manifest_to_markdown(self._make_manifest())
        assert "underpowered" in md


# ---------------------------------------------------------------------------
# 6. anchor_report_markdown
# ---------------------------------------------------------------------------

class TestAnchorReportMarkdown:
    def _make_anchors(self) -> list[EventAnchor]:
        return [
            EventAnchor(timestamps=[0.5], uncertainty=[0.02], source="foleybench_metadata"),
            EventAnchor(timestamps=[1.0], uncertainty=[0.03], source="foleybench_metadata"),
            EventAnchor(timestamps=[0.7, 2.1], uncertainty=[0.04, 0.05], source="visual_onset_detector"),
        ]

    def test_returns_str(self):
        md = anchor_report_markdown(self._make_anchors())
        assert isinstance(md, str)

    def test_contains_title(self):
        md = anchor_report_markdown(self._make_anchors())
        assert "# Event Anchor Validation Report" in md

    def test_contains_total_clips(self):
        md = anchor_report_markdown(self._make_anchors())
        assert "3" in md   # 3 clips

    def test_contains_source(self):
        md = anchor_report_markdown(self._make_anchors())
        assert "foleybench_metadata" in md
        assert "visual_onset_detector" in md

    def test_single_vs_multi_split(self):
        md = anchor_report_markdown(self._make_anchors())
        assert "Single-event clips" in md
        assert "Multi-event clips" in md

    def test_dict_input(self):
        d = {
            "vid001": EventAnchor(timestamps=[0.5], uncertainty=[0.02], source="synthetic"),
            "vid002": EventAnchor(timestamps=[1.0], uncertainty=[0.03], source="synthetic"),
        }
        md = anchor_report_markdown(d)
        assert "# Event Anchor Validation Report" in md
        assert "2" in md

    def test_empty_anchors(self):
        md = anchor_report_markdown([])
        assert "No anchors" in md

    def test_mean_uncertainty_present(self):
        md = anchor_report_markdown(self._make_anchors())
        assert "Mean uncertainty" in md


# ---------------------------------------------------------------------------
# 7. DatasetManifest.usable_n
# ---------------------------------------------------------------------------

class TestUsableN:
    def test_known_axis(self):
        m = build_manifest(_load_dataset_cfg())
        assert m.usable_n("timing") == 40

    def test_unknown_axis_returns_zero(self):
        m = build_manifest(_load_dataset_cfg())
        assert m.usable_n("not_an_axis") == 0


# ---------------------------------------------------------------------------
# 8. SubsetSpec fields
# ---------------------------------------------------------------------------

class TestSubsetSpec:
    def test_single_event_clip_range(self):
        m = build_manifest(_load_dataset_cfg())
        se = next(ss for ss in m.subsets if ss.name == "single_event")
        assert se.clip_duration_range_s == (1.0, 10.0)

    def test_two_event_purpose(self):
        m = build_manifest(_load_dataset_cfg())
        te = next(ss for ss in m.subsets if ss.name == "two_event")
        assert "binding" in te.purpose.lower() or "tier" in te.purpose.lower()
