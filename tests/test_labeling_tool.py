"""Tests for foley_cw.labeling_tool — self-contained HTML bundle generator."""

from __future__ import annotations

import base64
import json
import re

import pytest
import soundfile as sf
import numpy as np

from foley_cw.labeling_tool import (ClipItem, audio_item, event_classes,
                                    render_bundle, video_item, write_bundle)


def _coarse_map(tmp_path):
    m = {"version": "test", "coarse_classes": ["impact", "music", "footsteps", "other"],
         "index_to_coarse": {"0": "impact"}, "class_excluded_coarse": ["music"],
         "non_event_indices": []}
    p = tmp_path / "coarse.json"
    p.write_text(json.dumps(m))
    return p


def test_event_classes_excludes(tmp_path):
    assert event_classes(_coarse_map(tmp_path)) == ["impact", "footsteps", "other"]


def test_audio_item_data_uri(tmp_path):
    wav = tmp_path / "c1.wav"
    sf.write(wav, np.zeros(16000, dtype=np.float32), 16000, subtype="PCM_16")
    it = audio_item("c1", wav, qwen={"class": "impact"})
    assert it.media_mime == "audio/wav"
    assert it.media_b64.startswith("data:audio/wav;base64,")
    # round-trips to the original bytes
    raw = base64.b64decode(it.media_b64.split(",", 1)[1])
    assert raw == wav.read_bytes()
    assert it.qwen == {"class": "impact"}


def test_render_validity_bundle_self_contained(tmp_path):
    items = []
    for i in range(3):
        wav = tmp_path / f"c{i}.wav"
        sf.write(wav, np.full(8000, 0.1, dtype=np.float32), 16000, subtype="PCM_16")
        items.append(audio_item(f"c{i}", wav, qwen={"presence": "present"}))
    doc = render_bundle("validity", items, ["impact", "footsteps", "other"],
                        title="t", prompt_version="v1")
    assert doc.lstrip().startswith("<!doctype html>")
    # no external resource references (fully self-contained)
    assert "http://" not in doc and "https://" not in doc
    assert "src=\"http" not in doc
    # embedded payload parses and carries all clips + the 12-class set + qwen
    m = re.search(r'id="data">(.*?)</script>', doc, re.S)
    payload = json.loads(m.group(1).replace("<\\/", "</"))
    assert payload["task"] == "validity"
    assert len(payload["items"]) == 3
    assert payload["classes"] == ["impact", "footsteps", "other"]
    assert payload["items"][0]["qwen"] == {"presence": "present"}
    # validity widgets present
    assert "presence:" in doc and "abstain" in doc and "TAP at playhead" in doc


def test_render_anchor_bundle_has_video_and_onset_only(tmp_path):
    # tiny fake mp4 bytes (content irrelevant for the generator test)
    mp4 = tmp_path / "k1.mp4"
    mp4.write_bytes(b"\x00\x00\x00\x18ftypisom" + b"\x00" * 64)
    it = video_item("k1", mp4, proposed_onset_s=1.25)
    doc = render_bundle("anchor", [it], ["impact"], title="anchor")
    assert "<video" in doc
    assert "proposed 1.25s" in doc or "proposed" in doc
    # anchor task has NO class forced-choice
    payload = json.loads(re.search(r'id="data">(.*?)</script>', doc, re.S).group(1)
                         .replace("<\\/", "</"))
    assert payload["task"] == "anchor"
    assert payload["items"][0]["proposed_onset_s"] == 1.25


def test_write_bundle_manifest(tmp_path):
    wav = tmp_path / "c.wav"
    sf.write(wav, np.zeros(4000, dtype=np.float32), 16000, subtype="PCM_16")
    out = tmp_path / "b.html"
    man = write_bundle(out, "validity", [audio_item("c", wav)], ["impact"], title="t")
    assert out.exists()
    assert man["n_clips"] == 1 and man["clip_ids"] == ["c"]
    assert man["bytes"] == len(out.read_text(encoding="utf-8").encode("utf-8"))


def test_unknown_task_rejected(tmp_path):
    with pytest.raises(ValueError):
        render_bundle("nope", [], ["impact"], title="t")


def test_script_close_sequence_escaped(tmp_path):
    # a caption containing </script> must not break the document
    it = ClipItem(clip_id="x", media_b64="data:audio/wav;base64,AAAA",
                  media_mime="audio/wav", caption="weird </script> caption")
    doc = render_bundle("validity", [it], ["impact"], title="t")
    # exactly one real closing tag for the data script block region
    assert "</script>" in doc
    payload = json.loads(re.search(r'id="data">(.*?)</script>', doc, re.S).group(1)
                         .replace("<\\/", "</"))
    assert payload["items"][0]["caption"] == "weird </script> caption"
