"""Tests for foley_cw.real_measurer with injected tagger/embed fns (CPU, no checkpoints).

Revised plan section 3.3: the class axis is an EVENT-RESTRICTED argmax (speech/
music/ambient coarse groups excluded via the frozen map's class_excluded_coarse
key, plus non_event_indices) with a CROSS-GROUP abstain margin delta — within-
group runner-ups never trigger abstain; the abstain label is a first-class
categorical value that flows through measure() and the determinism contract.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from foley_cw.real_measurer import (ABSENT, ABSTAIN, AST_MODEL_ID, AST_REVISION,
                                    CLASS_ABSTAIN_DELTA, CLAP_MODEL_ID, CLAP_REVISION,
                                    NO_ONSET_LABEL, PRESENT, RealFoleyMeasurer, load_coarse_map)
from foley_cw.types import AgreementMetric, Axis, AxisKind, AxisTier

SR = 16000

PRESENCE = Axis(id="presence", name="presence", tier=AxisTier.TIER1, kind=AxisKind.CATEGORICAL,
                agreement=AgreementMetric.EXACT_MATCH, measure="presence_detector")
CLASS = Axis(id="class", name="class", tier=AxisTier.TIER1, kind=AxisKind.CATEGORICAL,
             agreement=AgreementMetric.KRIPPENDORFF_ALPHA, measure="audio_tagger_top1")
TIMING = Axis(id="timing", name="timing", tier=AxisTier.TIER1, kind=AxisKind.CATEGORICAL,
              agreement=AgreementMetric.EXACT_MATCH, measure="onset_timing_bin")
MATERIAL = Axis(id="material", name="material", tier=AxisTier.TIER2, kind=AxisKind.EMBEDDING,
                agreement=AgreementMetric.MEAN_PAIRWISE_COSINE, measure="audio_embedding")
BINDING = Axis(id="binding", name="binding", tier=AxisTier.TIER3, kind=AxisKind.CATEGORICAL,
               agreement=AgreementMetric.EXACT_MATCH, measure="binding_label")

# Toy 6-class prob vector layout:
#   0, 1 -> impact (same coarse group)   2 -> footsteps   3 -> machine
#   4    -> non-event index (unmapped)   5 -> speech (excluded coarse group)
N_PROBS = 6


@pytest.fixture()
def coarse_map(tmp_path):
    m = {"version": "test-v3",
         "coarse_classes": ["impact", "footsteps", "machine", "speech"],
         "index_to_coarse": {"0": "impact", "1": "impact", "2": "footsteps",
                             "3": "machine", "5": "speech"},
         "class_excluded_coarse": ["speech"],
         "non_event_indices": [4]}
    p = tmp_path / "coarse_class_map.json"
    p.write_text(json.dumps(m))
    return p


def make_measurer(coarse_map, probs=None, emb=None):
    probs = probs if probs is not None else np.zeros(N_PROBS)
    def tagger_fn(audio):
        return np.asarray(probs, dtype=float), np.arange(2048, dtype=float)
    def embed_fn(audio):
        return np.array([3.0, 4.0])
    return RealFoleyMeasurer(coarse_map_path=coarse_map, tagger_fn=tagger_fn,
                             embed_fn=embed_fn, tagger2_fn=lambda a: np.asarray(probs, dtype=float))


def burst(amp=0.5, onset_s=1.0, dur_s=4.0):
    """Silence then a sine burst starting at onset_s."""
    n = int(SR * dur_s)
    x = np.zeros(n, dtype=np.float32)
    i0 = int(SR * onset_s)
    t = np.arange(n - i0) / SR
    x[i0:] = amp * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    return x


def test_load_coarse_map_validates(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"coarse_classes": []}))
    with pytest.raises(ValueError):
        load_coarse_map(p)


def test_load_coarse_map_defaults_event_restriction_keys(tmp_path):
    """Older maps without the v3 keys load with empty exclusions (no crash)."""
    p = tmp_path / "old.json"
    p.write_text(json.dumps({"version": "v1", "coarse_classes": ["impact"],
                             "index_to_coarse": {"0": "impact"}}))
    d = load_coarse_map(p)
    assert d["class_excluded_coarse"] == []
    assert d["non_event_indices"] == []


def test_sr_guard(coarse_map):
    with pytest.raises(ValueError):
        RealFoleyMeasurer(sr=32000, coarse_map_path=coarse_map)


@pytest.mark.parametrize(
    ("model_id", "revision"),
    [(CLAP_MODEL_ID, CLAP_REVISION), (AST_MODEL_ID, AST_REVISION)],
)
def test_hf_pretrained_specs_are_pinned_and_local_only(
    coarse_map, monkeypatch, model_id, revision
):
    monkeypatch.setenv("FOLEY_CW_WEIGHTS_SOURCE", "hf")
    name, kwargs = make_measurer(coarse_map)._pretrained_spec(model_id)
    assert name == model_id
    assert kwargs == {"revision": revision, "local_files_only": True}


def test_modelscope_source_prefers_local_mirror(coarse_map, tmp_path, monkeypatch):
    mirror = tmp_path / "modelscope"
    local_model = mirror / CLAP_MODEL_ID
    local_model.mkdir(parents=True)
    monkeypatch.setenv("FOLEY_CW_WEIGHTS_SOURCE", "modelscope")
    monkeypatch.setenv("FOLEY_CW_MODELSCOPE_ROOT", str(mirror))
    name, kwargs = make_measurer(coarse_map)._pretrained_spec(CLAP_MODEL_ID)
    assert name == str(local_model)
    assert kwargs == {"revision": CLAP_REVISION, "local_files_only": True}


def test_modelscope_source_never_falls_through_to_download(
    coarse_map, tmp_path, monkeypatch
):
    monkeypatch.setenv("FOLEY_CW_WEIGHTS_SOURCE", "modelscope")
    monkeypatch.setenv("FOLEY_CW_MODELSCOPE_ROOT", str(tmp_path / "missing"))
    with pytest.raises(FileNotFoundError, match="Downloads are disabled"):
        make_measurer(coarse_map)._pretrained_spec(AST_MODEL_ID)


def test_presence_present_and_absent(coarse_map):
    m = make_measurer(coarse_map, probs=[0.9, 0, 0, 0, 0, 0])
    assert m.measure(burst(), PRESENCE).label == PRESENT
    # loud audio but only the non-event class fires -> absent
    m2 = make_measurer(coarse_map, probs=[0.0, 0, 0, 0, 0.9, 0])
    assert m2.measure(burst(), PRESENCE).label == ABSENT
    # silence fails the energy gate even with tagger prob
    assert m.measure(np.zeros(SR, dtype=np.float32), PRESENCE).label == ABSENT


# ------------------------------------------------------------------ class axis
def test_class_event_restricted_argmax(coarse_map):
    """Revised 3.3: argmax is over EVENT classes only. The speech class (idx 5)
    has the globally highest prob but its coarse group is excluded, so the best
    event class (idx 3 = machine) decides; margin 0.5 - 0.3 >= delta, no abstain."""
    m = make_measurer(coarse_map, probs=[0.3, 0.3, 0.0, 0.5, 0.0, 0.9])
    assert m.measure(burst(), CLASS).label == "machine"
    # a hot non-event index (idx 4) is equally invisible to the class decision
    m2 = make_measurer(coarse_map, probs=[0.1, 0.0, 0.0, 0.5, 0.9, 0.0])
    assert m2.measure(burst(), CLASS).label == "machine"


def test_class_no_event_signal_abstains(coarse_map):
    """All event-class probs at zero (everything on speech + non-event): the
    cross-group margin is 0 < delta -> abstain, never a knife-edge event label."""
    m = make_measurer(coarse_map, probs=[0.0, 0.0, 0.0, 0.0, 0.9, 0.9])
    assert m.measure(burst(), CLASS).label == ABSTAIN


def test_within_group_runner_up_does_not_abstain(coarse_map):
    """Two near-tied top classes in the SAME coarse group (idx 0/1 -> impact):
    a within-group flip cannot change the label, so the margin is taken against
    the best CROSS-GROUP class (machine at 0.10) and there is no abstain."""
    m = make_measurer(coarse_map, probs=[0.50, 0.48, 0.0, 0.10, 0.0, 0.0])
    tgt = m.measure(burst(), CLASS)
    assert tgt.label == "impact"
    d = m.class_diagnostics(np.array([0.50, 0.48, 0.0, 0.10, 0.0, 0.0]))
    assert d["margin"] == pytest.approx(0.40)  # 0.50 - 0.10, NOT 0.50 - 0.48


def test_cross_group_close_call_abstains(coarse_map):
    """Top-2 in DIFFERENT groups closer than delta -> abstain."""
    m = make_measurer(coarse_map, probs=[0.50, 0.0, 0.0, 0.48, 0.0, 0.0])
    assert m.measure(burst(), CLASS).label == ABSTAIN
    d = m.class_diagnostics(np.array([0.50, 0.0, 0.0, 0.48, 0.0, 0.0]))
    assert d["label"] == ABSTAIN
    assert d["margin"] == pytest.approx(0.02)
    assert d["margin"] < CLASS_ABSTAIN_DELTA


def test_class_diagnostics_values(coarse_map):
    """Non-gating instruments are sane: margin/entropy/top1_prob/concentration."""
    m = make_measurer(coarse_map)
    probs = np.array([0.5, 0.2, 0.1, 0.3, 0.7, 0.9])  # event probs: [0.5,0.2,0.1,0.3]
    d = m.class_diagnostics(probs)
    assert d["label"] == "impact"
    assert d["top1_index"] == 0 and d["top1_group"] == "impact"
    assert d["top1_prob"] == pytest.approx(0.5)
    # runner-up is the best cross-group event class: machine (idx 3, 0.3)
    assert d["runner_index"] == 3
    assert d["runner_prob"] == pytest.approx(0.3)
    assert d["margin"] == pytest.approx(0.2)
    # concentration = top1 / sum(event probs)
    assert d["concentration"] == pytest.approx(0.5 / 1.1)
    # entropy of the normalized event distribution: 0 < H <= log(n_event)
    assert 0.0 < d["entropy"] <= np.log(4) + 1e-9
    # excluded/non-event indices never appear in the decision fields
    assert d["top1_index"] not in (4, 5) and d["runner_index"] not in (4, 5)


def test_custom_abstain_delta(coarse_map):
    """delta is a constructor parameter; a wider delta turns a confident call
    into an abstain on the same probs."""
    probs = [0.50, 0.0, 0.0, 0.30, 0.0, 0.0]  # margin 0.20
    def tagger_fn(audio):
        return np.asarray(probs, dtype=float), np.zeros(2048)
    m = RealFoleyMeasurer(coarse_map_path=coarse_map, tagger_fn=tagger_fn)
    assert m.measure(burst(), CLASS).label == "impact"
    wide = RealFoleyMeasurer(coarse_map_path=coarse_map, tagger_fn=tagger_fn,
                             class_abstain_delta=0.25)
    assert wide.measure(burst(), CLASS).label == ABSTAIN


def test_timing_onset_bin_and_none(coarse_map):
    pytest.importorskip("librosa")
    m = make_measurer(coarse_map, probs=[0.9, 0, 0, 0, 0, 0])
    tgt = m.measure(burst(onset_s=1.0), TIMING)
    assert tgt.label == int(1.0 // 0.5)  # bin 2
    assert m.measure(np.zeros(SR, dtype=np.float32), TIMING).label == NO_ONSET_LABEL


def test_material_unit_norm(coarse_map):
    m = make_measurer(coarse_map)
    tgt = m.measure(burst(), MATERIAL)
    assert tgt.kind is AxisKind.EMBEDDING
    assert np.linalg.norm(tgt.embedding) == pytest.approx(1.0)
    np.testing.assert_allclose(tgt.embedding, [0.6, 0.8])


def test_binding_not_implemented(coarse_map):
    m = make_measurer(coarse_map)
    with pytest.raises(NotImplementedError):
        m.measure(burst(), BINDING)


def test_determinism_and_forward_cache(coarse_map):
    calls = {"n": 0}
    def tagger_fn(audio):
        calls["n"] += 1
        return np.array([0.9, 0, 0, 0, 0, 0]), np.zeros(2048)
    m = RealFoleyMeasurer(coarse_map_path=coarse_map, tagger_fn=tagger_fn)
    a = burst()
    l1 = m.measure(a, PRESENCE).label
    l2 = m.measure(a, CLASS).label
    l3 = m.measure(a, PRESENCE).label
    posterior = m.panns_posterior(a)
    assert l1 == l3 == PRESENT and l2 == "impact"
    np.testing.assert_allclose(posterior, [0.9, 0, 0, 0, 0, 0])
    assert calls["n"] == 1  # presence+class+posterior+repeat share the cache


def test_determinism_includes_abstain(coarse_map):
    """Stage-M criterion 4 uses the extended alphabet: identical audio must give
    identical labels INCLUDING abstain (the knife-edge case the rule exists for)."""
    def tagger_fn(audio):
        return np.array([0.50, 0.0, 0.0, 0.48, 0.0, 0.0]), np.zeros(2048)
    m = RealFoleyMeasurer(coarse_map_path=coarse_map, tagger_fn=tagger_fn)
    a = burst()
    first = m.measure(a, CLASS).label
    second = m.measure(np.copy(a), CLASS).label  # same bytes, fresh array
    assert first == second == ABSTAIN


def test_abstain_flows_through_measure_as_categorical(coarse_map):
    m = make_measurer(coarse_map, probs=[0.50, 0.0, 0.0, 0.48, 0.0, 0.0])
    tgt = m.measure(burst(), CLASS)
    assert tgt.axis_id == "class"
    assert tgt.kind is AxisKind.CATEGORICAL
    assert tgt.label == ABSTAIN


def test_panns_embedding_exposed(coarse_map):
    m = make_measurer(coarse_map)
    emb = m.panns_embedding(burst())
    assert emb.shape == (2048,)


def test_second_tagger_uses_same_coarse_map(coarse_map):
    m = make_measurer(coarse_map, probs=[0.0, 0.0, 0.9, 0.0, 0.0, 0.0])
    assert m.coarse_label_second_tagger(burst()) == "footsteps"
    # the second tagger inherits the event restriction + abstain rule too
    m2 = make_measurer(coarse_map, probs=[0.50, 0.0, 0.0, 0.48, 0.0, 0.9])
    assert m2.coarse_label_second_tagger(burst()) == ABSTAIN
