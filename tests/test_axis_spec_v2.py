"""Executable invariants for the draft event-axis specification v2."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from foley_cw.axis_spec_contract import (
    binding_observation,
    class_observation,
    material_2afc,
    presence_state,
    timing_observation,
)


ROOT = Path(__file__).resolve().parents[1]
SPEC_DIR = ROOT / "experiment" / "axis_spec_v2"
TERMINAL_STATUSES = {
    "PASS",
    "UNRESOLVED",
    "INVALID_MEASUREMENT",
    "INCOMPLETE_ARTIFACTS",
    "ENGINEERING_FAILURE",
}


def _load(name: str):
    with (SPEC_DIR / name).open(encoding="utf-8") as handle:
        return json.load(handle)


def _assert_nested(actual, expected):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_nested(actual[key], expected[key])
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            _assert_nested(left, right)
    elif isinstance(expected, float):
        assert actual == pytest.approx(expected)
    else:
        assert actual == expected


def test_all_machine_contracts_are_valid_json():
    names = {
        path.name for path in SPEC_DIR.glob("*.json")
    }
    assert {
        "axis_spec_v2.json",
        "axis_spec_v2.schema.json",
        "axis_spec_v2.freeze.schema.json",
        "measurement_record_v2.schema.json",
        "reference_manifest_v2.schema.json",
        "semantic_edge_cases.json",
    } <= names
    for name in names:
        _load(name)


def test_goal1_draft_cannot_authorize_execution():
    spec = _load("axis_spec_v2.json")
    freeze = spec["freeze"]
    assert spec["status"] == "DRAFT_AWAITING_DUAL_PI_SIGNATURE"
    assert freeze["authority"] == "separate_signed_envelope"
    assert freeze["envelope_present"] is False
    assert freeze["measurement_authorized"] is False
    assert freeze["quarantine_release_authorized"] is False
    assert "freeze_sha256" not in freeze
    assert not (SPEC_DIR / "axis_spec_v2.freeze.json").exists()

    schema = _load("axis_spec_v2.schema.json")
    assert schema["properties"]["status"]["const"] == spec["status"]
    freeze_properties = schema["properties"]["freeze"]["properties"]
    assert freeze_properties["measurement_authorized"]["const"] is False
    assert freeze_properties["quarantine_release_authorized"]["const"] is False


def test_freeze_envelope_keeps_b6_causal_and_confirmatory_work_closed():
    schema = _load("axis_spec_v2.freeze.schema.json")
    auth = schema["properties"]["authorization"]["properties"]
    assert auth["goal_2_measurement"]["const"] is True
    assert auth["b2_measurement_only_release"]["const"] is True
    assert auth["b6_release"]["const"] is False
    assert auth["causal_analysis"]["const"] is False
    assert auth["confirmatory_execution"]["const"] is False


def test_every_axis_has_selection_persistence_and_fail_closed_rules():
    spec = _load("axis_spec_v2.json")
    assert set(spec["axes"]) == {
        "presence", "timing", "class", "material", "binding"
    }
    assert spec["shared_contract"]["threshold_and_model_selection"][
        "historical_numeric_thresholds_carried_forward"
    ] is False
    for axis in spec["axes"].values():
        assert axis["selection_procedure"]
        assert len(axis["minimum_persisted_fields"]) >= 8
        assert TERMINAL_STATUSES <= set(axis["disposition_rules"])
        assert axis["disposition_rules"]["redesign_or_demotion"]


def test_measurement_schema_enforces_axis_specific_payload_lineage():
    schema = _load("measurement_record_v2.schema.json")
    defs = schema["$defs"]
    assert "generation_id" in defs["generation"]["required"]
    assert {
        "audio_onset_source", "audio_onset_provenance", "delta_t_s"
    } <= set(defs["timingPayload"]["required"])
    assert {
        "clipwise_output_527", "coarse_posterior", "coarse_map_sha256",
        "abstention_rule_id",
    } <= set(defs["classPayload"]["required"])
    assert {
        "candidate", "positive_reference", "negative_reference", "presentation",
        "score_a", "score_b", "decision_choice", "indifference_margin_rule_id",
    } <= set(defs["materialPayload"]["required"])
    assert {
        "event_set_id", "event_set", "event_set_state", "identity_assignment",
        "primary_eligible", "primary_correct",
    } <= set(defs["bindingPayload"]["required"])
    assert {
        "pair_id", "source", "donor", "swap", "measured_role"
    } <= set(defs["conditionSwapLineage"]["required"])
    assert "axis_spec_content_sha256" in defs["provenance"]["required"]
    assert "freeze_envelope_sha256" in defs["provenance"]["required"]


def test_reference_schema_requires_auditable_frozen_trials():
    schema = _load("reference_manifest_v2.schema.json")
    defs = schema["$defs"]
    frozen_guard = schema["allOf"][1]["then"]["properties"]
    assert frozen_guard["trials"]["minItems"] == 1
    assert "axis_spec_content_sha256" in frozen_guard
    assert "freeze_envelope_sha256" in frozen_guard
    assert {
        "candidate", "positive_reference", "negative_reference",
        "matching_metadata", "presentation", "disjoint_references",
    } <= set(defs["materialTrial"]["required"])
    assert {
        "randomization_seed", "option_a_reference_id", "option_b_reference_id",
        "correct_choice", "orientation_sha256",
    } <= set(defs["abPresentation"]["required"])
    assert {
        "composite_event_set", "correct_assignment", "swapped_assignment",
        "b6_lineage",
    } <= set(defs["bindingTrial"]["required"])


def test_axis_specific_nonnegotiable_semantics():
    axes = _load("axis_spec_v2.json")["axes"]
    assert "background" in axes["presence"]["scientific_meaning"]
    assert axes["timing"]["target_type"]["categorical_bins"] is None
    assert "presence" in axes["timing"]["missingness"]
    assert "clipwise_output_527" in axes["class"]["minimum_persisted_fields"]
    assert "never a class" in axes["class"]["target_type"]["abstain"]
    assert axes["material"]["target_type"]["chance_level"] == 0.5
    assert "Bernoulli" in axes["material"]["target_type"]["forbidden_conversion"]
    assert "both target events present" in axes["binding"]["target_type"]["primary"]
    assert "never recoded as swapped" in axes["binding"]["missingness"]


def test_declared_semantic_edge_cases_execute():
    manifest = _load("semantic_edge_cases.json")
    for case in manifest["cases"]:
        axis = case["axis_id"]
        if axis == "presence":
            actual = presence_state(**case["input"])
        elif axis == "timing":
            actual = timing_observation(**case["input"])
        elif axis == "class":
            actual = class_observation(**case["input"])
        elif axis == "material":
            assert case["example_threshold_only"] is True
            actual = material_2afc(**case["input"])
        elif axis == "binding":
            actual = binding_observation(**case["input"])
        else:
            raise AssertionError(f"unknown edge-case axis: {axis}")
        _assert_nested(actual, case["expected"])


def test_background_evidence_cannot_turn_target_absence_into_presence():
    assert presence_state(
        False, evidence_confident=True, background_event_detected=False
    ) == presence_state(
        False, evidence_confident=True, background_event_detected=True
    ) == "absent"


def test_invalid_intervals_and_material_thresholds_fail_closed():
    with pytest.raises(ValueError, match="ordered interval"):
        timing_observation(
            presence="present",
            anchor_interval_s=[2.0, 1.0],
            onset_interval_s=[2.1, 2.2],
        )
    with pytest.raises(ValueError, match="non-negative"):
        material_2afc(
            similarity_positive=0.8,
            similarity_negative=0.5,
            indifference_margin=-0.01,
        )


def test_binding_rejects_truthy_nonbooleans_and_fractional_counts():
    with pytest.raises(TypeError, match="bool or None"):
        binding_observation(
            event_presence=[True, "uncertain"], assignment="correct"
        )
    with pytest.raises(TypeError, match="integer"):
        binding_observation(
            event_presence=[True, True], assignment="correct", extra_event_count=0.5
        )


def test_abstention_and_missing_events_never_become_semantic_classes():
    assert class_observation(label="abstain", posterior_persisted=True) == {
        "scorable": False,
        "reason": "missing_evidence",
    }
    result = binding_observation(
        event_presence=[True, False],
        assignment="swapped",
        temporal_order="correct",
    )
    assert result["outcome"] == "missing"
    assert result["identity_assignment"] is None
    assert result["primary_eligible"] is False
