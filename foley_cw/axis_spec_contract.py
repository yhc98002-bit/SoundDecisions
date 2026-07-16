"""Pure semantic contracts for the draft Axis Specification v2.

This module is not a measurement implementation.  It encodes only the edge-case
semantics that must remain invariant when the v2 measurers are built after PI
freeze: missing evidence, interval propagation, abstention, 2AFC indifference,
and binding-score eligibility.  No performance or scientific decision threshold
is frozen here.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence


PRESENCE_STATES = frozenset({"present", "absent", "uncertain"})
BINDING_ASSIGNMENTS = frozenset({"correct", "swapped", "ambiguous"})
TEMPORAL_ORDER_STATES = frozenset({"correct", "reversed", "ambiguous"})


def presence_state(
    target_event_detected: Optional[bool],
    *,
    evidence_confident: bool,
    background_event_detected: bool = False,
) -> str:
    """Return the target-event state; unrelated audio is not an input.

    Callers must determine whether the specified visible event, rather than any
    background sound, was detected.  ``background_event_detected`` is accepted
    only to pin the semantic invariant that off-target evidence cannot change
    the target state.  Goal-2 measurer tests must establish that the detector
    makes this distinction from audio.  ``None`` or low-confidence evidence is
    missing evidence and therefore maps to ``uncertain``.
    """

    if target_event_detected is None or not evidence_confident:
        return "uncertain"
    return "present" if target_event_detected else "absent"


def _interval(value: Sequence[float], name: str) -> tuple[float, float]:
    if len(value) != 2:
        raise ValueError(f"{name} must contain [lo, hi]")
    lo, hi = float(value[0]), float(value[1])
    if not (math.isfinite(lo) and math.isfinite(hi) and lo <= hi):
        raise ValueError(f"{name} must be a finite ordered interval")
    return lo, hi


def timing_observation(
    *,
    presence: str,
    anchor_interval_s: Sequence[float],
    onset_interval_s: Optional[Sequence[float]],
) -> dict:
    """Propagate anchor/onset intervals into an anchor-relative onset offset.

    Timing is defined only for confidently present target events with a detected
    onset.  The continuous center offset is accompanied by the full conservative
    interval; missing onsets are never assigned to a timing bin.
    """

    if presence not in PRESENCE_STATES:
        raise ValueError(f"unknown presence state: {presence!r}")
    anchor_lo, anchor_hi = _interval(anchor_interval_s, "anchor_interval_s")
    if presence != "present":
        return {
            "defined": False,
            "reason": "presence_not_confidently_positive",
            "delta_t_s": None,
            "delta_t_interval_s": None,
        }
    if onset_interval_s is None:
        return {
            "defined": False,
            "reason": "missing_onset",
            "delta_t_s": None,
            "delta_t_interval_s": None,
        }
    onset_lo, onset_hi = _interval(onset_interval_s, "onset_interval_s")
    anchor_center = (anchor_lo + anchor_hi) / 2.0
    onset_center = (onset_lo + onset_hi) / 2.0
    return {
        "defined": True,
        "reason": None,
        "delta_t_s": onset_center - anchor_center,
        "delta_t_interval_s": [onset_lo - anchor_hi, onset_hi - anchor_lo],
    }


def class_observation(*, label: Optional[str], posterior_persisted: bool) -> dict:
    """Return whether one v2 class observation is usable as evidence."""

    if label is None or label == "abstain":
        return {"scorable": False, "reason": "missing_evidence"}
    if not posterior_persisted:
        return {"scorable": False, "reason": "incomplete_posterior_artifact"}
    return {"scorable": True, "reason": None}


def material_2afc(
    *,
    similarity_positive: float,
    similarity_negative: float,
    indifference_margin: float,
) -> dict:
    """Apply the frozen-by-procedure 2AFC margin without making a probability."""

    pos = float(similarity_positive)
    neg = float(similarity_negative)
    tie = float(indifference_margin)
    if not (math.isfinite(pos) and math.isfinite(neg)):
        raise ValueError("2AFC similarities must be finite")
    if not math.isfinite(tie) or tie < 0.0:
        raise ValueError("indifference_margin must be finite and non-negative")
    margin = pos - neg
    if margin > tie:
        decision = "positive"
    elif margin < -tie:
        decision = "negative"
    else:
        decision = "indifferent"
    return {
        "decision": decision,
        "margin": margin,
        "is_scorable": decision != "indifferent",
        "probability": None,
    }


def binding_observation(
    *,
    event_presence: Sequence[Optional[bool]],
    assignment: Optional[str],
    extra_event_count: int = 0,
    anchors_usable: bool = True,
    temporal_order: Optional[str] = None,
) -> dict:
    """Separate event-set state from conditional two-event assignment accuracy."""

    if len(event_presence) != 2:
        raise ValueError("binding requires exactly two target-event presence states")
    if any(value is not None and type(value) is not bool for value in event_presence):
        raise TypeError("event_presence values must be bool or None")
    if assignment is not None and assignment not in BINDING_ASSIGNMENTS:
        raise ValueError(f"unknown binding assignment: {assignment!r}")
    if temporal_order is not None and temporal_order not in TEMPORAL_ORDER_STATES:
        raise ValueError(f"unknown temporal order state: {temporal_order!r}")
    if type(extra_event_count) is not int:
        raise TypeError("extra_event_count must be an integer")
    if extra_event_count < 0:
        raise ValueError("extra_event_count must be non-negative")

    if not anchors_usable or any(value is None for value in event_presence):
        return {
            "outcome": "ambiguous",
            "event_set_state": "ambiguous",
            "temporal_order": None,
            "identity_assignment": None,
            "primary_eligible": False,
            "primary_correct": None,
        }

    both_present = all(bool(value) for value in event_presence)
    if not both_present:
        return {
            "outcome": "missing",
            "event_set_state": "missing",
            "temporal_order": temporal_order,
            "identity_assignment": None,
            "primary_eligible": False,
            "primary_correct": None,
        }

    decisive = assignment in {"correct", "swapped"}
    if not decisive:
        return {
            "outcome": "ambiguous",
            "event_set_state": "both_present",
            "temporal_order": temporal_order,
            "identity_assignment": "ambiguous",
            "primary_eligible": False,
            "primary_correct": None,
        }

    outcome = "extra" if extra_event_count > 0 else str(assignment)
    return {
        "outcome": outcome,
        "event_set_state": "both_present",
        "temporal_order": temporal_order,
        "identity_assignment": assignment,
        "primary_eligible": True,
        "primary_correct": assignment == "correct",
    }
