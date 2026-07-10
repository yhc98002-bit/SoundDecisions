"""Tests for foley_cw.kernel_provenance — the (cfg, schedule) certification guard."""

from __future__ import annotations

import json

import pytest

from foley_cw.kernel_provenance import (KernelNotCertifiedError, assert_certified_kernel,
                                        find_certification)

LEDGER = {
    "_doc": "test ledger",
    "headline": {"cfg": 1.0, "schedule": "sqrt_down",
                 "token": "CFG_KERNEL_OK(cfg=1, schedule=sqrt_down)",
                 "ok": True, "ratified": True, "scope": "backbone"},
    "deployed": {"cfg": 4.5, "schedule": "sqrt_down",
                 "token": "CFG_KERNEL_OK(cfg=4.5, schedule=sqrt_down)",
                 "ok": True, "ratified": False, "scope": "CANDIDATE"},
}


@pytest.fixture()
def ledger_path(tmp_path):
    p = tmp_path / "certified_kernels.json"
    p.write_text(json.dumps(LEDGER))
    return p


def test_ratified_headline_passes(ledger_path):
    c = assert_certified_kernel(1.0, "sqrt_down", ledger_path)
    assert c["ratified"] is True


def test_candidate_deployed_rejected_when_ratified_required(ledger_path):
    with pytest.raises(KernelNotCertifiedError, match="CANDIDATE-only"):
        assert_certified_kernel(4.5, "sqrt_down", ledger_path)
    # but allowed for an explicitly non-gating diagnostic run
    c = assert_certified_kernel(4.5, "sqrt_down", ledger_path, require_ratified=False)
    assert c["ok"] is True


def test_wrong_schedule_rejected(ledger_path):
    with pytest.raises(KernelNotCertifiedError, match="no kernel certification"):
        assert_certified_kernel(1.0, "constant", ledger_path)
    with pytest.raises(KernelNotCertifiedError, match="no kernel certification"):
        assert_certified_kernel(2.5, "sqrt_down", ledger_path)


def test_missing_ledger_rejected(tmp_path):
    with pytest.raises(KernelNotCertifiedError, match="no certification ledger"):
        assert_certified_kernel(1.0, "sqrt_down", tmp_path / "nope.json")


def test_failed_kernel_rejected(tmp_path):
    bad = dict(LEDGER)
    bad["headline"] = {**LEDGER["headline"], "ok": False, "ratified": False,
                       "token": "CFG_KERNEL_FAIL(cfg=1, schedule=sqrt_down)"}
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(bad))
    with pytest.raises(KernelNotCertifiedError, match="not a certified OK"):
        assert_certified_kernel(1.0, "sqrt_down", p)


def test_find_certification_matches_float_tolerance(ledger_path):
    assert find_certification(1.0, "sqrt_down", ledger_path) is not None
    assert find_certification(4.5, "sqrt_down", ledger_path) is not None
    assert find_certification(3.0, "sqrt_down", ledger_path) is None


def test_token_text_validated_not_just_booleans(tmp_path):
    """Codex T1 finding: an inconsistent ledger (ok=true but a FAIL token, or a
    token missing the schedule= suffix) must be rejected, not trusted."""
    # ok=true but FAIL token
    bad1 = {"h": {"cfg": 1.0, "schedule": "sqrt_down", "ok": True, "ratified": True,
                  "token": "CFG_KERNEL_FAIL(cfg=1, schedule=sqrt_down)"}}
    p1 = tmp_path / "bad1.json"; p1.write_text(json.dumps(bad1))
    with pytest.raises(KernelNotCertifiedError, match="not a certified OK"):
        assert_certified_kernel(1.0, "sqrt_down", p1)
    # token missing the mandatory schedule= suffix
    bad2 = {"h": {"cfg": 1.0, "schedule": "sqrt_down", "ok": True, "ratified": True,
                  "token": "CFG_KERNEL_OK(cfg=1)"}}
    p2 = tmp_path / "bad2.json"; p2.write_text(json.dumps(bad2))
    with pytest.raises(KernelNotCertifiedError, match="missing the mandatory"):
        assert_certified_kernel(1.0, "sqrt_down", p2)
