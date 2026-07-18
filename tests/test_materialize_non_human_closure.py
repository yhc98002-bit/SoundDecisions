"""Unit tests for create-only non-human result materialization helpers."""

from __future__ import annotations

import json

import pytest

from scripts import materialize_non_human_closure as materialize


def test_create_only_json_and_copy(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"evidence")
    copied = materialize.copy_create(source, tmp_path / "nested" / "copy.bin")
    assert copied.read_bytes() == b"evidence"
    assert materialize.sha256_file(source) == materialize.sha256_file(copied)
    written = materialize.write_json_create(tmp_path / "report.json", {"b": 2, "a": 1})
    assert json.loads(written.read_text()) == {"a": 1, "b": 2}
    with pytest.raises(FileExistsError):
        materialize.write_json_create(written, {"a": 1})
    with pytest.raises(FileExistsError):
        materialize.copy_create(source, copied)


def test_numeric_summary_and_counts_are_deterministic():
    import numpy as np

    assert materialize._counts(["b", "a", "b"]) == {"a": 1, "b": 2}
    assert materialize._numeric_summary(np.asarray([3.0, 1.0, 2.0])) == {
        "min": 1.0,
        "mean": 2.0,
        "max": 3.0,
    }


def test_require_fails_closed():
    materialize.require(True, "ok")
    with pytest.raises(materialize.MaterializationError, match="evidence"):
        materialize.require(False, "missing evidence")


def test_canonical_output_contract_is_exact_and_duplicate_safe():
    expected = sorted(materialize.EXPECTED_MATERIALIZED_OUTPUTS)
    assert len(expected) == 37
    materialize.require_canonical_output_paths(expected)
    with pytest.raises(materialize.MaterializationError, match="duplicate"):
        materialize.require_canonical_output_paths(expected + [expected[0]])
    with pytest.raises(materialize.MaterializationError, match="37-file"):
        materialize.require_canonical_output_paths(expected[:-1])
