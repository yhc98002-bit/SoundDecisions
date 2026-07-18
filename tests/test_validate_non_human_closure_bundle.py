"""Unit and corruption tests for the closure-bundle validator."""

import hashlib

import pytest

from scripts import validate_non_human_closure_bundle as bundle


def test_checksum_parser_and_validator(tmp_path):
    target = tmp_path / "evidence.json"
    target.write_text("{}\n", encoding="utf-8")
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    checksums = tmp_path / "CHECKSUMS.sha256"
    checksums.write_text(f"{digest}  evidence.json\n", encoding="utf-8")
    assert bundle.validate_checksum_file(tmp_path, checksums, exact=True) == 1


def test_checksum_validator_rejects_corruption(tmp_path):
    target = tmp_path / "evidence.json"
    target.write_text("{}\n", encoding="utf-8")
    checksums = tmp_path / "CHECKSUMS.sha256"
    checksums.write_text(f"{'0' * 64}  evidence.json\n", encoding="utf-8")
    with pytest.raises(bundle.BundleError, match="checksum mismatch"):
        bundle.validate_checksum_file(tmp_path, checksums, exact=True)


def test_bundle_validator_rejects_partial_directory(tmp_path):
    with pytest.raises(bundle.BundleError, match="required closure files missing"):
        bundle.validate_bundle(tmp_path)


def test_probability_simplex_uses_float32_producer_bound():
    bundle.require_probability_simplex([0.5, 0.5000001] + [0.0] * 13, "float32-softmax")
    with pytest.raises(bundle.BundleError, match="float32 bound"):
        bundle.require_probability_simplex([0.5, 0.50001] + [0.0] * 13, "corrupt-softmax")
    with pytest.raises(bundle.BundleError, match="invalid probabilities"):
        bundle.require_probability_simplex([1.1, -0.1] + [0.0] * 13, "negative")
    with pytest.raises(bundle.BundleError, match="invalid probabilities"):
        bundle.require_probability_simplex([1.0000001] + [0.0] * 14, "above-one")
    with pytest.raises(bundle.BundleError, match="width mismatch"):
        bundle.require_probability_simplex([0.5, 0.5], "wrong-width")
