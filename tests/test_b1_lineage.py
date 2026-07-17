"""CPU-safe tests for the B-1 same-forward lineage engineering gate."""

from __future__ import annotations

import json
import types
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

import foley_cw.b1_lineage as lineage


def _packet_arrays(value: float = 1.0) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {
        "x_s": np.full((4, 3), value, dtype=np.float32),
        "model_time": np.asarray(0.35, dtype=np.float32),
    }
    for prefix in ("conditions", "empty_conditions"):
        for index, name in enumerate(lineage.CONDITION_FIELDS):
            shape = (1, 2, 3) if name.endswith("_f") else (1, 3)
            arrays[f"{prefix}__{name}"] = np.full(
                shape, value + index, dtype=np.float32
            )
    return arrays


def _make_packet_attempt(tmp_path: Path, clips: tuple[str, ...],
                         points: tuple[float, ...] = lineage.S_POINTS) -> Path:
    root = lineage.create_attempt(tmp_path, "packets", "packets")
    for clip in clips:
        role = "heldout" if clip == lineage.HELDOUT_CLIP else "calibration"
        for s in points:
            arrays = _packet_arrays(float(int(clip) % 7))
            arrays["model_time"] = np.asarray(s, dtype=np.float32)
            lineage.write_packet_unit(
                root, clip_id=clip, role=role, s=s, arrays=arrays,
                parent_hashes={"selection_manifest_sha256": "a" * 64},
                provenance={"test": True}, video_sha256="b" * 64,
            )
    lineage.finish_attempt(root, "packets", len(clips) * len(points))
    return root


def _fake_capture(diff: float = 0.0) -> tuple[dict[str, np.ndarray], dict]:
    original = np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32)
    repaired = original.copy()
    repaired[0, 0] += diff
    arrays = {
        "packet_x_s_fp32": np.ones((4, 3), dtype=np.float32),
        "packet_model_time_fp32": np.asarray(0.35, dtype=np.float32),
        "device_latent_fp32": np.ones((1, 4, 3), dtype=np.float32),
        "device_time_fp32": np.asarray(0.35, dtype=np.float32),
        "returned_velocity_fp32": np.zeros((1, 4, 3), dtype=np.float32),
        "tweedie_latent_fp32": np.ones((1, 4, 3), dtype=np.float32),
        "pooled_original_fp32__conditional__joint_0": original,
        "pooled_repaired_fp32__conditional__joint_0": repaired,
        "panns_clipwise_output_fp32": np.zeros((1, 527), dtype=np.float32),
        "panns_embedding_fp32": np.zeros((1, 2048), dtype=np.float32),
    }
    capture = {
        "capture_nonce": "c" * 32,
        "one_ode_wrapper_evaluation": True,
        "expected_passes": 2,
        "observed_passes": 2,
        "pass_roles": list(lineage.PASS_ROLES),
        "post_block_count": 1,
        "post_block_count_by_role": {"conditional": 1, "empty": 0},
        "joint_blocks": 1,
        "fused_blocks": 1,
        "selected_attention_sites": ["joint.0", "fused.0"],
        "attention_capture_count": 0,
        "tokens": [{
            "capture_nonce": "c" * 32,
            "pass_index": 0,
            "pass_role": "conditional",
            "site": "joint.0",
            "pooled_original": "pooled_original_fp32__conditional__joint_0",
            "pooled_repaired": "pooled_repaired_fp32__conditional__joint_0",
            "pool_operation_original": "native_tensor.float().mean(dim=1)",
            "pool_operation_repaired": "torch.mean(native_tensor.float(), dim=1)",
        }],
        "attention": [],
    }
    return arrays, capture


def _make_replay_attempt(tmp_path: Path, packet_attempt: Path, role: str,
                         name: str, diff: float = 0.0) -> Path:
    root = lineage.create_attempt(tmp_path, "replay", name)
    count = 0
    for packet_root in sorted((packet_attempt / "units").iterdir()):
        manifest, packet_arrays = lineage.validate_packet_unit(packet_root)
        if manifest["identity"]["role"] != role:
            continue
        arrays, capture = _fake_capture(diff)
        lineage.write_replay_unit(
            root, packet_manifest=manifest, packet_arrays=packet_arrays,
            packet_unit_root=packet_root, repeat_index=0, arrays=arrays,
            capture_metadata=capture, preview_wav=np.zeros(32, dtype=np.float32),
            provenance={"device": "cpu", "repeat_count": 1},
        )
        count += 1
    lineage.finish_attempt(root, "replay", count)
    return root


class _Joint(nn.Module):
    def __init__(self, attention_module):
        super().__init__()
        self.attention_module = attention_module

    def forward(self, latent, clip, text):
        q = latent.unsqueeze(1)
        out = self.attention_module.attention(q, q, q)
        return latent + out, clip, text


class _Fused(nn.Module):
    def __init__(self, attention_module):
        super().__init__()
        self.attention_module = attention_module

    def forward(self, latent):
        q = latent.unsqueeze(1)
        return latent + self.attention_module.attention(q, q, q)


def _fake_attention(q, k, v):
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    return out.transpose(1, 2).reshape(q.shape[0], q.shape[2], -1).contiguous()


class _Net(nn.Module):
    def __init__(self, attention_module):
        super().__init__()
        self.joint_blocks = nn.ModuleList([_Joint(attention_module) for _ in range(2)])
        self.fused_blocks = nn.ModuleList([_Fused(attention_module) for _ in range(2)])

    def predict_flow(self, latent):
        clip = text = torch.zeros_like(latent)
        for block in self.joint_blocks:
            latent, clip, text = block(latent, clip, text)
        for block in self.fused_blocks:
            latent = block(latent)
        return latent

    def ode_wrapper(self, _time, latent, _conditions, _empty, cfg):
        # Real MMAudio executes both branches for cfg >= 1, including cfg == 1.
        return cfg * self.predict_flow(latent) + (1.0 - cfg) * self.predict_flow(latent)


def test_same_forward_capture_actual_attention_and_equivalent_pooling():
    attention_module = types.SimpleNamespace(attention=_fake_attention)
    original = attention_module.attention
    net = _Net(attention_module)
    capture = lineage.SameForwardCapture(net, attention_module)
    x = torch.arange(12, dtype=torch.float32).reshape(1, 4, 3)
    with capture.armed("nonce"):
        velocity = net.ode_wrapper(torch.tensor(0.35), x, None, None, 1.0)
    arrays, metadata = capture.finish()
    assert velocity.shape == x.shape
    assert attention_module.attention is original
    assert metadata["observed_passes"] == 2
    assert metadata["post_block_count_by_role"] == {"conditional": 4, "empty": 4}
    assert metadata["selected_attention_sites"] == ["fused.1", "joint.0", "joint.1"]
    assert metadata["attention_capture_count"] == 6
    for record in metadata["tokens"]:
        assert np.array_equal(arrays[record["pooled_original"]],
                              arrays[record["pooled_repaired"]])
        assert record["capture_nonce"] == "nonce"
    for record in metadata["attention"]:
        assert "RECOMPUTED_DERIVED" in record["probability_map_provenance"]
        assert record["actual_attention_output"] in arrays
        assert record["q"] in arrays and record["k"] in arrays and record["v"] in arrays


def test_phase1_rng_is_exact_and_deterministic():
    got1 = lineage.phase1_rng(0, "3780", "ind", 12).standard_normal(8)
    got2 = lineage.phase1_rng(0, "3780", "ind", 12).standard_normal(8)
    entropy = [0, zlib_crc("3780"), zlib_crc("ind"), zlib_crc("12")]
    expected = np.random.default_rng(np.random.SeedSequence(entropy)).standard_normal(8)
    assert np.array_equal(got1, got2)
    assert np.array_equal(got1, expected)


def zlib_crc(value: str) -> int:
    import zlib
    return zlib.crc32(value.encode())


def test_packet_schema_complete_and_immutable(tmp_path):
    root = lineage.create_attempt(tmp_path, "packets", "a")
    arrays = _packet_arrays()
    unit = lineage.write_packet_unit(
        root, clip_id="3780", role="calibration", s=0.35, arrays=arrays,
        parent_hashes={"selection_manifest_sha256": "a" * 64},
        provenance={"test": True}, video_sha256="b" * 64,
    )
    manifest, loaded = lineage.validate_packet_unit(unit)
    assert manifest["schema"] == lineage.PACKET_SCHEMA
    assert set(loaded) == set(arrays)
    with pytest.raises(lineage.ImmutableArtifactError):
        lineage.write_packet_unit(
            root, clip_id="3780", role="calibration", s=0.35, arrays=arrays,
            parent_hashes={}, provenance={}, video_sha256="b" * 64,
        )
    bad = dict(arrays)
    bad.pop("empty_conditions__text_f_c")
    with pytest.raises(lineage.ArtifactValidationError):
        lineage.write_packet_unit(
            root, clip_id="1813", role="calibration", s=0.35, arrays=bad,
            parent_hashes={}, provenance={}, video_sha256="b" * 64,
        )


def test_partial_and_corrupt_attempts_rejected(tmp_path):
    partial = lineage.create_attempt(tmp_path, "packets", "partial")
    with pytest.raises(lineage.ArtifactValidationError, match="no completion"):
        lineage.validate_attempt(partial)
    packet = _make_packet_attempt(tmp_path / "complete", ("3780",), (0.05,))
    unit_npz = next((packet / "units").glob("*/arrays.npz"))
    with unit_npz.open("ab") as handle:
        handle.write(b"corruption")
    with pytest.raises(lineage.ArtifactValidationError, match="inventory/hash mismatch"):
        lineage.validate_attempt(packet)


def test_conflicting_scientific_identity_rejected(tmp_path):
    root = lineage.create_attempt(tmp_path, "packets", "conflict")
    for clip in ("3780", "1813"):
        lineage.write_packet_unit(
            root, clip_id=clip, role="calibration", s=0.05, arrays=_packet_arrays(),
            parent_hashes={}, provenance={}, video_sha256="b" * 64,
        )
    first, second = sorted((root / "units").iterdir())
    first_identity = json.loads((first / "manifest.json").read_text())["identity"]
    second_manifest = json.loads((second / "manifest.json").read_text())
    second_manifest["identity"] = first_identity
    (second / "manifest.json").write_text(json.dumps(second_manifest, indent=2, sort_keys=True) + "\n")
    (second / "COMPLETED.json").unlink()
    lineage._finish_unit(second, "packets", first_identity)
    lineage.finish_attempt(root, "packets", 2)
    with pytest.raises(lineage.ArtifactValidationError, match="duplicate scientific identity"):
        lineage.validate_attempt(root)


def test_higher_quantile_times_two_uses_higher():
    raw, tolerance = lineage.higher_quantile_times_two([0.0, 0.1, 0.2, 0.3])
    assert raw == pytest.approx(0.3)
    assert tolerance == pytest.approx(0.6)


def test_calibration_physically_rejects_heldout(tmp_path):
    packets = _make_packet_attempt(tmp_path / "p", (lineage.HELDOUT_CLIP,))
    replay = _make_replay_attempt(tmp_path / "r", packets, "heldout", "heldout")
    with pytest.raises(lineage.HeldoutLeakageError, match="physically rejected"):
        lineage.calibrate_attempt(replay, tmp_path / "out", "cal")


def test_calibration_and_heldout_hash_immutable_and_localized(tmp_path):
    calibration_packets = _make_packet_attempt(tmp_path / "pc", lineage.CALIBRATION_CLIPS)
    calibration_replay = _make_replay_attempt(
        tmp_path / "rc", calibration_packets, "calibration", "calibration"
    )
    calibration = lineage.calibrate_attempt(
        calibration_replay, tmp_path / "reports", "tolerance"
    )
    tolerance = calibration / "TOLERANCE.json"
    frozen_bytes = tolerance.read_bytes()
    frozen_hash = lineage.sha256_file(tolerance)

    heldout_packets = _make_packet_attempt(tmp_path / "ph", (lineage.HELDOUT_CLIP,))
    heldout_replay = _make_replay_attempt(
        tmp_path / "rh", heldout_packets, "heldout", "heldout", diff=0.25
    )
    report_root = lineage.heldout_attempt(
        heldout_replay, tolerance, tmp_path / "reports", "heldout",
        expected_tolerance_sha256=frozen_hash,
    )
    assert tolerance.read_bytes() == frozen_bytes
    report = json.loads((report_root / "HELDOUT_REPORT.json").read_text())
    assert report["status"] == "ENGINEERING_FAILURE"
    assert report["tolerance_sha256"] == frozen_hash
    assert report["failure_count"] > 0
    failure = report["failures"][0]
    assert failure["clip_id"] == lineage.HELDOUT_CLIP
    assert failure["pass_role"] == "conditional"
    assert failure["site"] == "joint.0"
    assert failure["metric"] in {"relative_l2", "max_abs"}


def test_forbidden_quantization_order_cannot_gate(tmp_path):
    packets = _make_packet_attempt(tmp_path / "p", ("3780",), (0.05,))
    replay = _make_replay_attempt(tmp_path / "r", packets, "calibration", "one")
    unit = next((replay / "units").iterdir())
    manifest = json.loads((unit / "manifest.json").read_text())
    manifest["gating_policy"]["forbidden_comparison_present"] = True
    (unit / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (unit / "COMPLETED.json").unlink()
    lineage._finish_unit(unit, "replay", manifest["identity"])
    with pytest.raises(lineage.ArtifactValidationError, match="forbidden reduction-order"):
        lineage.validate_replay_unit(unit)
