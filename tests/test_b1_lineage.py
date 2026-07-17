"""CPU-only launch-contract tests for the B-1 same-forward lineage gate."""

from __future__ import annotations

import json
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

import foley_cw.b1_lineage as lineage


REPO = Path(__file__).resolve().parents[1]
PROTOCOL = REPO / "experiment" / "non_human_closure" / "PROTOCOL.json"
PROTOCOL_SHA = lineage.sha256_file(PROTOCOL)


def _binding() -> dict:
    return lineage.load_protocol(PROTOCOL, PROTOCOL_SHA)


def _bound_attempt(root: Path, stage: str, name: str) -> Path:
    return lineage.create_bound_attempt(root, stage, name, _binding())


def _packet_arrays(value: float = 1.0, s: float = 0.35) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {
        "x_s": np.full((4, 3), value, dtype=np.float32),
        "model_time": np.asarray(s, dtype=np.float32),
    }
    for prefix in ("conditions", "empty_conditions"):
        for index, name in enumerate(lineage.CONDITION_FIELDS):
            if name in {"clip_f", "text_f"}:
                shape = (1, 2, 3)
            elif name == "sync_f":
                shape = (1, 4, 3)
            else:
                shape = (1, 3)
            arrays[f"{prefix}__{name}"] = np.full(
                shape, value + index + (10 if prefix == "empty_conditions" else 0),
                dtype=np.float32,
            )
    return arrays


def _make_packets(root: Path, clips: tuple[str, ...]) -> Path:
    attempt = _bound_attempt(root, "packets", "packets")
    for clip in clips:
        role = "heldout" if clip == lineage.HELDOUT_CLIP else "calibration"
        for s in lineage.S_POINTS:
            lineage.write_packet_unit(
                attempt, clip_id=clip, role=role, s=s,
                arrays=_packet_arrays(float(int(clip) % 7 + 1), s),
                parent_hashes={"selection_manifest_sha256": "a" * 64},
                provenance={"test": True}, video_sha256="b" * 64,
            )
    lineage.finish_attempt(attempt, "packets", len(clips) * len(lineage.S_POINTS))
    return attempt


def _attention(q, k, v):
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    return out.transpose(1, 2).reshape(q.shape[0], q.shape[2], -1).contiguous()


class _Joint(nn.Module):
    def __init__(self, attention_module):
        super().__init__()
        self.attention_module = attention_module

    def forward(self, latent, clip, text, _global, _extended, _latent_rot, _clip_rot):
        merged = torch.cat((latent, clip, text), dim=1)
        q = merged.unsqueeze(1)
        out = self.attention_module.attention(q, q, q)
        n_latent, n_clip = latent.shape[1], clip.shape[1]
        return (
            latent + out[:, :n_latent],
            clip + out[:, n_latent:n_latent + n_clip],
            text + out[:, n_latent + n_clip:],
        )


class _Fused(nn.Module):
    def __init__(self, attention_module):
        super().__init__()
        self.attention_module = attention_module

    def forward(self, latent, _extended, _rot):
        q = latent.unsqueeze(1)
        return latent + self.attention_module.attention(q, q, q)


class _Net(nn.Module):
    def __init__(self, attention_module):
        super().__init__()
        self.joint_blocks = nn.ModuleList([_Joint(attention_module) for _ in range(2)])
        self.fused_blocks = nn.ModuleList([_Fused(attention_module) for _ in range(2)])
        self.decoder_seen = None

    def predict_flow(self, latent, t, conditions):
        clip, text = conditions.clip_f, conditions.text_f
        global_c = conditions.clip_f_c + conditions.text_f_c + t[:, None]
        extended_c = global_c[:, None, :] + conditions.sync_f
        for block in self.joint_blocks:
            latent, clip, text = block(
                latent, clip, text, global_c, extended_c, None, None
            )
        for block in self.fused_blocks:
            latent = block(latent, extended_c, None)
        return latent

    def ode_wrapper(self, t, latent, conditions, empty, cfg):
        broadcast = t * torch.ones(len(latent), device=latent.device, dtype=latent.dtype)
        return (cfg * self.predict_flow(latent, broadcast, conditions) +
                (1.0 - cfg) * self.predict_flow(latent, broadcast, empty))

    def unnormalize(self, value):
        return value.mul_(2.0).add_(1.0)


@dataclass
class _Conditions:
    clip_f: torch.Tensor
    sync_f: torch.Tensor
    text_f: torch.Tensor
    clip_f_c: torch.Tensor
    text_f_c: torch.Tensor


def _capture(packet: dict[str, np.ndarray], nonce: str,
             output_delta: float = 0.0) -> tuple[dict[str, np.ndarray], dict]:
    attention_module = types.SimpleNamespace(attention=_attention)
    net = _Net(attention_module)
    cond = _Conditions(**{
        name: torch.from_numpy(packet[f"conditions__{name}"]) for name in lineage.CONDITION_FIELDS
    })
    empty = _Conditions(**{
        name: torch.from_numpy(packet[f"empty_conditions__{name}"])
        for name in lineage.CONDITION_FIELDS
    })
    x = torch.from_numpy(packet["x_s"]).unsqueeze(0)
    t = torch.tensor(float(packet["model_time"]), dtype=torch.float32)
    collector = lineage.SameForwardCapture(net, attention_module)
    with collector.armed(nonce):
        velocity = net.ode_wrapper(t, x, cond, empty, 1.0) + output_delta
    capture_arrays, metadata = collector.finish()
    tweedie = x + (1.0 - t) * velocity
    arrays = {
        "packet_x_s_fp32": packet["x_s"].copy(),
        "packet_model_time_fp32": packet["model_time"].copy(),
        "device_latent_fp32": x.numpy().copy(),
        "device_time_fp32": t.numpy().copy(),
        "device_latent_native": x.numpy().copy(),
        "device_time_native": t.numpy().copy(),
        "returned_velocity_native": velocity.numpy().copy(),
        "returned_velocity_fp32": velocity.numpy().copy(),
        "tweedie_latent_normalized_native": tweedie.numpy().copy(),
        "tweedie_latent_fp32": tweedie.numpy().copy(),
        "tweedie_latent_unnormalized_fp32": (tweedie * 2 + 1).numpy().copy(),
        "panns_clipwise_output_fp32": np.full((1, 527), output_delta, np.float32),
        "panns_embedding_fp32": np.full((1, 8), output_delta, np.float32),
        **capture_arrays,
    }
    for key, value in packet.items():
        if key.startswith("conditions__") or key.startswith("empty_conditions__"):
            arrays[key] = value.copy()
    metadata.update({
        "actual_inputs": {}, "returned_velocity": {}, "tweedie_latent": {},
        "conditioning_fields": list(lineage.CONDITION_FIELDS),
        "conditioning_complete": True,
        "tweedie_contract": {"normalized_clone_before_unnormalize": True},
        "runtime": {"ode_wrapper_evaluations": 1, "predict_flow_nfe": 2},
    })
    return arrays, metadata


def _make_replay(root: Path, packets: Path, role: str, name: str, *,
                 delta: float = 0.0, authority: dict | None = None) -> Path:
    attempt = _bound_attempt(root, "replay", name)
    count = 0
    for packet_root in sorted((packets / "units").iterdir()):
        manifest, packet = lineage.validate_packet_unit(packet_root)
        if manifest["identity"]["role"] != role:
            continue
        nonce = lineage.sha256_bytes(
            f"{name}:{manifest['unit_id']}".encode()
        )[:32]
        arrays, capture = _capture(packet, nonce, delta)
        lineage.write_replay_unit(
            attempt, packet_manifest=manifest, packet_arrays=packet,
            packet_unit_root=packet_root, repeat_index=0, arrays=arrays,
            capture_metadata=capture, preview_wav=np.zeros(32, np.float32),
            provenance={
                "hostname": "node0", "device_argument": f"cpu:{name}",
                "cuda_device_uuid": name, "replay_instance_id": name,
                "PYTHONHASHSEED": "0", "calibration_authority": authority,
            },
        )
        count += 1
    lineage.finish_attempt(attempt, "replay", count)
    return attempt


def test_protocol_hash_and_offline_guard(monkeypatch, tmp_path):
    with pytest.raises(lineage.ArtifactValidationError, match="protocol hash mismatch"):
        lineage.load_protocol(PROTOCOL, "0" * 64)
    contract = _binding()["pilot"]["asset_contract"]
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    with pytest.raises(lineage.ArtifactValidationError, match="offline guard"):
        lineage.require_offline_environment(contract)
    attempt = _bound_attempt(tmp_path, "packets", "bound")
    lineage.finish_attempt(attempt, "packets", 0)
    assert lineage.validate_attempt(attempt, expected_protocol_sha256=PROTOCOL_SHA)["status"] == "PASS"


def test_environment_provenance_serializes_cuda_uuid(monkeypatch):
    class NonJsonCudaUuid:
        def __str__(self):
            return "GPU-test-uuid"

    props = types.SimpleNamespace(uuid=NonJsonCudaUuid(), total_memory=123)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _index: "test-gpu")
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _index: props)
    monkeypatch.setattr(
        lineage.subprocess,
        "check_output",
        lambda *args, **kwargs: "0, GPU-test-uuid, test-gpu, 0, 123, 100\n",
    )

    provenance = lineage.environment_provenance("cuda:0")
    assert provenance["cuda_device_uuid"] == "GPU-test-uuid"
    json.dumps(provenance, allow_nan=False)


def test_same_forward_real_api_contract_and_historical_pool():
    packet = _packet_arrays()
    arrays, metadata = _capture(packet, "nonce")
    assert [row["pass_role"] for row in metadata["passes"]] == list(lineage.PASS_ROLES)
    for row in metadata["passes"]:
        assert arrays[row["actual_broadcast_time"]].shape == (1,)
        assert set(row["conditions_as_consumed"]) == set(lineage.CONDITION_FIELDS)
        assert row["network_consumed"]["global_condition"] in arrays
        assert row["network_consumed"]["extended_condition"] in arrays
    for row in metadata["tokens"]:
        assert row["pool_operation_original"] == "latent.detach().mean(dim=1)[0].float()"
        assert np.array_equal(arrays[row["pooled_original"]], arrays[row["pooled_repaired"]])
    for row in metadata["attention"]:
        assert row["actual_latent_query_output"] in arrays
        if row["site"].startswith("joint."):
            assert row["probability_summary"] in arrays
            assert "RECOMPUTED_DERIVED" in row["probability_map_provenance"]
        else:
            assert row["probability_summary"] is None
            assert "NOT_APPLICABLE" in row["probability_map_provenance"]


def test_real_replay_preserves_normalized_tweedie_before_inplace_unnormalize(tmp_path):
    packets = _make_packets(tmp_path / "packets", ("3780",))
    packet_root = next((packets / "units").iterdir())
    attention_module = types.SimpleNamespace(attention=_attention)
    net = _Net(attention_module)

    class Features:
        def decode(self, value):
            self.seen = value.detach().clone()
            return value.mean(dim=-1)

        def vocode(self, value):
            return value.reshape(1, -1)

    backend = types.SimpleNamespace(
        dtype=torch.float32, net=net, feature_utils=Features()
    )

    class Panns:
        def __call__(self, value):
            return {"clipwise_output": torch.zeros((1, 527)), "embedding": torch.zeros((1, 8))}

    arrays, capture, _ = lineage._real_replay_one(
        backend, Panns(), packet_root, 0, "cpu", replay_instance_id="mock",
        condition_type=_Conditions, attention_module=attention_module,
    )
    normalized = arrays["tweedie_latent_normalized_native"]
    assert np.array_equal(arrays["tweedie_latent_fp32"], normalized)
    assert np.allclose(backend.feature_utils.seen.numpy(), normalized * 2 + 1)
    assert capture["tweedie_contract"]["normalized_clone_before_unnormalize"] is True


def test_schema_corruption_nonfinite_and_metric_recompute_rejected(tmp_path):
    packets = _make_packets(tmp_path / "p", ("3780",))
    replay = _make_replay(tmp_path / "r", packets, "calibration", "a")
    unit = next((replay / "units").iterdir())
    manifest = json.loads((unit / "manifest.json").read_text())
    manifest["comparisons"][0]["metrics"]["relative_l2"] = 1.0
    (unit / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (unit / "COMPLETED.json").unlink()
    lineage._finish_unit(unit, "replay", {
        **manifest["identity"], "capture_nonce": manifest["capture"]["capture_nonce"]
    })
    with pytest.raises(lineage.ArtifactValidationError, match="comparison metrics"):
        lineage.validate_replay_unit(unit)
    with pytest.raises(lineage.ArtifactValidationError, match="nonfinite"):
        lineage.describe_array(np.asarray([np.nan], np.float32))


def test_multireplay_calibration_and_heldout_authority(tmp_path):
    calibration_packets = _make_packets(tmp_path / "pc", lineage.CALIBRATION_CLIPS)
    replay_a = _make_replay(tmp_path / "rca", calibration_packets, "calibration", "a")
    replay_b = _make_replay(
        tmp_path / "rcb", calibration_packets, "calibration", "b", delta=0.01
    )
    with pytest.raises(lineage.ArtifactValidationError, match="at least two fresh"):
        lineage.calibrate_attempt(
            [replay_a], tmp_path / "reports", "bad", protocol_path=PROTOCOL,
            protocol_sha256=PROTOCOL_SHA,
        )
    calibration = lineage.calibrate_attempt(
        [replay_a, replay_b], tmp_path / "reports", "calibration",
        protocol_path=PROTOCOL, protocol_sha256=PROTOCOL_SHA,
    )
    tolerance_hash = lineage.sha256_file(calibration / "TOLERANCE.json")
    authority = lineage.validate_calibration_authority(
        calibration, tolerance_hash, expected_protocol_sha256=PROTOCOL_SHA
    )
    tolerance = json.loads((calibration / "TOLERANCE.json").read_text())
    assert tolerance["cross_replay_rows"] > 0
    assert any(key.startswith(lineage.CROSS_REPLAY_COMPARISON + ":")
               for key in tolerance["thresholds"])

    heldout_packets = _make_packets(tmp_path / "ph", (lineage.HELDOUT_CLIP,))
    heldout_a = _make_replay(
        tmp_path / "rha", heldout_packets, "heldout", "ha", authority=authority
    )
    heldout_b = _make_replay(
        tmp_path / "rhb", heldout_packets, "heldout", "hb", delta=0.5,
        authority=authority,
    )
    report_root = lineage.heldout_attempt(
        [heldout_a, heldout_b], calibration, tmp_path / "reports", "heldout",
        tolerance_sha256=tolerance_hash, protocol_path=PROTOCOL,
        protocol_sha256=PROTOCOL_SHA,
    )
    report = json.loads((report_root / "HELDOUT_REPORT.json").read_text())
    assert report["status"] == "ENGINEERING_FAILURE"
    assert report["failure_count"] > 0
    with pytest.raises(lineage.ArtifactValidationError, match="mandatory tolerance SHA"):
        lineage.validate_calibration_authority(
            calibration, "0" * 64, expected_protocol_sha256=PROTOCOL_SHA
        )


def test_calibration_physically_rejects_heldout(tmp_path):
    packets = _make_packets(tmp_path / "p", (lineage.HELDOUT_CLIP,))
    first = _make_replay(tmp_path / "r1", packets, "heldout", "a")
    second = _make_replay(tmp_path / "r2", packets, "heldout", "b")
    with pytest.raises(lineage.HeldoutLeakageError, match="physically rejected"):
        lineage.calibrate_attempt(
            [first, second], tmp_path / "out", "cal", protocol_path=PROTOCOL,
            protocol_sha256=PROTOCOL_SHA,
        )


def test_phase1_rng_and_higher_quantile_contract():
    assert np.array_equal(
        lineage.phase1_rng(0, "3780", "ind", 12).standard_normal(8),
        lineage.phase1_rng(0, "3780", "ind", 12).standard_normal(8),
    )
    raw, tolerance = lineage.higher_quantile_times_two([0.0, 0.1, 0.2, 0.3])
    assert raw == pytest.approx(0.3)
    assert tolerance == pytest.approx(0.6)
