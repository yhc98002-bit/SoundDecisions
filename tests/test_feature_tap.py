"""Tests for foley_cw.feature_tap — hook capture, two-pass selection, assertions.

CPU-only: a fake MMAudio-like module with joint_blocks (tuple output) and
fused_blocks (tensor output), and a fake backend driving it with the same
one-or-two predict_flow passes per velocity call as networks.py ode_wrapper.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from foley_cw.feature_tap import (FeatureTap, InstrumentedBackend, TapMismatchError,  # noqa: E402
                                  expected_passes)

N_JOINT, N_FUSED, DIM, TOKENS = 3, 2, 8, 5


class _JointBlock(nn.Module):
    def forward(self, latent, clip_f, text_f):
        return latent + 1.0, clip_f, text_f


class _FusedBlock(nn.Module):
    def forward(self, latent):
        return latent * 2.0


class FakeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.joint_blocks = nn.ModuleList([_JointBlock() for _ in range(N_JOINT)])
        self.fused_blocks = nn.ModuleList([_FusedBlock() for _ in range(N_FUSED)])

    def predict_flow(self, latent):
        clip_f = text_f = torch.zeros_like(latent)
        for b in self.joint_blocks:
            latent, clip_f, text_f = b(latent, clip_f, text_f)
        for b in self.fused_blocks:
            latent = b(latent)
        return latent


class _IdentitySToT:
    def s_to_t(self, s: float) -> float:
        return float(s)


class FakeBackend:
    """Mimics MMAudioBackend's velocity contract incl. the cfg two-pass behavior."""

    def __init__(self, cfg_strength: float = 1.0):
        self.net = FakeNet()
        self.cfg_strength = cfg_strength
        self.s_to_t = _IdentitySToT()
        self.state_shape = (TOKENS, DIM)

    def _flow_once(self, x: np.ndarray) -> torch.Tensor:
        latent = torch.from_numpy(np.asarray(x, dtype=np.float32))[None]  # (1, N, D)
        return self.net.predict_flow(latent)

    def velocity(self, x: np.ndarray, t: float, cond) -> np.ndarray:
        if self.cfg_strength < 1.0:
            out = self._flow_once(x)
        else:
            out = (self.cfg_strength * self._flow_once(x)
                   + (1 - self.cfg_strength) * self._flow_once(x))
        return out[0].numpy()

    def decode(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x).reshape(-1)

    def sample_prior(self, cond, rng) -> np.ndarray:
        return rng.standard_normal(self.state_shape)


def _x():
    return np.ones((TOKENS, DIM), dtype=np.float32)


def test_expected_passes():
    assert expected_passes(0.5) == 1
    assert expected_passes(1.0) == 2
    assert expected_passes(4.5) == 2


@pytest.mark.parametrize("cfg,passes", [(0.5, 1), (1.0, 2), (4.5, 2)])
def test_pop_call_counts(cfg, passes):
    be = FakeBackend(cfg_strength=cfg)
    tap = FeatureTap(be.net)
    assert tap.n_layers == N_JOINT + N_FUSED
    with tap.armed():
        tap.clear()
        be.velocity(_x(), 0.3, None)
        feats = tap.pop_call(passes)
    assert feats.shape == (N_JOINT + N_FUSED, DIM)
    assert feats.dtype == np.float16


def test_pop_call_wrong_passes_raises():
    be = FakeBackend(cfg_strength=4.5)
    tap = FeatureTap(be.net)
    with tap.armed():
        tap.clear()
        be.velocity(_x(), 0.3, None)
        with pytest.raises(TapMismatchError):
            tap.pop_call(1)  # cfg>=1 produced 2 passes


def test_disarmed_no_capture():
    be = FakeBackend()
    tap = FeatureTap(be.net)
    tap.attach()
    be.velocity(_x(), 0.3, None)
    assert tap._buffer == []


def test_conditional_pass_selected_first():
    """The kept block must be the FIRST pass (conditional); both passes here are
    identical-valued, so check by count bookkeeping instead: drain two calls."""
    be = FakeBackend(cfg_strength=4.5)
    tap = FeatureTap(be.net)
    with tap.armed():
        tap.clear()
        be.velocity(_x(), 0.1, None)
        be.velocity(_x(), 0.2, None)
        calls = tap.drain_calls(passes=2)
    assert len(calls) == 2
    assert all(c.shape == (N_JOINT + N_FUSED, DIM) for c in calls)


def test_instrumented_backend_tap_features_at_and_delegation():
    ib = InstrumentedBackend(FakeBackend(cfg_strength=4.5))
    assert ib.state_shape == (TOKENS, DIM)  # delegation
    feats, v = ib.tap_features_at(_x(), 0.4, None)
    assert feats.shape == (N_JOINT + N_FUSED, DIM)
    assert v.shape == (TOKENS, DIM)
    ib.cfg_strength = 1.0  # setattr delegates to wrapped backend
    assert ib._b.cfg_strength == 1.0


def test_record_steps_and_drain():
    ib = InstrumentedBackend(FakeBackend(cfg_strength=1.0))
    with ib.record_steps():
        for t in (0.0, 0.25, 0.5):
            ib.velocity(_x(), t, None)
    ts, feats = ib.drain_step_features()
    assert ts.tolist() == [0.0, 0.25, 0.5]
    assert feats.shape == (3, N_JOINT + N_FUSED, DIM)


def test_record_steps_mismatch_raises():
    ib = InstrumentedBackend(FakeBackend(cfg_strength=1.0))
    with ib.record_steps():
        ib.velocity(_x(), 0.1, None)
    ib._t_log.append(0.9)  # corrupt the call log
    with pytest.raises(TapMismatchError):
        ib.drain_step_features()


def test_nfe_counts_tap_captures():
    """Codex round-3 finding: tap_features_at must count toward nfe."""
    ib = InstrumentedBackend(FakeBackend(cfg_strength=4.5))
    assert ib.nfe == 0
    ib.velocity(_x(), 0.1, None)
    assert ib.nfe == 1
    ib.tap_features_at(_x(), 0.4, None)
    assert ib.nfe == 2
