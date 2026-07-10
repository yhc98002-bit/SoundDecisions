"""Pooled per-layer hidden-state capture for MMAudio (logging contract, manual 1.4).

Captures mean-over-tokens pooled activations from every transformer block of the
MMAudio flow network via forward hooks — WITHOUT modifying third_party/. Hook
targets (verified against third_party/MMAudio/mmaudio/model/networks.py):

  * net.joint_blocks[i]  — returns a tuple (latent, clip_f, text_f); we pool output[0]
  * net.fused_blocks[j]  — returns the latent tensor; we pool it directly

small_16k: 12 joint + 8 fused = 20 layers, hidden_dim 448.

THE TWO-PASS SUBTLETY (load-bearing). ``net.ode_wrapper`` calls ``predict_flow``
TWICE per velocity evaluation whenever ``cfg_strength >= 1.0`` (conditional pass
first, empty pass second; networks.py:332-340) and ONCE when ``cfg_strength < 1.0``.
The tap therefore expects ``passes * n_layers`` captures per armed velocity call,
asserts that count exactly, and keeps the FIRST ``n_layers`` (the conditional pass).
An assertion failure halts the run — never guess which pass is which.

Usage:
    ib = InstrumentedBackend(backend)            # composition; delegates everything
    feats, v = ib.tap_features_at(x_s, s, cond)  # one armed velocity call: (20,448) fp16 + v
    with ib.record_steps():                      # every-step capture (base trajectories)
        traj = score_sde.generate_trajectory(ib, cond, schedule, rng, alpha=0.0, ...)
    ts, step_feats = ib.drain_step_features()    # (n_calls,), (n_calls, 20, 448) fp16
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Optional

import numpy as np

try:
    import torch
except ImportError as e:  # pragma: no cover - torch is present wherever this runs
    raise ImportError("feature_tap requires torch (GPU seam module)") from e


def expected_passes(cfg_strength: float) -> int:
    """predict_flow calls per ode_wrapper velocity evaluation (networks.py:332)."""
    return 1 if cfg_strength < 1.0 else 2


class TapMismatchError(RuntimeError):
    """Capture count did not match passes * n_layers — halt, never guess."""


class FeatureTap:
    """Forward-hook tap pooling per-layer hidden states of an MMAudio module."""

    def __init__(self, net: "torch.nn.Module") -> None:
        self._net = net
        self._joint = list(net.joint_blocks)
        self._fused = list(net.fused_blocks)
        self.n_layers = len(self._joint) + len(self._fused)
        self._armed = False
        self._buffer: list[np.ndarray] = []
        self._handles: list[Any] = []

    # ------------------------------------------------------------------
    def _pool(self, latent: "torch.Tensor") -> np.ndarray:
        # latent: (B, N_tokens, D); pool tokens -> (D,) for B==1 (runner contract).
        if latent.ndim != 3 or latent.shape[0] != 1:
            raise TapMismatchError(f"expected (1, N, D) latent, got {tuple(latent.shape)}")
        pooled = latent.detach().mean(dim=1)[0]
        return pooled.float().cpu().numpy().astype(np.float16)

    def _hook_joint(self, _mod: Any, _inp: Any, out: Any) -> None:
        if self._armed:
            self._buffer.append(self._pool(out[0]))

    def _hook_fused(self, _mod: Any, _inp: Any, out: Any) -> None:
        if self._armed:
            self._buffer.append(self._pool(out))

    def attach(self) -> None:
        if self._handles:
            return
        for blk in self._joint:
            self._handles.append(blk.register_forward_hook(self._hook_joint))
        for blk in self._fused:
            self._handles.append(blk.register_forward_hook(self._hook_fused))

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    # ------------------------------------------------------------------
    @contextmanager
    def armed(self):
        self.attach()
        self._armed = True
        try:
            yield self
        finally:
            self._armed = False

    def clear(self) -> None:
        self._buffer = []

    def pop_call(self, passes: int) -> np.ndarray:
        """Pooled features of ONE velocity call: (n_layers, D) fp16, conditional pass.

        Asserts the buffer holds exactly passes * n_layers captures.
        """
        want = passes * self.n_layers
        if len(self._buffer) != want:
            raise TapMismatchError(
                f"captured {len(self._buffer)} layer outputs, expected {want} "
                f"({passes} passes x {self.n_layers} layers) — cfg branch mismatch?")
        feats = np.stack(self._buffer[: self.n_layers])
        self._buffer = []
        return feats

    def drain_calls(self, passes: int) -> list[np.ndarray]:
        """Group the buffer into per-call (n_layers, D) blocks (conditional pass only)."""
        per_call = passes * self.n_layers
        if len(self._buffer) % per_call != 0:
            raise TapMismatchError(
                f"buffer size {len(self._buffer)} not divisible by {per_call} "
                f"({passes} passes x {self.n_layers} layers)")
        out = []
        for i in range(0, len(self._buffer), per_call):
            out.append(np.stack(self._buffer[i: i + self.n_layers]))
        self._buffer = []
        return out


class InstrumentedBackend:
    """Composition wrapper adding the feature tap to an MMAudioBackend.

    Delegates the FlowModelBackend interface (velocity / decode / sample_prior /
    s_to_t / state_shape / cond builders) to the wrapped backend, so it drops into
    every score_sde primitive unchanged. The wrapped backend's audited file is not
    modified. ``cfg_strength`` stays mutable on the wrapped instance; the expected
    pass count is derived at call time.
    """

    def __init__(self, backend: Any) -> None:
        object.__setattr__(self, "_b", backend)
        object.__setattr__(self, "tap", FeatureTap(backend.net))
        object.__setattr__(self, "_t_log", [])
        object.__setattr__(self, "_recording", False)
        object.__setattr__(self, "_record_passes", None)
        object.__setattr__(self, "_nfe", 0)

    # -- delegation ------------------------------------------------------
    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_b"), name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in ("_b", "tap", "_t_log", "_recording", "_record_passes", "_nfe"):
            object.__setattr__(self, name, value)
        elif name == "cfg_strength" and object.__getattribute__(self, "_recording"):
            raise TapMismatchError("cfg_strength must not change during record_steps() "
                                   "(pass count is frozen for the recording)")
        else:
            setattr(object.__getattribute__(self, "_b"), name, value)

    @property
    def nfe(self) -> int:
        """Velocity calls made through this wrapper (instrumentation included)."""
        return self._nfe

    def velocity(self, x: np.ndarray, t: float, cond: Any) -> np.ndarray:
        if self._recording:
            cur = expected_passes(float(self._b.cfg_strength))
            if cur != self._record_passes:
                raise TapMismatchError(
                    f"cfg pass count changed mid-recording ({self._record_passes} -> {cur})")
        v = self._b.velocity(x, t, cond)
        self._nfe += 1
        if self._recording:
            self._t_log.append(float(t))
        return v

    def decode(self, x: np.ndarray) -> np.ndarray:
        return self._b.decode(x)

    def sample_prior(self, cond: Any, rng: np.random.Generator) -> np.ndarray:
        return self._b.sample_prior(cond, rng)

    # -- grid-point capture (one extra NFE; v is returned for the preview) ----
    def tap_features_at(self, x_s: np.ndarray, s: float, cond: Any
                        ) -> tuple[np.ndarray, np.ndarray]:
        """One armed velocity call at (x_s, s) -> ((n_layers, D) fp16, velocity).

        The returned velocity is the same quantity the Tweedie preview needs
        (x0 = x + (1-t) v), so the preview costs no additional NFE.
        """
        passes = expected_passes(float(self._b.cfg_strength))
        t = self._b.s_to_t.s_to_t(float(s))
        with self.tap.armed():
            self.tap.clear()
            v = self._b.velocity(x_s, t, cond)
            self._nfe += 1  # bypasses self.velocity (avoids polluting _t_log); count here
            feats = self.tap.pop_call(passes)
        return feats, v

    # -- every-step capture (base trajectories only, manual 1.4) --------------
    @contextmanager
    def record_steps(self):
        """Pass count is frozen at entry; partial buffers are cleared on exception."""
        self._record_passes = expected_passes(float(self._b.cfg_strength))
        self._t_log = []
        self._recording = True
        with self.tap.armed():
            self.tap.clear()
            try:
                yield self
            except BaseException:
                self.tap.clear()
                self._t_log = []
                raise
            finally:
                self._recording = False

    def drain_step_features(self) -> tuple[np.ndarray, np.ndarray]:
        """(ts (n_calls,), feats (n_calls, n_layers, D) fp16) for the recorded run."""
        if self._record_passes is None:
            raise TapMismatchError("drain_step_features without a record_steps() recording")
        calls = self.tap.drain_calls(self._record_passes)
        self._record_passes = None
        ts = np.array(self._t_log, dtype=np.float64)
        if len(calls) != ts.size:
            raise TapMismatchError(
                f"{len(calls)} feature calls vs {ts.size} velocity calls recorded")
        self._t_log = []
        return ts, np.stack(calls) if calls else np.zeros((0, self.tap.n_layers, 0), np.float16)
