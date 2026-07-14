"""Real MMAudio v1 backend for foley_cw (GPU; torch + MMAudio required).

This module is NOT imported by the numpy-only core; import it explicitly only on a GPU box
with torch + the vendored `third_party/MMAudio` installed. It wires MMAudio's flow-matching
network as a foley_cw `FlowModelBackend`, so the model-agnostic commitment/readout/SDE
machinery (score_sde.integrate_segment / fork_tail) drives the real model unchanged.

AUDITED CONVENTION (Phase 0.1/0.2; against third_party/MMAudio/mmaudio/model/flow_matching.py
and eval_utils.py / networks.py, MMAudio commit pinned in third_party/MMAudio):

  * Interpolant: x_t = (1 - t) * x0 + t * x1, with x0 = noise (torch.randn), x1 = data latent
    (min_sigma = 0 in demo.py / FlowMatching). So t=0 is noise, t=1 is audio -> ASCENDING,
    and progress s maps to native time t by the IDENTITY (s == t). The plan's "some models
    integrate t: 1->0" risk does NOT apply: MMAudio integrates `to_data` t: 0 -> 1, Euler
    x <- x + dt * v with dt > 0.
  * Velocity target v = x1 - x0 (rectified-flow linear interpolant). This is exactly the
    convention foley_cw.synthetic_backend and score_sde were derived for, so:
        score_from_velocity(v, x, t) = (t*v - x)/(1 - t)   is EXACT (no sign/direction flip),
        tweedie_x0(v, x, t)          = x + (1 - t)*v,
        marginal-preserving fork drift = v + 1/2 sigma^2 * score.
    => time_map.IdentitySToT is the verified MMAudio mapping.
  * net.ode_wrapper(t, x, cond, empty_cond, cfg_strength): at cfg_strength == 1.0 it returns
    the PURE conditional velocity predict_flow(x, t, cond) (no classifier-free guidance), which
    is the conditional-expectation velocity of the conditional linear-interpolant marginal ->
    the score identity above holds exactly. The fork SDE therefore defaults to cfg=1.0. The
    deployed cfg (e.g. 4.5) yields a TILTED PSEUDO-SCORE with no exact marginal-preservation
    guarantee. Per the CFG doctrine (experiment/LONG_RANGE_EXPERIMENT_PLAN.md section 1.2,
    superseding earlier wording here), forks at cfg > 1 ARE permitted for headline maps, but
    only after that cfg passes Gate A at that cfg: small-alpha continuity, fork audio
    validity, nontrivial diversity, AND distributional match of fork-finals vs independent
    ODE-finals calibrated against the cfg=1.0 reference (foley_cw/gate_a.py). Gate-A failure
    emits CFG_KERNEL_FAIL(cfg=x) and routes to the cfg=1.0 fallback. Phase 0.2 validated the
    exact-kernel regime; a Phase-0.2 failure must emit FIX_SCORE_CONVERSION.

State convention: foley_cw operates in numpy on the NORMALIZED latent (shape
(latent_seq_len, latent_dim)); the network predicts flow in normalized space (generate() only
calls net.unnormalize() at the very end before VAE decode). velocity()/decode() bridge
numpy<->torch at the boundary and run the network on the GPU per call. sample_prior() returns
a standard-normal latent (the t=0 prior).
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch

from .model_adapter import FlowModelBackend
from .time_map import IdentitySToT

# Make the vendored MMAudio importable without requiring an editable install.
_MMAUDIO_DIR = os.path.abspath(os.environ.get(
    "FOLEY_CW_MMAUDIO_ROOT",
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "third_party",
        "MMAudio",
    ),
))
if os.path.isdir(_MMAUDIO_DIR) and _MMAUDIO_DIR not in sys.path:
    sys.path.insert(0, _MMAUDIO_DIR)


@dataclass
class MMAudioCond:
    """Per-video (or per-prompt) conditioning for foley_cw, holding the preprocessed MMAudio
    conditions plus the empty (CFG) conditions. `video_id` is the bookkeeping key."""

    video_id: str
    conditions: Any           # PreprocessedConditions
    empty_conditions: Any     # PreprocessedConditions
    prompt: str = ""


class MMAudioBackend(FlowModelBackend):
    """MMAudio v1 as a foley_cw FlowModelBackend (text- or video-conditioned)."""

    def __init__(
        self,
        variant: str = "small_16k",
        device: str = "cuda",
        full_precision: bool = True,   # float32 for the SDE-validation precision
        cfg_strength: float = 1.0,     # fork SDE uses pure conditional velocity (see module docstring)
        num_steps: int = 25,
        duration_sec: float = 8.0,
        weights_root: Optional[str] = None,
        enable_conditions: bool = False,  # False = no CLIP/synchformer (empty/unconditional cond);
        #                                   sufficient for the Phase-0.1/0.2 trajectory+SDE crux and
        #                                   avoids the CLIP download. Set True for text/video maps.
    ) -> None:
        from mmaudio.eval_utils import ModelConfig, all_model_cfg
        from mmaudio.model.flow_matching import FlowMatching
        from mmaudio.model.networks import get_my_mmaudio
        from mmaudio.model.utils.features_utils import FeaturesUtils

        if variant not in all_model_cfg:
            raise ValueError(f"unknown MMAudio variant {variant!r}; choices: {list(all_model_cfg)}")
        cfg: ModelConfig = all_model_cfg[variant]
        # Resolve weight paths relative to the vendored MMAudio dir (where we downloaded them).
        root = weights_root or _MMAUDIO_DIR
        cfg.model_path = _resolve(root, cfg.model_path)
        cfg.vae_path = _resolve(root, cfg.vae_path)
        if cfg.bigvgan_16k_path is not None:
            cfg.bigvgan_16k_path = _resolve(root, cfg.bigvgan_16k_path)
        cfg.synchformer_ckpt = _resolve(root, cfg.synchformer_ckpt)

        self.variant = variant
        self.device = device
        self.dtype = torch.float32 if full_precision else torch.bfloat16
        self.cfg_strength = float(cfg_strength)
        self._mode = cfg.mode
        self.s_to_t = IdentitySToT  # AUDITED: MMAudio integrates t:0(noise)->1(audio), t == s

        net = get_my_mmaudio(cfg.model_name).to(device, self.dtype).eval()
        net.load_weights(torch.load(cfg.model_path, map_location=device, weights_only=True))
        seq_cfg = cfg.seq_cfg
        seq_cfg.duration = duration_sec
        net.update_seq_lengths(seq_cfg.latent_seq_len, seq_cfg.clip_seq_len, seq_cfg.sync_seq_len)
        self.net = net
        self.seq_cfg = seq_cfg

        self.enable_conditions = bool(enable_conditions)
        self.feature_utils = FeaturesUtils(
            tod_vae_ckpt=cfg.vae_path,
            synchformer_ckpt=cfg.synchformer_ckpt,
            enable_conditions=self.enable_conditions,
            mode=cfg.mode,
            bigvgan_vocoder_ckpt=cfg.bigvgan_16k_path,
            need_vae_encoder=False,
        ).to(device, self.dtype).eval()

        self.fm = FlowMatching(min_sigma=0, inference_mode="euler", num_steps=num_steps)
        self._latent_seq_len = int(net.latent_seq_len)
        self._latent_dim = int(net.latent_dim)

    # -- FlowModelBackend interface ----------------------------------------------------
    @property
    def state_shape(self) -> tuple[int, ...]:
        return (self._latent_seq_len, self._latent_dim)

    def _to_torch(self, x: np.ndarray) -> torch.Tensor:
        # (N, C) -> (1, N, C) on device
        t = torch.from_numpy(np.ascontiguousarray(x)).to(self.device, self.dtype)
        return t.unsqueeze(0)

    @torch.inference_mode()
    def sample_prior(self, cond: MMAudioCond, rng: np.random.Generator) -> np.ndarray:
        # Standard-normal latent (t=0 prior). Seed a torch generator from the numpy rng for
        # reproducibility while keeping foley_cw's numpy rng as the single seed source.
        seed = int(rng.integers(0, 2**31 - 1))
        g = torch.Generator(device=self.device).manual_seed(seed)
        x = torch.randn(1, self._latent_seq_len, self._latent_dim,
                        device=self.device, dtype=self.dtype, generator=g)
        return x.squeeze(0).float().cpu().numpy()

    @torch.inference_mode()
    def velocity(self, x: np.ndarray, t: float, cond: MMAudioCond) -> np.ndarray:
        xt = self._to_torch(x)
        tt = torch.tensor(float(t), device=self.device, dtype=self.dtype)
        v = self.net.ode_wrapper(tt, xt, cond.conditions, cond.empty_conditions, self.cfg_strength)
        return v.squeeze(0).float().cpu().numpy()

    @torch.inference_mode()
    def decode(self, x: np.ndarray) -> np.ndarray:
        xt = self._to_torch(x)
        x1 = self.net.unnormalize(xt)
        spec = self.feature_utils.decode(x1)
        audio = self.feature_utils.vocode(spec)  # (1, T) or (1, 1, T)
        audio = audio.float().cpu().numpy().reshape(-1)
        return audio

    # -- conditioning builders ---------------------------------------------------------
    @torch.inference_mode()
    def make_empty_cond(self, video_id: str = "empty") -> MMAudioCond:
        """Unconditional cond from empty CLIP/sync/text sequences only — needs NO CLIP or
        synchformer (works with enable_conditions=False). Sufficient for the Phase-0.1/0.2
        trajectory-access + velocity->score SDE crux, which tests the flow network mechanics,
        not the conditioning content."""
        net = self.net
        conditions = net.preprocess_conditions(
            net.get_empty_clip_sequence(1),
            net.get_empty_sync_sequence(1),
            net.get_empty_string_sequence(1),
        )
        empty_conditions = net.get_empty_conditions(1)
        return MMAudioCond(video_id=video_id, conditions=conditions,
                           empty_conditions=empty_conditions, prompt="")

    @torch.inference_mode()
    def make_text_cond(self, prompt: str = "", negative_prompt: str = "",
                       video_id: Optional[str] = None) -> MMAudioCond:
        """Text-to-audio conditioning (empty clip/sync sequences). Sufficient for Phase 0.1/0.2
        trajectory-access + SDE validation (no FoleyBench video required)."""
        net = self.net
        clip_features = net.get_empty_clip_sequence(1)
        sync_features = net.get_empty_sync_sequence(1)
        text_features = (self.feature_utils.encode_text([prompt]) if prompt
                         else net.get_empty_string_sequence(1))
        conditions = net.preprocess_conditions(clip_features, sync_features, text_features)
        neg = (self.feature_utils.encode_text([negative_prompt]) if negative_prompt else None)
        empty_conditions = net.get_empty_conditions(1, negative_text_features=neg)
        return MMAudioCond(video_id=video_id or f"text:{prompt[:24]}",
                           conditions=conditions, empty_conditions=empty_conditions, prompt=prompt)

    @torch.inference_mode()
    def make_video_cond(self, video_path: str, prompt: str = "", negative_prompt: str = "",
                        duration_sec: Optional[float] = None,
                        video_id: Optional[str] = None) -> MMAudioCond:
        """Video-conditioned conditioning (Phases 1-3 maps). Encodes CLIP + sync features."""
        from mmaudio.eval_utils import load_video
        net = self.net
        dur = duration_sec or self.seq_cfg.duration
        info = load_video(video_path, dur)
        clip_frames = info.clip_frames.unsqueeze(0)
        sync_frames = info.sync_frames.unsqueeze(0)
        fu = self.feature_utils
        clip_features = fu.encode_video_with_clip(clip_frames.to(self.device, self.dtype))
        sync_features = fu.encode_video_with_sync(sync_frames.to(self.device, self.dtype))
        # Sequence-length normalization at the FEATURE level (post-encode): clips
        # whose duration/fps rounds off yield a feature count one off the required
        # length (e.g. CLIP 63 vs 64; sync 184 vs 192 — the synchformer downsamples,
        # so this cannot be fixed by padding input frames), which otherwise crashes
        # preprocess_conditions on real FoleyBench clips. Pad by repeating the last
        # feature (tail of the clip) or truncate to the exact expected counts.
        # Deterministic; touches only the boundary, not the conditioning content.
        clip_features = _fit_seq(clip_features, self.seq_cfg.clip_seq_len)
        sync_features = _fit_seq(sync_features, self.seq_cfg.sync_seq_len)
        text_features = (fu.encode_text([prompt]) if prompt else net.get_empty_string_sequence(1))
        conditions = net.preprocess_conditions(clip_features, sync_features, text_features)
        neg = (fu.encode_text([negative_prompt]) if negative_prompt else None)
        empty_conditions = net.get_empty_conditions(1, negative_text_features=neg)
        return MMAudioCond(video_id=video_id or os.path.basename(video_path),
                           conditions=conditions, empty_conditions=empty_conditions, prompt=prompt)


def _fit_seq(feats, target_len: int):
    """Pad (repeat last) or truncate an encoded-feature tensor (B, T, D) along the
    sequence axis (dim=1) to target_len. Normalizes the off-by-one feature counts
    MMAudio's CLIP/sync encoders produce for clips whose duration/fps rounds."""
    t = feats.shape[1]
    if t == target_len:
        return feats
    if t > target_len:
        return feats[:, :target_len]
    import torch
    pad = feats[:, -1:].repeat((1, target_len - t) + (1,) * (feats.ndim - 2))
    return torch.cat([feats, pad], dim=1)


def _resolve(root: str, p) -> str:
    """Resolve a MMAudio ModelConfig relative path (e.g. './weights/x.pth') under `root`."""
    s = str(p)
    if os.path.isabs(s) and os.path.exists(s):
        return s
    cand = os.path.join(root, s.lstrip("./"))
    return cand
