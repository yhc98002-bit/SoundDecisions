#!/usr/bin/env python
"""Arc-3 Tier-B B1 — UN-POOLED per-token + cross-attention re-tap (pre-reg §B1 families 3-4).

GPU re-tap of the MMAudio DiT internal activations at TOKEN level (manual §1.4 token-level
activations) on the 200-clip cfg=1.0 Phase-1 independents pool, for the per-token probe
(family 3) and the cross-attention-map probe (family 4) that the POOLED features cannot
support. The CPU driver scripts/b1_class_readability.py already covers families 1-2 from the
cached pooled features; THIS script supplies the artifacts families 3-4 need.

  Family 3 (per-token): for each (clip, ind j, s, layer) keep the latent (audio) token
    activations un-pooled. Stored as: token_mean_max = concat(mean_t, max_t) per layer
    ((n_layers, 2D), the pre-reg's "token-mean-max concat"), plus a fixed-count token
    subsample tokens_sub ((n_layers, T_sub, D)) so the orchestrator's downstream probe can
    also flatten the full (T, D) field if it chooses. Mean removed -> kept per token: we
    additionally store token_mean ((n_layers, D)) so a "mean removed" residual field is
    recoverable without a re-tap.

  Family 4 (cross-attention map): MMAudio uses JOINT self-attention over concatenated
    [latent | clip | text] tokens (transformer_layers.JointBlock.forward); the "video->audio
    cross-attention" is the latent-query x clip-key sub-block of that joint attention map.
    Per joint block we recompute softmax(QK^T/sqrt(d_head)) for latent queries against clip
    keys, average over heads, and pool: xattn_clip = mean over latent-query rows of the
    attention mass each clip key receives ((n_joint, n_clip_tokens)) and the per-layer summary
    xattn_to_clip_frac = total attention mass latent queries place on clip vs (clip+text+self).
    This is "where the model routes conditioning", the family-4 probe input.

EXACTLY mirrors scripts/phase1_commitment.py: --shard i/n over the 200 single_event clips,
rng_for(seed, clip, "ind", j) so the re-tapped independents are the SAME trajectories whose
pooled features are cached (1:1 with <clip>__p1cfg1_ind<j>), journaled per clip, kernel-
guarded via assert_certified_kernel (cfg=1.0 sqrt_down). Writes under results/stage0/arc3/
pertoken/ via RunStore.put_npz (budget-accounted; frozen files untouched).

DO NOT run here (GPU). The orchestrator runs it sharded on an17/an29, then a follow-up CPU
probe consumes results/stage0/arc3/pertoken/*.npz with the same frozen-split / abstain
convention as b1_class_readability.py.

Run sharded (orchestrator):
  scripts/run_on_node.sh an17 'for i in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$i \
    python scripts/b1_pertoken_collect.py --shard $i/8 > logs/b1pt_$i.log 2>&1 & done; wait'
Aggregate (CPU): python scripts/b1_pertoken_collect.py --aggregate
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zlib
from contextlib import contextmanager
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from foley_cw import score_sde as K  # noqa: E402
from foley_cw.config import load_config  # noqa: E402
from foley_cw.feature_tap import (InstrumentedBackend, TapMismatchError,  # noqa: E402
                                  expected_passes)
from foley_cw.kernel_provenance import assert_certified_kernel  # noqa: E402
from foley_cw.run_store import RunStore  # noqa: E402
from foley_cw.storage_budget import StorageBudget  # noqa: E402
from foley_cw.types import ScheduleSpec  # noqa: E402

PHASE1_S_GRID = (0.05, 0.15, 0.25, 0.35, 0.45, 0.60, 0.75, 0.90)
N_INDEPENDENT = 16
T_SUB = 64                       # fixed latent-token subsample kept per (layer, s)
PERTOKEN_SUBDIR = "arc3/pertoken"


def rng_for(seed: int, *parts) -> np.random.Generator:
    """IDENTICAL to scripts/phase1_commitment.rng_for so re-tapped independents match the
    cached pooled features (same x_init / same trajectory per (clip, j))."""
    entropy = [seed] + [zlib.crc32(str(p).encode()) for p in parts]
    return np.random.default_rng(np.random.SeedSequence(entropy))


class PerTokenTap:
    """Forward-hook tap keeping UN-POOLED latent-token activations per joint/fused block,
    plus the latent->clip cross-attention map per joint block.

    Does NOT modify foley_cw.feature_tap (frozen); it registers its OWN hooks on the same
    net.joint_blocks / net.fused_blocks. The latent-token count per block is read from the
    forward inputs (JointBlock.forward(latent, clip_f, text_f, ...)); the cross-attention
    map is recomputed from each JointBlock's pre_attention qkv (transformer_layers.py).
    """

    def __init__(self, net) -> None:
        import torch  # noqa: F401  (GPU-only import, deferred)
        self._net = net
        self._joint = list(net.joint_blocks)
        self._fused = list(net.fused_blocks)
        self.n_joint = len(self._joint)
        self.n_layers = self.n_joint + len(self._fused)
        self._armed = False
        self._tok: list[np.ndarray] = []        # per joint/fused: latent tokens (N1, D) fp16
        self._xattn_clip: list[np.ndarray] = []  # per joint: (n_clip,) clip-key attention mass
        self._xattn_frac: list[float] = []       # per joint: latent->clip mass fraction
        self._handles: list = []

    # ---- joint block: pre-hook captures inputs to recompute the cross-attn map -----
    def _pre_joint(self, blk):
        import torch

        def hook(_module, args, kwargs):
            if not self._armed:
                return
            latent = kwargs.get("latent", args[0] if args else None)
            clip_f = kwargs.get("clip_f", args[1] if len(args) > 1 else None)
            text_f = kwargs.get("text_f", args[2] if len(args) > 2 else None)
            latent_rot = kwargs.get("latent_rot")
            clip_rot = kwargs.get("clip_rot")
            extended_c = kwargs.get("extended_c")
            global_c = kwargs.get("global_c")
            if latent_rot is None or clip_rot is None:
                # positional fallback: forward(latent, clip_f, text_f, global_c,
                # extended_c, latent_rot, clip_rot)
                global_c = args[3] if len(args) > 3 else global_c
                extended_c = args[4] if len(args) > 4 else extended_c
                latent_rot = args[5] if len(args) > 5 else latent_rot
                clip_rot = args[6] if len(args) > 6 else clip_rot
            with torch.no_grad():
                x_qkv, _ = blk.latent_block.pre_attention(latent, extended_c, latent_rot)
                c_qkv, _ = blk.clip_block.pre_attention(clip_f, global_c, clip_rot)
                t_qkv, _ = blk.text_block.pre_attention(text_f, global_c, rot=None)
                # q,k: (B, heads, N, d_head)  (MMAudio attention() layout)
                q = x_qkv[0]                      # latent queries
                kx, kc, kt = x_qkv[1], c_qkv[1], t_qkv[1]   # keys per stream
                k_all = torch.cat([kx, kc, kt], dim=2)
                d_head = q.shape[-1]
                scores = (q @ k_all.transpose(-1, -2)) / (d_head ** 0.5)
                attn = torch.softmax(scores.float(), dim=-1)   # (B, heads, N1, N_all)
                n1 = q.shape[2]; nc = kc.shape[2]
                clip_slice = attn[..., n1:n1 + nc]             # latent->clip block
                # average over heads + latent queries -> per clip-key mass
                clip_mass = clip_slice.mean(dim=(0, 1, 2))     # (nc,)
                # fraction of latent-query attention placed on clip keys overall
                frac = float(clip_slice.sum(dim=-1).mean().item())
                self._xattn_clip.append(clip_mass.cpu().numpy().astype(np.float16))
                self._xattn_frac.append(frac)

        return blk.register_forward_pre_hook(hook, with_kwargs=True)

    # ---- output hooks capture un-pooled latent tokens -----------------------------
    def _post_joint(self, _mod, _inp, out):
        if not self._armed:
            return
        latent = out[0]                                   # (1, N1, D)
        if latent.ndim != 3 or latent.shape[0] != 1:
            raise TapMismatchError(f"joint latent {tuple(latent.shape)} != (1,N,D)")
        self._tok.append(latent.detach()[0].float().cpu().numpy().astype(np.float16))

    def _post_fused(self, _mod, _inp, out):
        if not self._armed:
            return
        if out.ndim != 3 or out.shape[0] != 1:
            raise TapMismatchError(f"fused out {tuple(out.shape)} != (1,N,D)")
        self._tok.append(out.detach()[0].float().cpu().numpy().astype(np.float16))

    def attach(self) -> None:
        if self._handles:
            return
        for blk in self._joint:
            self._handles.append(self._pre_joint(blk))
            self._handles.append(blk.register_forward_hook(self._post_joint))
        for blk in self._fused:
            self._handles.append(blk.register_forward_hook(self._post_fused))

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    @contextmanager
    def armed(self):
        self.attach()
        self._armed = True
        try:
            yield self
        finally:
            self._armed = False

    def clear(self) -> None:
        self._tok = []; self._xattn_clip = []; self._xattn_frac = []

    def pop_call(self, passes: int) -> dict:
        """Collapse one velocity call's captures (conditional pass only) into arrays.

        Returns per-(clip,j,s) artifacts:
          token_mean      (n_layers, D)
          token_mean_max  (n_layers, 2D)     <- pre-reg family-3 probe input
          tokens_sub      (n_layers, T_sub, D)
          xattn_clip      (n_joint, n_clip)  <- pre-reg family-4 probe input
          xattn_frac      (n_joint,)
        """
        want_tok = passes * self.n_layers
        if len(self._tok) != want_tok:
            raise TapMismatchError(
                f"captured {len(self._tok)} token layers, expected {want_tok} "
                f"({passes} passes x {self.n_layers}) — cfg branch mismatch?")
        want_xa = passes * self.n_joint
        if len(self._xattn_clip) != want_xa:
            raise TapMismatchError(
                f"captured {len(self._xattn_clip)} xattn maps, expected {want_xa}")
        tok = self._tok[: self.n_layers]        # conditional pass = first block
        xclip = self._xattn_clip[: self.n_joint]
        xfrac = self._xattn_frac[: self.n_joint]

        means, meanmax, subs = [], [], []
        for layer_tokens in tok:                 # (N1, D) fp16
            f = layer_tokens.astype(np.float32)
            mu = f.mean(axis=0); mx = f.max(axis=0)
            means.append(mu.astype(np.float16))
            meanmax.append(np.concatenate([mu, mx]).astype(np.float16))
            n = f.shape[0]
            if n >= T_SUB:
                idx = np.linspace(0, n - 1, T_SUB).round().astype(int)
            else:
                idx = np.concatenate([np.arange(n), np.zeros(T_SUB - n, dtype=int)])
            subs.append(f[idx].astype(np.float16))
        self.clear()
        return {
            "token_mean": np.stack(means),
            "token_mean_max": np.stack(meanmax),
            "tokens_sub": np.stack(subs),
            "xattn_clip": np.stack(xclip),
            "xattn_frac": np.asarray(xfrac, dtype=np.float32),
        }


def tap_pertoken_at(ib: InstrumentedBackend, tap: PerTokenTap, x_s, s: float, cond) -> dict:
    """One armed velocity call at (x_s, s); returns the per-token + cross-attn artifacts.
    Mirrors InstrumentedBackend.tap_features_at but with the un-pooled tap."""
    passes = expected_passes(float(ib.cfg_strength))
    t = ib.s_to_t.s_to_t(float(s))
    with tap.armed():
        tap.clear()
        ib._b.velocity(x_s, t, cond)
        ib._nfe += 1
        return tap.pop_call(passes)


def run_clip(ib, tap, store: RunStore, clip: str, video: Path, schedule: ScheduleSpec,
             seed: int, cfg: float, tag: str, n_independent: int) -> dict:
    t0 = time.time(); nfe0 = ib.nfe
    ib.cfg_strength = cfg
    cond = ib.make_video_cond(str(video), video_id=clip)
    saved = 0
    for j in range(n_independent):
        gid = f"{clip}__{tag}_ind{j}"
        tr = K.generate_trajectory(ib, cond, schedule, rng_for(seed, clip, "ind", j),
                                   alpha=0.0, record_points=PHASE1_S_GRID)
        for s in PHASE1_S_GRID:
            art = tap_pertoken_at(ib, tap, tr["states"][s], s, cond)
            store.put_npz(PERTOKEN_SUBDIR, f"{gid}__s{s:.2f}", **art)
            saved += 1
    nfe = ib.nfe - nfe0
    elapsed = time.time() - t0
    print(f"[b1pt {clip}] {elapsed:.0f}s nfe={nfe} saved={saved}", flush=True)
    return {"clip": clip, "cfg": cfg, "tag": tag, "saved": saved,
            "elapsed_s": round(elapsed, 1), "nfe_velocity_calls": int(nfe)}


def aggregate(out: Path, clips: list[str], tag: str) -> int:
    """Manifest of collected per-token bundles (CPU). The downstream probe reads the npz."""
    store = RunStore(out)
    have, missing = [], []
    for clip in clips:
        unit = f"{tag}_pertoken__{clip}"
        (have if store.is_done(unit) else missing).append(clip)
    man = {"_doc": "Arc-3 B1 per-token/cross-attn collection manifest (§B1 families 3-4).",
           "tag": tag, "subdir": PERTOKEN_SUBDIR, "n_done": len(have),
           "n_missing": len(missing), "missing": missing[:20]}
    out_dir = out / "arc3"; out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"b1_pertoken_manifest_{tag}.json").write_text(json.dumps(man, indent=2))
    print(f"[aggregate] done {len(have)} / missing {len(missing)} (first {missing[:5]})")
    return 0 if not missing else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path,
                    default=Path("data/manifests/phase1_manifest_frozen.json"))
    ap.add_argument("--clips-root", type=Path, default=Path("data/FoleyBench/clips"))
    ap.add_argument("--out", type=Path, default=Path("results/stage0"))
    ap.add_argument("--certified", type=Path,
                    default=Path("results/stage_m_rerun/certified_kernels.json"))
    ap.add_argument("--cfg", type=float, default=1.0)
    ap.add_argument("--schedule", default="sqrt_down")
    ap.add_argument("--variant", default="small_16k")
    ap.add_argument("--duration", type=float, default=8.0)
    ap.add_argument("--num-steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-independent", type=int, default=N_INDEPENDENT)
    ap.add_argument("--k-forks", type=int, default=12)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--clip-set", default="single_event",
                    choices=["single_event", "two_event", "both"])
    ap.add_argument("--tag", default=None, help="default p1cfg<cfg>")
    ap.add_argument("--require-ratified", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--aggregate", action="store_true")
    args = ap.parse_args()

    tag = args.tag or f"p1cfg{args.cfg:g}"
    man = json.loads(args.manifest.read_text())
    if args.clip_set == "both":
        clips = sorted(set(man["clips"]["single_event"]) | set(man["clips"]["two_event"]))
    else:
        clips = sorted(str(c) for c in man["clips"][args.clip_set])

    if args.aggregate:
        return aggregate(args.out, clips, tag)

    # Provenance guard (§15.8): refuse an uncertified (cfg, schedule). cfg=1.0 is ratified.
    cert = assert_certified_kernel(args.cfg, args.schedule, args.certified,
                                   require_ratified=args.require_ratified)
    print(f"[b1pt] kernel OK: {cert['token']} (ratified={cert['ratified']})", flush=True)

    shard_i, shard_n = (int(x) for x in args.shard.split("/"))
    schedule = ScheduleSpec(n_steps=args.num_steps, scan_points=PHASE1_S_GRID,
                            K_forks=args.k_forks, N_independent=args.n_independent,
                            g_kind=args.schedule, g_value=1.0)
    grid = schedule.integration_s_grid()
    for s in PHASE1_S_GRID:
        assert np.any(np.isclose(grid, s, atol=1e-9)), f"s={s} off integration grid"

    budget = StorageBudget(cap_gb=100.0)
    store = RunStore(args.out, budget=budget)
    store.account_preexisting_tree()

    todo = [c for i, c in enumerate(clips) if i % shard_n == shard_i]
    if args.limit:
        todo = todo[: args.limit]
    todo = [c for c in todo if not store.is_done(f"{tag}_pertoken__{c}")]
    print(f"[b1pt] shard {args.shard}: {len(todo)} clips (tag={tag}, cfg={args.cfg})",
          flush=True)
    if not todo:
        return 0

    from foley_cw.mmaudio_backend import MMAudioBackend

    backend = MMAudioBackend(variant=args.variant, device=args.device, full_precision=True,
                             cfg_strength=args.cfg, num_steps=args.num_steps,
                             duration_sec=args.duration, enable_conditions=True)
    ib = InstrumentedBackend(backend)
    tap = PerTokenTap(backend.net)
    _ = load_config()  # parity with phase1 (axis config load); not gated here

    for clip in todo:
        payload = run_clip(ib, tap, store, clip, args.clips_root / f"{clip}.mp4",
                           schedule, args.seed, cfg=args.cfg, tag=tag,
                           n_independent=args.n_independent)
        payload["budget"] = budget.summary()
        store.journal_done(f"{tag}_pertoken__{clip}", payload)
    print(f"[b1pt] shard {args.shard} complete; budget: {budget.summary()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
