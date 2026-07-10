#!/usr/bin/env python
"""Stage-M micro-map GPU runner — REVISED protocol (revised manual section 2).

Per (clip, cfg) journal unit:
  * encode video conditioning once;
  * 1 BASE trajectory (alpha=0): every-step features, grid features + previews,
    final wav, measurements;
  * N=8 INDEPENDENTS (alpha=0): grid features + previews + final wavs +
    measurements + PANNs embeddings (conditioning floor A_ind_emb) + 527-dim
    class prob vectors;
  * 8 FRESH REFERENCE independents (alpha=0, distinct RNG stream): the Gate-A
    reference pool (spec-literal "fresh independents") — grid features +
    previews + final wavs + measurements + prob vectors;
  * per-seed COMMITMENT forks: K=8 from the base state at every grid s
    (alpha=PRIMARY, marginal-preserving SDE) — measurements + embeddings
    (A_fork_emb seed-floor/growth) + 10% wav audit;
  * GATE-A forks: ONE tail per independent at s in --gate-a-s (default
    {0.05, 0.90}) — seed-marginalized pool; prob vectors + labels only.

Confident-subset agreements (class abstains excluded; frozen interpretation #3)
are journaled per cell together with abstain rates, confident counts, embedding
cosines, and per-measurement class diagnostics (margin/entropy/top1).

Default --seed 1 (the first run used seed 0; its wavs informed the redesign —
frozen interpretation #4). Budget ~1.55k FGE total (~9 min wall on 8 shards).
Stage-M outputs are engineering diagnostics, never scientific evidence.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from foley_cw import score_sde as K  # noqa: E402
from foley_cw import validation as V  # noqa: E402
from foley_cw.agreement import confident_agreement, mean_pairwise_cosine  # noqa: E402
from foley_cw.config import load_config  # noqa: E402
from foley_cw.feature_tap import InstrumentedBackend  # noqa: E402
from foley_cw.real_measurer import ABSTAIN, RealFoleyMeasurer  # noqa: E402
from foley_cw.reliability import determinism  # noqa: E402
from foley_cw.run_store import RunStore  # noqa: E402
from foley_cw.storage_budget import StorageBudget  # noqa: E402
from foley_cw.types import AgreementMetric, ScheduleSpec  # noqa: E402

STAGE_M_AXIS_IDS = ("presence", "class")
N_FRESH_REFS = 8


def rng_for(seed: int, *parts) -> np.random.Generator:
    entropy = [seed] + [zlib.crc32(str(p).encode()) for p in parts]
    return np.random.default_rng(np.random.SeedSequence(entropy))


def parse_floats(s: str) -> tuple[float, ...]:
    return tuple(float(x) for x in s.split(","))


def assert_grid_alignment(schedule: ScheduleSpec, points: tuple[float, ...]) -> None:
    grid = schedule.integration_s_grid()
    for s in points:
        if not np.any(np.isclose(grid, s, atol=1e-9)):
            raise SystemExit(f"s={s} not on the integration grid (n_steps={schedule.n_steps})")


def run_unit(ib, measurer, axes, store: RunStore, clip: str, cfg: float, video: Path,
             schedule: ScheduleSpec, s_grid: tuple[float, ...], alpha: float,
             k_forks: int, n_independent: int, gate_a_s: tuple[float, ...],
             seed: int) -> dict:
    t0 = time.time()
    nfe0 = ib.nfe
    ib.cfg_strength = cfg
    assert float(ib.cfg_strength) == cfg
    g = K.make_g(schedule.g_kind, schedule.g_value)
    axes_by_id = {a.id: a for a in axes}

    print(f"[unit {clip} cfg={cfg:g}] encoding conditioning ...", flush=True)
    cond = ib.make_video_cond(str(video), video_id=clip)

    def capture_grid_point(gen_id: str, x_s: np.ndarray, s: float, preview: bool) -> None:
        feats, v = ib.tap_features_at(x_s, s, cond)
        store.put_features(gen_id, s, feats)
        if preview:
            t = ib.s_to_t.s_to_t(s)
            store.put_preview(gen_id, s, ib.decode(K.tweedie_x0(v, x_s, t)))

    def measure_all(gen_id: str, audio: np.ndarray, extra: dict) -> dict:
        """Measure both axes; class measurements carry the diagnostic instruments."""
        labels = {}
        probs, _ = measurer._panns_forward(audio)
        for axis in axes:
            tgt = measurer.measure(audio, axis)
            labels[axis.id] = tgt.label
            ex = dict(extra)
            if axis.id == "class":
                diag = measurer.class_diagnostics(probs)
                ex.update({k: diag[k] for k in ("margin", "entropy", "top1_index",
                                                "top1_prob", "concentration")})
            store.record_measurement(gen_id, axis.id, tgt, extra=ex)
        return {"labels": labels, "probs": probs,
                "emb": measurer.panns_embedding(audio)}

    def conf_agree(axis_id: str, labels: list) -> tuple[float, int, float]:
        """(confident exact-match | NaN, n_conf, abstain_rate)."""
        val, n_conf = confident_agreement(labels, AgreementMetric.EXACT_MATCH,
                                          abstain=ABSTAIN)
        abst = sum(1 for l in labels if l == ABSTAIN) / max(len(labels), 1)
        return val, n_conf, abst

    # ---- BASE ------------------------------------------------------------------
    base_id = f"{clip}__cfg{cfg:g}__base"
    with ib.record_steps():
        traj = K.generate_trajectory(ib, cond, schedule, rng_for(seed, clip, cfg, "base"),
                                     alpha=0.0, record_points=s_grid)
    ts, step_feats = ib.drain_step_features()
    store.put_step_features(base_id, ts, step_feats)
    for s in s_grid:
        capture_grid_point(base_id, traj["states"][s], s, preview=True)
    store.put_final_wav(base_id, traj["audio"])
    measure_all(base_id, traj["audio"], {"role": "base", "clip": clip, "cfg": cfg})

    # ---- INDEPENDENTS (conditioning floor + Gate-A fork parents) ----------------
    ind_states: list[dict] = []
    ind_labels = {a.id: [] for a in axes}
    ind_embs, ind_probs = [], []
    for j in range(n_independent):
        gid = f"{clip}__cfg{cfg:g}__ind{j}"
        tr = K.generate_trajectory(ib, cond, schedule, rng_for(seed, clip, cfg, "ind", j),
                                   alpha=0.0, record_points=tuple(sorted(set(s_grid)
                                                                         | set(gate_a_s))))
        for s in s_grid:
            capture_grid_point(gid, tr["states"][s], s, preview=True)
        m = measure_all(gid, tr["audio"], {"role": "independent", "j": j,
                                           "clip": clip, "cfg": cfg})
        for a in axes:
            ind_labels[a.id].append(m["labels"][a.id])
        ind_embs.append(m["emb"])
        ind_probs.append(m["probs"])
        store.put_final_wav(gid, tr["audio"])
        ind_states.append({s: tr["states"][s] for s in gate_a_s})

    a_independent, n_conf_ind, abstain_ind = {}, {}, {}
    for a in axes:
        v, n, ab = conf_agree(a.id, ind_labels[a.id])
        a_independent[a.id], n_conf_ind[a.id], abstain_ind[a.id] = v, n, ab
    a_ind_emb = float(mean_pairwise_cosine(np.stack(ind_embs)))

    # ---- FRESH REFERENCE INDEPENDENTS (Gate-A reference pool) -------------------
    ref_labels_class, ref_probs = [], []
    for j in range(N_FRESH_REFS):
        gid = f"{clip}__cfg{cfg:g}__ref{j}"
        tr = K.generate_trajectory(ib, cond, schedule,
                                   rng_for(seed, clip, cfg, "gateref", j),
                                   alpha=0.0, record_points=s_grid)
        for s in s_grid:
            capture_grid_point(gid, tr["states"][s], s, preview=True)
        m = measure_all(gid, tr["audio"], {"role": "gate_ref", "j": j,
                                           "clip": clip, "cfg": cfg})
        ref_labels_class.append(m["labels"]["class"])
        ref_probs.append(m["probs"])
        store.put_final_wav(gid, tr["audio"])

    # ---- PER-SEED COMMITMENT FORKS (K from the base seed at every grid s) -------
    a_fork, n_conf_fork, abstain_fork, a_fork_emb = {}, {}, {}, {}
    for s in s_grid:
        x_s = traj["states"][s]
        fork_labels = {a.id: [] for a in axes}
        embs = []
        later = [p for p in s_grid if p > s + 1e-9]
        for k in range(k_forks):
            gid = f"{clip}__cfg{cfg:g}__fork_s{s:g}_{k}"
            rng_k = rng_for(seed, clip, cfg, "fork", s, k)
            x = np.array(x_s, copy=True)
            s_prev = s
            for s_next in later:
                x = K.integrate_segment(ib, x, cond, s_prev, s_next, schedule, alpha, g, rng_k)
                capture_grid_point(gid, x, s_next, preview=False)
                s_prev = s_next
            x = K.integrate_segment(ib, x, cond, s_prev, 1.0, schedule, alpha, g, rng_k)
            audio = ib.decode(x)
            store.put_final_wav(gid, audio, audit_only=True)
            m = measure_all(gid, audio, {"role": "fork", "s": s, "k": k,
                                         "clip": clip, "cfg": cfg, "alpha": alpha})
            for a in axes:
                fork_labels[a.id].append(m["labels"][a.id])
            embs.append(m["emb"])
        for a in axes:
            v, n, ab = conf_agree(a.id, fork_labels[a.id])
            a_fork[f"{a.id}|{s:g}"] = v
            n_conf_fork[f"{a.id}|{s:g}"] = n
            abstain_fork[f"{a.id}|{s:g}"] = ab
        a_fork_emb[f"{s:g}"] = float(mean_pairwise_cosine(np.stack(embs)))

    # ---- GATE-A FORKS (one tail per independent, seed-marginalized pool) --------
    ga_probs_by_s, ga_labels_by_s = {}, {}
    for s in gate_a_s:
        probs_pool, labels_pool = [], []
        for j, st in enumerate(ind_states):
            gid = f"{clip}__cfg{cfg:g}__gafork_s{s:g}_{j}"
            rng_j = rng_for(seed, clip, cfg, "gatea", j, s)
            x = K.integrate_segment(ib, np.array(st[s], copy=True), cond, s, 1.0,
                                    schedule, alpha, g, rng_j)
            audio = ib.decode(x)
            store.put_final_wav(gid, audio, audit_only=True)  # 1.4: fork finals 10% audit
            m = measure_all(gid, audio, {"role": "gate_fork", "s": s, "j": j,
                                         "clip": clip, "cfg": cfg, "alpha": alpha})
            probs_pool.append(m["probs"])
            labels_pool.append(m["labels"]["class"])
        ga_probs_by_s[s] = np.stack(probs_pool)
        ga_labels_by_s[s] = labels_pool

    # ---- Gate-A / embedding bundle (budget-accounted) ---------------------------
    ga_path = store.put_npz(
        "gate_a", f"{clip}__cfg{cfg:g}",
        probs_ref=np.stack(ref_probs).astype(np.float32),
        probs_ind=np.stack(ind_probs).astype(np.float32),
        emb_ind=np.stack(ind_embs).astype(np.float32),
        **{f"probs_gafork_s{s:g}": ga_probs_by_s[s].astype(np.float32) for s in gate_a_s},
    )
    gate_a_labels = {"ref": ref_labels_class,
                     **{f"s{s:g}": ga_labels_by_s[s] for s in gate_a_s}}

    elapsed = time.time() - t0
    nfe = ib.nfe - nfe0
    print(f"[unit {clip} cfg={cfg:g}] done in {elapsed:.0f}s nfe={nfe} "
          f"(~{nfe/schedule.n_steps:.1f} FGE) "
          f"A_ind={ {k: (round(v, 3) if np.isfinite(v) else 'NaN') for k, v in a_independent.items()} } "
          f"abst_ind={ {k: round(v, 2) for k, v in abstain_ind.items()} }", flush=True)
    return {"clip": clip, "cfg": cfg, "alpha": alpha, "g_kind": schedule.g_kind,
            "a_independent": a_independent, "n_conf_ind": n_conf_ind,
            "abstain_ind": abstain_ind, "a_ind_emb": a_ind_emb,
            "a_fork": a_fork, "n_conf_fork": n_conf_fork, "abstain_fork": abstain_fork,
            "a_fork_emb": a_fork_emb, "gate_a_labels": gate_a_labels,
            "gate_a_s": list(gate_a_s), "elapsed_s": round(elapsed, 1),
            "nfe_velocity_calls": int(nfe),
            "full_gen_equivalents": round(nfe / schedule.n_steps, 2),
            "gate_a_npz": str(ga_path)}


def run_extras(ib, measurer, axes, store: RunStore, clip: str, video: Path,
               schedule: ScheduleSpec, alpha: float, seed: int) -> dict:
    """Once per shard: SDE re-validation (Gate-A items i-iii) at BOTH cfgs +
    measurer determinism (extended alphabet)."""
    payload = {}
    cond = None
    for cfgv, key in ((1.0, "sde_token_cfg1"), (4.5, "sde_token_cfg45")):
        ib.cfg_strength = cfgv
        cond = ib.make_video_cond(str(video), video_id=clip)
        checks, token = V.run_sde_validation(ib, cond, schedule,
                                             rng_for(seed, "sde", cfgv), alpha=alpha)
        payload[key] = token
        payload[key + "_checks"] = [{"name": c.name, "passed": bool(c.passed),
                                     "value": float(c.value), "detail": c.detail}
                                    for c in checks]
        print(f"[extras] SDE@cfg{cfgv:g} token={token}", flush=True)
    ib.cfg_strength = 1.0
    base = K.generate_trajectory(ib, cond, schedule, rng_for(seed, clip, "det"),
                                 alpha=0.0, record_points=(0.0, 1.0))
    det = {a.id: float(determinism(measurer, base["audio"], a, repeats=5)) for a in axes}
    payload["determinism"] = det
    print(f"[extras] determinism={det}", flush=True)
    return payload


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips-json", type=Path, default=Path("data/manifests/stage_m_clips.json"))
    ap.add_argument("--clips-root", type=Path, default=Path("data/FoleyBench/clips"))
    ap.add_argument("--out", type=Path, default=Path("results/stage_m_rerun"))
    ap.add_argument("--variant", default="small_16k")
    ap.add_argument("--duration", type=float, default=8.0)
    ap.add_argument("--num-steps", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=1.6,
                    help="PRIMARY_ALPHA at the headline cfg (alpha-pilot token)")
    ap.add_argument("--g-kind", default="constant",
                    choices=["constant", "linear_down", "sqrt_down"],
                    help="fork-noise schedule g(s); early-heavy schedules are the "
                         "manual-1.3 kernel-redesign route")
    ap.add_argument("--k-forks", type=int, default=8)
    ap.add_argument("--n-independent", type=int, default=8)
    ap.add_argument("--s-grid", default="0.05,0.30,0.60,0.90")
    ap.add_argument("--gate-a-s", default="0.05,0.90",
                    help="Gate-A test s-points (one fork per independent each)")
    ap.add_argument("--cfgs", default="1.0,4.5")
    ap.add_argument("--seed", type=int, default=1,
                    help="seed 1: the first run's seed-0 wavs informed the redesign")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    s_grid = parse_floats(args.s_grid)
    gate_a_s = parse_floats(args.gate_a_s)
    cfgs = parse_floats(args.cfgs)
    shard_i, shard_n = (int(x) for x in args.shard.split("/"))

    cfg_all = load_config()
    axes = [a for a in cfg_all.axes if a.id in STAGE_M_AXIS_IDS]
    assert len(axes) == len(STAGE_M_AXIS_IDS)

    clips = json.loads(args.clips_json.read_text())["clips"]
    schedule = ScheduleSpec(n_steps=args.num_steps, scan_points=s_grid,
                            K_forks=args.k_forks, N_independent=args.n_independent,
                            g_kind=args.g_kind)
    assert_grid_alignment(schedule, tuple(sorted(set(s_grid) | set(gate_a_s))))

    budget = StorageBudget(cap_gb=100.0)
    store = RunStore(args.out, budget=budget)
    pre = store.account_preexisting_tree()
    if pre:
        print(f"[stage-m] accounted {pre/1e6:.1f} MB preexisting", flush=True)

    units = [(c, g) for c in clips for g in cfgs]
    units = [u for i, u in enumerate(sorted(units)) if i % shard_n == shard_i]
    if args.limit:
        units = units[: args.limit]
    todo = [u for u in units if not store.is_done(f"{u[0]}__cfg{u[1]:g}")]
    print(f"[stage-m] shard {args.shard}: {len(todo)}/{len(units)} units to run", flush=True)
    if not todo and store.is_done(f"extras__shard{shard_i}"):
        print("[stage-m] nothing to do")
        return 0

    from foley_cw.mmaudio_backend import MMAudioBackend
    t0 = time.time()
    backend = MMAudioBackend(variant=args.variant, device=args.device, full_precision=True,
                             cfg_strength=cfgs[0], num_steps=args.num_steps,
                             duration_sec=args.duration, enable_conditions=True)
    ib = InstrumentedBackend(backend)
    measurer = RealFoleyMeasurer(device=args.device)
    print(f"[stage-m] backend+measurer ready in {time.time()-t0:.0f}s", flush=True)

    for clip, g in todo:
        payload = run_unit(ib, measurer, axes, store, clip, g,
                           args.clips_root / f"{clip}.mp4", schedule, s_grid,
                           args.alpha, args.k_forks, args.n_independent, gate_a_s, args.seed)
        payload["budget"] = budget.summary()
        store.journal_done(f"{clip}__cfg{g:g}", payload)

    extras_id = f"extras__shard{shard_i}"
    if not store.is_done(extras_id):
        clip0 = sorted(clips)[shard_i % len(clips)]
        payload = run_extras(ib, measurer, axes, store, clip0,
                             args.clips_root / f"{clip0}.mp4", schedule, args.alpha, args.seed)
        store.journal_done(extras_id, payload)

    print(f"[stage-m] shard {args.shard} complete; budget: {budget.summary()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
