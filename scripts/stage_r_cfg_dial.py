#!/usr/bin/env python
"""Stage R — F-1 cfg-dial collection (manual §8.3 part b; Fig 2 α*(cfg)). REAL GPU runner.

The Decision-1 disambiguator: does guidance move the class decision into the SEED? Per
clip × cfg ∈ {1.0,1.5,2.0,2.5,3.0,4.5} (≈24 clips), instrument-light (no marginal-preserving
kernel guarantee needed at high cfg — this is a diagnostic, not a headline map), collect the
three F-1 dial quantities:

  (1) SEED-PREDICTABILITY of class: N_independent ODE (alpha=0) generations; store each
      INITIAL noise latent x(s=0) (the probe input) + the final class label (the probe
      target). The (noise, video) -> class linear probe is fit at --aggregate via
      foley_cw.internal_probes.probe_accuracy. Also the earliest-s fork-agreement of class
      (already computable from A_fork) from a small fork sweep on one base trajectory.
  (2) α*(cfg): from a base trajectory, K tail-forks at s_min under each pilot alpha; tail
      DIVERSITY (1 − class fork-agreement) per alpha → α* = smallest alpha clearing
      diversity_min (computed at --aggregate by foley_cw.cfg_dial.alpha_star).
  (3) SEED SHARE per cfg is read from the two headline commitment grids (§8.3 part a) at
      --aggregate, NOT re-collected here.

Reuses the phase1_commitment pattern verbatim (InstrumentedBackend optional / MMAudioBackend +
RealFoleyMeasurer + RunStore + StorageBudget; --shard i/n over an17/an29; rng_for; journaled,
resumable). cfg>1 forks are diagnostic-only here (the F-1 dial is explicitly instrument-light
per §8.3) and are NOT cited as a headline kernel — no Gate-A certification is asserted; the
module/aggregate emit only a SUGGESTED token (the PI holds the binding F-1 call).

Run sharded (GPU; do NOT certify a kernel — diagnostic dial):
  scripts/run_on_node.sh an17 'for i in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$i \
    python scripts/stage_r_cfg_dial.py --shard $i/8 > logs/dial_$i.log 2>&1 & done; wait'
Aggregate (CPU, numpy-only — fits probes, computes α*, suggests token):
  python scripts/stage_r_cfg_dial.py --aggregate
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

from foley_cw import cfg_dial  # noqa: E402
from foley_cw import score_sde as K  # noqa: E402
from foley_cw.agreement import agreement  # noqa: E402
from foley_cw.config import load_config  # noqa: E402
from foley_cw.internal_probes import probe_accuracy  # noqa: E402
from foley_cw.real_measurer import ABSTAIN  # noqa: E402
from foley_cw.run_store import RunStore  # noqa: E402
from foley_cw.storage_budget import StorageBudget  # noqa: E402
from foley_cw.types import AgreementMetric, ScheduleSpec  # noqa: E402

CLASS_AXIS = "class"
DIAL_CFGS = (1.0, 1.5, 2.0, 2.5, 3.0, 4.5)
DIAL_ALPHAS = (0.05, 0.1, 0.2, 0.4, 0.8, 1.6)   # pilot unlocking grid (alpha=0 ≡ no diversity)
PHASE1_S_GRID = (0.05, 0.15, 0.25, 0.35, 0.45, 0.60, 0.75, 0.90)
S_MIN = PHASE1_S_GRID[0]                          # seed floor for α* forks
N_INDEPENDENT = 16
K_FORKS = 12
N_DIAL_CLIPS = 24


def rng_for(seed: int, *parts) -> np.random.Generator:
    entropy = [seed] + [zlib.crc32(str(p).encode()) for p in parts]
    return np.random.default_rng(np.random.SeedSequence(entropy))


def _class_diversity(labels: list[str]) -> float:
    """Tail diversity of class forks = 1 − exact-match self-agreement (abstains kept as a
    distinct value so an all-abstain fork reads as zero diversity, consistent with §3.3)."""
    if len(labels) < 2:
        return 0.0
    return float(1.0 - agreement_labels(labels))


def agreement_labels(labels: list[str]) -> float:
    from foley_cw.agreement import categorical_agreement
    return categorical_agreement(labels)


def _earliest_fork_agreement(a_fork_by_s: dict[str, float], theta: float) -> float:
    """Earliest s whose class fork-agreement ≥ theta (NaN if never): a seed-predictability
    proxy that needs no probe — low s crossing means the value is fixed near the seed."""
    for s in sorted(a_fork_by_s, key=float):
        v = a_fork_by_s[s]
        if v is not None and np.isfinite(v) and v >= theta:
            return float(s)
    return float("nan")


def run_clip_cfg(ib, measurer, class_axis, store: RunStore, clip: str, video: Path,
                 schedule: ScheduleSpec, seed: int, cfg: float, tag: str,
                 n_independent: int = N_INDEPENDENT, k_forks: int = K_FORKS,
                 alphas: tuple = DIAL_ALPHAS) -> dict:
    """Collect the F-1 dial quantities for one (clip, cfg). The (noise, label) probe pairs
    are stored as an npz bundle; α* diversity + earliest fork-agreement are journaled."""
    t0 = time.time(); nfe0 = getattr(ib, "nfe", 0)
    ib.cfg_strength = cfg
    cond = ib.make_video_cond(str(video), video_id=clip)
    g = K.make_g(schedule.g_kind, schedule.g_value)

    # --- (1) independents: initial noise x(s=0) (probe input) + final class (probe target) ---
    noise_rows, label_rows = [], []
    for j in range(n_independent):
        tr = K.generate_trajectory(ib, cond, schedule, rng_for(seed, clip, cfg, "ind", j),
                                   alpha=0.0, record_points=(S_MIN,))
        x0_noise = np.asarray(tr["states"][0.0], dtype=np.float32).ravel()   # s=0 prior latent
        tgt = measurer.measure(tr["audio"], class_axis)
        noise_rows.append(x0_noise)
        label_rows.append(str(tgt.label))
        store.record_measurement(f"{clip}__{tag}_ind{j}", CLASS_AXIS, tgt,
                                 extra={"role": f"{tag}_independent", "j": j, "clip": clip,
                                        "cfg": cfg})
    store.put_npz("gate_a", f"dial_noise__{tag}__{clip}",
                  noise=np.stack(noise_rows, axis=0),
                  labels=np.array(label_rows, dtype=object))

    # --- base trajectory (fork source for α* + earliest-s agreement) ---
    base = K.generate_trajectory(ib, cond, schedule, rng_for(seed, clip, cfg, "base"),
                                 alpha=0.0, record_points=PHASE1_S_GRID)

    # --- (2) α*: K class-forks at s_min under each pilot alpha → tail diversity ---
    diversity_by_alpha: dict[str, float] = {}
    for a in alphas:
        audios = K.fork_tail(ib, base["states"][S_MIN], S_MIN, cond, a, k_forks, schedule,
                             rng_for(seed, clip, cfg, "astar", a), g=g)
        labels = [str(measurer.measure(au, class_axis).label) for au in audios]
        for k, au in enumerate(audios):
            fgid = f"{clip}__{tag}_astar_a{a:g}_k{k}"
            if store.audit_selected(fgid):
                store.put_final_wav(fgid, au, audit_only=False)
        diversity_by_alpha[f"{a:g}"] = _class_diversity(labels)

    # --- earliest-s class fork-agreement (probe-free seed-predictability proxy) ---
    fork_alpha = float(alphas[len(alphas) // 2])     # mid pilot alpha for the s-sweep
    a_fork_by_s: dict[str, float] = {}
    for s in PHASE1_S_GRID:
        audios = K.fork_tail(ib, base["states"][s], s, cond, fork_alpha, k_forks, schedule,
                             rng_for(seed, clip, cfg, "sfork", s), g=g)
        labels = [str(measurer.measure(au, class_axis).label) for au in audios]
        a_fork_by_s[f"{s:.2f}"] = agreement_labels(labels)

    nfe = getattr(ib, "nfe", 0) - nfe0
    elapsed = time.time() - t0
    print(f"[dial {clip} cfg={cfg:g}] {elapsed:.0f}s nfe={nfe} "
          f"div={ {a: round(d, 3) for a, d in diversity_by_alpha.items()} }", flush=True)
    return {"clip": clip, "cfg": cfg, "schedule": schedule.g_kind, "alphas": list(alphas),
            "fork_alpha": fork_alpha, "n_independent": n_independent, "k_forks": k_forks,
            "diversity_by_alpha": diversity_by_alpha, "a_fork_by_s": a_fork_by_s,
            "elapsed_s": round(elapsed, 1), "nfe_velocity_calls": int(nfe)}


# ---------------------------------------------------------------------------
# CPU aggregate: fit probes, compute α*, read seed share, suggest the F-1 token
# ---------------------------------------------------------------------------
def _seed_share_by_cfg(determination_json: Path) -> dict[float, float]:
    """Read the class seed share per cfg from the §8.3-part-a determination budgets.

    Expects a JSON mapping {"<cfg>": {"<axis>": {"budget": {"seed_share": {"mean": x}}}}}
    (the foley_cw.determination.build_determination_budget output dumped per cfg). Missing
    cfgs/axes are skipped — the trend is fit over whatever cfgs are present."""
    if not determination_json or not Path(determination_json).exists():
        return {}
    raw = json.loads(Path(determination_json).read_text())
    out: dict[float, float] = {}
    for cfg_key, by_axis in raw.items():
        ax = by_axis.get(CLASS_AXIS, {})
        ss = ax.get("budget", {}).get("seed_share", {})
        m = ss.get("mean") if isinstance(ss, dict) else ss
        if m is not None and np.isfinite(float(m)):
            out[float(cfg_key)] = float(m)
    return out


def aggregate(out: Path, clips: list[str], tag_fmt: str, cfgs: tuple, lam: float,
              diversity_min: float, theta_agree: float,
              determination_json: Path | None) -> int:
    store = RunStore(out)
    # Pool (noise, label) probe pairs across clips per cfg, train/eval split by clip.
    probe_acc_by_cfg: dict[float, float] = {}
    alpha_star_by_cfg: dict[float, float] = {}
    earliest_agree_by_cfg: dict[float, list[float]] = {}
    missing = []
    for cfg in cfgs:
        tag = tag_fmt.format(cfg=cfg)
        Xtr, ytr, Xte, yte = [], [], [], []
        div_by_alpha_pooled: dict[str, list[float]] = {}
        earliest: list[float] = []
        n_clips = 0
        for ci, clip in enumerate(clips):
            unit = f"{tag}__{clip}"
            if not store.is_done(unit):
                missing.append(unit); continue
            d = store.load_journal(unit)
            n_clips += 1
            # α* diversity + earliest-s agreement
            for a, dv in d["diversity_by_alpha"].items():
                div_by_alpha_pooled.setdefault(a, []).append(float(dv))
            earliest.append(_earliest_fork_agreement(d["a_fork_by_s"], theta_agree))
            # (noise, label) probe pairs (split by clip: 60/40-style alternation)
            npz = out / "gate_a" / f"dial_noise__{tag}__{clip}.npz"
            if npz.exists():
                z = np.load(npz, allow_pickle=True)
                noise = np.asarray(z["noise"], dtype=float)
                labels = [str(x) for x in z["labels"].tolist()]
                if ci % 5 < 3:                       # ~60% of clips → probe train
                    Xtr.append(noise); ytr += labels
                else:
                    Xte.append(noise); yte += labels
        if not div_by_alpha_pooled:
            continue
        # α*: pooled mean diversity per alpha → smallest alpha clearing diversity_min
        pooled = {float(a): float(np.mean(v)) for a, v in div_by_alpha_pooled.items()}
        astar = cfg_dial.alpha_star({cfg: pooled}, diversity_min=diversity_min)
        alpha_star_by_cfg[cfg] = astar["by_cfg"][cfg]
        earliest_agree_by_cfg[cfg] = earliest
        # probe accuracy (noise → class) pooled across clips at this cfg
        if Xtr and Xte:
            acc = probe_accuracy(np.concatenate(Xtr), ytr,
                                 np.concatenate(Xte), yte, lam=lam)
            if np.isfinite(acc):
                probe_acc_by_cfg[cfg] = float(acc)

    # seed share (read from §8.3 part-a determination budgets, not re-collected)
    seed_share_by_cfg = _seed_share_by_cfg(determination_json) if determination_json else {}
    # chance baseline for the probe = majority class prior over the pooled eval labels
    chance = None

    verdict = cfg_dial.f1_verdict(probe_acc_by_cfg, alpha_star_by_cfg, seed_share_by_cfg,
                                  diversity_min=diversity_min, chance=chance)
    payload = {
        "probe_acc_by_cfg": {str(k): v for k, v in probe_acc_by_cfg.items()},
        "alpha_star_by_cfg": {str(k): v for k, v in alpha_star_by_cfg.items()},
        "seed_share_by_cfg": {str(k): v for k, v in seed_share_by_cfg.items()},
        "earliest_fork_agreement_mean_by_cfg": {
            str(k): (float(np.nanmean(v)) if v else float("nan"))
            for k, v in earliest_agree_by_cfg.items()},
        "diversity_min": diversity_min, "theta_agree": theta_agree, "lam": lam,
        "suggested_token": verdict["suggested_token"], "rationale": verdict["rationale"],
        "predictions_met": verdict["predictions_met"], "trends": verdict["trends"],
        "n_missing_units": len(missing),
    }
    out_dir = out / "stage_r"; out_dir.mkdir(parents=True, exist_ok=True)
    res_path = out_dir / "cfg_dial_f1.json"
    res_path.write_text(json.dumps(payload, indent=2, default=lambda o: float(o)
                                   if isinstance(o, np.floating) else str(o)))
    print(f"[dial aggregate] suggested={verdict['suggested_token']} "
          f"met={verdict['predictions_met']} → {res_path} "
          f"(missing {len(missing)} units; first {missing[:3]})")
    return 0 if not missing else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path,
                    default=Path("data/manifests/phase1_manifest_frozen.json"))
    ap.add_argument("--clips-root", type=Path, default=Path("data/FoleyBench/clips"))
    ap.add_argument("--out", type=Path, default=Path("results/stage0"))
    ap.add_argument("--schedule", default="sqrt_down")
    ap.add_argument("--variant", default="small_16k")
    ap.add_argument("--duration", type=float, default=8.0)
    ap.add_argument("--num-steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-independent", type=int, default=N_INDEPENDENT)
    ap.add_argument("--k-forks", type=int, default=K_FORKS)
    ap.add_argument("--n-clips", type=int, default=N_DIAL_CLIPS)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tag-fmt", default="dial_cfg{cfg:g}")
    ap.add_argument("--lam", type=float, default=1.0, help="probe ridge regularization")
    ap.add_argument("--diversity-min", type=float, default=cfg_dial.DEFAULT_DIVERSITY_MIN)
    ap.add_argument("--theta-agree", type=float, default=0.8,
                    help="class fork-agreement threshold for the earliest-s proxy")
    ap.add_argument("--determination-json", type=Path, default=None,
                    help="per-cfg determination budgets (§8.3 part a) for the seed-share trend")
    ap.add_argument("--aggregate", action="store_true")
    args = ap.parse_args()

    man = json.loads(args.manifest.read_text())
    # F-1 dial uses single-event clips (class is the dial's axis); take the first N by id.
    clips = sorted(str(c) for c in man["clips"]["single_event"])[: args.n_clips]

    if args.aggregate:
        return aggregate(args.out, clips, args.tag_fmt, DIAL_CFGS, args.lam,
                         args.diversity_min, args.theta_agree, args.determination_json)

    shard_i, shard_n = (int(x) for x in args.shard.split("/"))
    schedule = ScheduleSpec(n_steps=args.num_steps, scan_points=PHASE1_S_GRID,
                            K_forks=args.k_forks, N_independent=args.n_independent,
                            g_kind=args.schedule, g_value=1.0)
    grid = schedule.integration_s_grid()
    for s in PHASE1_S_GRID:
        assert np.any(np.isclose(grid, s, atol=1e-9)), f"s={s} off integration grid"

    # Work units are (clip, cfg) pairs; shard over the flattened list (resume by journal).
    units = [(clip, cfg) for clip in clips for cfg in DIAL_CFGS]
    todo = [u for i, u in enumerate(units) if i % shard_n == shard_i]
    budget = StorageBudget(cap_gb=100.0)
    store = RunStore(args.out, budget=budget)
    store.account_preexisting_tree()
    todo = [(c, cfg) for (c, cfg) in todo
            if not store.is_done(f"{args.tag_fmt.format(cfg=cfg)}__{c}")]
    print(f"[dial] shard {args.shard}: {len(todo)} (clip,cfg) units "
          f"(cfgs={DIAL_CFGS}, schedule={args.schedule})", flush=True)
    if not todo:
        return 0

    from foley_cw.feature_tap import InstrumentedBackend
    from foley_cw.mmaudio_backend import MMAudioBackend
    from foley_cw.real_measurer import RealFoleyMeasurer

    class_axis = next(a for a in load_config().axes if a.id == CLASS_AXIS)
    backend = MMAudioBackend(variant=args.variant, device=args.device, full_precision=True,
                             cfg_strength=DIAL_CFGS[0], num_steps=args.num_steps,
                             duration_sec=args.duration, enable_conditions=True)
    ib = InstrumentedBackend(backend)
    measurer = RealFoleyMeasurer(device=args.device)

    for clip, cfg in todo:
        tag = args.tag_fmt.format(cfg=cfg)
        payload = run_clip_cfg(ib, measurer, class_axis, store, clip,
                               args.clips_root / f"{clip}.mp4", schedule, args.seed,
                               cfg=cfg, tag=tag, n_independent=args.n_independent,
                               k_forks=args.k_forks)
        payload["budget"] = budget.summary()
        store.journal_done(f"{tag}__{clip}", payload)
    print(f"[dial] shard {args.shard} complete; budget: {budget.summary()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
