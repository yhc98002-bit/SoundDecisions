#!/usr/bin/env python
"""Arc-3 Tier-B §B2 — conditioning-channel audit (MMAudio bottleneck).

FROZEN pre-registration: experiment/preregistered/arc3_tierB_preregistration.md §B2.
Question: do the RAW video-conditioning features (CLIP + Synchformer, pre-DiT) carry
the final class at all? Probe (ridge + MLP, B1 family) predicts the clip's MEASURED
class from the pooled raw CLIP/Synchformer conditioning, on the FROZEN 60/40 clip
split; metric = held-out eval accuracy vs chance (eval majority-class prior), bootstrap
by video. Decision: cond class acc <= chance+0.15 AND substantially below B1 best ->
COND_BOTTLENECK. CONTINUE regardless (no pause).

Two phases, exactly like scripts/phase1_commitment.py:

  GPU extraction (sharded; mirror phase1_commitment.py --shard i/n, rng_for, journaled,
  kernel-guarded by assert_certified_kernel): per clip, call MMAudioBackend.make_video_cond
  to build the raw conditioning, pull cond.conditions.clip_f (B,VN,D projected CLIP),
  sync_f (B,N,D upsampled Synchformer), and clip_f_c (B,D global CLIP cond) off the
  PreprocessedConditions, pool (mean+max), store npz -> results/stage0/arc3/cond_feats/<clip>.npz.
  Journaled under unit 'b2cond__<clip>' so it is resumable & shardable; rng_for keeps the
  per-clip seed lineage even though make_video_cond is deterministic (parity with phase1).

  CPU aggregate (--aggregate, no GPU): assemble the pooled cond features + per-clip
  measured-class labels (majority self-target over each clip's p1cfg1 independents),
  run ridge + MLP probes on the frozen split vs chance, bootstrap-by-video CIs, compare
  to the B1 internal best, and emit the decision token. Outputs ->
  results/stage0/arc3/b2_cond_audit.json + b2_cond_audit_report.md.

Run (orchestrator) — extraction on a GPU node, then aggregate on CPU:

  scripts/run_on_node.sh an17 'for i in 0 1 2 3 4 5 6 7; do CUDA_VISIBLE_DEVICES=$i \
    .venv/bin/python scripts/b2_cond_audit.py --shard $i/8 > logs/b2_$i.log 2>&1 & done; wait'
  .venv/bin/python scripts/b2_cond_audit.py --aggregate

This script is CPU-clean to import; torch/MMAudio are imported only inside the
extraction path (never at module load, never in --aggregate).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import zlib
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from foley_cw.cond_features import (  # noqa: E402
    build_cond_feature, clip_class_label, decide_cond_bottleneck, run_cond_probe,
)

# The conditioning parts pulled off PreprocessedConditions, in fixed feature order.
# clip_f: projected CLIP seq (B,VN,D); sync_f: upsampled Synchformer seq (B,N,D);
# clip_f_c: global CLIP conditioning vector (B,D). All three are pre-DiT (cached by
# preprocess_conditions before predict_flow runs), i.e. the raw conditioning channel.
COND_KEYS = ("clip_f", "sync_f", "clip_f_c")
P1CFG1_ROLE = "p1cfg1_independent"
N_INDEPENDENT = 16


def rng_for(seed: int, *parts) -> np.random.Generator:
    entropy = [seed] + [zlib.crc32(str(p).encode()) for p in parts]
    return np.random.default_rng(np.random.SeedSequence(entropy))


# ---------------------------------------------------------------------------
# GPU extraction (one npz per clip)
# ---------------------------------------------------------------------------
def extract_clip(backend, clip: str, video: Path, seed: int) -> dict:
    """Build raw conditioning for one clip, pool clip_f/sync_f/clip_f_c, return arrays."""
    import torch  # GPU-only path

    _ = rng_for(seed, clip, "cond")  # seed lineage parity with phase1 (deterministic cond)
    cond = backend.make_video_cond(str(video), video_id=clip)
    pc = cond.conditions  # PreprocessedConditions(clip_f, sync_f, text_f, clip_f_c, text_f_c)
    parts: dict[str, np.ndarray] = {}
    raw_shapes: dict[str, list] = {}
    for key in COND_KEYS:
        t = getattr(pc, key)
        if isinstance(t, torch.Tensor):
            a = t.detach().float().cpu().numpy()
        else:
            a = np.asarray(t, dtype=np.float32)
        raw_shapes[key] = list(a.shape)
        parts[key] = a
    return {"parts": parts, "raw_shapes": raw_shapes}


def run_extraction(args, clips: list[str]) -> int:
    from foley_cw.kernel_provenance import assert_certified_kernel

    # Provenance guard (parity with phase1_commitment.py §15.8). Extraction does not
    # fork at cfg>1, but we keep the same guard so the run is kernel-pinned & audited.
    cert = assert_certified_kernel(args.cfg, args.schedule, args.certified,
                                   require_ratified=args.require_ratified)
    print(f"[b2] kernel OK: {cert['token']} (ratified={cert['ratified']}, "
          f"require_ratified={args.require_ratified})", flush=True)

    out_dir = args.out / "arc3" / "cond_feats"
    out_dir.mkdir(parents=True, exist_ok=True)

    shard_i, shard_n = (int(x) for x in args.shard.split("/"))
    todo = [c for i, c in enumerate(clips) if i % shard_n == shard_i]
    if args.limit:
        todo = todo[: args.limit]
    todo = [c for c in todo if not (out_dir / f"{c}.npz").exists()]
    print(f"[b2] shard {args.shard}: {len(todo)} clips -> {out_dir}", flush=True)
    if not todo:
        return 0

    from foley_cw.mmaudio_backend import MMAudioBackend
    backend = MMAudioBackend(variant=args.variant, device=args.device, full_precision=True,
                             cfg_strength=args.cfg, num_steps=args.num_steps,
                             duration_sec=args.duration, enable_conditions=True)

    for clip in todo:
        t0 = time.time()
        video = args.clips_root / f"{clip}.mp4"
        res = extract_clip(backend, clip, video, args.seed)
        pooled = build_cond_feature(res["parts"], list(COND_KEYS))
        per_part = {k: build_cond_feature(res["parts"], [k]) for k in COND_KEYS}
        # tmp MUST end in .npz: np.savez_compressed auto-appends '.npz' to any name not
        # ending in it (so '<clip>.npz.tmp' is written as '<clip>.npz.tmp.npz', and the
        # os.replace below then fails to find the tmp file).
        tmp = out_dir / f"{clip}.tmp.npz"
        np.savez_compressed(
            tmp,
            pooled=pooled.astype(np.float32),
            clip_f=per_part["clip_f"].astype(np.float32),
            sync_f=per_part["sync_f"].astype(np.float32),
            clip_f_c=per_part["clip_f_c"].astype(np.float32),
            cond_keys=np.array(list(COND_KEYS)),
            raw_shapes=np.array(json.dumps(res["raw_shapes"])),
        )
        os.replace(tmp, out_dir / f"{clip}.npz")
        print(f"[b2 {clip}] {time.time()-t0:.1f}s pooled_dim={pooled.shape[0]} "
              f"shapes={res['raw_shapes']}", flush=True)
    print(f"[b2] shard {args.shard} complete", flush=True)
    return 0


# ---------------------------------------------------------------------------
# CPU aggregate (probe + decision)
# ---------------------------------------------------------------------------
def load_clip_labels(measurements: Path, role: str, axis: str = "class") -> dict[str, str]:
    """clip -> majority measured-class self-target over its p1cfg1 independents."""
    votes: dict[str, list[str]] = {}
    with measurements.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            d = json.loads(line)
            e = d.get("extra") or {}
            if e.get("role") != role or d.get("axis_id") != axis:
                continue
            lab = (d.get("target") or {}).get("label")
            if lab is None:
                continue
            clip = str(e.get("clip"))
            votes.setdefault(clip, []).append(str(lab))
    return {c: lab for c, v in votes.items() if (lab := clip_class_label(v)) is not None}


def b1_best_acc(out: Path, tag: str, axis: str = "class") -> float:
    """Best DiT-internal class accuracy from B1 (track_p_<tag>.json best_layer_acc max).
    The B2 decision compares cond accuracy to this. Returns NaN if B1 not yet produced."""
    p = out / "phase1" / f"track_p_{tag}.json"
    if not p.exists():
        return float("nan")
    try:
        d = json.loads(p.read_text())
        bla = d.get("axes", {}).get(axis, {}).get("best_layer_acc", {})
        vals = [v for v in bla.values() if isinstance(v, (int, float)) and not math.isnan(v)]
        return float(max(vals)) if vals else float("nan")
    except Exception:
        return float("nan")


def run_aggregate(args, clips: list[str], train_clips: set, eval_clips_set: set) -> int:
    feat_dir = args.out / "arc3" / "cond_feats"
    labels = load_clip_labels(args.out / "measurements" / "measurements.jsonl",
                              args.label_role, axis="class")

    # Assemble per-clip feature matrices (pooled + each part), frozen split applied
    # downstream inside run_cond_probe. Only clips with BOTH a cond feature and a label.
    variants = {"pooled_all": "pooled", "clip_f": "clip_f", "sync_f": "sync_f",
                "clip_f_c": "clip_f_c"}
    matrices: dict[str, dict] = {v: {"X": [], "y": [], "clips": []} for v in variants}
    missing_feat, missing_label = [], []
    for clip in clips:
        p = feat_dir / f"{clip}.npz"
        if not p.exists():
            missing_feat.append(clip); continue
        if clip not in labels:
            missing_label.append(clip); continue
        z = np.load(p, allow_pickle=True)
        for vname, key in variants.items():
            matrices[vname]["X"].append(z[key].astype(np.float64))
            matrices[vname]["y"].append(labels[clip])
            matrices[vname]["clips"].append(clip)

    b1 = b1_best_acc(args.out, args.b1_tag, axis="class")
    results = {"_doc": "Arc-3 Tier-B §B2 conditioning-channel audit (frozen pre-reg). "
                       "Ridge+MLP probe of MEASURED class from raw pooled CLIP/Synchformer "
                       "conditioning, frozen 60/40 split, vs chance; bootstrap by video.",
               "label_role": args.label_role, "b1_tag": args.b1_tag,
               "b1_internal_best_class_acc": (None if math.isnan(b1) else b1),
               "n_clips_with_feat_and_label": len(matrices["pooled_all"]["clips"]),
               "missing_feat": missing_feat, "missing_label": missing_label,
               "variants": {}}

    if not matrices["pooled_all"]["clips"]:
        results["status"] = "NO_FEATURES_YET"
        _write_aggregate(args.out, results)
        print("[b2 aggregate] no cond features found yet "
              f"(missing_feat={len(missing_feat)}). Run extraction first.")
        return 1

    # label distribution / chance context
    y_all = matrices["pooled_all"]["y"]
    cl_all = matrices["pooled_all"]["clips"]
    eval_y = [y for y, c in zip(y_all, cl_all) if c in eval_clips_set]
    results["label_dist_all"] = dict(Counter(y_all).most_common())
    results["n_classes"] = len(set(y_all))
    results["eval_majority_prior"] = (
        float(Counter(eval_y).most_common(1)[0][1] / len(eval_y)) if eval_y else None)

    best_cond_acc = float("-inf"); best_cond_key = None
    for vname in variants:
        m = matrices[vname]
        X = np.stack(m["X"]) if m["X"] else np.zeros((0, 1))
        per_family = {}
        for fam in ("ridge", "mlp"):
            r = run_cond_probe(X, m["y"], m["clips"], train_clips, eval_clips_set,
                               family=fam, lam=args.lam, n_boot=args.n_boot,
                               boot_seed=args.boot_seed)
            per_family[fam] = r
            if np.isfinite(r["accuracy"]) and r["accuracy"] > best_cond_acc:
                best_cond_acc = r["accuracy"]; best_cond_key = f"{vname}/{fam}"
        results["variants"][vname] = per_family

    # Decision uses the BEST cond probe (any variant/family) per pre-reg "cond-feature
    # class acc" (the strongest fair readout from the conditioning channel) vs chance.
    # Chance = the eval majority prior on the pooled_all matrix (same eval clips).
    chance = results["variants"]["pooled_all"]["ridge"]["chance"]
    decision = decide_cond_bottleneck(best_cond_acc, chance, b1,
                                      margin=args.margin, b1_gap=args.b1_gap)
    decision["best_cond_probe"] = best_cond_key
    results["chance_for_decision"] = chance
    results["decision"] = decision

    _write_aggregate(args.out, results)
    _write_report(args.out, results)
    print(f"[b2 aggregate] best cond probe = {best_cond_key} acc={best_cond_acc:.3f} "
          f"chance={chance:.3f} B1_best={b1 if not math.isnan(b1) else 'NA'} "
          f"-> {decision['token']} (near_chance={decision['near_chance']}, "
          f"below_b1={decision['below_b1']})")
    return 0


def _write_aggregate(out: Path, results: dict) -> None:
    d = out / "arc3"; d.mkdir(parents=True, exist_ok=True)
    (d / "b2_cond_audit.json").write_text(json.dumps(results, indent=2, default=str))


def _write_report(out: Path, r: dict) -> None:
    L = ["# Arc-3 Tier-B §B2 — Conditioning-Channel Audit (MMAudio bottleneck)", "",
         "Ridge + MLP probe of the clip's MEASURED class from the pooled RAW CLIP + "
         "Synchformer conditioning (pre-DiT), frozen 60/40 clip split; chance = eval "
         "majority-class prior; bootstrap unit = video.", "",
         f"- clips (feature & label): {r['n_clips_with_feat_and_label']}  "
         f"classes: {r.get('n_classes')}  eval majority prior: {r.get('eval_majority_prior')}",
         f"- B1 DiT-internal best class acc: {r.get('b1_internal_best_class_acc')}", "",
         "| conditioning variant | family | eval acc | chance | Δ over chance | 95% CI (by video) | n_eval |",
         "|---|---|---|---|---|---|---|"]
    for vname, fams in r.get("variants", {}).items():
        for fam, m in fams.items():
            ci = m.get("ci95", [float("nan"), float("nan")])
            L.append(f"| {vname} | {fam} | {m['accuracy']:.3f} | {m['chance']:.3f} | "
                     f"{m['delta_over_chance']:.3f} | [{ci[0]:.3f}, {ci[1]:.3f}] | {m['n_eval']} |")
    dec = r.get("decision", {})
    L += ["", "## Decision (frozen pre-reg §B2)",
          f"- best cond probe: {dec.get('best_cond_probe')} (acc {dec.get('cond_best_acc')})",
          f"- near chance (<= chance+{dec.get('margin')}): {dec.get('near_chance')}",
          f"- substantially below B1 (<= B1_best-{dec.get('b1_gap')}): {dec.get('below_b1')}",
          f"- **TOKEN: {dec.get('token')}**  (CONTINUE regardless; offline, no pause)"]
    (out / "arc3" / "b2_cond_audit_report.md").write_text("\n".join(L) + "\n")


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
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--clip-set", default="single_event",
                    choices=["single_event", "two_event", "both"])
    ap.add_argument("--require-ratified", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--label-role", default=P1CFG1_ROLE,
                    help="measurement role giving the per-clip class self-target")
    ap.add_argument("--b1-tag", default="p1cfg1",
                    help="track_p tag whose best class acc is the B1 internal reference")
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--boot-seed", type=int, default=0)
    ap.add_argument("--margin", type=float, default=0.15, help="chance+margin near-chance band")
    ap.add_argument("--b1-gap", type=float, default=0.15,
                    help="cond <= B1_best - b1_gap == 'substantially below B1'")
    ap.add_argument("--aggregate", action="store_true")
    args = ap.parse_args()

    man = json.loads(args.manifest.read_text())
    if args.clip_set == "both":
        clips = sorted(set(man["clips"]["single_event"]) | set(man["clips"]["two_event"]))
    else:
        clips = sorted(str(c) for c in man["clips"][args.clip_set])
    split = man["split_60_40_by_clip"]
    train_clips = set(str(c) for c in split["probe_train"])
    eval_clips_set = set(str(c) for c in split["eval"])

    if args.aggregate:
        return run_aggregate(args, clips, train_clips, eval_clips_set)
    return run_extraction(args, clips)


if __name__ == "__main__":
    raise SystemExit(main())
