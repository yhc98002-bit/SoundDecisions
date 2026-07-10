#!/usr/bin/env python
"""Phase 2 — Readout maps (manual §5). External probe on cached x̂0(s) previews.

Does an external probe reading the blurry Tweedie preview x̂0(s) recover the trajectory's
final self-target, and at what earliest s (s_read)? Subjects = the Phase-1 independents
(each an ODE trajectory; ODE-target = its own final self-target). The audio-tagger probe
is the free full-grid rung (RealFoleyMeasurer on the preview wav); qwen-on-preview (the
primary external probe) is added on a budget-limited subset by --probe qwen.

Per (clip, subject j, s): load previews/<gid>__s{s}.wav, run the probe → predicted label
(or embedding), compare to the final self-target recorded for that independent. Accuracy
per (axis, s) bootstrapped by clip; s_read(axis, probe, ode) = min s with acc ≥ θ_read.

Sharded like the runner; --aggregate (CPU) builds readout_map.csv + emits READOUT_MAP_DONE.
Run:  scripts/run_on_node.sh an17 'for i in 0..7; do CUDA_VISIBLE_DEVICES=$i \
        python scripts/phase2_readout.py --shard $i/16 ... & done; wait'
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from foley_cw.run_store import RunStore, sanitize_unit_id  # noqa: E402
from foley_cw.storage_budget import StorageBudget  # noqa: E402

SUBJECT_TAG = "p1cfg1"   # readout subjects = the Phase-1 cfg=1.0 independents

PHASE1_S_GRID = (0.05, 0.15, 0.25, 0.35, 0.45, 0.60, 0.75, 0.90)
LABEL_AXES = ("presence", "timing", "class")
EMB_AXES = ("material",)


def load_final_targets(measurements: Path, tag: str) -> dict:
    """label[gid][axis] and emb[gid]['material'] for the independents (final self-target)."""
    lab: dict = defaultdict(dict); emb: dict = defaultdict(dict)
    role = f"{tag}_independent"
    with measurements.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            d = json.loads(line)
            if (d.get("extra") or {}).get("role") != role:
                continue
            ax = d["axis_id"]; tgt = d.get("target") or {}
            if ax in LABEL_AXES and tgt.get("label") is not None:
                lab[d["gen_id"]][ax] = str(tgt["label"])
            elif ax in EMB_AXES and tgt.get("embedding") is not None:
                emb[d["gen_id"]][ax] = np.asarray(tgt["embedding"], dtype=np.float32)
    return lab, emb


def _cos(a, b) -> float:
    a = np.asarray(a, float).ravel(); b = np.asarray(b, float).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na and nb else 0.0


def run_clip(measurer, axes, store, clip, finals_lab, finals_emb, n_subj, tag, sf):
    import numpy as np
    rows = []
    prev_dir = store._dir("previews")
    for j in range(n_subj):
        gid = f"{clip}__{SUBJECT_TAG}_ind{j}"   # subject = p1 independent
        lab = finals_lab.get(gid, {}); emb = finals_emb.get(gid, {})
        if not lab:
            continue
        for s in PHASE1_S_GRID:
            prev = prev_dir / f"{sanitize_unit_id(gid)}__s{s:.2f}.wav"
            if not prev.exists():
                continue
            wav, _sr = sf.read(prev, dtype="float32")
            audio = np.asarray(wav, dtype=np.float32)
            for a in axes:
                tgt = measurer.measure(audio, a)
                if a.id in LABEL_AXES:
                    correct = 1.0 if str(tgt.label) == lab.get(a.id) else 0.0
                else:
                    correct = _cos(tgt.embedding, emb.get(a.id)) if a.id in emb else float("nan")
                rows.append({"clip": clip, "j": j, "axis_id": a.id, "s": s,
                             "probe": "audio_tagger", "target": "ode", "correct": correct})
    return rows


def aggregate(out: Path, clips, tag, theta_read):
    store = RunStore(out)
    by = defaultdict(lambda: defaultdict(list))   # (axis,probe,target) -> s -> [correct]
    missing = []
    for clip in clips:
        unit = f"{tag}__{clip}"
        if not store.is_done(unit):
            missing.append(clip); continue
        for r in store.load_journal(unit).get("rows", []):
            key = (r["axis_id"], r["probe"], r["target"])
            by[key][r["s"]].append(r["correct"])
    p2 = out / "phase1"; p2.mkdir(parents=True, exist_ok=True)
    rows, sread = [], {}
    for key, by_s in by.items():
        ax, probe, target = key
        for s in PHASE1_S_GRID:
            vals = [v for v in by_s.get(s, []) if np.isfinite(v)]
            acc = float(np.mean(vals)) if vals else float("nan")
            rows.append({"axis_id": ax, "probe": probe, "target": target, "s": s,
                         "accuracy": acc, "n": len(vals)})
        crossed = [s for s in PHASE1_S_GRID
                   if by_s.get(s) and np.isfinite(np.mean([v for v in by_s[s] if np.isfinite(v)]))
                   and np.mean([v for v in by_s[s] if np.isfinite(v)]) >= theta_read]
        sread[key] = min(crossed) if crossed else float("nan")
    with (p2 / f"readout_map_{tag}.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["axis_id", "probe", "target", "s", "accuracy", "n"])
        w.writeheader(); w.writerows(rows)
    complete = not missing
    L = [f"# Phase-2 Readout Map ({tag}) — s_read external", "",
         f"θ_read = {theta_read}; ODE-target; audio-tagger probe on x̂0(s) previews. "
         f"{len(clips)-len(missing)}/{len(clips)} clips.", "",
         "| axis | probe | s_read_external |", "|---|---|---|"]
    for key, sr in sorted(sread.items()):
        L.append(f"| {key[0]} | {key[1]} | {'never' if math.isnan(sr) else sr} |")
    (p2 / f"readout_report_{tag}.md").write_text("\n".join(L) + "\n")
    tokens = ["READOUT_MAP_DONE"] if complete else []
    (p2 / f"tokens_readout_{tag}.json").write_text(json.dumps(
        {"tokens": tokens, "s_read": {f"{k[0]}/{k[1]}/{k[2]}": v for k, v in sread.items()},
         "complete": complete, "missing": missing[:10]}, indent=2, default=str))
    print("\n".join(L)); print(f"[readout] tokens={tokens or '(incomplete)'} missing={len(missing)}")
    return 0 if complete else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=Path("data/manifests/phase1_manifest_frozen.json"))
    ap.add_argument("--out", type=Path, default=Path("results/stage0"))
    ap.add_argument("--thresholds", type=Path, default=Path("configs/thresholds.json"))
    ap.add_argument("--tag", default="p2cfg1")
    ap.add_argument("--n-subjects", type=int, default=4)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--aggregate", action="store_true")
    args = ap.parse_args()

    theta_read = json.loads(args.thresholds.read_text())["theta_read"]
    clips = sorted(str(c) for c in json.loads(args.manifest.read_text())["clips"]["single_event"])
    if args.aggregate:
        return aggregate(args.out, clips, args.tag, theta_read)

    import soundfile as sf
    from foley_cw.config import load_config
    from foley_cw.real_measurer import RealFoleyMeasurer

    shard_i, shard_n = (int(x) for x in args.shard.split("/"))
    store = RunStore(args.out, budget=StorageBudget(cap_gb=100.0))
    store.account_preexisting_tree()
    todo = [c for i, c in enumerate(clips) if i % shard_n == shard_i]
    if args.limit:
        todo = todo[: args.limit]
    todo = [c for c in todo if not store.is_done(f"{args.tag}__{c}")]
    print(f"[readout] shard {args.shard}: {len(todo)} clips", flush=True)
    if not todo:
        return 0
    axes = [a for a in load_config().axes if a.id in (LABEL_AXES + EMB_AXES)]
    measurer = RealFoleyMeasurer(device=args.device)
    finals_lab, finals_emb = load_final_targets(args.out / "measurements" / "measurements.jsonl",
                                                "p1cfg1")
    for clip in todo:
        rows = run_clip(measurer, axes, store, clip, finals_lab, finals_emb,
                        args.n_subjects, args.tag, sf)
        store.journal_done(f"{args.tag}__{clip}", {"clip": clip, "rows": rows})
        print(f"[readout {clip}] {len(rows)} rows", flush=True)
    print(f"[readout] shard {args.shard} complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
