#!/usr/bin/env python
"""Arc-3 Tier-B B1 — class readability, fairly tested (pre-reg §B1). CPU-only, NON-GATING.

Reads the CACHED cfg=1.0 pooled per-layer features and the measured final class
self-targets, trains TWO probe families (linear ridge baseline + a 1-hidden-layer width-256
ReLU MLP, weight-decay 1e-3, early-stopped) per (layer, s) on the FROZEN probe_train clips'
independents, evaluates on the eval clips' independents, and computes the §B1 decision:

  s_read_internal_class = min s with best (probe x layer) eval acc >= theta_read (0.70)
  AND >= chance + 0.15;  <= 0.45 -> CLASS_INTERNAL_READOUT_FOUND (record the winning
  (probe, layer) as the B4 class-head spec); else R2_CLASS_CONFIRMED.

Feature/label loading mirrors scripts/track_p.py exactly (frozen split, single_event 200,
16 independents, labels kept verbatim incl. the measurer's 'abstain' outcome). Output:
results/stage0/arc3/b1_class_readability.json. Outputs never feed decision tokens.

Run (CPU):
  .venv/bin/python scripts/b1_class_readability.py
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from foley_cw.class_probes import (  # noqa: E402
    class_readability_curve, s_read_internal_class)

PHASE1_S_GRID = (0.05, 0.15, 0.25, 0.35, 0.45, 0.60, 0.75, 0.90)
S_READOUT_EXTERNAL = 0.75   # reference: external audio-tagger crosses θ_read only at 0.90


def load_class_labels(measurements: Path, tag: str) -> dict:
    """label[gid] = final class self-target (str, verbatim incl. 'abstain'), independents only.
    Mirrors scripts/track_p.py.load_labels restricted to the class axis."""
    lab: dict[str, str] = {}
    role = f"{tag}_independent"
    with measurements.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            d = json.loads(line)
            e = d.get("extra") or {}
            if e.get("role") != role or d.get("axis_id") != "class":
                continue
            v = (d.get("target") or {}).get("label")
            if v is not None:
                lab[d["gen_id"]] = str(v)
    return lab


def build_feats(out: Path, tag: str, single: list[str], train_clips: set,
                eval_clips: set, labels: dict, n_independent: int):
    """feats_by_layer_s[(L,s)] = {"train": (X,y), "eval": (X,y)} from cached pooled npz.
    Built per-s (memory-bounded), layers sliced from the (N, n_layers, D) stack."""
    feat_dir = out / "features"
    feats: dict = {}
    layers: list[int] = []
    counts = {"train": {}, "eval": {}, "missing_feat": 0, "missing_label": 0}
    for s in PHASE1_S_GRID:
        rows, meta = [], []
        for clip in single:
            for j in range(n_independent):
                gid = f"{clip}__{tag}_ind{j}"
                p = feat_dir / f"{gid}__s{s:.2f}.npz"
                if not p.exists():
                    counts["missing_feat"] += 1
                    continue
                rows.append(np.load(p)["pooled"].astype(np.float32))   # (n_layers, D)
                meta.append((clip, gid))
        if not rows:
            continue
        X = np.stack(rows)                                              # (N, n_layers, D)
        if not layers:
            layers = list(range(X.shape[1]))
        is_train = np.array([c in train_clips for c, _ in meta])
        is_eval = np.array([c in eval_clips for c, _ in meta])
        y_all = [labels.get(gid) for _, gid in meta]
        ok = np.array([v is not None for v in y_all])
        counts["missing_label"] += int((~ok).sum())
        for L in layers:
            XL = X[:, L, :]
            tr = ok & is_train; te = ok & is_eval
            feats[(L, s)] = {
                "train": (XL[tr], [y_all[i] for i in np.where(tr)[0]]),
                "eval": (XL[te], [y_all[i] for i in np.where(te)[0]]),
            }
        counts["train"][f"{s:.2f}"] = int((ok & is_train).sum())
        counts["eval"][f"{s:.2f}"] = int((ok & is_eval).sum())
        del X, rows
    return feats, layers, counts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("results/stage0"))
    ap.add_argument("--manifest", type=Path,
                    default=Path("data/manifests/phase1_manifest_frozen.json"))
    ap.add_argument("--thresholds", type=Path, default=Path("configs/thresholds.json"))
    ap.add_argument("--tag", default="p1cfg1")
    ap.add_argument("--n-independent", type=int, default=16)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--mlp-seed", type=int, default=0)
    ap.add_argument("--margin", type=float, default=0.15)
    args = ap.parse_args()

    man = json.loads(args.manifest.read_text())
    split = man["split_60_40_by_clip"]
    train_clips = set(str(c) for c in split["probe_train"])
    eval_clips = set(str(c) for c in split["eval"])
    single = [str(c) for c in man["clips"]["single_event"]]
    theta_read = json.loads(args.thresholds.read_text())["theta_read"]

    # data-leakage guard: train and eval clip sets must be disjoint (frozen split).
    assert not (train_clips & eval_clips), "train/eval clip overlap — split leakage"

    labels = load_class_labels(args.out / "measurements" / "measurements.jsonl", args.tag)
    feats, layers, counts = build_feats(args.out, args.tag, single, train_clips,
                                        eval_clips, labels, args.n_independent)

    curve = class_readability_curve(feats, layers, list(PHASE1_S_GRID),
                                    lam=args.lam, mlp_seed=args.mlp_seed)
    sri = s_read_internal_class(curve["best_acc"], curve["chance"],
                                theta_read=theta_read, margin=args.margin)

    found = math.isfinite(sri) and sri <= 0.45
    if found:
        token = "CLASS_INTERNAL_READOUT_FOUND"
        win = curve["best_spec"][sri]
        class_head_spec = {"probe": win["probe"], "layer": win["layer"], "s": sri,
                           "acc": win["acc"], "chance": curve["chance"][sri]}
    else:
        token = "R2_CLASS_CONFIRMED"
        class_head_spec = None

    # also surface the overall best (probe x layer x s) for the report, even if it never
    # satisfies the early-readout rule.
    fin = {s: a for s, a in curve["best_acc"].items() if np.isfinite(a)}
    overall_best_s = max(fin, key=fin.get) if fin else None
    overall_best = curve["best_spec"].get(overall_best_s) if overall_best_s is not None else None

    out = {
        "_doc": "Arc-3 Tier-B B1 class readability (pre-reg §B1) — NON-GATING. Linear ridge "
                "baseline + MLP (1x256 ReLU, wd 1e-3, early-stopped) on cached cfg=1.0 pooled "
                "per-layer features, frozen 60/40 split. labels = each independent's measured "
                "final class self-target (verbatim, incl. 'abstain'), matching Track-P.",
        "tag": args.tag, "theta_read": theta_read, "margin": args.margin,
        "n_layers": len(layers), "s_grid": list(PHASE1_S_GRID),
        "s_readout_external_ref": S_READOUT_EXTERNAL,
        "counts": counts,
        "chance_by_s": {f"{s:.2f}": curve["chance"][s] for s in PHASE1_S_GRID},
        "best_acc_by_s": {f"{s:.2f}": curve["best_acc"][s] for s in PHASE1_S_GRID},
        "best_spec_by_s": {f"{s:.2f}": curve["best_spec"][s] for s in PHASE1_S_GRID},
        "ridge_best_by_s": {f"{s:.2f}": curve["ridge_best"][s] for s in PHASE1_S_GRID},
        "mlp_best_by_s": {f"{s:.2f}": curve["mlp_best"][s] for s in PHASE1_S_GRID},
        "per_s_per_layer": {
            f"{s:.2f}": {
                "ridge": {str(L): curve["per_s"][s]["ridge"].get(L) for L in layers},
                "mlp": {str(L): curve["per_s"][s]["mlp"].get(L) for L in layers},
            } for s in PHASE1_S_GRID},
        "s_read_internal_class": sri,
        "overall_best": overall_best,
        "decision": {
            "rule": "s_read_internal_class<=0.45 -> CLASS_INTERNAL_READOUT_FOUND else "
                    "R2_CLASS_CONFIRMED",
            "suggested_token": token,
            "class_head_spec_for_b4": class_head_spec,
        },
    }

    out_dir = args.out / "arc3"; out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "b1_class_readability.json").write_text(json.dumps(out, indent=2, default=str))

    # console summary (per-s best-acc curve + chance)
    print("# B1 class readability (pre-reg §B1) — NON-GATING")
    print(f"theta_read={theta_read}  margin={args.margin}  n_layers={len(layers)}")
    print("s     | best_acc | chance | probe@layer | ridge_best | mlp_best")
    for s in PHASE1_S_GRID:
        sp = curve["best_spec"][s]
        print(f"{s:<5} | {curve['best_acc'][s]:.3f}    | {curve['chance'][s]:.3f}  | "
              f"{str(sp['probe'])}@{sp['layer']:<3} | {curve['ridge_best'][s]:.3f}      | "
              f"{curve['mlp_best'][s]:.3f}")
    print(f"s_read_internal_class = "
          f"{'never' if math.isnan(sri) else sri}  ->  {token}")
    if class_head_spec:
        print(f"B4 class-head spec: {class_head_spec}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
