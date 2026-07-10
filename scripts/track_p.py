#!/usr/bin/env python
"""Track P — internal-feature probes (manual §7). Parallel, NON-GATING.

Trains a linear probe per (axis, layer, s) on the cached pooled per-layer features of the
Phase-1 independents pool, predicting each independent's OWN final self-target, on the
frozen 60/40 clip split. Reports the best-layer internal-readout curve and
s_read_internal(axis). The headline: is s_read_internal ≈ s_commit ≪ s_read_external?
Outputs NEVER feed decision tokens. CPU-only (reads cached features).

Output: results/stage0/phase1/internal_probe_report.md + track_p_<tag>.json (Fig-4 data).
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

from foley_cw.internal_probes import best_layer_curve, s_read_internal  # noqa: E402

PHASE1_S_GRID = (0.05, 0.15, 0.25, 0.35, 0.45, 0.60, 0.75, 0.90)
CATEGORICAL_AXES = ("presence", "timing", "class")
N_LAYERS = 20


def load_labels(measurements: Path, tag: str) -> dict:
    """label[axis_id][gid] = final self-target label, for the independents (role *_independent)."""
    lab: dict[str, dict[str, object]] = {a: {} for a in CATEGORICAL_AXES}
    role = f"{tag}_independent"
    with measurements.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            d = json.loads(line)
            e = d.get("extra") or {}
            if e.get("role") != role:
                continue
            ax = d["axis_id"]
            if ax in lab and (d.get("target") or {}).get("label") is not None:
                lab[ax][d["gen_id"]] = str(d["target"]["label"])
    return lab


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("results/stage0"))
    ap.add_argument("--manifest", type=Path, default=Path("data/manifests/phase1_manifest_frozen.json"))
    ap.add_argument("--thresholds", type=Path, default=Path("configs/thresholds.json"))
    ap.add_argument("--tag", default="p1cfg1")
    ap.add_argument("--n-independent", type=int, default=16)
    ap.add_argument("--lam", type=float, default=1.0)
    args = ap.parse_args()

    man = json.loads(args.manifest.read_text())
    split = man["split_60_40_by_clip"]
    train_clips = set(str(c) for c in split["probe_train"])
    eval_clips = set(str(c) for c in split["eval"])
    single = [str(c) for c in man["clips"]["single_event"]]
    theta_read = json.loads(args.thresholds.read_text())["theta_read"]

    feat_dir = args.out / "features"
    labels = load_labels(args.out / "measurements" / "measurements.jsonl", args.tag)

    # Per axis: feats_by_layer_s[(L,s)] = {"train": (X,y), "eval": (X,y)}.
    # Built per-s to bound memory; layers sliced from the (N, n_layers, D) stack
    # (n_layers read from the stored features — small_16k taps the 12 joint blocks).
    layers: list[int] = []
    feats_by_axis = {ax: {} for ax in CATEGORICAL_AXES}
    for s in PHASE1_S_GRID:
        rows, meta = [], []
        for clip in single:
            for j in range(args.n_independent):
                gid = f"{clip}__{args.tag}_ind{j}"
                p = feat_dir / f"{gid}__s{s:.2f}.npz"
                if not p.exists():
                    continue
                rows.append(np.load(p)["pooled"].astype(np.float32))   # (20, D)
                meta.append((clip, gid))
        if not rows:
            continue
        X = np.stack(rows)                                              # (N, n_layers, D)
        if not layers:
            layers = list(range(X.shape[1]))
        is_train = np.array([c in train_clips for c, _ in meta])
        is_eval = np.array([c in eval_clips for c, _ in meta])
        for ax in CATEGORICAL_AXES:
            y_all = [labels[ax].get(gid) for _, gid in meta]
            ok = np.array([v is not None for v in y_all])
            for L in layers:
                XL = X[:, L, :]
                tr = ok & is_train; te = ok & is_eval
                feats_by_axis[ax][(L, s)] = {
                    "train": (XL[tr], [y_all[i] for i in np.where(tr)[0]]),
                    "eval": (XL[te], [y_all[i] for i in np.where(te)[0]]),
                }
        del X, rows

    out = {"_doc": "Track P internal-feature probes (§7) — NON-GATING. best-layer linear "
                   "probe accuracy per (axis, s); s_read_internal vs θ_read.",
           "theta_read": theta_read, "tag": args.tag, "axes": {}}
    for ax in CATEGORICAL_AXES:
        curve = best_layer_curve(feats_by_axis[ax], layers, list(PHASE1_S_GRID), lam=args.lam)
        sri = s_read_internal(curve["best_layer_acc"], theta_read)
        out["axes"][ax] = {"best_layer_acc": curve["best_layer_acc"],
                           "best_layer": curve["best_layer"], "s_read_internal": sri}

    phase1_dir = args.out / "phase1"; phase1_dir.mkdir(parents=True, exist_ok=True)
    (phase1_dir / f"track_p_{args.tag}.json").write_text(json.dumps(out, indent=2, default=str))

    L = [f"# Track P — Internal-Feature Probes ({args.tag}) — Fig 4 (NON-GATING)", "",
         f"Best-layer linear-probe accuracy on the frozen 60/40 split; θ_read = {theta_read}. "
         "s_read_internal = earliest s the generator's own features predict its final "
         "self-target. Compare to s_commit (Phase 1) and s_read_external (Phase 2).", "",
         "| axis | " + " | ".join(f"s={s}" for s in PHASE1_S_GRID) + " | s_read_internal |",
         "|---|" + "---|" * (len(PHASE1_S_GRID) + 1)]
    for ax in CATEGORICAL_AXES:
        a = out["axes"][ax]
        accs = " | ".join(
            ("nan" if math.isnan(a["best_layer_acc"].get(s, float("nan")))
             else f"{a['best_layer_acc'][s]:.2f}") for s in PHASE1_S_GRID)
        sri = "never" if math.isnan(a["s_read_internal"]) else f"{a['s_read_internal']}"
        L.append(f"| {ax} | {accs} | {sri} |")
    (phase1_dir / "internal_probe_report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
