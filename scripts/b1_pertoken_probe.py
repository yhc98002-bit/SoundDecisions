#!/usr/bin/env python
"""B1 families 3-4 — per-token + cross-attention class probe (pre-reg §B1; discharges R2).

Consumes the un-pooled re-tap (results/stage0/arc3/pertoken/<gid>__s<S>.npz): family 3 =
token_mean_max (n_layers, 2D) per layer; family 4 = xattn_clip (n_xlayers, T) cross-attention
map. Trains a ridge probe per (family, layer, s) predicting each cfg=1.0 independent's final
class on the FROZEN 60/40 split (train probe_train clips, eval eval clips), chance = eval
majority prior. Decision (pre-reg §B1): best eval acc across families×layers reaching θ_read=0.70
AND >= chance+0.15 early (s<=0.45) -> CLASS_INTERNAL_READOUT_FOUND, else R2 confirmed on per-token.
"""
from __future__ import annotations
import json, os, sys, glob
from pathlib import Path
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from foley_cw.internal_probes import probe_accuracy  # noqa: E402

S_GRID = (0.05, 0.15, 0.25, 0.35, 0.45, 0.60, 0.75, 0.90)
THETA_READ, MARGIN = 0.70, 0.15
FEAT_DIR = Path("results/stage0/arc3/pertoken")


def load_labels(meas: Path, tag="p1cfg1") -> dict:
    lab = {}
    for line in meas.open():
        if not line.strip():
            continue
        d = json.loads(line); e = d.get("extra") or {}
        if e.get("role") == f"{tag}_independent" and d["axis_id"] == "class" \
                and (d.get("target") or {}).get("label") is not None:
            lab[d["gen_id"]] = str(d["target"]["label"])
    return lab


def main() -> int:
    man = json.loads(Path("data/manifests/phase1_manifest_frozen.json").read_text())
    split = man["split_60_40_by_clip"]
    train_c = set(map(str, split["probe_train"])); eval_c = set(map(str, split["eval"]))
    single = [str(c) for c in man["clips"]["single_event"]]
    labels = load_labels(Path("results/stage0/measurements/measurements.jsonl"))

    # discover layer counts from one file
    sample = sorted(glob.glob(str(FEAT_DIR / "*__s0.05.npz")))
    if not sample:
        print("[b1-pertoken-probe] no per-token npz found"); return 2
    z0 = np.load(sample[0]); n_lay_tmm = z0["token_mean_max"].shape[0]
    n_lay_x = z0["xattn_clip"].shape[0]

    families = {"token_mean_max": n_lay_tmm, "xattn_clip": n_lay_x}
    best = {"acc": -1.0, "family": None, "layer": None, "s": None}
    per_s_best = {}
    chance_by_s = {}
    for s in S_GRID:
        # gather feature rows per family+layer with frozen-split membership
        rows, meta = [], []
        for clip in single:
            for j in range(16):
                gid = f"{clip}__p1cfg1_ind{j}"
                p = FEAT_DIR / f"{gid}__s{s:.2f}.npz"
                if not p.exists() or gid not in labels:
                    continue
                rows.append(p); meta.append((clip, gid))
        if not rows:
            continue
        is_tr = np.array([c in train_c for c, _ in meta])
        is_ev = np.array([c in eval_c for c, _ in meta])
        y = [labels[g] for _, g in meta]
        ev_labels = [y[i] for i in np.where(is_ev)[0]]
        chance = max((ev_labels.count(l) for l in set(ev_labels)), default=0) / max(len(ev_labels), 1)
        chance_by_s[s] = chance
        s_best = chance
        for fam, n_lay in families.items():
            arrs = [np.load(p)[fam] for p in rows]      # (N, n_lay, D)
            X = np.stack(arrs)
            for L in range(n_lay):
                XL = X[:, L, :]
                acc = probe_accuracy(XL[is_tr], [y[i] for i in np.where(is_tr)[0]],
                                     XL[is_ev], ev_labels, lam=1.0)
                if np.isfinite(acc) and acc > s_best:
                    s_best = acc
                if np.isfinite(acc) and acc > best["acc"]:
                    best = {"acc": float(acc), "family": fam, "layer": int(L), "s": float(s)}
        per_s_best[f"{s:g}"] = s_best
        print(f"[b1-pertoken] s={s}: best acc {s_best:.3f} (chance {chance:.3f})", flush=True)

    # decision: earliest s where best-family reaches threshold
    s_read = float("nan")
    for s in S_GRID:
        sb = per_s_best.get(f"{s:g}", float("nan"))
        if np.isfinite(sb) and sb >= THETA_READ and sb >= chance_by_s.get(s, 1.0) + MARGIN:
            s_read = float(s); break
    found = np.isfinite(s_read) and s_read <= 0.45
    token = "CLASS_INTERNAL_READOUT_FOUND" if found else "R2_CLASS_CONFIRMED"
    out = {"_doc": "B1 families 3-4 (per-token + cross-attn) class probe — discharges R2.",
           "best": best, "per_s_best_acc": per_s_best, "chance_by_s": chance_by_s,
           "s_read_internal_class": s_read, "theta_read": THETA_READ,
           "suggested_token": token}
    Path("results/stage0/arc3").mkdir(parents=True, exist_ok=True)
    Path("results/stage0/arc3/b1_pertoken_probe.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\n[b1-pertoken] GLOBAL best {best['acc']:.3f} ({best['family']} L{best['layer']} "
          f"s={best['s']}) vs chance ~{np.mean(list(chance_by_s.values())):.3f} -> {token}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
