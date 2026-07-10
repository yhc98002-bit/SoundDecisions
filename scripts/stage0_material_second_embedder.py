#!/usr/bin/env python
"""Stage-0 material second-embedder validity (manual §3.3) — CLAP vs PANNs consistency.

The material/timbre axis has no MLLM-judgeable gold, and (being ~0% video-pinned) it is
the axis most likely to carry the trajectory-share window story — so §3.3 validates it
via a SECOND embedder rather than relegating it. CLAP (512-d) and PANNs (2048-d) live in
different spaces, so a direct cosine is undefined; the principled cross-embedder
consistency is **representational-similarity analysis (RSA)**: if two independent
embedders induce the same *geometry* over clips, the material axis's similarity structure
is real, not embedder-specific noise.

Method: for N clips compute each embedder's pairwise-cosine matrix, then the Spearman
rank correlation between the two matrices' off-diagonal entries (bootstrap over clips for
a CI). High RSA ρ ⇒ material validated. Diagnostic / correctness-layer (NOT a
GO_MAPS_PHASE precondition).

Needs the venv (torch); runs on CPU. Output: results/stage0/material_second_embedder.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _pairwise_cosine_offdiag(embs: np.ndarray) -> np.ndarray:
    """Flattened upper-triangle (i<j) cosines of row-normalized embeddings."""
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    g = (embs / norms) @ (embs / norms).T
    iu = np.triu_indices(embs.shape[0], k=1)
    return np.clip(g[iu], -1.0, 1.0)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman ρ via rank+Pearson (numpy-only; ties via average rank)."""
    def rank(a):
        order = np.argsort(a, kind="mergesort")
        r = np.empty(len(a), dtype=float)
        r[order] = np.arange(len(a), dtype=float)
        # average tied ranks
        _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
        csum = np.cumsum(counts)
        starts = csum - counts
        avg = (starts + csum - 1) / 2.0
        return avg[inv]
    rx, ry = rank(x), rank(y)
    rx -= rx.mean(); ry -= ry.mean()
    denom = float(np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))
    return float((rx * ry).sum() / denom) if denom > 0 else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--human-jsonl", type=Path,
                    default=Path("results/labeling/labels_validity_v1.jsonl"),
                    help="clip set to validate on (the validity finals)")
    ap.add_argument("--finals-dir", type=Path, default=Path("results/stage0/finals"))
    ap.add_argument("--out", type=Path, default=Path("results/stage0"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import soundfile as sf

    from foley_cw.config import load_config
    from foley_cw.real_measurer import RealFoleyMeasurer

    clips = []
    for line in args.human_jsonl.read_text().splitlines():
        if line.strip():
            clips.append(str(json.loads(line)["clip_id"]))
    clips = sorted(set(clips))

    material_axis = next(a for a in load_config().axes if a.id == "material")
    measurer = RealFoleyMeasurer(device=args.device)

    clap, panns, used = [], [], []
    for i, clip in enumerate(clips):
        p = args.finals_dir / f"{clip}__screen_ind0.wav"
        if not p.exists():
            continue
        wav, _sr = sf.read(p, dtype="float32")
        audio = np.asarray(wav, dtype=np.float32)
        clap.append(np.asarray(measurer.measure(audio, material_axis).embedding, dtype=float))
        panns.append(np.asarray(measurer.panns_embedding(audio), dtype=float))
        used.append(clip)
        if (i + 1) % 10 == 0:
            print(f"[material-2emb] {i + 1}/{len(clips)} embedded", flush=True)

    clap = np.stack(clap); panns = np.stack(panns)
    n = len(used)
    cc = _pairwise_cosine_offdiag(clap)
    pc = _pairwise_cosine_offdiag(panns)
    rsa = _spearman(cc, pc)

    # bootstrap over CLIPS (resample clips, recompute both off-diag sets, recorrelate)
    rng = np.random.default_rng(args.seed)
    boots = []
    for _ in range(args.n_boot):
        idx = rng.integers(0, n, n)
        ci = _pairwise_cosine_offdiag(clap[idx])
        pi = _pairwise_cosine_offdiag(panns[idx])
        r = _spearman(ci, pi)
        if np.isfinite(r):
            boots.append(r)
    lo, hi = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))) \
        if boots else (float("nan"), float("nan"))

    out = {
        "_doc": "Material axis second-embedder validity (§3.3): CLAP-vs-PANNs RSA "
                "(Spearman ρ of pairwise-cosine geometries). Diagnostic / correctness "
                "layer; NOT a GO_MAPS_PHASE precondition.",
        "n_clips": n, "n_pairs": int(n * (n - 1) // 2),
        "rsa_spearman_rho": rsa,
        "rsa_ci95": [lo, hi],
        "mean_clap_cos": float(np.mean(cc)), "mean_panns_cos": float(np.mean(pc)),
        "interpretation": ("two independent embedders agree on the material geometry "
                           "(validated)" if rsa >= 0.5 else
                           "weak cross-embedder agreement — material geometry is "
                           "embedder-sensitive (report as a caveat)"),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "material_second_embedder.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
