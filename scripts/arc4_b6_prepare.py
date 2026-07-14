#!/usr/bin/env python
"""Freeze the Arc-4 B6 balanced, cross-class condition-swap pair manifest."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from foley_cw.arc4_gpu import (  # noqa: E402
    B6_S_GRID,
    load_confident_clip_labels,
    select_balanced_pairs,
    sha256_file,
    validate_pair_manifest,
)

ROLE_BY_CFG = {1.0: "p1cfg1_independent", 4.5: "p1cfg45_independent"}


def build_manifest(measurements: Path, n_pairs: int, seed: int) -> dict:
    pairs = []
    eligible = {}
    for cfg, role in ROLE_BY_CFG.items():
        labels = load_confident_clip_labels(measurements, role)
        eligible[f"{cfg:g}"] = len(labels)
        cfg_pairs = select_balanced_pairs(labels, cfg=cfg, n_pairs=n_pairs, seed=seed)
        for pair in cfg_pairs:
            pair["cached_label_role"] = role
        pairs.extend(cfg_pairs)
    manifest = {
        "_doc": "Arc-4 B6 frozen generation manifest. Design metadata only; no swap outputs.",
        "seed": seed,
        "n_pairs_per_cfg": n_pairs,
        "cfgs": [1.0, 4.5],
        "s_grid": list(B6_S_GRID),
        "pair_order": "ordered source/donor",
        "label_rule": (
            "per-clip lexicographic-tiebroken majority of non-abstain class labels over "
            "all 16 cached cfg-specific Phase-1 independents; clips with incomplete rows "
            "or no confident vote are excluded"
        ),
        "balance_rule": (
            "greedy balance of source-class marginal, donor-class marginal, class-pair "
            "cell, and clip reuse; SHA256(seed,cfg,source,donor) deterministic tie-break"
        ),
        "eligible_clips_by_cfg": eligible,
        "measurements_path": str(measurements.resolve()),
        "pairs": pairs,
    }
    validate_pair_manifest(manifest, expected_pairs_per_cfg=n_pairs)
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--measurements", type=Path, required=True)
    ap.add_argument("--out", type=Path,
                    default=Path("results/arc4_quarantine/b6/pair_manifest.json"))
    ap.add_argument("--n-pairs", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.seed != 0:
        raise ValueError("Arc-4 B6 manifest seed is frozen at 0")
    if "arc4_quarantine" not in args.out.parts:
        raise ValueError("B6 pair manifest must live under results/arc4_quarantine/")

    manifest = build_manifest(args.measurements, args.n_pairs, args.seed)
    data = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        if args.out.read_text() != data:
            raise FileExistsError(f"refusing to replace frozen manifest {args.out}")
    else:
        tmp = args.out.with_suffix(".json.tmp")
        tmp.write_text(data)
        os.replace(tmp, args.out)
    digest = sha256_file(args.out)
    sidecar = args.out.with_suffix(".sha256")
    sidecar_data = f"{digest}  {args.out.name}\n"
    if sidecar.exists() and sidecar.read_text() != sidecar_data:
        raise FileExistsError(f"refusing to replace frozen hash {sidecar}")
    sidecar.write_text(sidecar_data)
    print(f"wrote {args.out} ({len(manifest['pairs'])} pairs) sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
