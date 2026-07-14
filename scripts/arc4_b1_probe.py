#!/usr/bin/env python
"""Frozen Arc-4 B1 completeness gate and streamed confirmatory probe."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from foley_cw.arc4_b1 import (  # noqa: E402
    accuracy_and_video_ci,
    inner_clip_split,
    mlp_predict,
    ridge_predict,
    select_spec,
)
from foley_cw.arc4_gpu import sha256_file, valid_b1_bundle  # noqa: E402

S_GRID = (0.05, 0.15, 0.25, 0.35, 0.45, 0.60, 0.75, 0.90)
N_INDEPENDENT = 16
EXPECTED_CLIPS = 200
EXPECTED_LABELS = EXPECTED_CLIPS * N_INDEPENDENT
EXPECTED_BUNDLES = EXPECTED_LABELS * len(S_GRID)
PROTOCOL_SHA256 = "b85eeece6f18ff7ce3ab254411d06f97cf2446d393f74eb81ad34048131cc03f"


def _expected(manifest: dict):
    clips = [str(clip) for clip in manifest["clips"]["single_event"]]
    if len(clips) != EXPECTED_CLIPS or len(set(clips)) != EXPECTED_CLIPS:
        raise ValueError(f"expected {EXPECTED_CLIPS} unique single-event clips")
    rows = [(clip, j, f"{clip}__p1cfg1_ind{j}")
            for clip in clips for j in range(N_INDEPENDENT)]
    return clips, rows


def _load_labels(path: Path) -> tuple[dict[str, str], int]:
    labels = {}
    duplicate = 0
    with path.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            extra = row.get("extra") or {}
            if row.get("axis_id") != "class" or extra.get("role") != "p1cfg1_independent":
                continue
            label = (row.get("target") or {}).get("label")
            gid = row.get("gen_id")
            if gid is None or label is None:
                continue
            if gid in labels:
                duplicate += 1
            labels[str(gid)] = str(label)
    return labels, duplicate


def _write_once_or_same(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text() != text:
            raise FileExistsError(f"refusing to replace existing B1 output {path}")
        return
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.link(tmp, path)
    tmp.unlink()


def run_gate(args, manifest: dict, clips: list[str], rows: list[tuple]) -> dict:
    if sha256_file(args.protocol) != PROTOCOL_SHA256:
        raise ValueError("B1 protocol hash does not match the frozen protocol")
    train = set(map(str, manifest["split_60_40_by_clip"]["probe_train"]))
    evaluation = set(map(str, manifest["split_60_40_by_clip"]["eval"]))
    single = set(clips)
    if train & evaluation or len(single & train) != 126 or len(single & evaluation) != 74:
        raise ValueError("frozen single-event train/eval split failed integrity checks")

    labels, duplicates = _load_labels(args.measurements)
    expected_gids = {gid for _, _, gid in rows}
    if duplicates or set(labels) != expected_gids:
        missing = len(expected_gids - set(labels))
        extra = len(set(labels) - expected_gids)
        raise RuntimeError(
            f"B1 label completeness failed: labels={len(labels)} duplicates={duplicates} "
            f"missing={missing} extra={extra}")

    pertoken_dir = args.b1_root / "arc3" / "pertoken"
    expected_pertoken = {
        f"{gid}__s{s:.2f}.npz" for _, _, gid in rows for s in S_GRID
    }
    actual_pertoken = {path.name for path in pertoken_dir.glob("*.npz")}
    if actual_pertoken != expected_pertoken:
        raise RuntimeError(
            f"B1 pertoken file-set failed: actual={len(actual_pertoken)} "
            f"missing={len(expected_pertoken-actual_pertoken)} "
            f"extra={len(actual_pertoken-expected_pertoken)}")

    pooled_dir = args.pooled_root / "features"
    expected_pooled = expected_pertoken
    actual_pooled = {
        path.name for path in pooled_dir.glob("*__p1cfg1_ind*__s*.npz")
    }
    if actual_pooled != expected_pooled:
        raise RuntimeError(
            f"B1 pooled file-set failed: actual={len(actual_pooled)} "
            f"missing={len(expected_pooled-actual_pooled)} "
            f"extra={len(actual_pooled-expected_pooled)}")

    for index, name in enumerate(sorted(expected_pertoken), start=1):
        if not valid_b1_bundle(pertoken_dir / name):
            raise RuntimeError(f"invalid B1 pertoken bundle: {name}")
        with np.load(pooled_dir / name, allow_pickle=False) as z:
            if set(z.files) != {"pooled"} or z["pooled"].shape != (12, 448) \
                    or not np.isfinite(z["pooled"]).all():
                raise RuntimeError(f"invalid B1 pooled bundle: {name}")
        if index % 1000 == 0:
            print(f"[b1 gate] validated {index}/{EXPECTED_BUNDLES} bundles", flush=True)

    journal_dir = args.b1_root / "journal"
    journals = list(journal_dir.glob("p1cfg1_pertoken__*.json"))
    if len(journals) != EXPECTED_CLIPS:
        raise RuntimeError(f"B1 journal gate expected 200, got {len(journals)}")
    journal_clips = set()
    for path in journals:
        row = json.loads(path.read_text())
        clip = str(row.get("clip"))
        if row.get("tag") != "p1cfg1" or not math.isclose(float(row.get("cfg")), 1.0) \
                or row.get("bundle_count") != 128:
            raise RuntimeError(f"invalid B1 journal metadata: {path}")
        journal_clips.add(clip)
    if journal_clips != single:
        raise RuntimeError("B1 journal clip set does not match the frozen population")

    gate = {
        "status": "PASS",
        "protocol_sha256": PROTOCOL_SHA256,
        "manifest_sha256": sha256_file(args.manifest),
        "measurements_path": str(args.measurements.resolve()),
        "b1_root": str(args.b1_root.resolve()),
        "pooled_root": str(args.pooled_root.resolve()),
        "n_clips": EXPECTED_CLIPS,
        "n_train_clips": 126,
        "n_eval_clips": 74,
        "n_labels": EXPECTED_LABELS,
        "n_pertoken_bundles": EXPECTED_BUNDLES,
        "n_pooled_bundles": EXPECTED_BUNDLES,
        "n_journals": EXPECTED_CLIPS,
        "schemas": {
            "pooled": [12, 448],
            "token_mean_max": [12, 896],
            "xattn_clip": [4, 64],
        },
    }
    _write_once_or_same(
        args.b1_root / "bundle_gate.json", json.dumps(gate, indent=2, sort_keys=True) + "\n")
    print("[b1 gate] PASS", flush=True)
    return gate


def _load_family(args, rows, s: float, family: str) -> np.ndarray:
    arrays = []
    if family == "pooled":
        directory = args.pooled_root / "features"
        key = "pooled"
    else:
        directory = args.b1_root / "arc3" / "pertoken"
        key = family
    for _, _, gid in rows:
        with np.load(directory / f"{gid}__s{s:.2f}.npz", allow_pickle=False) as z:
            arrays.append(z[key].astype(np.float32))
    return np.stack(arrays)


def run_evaluation(args, manifest: dict, rows: list[tuple], labels: dict[str, str]) -> dict:
    train_clips = set(map(str, manifest["split_60_40_by_clip"]["probe_train"]))
    eval_clips = set(map(str, manifest["split_60_40_by_clip"]["eval"]))
    fit_clips, validation_clips = inner_clip_split(
        {clip for clip in train_clips if any(clip == row[0] for row in rows)})
    row_clips = np.asarray([clip for clip, _, _ in rows], dtype=object)
    y = [labels[gid] for _, _, gid in rows]
    y_array = np.asarray(y, dtype=object)
    is_train = np.asarray([clip in train_clips for clip in row_clips])
    is_eval = np.asarray([clip in eval_clips for clip in row_clips])
    is_fit = np.asarray([clip in fit_clips for clip in row_clips])
    is_validation = np.asarray([clip in validation_clips for clip in row_clips])
    eval_targets = y_array[is_eval].tolist()
    eval_clip_ids = row_clips[is_eval].tolist()
    chance = Counter(eval_targets).most_common(1)[0][1] / len(eval_targets)
    outer_classes = sorted(set(y_array[is_train].tolist()))

    all_specs = []
    per_s = {}
    winners = {}
    for s in S_GRID:
        s_specs = []
        per_s[f"{s:.2f}"] = {}
        for family in ("pooled", "token_mean_max", "xattn_clip"):
            features = _load_family(args, rows, s, family)
            layer_count = features.shape[1]
            per_s[f"{s:.2f}"][family] = {"ridge": {}, "mlp": {}}
            for layer in range(layer_count):
                X = features[:, layer, :]
                ridge = ridge_predict(
                    X[is_train], y_array[is_train].tolist(), X[is_eval], lam=1.0)
                ridge_stats = accuracy_and_video_ci(
                    ridge, eval_targets, eval_clip_ids, n_boot=1000, seed=0)
                ridge_spec = {
                    "s": s, "family": family, "probe": "ridge", "layer": layer,
                    "chance": chance, **ridge_stats,
                }
                per_s[f"{s:.2f}"][family]["ridge"][str(layer)] = ridge_spec
                s_specs.append(ridge_spec); all_specs.append(ridge_spec)

                mlp, early = mlp_predict(
                    X[is_fit], y_array[is_fit].tolist(),
                    X[is_validation], y_array[is_validation].tolist(),
                    X[is_eval], outer_classes=outer_classes, seed=0)
                mlp_stats = accuracy_and_video_ci(
                    mlp, eval_targets, eval_clip_ids, n_boot=1000, seed=0)
                mlp_spec = {
                    "s": s, "family": family, "probe": "mlp", "layer": layer,
                    "chance": chance, **mlp_stats, "early_stopping": early,
                }
                per_s[f"{s:.2f}"][family]["mlp"][str(layer)] = mlp_spec
                s_specs.append(mlp_spec); all_specs.append(mlp_spec)
                print(f"[b1 probe] s={s:.2f} family={family} layer={layer} complete",
                      flush=True)
            del features
        winners[f"{s:.2f}"] = select_spec(s_specs)

    s_read = None
    for s in S_GRID:
        winner = winners[f"{s:.2f}"]
        if winner["accuracy"] >= 0.70 and winner["accuracy"] >= chance + 0.15:
            s_read = s
            break
    token = ("CLASS_INTERNAL_READOUT_FOUND"
             if s_read is not None and s_read <= 0.45 else "R2_CLASS_CONFIRMED")
    family_rank = {"pooled": 0, "token_mean_max": 1, "xattn_clip": 2}
    probe_rank = {"ridge": 0, "mlp": 1}
    global_winner = min(
        all_specs,
        key=lambda spec: (-spec["accuracy"], spec["s"], family_rank[spec["family"]],
                          probe_rank[spec["probe"]], spec["layer"]),
    )
    return {
        "_doc": "Arc-4 B1 confirmatory result under B1_PROTOCOL.md.",
        "protocol_sha256": PROTOCOL_SHA256,
        "chance": chance,
        "theta_read": 0.70,
        "margin": 0.15,
        "s_grid": list(S_GRID),
        "counts": {"train_clips": 126, "fit_clips": 100, "validation_clips": 26,
                   "eval_clips": 74, "train_trajectories": 2016,
                   "eval_trajectories": 1184},
        "per_s": per_s,
        "winner_by_s": winners,
        "global_winner": global_winner,
        "s_read_internal_class": s_read,
        "decision": {"token": token, "complete": True},
    }


def _report(result: dict) -> str:
    lines = [
        "# Arc-4 B1 confirmatory probe",
        "",
        f"Protocol SHA256: `{result['protocol_sha256']}`",
        "",
        "Bootstrap unit: eval video (1,000 draws, seed 0). Labels are trajectory "
        "self-targets and include abstain. The registered search spans all family, "
        "probe, layer, and progress specifications; reported CIs are fixed-specification "
        "intervals and do not correct for search multiplicity.",
        "",
        "| s | accuracy | 95% CI | chance | family | probe | layer |",
        "|---:|---:|---:|---:|---|---|---:|",
    ]
    for s in S_GRID:
        row = result["winner_by_s"][f"{s:.2f}"]
        lines.append(
            f"| {s:.2f} | {row['accuracy']:.4f} | [{row['ci_lo']:.4f}, "
            f"{row['ci_hi']:.4f}] | {row['chance']:.4f} | {row['family']} | "
            f"{row['probe']} | {row['layer']} |")
    lines += [
        "",
        f"s_read_internal_class: `{result['s_read_internal_class']}`",
        f"Decision token: **{result['decision']['token']}**",
        "",
        "Scope: one MMAudio checkpoint and frozen self-target labels; this is not a "
        "human-perceptual class claim.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--b1-root", type=Path, default=Path("results/arc4_b1"))
    ap.add_argument("--pooled-root", type=Path, required=True)
    ap.add_argument("--measurements", type=Path, required=True)
    ap.add_argument("--manifest", type=Path,
                    default=Path("data/manifests/phase1_manifest_frozen.json"))
    ap.add_argument("--protocol", type=Path,
                    default=Path("experiment/preregistered/B1_PROTOCOL.md"))
    ap.add_argument("--evaluate", action="store_true",
                    help="run probes after the mandatory full gate passes")
    args = ap.parse_args()

    manifest = json.loads(args.manifest.read_text())
    clips, rows = _expected(manifest)
    run_gate(args, manifest, clips, rows)
    if not args.evaluate:
        print("[b1] gate-only mode; no evaluation run", flush=True)
        return 0
    labels, _ = _load_labels(args.measurements)
    result = run_evaluation(args, manifest, rows, labels)
    _write_once_or_same(
        args.b1_root / "b1_probe.json", json.dumps(result, indent=2, sort_keys=True) + "\n")
    _write_once_or_same(args.b1_root / "b1_probe.md", _report(result))
    print(f"[b1] decision={result['decision']['token']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
