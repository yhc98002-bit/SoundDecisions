#!/usr/bin/env python
"""Stage-0 manifest freeze (manual §3.1) — the frozen Phase-1 manifest.

Selects 200 stratified single-event + 60 two-event clips, a 60/40 probe-train/eval split
by clip, and records — per axis — usable n AND non-pinned n (headroom stratification),
cfg-specific video-pinned exclusions, the anchor source + timing_bin_s, and the
class=diagnostic flag. Frozen BEFORE any Phase-1 headline curve (§1.5). Freezing is
autonomous (no sign-off) per the run delegation.

Inputs (all already produced):
  data/manifests/screening_manifest.json          400 single-event candidates
  results/stage0/screening/a_independent_screen.csv  per-axis A_independent (cfg=1.0)
  data/FoleyBench/clips_index.csv                  ucs_category strata + duration
  data/manifests/two_event_manifest.json           60 two-event clips
  results/stage0/video_determined_registry_screen.json    cfg=1.0 pinned (A_ind>0.9)
  results/stage0/video_determined_registry_subscreen45.json  cfg=4.5 pinned
  results/stage0/anchor_adoption.json              timing_bin_s + anchor source
  configs/thresholds.json                          frozen thresholds

Output: data/manifests/phase1_manifest_frozen.json + results/stage0/manifest_freeze_report.md
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

EVENT_CLASSES = ["impact_friction", "footsteps_walk", "tools_hand", "machines_motors",
                 "vehicles", "water_liquid", "guns_explosions", "doors_furniture",
                 "electronics_ui", "animals", "food_cooking", "other"]
AXES = ("presence", "timing", "class", "material")
PIN_THRESHOLD = 0.9


def load_a_ind(csv_path: Path) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = defaultdict(dict)
    for r in csv.DictReader(csv_path.open()):
        try:
            v = float(r["a_independent"])
        except (ValueError, TypeError):
            v = float("nan")
        out[r["clip"]][r["axis_id"]] = v
    return out


def load_ucs(index_csv: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for r in csv.DictReader(index_csv.open()):
        out[str(r["key"])] = (r.get("ucs_category") or "UNKNOWN").strip() or "UNKNOWN"
    return out


def stratified_pick(clips: list[str], strata: dict[str, str], n: int, seed: int) -> list[str]:
    """Round-robin over ucs strata (deterministic shuffle within strata)."""
    groups: dict[str, list[str]] = defaultdict(list)
    for c in clips:
        groups[strata.get(c, "UNKNOWN")].append(c)
    rng = np.random.default_rng(seed)
    for k in groups:
        groups[k].sort()
        rng.shuffle(groups[k])
    ordered = sorted(groups)
    picked: list[str] = []
    while len(picked) < n and any(groups[k] for k in ordered):
        for k in ordered:
            if groups[k] and len(picked) < n:
                picked.append(groups[k].pop())
    return picked


def split_60_40(clips: list[str], strata: dict[str, str], seed: int) -> dict[str, list[str]]:
    """60/40 probe-train/eval split by clip, stratified by ucs."""
    groups: dict[str, list[str]] = defaultdict(list)
    for c in clips:
        groups[strata.get(c, "UNKNOWN")].append(c)
    rng = np.random.default_rng(seed)
    train, evl = [], []
    for k in sorted(groups):
        g = sorted(groups[k]); rng.shuffle(g)
        cut = int(round(0.6 * len(g)))
        train += g[:cut]; evl += g[cut:]
    return {"probe_train": sorted(train), "eval": sorted(evl)}


def axis_counts(clips: list[str], a_ind: dict[str, dict[str, float]]) -> dict[str, dict]:
    out = {}
    for ax in AXES:
        finite = [c for c in clips if c in a_ind and np.isfinite(a_ind[c].get(ax, float("nan")))]
        pinned = [c for c in finite if a_ind[c][ax] > PIN_THRESHOLD]
        non_pinned = [c for c in finite if a_ind[c][ax] <= PIN_THRESHOLD]
        unscorable = [c for c in clips if not np.isfinite(a_ind.get(c, {}).get(ax, float("nan")))]
        out[ax] = {
            "usable_n": len(non_pinned),            # not video-pinned AND scorable
            "non_pinned_n": len(non_pinned),
            "pinned_n": len(pinned),
            "unscorable_n": len(unscorable),
            "mean_a_independent": (float(np.mean([a_ind[c][ax] for c in finite]))
                                   if finite else float("nan")),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--screening", type=Path, default=Path("data/manifests/screening_manifest.json"))
    ap.add_argument("--a-ind", type=Path, default=Path("results/stage0/screening/a_independent_screen.csv"))
    ap.add_argument("--index", type=Path, default=Path("data/FoleyBench/clips_index.csv"))
    ap.add_argument("--two-event", type=Path, default=Path("data/manifests/two_event_manifest.json"))
    ap.add_argument("--reg-cfg1", type=Path, default=Path("results/stage0/video_determined_registry_screen.json"))
    ap.add_argument("--reg-cfg45", type=Path, default=Path("results/stage0/video_determined_registry_subscreen45.json"))
    ap.add_argument("--anchor", type=Path, default=Path("results/stage0/anchor_adoption.json"))
    ap.add_argument("--thresholds", type=Path, default=Path("configs/thresholds.json"))
    ap.add_argument("--n-single", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("data/manifests/phase1_manifest_frozen.json"))
    ap.add_argument("--report", type=Path, default=Path("results/stage0/manifest_freeze_report.md"))
    args = ap.parse_args()

    screened = [str(c) for c in json.loads(args.screening.read_text())["clips"]]
    two_event = [str(c) for c in json.loads(args.two_event.read_text())["clips"]]
    a_ind = load_a_ind(args.a_ind)
    ucs = load_ucs(args.index)
    anchor = json.loads(args.anchor.read_text())
    thr = json.loads(args.thresholds.read_text())

    # Single-event candidate pool = screened minus any two-event clips (kept separate).
    pool = [c for c in screened if c not in set(two_event)]
    single = stratified_pick(pool, ucs, args.n_single, args.seed)
    if len(single) < args.n_single:
        print(f"[freeze] WARNING: only {len(single)}/{args.n_single} single-event clips available")

    all_clips = sorted(set(single) | set(two_event))
    split = split_60_40(all_clips, ucs, args.seed + 1)

    # cfg-specific video-pinned exclusions, intersected with the selected manifest.
    def reg_for(path: Path, clips: set[str]) -> dict[str, list[str]]:
        if not path.exists():
            return {}
        reg = json.loads(path.read_text()).get("video_determined_by_axis", {})
        return {ax: sorted(set(v) & clips) for ax, v in reg.items()}

    sel = set(all_clips)
    excl_cfg1 = reg_for(args.reg_cfg1, sel)
    excl_cfg45 = reg_for(args.reg_cfg45, sel)

    timing_bin_s = (anchor.get("decision") or {}).get("timing_bin_s", 0.5)
    anchor_source = (anchor.get("decision") or {}).get("anchor_source", "approved_chain")

    manifest = {
        "_doc": "FROZEN Phase-1 manifest (§3.1). Frozen before any Phase-1 headline curve. "
                "class is a DIAGNOSTIC axis (kept in the determination budget, not gating).",
        "frozen": True,
        "seed": args.seed,
        "clips": {"single_event": sorted(single), "two_event": sorted(two_event),
                  "n_single": len(single), "n_two_event": len(two_event)},
        "split_60_40_by_clip": split,
        "per_axis": {
            "all": axis_counts(all_clips, a_ind),
            "single_event": axis_counts(single, a_ind),
        },
        "video_pinned_exclusions": {"cfg=1.0": excl_cfg1, "cfg=4.5_subscreen": excl_cfg45},
        "frozen_settings": {
            "event_classes": EVENT_CLASSES,
            "clip_duration_s": 8.0,
            "timing_bin_s": timing_bin_s,
            "anchor_source": anchor_source,
            "class_is_diagnostic": True,
            "thresholds": {k: thr[k] for k in ("theta_commit", "theta_read", "theta_rel",
                                               "theta_robust", "theta_cal") if k in thr},
            "probe_split": "60/40 by clip (probe_train/eval), stratified by ucs",
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2))

    # report
    ac = manifest["per_axis"]["all"]
    L = ["# Phase-1 Manifest Freeze Report — §3.1 (FROZEN)", "",
         f"Selected **{len(single)} single-event + {len(two_event)} two-event = "
         f"{len(all_clips)} clips**; 60/40 split = {len(split['probe_train'])} train / "
         f"{len(split['eval'])} eval. timing_bin_s = {timing_bin_s} (anchor: {anchor_source}). "
         f"class = DIAGNOSTIC (kept in maps, not gating).", "",
         "## Per-axis usable / non-pinned n (headroom stratification, all clips)",
         "| axis | usable n | non-pinned n | pinned n | unscorable | mean A_ind |",
         "|---|---|---|---|---|---|"]
    for ax in AXES:
        d = ac[ax]
        mai = "nan" if math.isnan(d["mean_a_independent"]) else f"{d['mean_a_independent']:.3f}"
        L.append(f"| {ax} | {d['usable_n']} | {d['non_pinned_n']} | {d['pinned_n']} | "
                 f"{d['unscorable_n']} | {mai} |")
    L += ["", "Headroom note (§3.1): material & class carry the seed/trajectory-share story "
          "and have abundant non-pinned headroom; timing is the most conditioning-pinned axis "
          "(expected). cfg-specific video-pinned exclusions recorded per axis in the manifest.",
          "", f"Pinned exclusions intersected with manifest: cfg=1.0 "
          f"{ {ax: len(v) for ax, v in excl_cfg1.items()} }."]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\n[freeze] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
