#!/usr/bin/env python
"""Build a self-contained labeling HTML bundle (June-13 manual section 13).

Tasks:
  anchor   — 30 clips from data/manifests/anchor_check_30.csv (original mp4,
             video+audio, onset-tap; pre-fills proposed_onset_s). Buildable now.
  validity — ~50 GENERATED-audio clips from screening finals
             (results/stage0/finals/<clip>__screen_ind0.wav), audio-only,
             presence + 12-class + abstain + onset-tap. Built after screening;
             qwen labels embedded from the MLLM sidecar.csv for the same clips.

Output: results/labeling/<task>_bundle.html (single self-contained file) +
results/labeling/<task>_manifest.json. CPU-only.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from foley_cw.labeling_tool import (audio_item, event_classes, video_item,  # noqa: E402
                                    write_bundle)


def build_anchor(args, classes) -> dict:
    rows = list(csv.DictReader(args.anchor_csv.open()))
    if args.n:
        rows = rows[: args.n]
    items = []
    for r in rows:
        onset = float(r["proposed_onset_s"]) if r.get("proposed_onset_s") else None
        items.append(video_item(r["key"], Path(r["path"]),
                                caption=r.get("notes", "") or "", proposed_onset_s=onset))
    out = args.out_dir / "anchor_bundle.html"
    return write_bundle(out, "anchor", items, classes,
                        title=f"foley-cw — anchor onset marks ({len(items)} clips)",
                        prompt_version=args.prompt_version)


def load_qwen(sidecar_csv: Path) -> dict[str, dict]:
    """clip -> {axis_id: label} from the MLLM sidecar (for read-only display)."""
    q: dict[str, dict] = {}
    if not sidecar_csv.exists():
        return q
    for r in csv.DictReader(sidecar_csv.open()):
        q.setdefault(r["clip"], {})[r["axis_id"]] = r["label"]
    return q


def build_validity(args, classes) -> dict:
    qwen = load_qwen(args.sidecar_csv)
    # prefer clips that qwen judged (so human+qwen are on the same pass)
    clips = sorted(qwen) if qwen else None
    if clips is None:
        clips = sorted(p.stem.replace("__screen_ind0", "")
                       for p in args.finals_dir.glob("*__screen_ind0.wav"))
    clips = clips[: args.n] if args.n else clips
    items = []
    for clip in clips:
        wav = args.finals_dir / f"{clip}__screen_ind0.wav"
        if not wav.exists():
            continue
        items.append(audio_item(clip, wav, qwen=qwen.get(clip, {})))
    if not items:
        raise SystemExit(f"no generated finals found in {args.finals_dir} — run screening first")
    out = args.out_dir / "validity_bundle.html"
    return write_bundle(out, "validity", items, classes,
                        title=f"foley-cw — validity labels ({len(items)} clips, audio only)",
                        prompt_version=args.prompt_version)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["anchor", "validity"])
    ap.add_argument("--out-dir", type=Path, default=Path("results/labeling"))
    ap.add_argument("--coarse-map", type=Path, default=Path("configs/coarse_class_map.json"))
    ap.add_argument("--anchor-csv", type=Path, default=Path("data/manifests/anchor_check_30.csv"))
    ap.add_argument("--finals-dir", type=Path, default=Path("results/stage0/finals"))
    ap.add_argument("--sidecar-csv", type=Path,
                    default=Path("results/stage0/mllm_sidecar/sidecar.csv"))
    ap.add_argument("--n", type=int, default=0, help="cap number of clips (0 = all)")
    ap.add_argument("--prompt-version", default="v1",
                    help="bundle version; sets the localStorage key so a demo bundle "
                         "(e.g. 'demo') cannot overwrite the real 'v1' bundle's progress")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    classes = event_classes(args.coarse_map)

    manifest = build_anchor(args, classes) if args.task == "anchor" else build_validity(args, classes)
    (args.out_dir / f"{args.task}_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[labeling] wrote {manifest['out']} — {manifest['n_clips']} clips, "
          f"{manifest['bytes']/1e6:.1f} MB; classes={classes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
