#!/usr/bin/env python
"""Stage 0 — event anchors over the FoleyBench clip index (manual §3.2, Phase 0.4).

For each ok clip in data/FoleyBench/clips_index.csv (optionally restricted by a
--clips manifest JSON), computes anchors_for_clip (audio-track onsets + visual
onset detector + per-clip sigma) and writes:

  results/stage0/anchors.json          per-clip records (key, audio/visual
                                       timestamps + uncertainties, sigma_s) +
                                       the summarize_sigma summary.
  results/stage0/anchor_report.md      dataset.anchor_report_markdown over the
                                       audio-source anchors + a σ_anchor section
                                       + the PROPOSED AMENDMENT note (the
                                       audio-track onset source is NOT in the
                                       approved §3.2 chain; pending PI approval).
  data/manifests/anchor_check_30.csv   30 clips stratified by ucs category for the
                                       light human marks pass (human column empty).

CPU-only, no GPU, no network. Usage (from the repo root):
    .venv/bin/python scripts/stage0_anchors.py
    .venv/bin/python scripts/stage0_anchors.py --max-clips 5   # smoke run
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import numpy as np

# Make foley_cw importable when run from the repo root.
_REPO_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# Index reading (defensive about column naming — the index is produced by a
# separate extraction step and its exact header is not frozen yet)
# ---------------------------------------------------------------------------

_KEY_COLS = ("key", "clip_id", "clip_key", "id")
_PATH_COLS = ("path", "video_path", "mp4_path", "clip_path", "file", "filepath")
_STATUS_COLS = ("status", "ok", "state")
_UCS_COLS = ("ucs", "ucs_category", "category")
_OK_VALUES = {"ok", "true", "1", "yes", "y"}


def _resolve_col(fieldnames: list[str], candidates: tuple[str, ...]) -> Optional[str]:
    lowered = {f.lower(): f for f in fieldnames}
    for cand in candidates:
        if cand in lowered:
            return lowered[cand]
    return None


def _resolve_path(raw: str, index_dir: Path) -> Path:
    p = Path(raw)
    if p.is_absolute():
        return p
    for base in (index_dir, _REPO_ROOT):
        cand = base / p
        if cand.exists():
            return cand
    return p  # left as-is; failure is recorded per clip


def read_clip_index(index_path: Path) -> list[dict[str, Any]]:
    """Read clips_index.csv into [{key, path, ucs, ok}, ...]."""
    with open(index_path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])
        key_col = _resolve_col(fieldnames, _KEY_COLS)
        path_col = _resolve_col(fieldnames, _PATH_COLS)
        status_col = _resolve_col(fieldnames, _STATUS_COLS)
        ucs_col = _resolve_col(fieldnames, _UCS_COLS)
        if key_col is None or path_col is None:
            raise ValueError(
                f"clips_index.csv must have a key column (one of {_KEY_COLS}) and a "
                f"path column (one of {_PATH_COLS}); found {fieldnames}"
            )
        rows: list[dict[str, Any]] = []
        for row in reader:
            ok = True
            if status_col is not None:
                ok = str(row.get(status_col, "")).strip().lower() in _OK_VALUES
            rows.append({
                "key": str(row[key_col]).strip(),
                "path": str(row[path_col]).strip(),
                "ucs": str(row.get(ucs_col, "") or "UNKNOWN").strip() or "UNKNOWN",
                "ok": ok,
            })
    return rows


def read_clips_manifest(manifest_path: Path) -> set[str]:
    """Read an optional --clips JSON: a list of keys, a list of {'key': ...} dicts,
    or {'clips': [...]} with either shape."""
    with open(manifest_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict):
        data = data.get("clips", [])
    keys: set[str] = set()
    for item in data:
        if isinstance(item, dict):
            keys.add(str(item.get("key", "")).strip())
        else:
            keys.add(str(item).strip())
    keys.discard("")
    return keys


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def _anchor_to_json(anchor: Any) -> Optional[dict[str, Any]]:
    if anchor is None:
        return None
    return {
        "timestamps": [float(t) for t in anchor.timestamps],
        "uncertainty": [float(u) for u in anchor.uncertainty],
        "source": anchor.source,
    }


def _fmt(x: float) -> str:
    return "nan" if not np.isfinite(x) else f"{x:.4f}"


def _json_safe(x: Any) -> Any:
    """NaN/inf → None so anchors.json stays strict JSON (json.dumps emits bare NaN)."""
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


# ---------------------------------------------------------------------------
# Stratified human-check manifest
# ---------------------------------------------------------------------------

def stratified_check_set(
    records: list[dict[str, Any]],
    n_check: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Pick *n_check* clips stratified by ucs (round-robin over shuffled strata).

    proposed_onset_s = primary audio-track onset when available, else primary
    visual onset; clips with neither are not checkable and are skipped.
    """
    candidates: list[dict[str, Any]] = []
    for rec in records:
        anchor = rec.get("audio") or rec.get("visual")
        if anchor is None or anchor.n_events == 0:
            continue
        candidates.append({
            "key": rec["key"],
            "path": rec["path"],
            "ucs": rec.get("ucs", "UNKNOWN"),
            "proposed_onset_s": float(anchor.timestamps[0]),
        })

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for cand in candidates:
        groups[cand["ucs"]].append(cand)

    rng = np.random.default_rng(seed)
    ordered_keys = sorted(groups)
    for k in ordered_keys:
        groups[k].sort(key=lambda c: c["key"])  # deterministic before shuffle
        rng.shuffle(groups[k])

    picked: list[dict[str, Any]] = []
    while len(picked) < n_check and any(groups[k] for k in ordered_keys):
        for k in ordered_keys:
            if groups[k] and len(picked) < n_check:
                picked.append(groups[k].pop())
    return picked


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------

def render_report(
    audio_anchors: dict[str, Any],
    summary: dict[str, float],
    n_errors: int,
) -> str:
    from foley_cw.dataset import anchor_report_markdown

    md = anchor_report_markdown(audio_anchors)
    extra = [
        "",
        "## Stage-0 σ_anchor Summary (audio-track vs visual onsets)",
        "",
        "σ per clip = |primary audio-track onset − nearest visual onset|; "
        "stats over clips with BOTH anchors.",
        "",
        "| stat | value |",
        "|---|---|",
        f"| n_clips | {summary['n_clips']} |",
        f"| median σ (s) | {_fmt(summary['median_sigma_s'])} |",
        f"| mean σ (s) | {_fmt(summary['mean_sigma_s'])} |",
        f"| max σ (s) | {_fmt(summary['max_sigma_s'])} |",
        f"| coverage audio | {_fmt(summary['coverage_audio'])} |",
        f"| coverage visual | {_fmt(summary['coverage_visual'])} |",
        f"| coverage both | {_fmt(summary['coverage_both'])} |",
        f"| recommended gross-timing bin (s) | {_fmt(summary['recommended_bin_s'])} |",
        f"| clips with processing errors | {n_errors} |",
        "",
        "Propagation rule (manual §3.2): gross-timing bins ≥ 2·σ_anchor; the "
        "recommended bin width is max(0.5 s, 2·median σ).",
        "",
        "## PROPOSED AMENDMENT (pending PI approval)",
        "",
        "The `foleybench_audio_onset` anchor source above detects onsets on the "
        "clip's OWN audio track. This source is **NOT** part of the approved "
        "manual §3.2 anchor chain (foleybench_metadata → visual_onset_detector → "
        "light_human_marks); it is recorded here as a **PROPOSED AMENDMENT pending "
        "PI approval**. Until approved, audio-track anchors are diagnostic evidence "
        "for σ_anchor estimation only; the frozen anchor source for the timing and "
        "binding axes remains the approved chain, validated against the 30-clip "
        "human check set (data/manifests/anchor_check_30.csv).",
        "",
    ]
    return md + "\n".join(extra)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", type=Path,
                    default=_REPO_ROOT / "data" / "FoleyBench" / "clips_index.csv")
    ap.add_argument("--clips", type=Path, default=None,
                    help="optional manifest JSON restricting which keys to process")
    ap.add_argument("--out-dir", type=Path, default=_REPO_ROOT / "results" / "stage0")
    ap.add_argument("--check-csv", type=Path,
                    default=_REPO_ROOT / "data" / "manifests" / "anchor_check_30.csv")
    ap.add_argument("--n-check", type=int, default=30)
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--seed", type=int, default=0,
                    help="seed for the stratified human-check sample")
    ap.add_argument("--max-clips", type=int, default=None,
                    help="cap processed clips (smoke runs)")
    args = ap.parse_args()

    from foley_cw.visual_anchors import anchors_for_clip, summarize_sigma

    if not args.index.exists():
        print(f"[stage0-anchors] ERROR: clip index not found: {args.index}", file=sys.stderr)
        print("[stage0-anchors] run the clip extraction step first.", file=sys.stderr)
        return 2

    rows = read_clip_index(args.index)
    n_total = len(rows)
    rows = [r for r in rows if r["ok"]]
    n_ok = len(rows)

    if args.clips is not None:
        wanted = read_clips_manifest(args.clips)
        rows = [r for r in rows if r["key"] in wanted]
    if args.max_clips is not None:
        rows = rows[: args.max_clips]

    index_dir = args.index.parent
    records: list[dict[str, Any]] = []     # EventAnchor objects (for summarize_sigma)
    records_json: list[dict[str, Any]] = []
    n_errors = 0
    for i, row in enumerate(rows):
        clip_path = _resolve_path(row["path"], index_dir)
        rec: dict[str, Any] = {
            "key": row["key"], "path": str(clip_path), "ucs": row["ucs"],
            "audio": None, "visual": None, "sigma_s": float("nan"),
        }
        err: Optional[str] = None
        try:
            anchors = anchors_for_clip(clip_path, sr=args.sr)
            rec.update(anchors)
        except Exception as e:  # noqa: BLE001 — per-clip isolation; errors are logged
            err = f"{type(e).__name__}: {e}"
            n_errors += 1
        records.append(rec)
        rec_json = {
            "key": rec["key"], "path": rec["path"], "ucs": rec["ucs"],
            "audio": _anchor_to_json(rec["audio"]),
            "visual": _anchor_to_json(rec["visual"]),
            "sigma_s": _json_safe(float(rec["sigma_s"])),
        }
        if err is not None:
            rec_json["error"] = err
        records_json.append(rec_json)
        if (i + 1) % 25 == 0 or (i + 1) == len(rows):
            print(f"[stage0-anchors] {i + 1}/{len(rows)} clips processed", flush=True)

    summary = summarize_sigma(records)

    # -- anchors.json
    args.out_dir.mkdir(parents=True, exist_ok=True)
    anchors_path = args.out_dir / "anchors.json"
    payload = {
        "meta": {
            "generated_by": "scripts/stage0_anchors.py",
            "index": str(args.index),
            "n_index_rows": n_total,
            "n_ok": n_ok,
            "n_processed": len(records),
            "n_errors": n_errors,
            "sr": args.sr,
            "seed": args.seed,
            "manual_section": "3.2 (Phase 0.4 — event anchors)",
        },
        "summary": {k: _json_safe(v) for k, v in summary.items()},
        "clips": records_json,
    }
    anchors_path.write_text(json.dumps(payload, indent=2))
    print(f"[stage0-anchors] wrote {anchors_path}")

    # -- anchor_report.md (audio-source anchors + sigma summary + amendment note)
    audio_anchors = {r["key"]: r["audio"] for r in records if r["audio"] is not None}
    report_path = args.out_dir / "anchor_report.md"
    report_path.write_text(render_report(audio_anchors, summary, n_errors))
    print(f"[stage0-anchors] wrote {report_path}")

    # -- anchor_check_30.csv (stratified by ucs; human column empty)
    picked = stratified_check_set(records, n_check=args.n_check, seed=args.seed)
    args.check_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.check_csv, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["key", "path", "proposed_onset_s", "human_onset_s", "notes"])
        for cand in picked:
            writer.writerow(
                [cand["key"], cand["path"], f"{cand['proposed_onset_s']:.3f}", "", ""]
            )
    print(f"[stage0-anchors] wrote {args.check_csv} ({len(picked)} clips)")
    if len(picked) < args.n_check:
        print(f"[stage0-anchors] WARNING: only {len(picked)}/{args.n_check} "
              "checkable clips available for the human check set.")

    print(f"[stage0-anchors] sigma summary: "
          f"median={_fmt(summary['median_sigma_s'])}s "
          f"coverage_both={_fmt(summary['coverage_both'])} "
          f"recommended_bin={_fmt(summary['recommended_bin_s'])}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
