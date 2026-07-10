#!/usr/bin/env python
"""FoleyBench extraction CLI (manual §2 Stage-M clip source + §3.1 candidate pool).

Drives foley_cw.foleybench_extract over the local FoleyBench snapshot:

  extract   decode parquet-embedded MP4s to data/FoleyBench/clips/<key>.mp4
            (atomic, idempotent index CSV)
  validate  PyAV second pass; HALTS with exit 2 if ok_rate < --min-ok-rate
  select    write Stage-M / exclusion / screening manifests under data/manifests
  all       extract -> validate (gated) -> select

Selection manifests:
  stage_m_clips.json        {clips, seed, n, pool: 'single_source_discrete'}
  stage_m_exclusions.json   {excluded_from_phase1_manifest, reason}
  screening_manifest.json   {clips, seed, n, exclusions_applied: true}

NO download, NO network: everything reads the already-present local snapshot.
Stage-M clips are diagnostics only and are excluded from the Phase-1 manifest
(manual §3.1: Stage-M clips are not carried forward).

Usage (from the repo root):
    .venv/bin/python scripts/extract_foleybench.py all
    .venv/bin/python scripts/extract_foleybench.py select --stage-m-n 16 --screening-n 400
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Make foley_cw importable when run from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_PARQUET_DIR = _REPO_ROOT / "data" / "FoleyBench" / "foleybench" / "data"
_DEFAULT_CSV = _REPO_ROOT / "data" / "FoleyBench" / "foleybench" / "foleybench.csv"
_DEFAULT_OUT_DIR = _REPO_ROOT / "data" / "FoleyBench" / "clips"
_DEFAULT_INDEX_CSV = _REPO_ROOT / "data" / "FoleyBench" / "clips_index.csv"
_DEFAULT_MANIFEST_DIR = _REPO_ROOT / "data" / "manifests"


def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--parquet-dir", type=Path, default=_DEFAULT_PARQUET_DIR,
                   help="FoleyBench parquet shard dir (data/train-*.parquet)")
    p.add_argument("--csv", type=Path, default=_DEFAULT_CSV,
                   help="FoleyBench metadata CSV (foleybench.csv)")
    p.add_argument("--out-dir", type=Path, default=_DEFAULT_OUT_DIR,
                   help="output dir for extracted <key>.mp4 clips")
    p.add_argument("--index-csv", type=Path, default=_DEFAULT_INDEX_CSV,
                   help="extraction index CSV (rewritten idempotently)")
    p.add_argument("--manifest-dir", type=Path, default=_DEFAULT_MANIFEST_DIR,
                   help="output dir for selection manifests")
    p.add_argument("--seed", type=int, default=0, help="selection RNG seed")
    p.add_argument("--stage-m-n", type=int, default=16, help="Stage-M clip count")
    p.add_argument("--screening-n", type=int, default=400,
                   help="screening candidate pool size (manual §3.1)")
    p.add_argument("--min-ok-rate", type=float, default=0.95,
                   help="halt threshold for the PyAV validation pass")
    p.add_argument("--include-non-discrete", action="store_true",
                   help="extract ALL rows, not just discrete_vs_rest == 'Discrete'")
    p.add_argument("--keys", type=Path, default=None,
                   help="optional newline-separated file of keys to restrict extraction to")


def _load_keys(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    keys = {line.strip() for line in path.read_text(encoding="utf-8").splitlines()}
    return {k for k in keys if k} or None


def run_extract(args: argparse.Namespace) -> int:
    from foley_cw.foleybench_extract import extract_clips

    report = extract_clips(
        parquet_dir=args.parquet_dir,
        csv_path=args.csv,
        out_dir=args.out_dir,
        index_csv=args.index_csv,
        only_discrete=not args.include_non_discrete,
        keys=_load_keys(args.keys),
    )
    print(
        f"[extract] seen={report.n_seen} extracted={report.n_extracted} "
        f"skipped_existing={report.n_skipped_existing} "
        f"decode_error={report.n_decode_error} filtered_out={report.n_filtered_out}",
        flush=True,
    )
    print(f"[extract] index: {args.index_csv}")
    return 0


def run_validate(args: argparse.Namespace) -> int:
    from foley_cw.foleybench_extract import validate_clips_av

    stats = validate_clips_av(args.index_csv)
    print(
        f"[validate] n_ok={stats['n_ok']} n_failed={stats['n_failed']} "
        f"ok_rate={stats['ok_rate']:.4f}",
        flush=True,
    )
    if stats["ok_rate"] < args.min_ok_rate:
        print(
            f"[validate] HALT: ok_rate {stats['ok_rate']:.4f} < required "
            f"{args.min_ok_rate:.4f} — the extracted pool is unhealthy; fix the "
            "snapshot/extraction before selection (failures are marked "
            "decode_error in the index).",
            file=sys.stderr,
        )
        return 2
    return 0


def run_select(args: argparse.Namespace) -> int:
    from foley_cw.foleybench_extract import (
        select_screening_pool,
        select_stage_m_clips,
        write_manifest_json,
    )

    stage_m = select_stage_m_clips(args.index_csv, n=args.stage_m_n, seed=args.seed)
    if len(stage_m) < args.stage_m_n:
        print(
            f"[select] WARNING: Stage-M pool yielded {len(stage_m)} < requested "
            f"{args.stage_m_n} clips",
            file=sys.stderr,
        )
    stage_m_path = args.manifest_dir / "stage_m_clips.json"
    write_manifest_json(stage_m_path, {
        "clips": stage_m,
        "seed": args.seed,
        "n": len(stage_m),
        "pool": "single_source_discrete",
    })

    exclusions_path = args.manifest_dir / "stage_m_exclusions.json"
    write_manifest_json(exclusions_path, {
        "excluded_from_phase1_manifest": stage_m,
        "reason": (
            "used in Stage M (manual §3.1: Stage-M clips are not carried forward)"
        ),
    })

    screening = select_screening_pool(
        args.index_csv, n=args.screening_n, exclude=set(stage_m), seed=args.seed,
    )
    if len(screening) < args.screening_n:
        print(
            f"[select] WARNING: screening pool yielded {len(screening)} < requested "
            f"{args.screening_n} clips",
            file=sys.stderr,
        )
    screening_path = args.manifest_dir / "screening_manifest.json"
    write_manifest_json(screening_path, {
        "clips": screening,
        "seed": args.seed,
        "n": len(screening),
        "exclusions_applied": True,
    })

    overlap = set(stage_m) & set(screening)
    if overlap:
        # Defensive: would indicate a selection bug; never ship such manifests.
        print(f"[select] ERROR: Stage-M/screening overlap: {sorted(overlap)}",
              file=sys.stderr)
        return 1

    for p in (stage_m_path, exclusions_path, screening_path):
        print(f"[select] wrote {p}")
    print(json.dumps({"stage_m_n": len(stage_m), "screening_n": len(screening),
                      "seed": args.seed}))
    return 0


def run_all(args: argparse.Namespace) -> int:
    rc = run_extract(args)
    if rc != 0:
        return rc
    rc = run_validate(args)
    if rc != 0:
        return rc
    return run_select(args)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="FoleyBench clip extraction / validation / selection",
    )
    sub = ap.add_subparsers(dest="command", required=True)
    for name, fn, doc in (
        ("extract", run_extract, "extract clips from parquet shards"),
        ("validate", run_validate, "PyAV second pass (exit 2 if ok_rate < threshold)"),
        ("select", run_select, "write Stage-M / exclusion / screening manifests"),
        ("all", run_all, "extract, validate (gated), then select"),
    ):
        p = sub.add_parser(name, help=doc)
        _add_common_args(p)
        p.set_defaults(fn=fn)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
