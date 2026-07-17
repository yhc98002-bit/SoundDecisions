#!/usr/bin/env python
"""Freeze or reject the outcome-blind legacy Material 2AFC references.

This is a CPU-only metadata gate.  It may decode original FoleyBench MP4 audio
to derive source RMS dBFS, but it never replays a generated preview and never
loads a Material embedding, cosine, margin, or decision.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(_REPO_ROOT))

from foley_cw.material_reference_feasibility import (  # noqa: E402
    ADMISSIBLE_MATERIALS,
    build_reference_manifest,
    collect_source_loudness,
    inventory_phase2_journals,
    load_clip_metadata,
    load_primary_detector_timing,
    scan_retained_finals,
    sha256_file,
    validate_protocol,
    write_feasibility_outputs,
)


def _git_revision() -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return proc.stdout.strip() if proc.returncode == 0 else "UNKNOWN"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--measurements",
        type=Path,
        default=_REPO_ROOT / "results" / "stage0" / "measurements" / "measurements.jsonl",
        help="legacy measurement JSONL containing retained p1cfg1 finals",
    )
    parser.add_argument(
        "--phase2-journal-dir",
        type=Path,
        default=_REPO_ROOT / "results" / "stage0" / "journal",
    )
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=_REPO_ROOT / "data" / "FoleyBench" / "clips_index.csv",
    )
    parser.add_argument(
        "--anchors-json",
        type=Path,
        default=_REPO_ROOT / "results" / "stage0" / "anchors.json",
    )
    parser.add_argument(
        "--clips-root",
        type=Path,
        default=_REPO_ROOT / "data" / "FoleyBench" / "clips",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=_REPO_ROOT / "experiment" / "non_human_closure" / "PROTOCOL.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_REPO_ROOT / "results" / "non_human_closure" / "material_reference_feasibility",
    )
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument(
        "--loudness-workers",
        type=int,
        default=4,
        help="parallel original-MP4 decoders; no generated audio is touched",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
        validate_protocol(protocol)
        protocol_sha = sha256_file(args.protocol)

        # Outcome-blind inventories happen before source-audio covariate work.
        phase2 = inventory_phase2_journals(args.phase2_journal_dir)
        finals = scan_retained_finals(args.measurements)
        metadata, metadata_sha, metadata_errors = load_clip_metadata(args.metadata_csv)
        timings, anchors_sha, timing_errors = load_primary_detector_timing(args.anchors_json)

        # Only exact admissible UCS clips can ever enter a positive/negative
        # Material trial.  Captions are not loaded by the metadata reader.
        loudness_clips = [
            clip for clip in phase2.clips
            if metadata.get(clip, {}).get("ucs_category") in ADMISSIBLE_MATERIALS
        ]
        loudness = collect_source_loudness(
            loudness_clips,
            metadata,
            args.metadata_csv,
            args.clips_root,
            ffmpeg_binary=args.ffmpeg,
            workers=args.loudness_workers,
        )
        material_protocol = protocol["material_continuity"]
        source_provenance = {
            "git_revision": _git_revision(),
            "tool": str(Path(__file__).resolve()),
            "tool_sha256": sha256_file(Path(__file__).resolve()),
            "module": str((_REPO_ROOT / "foley_cw" / "material_reference_feasibility.py").resolve()),
            "module_sha256": sha256_file(
                _REPO_ROOT / "foley_cw" / "material_reference_feasibility.py"
            ),
            "measurements_path": str(args.measurements.resolve()),
            "phase2_journal_dir": str(args.phase2_journal_dir.resolve()),
            "phase2_outcome_blind_inventory_hashes": phase2.journal_hashes,
            "metadata_csv": str(args.metadata_csv.resolve()),
            "metadata_csv_sha256": metadata_sha,
            "metadata_input_errors": list(metadata_errors),
            "anchors_json": str(args.anchors_json.resolve()),
            "anchors_json_sha256": anchors_sha,
            "timing_input_errors": list(timing_errors),
            "clips_root": str(args.clips_root.resolve()),
            "loudness_command_contract": (
                "ffmpeg original MP4 audio -> mono 16 kHz pcm_f32le; RMS dBFS; full retained clip"
            ),
        }
        report, manifest = build_reference_manifest(
            phase2=phase2,
            finals=finals,
            metadata=metadata,
            timings=timings,
            loudness=loudness,
            source_provenance=source_provenance,
            protocol_sha256=protocol_sha,
            orientation_seed=int(material_protocol["orientation_seed"]),
        )
        outputs = write_feasibility_outputs(args.out_dir, report, manifest, loudness)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"[material-reference-feasibility] ENGINEERING ERROR: {exc}", file=sys.stderr)
        return 2

    print(json.dumps({
        "status": report["status"],
        "coverage": report["coverage"],
        "outputs": outputs,
    }, indent=2, sort_keys=True))
    # Exit 3 is an expected evidence-backed scientific artifact blocker, not a
    # script crash.  No manifest has been written in this case.
    return 0 if manifest is not None else 3


if __name__ == "__main__":
    raise SystemExit(main())
