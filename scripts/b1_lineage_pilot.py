#!/usr/bin/env python
"""CLI for the five-video B-1 same-forward identity pilot.

No command downloads assets.  Model/checkpoint/video roots are explicit so a
clean execution worktree cannot silently fall back to another checkout.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from foley_cw.b1_lineage import (  # noqa: E402
    calibrate_attempt,
    create_selection_attempt,
    heldout_attempt,
    make_packet_attempt,
    replay_attempt,
    validate_attempt,
)


def _assets(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mmaudio-root", type=Path, required=True)
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--clips-root", type=Path, required=True)


def _destination(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--attempt-id", required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)

    select = sub.add_parser("select", help="freeze deterministic 4+1 pilot selection")
    _assets(select)
    _destination(select)

    packets = sub.add_parser("make-packets", help="make 40 canonical replay packets")
    _assets(packets)
    _destination(packets)
    packets.add_argument("--selection-attempt", type=Path, required=True)
    packets.add_argument("--device", required=True,
                         help="explicit torch device, e.g. cuda:0")

    replay = sub.add_parser("replay", help="run same-forward capture for one pilot role")
    _assets(replay)
    _destination(replay)
    replay.add_argument("--packet-attempt", type=Path, required=True)
    replay.add_argument("--role", choices=("calibration", "heldout"), required=True)
    replay.add_argument("--device", required=True)
    replay.add_argument("--repeats", type=int, default=1)
    replay.add_argument("--repeat-offset", type=int, default=0)

    calibrate = sub.add_parser("calibrate", help="freeze q0.999(higher)*2 tolerance")
    _destination(calibrate)
    calibrate.add_argument("--replay-attempt", type=Path, required=True)

    heldout = sub.add_parser("heldout", help="apply frozen tolerance to clip 1002")
    _destination(heldout)
    heldout.add_argument("--replay-attempt", type=Path, required=True)
    heldout.add_argument("--tolerance-file", type=Path, required=True)
    heldout.add_argument("--tolerance-sha256",
                         help="optional externally recorded expected tolerance hash")

    validate = sub.add_parser("validate", help="validate immutable attempt recursively")
    validate.add_argument("--attempt-root", type=Path, required=True)
    validate.add_argument("--expected-stage",
                          choices=("selection", "packets", "replay", "calibration", "heldout"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.stage == "select":
        result = create_selection_attempt(
            args.output_root, args.attempt_id, mmaudio_root=args.mmaudio_root,
            weights_dir=args.weights_dir, clips_root=args.clips_root,
        )
    elif args.stage == "make-packets":
        result = make_packet_attempt(
            args.selection_attempt, args.output_root, args.attempt_id,
            mmaudio_root=args.mmaudio_root, weights_dir=args.weights_dir,
            clips_root=args.clips_root, device=args.device,
        )
    elif args.stage == "replay":
        result = replay_attempt(
            args.packet_attempt, args.output_root, args.attempt_id, role=args.role,
            mmaudio_root=args.mmaudio_root, weights_dir=args.weights_dir,
            clips_root=args.clips_root, device=args.device, repeats=args.repeats,
            repeat_offset=args.repeat_offset,
        )
    elif args.stage == "calibrate":
        result = calibrate_attempt(
            args.replay_attempt, args.output_root, args.attempt_id,
        )
    elif args.stage == "heldout":
        result = heldout_attempt(
            args.replay_attempt, args.tolerance_file, args.output_root, args.attempt_id,
            expected_tolerance_sha256=args.tolerance_sha256,
        )
    else:
        summary = validate_attempt(args.attempt_root, expected_stage=args.expected_stage)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    print(str(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
