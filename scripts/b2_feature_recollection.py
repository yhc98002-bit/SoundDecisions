#!/usr/bin/env python3
"""Collect, validate, and reduce lineage-valid B2 internal features."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from foley_cw.b2_feature_recollection import (  # noqa: E402
    collect_feature_shard,
    merge_feature_shards,
    validate_feature_shard,
)


def _shard(value: str) -> tuple[int, int]:
    try:
        index, count = (int(part) for part in value.split("/"))
    except Exception as exc:
        raise argparse.ArgumentTypeError("shard must be INDEX/COUNT") from exc
    if count < 1 or not 0 <= index < count:
        raise argparse.ArgumentTypeError("shard must satisfy 0 <= INDEX < COUNT")
    return index, count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    collect = sub.add_parser("collect")
    collect.add_argument("--inventory-manifest", type=Path, required=True)
    collect.add_argument("--heldout-attempt", type=Path, required=True)
    collect.add_argument("--output-root", type=Path, required=True)
    collect.add_argument("--attempt-id", required=True)
    collect.add_argument("--shard", type=_shard, required=True)
    collect.add_argument("--mmaudio-root", type=Path, required=True)
    collect.add_argument("--weights-dir", type=Path, required=True)
    collect.add_argument("--clips-root", type=Path, required=True)
    collect.add_argument("--device", required=True)
    collect.add_argument("--protocol", type=Path, required=True)
    collect.add_argument("--protocol-sha256", required=True)

    validate = sub.add_parser("validate-shard")
    validate.add_argument("--completion", type=Path, required=True)
    validate.add_argument("--deep", action="store_true")

    merge = sub.add_parser("merge")
    merge.add_argument("--completion", type=Path, action="append", required=True)
    merge.add_argument("--out-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "collect":
        index, count = args.shard
        result = collect_feature_shard(
            args.inventory_manifest,
            args.heldout_attempt,
            args.output_root,
            args.attempt_id,
            shard_index=index,
            shard_count=count,
            mmaudio_root=args.mmaudio_root,
            weights_dir=args.weights_dir,
            clips_root=args.clips_root,
            device=args.device,
            protocol_path=args.protocol,
            protocol_sha256=args.protocol_sha256,
        )
        payload = {"status": "COMPLETE", "root": str(result)}
    elif args.command == "validate-shard":
        completion, manifests = validate_feature_shard(args.completion, deep=args.deep)
        payload = {
            "status": "VALID",
            "shard_index": completion["shard_index"],
            "unit_count": len(manifests),
            "completion": str(args.completion.resolve()),
        }
    else:
        result = merge_feature_shards(args.completion, args.out_dir)
        payload = {"status": "COMPLETE", "root": str(result)}
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
