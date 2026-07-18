#!/usr/bin/env python
"""Prepare, fit, validate, and merge nested-CV B2 Class readout shards."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from foley_cw.class_internal_readout import (  # noqa: E402
    fit_progress_shard,
    merge_readout_shards,
    prepare_targets,
    validate_readout_shard,
    validate_targets,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare-targets")
    prepare.add_argument("--class-completion", type=Path, required=True)
    prepare.add_argument("--feature-completion", type=Path, required=True)
    prepare.add_argument("--protocol", type=Path, required=True)
    prepare.add_argument("--implementation", type=Path, required=True)
    prepare.add_argument("--out-dir", type=Path, required=True)

    fit = sub.add_parser("fit-progress")
    fit.add_argument("--feature-completion", type=Path, required=True)
    fit.add_argument("--target-completion", type=Path, required=True)
    fit.add_argument("--protocol", type=Path, required=True)
    fit.add_argument("--implementation", type=Path, required=True)
    fit.add_argument("--out-dir", type=Path, required=True)
    fit.add_argument("--progress", type=float, required=True)
    fit.add_argument("--device", default="cpu")

    merge = sub.add_parser("merge")
    merge.add_argument("--completion", type=Path, action="append", required=True)
    merge.add_argument("--out-dir", type=Path, required=True)
    merge.add_argument("--bootstrap-draws", type=int, default=5000)
    merge.add_argument("--bootstrap-seed", type=int, default=20260717)

    validate = sub.add_parser("validate")
    validate.add_argument("--completion", type=Path, required=True)
    validate.add_argument("--kind", choices=("targets", "shard"), required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare-targets":
        result = prepare_targets(
            args.class_completion,
            args.feature_completion,
            args.protocol,
            args.implementation,
            args.out_dir,
        )
        print(result)
    elif args.command == "fit-progress":
        result = fit_progress_shard(
            args.feature_completion,
            args.target_completion,
            args.protocol,
            args.implementation,
            args.out_dir,
            progress=args.progress,
            device=args.device,
        )
        print(result)
    elif args.command == "merge":
        result = merge_readout_shards(
            args.completion,
            args.out_dir,
            bootstrap_draws=args.bootstrap_draws,
            bootstrap_seed=args.bootstrap_seed,
        )
        print(result)
    else:
        if args.kind == "targets":
            completion, rows = validate_targets(args.completion)
        else:
            completion, rows = validate_readout_shard(args.completion)
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "schema": completion.get("schema"),
                    "rows": len(rows),
                    "completion": str(args.completion.resolve()),
                },
                indent=2,
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
