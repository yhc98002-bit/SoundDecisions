#!/usr/bin/env python3
"""Inventory, measure, reduce, and analyse the immutable Arc-4 B2 Class bank.

No subcommand generates or replays audio.  Measurement workers must each use a
different output directory; every scientific output is create-only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from foley_cw.b2_class_closure import (  # noqa: E402
    DEFAULT_BOOTSTRAP_DRAWS,
    DEFAULT_BOOTSTRAP_SEED,
    PannsBatchPredictor,
    SENSITIVITY_THRESHOLDS,
    load_merged_posteriors,
    measure_inventory_shard,
    merge_posterior_shards,
    parse_shard,
    runtime_provenance,
    sha256_file,
    validate_shard_completion,
    validate_class_protocol,
    write_inventory,
    write_multiseed_analysis,
)


def _thresholds(value: str) -> tuple[float, ...]:
    try:
        thresholds = tuple(float(part) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("thresholds must be comma-separated numbers") from exc
    if not thresholds or any(not 0.0 < threshold < 1.0 for threshold in thresholds):
        raise argparse.ArgumentTypeError("thresholds must lie strictly inside (0,1)")
    if tuple(sorted(set(thresholds))) != thresholds:
        raise argparse.ArgumentTypeError("thresholds must be unique and increasing")
    return thresholds


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    inventory = subparsers.add_parser(
        "inventory", help="read-only validation and inventory of four B2 roots"
    )
    inventory.add_argument("--root", type=Path, action="append", required=True)
    inventory.add_argument("--out-dir", type=Path, required=True)
    inventory.add_argument(
        "--noncanonical",
        action="store_true",
        help="allow a synthetic/noncanonical design (never use for the scientific B2 bank)",
    )
    inventory.add_argument(
        "--skip-wav-header-validation",
        action="store_true",
        help="test-only escape hatch; hashes and byte counts are still validated",
    )

    measure = subparsers.add_parser("measure", help="measure one immutable inventory shard")
    measure.add_argument("--inventory-manifest", type=Path, required=True)
    measure.add_argument("--protocol", type=Path, required=True)
    measure.add_argument("--out-dir", type=Path, required=True)
    measure.add_argument("--shard", required=True, help="zero-based INDEX/COUNT")
    measure.add_argument("--device", default="cuda:0")
    measure.add_argument("--batch-size", type=int, default=8)
    measure.add_argument("--checkpoint", type=Path, required=True)
    measure.add_argument("--abstain-delta", type=float, default=0.05)
    measure.add_argument(
        "--noncanonical",
        action="store_true",
        help="explicit synthetic-test mode; canonical pinned-asset checks are disabled",
    )
    measure.add_argument(
        "--coarse-map", type=Path, default=ROOT / "configs" / "coarse_class_map.json"
    )

    validate = subparsers.add_parser("validate-shard", help="fail-closed shard validation")
    validate.add_argument("--completion", type=Path, required=True)
    validate.add_argument("--inventory-manifest", type=Path)

    merge = subparsers.add_parser("merge", help="validate and merge a complete shard set")
    merge.add_argument("--inventory-manifest", type=Path, required=True)
    merge.add_argument("--completion", type=Path, action="append", required=True)
    merge.add_argument("--out-dir", type=Path, required=True)

    validate_merge = subparsers.add_parser(
        "validate-merge", help="validate a merged posterior completion"
    )
    validate_merge.add_argument("--completion", type=Path, required=True)

    analyze = subparsers.add_parser("analyze", help="multi-seed continuity analysis")
    analyze.add_argument("--merged-completion", type=Path, required=True)
    analyze.add_argument("--protocol", type=Path, required=True)
    analyze.add_argument("--out-dir", type=Path, required=True)
    analyze.add_argument(
        "--thresholds",
        type=_thresholds,
        default=SENSITIVITY_THRESHOLDS,
        help="predeclared increasing threshold grid (default: 0.50,...,0.90)",
    )
    analyze.add_argument(
        "--video-bootstrap-draws", type=int, default=DEFAULT_BOOTSTRAP_DRAWS
    )
    analyze.add_argument(
        "--fork-bootstrap-draws", type=int, default=DEFAULT_BOOTSTRAP_DRAWS
    )
    analyze.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    analyze.add_argument("--historical-json", type=Path, action="append", default=[])
    analyze.add_argument(
        "--noncanonical",
        action="store_true",
        help="explicit synthetic-test mode; production cardinality/provenance gates are disabled",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "inventory":
        result = write_inventory(
            args.root,
            args.out_dir,
            canonical=not args.noncanonical,
            verify_wav_headers=not args.skip_wav_header_validation,
        )
    elif args.command == "measure":
        shard_index, shard_count = parse_shard(args.shard)
        # This hash/pin gate deliberately precedes PANNs construction and any
        # output creation.
        validate_class_protocol(
            args.protocol,
            checkpoint_path=args.checkpoint,
            coarse_map_path=args.coarse_map,
            abstain_delta=args.abstain_delta,
            canonical=not args.noncanonical,
        )
        predictor = PannsBatchPredictor(args.checkpoint, args.device)
        provenance = runtime_provenance(ROOT, command=sys.argv, device=args.device)
        provenance.update(
            {
                "b2_class_module_sha256": sha256_file(
                    ROOT / "foley_cw" / "b2_class_closure.py"
                ),
                "real_measurer_module_sha256": sha256_file(
                    ROOT / "foley_cw" / "real_measurer.py"
                ),
                "panns_module_sha256": sha256_file(
                    ROOT / "foley_cw" / "measurers_panns_cnn14.py"
                ),
                "network_downloads": "forbidden; local checkpoint only",
            }
        )
        result = measure_inventory_shard(
            args.inventory_manifest,
            args.out_dir,
            protocol_path=args.protocol,
            canonical=not args.noncanonical,
            shard_index=shard_index,
            shard_count=shard_count,
            coarse_map_path=args.coarse_map,
            posterior_fn=predictor,
            batch_size=args.batch_size,
            tagger_revision=predictor.revision,
            tagger_checkpoint_sha256=predictor.checkpoint_sha256,
            measurer_revision=provenance["git_commit"],
            abstain_delta=args.abstain_delta,
            provenance=provenance,
        )
    elif args.command == "validate-shard":
        completion, arrays = validate_shard_completion(
            args.completion, inventory_manifest_path=args.inventory_manifest
        )
        result = {
            "status": "VALID",
            "completion": str(args.completion.resolve()),
            "completion_sha256": sha256_file(args.completion),
            "record_count": int(arrays["record_id"].size),
            "data_sha256": completion["data_sha256"],
        }
    elif args.command == "merge":
        result = merge_posterior_shards(
            args.inventory_manifest, args.completion, args.out_dir
        )
    elif args.command == "validate-merge":
        arrays, completion = load_merged_posteriors(args.completion)
        result = {
            "status": "VALID",
            "completion": str(args.completion.resolve()),
            "completion_sha256": sha256_file(args.completion),
            "record_count": int(arrays["record_id"].size),
            "data_sha256": completion["data_sha256"],
        }
    elif args.command == "analyze":
        if args.video_bootstrap_draws <= 0 or args.fork_bootstrap_draws <= 1:
            raise ValueError("bootstrap draw counts must be positive (fork > 1)")
        result = write_multiseed_analysis(
            args.merged_completion,
            args.out_dir,
            protocol_path=args.protocol,
            canonical=not args.noncanonical,
            thresholds=args.thresholds,
            n_video_boot=args.video_bootstrap_draws,
            n_fork_boot=args.fork_bootstrap_draws,
            seed=args.bootstrap_seed,
            historical_jsons=args.historical_json,
        )
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
