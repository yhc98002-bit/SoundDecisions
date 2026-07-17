#!/usr/bin/env python3
"""Merge four immutable one-root B2 inventories without reopening WAVs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from foley_cw.b2_inventory_merge import merge_partial_inventories  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--part-manifest",
        type=Path,
        action="append",
        required=True,
        help="one B2_WAV_INVENTORY_MANIFEST.json; repeat exactly four times",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = merge_partial_inventories(args.part_manifest, args.out_dir)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
