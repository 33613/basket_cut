"""Verify and inspect the downloaded KPR checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.models.kpr_checkpoint import DEFAULT_CHECKPOINT, PUBLISHED_SHA256


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--expected-sha256", default=PUBLISHED_SHA256)
    parser.add_argument("--skip-tensor-scan", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    from tools.models.kpr_checkpoint import (
        KPRCheckpointOptions,
        inspect_kpr_checkpoint,
    )

    result = inspect_kpr_checkpoint(
        KPRCheckpointOptions(
            checkpoint=args.checkpoint,
            expected_sha256=args.expected_sha256,
            skip_tensor_scan=args.skip_tensor_scan,
            output=args.output,
        )
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
