"""Selectively download SHOT data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.datasets.shot import DEFAULT_REPO_ID, DEFAULT_REVISION, DEFAULT_SAMPLE


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Selectively download SHOT basketball samples"
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/root/autodl-tmp/data/basket_cut/SHOT"),
    )
    parser.add_argument(
        "--sample",
        action="append",
        help=f"Repeat for multiple samples; default: {DEFAULT_SAMPLE}",
    )
    parser.add_argument("--max-samples", type=int)
    parser.add_argument(
        "--profile", choices=("video", "tracking-eval", "full"), default="tracking-eval"
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    from tools.datasets.shot import ShotDownloadOptions, download_shot

    result = download_shot(
        ShotDownloadOptions(
            repo_id=args.repo_id,
            revision=args.revision,
            output_dir=args.output_dir,
            samples=tuple(args.sample or ()),
            max_samples=args.max_samples,
            profile=args.profile,
            dry_run=args.dry_run,
        )
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
