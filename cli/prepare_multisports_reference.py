"""Export selected MultiSports GT tubes as canonical reference events."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert selected official MultiSports action tubes into canonical "
            "event reference JSONL"
        )
    )
    parser.add_argument("--annotation", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--video",
        action="append",
        help="Official MultiSports video key; repeat to select more than one",
    )
    parser.add_argument(
        "--list-videos",
        action="store_true",
        help="List basketball video keys without writing a reference file",
    )
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--label-prefix", default="basketball")
    parser.add_argument(
        "--split",
        choices=("all", "train", "validation"),
        default="all",
        help=(
            "Video split used by --list-videos. MultiSports calls the public "
            "validation list test_videos internally."
        ),
    )
    parser.add_argument(
        "--include-unevaluated-labels",
        action="store_true",
        help=(
            "Also export basketball_save and basketball_jump_ball, which the "
            "official MultiSports evaluator excludes"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    from tools.datasets.multisports import (
        MultiSportsReferenceOptions,
        list_multisports_videos,
        prepare_multisports_reference,
    )

    if args.list_videos:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        videos = list_multisports_videos(
            args.annotation,
            label_prefix=args.label_prefix,
            split=args.split,
        )
        print(
            json.dumps(
                {
                    "split": args.split,
                    "video_count": len(videos),
                    "videos": videos[: args.limit],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if args.output is None or not args.video:
        raise ValueError(
            "Export requires --output and at least one --video; use "
            "--list-videos to inspect valid keys"
        )

    result = prepare_multisports_reference(
        MultiSportsReferenceOptions(
            annotation=args.annotation,
            output=args.output,
            videos=tuple(args.video),
            fps=args.fps,
            label_prefix=args.label_prefix,
            include_unevaluated_labels=args.include_unevaluated_labels,
            overwrite=args.overwrite,
        )
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
