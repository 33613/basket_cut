"""Render tracking, identity, and action artifacts onto a video."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render modular pipeline artifacts without rerunning models"
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--tracks", required=True, type=Path)
    parser.add_argument("--actions", type=Path)
    parser.add_argument("--identity-map", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-action-gap", type=int, default=4)
    parser.add_argument("--show-top-candidate", action="store_true")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    from analysis.visualization.render import RenderOptions, render_results

    result = render_results(
        RenderOptions(
            input=args.input,
            tracks=args.tracks,
            actions=args.actions,
            identity_map=args.identity_map,
            output=args.output,
            max_action_gap=args.max_action_gap,
            show_top_candidate=args.show_top_candidate,
            max_frames=args.max_frames,
            overwrite=args.overwrite,
        )
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
