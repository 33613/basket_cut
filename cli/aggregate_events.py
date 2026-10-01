"""Aggregate sparse MMAction2 action points into temporal events."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert MMAction2 per-person sampled predictions into "
            "[id, event, start, end, raw_score] temporal events"
        )
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--video-meta", type=Path)
    parser.add_argument("--fps", type=float)
    parser.add_argument("--cadence-frames", type=int)
    parser.add_argument("--score-threshold", type=float, default=0.2)
    parser.add_argument("--max-missing-steps", type=int, default=1)
    parser.add_argument("--min-support", type=int, default=1)
    parser.add_argument("--min-duration-s", type=float, default=0.0)
    parser.add_argument(
        "--score-reducer", choices=("mean", "median", "max"), default="mean"
    )
    parser.add_argument(
        "--require-person-id",
        action="store_true",
        help="Fail instead of falling back to the raw MOTIP track ID",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    from adapters.mmaction2.temporal_events import TemporalAggregationOptions
    from workflows.events import build_temporal_events

    result = build_temporal_events(
        TemporalAggregationOptions(
            input=args.input,
            output=args.output,
            video_meta=args.video_meta,
            fps=args.fps,
            cadence_frames=args.cadence_frames,
            score_threshold=args.score_threshold,
            max_missing_steps=args.max_missing_steps,
            min_support=args.min_support,
            min_duration_s=args.min_duration_s,
            score_reducer=args.score_reducer,
            allow_track_id_fallback=not args.require_person_id,
            overwrite=args.overwrite,
        )
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
