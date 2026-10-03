"""Evaluate canonical temporal events with a product-facing scorecard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate event class, temporal interval and actor with one "
            "headline usable-event F1 score"
        )
    )
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--prediction", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--errors-output", type=Path)
    parser.add_argument("--tracks", type=Path)
    parser.add_argument("--temporal-iou-threshold", type=float, default=0.5)
    parser.add_argument(
        "--actor-mode",
        choices=("auto", "exact", "tube", "ignore"),
        default="auto",
    )
    parser.add_argument("--actor-iou-threshold", type=float, default=0.5)
    parser.add_argument("--actor-min-coverage", type=float, default=0.8)
    parser.add_argument("--min-raw-score", type=float, default=0.0)
    parser.add_argument(
        "--profile", choices=("generic", "multisports"), default="generic"
    )
    parser.add_argument("--exclude-label", action="append", default=[])
    parser.add_argument("--reference-video-id")
    parser.add_argument("--prediction-video-id")
    parser.add_argument("--video-meta", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    from analysis.evaluation.events import (
        EventEvaluationOptions,
        evaluate_events,
    )

    result = evaluate_events(
        EventEvaluationOptions(
            reference=args.reference,
            prediction=args.prediction,
            output=args.output,
            errors_output=args.errors_output,
            tracks=args.tracks,
            temporal_iou_threshold=args.temporal_iou_threshold,
            actor_mode=args.actor_mode,
            actor_iou_threshold=args.actor_iou_threshold,
            min_raw_score=args.min_raw_score,
            actor_min_coverage=args.actor_min_coverage,
            profile=args.profile,
            exclude_labels=tuple(args.exclude_label),
            reference_video_id=args.reference_video_id,
            prediction_video_id=args.prediction_video_id,
            video_meta=args.video_meta,
        )
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
