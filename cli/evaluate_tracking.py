"""Evaluate tracking artifacts from the command line."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate MOTIP JSONL tracks against SHOT reference tracks"
    )
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--prediction", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--events-output", type=Path)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--reference-frame-base", type=int, default=1)
    parser.add_argument("--min-pred-score", type=float, default=0.0)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--video-meta", type=Path)
    parser.add_argument("--ignore-video-meta", action="store_true")
    parser.add_argument("--sequence-name")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    from analysis.evaluation.tracking import (
        TrackingEvaluationOptions,
        evaluate_tracks,
    )

    result = evaluate_tracks(
        TrackingEvaluationOptions(
            reference=args.reference,
            prediction=args.prediction,
            output=args.output,
            events_output=args.events_output,
            iou_threshold=args.iou_threshold,
            reference_frame_base=args.reference_frame_base,
            min_pred_score=args.min_pred_score,
            max_frames=args.max_frames,
            video_meta=args.video_meta,
            ignore_video_meta=args.ignore_video_meta,
            sequence_name=args.sequence_name,
        )
    )
    print(result.pop("rendered_summary"))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
