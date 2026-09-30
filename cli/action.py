"""Run the action-recognition workflow from the command line."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply MultiSports SlowFast to MOTIP person tracks"
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--tracks", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--label-map", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--short-side", type=int, default=256)
    parser.add_argument("--predict-stepsize", type=int, default=8)
    parser.add_argument("--proposal-max-gap", type=int, default=2)
    parser.add_argument("--min-det-score", type=float, default=0.3)
    parser.add_argument("--action-threshold", type=float, default=0.2)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--label-prefix", default="basketball_")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    from pipeline.action.service import ActionOptions
    from workflows.action import run_action_recognition

    result = run_action_recognition(
        ActionOptions(
            input=args.input,
            tracks=args.tracks,
            config=args.config,
            checkpoint=args.checkpoint,
            label_map=args.label_map,
            output=args.output,
            device=args.device,
            short_side=args.short_side,
            predict_stepsize=args.predict_stepsize,
            proposal_max_gap=args.proposal_max_gap,
            min_det_score=args.min_det_score,
            action_threshold=args.action_threshold,
            top_k=args.top_k,
            label_prefix=args.label_prefix,
            overwrite=args.overwrite,
        )
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
