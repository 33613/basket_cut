"""Run event evaluation from a fixed video manifest, no GPU required."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from analysis.evaluation.batch import BatchEventEvaluationOptions, evaluate_event_batch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--profile", choices=("generic", "multisports"), default="multisports"
    )
    parser.add_argument(
        "--actor-mode", choices=("tube", "exact", "ignore"), default="tube"
    )
    parser.add_argument("--temporal-iou-threshold", type=float, default=0.5)
    parser.add_argument("--actor-iou-threshold", type=float, default=0.5)
    parser.add_argument("--actor-min-coverage", type=float, default=0.8)
    parser.add_argument("--min-raw-score", type=float, default=0.0)
    parser.add_argument("--exclude-label", action="append", default=[])
    args = parser.parse_args()
    result = evaluate_event_batch(
        BatchEventEvaluationOptions(
            manifest=args.manifest,
            output_dir=args.output_dir,
            profile=args.profile,
            actor_mode=args.actor_mode,
            temporal_iou_threshold=args.temporal_iou_threshold,
            actor_iou_threshold=args.actor_iou_threshold,
            actor_min_coverage=args.actor_min_coverage,
            min_raw_score=args.min_raw_score,
            exclude_labels=tuple(args.exclude_label),
        )
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
