"""Run the tracking workflow from the command line."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export MOTIP person tracks from one video"
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--motip-root", type=Path, default=REPOSITORY_ROOT / "MOTIP")
    parser.add_argument(
        "--config", default="configs/r50_deformable_detr_motip_sportsmot.yaml"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--assignment-protocol", default="object-max")
    parser.add_argument("--miss-tolerance", type=int, default=60)
    parser.add_argument("--det-thresh", type=float, default=0.3)
    parser.add_argument("--newborn-thresh", type=float, default=0.6)
    parser.add_argument("--id-thresh", type=float, default=0.2)
    parser.add_argument("--area-thresh", type=int, default=0)
    parser.add_argument("--max-shorter", type=int, default=800)
    parser.add_argument("--max-longer", type=int, default=1440)
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--no-visualization", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    from pipeline.tracking.service import TrackingOptions
    from workflows.tracking import run_tracking

    result = run_tracking(
        TrackingOptions(
            input=args.input,
            checkpoint=args.checkpoint,
            output_dir=args.output_dir,
            motip_root=args.motip_root,
            config=args.config,
            device=args.device,
            assignment_protocol=args.assignment_protocol,
            miss_tolerance=args.miss_tolerance,
            det_thresh=args.det_thresh,
            newborn_thresh=args.newborn_thresh,
            id_thresh=args.id_thresh,
            area_thresh=args.area_thresh,
            max_shorter=args.max_shorter,
            max_longer=args.max_longer,
            fp32=args.fp32,
            max_frames=args.max_frames,
            overwrite=args.overwrite,
        ),
        create_visualization=not args.no_visualization,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
