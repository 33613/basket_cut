"""Build a KPR identity archive from the command line."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from contracts.paths import KPR_FILENAME, model_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract KPR track prototypes and identity archives"
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--tracks", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--kpr-root", type=Path, default=Path("KPR"))
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/kpr/multidataset_sports_test.yaml"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=model_path("kpr", KPR_FILENAME),
    )
    parser.add_argument("--prompt-mode", choices=("none", "keypoints"), default="none")
    parser.add_argument("--keypoints", type=Path)
    parser.add_argument("--samples-per-track", type=int, default=8)
    parser.add_argument("--sample-min-gap-s", type=float, default=0.25)
    parser.add_argument("--archive-exemplars", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--min-det-score", type=float, default=0.5)
    parser.add_argument("--min-box-width", type=float, default=16.0)
    parser.add_argument("--min-box-height", type=float, default=40.0)
    parser.add_argument("--crop-padding", type=float, default=0.05)
    parser.add_argument("--context-padding", type=float, default=0.5)
    parser.add_argument("--max-gap-frames", type=int, default=90)
    parser.add_argument("--max-overlap-frames", type=int, default=0)
    parser.add_argument("--candidate-threshold", type=float)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    from pipeline.identity.archive import IdentityArchiveOptions
    from workflows.identity import run_identity_archive

    result = run_identity_archive(
        IdentityArchiveOptions(
            input=args.input,
            tracks=args.tracks,
            output_dir=args.output_dir,
            kpr_root=args.kpr_root,
            config=args.config,
            checkpoint=args.checkpoint,
            prompt_mode=args.prompt_mode,
            keypoints=args.keypoints,
            samples_per_track=args.samples_per_track,
            sample_min_gap_s=args.sample_min_gap_s,
            archive_exemplars=args.archive_exemplars,
            batch_size=args.batch_size,
            min_det_score=args.min_det_score,
            min_box_width=args.min_box_width,
            min_box_height=args.min_box_height,
            crop_padding=args.crop_padding,
            context_padding=args.context_padding,
            max_gap_frames=args.max_gap_frames,
            max_overlap_frames=args.max_overlap_frames,
            candidate_threshold=args.candidate_threshold,
            cpu=args.cpu,
            prepare_only=args.prepare_only,
            overwrite=args.overwrite,
        )
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
