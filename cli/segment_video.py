"""Qwen3-VL coarse windows, boundary review and source-mapped video export."""

import argparse
import json
from pathlib import Path

from contracts.execution import ExecutionSettings, normalize_omp_threads
from pipeline.media import probe_video, preflight_video
from pipeline.segmentation.service import SegmentOptions, analyze_video, export_segments, plan_windows


def add_options(parser):
    parser.add_argument("--coarse-method", choices=("boundaries", "states"), default="boundaries")
    parser.add_argument("--video-tokens", type=int, default=2048)
    parser.add_argument("--window-s", type=float, default=20)
    parser.add_argument("--overlap-s", type=float, default=6)
    parser.add_argument("--review-radius-s", type=float, default=4)
    parser.add_argument("--coarse-frames", type=int, default=16)
    parser.add_argument("--review-frames", type=int, default=32)
    parser.add_argument("--min-clip-s", type=float, default=.5)
    parser.add_argument("--merge-tolerance-s", type=float, default=.75)
    parser.add_argument("--min-free-gb", type=float, default=2)
    parser.add_argument("--context", default="5v5 basketball; team identities unknown",
                        help="Basketball rules/team colors/camera context, up to 2000 characters")


def options_from(args):
    return SegmentOptions(**{key: getattr(args, key) for key in SegmentOptions.__dataclass_fields__})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--stage", choices=("analyze", "export", "all", "status"), default="all")
    parser.add_argument("--review-file", type=Path, help="Optional manually corrected contiguous source intervals")
    parser.add_argument("--dry-run", action="store_true")
    add_options(parser)
    args = parser.parse_args()
    if args.env_file:
        from cli.serve_web import load_environment
        load_environment(args.env_file)
    settings = ExecutionSettings()
    options = options_from(args)
    options.validate()
    if args.stage == "status":
        result = {}
        for name in ("status.json", "clips_manifest.json"):
            path = args.output_dir / name
            if path.is_file():
                value = json.loads(path.read_text())
                result[name] = ({k: v for k, v in value.items() if k != "videos"}
                                | {"clip_count": len(value["videos"])}) if "videos" in value else value
        print(json.dumps(result or {"status": "not_started"}, ensure_ascii=False, indent=2))
        return
    if args.stage in ("analyze", "all") and not args.input:
        parser.error("--input is required for analysis")
    if args.dry_run:
        meta = probe_video(args.input, ffprobe=settings.ffprobe) if args.input else None
        print(json.dumps({"dry_run": True, "stage": args.stage, "source": meta,
            "windows": plan_windows(meta["duration_s"], options) if meta else [],
            "model_dir": str(args.model_dir or settings.qwen_model_dir),
            "output_dir": str(args.output_dir.resolve())}, ensure_ascii=False, indent=2))
        return
    if args.stage in ("analyze", "all"):
        normalize_omp_threads()
        check = preflight_video(args.input, ffmpeg=settings.ffmpeg, ffprobe=settings.ffprobe)
        print("[segmentation] CPU media preflight passed: " + json.dumps(check), flush=True)
        from pipeline.identity.qwen_backend import QwenBackend, ensure_gpu_idle
        ensure_gpu_idle()
        backend = QwenBackend(args.model_dir or settings.qwen_model_dir)
        result = analyze_video(args.input, args.output_dir, backend, options=options,
                               ffmpeg=settings.ffmpeg, ffprobe=settings.ffprobe)
        print(json.dumps({"status": result["status"], "clip_count": len(result["segments"]),
                          "unresolved_count": len(result["unresolved"])}, ensure_ascii=False, indent=2))
    if args.stage in ("export", "all"):
        manifest = export_segments(args.output_dir, review_file=args.review_file,
            ffmpeg=settings.ffmpeg, ffprobe=settings.ffprobe, min_free_gb=args.min_free_gb)
        print(json.dumps({"status": manifest["status"], "clip_count": len(manifest["videos"]),
                          "clips_manifest": str(args.output_dir.resolve() / "clips_manifest.json")}, indent=2))


if __name__ == "__main__":
    main()
