"""Diagnose a small interval using original windows, prompts and actual Qwen inputs."""

import argparse
import json
from pathlib import Path

from analysis.segmentation_diagnostic import diagnostic_plan, diagnose
from contracts.execution import ExecutionSettings, normalize_omp_threads
from pipeline.media import preflight_video


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segmentation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--start-s", type=float, required=True)
    parser.add_argument("--end-s", type=float, required=True)
    parser.add_argument("--frames", type=int, help="Optional sampling ablation; default preserves original frame count")
    parser.add_argument("--max-windows", type=int, default=8)
    parser.add_argument("--video-tokens", type=int,
                        help="Optional diagnostic total visual-token cap (256–8192)")
    parser.add_argument("--queries", nargs="+", choices=("description", "boundary"),
                        default=["description", "boundary"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.env_file:
        from cli.serve_web import load_environment
        load_environment(args.env_file)
    settings = ExecutionSettings()
    protocol, windows = diagnostic_plan(args.segmentation_dir, args.start_s, args.end_s)
    if not 1 <= len(windows) <= args.max_windows:
        parser.error(f"Selected {len(windows)} windows; choose a smaller interval or increase --max-windows")
    if args.frames is not None and (args.frames % 2 or not 4 <= args.frames <= 64):
        parser.error("--frames must be an even integer from 4 to 64")
    if len(set(args.queries)) != len(args.queries):
        parser.error("--queries must not contain duplicates")
    if args.video_tokens is not None and not 256 <= args.video_tokens <= 8192:
        parser.error("--video-tokens must be between 256 and 8192")
    print(json.dumps({"windows": windows, "model_queries": len(args.queries) * len(windows),
        "num_frames": args.frames or protocol["options"]["coarse_frames"],
        "queries": args.queries, "requested_video_tokens": args.video_tokens,
        "output_dir": str(args.output_dir.resolve())}, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        return
    # Fail cheap before loading the 8B model; diagnose() also checks before writes.
    output = args.output_dir.resolve()
    original = args.segmentation_dir.resolve()
    if output == original or output.is_relative_to(original) or original.is_relative_to(output):
        parser.error("Use a diagnostic directory separate from the segmentation run")
    if output.exists() and any(output.iterdir()):
        parser.error("Use a new, empty diagnostic output directory")
    preflight_video(protocol["source"]["path"], ffmpeg=settings.ffmpeg, ffprobe=settings.ffprobe)
    normalize_omp_threads()
    from pipeline.identity.qwen_backend import QwenBackend, ensure_gpu_idle
    ensure_gpu_idle()
    print("Loading local Qwen once for all diagnostic windows", flush=True)
    backend = QwenBackend(settings.qwen_model_dir)
    diagnose(args.segmentation_dir, args.output_dir, windows, protocol, backend,
             frames=args.frames, ffmpeg=settings.ffmpeg, ffprobe=settings.ffprobe,
             video_tokens=args.video_tokens, query_types=args.queries)


if __name__ == "__main__":
    main()
