"""Independently review state candidates without exporting or registering clips."""

import argparse
import json
from pathlib import Path

from analysis.possession_review import review_plan, review_candidates
from contracts.execution import ExecutionSettings, normalize_omp_threads
from pipeline.media import preflight_video


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--padding-s", type=float, default=2)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--video-tokens", type=int, default=8192)
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    original, output = args.diagnostic_dir.resolve(), args.output_dir.resolve()
    if output == original or output.is_relative_to(original) or original.is_relative_to(output):
        parser.error("Use a review directory separate from the state diagnostic")
    if output.exists() and any(output.iterdir()):
        parser.error("Use a new, empty review directory")
    diagnostic = json.loads((original / "diagnostic.json").read_text())
    if diagnostic.get("status") != "completed_not_accuracy_evaluated":
        parser.error("State diagnostic has not completed")
    candidates = review_plan(diagnostic, args.padding_s)
    if len(candidates) > args.max_candidates or not candidates:
        parser.error(f"Found {len(candidates)} candidates; expected 1 through --max-candidates")
    if args.frames % 2 or not 4 <= args.frames <= 64 or not 256 <= args.video_tokens <= 8192:
        parser.error("Use 4–64 even frames and a 256–8192 video-token cap")
    print(json.dumps({"model_queries": len(candidates), "candidates": candidates}, indent=2), flush=True)
    if args.dry_run:
        return
    if args.env_file:
        from cli.serve_web import load_environment
        load_environment(args.env_file)
    settings = ExecutionSettings()
    preflight_video(diagnostic["source"]["path"], ffmpeg=settings.ffmpeg, ffprobe=settings.ffprobe)
    normalize_omp_threads()
    from pipeline.identity.qwen_backend import QwenBackend, ensure_gpu_idle
    ensure_gpu_idle()
    print("Loading local Qwen once for independent candidate review", flush=True)
    backend = QwenBackend(settings.qwen_model_dir)
    result = review_candidates(diagnostic, candidates, output, backend, frames=args.frames,
        video_tokens=args.video_tokens, ffmpeg=settings.ffmpeg, ffprobe=settings.ffprobe)
    decisions = [row.get("review", {}).get("decision", "invalid_response") for row in result["results"]]
    print(json.dumps({"status": result["status"], "decisions": decisions,
                      "output": str(output / "candidate-review.json")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
