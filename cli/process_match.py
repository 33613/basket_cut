"""Long match -> Qwen clips -> serial tracking, identity, actions and library."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess

from cli.segment_video import add_options, options_from
from contracts.execution import ExecutionSettings, normalize_omp_threads
from workflows.batch import run_batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--match-id", required=True)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--stage", choices=("segment", "process", "full"), default="full")
    parser.add_argument("--review-file", type=Path)
    parser.add_argument("--limit", type=int, help="Optional first N exported clips for an initial GPU run")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--player-match-distance", type=float, default=.2)
    parser.add_argument("--player-novelty-distance", type=float, default=.5)
    parser.add_argument("--player-min-margin", type=float, default=.05)
    add_options(parser)
    args = parser.parse_args()
    if args.env_file:
        from cli.serve_web import load_environment
        load_environment(args.env_file)
    options_from(args).validate()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    settings = ExecutionSettings()
    root = args.output_dir.resolve()
    segmentation, results = root / "segmentation", root / "results"
    manifest = segmentation / "clips_manifest.json"
    previous_batch = results / "batch_manifest.json"
    if previous_batch.is_file() and json.loads(previous_batch.read_text()).get("match_id") != args.match_id:
        raise ValueError("Output directory belongs to a different match")
    command = [str(settings.qwen_python), "-u", "-m", "cli.segment_video",
               "--input", str(args.input.resolve()), "--output-dir", str(segmentation),
               "--model-dir", str(settings.qwen_model_dir)]
    for key, value in vars(options_from(args)).items():
        command.extend(["--" + key.replace("_", "-"), str(value)])
    if args.env_file:
        command.extend(["--env-file", str(args.env_file.resolve())])
    if args.review_file:
        command.extend(["--review-file", str(args.review_file.resolve())])
    if args.dry_run:
        print(json.dumps({"dry_run": True, "stage": args.stage,
            "segmentation_command": shlex.join(command), "results_dir": str(results),
            "next": "Each exported clip resets MOTIP. KPR + Qwen evidence update one match library; actions join by clip/track."}, indent=2))
        return
    normalize_omp_threads()
    if args.stage in ("segment", "full"):
        root.mkdir(parents=True, exist_ok=True)
        # Subprocess exit releases the Qwen model before MOTIP starts.
        subprocess.run(command, cwd=settings.repository_root, check=True)
    if args.stage in ("process", "full"):
        exported = json.loads(manifest.read_text())
        if Path(exported["source"]["path"]).resolve() != args.input.resolve():
            raise ValueError("--input differs from the source in the exported clip manifest")
        count = len(exported["videos"])
        result = run_batch(settings, segmentation, results, limit=args.limit or count,
            clips_manifest=manifest, options={"match_id": args.match_id,
                "jersey_qwen": True, "min_free_gb": args.min_free_gb,
                "player_match_distance": args.player_match_distance,
                "player_novelty_distance": args.player_novelty_distance,
                "player_min_margin": args.player_min_margin})
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if result["failed_clips"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
