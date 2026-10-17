"""Long match -> Qwen clips -> serial tracking, identity, actions and library."""

import argparse
import fcntl
import json
from pathlib import Path
import shlex
import subprocess
import sys

from cli.segment_video import add_options, options_from
from contracts.execution import ExecutionSettings, normalize_omp_threads
from workflows.batch import run_batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--match-id")
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--stage", choices=("initialize", "segment", "process", "full", "status"), default="full")
    parser.add_argument("--bootstrap-windows", type=int, choices=(6,), default=6)
    parser.add_argument("--bootstrap-window-s", type=float, default=4.)
    parser.add_argument("--bootstrap-windows-file", type=Path)
    parser.add_argument("--bootstrap-team-map", type=Path)
    parser.add_argument("--skip-bootstrap", action="store_true")
    parser.add_argument("--allow-unreviewed", action="store_true", help="Explicitly run exploratory downstream processing with unresolved segmentation")
    parser.add_argument("--review-file", type=Path)
    parser.add_argument("--limit", type=int, help="Optional first N exported clips for an initial GPU run")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--player-match-distance", type=float, default=.2)
    parser.add_argument("--player-novelty-distance", type=float, default=.5)
    parser.add_argument("--player-min-margin", type=float, default=.05)
    add_options(parser)
    parser.set_defaults(coarse_method="states", video_tokens=8192, review_frames=16)
    args = parser.parse_args()
    if args.stage != "status" and (args.input is None or not args.match_id):
        parser.error("--input and --match-id are required except for --stage status")
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
    if args.stage == "status":
        from workflows.match_report import match_report
        print(json.dumps(match_report(root), ensure_ascii=False, indent=2))
        return
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
            "bootstrap": {"enabled": not args.skip_bootstrap, "mode": "six_clear_tracking_windows", "windows": args.bootstrap_windows, "window_s": args.bootstrap_window_s},
            "next": "Six native-resolution windows seed clear player tracklets; team/number consensus anchors identities; KPR supplements association. Possession clips update this library."}, indent=2))
        return
    normalize_omp_threads()
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".match.lock").open("a") as run_lock:
        try:
            fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another process_match is writing this match; inspect its progress before resuming") from exc
        from pipeline.media import atomic_json
        match_file = root / "match.json"
        identity = {"match_id": args.match_id, "input": str(args.input.resolve())}
        if match_file.is_file() and json.loads(match_file.read_text()) != identity:
            raise ValueError("Output directory belongs to another match/source")
        atomic_json(match_file, identity)
        if args.stage in ("initialize", "full") and not args.skip_bootstrap:
            bootstrap_command = [sys.executable, "-u", "-m", "cli.bootstrap_tracklets",
                "--input", str(args.input.resolve()), "--run-dir", str(results), "--match-id", args.match_id,
                "--windows", str(args.bootstrap_windows), "--window-s", str(args.bootstrap_window_s)]
            for key,value in (("--windows-file",args.bootstrap_windows_file),("--team-map",args.bootstrap_team_map)):
                if value: bootstrap_command.extend([key,str(value.resolve())])
            if args.env_file:
                bootstrap_command.extend(["--env-file", str(args.env_file.resolve())])
            subprocess.run(bootstrap_command, cwd=settings.repository_root, check=True)
            registration = results / "bootstrap_tracklets/registration.json"
            if args.stage == "full" and registration.is_file() and not json.loads(registration.read_text()).get("ready_for_incremental"):
                raise RuntimeError("Initialization produced no clear team/number identity anchors; inspect bootstrap_tracklets before full processing")
        if args.stage in ("segment", "full"):
            root.mkdir(parents=True, exist_ok=True)
            # Subprocess exit releases the Qwen model before MOTIP starts.
            subprocess.run(command, cwd=settings.repository_root, check=True)
        plan_path = segmentation / "segmentation.json"
        if plan_path.is_file():
            from analysis.possession_evaluation import annotation_template, write_evaluation
            from pipeline.segmentation.possession import BOUNDARY_CONVENTION
            analyzed = json.loads(plan_path.read_text())
            if analyzed.get("boundary_convention") == BOUNDARY_CONVENTION:
                reference = root / "possession-reference.json"
                if not reference.exists():
                    atomic_json(reference, annotation_template(analyzed["source"]))
                write_evaluation(segmentation, reference)
        if args.stage in ("process", "full"):
            exported = json.loads(manifest.read_text())
            if Path(exported["source"]["path"]).resolve() != args.input.resolve():
                raise ValueError("--input differs from the source in the exported clip manifest")
            from pipeline.media import sha256_file
            if sha256_file(args.input) != exported["source"]["sha256"]:
                raise ValueError("Original video changed after clip export")
            plan = json.loads((segmentation / "segmentation.json").read_text())
            uncertain = bool(plan.get("unresolved")) or any(r.get("needs_review") for r in exported["videos"])
            no_changes = len(exported["videos"]) == 1 and exported["source"]["duration_s"] > 2*args.window_s
            if (uncertain or no_changes) and not exported.get("manually_reviewed") and not args.allow_unreviewed:
                raise RuntimeError("Segmentation needs review or found no changes. Inspect segmentation.json; use --allow-unreviewed only for an exploratory run")
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
        from workflows.match_report import match_report
        from pipeline.media import atomic_json
        report = match_report(root)
        atomic_json(root / "match_report.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
