"""Initialize players from six clear continuous MOTIP windows, without photo identities."""

import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys

from contracts.execution import ExecutionSettings, normalize_omp_threads


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--match-id", required=True)
    p.add_argument("--env-file", type=Path)
    p.add_argument(
        "--stage",
        choices=("all", "track", "assess", "features", "register"),
        default="all",
    )
    p.add_argument("--windows", type=int, default=6)
    p.add_argument("--window-s", type=float, default=4.0)
    p.add_argument(
        "--windows-file",
        type=Path,
        help="JSON list of six explicit source start seconds",
    )
    p.add_argument(
        "--team-map",
        type=Path,
        help='Optional JSON mapping shirt colors to team IDs, e.g. {"black":"home","white":"away"}',
    )
    p.add_argument("--min-height", type=int, default=128)
    p.add_argument("--min-width", type=int, default=40)
    p.add_argument("--min-sharpness", type=float, default=30.0)
    p.add_argument("--samples-per-track", type=int, default=4)
    args = p.parse_args()
    if args.env_file:
        from cli.serve_web import load_environment

        load_environment(args.env_file)
    normalize_omp_threads()
    settings = ExecutionSettings()
    from workflows import tracklet_bootstrap as flow

    policy = flow.WindowPolicy(
        windows=args.windows,
        window_s=args.window_s,
        min_height=args.min_height,
        min_width=args.min_width,
        min_sharpness=args.min_sharpness,
        samples_per_track=args.samples_per_track,
    ).validate()
    root = flow.folder(args.run_dir)
    root.mkdir(parents=True, exist_ok=True)
    if args.stage == "all":
        with (root / ".lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("Initialization already running") from exc
            # Validate completed evidence without loading three GPU models.
            if (root / "registration.json").exists():
                from pipeline.identity.qwen_backend import model_provenance
                from pipeline.identity.player_evidence import assessment_provenance

                manifest = flow.load_manifest(args.run_dir)
                contract = json.loads((root / "feature_contract.json").read_text())
                assessments = json.loads((root / "assessments.json").read_text())
                requested_map = (
                    json.loads(args.team_map.read_text()) if args.team_map else None
                )
                source = manifest["protocol"]["source"]
                groups = (
                    flow.manual_starts(args.windows_file, source["duration_s"], policy)
                    if args.windows_file
                    else flow.candidate_starts(source["duration_s"], policy)
                )
                if (
                    Path(source["path"]).resolve() != args.input.resolve()
                    or manifest["protocol"]["policy"] != flow.asdict(policy)
                    or manifest["protocol"]["candidate_starts"] != groups
                    or manifest["protocol"]["detector"]["checkpoint_sha256"]
                    != flow.sha256_file(settings.motip_checkpoint)
                    or contract["signature"]["checkpoint"]
                    != flow.sha256_file(settings.kpr_checkpoint)
                    or assessments["signature"]["model"]
                    != assessment_provenance(model_provenance(settings.qwen_model_dir))
                    or assessments["signature"]["team_map"] != requested_map
                ):
                    raise ValueError(
                        "Registered initialization settings/model changed; use a new output directory"
                    )
                print(
                    json.dumps(
                        flow.register(args.run_dir, args.match_id),
                        ensure_ascii=False,
                        indent=2,
                    )
                )
                return
            for stage, python in (
                ("track", settings.motip_python),
                ("assess", settings.qwen_python),
                ("features", settings.kpr_python),
                ("register", sys.executable),
            ):
                command = [
                    str(python),
                    "-u",
                    "-m",
                    "cli.bootstrap_tracklets",
                    *sys.argv[1:],
                ]
                if "--stage" in command:
                    command[command.index("--stage") + 1] = stage
                else:
                    command.extend(["--stage", stage])
                subprocess.run(command, cwd=settings.repository_root, check=True)
        return
    if (root / "registration.json").exists() and args.stage != "register":
        raise ValueError(
            "Registered model evidence is immutable; resume with --stage all"
        )
    if args.stage == "track":
        from pipeline.tracking.motip_backend import MotipBackend, MotipRuntimeConfig

        config = (
            settings.repository_root
            / "MOTIP/configs/r50_deformable_detr_motip_sportsmot.yaml"
        )
        runtime = MotipRuntimeConfig()
        backend = MotipBackend(
            motip_root=settings.repository_root / "MOTIP",
            config_path=config,
            checkpoint_path=settings.motip_checkpoint,
            runtime_config=runtime,
            preprocessing="upstream-pil",
        )
        flow.prepare(
            args.input,
            args.run_dir,
            backend,
            policy=policy,
            ffmpeg=settings.ffmpeg,
            ffprobe=settings.ffprobe,
            windows_file=args.windows_file,
            detector={
                "checkpoint_sha256": flow.sha256_file(settings.motip_checkpoint),
                "config_sha256": flow.sha256_file(config),
                "runtime": runtime.to_dict(),
                "preprocessing": "upstream-pil",
            },
        )
    elif args.stage == "assess":
        from pipeline.identity.qwen_backend import QwenBackend, ensure_gpu_idle

        ensure_gpu_idle()
        flow.assess(
            args.run_dir,
            QwenBackend(settings.qwen_model_dir),
            team_map=json.loads(args.team_map.read_text()) if args.team_map else None,
        )
    elif args.stage == "features":
        from pipeline.identity.kpr_backend import KPRBackend

        backend = KPRBackend(
            kpr_root=settings.repository_root / "KPR",
            config_path=settings.repository_root
            / "configs/kpr/multidataset_sports_test.yaml",
            checkpoint_path=settings.kpr_checkpoint,
        )
        flow.extract_features(args.run_dir, backend, settings.kpr_checkpoint)
    else:
        print(
            json.dumps(
                flow.register(args.run_dir, args.match_id), ensure_ascii=False, indent=2
            )
        )


if __name__ == "__main__":
    main()
