"""Initialize a match gallery from independent random photographs."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from contracts.execution import ExecutionSettings, normalize_omp_threads


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--match-id", required=True)
    p.add_argument("--env-file", type=Path)
    p.add_argument("--stage", choices=("all", "prepare", "features", "jersey", "register"), default="all")
    p.add_argument("--photos", type=int, default=24)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--player-match-distance", type=float, default=.2)
    p.add_argument("--player-novelty-distance", type=float, default=.5)
    p.add_argument("--player-min-margin", type=float, default=.05)
    args = p.parse_args()
    if args.env_file:
        from cli.serve_web import load_environment
        load_environment(args.env_file)
    normalize_omp_threads()
    settings = ExecutionSettings()
    if args.stage == "all":
        if (args.run_dir / "bootstrap/registration.json").is_file():
            from workflows import bootstrap
            from pipeline.identity.qwen_backend import model_provenance
            photos = bootstrap.verified_photos(args.run_dir)
            original = photos["protocol"]
            features = json.loads((args.run_dir / "bootstrap/feature_contract.json").read_text())
            numbers = json.loads((args.run_dir / "bootstrap/numbers.json").read_text())
            if (Path(original["source"]["path"]).resolve() != args.input.resolve()
                    or original["count"] != args.photos or original["seed"] != args.seed
                    or original["code_sha256"] != bootstrap.sha256_file(Path(bootstrap.__file__))
                    or (original.get("detector") or {}).get("checkpoint_sha256") != bootstrap.sha256_file(settings.motip_checkpoint)
                    or features["metadata"]["checkpoint_sha256"] != bootstrap.sha256_file(settings.kpr_checkpoint)
                    or numbers["signature"]["model"] != model_provenance(settings.qwen_model_dir)):
                raise ValueError("Registered bootstrap settings/model changed; use a new match run")
            bootstrap.register(args.run_dir, args.match_id, max_distance=args.player_match_distance,
                novelty_distance=args.player_novelty_distance, min_margin=args.player_min_margin)
            print("Bootstrap evidence verified and reused without loading GPU models", flush=True)
            return
        for stage, python in (("prepare", settings.motip_python), ("features", settings.kpr_python),
                              ("jersey", settings.qwen_python), ("register", sys.executable)):
            command = [str(python), "-u", "-m", "cli.bootstrap_players", *sys.argv[1:]]
            if "--stage" in command:
                command[command.index("--stage") + 1] = stage
            else:
                command.extend(["--stage", stage])
            subprocess.run(command, cwd=settings.repository_root, check=True)
        return
    from workflows import bootstrap
    if args.stage == "prepare":
        from pipeline.tracking.motip_backend import MotipBackend, MotipRuntimeConfig
        config = settings.repository_root / "MOTIP/configs/r50_deformable_detr_motip_sportsmot.yaml"
        runtime = MotipRuntimeConfig(only_detr=True)
        backend = MotipBackend(motip_root=settings.repository_root / "MOTIP", config_path=config,
            checkpoint_path=settings.motip_checkpoint, runtime_config=runtime, preprocessing="upstream-pil")
        bootstrap.prepare(args.input, args.run_dir, backend, count=args.photos, seed=args.seed,
            ffprobe=settings.ffprobe, detector_contract={"checkpoint_sha256": bootstrap.sha256_file(settings.motip_checkpoint),
                "config_sha256": bootstrap.sha256_file(config), "runtime": runtime.to_dict(), "preprocessing": "upstream-pil"})
    elif args.stage == "features":
        if (args.run_dir / "bootstrap/registration.json").is_file():
            print("Bootstrap already registered; retaining immutable KPR evidence")
            return
        from pipeline.identity.kpr_backend import KPRBackend
        backend = KPRBackend(kpr_root=settings.repository_root / "KPR",
            config_path=settings.repository_root / "configs/kpr/multidataset_sports_test.yaml",
            checkpoint_path=settings.kpr_checkpoint)
        bootstrap.extract_features(args.run_dir, backend, settings.kpr_checkpoint)
    elif args.stage == "jersey":
        if (args.run_dir / "bootstrap/registration.json").is_file():
            print("Bootstrap already registered; retaining immutable Qwen evidence")
            return
        from pipeline.identity.qwen_backend import QwenBackend, ensure_gpu_idle
        ensure_gpu_idle()
        bootstrap.read_numbers(args.run_dir, QwenBackend(settings.qwen_model_dir))
    else:
        result = bootstrap.register(args.run_dir, args.match_id,
            max_distance=args.player_match_distance, novelty_distance=args.player_novelty_distance,
            min_margin=args.player_min_margin)
        print(json.dumps({"match_id": result["match_id"], "candidate_players": result["global_person_count"]}))


if __name__ == "__main__":
    main()
