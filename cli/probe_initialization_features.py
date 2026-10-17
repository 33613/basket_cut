"""Compare bounded original initialization crops with KPR without changing the gallery."""

import argparse
import json
from pathlib import Path

from contracts.execution import ExecutionSettings, normalize_omp_threads


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--quality-probe", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--env-file", type=Path)
    args = parser.parse_args()
    if args.env_file:
        from cli.serve_web import load_environment

        load_environment(args.env_file)
    normalize_omp_threads()
    settings = ExecutionSettings()
    from analysis.initialization_feature_probe import probe_features
    from pipeline.identity.kpr_backend import KPRBackend

    backend = KPRBackend(
        kpr_root=settings.repository_root / "KPR",
        config_path=settings.repository_root
        / "configs/kpr/multidataset_sports_test.yaml",
        checkpoint_path=settings.kpr_checkpoint,
    )
    result = probe_features(
        args.run_dir,
        args.quality_probe,
        args.output_dir,
        backend,
        settings.kpr_checkpoint,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("Audit archive:", args.output_dir.with_name(args.output_dir.name + ".tar.gz"))


if __name__ == "__main__":
    main()
