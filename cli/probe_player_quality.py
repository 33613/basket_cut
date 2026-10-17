"""Test native player crops with local Qwen (default: up to 12) before initialization."""

import argparse
import json
from pathlib import Path

from contracts.execution import ExecutionSettings, normalize_omp_threads


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--env-file", type=Path)
    p.add_argument("--per-window", type=int, choices=(1, 2, 3), default=2)
    p.add_argument(
        "--baseline",
        type=Path,
        help="Completed old probe.json; reuse its exact image selection",
    )
    p.add_argument(
        "--compare-number",
        action="store_true",
        help="Independent number prompt; diagnostic only",
    )
    p.add_argument(
        "--capture-inputs",
        action="store_true",
        help="Save the actual processed image tensors as PNG",
    )
    args = p.parse_args()
    if args.env_file:
        from cli.serve_web import load_environment

        load_environment(args.env_file)
    normalize_omp_threads()
    output = args.output_dir.resolve()
    run = args.run_dir.resolve()
    if output.is_relative_to(run) or run.is_relative_to(output):
        p.error("Use a separate diagnostic output directory")
    if output.exists() and any(output.iterdir()):
        p.error("Use a new probe output directory")
    from analysis.player_quality_probe import probe, select_samples
    from workflows.tracklet_bootstrap import load_manifest

    samples = select_samples(load_manifest(args.run_dir), args.per_window)
    if not samples:
        p.error(
            "No quality-selected track crops; inspect tracking/window selection first"
        )
    from pipeline.identity.qwen_backend import QwenBackend, ensure_gpu_idle

    ensure_gpu_idle()
    report = probe(
        args.run_dir,
        args.output_dir,
        backend=QwenBackend(ExecutionSettings().qwen_model_dir),
        per_window=args.per_window,
        baseline=args.baseline,
        compare_number=args.compare_number,
        capture_inputs=args.capture_inputs,
    )
    print(
        json.dumps(
            {k: v for k, v in report.items() if k != "samples"},
            ensure_ascii=False,
            indent=2,
        )
    )
    print("Read-only probe:", args.output_dir / "index.html")
    print("Audit archive:", args.output_dir.with_name(args.output_dir.name + ".tar.gz"))


if __name__ == "__main__":
    main()
