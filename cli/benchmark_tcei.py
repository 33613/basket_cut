"""Paired MOTIP/TCEI on SportsMOT frames or TrackID Indoor native videos."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--dataset", choices=("sportsmot", "trackid-indoor"), default="sportsmot"
    )
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument(
        "--baseline-run",
        type=Path,
        help="Historical SportsMOT run containing dataset.json",
    )
    parser.add_argument("--source-root", type=Path)
    parser.add_argument(
        "--baseline-search-root",
        nargs="+",
        type=Path,
        help="Search for a full completed baseline if --baseline-run is omitted",
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument(
        "--limit", type=int, default=1, help="First N declared sequences; 0=all"
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=("baseline", "off", "object", "id", "dual"),
        default=["baseline", "off", "dual"],
    )
    parser.add_argument(
        "--cache-scope", choices=("sequence", "stream"), default="sequence"
    )
    parser.add_argument(
        "--stage", choices=("all", "infer", "evaluate", "status"), default="all"
    )
    parser.add_argument(
        "--variant", choices=("baseline", "off", "object", "id", "dual")
    )
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    if args.dataset == "trackid-indoor":
        from analysis.trackid.tcei import run

        run(args)
        return
    if args.stage == "status":
        path = output / "status.json"
        print(path.read_text() if path.exists() else '{"status": "not_started"}')
        protocol_path = output / "protocol.json"
        if protocol_path.exists():
            protocol = json.loads(protocol_path.read_text())
            for variant in protocol["variants"]:
                count = len(list((output / variant / "val/completed").glob("*.json")))
                print(
                    f"{variant}: {count}/{len(protocol['dataset']['sequences'])} sequences"
                )
        return
    if args.stage == "infer":
        if args.variant is None:
            parser.error("--variant is required for internal infer worker")
        from analysis.sportsmot.tcei import infer

        infer(output, args.variant)
        return
    from analysis.sportsmot.tcei import evaluate

    if args.stage == "evaluate":
        evaluate(output)
        return
    if (
        args.baseline_run is None and not args.baseline_search_root
    ) or args.source_root is None:
        parser.error(
            "--source-root and either --baseline-run or --baseline-search-root are required"
        )
    if args.limit < 0 or len(set(args.variants)) != len(args.variants):
        parser.error("limit must be >=0 and variants must be unique")
    if not {"baseline", "off"}.issubset(args.variants):
        parser.error("baseline and off are required for the parity gate")
    if args.env_file:
        from cli.serve_web import load_environment

        load_environment(args.env_file)
    from analysis.sportsmot.tcei import discover_baseline, parity, prepare
    from analysis.teamtrack.dataset import save_json
    from analysis.teamtrack.metrics import trackeval_module
    from contracts.execution import ExecutionSettings

    settings = ExecutionSettings()
    trackeval_module()
    checkpoint = (args.checkpoint or settings.motip_checkpoint).expanduser().resolve()
    baseline = (
        args.baseline_run.expanduser().resolve()
        if args.baseline_run
        else discover_baseline(
            [p.expanduser().resolve() for p in args.baseline_search_root], checkpoint
        )
    )
    protocol = prepare(
        output,
        baseline,
        args.source_root.expanduser().resolve(),
        checkpoint,
        args.limit,
        [
            "baseline",
            "off",
            *[v for v in args.variants if v not in {"baseline", "off"}],
        ],
        args.cache_scope,
    )
    try:
        for variant in protocol["variants"]:
            save_json(output / "status.json", {"status": "running", "variant": variant})
            subprocess.run(
                [
                    str(settings.motip_python),
                    "-u",
                    "-m",
                    "cli.benchmark_tcei",
                    "--stage",
                    "infer",
                    "--variant",
                    variant,
                    "--output-dir",
                    str(output),
                ],
                cwd=settings.repository_root,
                env=dict(os.environ, WANDB_MODE="disabled"),
                check=True,
            )
            if variant == "off":
                parity(output, protocol["dataset"]["sequences"])
        evaluate(output)
    except Exception as error:
        save_json(
            output / "status.json",
            {"status": "failed", "error": f"{type(error).__name__}: {error}"},
        )
        raise


if __name__ == "__main__":
    main()
