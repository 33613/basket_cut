"""Repair library registration and downstream outputs without rerunning models."""

import argparse
import fcntl
import json
from pathlib import Path

from contracts.execution import ExecutionSettings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True, help='Match root containing results/')
    parser.add_argument('--env-file', type=Path)
    parser.add_argument('--wait', action='store_true', help='Wait for the active match controller to release its lock')
    args = parser.parse_args()
    if args.env_file:
        from cli.serve_web import load_environment
        load_environment(args.env_file)
    root = args.output_dir.resolve()
    if not (root / 'results/batch_manifest.json').is_file():
        raise FileNotFoundError('No processing manifest; wait for segmentation to finish')
    with (root / '.match.lock').open('a') as lock:
        if args.wait:
            print('Waiting for the active match controller; existing tasks are not stopped.', flush=True)
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | (0 if args.wait else fcntl.LOCK_NB))
        except BlockingIOError as exc:
            raise RuntimeError('Match controller is still active; use --wait or retry after it exits') from exc
        from workflows.recovery import repair_player_stages
        report = repair_player_stages(ExecutionSettings(), root / 'results')
        from workflows.match_report import match_report
        from pipeline.media import atomic_json
        atomic_json(root / 'match_report.json', match_report(root))
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if report['failed_clips']:
            raise SystemExit(1)


if __name__ == '__main__':
    main()
