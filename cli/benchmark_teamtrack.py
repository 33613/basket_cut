"""Prepare, track and evaluate raw TeamTrack basketball using the SportsMOT checkpoint."""
import argparse
import json
import subprocess
from pathlib import Path

from contracts.execution import ExecutionSettings


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--split', choices=('val', 'test'), required=True)
    p.add_argument('--stage', choices=('prepare', 'infer', 'evaluate', 'all'), default='all')
    p.add_argument('--env-file', type=Path)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    if args.env_file:
        from cli.serve_web import load_environment
        load_environment(args.env_file)
    settings = ExecutionSettings()
    if args.stage in ('evaluate', 'all'):
        # Fail on missing CPU evaluator dependencies before starting expensive GPU work.
        from analysis.teamtrack.metrics import evaluate, trackeval_module
        trackeval_module()
    if args.stage in ('prepare', 'all'):
        from analysis.teamtrack.dataset import prepare
        prepare(args.dataset_root, args.output_dir, args.split)
    if args.stage == 'infer':
        from analysis.teamtrack.inference import infer
        infer(args.output_dir, args.split, args.checkpoint or settings.motip_checkpoint, args.device)
    elif args.stage == 'all':
        command = [str(settings.motip_python), '-u', '-m', 'cli.benchmark_teamtrack',
                   '--stage', 'infer', '--dataset-root', str(args.dataset_root.resolve()),
                   '--output-dir', str(args.output_dir.resolve()), '--split', args.split,
                   '--checkpoint', str((args.checkpoint or settings.motip_checkpoint).resolve()),
                   '--device', args.device]
        subprocess.run(command, cwd=settings.repository_root, check=True)
    if args.stage in ('evaluate', 'all'):
        summaries = evaluate(args.output_dir, args.split)
        print(json.dumps([r for r in summaries if r['scope'] != 'sequence'], ensure_ascii=False, indent=2))
        print('Results:', args.output_dir / args.split / 'metrics/summary.csv')


if __name__ == '__main__':
    main()
