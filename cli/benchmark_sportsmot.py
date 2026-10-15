"""Audit official SportsMOT basketball val using original image frames."""
import argparse
import json
import subprocess
from pathlib import Path

from contracts.execution import ExecutionSettings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--env-file', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--limit', type=int, default=5, help='Sorted first N downloaded official basketball val sequences; 0=all')
    parser.add_argument('--mode', choices=('parity', 'official', 'adapter'), default='parity')
    parser.add_argument('--stage', choices=('prepare', 'infer', 'evaluate', 'status', 'all'), default='all')
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    if args.stage == 'status':
        path = output / 'status.json'
        status = json.loads(path.read_text()) if path.exists() else dict(status='not_completed')
        print(json.dumps(status, ensure_ascii=False, indent=2))
        manifest = output / 'dataset.json'
        if manifest.exists():
            count = len(json.loads(manifest.read_text())['sequences'])
            methods = (status['mode'],) if status.get('mode') in ('official', 'adapter') else ('official', 'adapter')
            for method in methods:
                completed = len(list((output / method / 'val/completed').glob('*.json')))
                print(f'{method}: {completed}/{count} complete markers')
        summary = output / 'summary.csv'
        if status.get('status') == 'completed':
            from analysis.teamtrack.dataset import sha256
            if not summary.exists() or sha256(summary) != status['summary_sha256']:
                raise ValueError('Completed summary checksum mismatch')
            import csv
            for row in csv.DictReader(summary.open()):
                if row['scope'] == 'overall':
                    print(json.dumps(row, ensure_ascii=False, indent=2))
        return
    if args.env_file:
        from cli.serve_web import load_environment
        load_environment(args.env_file)
    settings = ExecutionSettings()
    checkpoint = (args.checkpoint or settings.motip_checkpoint).expanduser().resolve()
    if args.stage in ('evaluate', 'all'):
        from analysis.teamtrack.metrics import trackeval_module
        trackeval_module()
    if args.stage == 'all' and args.mode in ('official', 'parity'):
        from analysis.sportsmot.runner import preflight
        preflight(settings.motip_python)
    if args.stage in ('prepare', 'all'):
        if args.dataset_root is None:
            parser.error('--dataset-root is required to prepare original frames')
        from analysis.sportsmot.dataset import prepare
        prepare(args.dataset_root, output, args.limit)
    if args.stage == 'infer':
        if args.mode == 'parity':
            parser.error('The internal infer worker requires --mode official or adapter')
        from analysis.sportsmot.runner import infer
        infer(output, checkpoint, args.mode)
    elif args.stage == 'all':
        from analysis.teamtrack.dataset import save_json
        save_json(output / 'status.json', dict(status='running', mode=args.mode))
        for method in (('official', 'adapter') if args.mode == 'parity' else (args.mode,)):
            command = [str(settings.motip_python), '-u', '-m', 'cli.benchmark_sportsmot',
                       '--output-dir', str(output), '--stage', 'infer', '--mode', method,
                       '--checkpoint', str(checkpoint)]
            subprocess.run(command, cwd=settings.repository_root, check=True)
    if args.stage in ('evaluate', 'all'):
        from analysis.sportsmot.evaluation import evaluate
        status = evaluate(output, args.mode)
        if status['parity_passed'] is False:
            raise SystemExit('Completed with differences: inspect parity.json and metric_deltas.csv before expanding the experiment.')


if __name__ == '__main__':
    main()
