"""Run first-five parity, then all basketball val only if parity succeeds."""
import argparse
import subprocess
import sys
from pathlib import Path

from analysis.teamtrack.dataset import save_json
from contracts.execution import REPOSITORY_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--env-file', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    args = parser.parse_args()
    root = args.output_root.expanduser().resolve()
    for name, mode, limit in [('parity-5', 'parity', 5), ('official-all', 'official', 0)]:
        save_json(root / 'suite_status.json', dict(status='running', stage=name))
        command = [sys.executable, '-u', '-m', 'cli.benchmark_sportsmot',
                   '--dataset-root', str(args.dataset_root.expanduser().resolve()),
                   '--output-dir', str(root / name), '--env-file', str(args.env_file.expanduser().resolve()),
                   '--mode', mode, '--limit', str(limit)]
        if args.checkpoint:
            command.extend(['--checkpoint', str(args.checkpoint.expanduser().resolve())])
        try:
            subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)
        except subprocess.CalledProcessError as error:
            save_json(root / 'suite_status.json', dict(status='stopped', stage=name, exit_code=error.returncode))
            raise SystemExit(error.returncode)
    save_json(root / 'suite_status.json', dict(status='completed', stage='official-all'))
    print('SUITE_COMPLETE: first-five parity passed; full official basketball val evaluation completed.', flush=True)


if __name__ == '__main__':
    main()
