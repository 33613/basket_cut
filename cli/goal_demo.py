"""Build a reviewed, match-scoped player highlight demo without rerunning MOTIP."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess

from contracts.execution import ExecutionSettings, normalize_omp_threads
from workflows import goal_demo as flow


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--link-run', type=Path)
    parser.add_argument('--review-csv', type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--env-file', type=Path)
    parser.add_argument('--stage', choices=('all', 'prepare', 'features', 'library', 'render'), default='all')
    parser.add_argument('--identity-mode', choices=('reviewed', 'kpr'), default='reviewed')
    parser.add_argument('--samples', type=int, default=6)
    parser.add_argument('--kpr-root', type=Path)
    parser.add_argument('--kpr-config', type=Path)
    parser.add_argument('--kpr-checkpoint', type=Path)
    parser.add_argument('--max-distance', type=float, default=.2)
    parser.add_argument('--player-id')
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.env_file:
        from cli.serve_web import load_environment
        load_environment(args.env_file)
    normalize_omp_threads()
    settings = ExecutionSettings()
    root = args.output_dir.expanduser().resolve()
    if args.width < 2 or args.height < 2 or args.width % 2 or args.height % 2:
        parser.error('Output dimensions must be positive even integers')
    if args.stage in ('all', 'prepare') and (not args.link_run or not args.review_csv):
        parser.error('--link-run and --review-csv are required for prepare/all')
    root.parent.mkdir(parents=True, exist_ok=True)
    with root.with_name('.'+root.name+'.demo.lock').open('a') as lock:
        if not args.worker:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                parser.error('This demo output is already being processed')
        if args.stage in ('all', 'prepare'):
            plan = flow.prepare(args.link_run, args.review_csv, root,
                                identity_mode=args.identity_mode, samples=args.samples)
            print(json.dumps(dict(selected=len(plan['items']), skipped=plan['skipped']), ensure_ascii=False), flush=True)
        plan = flow.read(root/'plan.json')
        if args.stage == 'features' or (args.stage == 'all' and plan['request']['identity_mode'] == 'kpr'):
            kpr_root = args.kpr_root or Path(os.environ.get('BASKET_KPR_ROOT', settings.repository_root/'KPR'))
            config = args.kpr_config or settings.repository_root/'configs/kpr/multidataset_sports_test.yaml'
            checkpoint = args.kpr_checkpoint or settings.kpr_checkpoint
            if args.stage == 'all':
                command = [str(settings.kpr_python), '-u', '-m', 'cli.goal_demo', '--stage', 'features',
                    '--worker', '--output-dir', str(root), '--kpr-root', str(kpr_root),
                    '--kpr-config', str(config), '--kpr-checkpoint', str(checkpoint)]
                if args.cpu:
                    command.append('--cpu')
                subprocess.run(command, cwd=settings.repository_root, check=True)
            else:
                flow.features(root, kpr_root=kpr_root, config=config, checkpoint=checkpoint, cpu=args.cpu)
        if args.stage in ('all', 'library'):
            result = flow.build_library(root, max_distance=args.max_distance)
            print(json.dumps(dict(players=result['global_person_count'], events=result['event_count'])), flush=True)
        if args.stage in ('all', 'render'):
            from pipeline.player_link.demo_render import render_player_videos
            print(json.dumps(render_player_videos(root, player_id=args.player_id,
                ffmpeg=settings.ffmpeg, ffprobe=settings.ffprobe,
                width=args.width, height=args.height), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
