"""Process and incrementally register same-match video clips on one GPU."""
import argparse
import json
import math
from pathlib import Path
from contracts.execution import ExecutionSettings
from workflows.batch import run_batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--match-id', required=True, help='One match per persistent output directory')
    parser.add_argument('--env-file', type=Path)
    parser.add_argument('--selection', type=Path)
    parser.add_argument('--limit', type=int, default=30)
    parser.add_argument('--target', choices=('identity', 'full'), default='full')
    parser.add_argument('--min-free-gb', type=float, default=2)
    parser.add_argument('--player-match-distance', type=float, default=.2)
    parser.add_argument('--player-novelty-distance', type=float, default=.5)
    parser.add_argument('--player-min-margin', type=float, default=.05)
    parser.add_argument('--skip-qwen', action='store_true', help='Explicit KPR-only mode; no number evidence')
    parser.add_argument('--force', action='store_true', help='Rerun stages; registered evidence cannot be overwritten')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.env_file:
        from cli.serve_web import load_environment
        load_environment(args.env_file)
    if not math.isfinite(args.min_free_gb) or args.min_free_gb < 0:
        parser.error('min-free-gb must be finite and nonnegative')
    result = run_batch(ExecutionSettings(), args.input_dir, args.output_dir, limit=args.limit,
        selection=args.selection, target=args.target, force=args.force, dry_run=args.dry_run,
        options={'match_id': args.match_id, 'min_free_gb': args.min_free_gb,
                 'jersey_qwen': not args.skip_qwen, 'player_match_distance': args.player_match_distance,
                 'player_novelty_distance': args.player_novelty_distance, 'player_min_margin': args.player_min_margin})
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result.get('failed_clips'):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
