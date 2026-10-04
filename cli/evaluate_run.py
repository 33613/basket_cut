"""Summarize a batch without inventing model accuracy from output counts."""

import argparse
import json
from pathlib import Path
from analysis.evaluation.run_summary import summarize_run


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', required=True, type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--web-output-root', type=Path, help='Read saved review sidecars without changing them')
    args = p.parse_args()
    result = summarize_run(args.run_dir, args.output or args.run_dir / 'run_evaluation.json', args.web_output_root)
    print(json.dumps({k: result[k] for k in ('expected_clips', 'completed_clips', 'failed_clips', 'headline', 'diagnostics_totals', 'warning')}, ensure_ascii=False, indent=2))
    if result['failed_clips']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
