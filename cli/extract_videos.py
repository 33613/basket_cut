"""Inspect or extract a bounded video subset from a local TAR/ZIP."""

import argparse
import json
from pathlib import Path
from tools.datasets.video_archive import extract_videos


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', required=True, type=Path)
    p.add_argument('--output-dir', required=True, type=Path)
    p.add_argument('--limit', type=int, default=30)
    p.add_argument('--max-gb', type=float, default=6)
    p.add_argument('--selection', type=Path, help='Official video-key list from prepare_multisports_reference --list-videos')
    p.add_argument('--list-only', action='store_true')
    print(json.dumps(extract_videos(**vars(p.parse_args())), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
