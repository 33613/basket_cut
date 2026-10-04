"""Prepare chronological evidence for manual track auditing; no GPU required."""

import argparse
import json
from pathlib import Path

from analysis.visualization.track_review import prepare_review


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--tracks", type=Path, required=True)
    parser.add_argument("--video-meta", type=Path, required=True)
    parser.add_argument("--quality", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples-per-track", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare_review(args.input, args.tracks, args.video_meta, args.output_dir,
                                   args.quality, args.samples_per_track, args.overwrite),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
