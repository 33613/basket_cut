"""Audit raw tracks and export a separate quality-filtered track file."""

import argparse
import json
from pathlib import Path

from pipeline.tracking.quality import QualityOptions
from workflows.quality import run_track_quality


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracks", type=Path, required=True)
    parser.add_argument("--video-meta", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-observations", type=int, default=3)
    parser.add_argument("--min-observed-seconds", type=float, default=0.1)
    parser.add_argument("--duplicate-iou", type=float, default=0.9)
    parser.add_argument("--jump-speed", type=float, default=60.0)
    parser.add_argument("--suppress-duplicates", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    print(
        json.dumps(
            run_track_quality(QualityOptions(**vars(parser.parse_args()))),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
