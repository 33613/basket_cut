"""Resolve raw KPR archives into conservative, clip-local person archives."""

import argparse
import json
from pathlib import Path

from pipeline.identity.resolution import ResolutionOptions
from workflows.quality import run_identity_resolution


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("tracks", "video-meta", "archive-dir", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--quality-dir", type=Path)
    parser.add_argument(
        "--max-distance",
        type=float,
        help="Explicit uncalibrated KPR threshold; omitted = no automatic merges",
    )
    parser.add_argument("--max-gap-seconds", type=float, default=1.5)
    parser.add_argument("--overlap-iou", type=float, default=0.9)
    parser.add_argument("--max-within-distance", type=float, default=0.4)
    parser.add_argument("--min-samples", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    print(
        json.dumps(
            run_identity_resolution(ResolutionOptions(**vars(parser.parse_args()))),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
