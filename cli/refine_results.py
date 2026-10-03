"""Refine a completed clip into a separate, reproducible result directory."""

import argparse
import json
from pathlib import Path

from workflows.refinement import RefinementOptions, refine_existing_clip


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--max-distance", type=float)
    parser.add_argument("--min-observations", type=int, default=3)
    parser.add_argument("--min-observed-seconds", type=float, default=0.1)
    parser.add_argument("--suppress-duplicates", action="store_true")
    parser.add_argument("--score-threshold", type=float, default=0.2)
    parser.add_argument("--min-support", type=int, default=1)
    print(
        json.dumps(
            refine_existing_clip(RefinementOptions(**vars(parser.parse_args()))),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
