"""Build/rebuild a match-scoped library without rerunning any model."""

import argparse
import json
from pathlib import Path

from workflows.person_library import PersonLibraryOptions, build_person_library


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-distance", type=float)
    parser.add_argument("--min-samples", type=int, default=2)
    parser.add_argument("--max-within-distance", type=float, default=0.4)
    parser.add_argument("--min-common-parts", type=int, default=2)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--allow-legacy-features", action="store_true")
    parser.add_argument("--use-jersey-evidence", action="store_true", help="Use consistent JNR number evidence as a cannot-link constraint")
    parser.add_argument("--overwrite", action="store_true")
    print(json.dumps(build_person_library(PersonLibraryOptions(**vars(parser.parse_args()))),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
