"""Evaluate identity grouping against independent same/different-person labels."""

import argparse
import json
from pathlib import Path

from analysis.evaluation.identity import evaluate_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference", "prediction", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            evaluate_identity(args.reference, args.prediction, args.output),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
