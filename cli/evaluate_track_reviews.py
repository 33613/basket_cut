"""Evaluate a frozen set of independent track reviews (CPU only)."""

import argparse
import json
from pathlib import Path

from analysis.evaluation.track_review import evaluate_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate_manifest(args.manifest, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["failed_clips"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
