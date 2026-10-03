"""Verify evaluation logic using synthetic data, without datasets or a GPU."""

import argparse

from analysis.evaluation.selfcheck import run_selfcheck


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--with-tracking",
        action="store_true",
        help="Also check IDF1/misses/ID switches; requires requirements-eval.txt",
    )
    args = parser.parse_args()
    if not run_selfcheck(include_tracking=args.with_tracking).wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
