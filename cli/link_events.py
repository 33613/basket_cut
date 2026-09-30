"""Attach identity archive IDs to action records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Attach canonical person IDs to action JSONL records"
    )
    parser.add_argument("--actions", required=True, type=Path)
    parser.add_argument("--identity-map", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--allow-unmapped", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    from workflows.link_events import LinkEventsOptions, link_actions_to_people

    result = link_actions_to_people(
        LinkEventsOptions(
            actions=args.actions,
            identity_map=args.identity_map,
            output=args.output,
            allow_unmapped=args.allow_unmapped,
            overwrite=args.overwrite,
        )
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
