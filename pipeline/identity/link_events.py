"""Attach canonical identity archive IDs to per-track action predictions.

This is intentionally a CPU-only join.  It can run after KPR and the action
backend in either order and never needs to reload either neural network.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from pipeline.common.schema import read_jsonl, write_json, write_jsonl_line


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Attach canonical person IDs to action JSONL records"
    )
    parser.add_argument("--actions", required=True, type=Path)
    parser.add_argument("--identity-map", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--allow-unmapped",
        action="store_true",
        help="Keep action records whose raw track has no identity mapping",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def load_identity_map(path: Path) -> dict[int, dict[str, Any]]:
    mappings: dict[int, dict[str, Any]] = {}
    for value in read_jsonl(path):
        track_id = int(value["raw_track_id"])
        if track_id in mappings:
            raise ValueError(f"Duplicate identity mapping for track {track_id}")
        mappings[track_id] = value
    if not mappings:
        raise ValueError(f"Identity map is empty: {path}")
    return mappings


def run(args: argparse.Namespace) -> dict[str, Any]:
    actions_path = args.actions.expanduser().resolve()
    identity_map_path = args.identity_map.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"{output_path} already exists; pass --overwrite to replace it"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mappings = load_identity_map(identity_map_path)

    record_count = 0
    mapped_count = 0
    unmapped_tracks: Counter[int] = Counter()
    people: Counter[str] = Counter()
    with output_path.open("w", encoding="utf-8") as handle:
        for action in read_jsonl(actions_path):
            record_count += 1
            raw_track_id = int(action["track_id"])
            mapping = mappings.get(raw_track_id)
            if mapping is None:
                unmapped_tracks[raw_track_id] += 1
                if not args.allow_unmapped:
                    continue
                person_id = None
                identity_status = "unmapped"
            else:
                mapped_count += 1
                person_id = str(mapping["person_id"])
                identity_status = str(mapping.get("status", "unlabeled"))
                people[person_id] += 1
            enriched = dict(action)
            enriched["raw_track_id"] = raw_track_id
            enriched["person_id"] = person_id
            enriched["identity_label"] = (
                mapping.get("identity_label") if mapping else None
            )
            enriched["identity_status"] = identity_status
            write_jsonl_line(handle, enriched)

    summary = {
        "actions": str(actions_path),
        "identity_map": str(identity_map_path),
        "output": str(output_path),
        "input_action_records": record_count,
        "mapped_action_records": mapped_count,
        "written_action_records": (
            mapped_count + sum(unmapped_tracks.values())
            if args.allow_unmapped
            else mapped_count
        ),
        "person_count": len(people),
        "records_per_person": dict(sorted(people.items())),
        "unmapped_tracks": {
            str(track_id): count for track_id, count in sorted(unmapped_tracks.items())
        },
        "allow_unmapped": bool(args.allow_unmapped),
    }
    write_json(output_path.with_suffix(".summary.json"), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
