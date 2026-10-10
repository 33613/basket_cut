"""Attach canonical identity archive IDs to per-track action predictions.

This is intentionally a CPU-only join.  It can run after KPR and the action
backend in either order and never needs to reload either neural network.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from contracts.schema import read_jsonl, write_json, write_jsonl_line


@dataclass(frozen=True)
class LinkEventsOptions:
    actions: Path
    identity_map: Path
    output: Path
    allow_unmapped: bool = False
    overwrite: bool = False
    player_map: Path | None = None
    clip_name: str | None = None


def load_identity_map(path: Path) -> dict[int, dict[str, Any]]:
    mappings: dict[int, dict[str, Any]] = {}
    for value in read_jsonl(path):
        track_id = int(value["raw_track_id"])
        if track_id in mappings:
            raise ValueError(f"Duplicate identity mapping for track {track_id}")
        mappings[track_id] = value
    return mappings


def link_actions_to_people(options: LinkEventsOptions) -> dict[str, Any]:
    actions_path = options.actions.expanduser().resolve()
    identity_map_path = options.identity_map.expanduser().resolve()
    output_path = options.output.expanduser().resolve()
    if output_path in {actions_path, identity_map_path}:
        raise ValueError("Linked output cannot overwrite its inputs")
    if output_path.exists() and not options.overwrite:
        raise FileExistsError(
            f"{output_path} already exists; pass --overwrite to replace it"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mappings = load_identity_map(identity_map_path)
    players = {}
    if options.player_map:
        for row in read_jsonl(options.player_map):
            if row["clip_name"] == options.clip_name:
                for tid in row["raw_track_ids"]:
                    players[int(tid)] = row["global_person_id"]

    record_count = 0
    mapped_count = 0
    unmapped_tracks: Counter[int] = Counter()
    people: Counter[str] = Counter()
    with output_path.open("w", encoding="utf-8") as handle:
        for action in read_jsonl(actions_path):
            record_count += 1
            raw_track_id = int(action["track_id"])
            mapping = mappings.get(raw_track_id)
            if (
                mapping
                and mapping.get("video_id")
                and mapping["video_id"] != action["video_id"]
            ):
                raise ValueError("Action and identity mapping video_id differ")
            if mapping is None:
                unmapped_tracks[raw_track_id] += 1
                if not options.allow_unmapped:
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
            enriched["player_id"] = players.get(raw_track_id)
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
            if options.allow_unmapped
            else mapped_count
        ),
        "person_count": len(people),
        "records_per_person": dict(sorted(people.items())),
        "unmapped_tracks": {
            str(track_id): count for track_id, count in sorted(unmapped_tracks.items())
        },
        "allow_unmapped": bool(options.allow_unmapped),
    }
    write_json(output_path.with_suffix(".summary.json"), summary)
    return summary
