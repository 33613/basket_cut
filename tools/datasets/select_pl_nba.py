#!/usr/bin/env python3
"""Match made-event inventory to local videos; emit rsync list and timing review.

Standalone standard-library tool, usable on a Mac without project dependencies.
Does not copy, rename, transcode, or upload videos, or invent basket timestamps.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            result.update(block)
    return result.hexdigest()


def candidate_time(value, base):
    seconds, fraction = map(int, value.split(":"))
    return seconds + fraction / base if fraction < base else ""


def select(inventory, input_dir, output_dir):
    inventory, root, output = (
        Path(inventory),
        Path(input_dir).expanduser().resolve(),
        Path(output_dir).expanduser().resolve(),
    )
    if (
        not root.is_dir()
        or output == root
        or output.is_relative_to(root)
        or root.is_relative_to(output)
    ):
        raise ValueError(
            "Input must exist and selection output must be a separate directory"
        )
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new/empty selection output directory")
    events = [
        json.loads(line) for line in inventory.read_text().splitlines() if line.strip()
    ]
    if not events or len({e["event_id"] for e in events}) != len(events):
        raise ValueError("Inventory must contain unique made events")
    if len({e["match_id"] for e in events}) != 1 or any(
        e["result"] != "made" for e in events
    ):
        raise ValueError("Select one match with made events only")
    videos = [
        p
        for p in root.rglob("*")
        if p.is_file()
        and p.suffix.lower() in {".mp4", ".mov", ".mkv", ".m4v", ".avi"}
        and not p.name.startswith("._")
        and "__MACOSX" not in p.parts
    ]
    matches, review, audit, checksums = {}, [], [], {}
    for event in events:
        vid = str(event["video_id"])
        found = [
            p
            for p in videos
            if p.stem == event["original_video_basename"]
            or p.stem == vid
            or p.stem.split("_", 1)[0] == vid
        ]
        status = (
            "matched" if len(found) == 1 else "missing" if not found else "ambiguous"
        )
        relative = found[0].relative_to(root).as_posix() if len(found) == 1 else ""
        audit.append(
            {
                "event_id": event["event_id"],
                "video_id": vid,
                "status": status,
                "candidates": [p.relative_to(root).as_posix() for p in found],
            }
        )
        if status != "matched":
            continue
        path = found[0]
        if not path.resolve().is_relative_to(root) or any(
            c in relative for c in "\r\n"
        ):
            raise ValueError("Unsafe video path: " + relative)
        matches[relative] = path
        checksums.setdefault(relative, None)
        raw = event["timestamp_video_raw"]
        review.append(
            {
                "event_id": event["event_id"],
                "video_id": vid,
                "filename": relative,
                "raw_action_start": raw["start"],
                "raw_action_end": raw["end"],
                "frames60_action_start_s": candidate_time(raw["start"], 60),
                "frames60_action_end_s": candidate_time(raw["end"], 60),
                "centiseconds_action_start_s": candidate_time(raw["start"], 100),
                "centiseconds_action_end_s": candidate_time(raw["end"], 100),
                "goal_time_s": "",
                "timing_checked": "",
                "confirmed_time_format": "",
                "evaluation_player_name": event.get("evaluation_player_name") or "",
                "evaluation_cohort": event.get("evaluation_cohort", "unassigned"),
                "action_label": event["action_label"],
                "annotation_source": "PL-NBA",
                "label_correction": "",
                "notes": "",
            }
        )
    output.mkdir(parents=True, exist_ok=True)
    failures = [row for row in audit if row["status"] != "matched"]
    manifest = {
        "schema_version": 1,
        "status": "incomplete" if failures else "ready_to_upload",
        "match_id": events[0]["match_id"],
        "input_dir": str(root),
        "inventory_sha256": digest(inventory),
        "planned_events": len(events),
        "matched_events": len(review),
        "matched_videos": len(matches),
        "video_audit": audit,
    }
    if not failures:
        for relative, path in matches.items():
            checksums[relative] = {"sha256": digest(path), "bytes": path.stat().st_size}
        manifest["files"] = checksums
        (output / "upload_files.txt").write_text(
            "".join(name + "\n" for name in sorted(matches))
        )
        (output / "made_events.jsonl").write_bytes(inventory.read_bytes())
        with (output / "timing_review.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(review[0]))
            writer.writeheader()
            writer.writerows(review)
    (output / "selection.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    if failures:
        raise ValueError(
            f"{len(failures)} missing/ambiguous events; inspect {output / 'selection.json'}. No upload list emitted."
        )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = select(args.inventory, args.input_dir, args.output_dir)
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "status",
                    "match_id",
                    "planned_events",
                    "matched_events",
                    "matched_videos",
                )
            },
            indent=2,
        )
    )
    print("Basket times remain blank. Review timing before running attribution.")


if __name__ == "__main__":
    main()
