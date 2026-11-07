"""Inventory PL-NBA made-shot intervals; never invent an exact basket timestamp."""

import argparse
import json
import math
import re
import shutil
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

from pipeline.media import atomic_json, sha256_file

SHOT_LABELS = {"2ptshot", "3ptshot", "layup", "dunk", "freethrow", "putback"}


def selection_package(output, development_video_ids=()):
    """Make a self-contained Mac selection ZIP on the server, outside Git."""
    output = Path(output)
    path = output / "made_events.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows or len({row["match_id"] for row in rows}) != 1:
        raise ValueError("Selection package requires made events from one match")
    development = set(map(str, development_video_ids))
    if development - {row["video_id"] for row in rows}:
        raise ValueError("Development IDs must exist in the selected made inventory")
    for row in rows:
        row["evaluation_cohort"] = (
            "development" if row["video_id"] in development else "new_test"
        )
    rows.sort(key=lambda row: row["video_id"])
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["evaluation_cohorts"] = dict(
        Counter(row["evaluation_cohort"] for row in rows)
    )
    atomic_json(summary_path, summary)
    selector = Path(__file__).resolve().parents[1] / "tools/datasets/select_pl_nba.py"
    shutil.copyfile(selector, output / "select_pl_nba.py")
    (output / "README.txt").write_text(
        "Run select_pl_nba.py with --inventory made_events.jsonl --input-dir VIDEO_DIR --output-dir NEW_DIR.\n"
        "Standard-library Python only. Does not modify or upload videos.\n"
        "Missing or ambiguous videos stop selection. Basket timestamps remain blank for review.\n"
    )
    archive = output.with_suffix(".zip")
    with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED) as bundle:
        for name in (
            "made_events.jsonl",
            "summary.json",
            "select_pl_nba.py",
            "README.txt",
        ):
            bundle.write(output / name, name)
    return {
        "archive": str(archive.resolve()),
        "sha256": sha256_file(archive),
        "events": len(rows),
        "cohorts": summary["evaluation_cohorts"],
    }


def parse_timestamp(value, time_format, fps=None):
    if not isinstance(value, str) or not re.fullmatch(r"\d+:\d{2}", value):
        raise ValueError(f"Unsupported PL-NBA time: {value!r}")
    seconds, fraction = map(int, value.split(":"))
    if time_format == "raw":
        return None
    base = 100 if time_format == "seconds-centiseconds" else fps
    if time_format not in {"seconds-centiseconds", "seconds-frames"}:
        raise ValueError("Unknown time format")
    if base is None or not math.isfinite(base) or base <= 0 or fraction >= base:
        raise ValueError(f"Invalid suffix {value!r} for timestamp base {base}")
    return seconds + fraction / base


def inventory(
    annotation_zip,
    output,
    match_id=None,
    video_ids=None,
    time_format="raw",
    timestamp_fps=None,
):
    annotation_zip, output = Path(annotation_zip), Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output must be a new/empty directory")
    if time_format == "seconds-frames" and (
        timestamp_fps is None or not math.isfinite(timestamp_fps) or timestamp_fps <= 0
    ):
        raise ValueError("seconds-frames requires an explicit --timestamp-fps")
    selected = None if video_ids is None else set(map(str, video_ids))
    if selected is not None and not match_id:
        raise ValueError(
            "Selecting clip numbers requires --match-id; IDs repeat across games"
        )
    rows, games, labels, matched_ids = [], Counter(), Counter(), set()
    with zipfile.ZipFile(annotation_zip) as archive:
        for name in sorted(archive.namelist()):
            if not name.endswith(".json") or "__MACOSX" in name:
                continue
            document = json.loads(archive.read(name))
            game = document["video_id"]
            if match_id and game != match_id:
                continue
            basename = PurePosixPath(name).stem
            clip_id = basename.split("_", 1)[0]
            if selected is not None and clip_id not in selected:
                continue
            games[game] += 1
            matched_ids.add(clip_id)
            for event in document["events"]:
                if (
                    event.get("result") != "made"
                    or str(event.get("event_label", "")).lower() not in SHOT_LABELS
                ):
                    continue
                raw = event["timestamp_video"]
                start = parse_timestamp(raw["start"], time_format, timestamp_fps)
                end = parse_timestamp(raw["end"], time_format, timestamp_fps)
                if start is not None and end < start:
                    raise ValueError(f"Reversed interval: {name}/{event['event_id']}")
                rows.append(
                    {
                        "event_id": f"{game}__{clip_id}__{event['event_id']}",
                        "match_id": game,
                        "video_id": clip_id,
                        "source_annotation": name,
                        "possession_id": document["possession_id"],
                        "original_video_basename": basename,
                        "action_label": event["event_label"],
                        "result": "made",
                        "evaluation_player_name": event.get("player"),
                        "timestamp_video_raw": raw,
                        "action_start_s": start,
                        "action_end_s": end,
                        "goal_time_s": None,
                        "timing_semantics": "action_interval_not_exact_basket_time",
                        "annotation_source": "PL-NBA",
                        "timestamp_format": time_format,
                        "timestamp_fps": timestamp_fps,
                    }
                )
                labels[event["event_label"]] += 1
    if selected is not None and selected - matched_ids:
        raise ValueError(f"Missing clip annotations: {sorted(selected - matched_ids)}")
    if not games:
        raise ValueError("No matching annotation files")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "made_events.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    summary = {
        "annotation_zip": str(annotation_zip.resolve()),
        "annotation_zip_sha256": sha256_file(annotation_zip),
        "matched_annotation_files": sum(games.values()),
        "games": dict(games),
        "made_events": len(rows),
        "labels": dict(labels),
        "timestamp_format": time_format,
        "timestamp_fps": timestamp_fps,
        "exact_goal_timestamps_available": False,
        "warnings": [
            "Time units are not specified by the repository README; raw is the default",
            "Player names and action types may disagree with manual review",
            "Match clip numbers against the actual video files before testing",
        ],
    }
    atomic_json(output / "summary.json", summary)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--annotation-zip", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--match-id")
    p.add_argument("--video-ids", nargs="+")
    p.add_argument(
        "--time-format",
        choices=("raw", "seconds-centiseconds", "seconds-frames"),
        default="raw",
    )
    p.add_argument("--timestamp-fps", type=float)
    p.add_argument(
        "--selection-package",
        action="store_true",
        help="Build a standalone Mac selection ZIP beside the output directory",
    )
    p.add_argument("--development-video-ids", nargs="*", default=[])
    a = p.parse_args()
    if a.selection_package:
        inventory(
            a.annotation_zip,
            a.output_dir,
            a.match_id,
            a.video_ids,
            a.time_format,
            a.timestamp_fps,
        )
        print(
            json.dumps(
                selection_package(a.output_dir, a.development_video_ids),
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    print(
        json.dumps(
            inventory(
                a.annotation_zip,
                a.output_dir,
                a.match_id,
                a.video_ids,
                a.time_format,
                a.timestamp_fps,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
