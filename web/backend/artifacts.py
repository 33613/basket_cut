"""Read model artifacts into compact dashboard-friendly summaries."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    values = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                values.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if limit is not None and len(values) >= limit:
                break
    return values


def best_action(record: dict[str, Any]) -> dict[str, Any] | None:
    selected = record.get("selected_actions") or []
    if selected:
        return max(selected, key=lambda item: float(item.get("score", 0)))
    candidates = record.get("action_candidates") or []
    if candidates:
        value = dict(candidates[0])
        value["below_threshold"] = True
        return value
    return None


def action_index(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    people: dict[str, dict[str, Any]] = {}
    label_counts: Counter[str] = Counter()
    for record in records:
        person_id = record.get("person_id")
        raw_track_id = int(record.get("raw_track_id", record.get("track_id", -1)))
        key = str(person_id or f"T{raw_track_id}")
        person = people.setdefault(
            key,
            {
                "person_id": person_id,
                "raw_track_ids": set(),
                "event_count": 0,
                "events": [],
                "labels": Counter(),
            },
        )
        person["raw_track_ids"].add(raw_track_id)
        action = best_action(record)
        if action is None:
            continue
        label = str(action.get("label", "unknown"))
        score = float(action.get("score", 0))
        person["event_count"] += 1
        person["labels"][label] += 1
        label_counts[label] += 1
        if len(person["events"]) < 60:
            person["events"].append(
                {
                    "frame_idx": int(record.get("frame_idx", 0)),
                    "timestamp_s": float(record.get("timestamp_s", 0)),
                    "track_id": int(record.get("track_id", raw_track_id)),
                    "label": label,
                    "score": score,
                    "below_threshold": bool(action.get("below_threshold", False)),
                }
            )
    result = []
    for value in people.values():
        value["raw_track_ids"] = sorted(value["raw_track_ids"])
        value["labels"] = dict(value["labels"].most_common())
        result.append(value)
    result.sort(key=lambda item: (-item["event_count"], str(item["person_id"])))
    return {
        "person_count": len(result),
        "event_count": sum(item["event_count"] for item in result),
        "label_counts": dict(label_counts.most_common()),
        "people": result,
    }


def collect_video_artifacts(video: dict[str, Any]) -> dict[str, Any]:
    output = Path(video["output_dir"])
    tracking = output / "tracking"
    identity = output / "identity"
    action = output / "action"
    visualization = output / "visualization"

    identities = read_jsonl(identity / "identities.jsonl")
    identity_by_id = {
        str(item.get("person_id")): item for item in identities if item.get("person_id")
    }
    action_path = action / "actions_with_identity.jsonl"
    if not action_path.is_file():
        action_path = action / "actions.jsonl"
    actions = read_jsonl(action_path)
    indexed_actions = action_index(actions)
    for person in indexed_actions["people"]:
        archive = identity_by_id.get(str(person.get("person_id")))
        person["identity"] = archive

    available = {
        "source": Path(video["source_path"]).is_file(),
        "tracks_video": (tracking / "tracks_vis.mp4").is_file(),
        "final_video": (visualization / "result_web.mp4").is_file()
        or (visualization / "result.mp4").is_file(),
        "tracks": (tracking / "tracks.jsonl").is_file(),
        "identities": bool(identities),
        "actions": bool(actions),
    }
    return {
        "available": available,
        "tracking": {
            "summary": load_json(tracking / "track_summary.json"),
            "video_meta": load_json(tracking / "video_meta.json"),
        },
        "identity": {
            "summary": load_json(identity / "kpr_summary.json")
            or load_json(identity / "kpr_prepare_summary.json"),
            "manifest": load_json(identity / "identity_archive_manifest.json"),
            "people": identities,
        },
        "action": {
            "summary": load_json(action / "actions.summary.json"),
            "identity_link_summary": load_json(
                action / "actions_with_identity.summary.json"
            ),
            "index": indexed_actions,
        },
    }


def project_people_index(manifest: dict[str, Any]) -> dict[str, Any]:
    """Aggregate current clip-local archives without inventing cross-clip IDs."""
    entries = []
    for video in manifest.get("videos", []):
        artifacts = collect_video_artifacts(video)
        for person in artifacts["action"]["index"]["people"]:
            entries.append(
                {
                    "video_id": video["video_id"],
                    "filename": video["filename"],
                    **person,
                }
            )
    return {
        "identity_scope": "clip_local",
        "cross_clip_resolution": "not_calibrated",
        "warning": (
            "The current archive IDs are local to one clip. Entries are not "
            "automatically merged across clips until KPR distance calibration "
            "or a named gallery is available."
        ),
        "entry_count": len(entries),
        "entries": entries,
    }
