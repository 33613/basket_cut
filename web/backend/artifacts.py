"""Read model artifacts into compact dashboard-friendly summaries."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from contracts.schema import EventRecord


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
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


def temporal_event_index(
    path: Path, video_id: str | None
) -> tuple[dict[str, Any], list[str]]:
    """Index actual intervals, never promote below-threshold action points."""
    people: dict[str, dict[str, Any]] = {}
    counts: Counter[str] = Counter()
    warnings = []
    if path.is_file():
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    record = EventRecord.from_dict(json.loads(line))
                    if video_id and record.video_id != video_id:
                        raise ValueError(
                            "event video_id differs from tracking metadata"
                        )
                except (ValueError, TypeError, KeyError) as exc:
                    warnings.append(f"events.jsonl:{line_number}: {exc}")
                    continue
                event = record.to_dict()
                person = people.setdefault(
                    record.identity_id,
                    {
                        "person_id": record.identity_id,
                        "raw_track_ids": set(),
                        "events": [],
                        "event_count": 0,
                        "labels": Counter(),
                    },
                )
                person["raw_track_ids"].update(record.raw_track_ids)
                person["events"].append(event)
                person["event_count"] += 1
                person["labels"][record.event] += 1
                counts[record.event] += 1
    result = []
    for person in people.values():
        person["raw_track_ids"] = sorted(person["raw_track_ids"])
        person["labels"] = dict(person["labels"])
        person["events"].sort(
            key=lambda item: (item["start"], item["end"], item["event_id"])
        )
        result.append(person)
    result.sort(key=lambda item: (-item["event_count"], item["person_id"]))
    return {
        "kind": "temporal_events",
        "person_count": len(result),
        "event_count": sum(item["event_count"] for item in result),
        "label_counts": dict(counts.most_common()),
        "people": result,
    }, warnings


def final_video_path(output: Path) -> Path | None:
    return next(
        (
            output / path
            for path in (
                "visualization/result_web.mp4",
                "visualization/result.mp4",
                "result.mp4",
            )
            if (output / path).is_file()
        ),
        None,
    )


def collect_video_artifacts(video: dict[str, Any]) -> dict[str, Any]:
    output = Path(video["output_dir"])
    tracking = output / "tracking"
    identity = output / "identity"
    action = output / "action"
    meta = load_json(tracking / "video_meta.json")

    identities = read_jsonl(identity / "identities.jsonl")
    identity_by_id = {
        str(item.get("person_id")): item for item in identities if item.get("person_id")
    }
    action_path = action / "actions_with_identity.jsonl"
    if not action_path.is_file():
        action_path = action / "actions.jsonl"
    actions = read_jsonl(action_path)
    indexed_actions = action_index(actions)
    indexed_events, warnings = temporal_event_index(
        action / "events.jsonl", (meta or {}).get("video_id")
    )
    for index in (indexed_actions, indexed_events):
        for person in index["people"]:
            person["identity"] = identity_by_id.get(str(person.get("person_id")))

    available = {
        "source": Path(video["source_path"]).is_file(),
        "tracks_video": (tracking / "tracks_vis.mp4").is_file(),
        "final_video": final_video_path(output) is not None,
        "tracks": (tracking / "tracks.jsonl").is_file(),
        "identities": bool(identities),
        "actions": bool(actions),
        "events": (action / "events.jsonl").is_file(),
    }
    return {
        "available": available,
        "tracking": {
            "summary": load_json(tracking / "track_summary.json"),
            "video_meta": meta,
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
            "points": indexed_actions,
            "index": indexed_events,
            "event_summary": load_json(action / "events.summary.json"),
        },
        "evaluation": {
            "tracking": load_json(output / "analysis/tracking_metrics.json"),
            "events": load_json(output / "analysis/event_metrics.json"),
        },
        "warnings": warnings,
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
