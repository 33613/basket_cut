"""Artifact presence and paged image evidence for the deployed clip workflow."""

import json
from pathlib import Path

from web.backend.artifacts import load_json


STAGE_OUTPUTS = {
    "tracking": ("tracking/tracks.jsonl", "tracking/video_meta.json"),
    "quality": ("quality/quality_summary.json", "quality/quality_tracks.jsonl"),
    "action": ("action/actions.jsonl", "action/actions.summary.json"),
    "aggregate": ("action/events.jsonl", "action/events.summary.json"),
    "identity": ("identity_raw/identity_archive_manifest.json", "identity_raw/kpr_summary.json"),
    "resolution": ("identity/identity_map.jsonl", "identity/resolution_summary.json"),
    "jersey": ("identity/jersey_summary.json", "identity/jersey_tracks.jsonl"),
    "players": ("identity/player_registration.json",),
    "link": ("action/actions_with_identity.jsonl", "action/actions_with_identity.summary.json"),
    "render": ("visualization/result.mp4",),
    "review": ("analysis/track_review/index.json",),
}


def workflow_artifacts(output):
    output = Path(output)
    result = {}
    for stage, names in STAGE_OUTPUTS.items():
        # Older runs wrote KPR directly into identity/; report their actual path.
        if stage == "identity" and not (output / names[0]).is_file():
            names = tuple(name.replace("identity_raw/", "identity/") for name in names)
        if stage == "render" and not (output / names[0]).is_file():
            names = (next((name for name in ("visualization/result_web.mp4", "result.mp4")
                           if (output / name).is_file()), names[0]),)
        files = []
        for name in names:
            path = output / name
            exists = path.is_file()
            files.append({"path": name, "exists": exists,
                          "size_bytes": path.stat().st_size if exists else None})
        result[stage] = {"files": files, "primary_exists": files[0]["exists"]}
    return {"stages": result,
            "registration": load_json(output / "identity/player_registration.json"),
            "kpr_summary": load_json(output / "identity_raw/kpr_summary.json")
                or load_json(output / "identity/kpr_summary.json")}


def _rows(path):
    if not path.is_file():
        return
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield row


def evidence_page(output, purpose, *, track_id=None, offset=0, limit=24):
    """Return one bounded page; do not load embeddings or image bytes."""
    output = Path(output)
    if purpose == "kpr":
        prefix = "identity" if (output / "identity/kpr_samples.jsonl").is_file() else "identity_raw"
        records = _rows(output / prefix / "kpr_samples.jsonl")
        tracks = list(_rows(output / prefix / "kpr_track_sampling.jsonl"))
    else:
        prefix = "identity"
        tracks = list(_rows(output / prefix / "jersey_tracks.jsonl"))
        records = (dict(reading, raw_track_id=track["raw_track_id"],
                        track_status=track.get("status"), track_number=track.get("number"))
                   for track in tracks for reading in track.get("readings", []))
    items, total = [], 0
    for row in records:
        tid = row.get("raw_track_id", row.get("track_id"))
        if track_id is not None and tid != track_id:
            continue
        if offset <= total < offset + limit:
            items.append(row)
        total += 1
    # Diagnostics stay small even when an individual track has many samples.
    statuses = [{key: row[key] for key in ("track_id", "raw_track_id", "status", "number",
                 "selected_sample_count", "observation_count", "exclusion_reasons") if key in row}
                for row in tracks if track_id is None or row.get("raw_track_id", row.get("track_id")) == track_id]
    return {"purpose": purpose, "items": items, "total": total, "offset": offset,
            "limit": limit, "media_prefix": prefix, "tracks": statuses,
            "available": (output / prefix / ("kpr_samples.jsonl" if purpose == "kpr"
                                              else "jersey_tracks.jsonl")).is_file()}
