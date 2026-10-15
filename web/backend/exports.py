"""Export one currently assigned player event into the dashboard cache."""

import json
from pathlib import Path

from web.backend.imports import source_media_path, within_root
from web.backend.person_library import library_context
from web.backend.store import safe_slug
from workflows.player_events import export_player_events


def export_event(settings, store, project_id, *, player_id, event_id, expected_revision,
                 before_s=1, after_s=1, scope="event", media_source="clip"):
    context = library_context(settings, store, project_id)
    if context is None:
        raise FileNotFoundError("Player library not built")
    row = next((r for r in context["events"] if r["event_id"] == event_id), None)
    if row is None:
        raise FileNotFoundError("Unknown event")
    video = context["videos"][row["clip_name"]]
    source_media_path(video["source_path"], settings, derived=video.get("generated_source", False))
    if media_source == "original":
        source_media_path(row.get("source_path", ""), settings)
    output = within_root(store.output_root / safe_slug(project_id) / "exports", store.output_root)
    result = export_player_events(context["run"], player_id, output,
        event_ids=[event_id], expected_revision=expected_revision,
        before_s=before_s, after_s=after_s, scope=scope, media_source=media_source,
        ffmpeg=settings.ffmpeg, ffprobe=settings.ffprobe)
    if len(result["videos"]) != 1:
        raise ValueError("Event is no longer assigned to this player")
    return {"download_url": f"/api/projects/{project_id}/exports/{Path(result['videos'][0]['path']).name}",
            "revision": result["revision"], "video": result["videos"][0]}


def segmentation_page(settings, store, project_id, *, offset=0, limit=20):
    project = store.get_project(project_id)
    run = (within_root(Path(project["imported_from"]), settings.import_root) if project.get("imported_from")
           else within_root(store.output_root / safe_slug(project_id) / "videos", store.output_root))
    path = run / "batch_manifest.json"
    batch = json.loads(path.read_text()) if path.is_file() else {}
    entry = batch.get("segmentation")
    if not entry:
        return {"available": False, "items": [], "total": 0}
    plan_path = source_media_path(entry["plan_path"], settings, derived=True)
    plan = json.loads(plan_path.read_text())
    boundaries = plan.get("boundaries", [])
    return {"available": True, "status": plan["status"], "source_video_id": plan["source"]["source_video_id"],
            "duration_s": plan["source"]["duration_s"], "window_count": len(plan["windows"]),
            "clip_count": len(plan["segments"]), "unresolved_count": len(plan["unresolved"]),
            "items": boundaries[offset:offset + limit], "total": len(boundaries), "offset": offset, "limit": limit}
