"""Read-only match library browsing and revision-checked review sidecars."""

from __future__ import annotations

import json
from pathlib import Path
import threading
from urllib.parse import quote

from contracts.schema import read_jsonl
from web.backend.imports import within_root
from web.backend.store import safe_slug
from workflows.person_library import export_library, library_is_current
from pipeline.identity.player_registry import PlayerRegistry
import fcntl

_lock = threading.RLock()


def library_context(settings, store, project_id):
    project = store.get_project(project_id)
    source = project.get("imported_from")
    if source:
        run = within_root(Path(source), settings.import_root)
    else:
        run = within_root(store.output_root / safe_slug(project_id) / "videos", store.output_root)
    base = within_root(run / "library", run)
    manifest_path = base / "library_manifest.json"
    if not manifest_path.is_file():
        return None
    # Cooperate with CLI writers so a multi-file export is read as one revision.
    with (base / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_SH)
        baseline = json.loads(manifest_path.read_text())
        if baseline.get("scope") != "match_incremental" or not library_is_current(run, baseline):
            raise ValueError("人物库证据已变更，请重新检查处理结果")
        current = within_root(base / baseline["index_dir"], base)
        people = list(read_jsonl(within_root(current / "people.jsonl", current)))
        events = list(read_jsonl(within_root(current / "events.jsonl", current)))
    manifest = baseline
    review = {"revision": baseline.get("review_revision")}
    review_path, revised, stale = None, None, False
    videos = {Path(video["output_dir"]).name: video for video in project["videos"]}
    for person in people:
        for member in person["members"]:
            video = videos.get(member["clip_name"])
            if not video or Path(video["output_dir"]).resolve() != (run / member["clip_name"]).resolve():
                raise ValueError("Person library references a clip outside this imported project")
    return {"run": run, "base": base, "directory": current, "manifest": manifest,
            "baseline": baseline, "people": people, "events": events, "videos": videos, "review": review,
            "review_path": review_path, "revised": revised, "stale_review": stale}


def decorate_member(project_id, context, member):
    video = context["videos"][member["clip_name"]]
    def media(sample, kind="identity"):
        result = dict(sample)
        media_root = Path(video["output_dir"]) / kind
        for key in ("crop_path", "context_path"):
            if result.get(key):
                within_root(media_root / result[key], media_root)
                result[key.replace("_path", "_url")] = (
                    f'/api/projects/{project_id}/videos/{video["video_id"]}/artifacts/{kind}/'
                    + quote(result[key], safe="/"))
        return result

    jersey = member.get("jersey")
    if jersey:
        jersey = {**jersey, "tracks": [
            {**track, "readings": [media(reading, "identity") for reading in track.get("readings", [])],
             "candidates": [{**candidate, "evidence": [media(reading, "identity") for reading in candidate.get("evidence", [])]}
                            for candidate in track.get("candidates", [])]}
            for track in jersey.get("tracks", [])]}
    return {**member, "web_video_id": video["video_id"], "cover": media(member.get("cover") or {}),
            "jersey": jersey,
            "exemplars": [media(sample) for sample in member.get("exemplars", [])]}


def library_catalog(settings, store, project_id, *, offset=0, limit=12, q="", status="all"):
    context = library_context(settings, store, project_id)
    if context is None:
        return {"available": False, "items": [], "total": 0, "offset": offset, "limit": limit,
                "message": "尚未建立跨片段人物库。逐个处理片段时自动更新；也可运行 cli.build_person_library。"}
    people = sorted(context["people"], key=lambda p: (-p["clip_count"], -p["event_count"], p["global_person_id"]))
    selected = []
    for person in people:
        text = " ".join([person["global_person_id"], person.get("identity_label") or "",
                         *((person.get("jersey") or {}).get("candidate_numbers", [])),
                         *(member["filename"] for member in person["members"])])
        matches = {"all": person["status"] != "non_player", "non_player": person["status"] == "non_player", "merged": person["local_archive_count"] > 1,
                   "singleton": person["local_archive_count"] == 1,
                   "needs_review": person["status"] == "needs_review",
                   "manual_grouping": person["status"] == "manual_grouping"}
        if matches.get(status, False) and (not q or q.casefold() in text.casefold()):
            selected.append(person)
    rows = []
    for person in selected[offset:offset + limit]:
        cover = next(member for member in person["members"] if member["node_id"] == person["cover_node_id"])
        rows.append({**{key: value for key, value in person.items() if key != "members"},
                     "cover": decorate_member(project_id, context, cover)["cover"],
                     "previews": [decorate_member(project_id, context, member)["cover"]
                                  for member in person["members"][:4]]})
    summary = {key: value for key, value in context["manifest"].items()
               if key not in {"source_artifacts", "merges", "feature_contract"}}
    return {"available": True, "items": rows, "offset": offset, "limit": limit,
            "total": len(selected), "summary": summary, "stale_review": context["stale_review"],
            "revision": context["review"].get("revision")}


def library_person(settings, store, project_id, person_id):
    context = library_context(settings, store, project_id)
    if context is None:
        raise FileNotFoundError("Person library not built")
    person_id = context["manifest"].get("player_aliases", {}).get(person_id, person_id)
    person = next((p for p in context["people"] if p["global_person_id"] == person_id), None)
    if person is None:
        raise FileNotFoundError("Unknown global person")
    nodes = {n["node_id"]: n for p in context["people"] for n in p["members"]}
    node_ids = {n["node_id"] for n in person["members"]}
    pairs = [pair for pair in read_jsonl(context["directory"] / "match_pairs.jsonl")
             if pair["node_a"] in node_ids or pair["node_b"] in node_ids]
    pairs.sort(key=lambda pair: (pair["same_global_person"], pair["distance"] if pair["distance"] is not None else 999))
    return {**person, "members": [decorate_member(project_id, context, member) for member in person["members"]],
            "pairs": [{**pair, "left": decorate_member(project_id, context, nodes[pair["node_a"]]),
                       "right": decorate_member(project_id, context, nodes[pair["node_b"]])}
                      for pair in pairs[:24]], "revision": context["review"].get("revision")}


def library_events(settings, store, project_id, *, person_id="", event="", q="", min_score=0,
                   offset=0, limit=20):
    context = library_context(settings, store, project_id)
    if context is None:
        return {"items": [], "total": 0, "offset": offset, "limit": limit}
    selected = []
    person_id = context["manifest"].get("player_aliases", {}).get(person_id, person_id)
    for row in context["events"]:
        video = context["videos"].get(row["clip_name"])
        if video is None:
            raise ValueError("Event references a video outside this project")
        if person_id and row["global_person_id"] != person_id:
            continue
        if event and row["event"] != event:
            continue
        if float(row["raw_score"]) < min_score:
            continue
        if q and q.casefold() not in video["filename"].casefold():
            continue
        selected.append({**row, "web_video_id": video["video_id"], "filename": video["filename"]})
    selected.sort(key=lambda row: (row["filename"], row["start"], row["event_id"]))
    return {"items": selected[offset:offset + limit], "total": len(selected), "offset": offset, "limit": limit,
            "time_scope": "source_clip_seconds", "warning": "Events are cached model predictions, not verified actions"}


def edit_library(settings, store, project_id, operation, *, expected_revision=None,
                 person_id=None, node_id=None, label=None):
    with _lock:
        context = library_context(settings, store, project_id)
        if context is None:
            raise FileNotFoundError("Player library not built")
        with (context["base"] / ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            with PlayerRegistry(context["base"] / "players.sqlite3", context["manifest"]["match_id"]) as registry:
                revision = registry.review(operation, expected_revision=expected_revision,
                    person_id=person_id, node_id=node_id, label=label)
                summary = export_library(registry, context["run"], context["base"])
        return {"revision": revision, "summary": summary}
