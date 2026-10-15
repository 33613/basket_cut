"""Query current match identities and cut their action videos without a GPU."""

import fcntl
import hashlib
import json
import math
from pathlib import Path
from functools import wraps

from pipeline.identity.player_registry import PlayerRegistry
from pipeline.media import atomic_json, cut_video, probe_video, sha256_file
from workflows.library_inputs import safe_relative
from workflows.person_library import library_is_current


def query_player_events(run_dir, player_id, *, event="", clip_id="", min_score=0, expected_revision=None):
    run = Path(run_dir).resolve()
    if not math.isfinite(min_score) or not 0 <= min_score <= 1:
        raise ValueError("min_score must be in [0,1]")
    base = run / "library"
    if not (base / "players.sqlite3").is_file():
        raise FileNotFoundError("Match player library has not been built")
    with (base / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_SH)
        manifest = json.loads((base / "library_manifest.json").read_text())
        if not library_is_current(run, manifest):
            raise ValueError("Library source evidence changed; rebuild/check the run first")
        with PlayerRegistry(base / "players.sqlite3", manifest["match_id"]) as registry:
            snapshot = registry.snapshot()
    if expected_revision is not None and expected_revision != snapshot["revision"]:
        raise RuntimeError("Player assignments changed; refresh before exporting")
    player_id = snapshot["aliases"].get(player_id, player_id)
    if player_id not in {p["player_id"] for p in snapshot["people"]}:
        raise ValueError("Unknown player ID in this match")
    items = [row for row in snapshot["events"] if row["player_id"] == player_id
             and (not event or row["event"] == event) and (not clip_id or row["clip_id"] == clip_id)
             and row["raw_score"] >= min_score]
    items.sort(key=lambda r: (r["clip_id"], r["start"], r["event_id"]))
    return {"match_id": snapshot["match_id"], "player_id": player_id,
            "revision": snapshot["revision"], "items": items, "total": len(items),
            "time_scope": "start/end are clip seconds; source_start_s/source_end_s are original video seconds when present"}


def serialized_export(function):
    @wraps(function)
    def wrapped(run_dir, player_id, output_dir, **kwargs):
        run, output = Path(run_dir).resolve(), Path(output_dir).resolve()
        if output == run or run.is_relative_to(output):
            raise ValueError("Export directory cannot replace the run tree")
        output.mkdir(parents=True, exist_ok=True)
        with (output / ".export.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            return function(run_dir, player_id, output_dir, **kwargs)
    return wrapped


@serialized_export
def export_player_events(run_dir, player_id, output_dir, *, event="", clip_id="", min_score=0,
                         before_s=1, after_s=1, scope="event", media_source="clip",
                         expected_revision=None, event_ids=None, ffmpeg="ffmpeg", ffprobe="ffprobe"):
    if any(not math.isfinite(v) or v < 0 for v in (before_s, after_s)):
        raise ValueError("Padding must be finite and nonnegative")
    if scope not in {"event", "clip"} or media_source not in {"clip", "original"}:
        raise ValueError("Invalid export scope or media source")
    run, output = Path(run_dir).resolve(), Path(output_dir).resolve()
    if output == run or run.is_relative_to(output):
        raise ValueError("Export directory cannot replace the run tree")
    query = query_player_events(run, player_id, event=event, clip_id=clip_id,
                               min_score=min_score, expected_revision=expected_revision)
    selected = query["items"]
    if event_ids is not None:
        requested = set(event_ids)
        selected = [e for e in selected if e["event_id"] in requested]
        if requested != {e["event_id"] for e in selected}:
            raise ValueError("Requested event does not belong to this player/filter")
    output.mkdir(parents=True, exist_ok=True)
    rendered, seen, probes, hashes = [], set(), {}, {}
    for row in selected:
        key = row["clip_id"] if scope == "clip" else row["event_id"]
        if key in seen:
            continue
        seen.add(key)
        clip_root = safe_relative(run, row["clip_name"])
        meta = json.loads((clip_root / "tracking/video_meta.json").read_text())
        if media_source == "original":
            if not row.get("source_path"):
                raise ValueError("This clip has no original-video provenance; export from clip instead")
            source = Path(row["source_path"]).resolve()
            if source not in hashes:
                hashes[source] = sha256_file(source)
            if hashes[source] != row["source_sha256"]:
                raise ValueError("Original video checksum changed")
            start, end = ((row["source_clip_start_s"], row["source_clip_end_s"]) if scope == "clip"
                          else (row["source_start_s"] - before_s, row["source_end_s"] + after_s))
        else:
            source = Path(meta["input_path"]).resolve()
            start, end = (0, None) if scope == "clip" else (row["start"] - before_s, row["end"] + after_s)
        if source not in probes:
            probes[source] = probe_video(source, ffprobe=ffprobe)
        media = probes[source]
        start, end = max(0, start), min(media["duration_s"], media["duration_s"] if end is None else end)
        if source not in hashes:
            hashes[source] = sha256_file(source)
        if media_source == "clip" and row.get("clip_sha256") and hashes[source] != row["clip_sha256"]:
            raise ValueError("Clip video changed after identity/event registration")
        record = {"player_id": query["player_id"], "match_id": query["match_id"],
                  "revision": query["revision"], "scope": scope, "clip_id": row["clip_id"],
                  "event_id": None if scope == "clip" else row["event_id"],
                  "media_source": media_source, "input": str(source), "input_sha256": hashes[source],
                  "start_s": start, "end_s": end, "encoding": "h264_cfr_crf18_v1"}
        fingerprint = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
        path = output / (fingerprint[:24] + ".mp4")
        sidecar = path.with_suffix(".json")
        old = json.loads(sidecar.read_text()) if sidecar.is_file() else {}
        if not (old.get("request") == record and path.is_file() and sha256_file(path) == old.get("sha256")):
            if path.exists():
                raise ValueError("Existing exported video changed")
            cut_video(source, path, start, end, metadata=media, ffmpeg=ffmpeg, ffprobe=ffprobe)
            atomic_json(sidecar, {"request": record, "sha256": sha256_file(path)})
        rendered.append({**record, "path": str(path), "sha256": sha256_file(path)})
    result = {"match_id": query["match_id"], "player_id": query["player_id"],
              "revision": query["revision"], "exported_videos": len(rendered), "videos": rendered}
    atomic_json(output / "export_manifest.json", result)
    return result
