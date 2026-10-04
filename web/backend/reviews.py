"""Persistent audit sidecars; imported model artifacts remain read-only."""

import io
import json
import threading
import uuid
import zipfile
from pathlib import Path

from analysis.evaluation.track_review import (
    evaluate_review, merge_review, review_index, summarize_reviews, validate_review,
)
from web.backend.artifacts import load_json

_lock = threading.RLock()


def review_paths(video: dict) -> dict:
    output = Path(video["output_dir"])
    return {"tracks": output / "tracking/tracks.jsonl",
            "video_meta": output / "tracking/video_meta.json",
            "quality": output / "quality/quality_tracks.jsonl",
            "identity_map": output / "identity/identity_map.jsonl",
            "evidence": output / "analysis/track_review/index.json"}


def sidecar(store, project_id: str, video_id: str) -> Path:
    # IDs are looked up, not accepted as filesystem paths.
    store.get_video(project_id, video_id)
    return store.output_root / project_id / "reviews" / f"{video_id}.json"


def collect_review(store, project_id: str, video: dict) -> dict:
    paths = review_paths(video)
    index = review_index(paths["tracks"], paths["video_meta"], paths["quality"])
    evidence = load_json(paths["evidence"])
    if evidence and evidence.get("fingerprint") == index["fingerprint"]:
        samples = {row["raw_track_id"]: row.get("samples", []) for row in evidence["tracks"]}
        for row in index["tracks"]:
            row["samples"] = samples.get(row["raw_track_id"], [])
        index["evidence_warnings"] = evidence.get("warnings", [])
    path = sidecar(store, project_id, video["video_id"])
    saved = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
    stale = bool(saved and saved.get("fingerprint") != index["fingerprint"])
    review = validate_review(index, None if stale else saved)
    metrics = evaluate_review(index, review,
                              paths["identity_map"] if paths["identity_map"].is_file() else None)
    return {"index": index, "review": review, "metrics": metrics,
            "stale_review": stale, "revision": saved.get("revision") if saved else None,
            "warning": "Review is a separate persistent sidecar. It does not alter tracks, archives or events."}


def save_review(store, project_id: str, video: dict, value: dict, expected_revision: str | None) -> dict:
    with _lock:
        current = collect_review(store, project_id, video)
        if expected_revision != current["revision"]:
            raise RuntimeError("Review changed in another window; reload before saving")
        clean = validate_review(current["index"], value)
        path = sidecar(store, project_id, video["video_id"])
        path.parent.mkdir(parents=True, exist_ok=True)
        clean["revision"] = uuid.uuid4().hex
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(clean, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
        return collect_review(store, project_id, video)


def project_reviews(store, manifest: dict) -> dict:
    results, failed = [], []
    for video in manifest["videos"]:
        try:
            result = collect_review(store, manifest["project_id"], video)
            results.append({"name": video["filename"], **result["metrics"],
                            "stale_review": result["stale_review"]})
        except (OSError, ValueError, KeyError, TypeError) as exc:
            failed.append({"name": video["filename"], "error": str(exc)})
    return {**summarize_reviews(results, expected_clips=len(manifest["videos"]), failed=failed),
            "clips": results,
            "processing_statuses": {video["video_id"]: video["status"] for video in manifest["videos"]}}


def review_bundle(store, manifest: dict) -> bytes:
    """Export a portable frozen audit manifest, not deployment paths or videos."""
    buffer, clips = io.BytesIO(), []
    total = 0
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        def add(name, data):
            nonlocal total
            if isinstance(data, dict):
                data = json.dumps(data, ensure_ascii=False, indent=2).encode()
            total += len(data)
            if total > 128 * 1024 * 1024:
                raise ValueError("Review bundle exceeds 128 MiB; export smaller projects")
            archive.writestr(name, data)
        for number, video in enumerate(manifest["videos"], 1):
            # Fail on missing raw results rather than silently exporting fewer clips.
            data = collect_review(store, manifest["project_id"], video)
            prefix = f"clips/{number:03d}"
            paths = review_paths(video)
            entry = {"name": f"{number:03d}: {video['filename']}"}
            for key in ("tracks", "quality", "identity_map"):
                if paths[key].is_file():
                    entry[key] = f"{prefix}/{key}.jsonl"
                    add(entry[key], paths[key].read_bytes())
            # Retain only the clock fields used by the fingerprint.
            meta = json.loads(paths["video_meta"].read_text(encoding="utf-8"))
            entry["video_meta"] = f"{prefix}/video_meta.json"
            add(entry["video_meta"], {k: meta[k] for k in (
                "video_id", "fps", "processed_frames", "frame_count", "width", "height"
            ) if k in meta})
            entry["review"] = f"{prefix}/review.json"
            add(entry["review"], data["review"])
            add(f"{prefix}/merge_review.json", merge_review(data["index"], data["review"]))
            clips.append(entry)
        add("manifest.json", {"schema_version": 1, "clips": clips})
        add("track_review_metrics.json", project_reviews(store, manifest))
    return buffer.getvalue()
