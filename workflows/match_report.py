"""Live workflow counts and explicit, provenance-checked evaluation availability."""

import json
from pathlib import Path


def load(path, default=None):
    return json.loads(path.read_text()) if path.is_file() else default


def match_report(match_root):
    root = Path(match_root).resolve()
    results = root / "results" if (root / "results").is_dir() else root
    batch = load(results / "batch_manifest.json", {})
    segmentation = root / "segmentation"
    entry = batch.get("segmentation")
    if entry:
        candidate = Path(entry["plan_path"]).resolve()
        if not candidate.is_relative_to(root):
            # Imported CLI projects point at their sibling segmentation tree.
            if not candidate.is_relative_to(root.parent):
                raise ValueError("Segmentation evidence lies outside the match result tree")
        segmentation = candidate.parent
    plan = load(segmentation / "segmentation.json", {})
    export = load(segmentation / "clips_manifest.json", {})
    library = load(results / "library/library_manifest.json", {})
    seed = load(results / "bootstrap/registration.json", {})
    rows = []
    for clip in batch.get("clips", []):
        name = clip["name"]
        path = (results / name).resolve()
        if not path.is_relative_to(results) or Path(name).name != name:
            raise ValueError("Unsafe clip in result manifest")
        state = load(path / "batch_status.json", {})
        quality = load(path / "quality/quality_summary.json", {})
        kpr = load(path / "identity_raw/kpr_summary.json", {})
        jersey = load(path / "identity/jersey_summary.json", {})
        meta = load(path / "tracking/video_meta.json", {})
        rows.append({"clip_id": clip.get("clip_id", name), "status": state.get("status", "not_started"),
            "stages": state.get("stages", {}), "stage_metrics": state.get("stage_metrics", {}),
            "processed_frames": meta.get("processed_frames"), "quality": quality,
            "kpr_sample_count": kpr.get("sample_count"), "kpr_track_count": kpr.get("track_count"),
            "number_samples": jersey.get("sample_count"), "number_inferences": jersey.get("inference_count"),
            "number_cache_hits": jersey.get("cache_hit_count"), "error": state.get("error")})
    evaluation = load(segmentation / "evaluation.json")
    if evaluation and (evaluation.get("source_sha256") != plan.get("source", {}).get("sha256")
            or evaluation.get("protocol_sha256") != plan.get("protocol_sha256")):
        evaluation = {"status": "stale_evaluation", "accuracy_evaluated": False}
    photos = load(results / "bootstrap/photos.json", {})
    expected = len(export.get("videos", [])) if export else len(rows)
    completed = sum(r["status"] == "completed" for r in rows)
    failed = sum(r["status"] == "failed" for r in rows)
    status = ("failed" if failed else "completed" if expected and completed == expected else
              "partial" if completed else "initialized" if seed and not expected else "incomplete")
    return {"schema_version": 1, "match_id": batch.get("match_id", seed.get("match_id")),
        "status": status,
        "source": plan.get("source", photos.get("protocol", {}).get("source")), "segmentation": {"status": load(segmentation / "status.json", {}),
            "boundary_convention": plan.get("boundary_convention"), "planned_clips": len(plan.get("segments", [])),
            "exported_clips": len(export.get("videos", [])), "unresolved_boundaries": len(plan.get("unresolved", [])),
            "coarse_candidates": len(plan.get("coarse_candidates", [])),
            "model_boundaries": len(plan.get("boundaries", [])), "manually_reviewed": export.get("manually_reviewed", False)},
        "bootstrap": seed, "planned_processing_clips": expected, "selected_processing_clips": len(rows),
        "completed_clips": completed, "failed_clips": failed, "clips": rows,
        "library": {k: library.get(k) for k in ("global_person_count", "bootstrap_photo_count", "event_count",
            "unmapped_event_count", "held_archive_count", "review_revision", "candidate_person_count",
            "memory_status_counts", "gallery_template_count", "memory_policy")},
        "possession_evaluation": evaluation or {"status": "not_annotated", "accuracy_evaluated": False},
        "tracking_identity_action_accuracy": {"status": "not_annotated", "accuracy_evaluated": False},
        "warning": "Workflow counts and successful execution are not detection, identity or action accuracy."}
