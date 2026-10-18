"""Explicit scopes: source video -> clip -> local track -> match player."""

import math
import hashlib
import json


def clip_id_for(source_id, start_s, end_s):
    value = [source_id, round(float(start_s), 9), round(float(end_s), 9)]
    return "CL-" + hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()[:20]


def clip_context(batch, clip_name):
    rows = [r for r in batch["clips"] if r["name"] == clip_name]
    if len(rows) != 1:
        raise ValueError("Clip must occur exactly once in the match manifest")
    row = rows[0]
    result = {"clip_id": row.get("clip_id", clip_name)}
    if row.get("lineage"):
        lineage = row["lineage"]
        start, end = float(lineage["start_s"]), float(lineage["end_s"])
        if not math.isfinite(start) or not math.isfinite(end) or not 0 <= start < end:
            raise ValueError("Invalid clip source interval")
        result.update({"source_video_id": lineage["source_video_id"],
                       "source_path": lineage["source_path"], "source_sha256": lineage["source_sha256"],
                       "source_clip_start_s": start, "source_clip_end_s": end,
                       "segmentation_needs_review": bool(lineage.get("needs_review", True))})
        result.update({k:lineage[k] for k in ("possession_id", "shot_id", "content_type",
                       "partial_possession_start", "partial_possession_end") if k in lineage})
        if lineage.get("sha256"):
            result["clip_sha256"] = lineage["sha256"]
        result["timestamp_quantization_s"] = float(lineage.get("encoded", {}).get("timestamp_quantization_s", .15))
    return result


def track_uid(clip_id, raw_track_id):
    tid = int(raw_track_id)
    if tid < 0:
        raise ValueError("Negative track ID")
    return f"{clip_id}:T{tid:04d}"


def event_lineage(context, start, end):
    result = dict(context)
    if "source_clip_start_s" in context:
        duration = context["source_clip_end_s"] - context["source_clip_start_s"]
        # CFR encoding can round up by one frame; clamp only to the real cut.
        if start >= duration or end > duration + context.get("timestamp_quantization_s", .15) + 1e-6:
            raise ValueError("Event falls outside its source clip interval")
        result.update(source_start_s=context["source_clip_start_s"] + start,
                      source_end_s=context["source_clip_start_s"] + min(end, duration))
    return result
