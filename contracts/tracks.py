"""Framework-independent track geometry and strict input validation."""

import math
from collections import defaultdict
from pathlib import Path

from contracts.schema import TrackRecord, read_jsonl


def box_iou(a, b) -> float:
    intersection = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0.0, min(a[3], b[3]) - max(a[1], b[1])
    )
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0.0


def load_tracks(path: Path, meta: dict) -> dict[int, list[TrackRecord]]:
    """Reject mixed videos, duplicate observations, wrong clocks and bounds."""
    fps = float(meta["fps"])
    if not math.isfinite(fps) or fps <= 0 or not meta.get("video_id"):
        raise ValueError("video_meta requires a video_id and positive finite fps")
    grouped = defaultdict(list)
    seen = set()
    frame_count = int(meta.get("processed_frames", meta.get("frame_count", 0)))
    if frame_count < 0:
        raise ValueError("Negative processed frame count")
    for value in read_jsonl(path):
        record = TrackRecord.from_dict(value)
        key = (record.track_id, record.frame_idx)
        if record.video_id != meta["video_id"]:
            raise ValueError("Track video_id does not match video_meta")
        if key in seen:
            raise ValueError(f"Duplicate track/frame: {key}")
        if (
            "processed_frames" in meta or frame_count
        ) and record.frame_idx >= frame_count:
            raise ValueError(f"Track frame outside processed range: {key}")
        if (
            not math.isfinite(record.timestamp_s)
            or abs(record.timestamp_s - record.frame_idx / fps) > 1 / fps
        ):
            raise ValueError(f"Track timestamp does not match frame clock: {key}")
        seen.add(key)
        grouped[record.track_id].append(record)
    return {
        key: sorted(rows, key=lambda r: r.frame_idx)
        for key, rows in sorted(grouped.items())
    }
