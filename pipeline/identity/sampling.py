"""Bounded CPU crop selection shared by ReID and jersey reading.

Scores are image-quality heuristics, never identity or number confidence.
No pose/occlusion model is required; box overlap is only a soft penalty.
"""

import math
from collections import defaultdict

from contracts.schema import TrackRecord

SAMPLING_STRATEGY = "quality_v1"


def crop_region(box, frame_shape, purpose):
    height, width = frame_shape[:2]
    x1, y1, x2, y2 = box
    if purpose == "jersey":
        box_width, box_height = x2 - x1, y2 - y1
        x1, x2 = x1 + .08 * box_width, x2 - .08 * box_width
        y1, y2 = y1 + .12 * box_height, y1 + .70 * box_height
    expected_area = (x2 - x1) * (y2 - y1)
    roi = (max(0, math.floor(x1)), max(0, math.floor(y1)),
           min(width, math.ceil(x2)), min(height, math.ceil(y2)))
    area = max(0, roi[2] - roi[0]) * max(0, roi[3] - roi[1])
    return roi, min(1., area / expected_area)


def _overlap_fraction(roi, record, others):
    area = (roi[2] - roi[0]) * (roi[3] - roi[1])
    largest = 0.
    for other in others:
        if other.track_id == record.track_id:
            continue
        box = other.bbox_xyxy
        intersection = max(0., min(roi[2], box[2]) - max(roi[0], box[0])) * max(
            0., min(roi[3], box[3]) - max(roi[1], box[1]))
        largest = max(largest, intersection / area)
    return largest


def _stability(record, previous):
    if previous is None:
        return 1.
    a, b = previous.bbox_xyxy, record.bbox_xyxy
    scale = max(1., math.hypot(a[2] - a[0], a[3] - a[1]))
    step = math.hypot((b[0] + b[2] - a[0] - a[2]) / 2,
                      (b[1] + b[3] - a[1] - a[3]) / 2) / scale
    # A long observation gap is not treated as a one-frame jump.
    step /= max(1, record.frame_idx - previous.frame_idx)
    size_change = abs(math.log(
        ((b[2] - b[0]) * (b[3] - b[1])) /
        ((a[2] - a[0]) * (a[3] - a[1]))
    )) / max(1, record.frame_idx - previous.frame_idx)
    return 1. / (1. + 4. * step + size_change)


def select_quality_samples(video, records: list[TrackRecord], *, purpose, sample_count,
                           min_score=.5, min_width=16., min_height=40.,
                           min_gap_s=.25, track_ids=None):
    """Return selected TrackRecords and per-frame quality metadata.

    Decode sequentially, score all eligible observations, keep one candidate
    per time bucket and at most 8 x sample_count candidates per track. Retained
    candidates contain tiny grayscale thumbnails, not full-resolution crops.
    """
    import cv2
    import numpy as np

    if purpose not in {"person", "jersey"}:
        raise ValueError("Unknown crop sampling purpose")
    if sample_count < 1 or not math.isfinite(min_gap_s) or min_gap_s <= 0:
        raise ValueError("Invalid sampling count or temporal gap")
    if not math.isfinite(min_score) or not 0 <= min_score <= 1:
        raise ValueError("Invalid detection score threshold")
    if any(not math.isfinite(x) or x <= 0 for x in (min_width, min_height)):
        raise ValueError("Invalid minimum crop size")

    by_frame = defaultdict(list)
    previous, stability = {}, {}
    seen, video_ids = set(), set()
    for record in sorted(records, key=lambda r: (r.frame_idx, r.track_id)):
        key = (record.track_id, record.frame_idx)
        if key in seen or not math.isfinite(record.timestamp_s):
            raise ValueError("Duplicate observation or nonfinite timestamp")
        seen.add(key)
        video_ids.add(record.video_id)
        by_frame[record.frame_idx].append(record)
        stability[key] = _stability(record, previous.get(record.track_id))
        previous[record.track_id] = record
    if len(video_ids) > 1:
        raise ValueError("Sampling tracks contain multiple videos")
    eligible_frames = {
        frame for frame, rows in by_frame.items()
        if any((track_ids is None or r.track_id in track_ids)
               and r.det_score >= min_score
               and r.bbox_xyxy[2] - r.bbox_xyxy[0] >= min_width
               and r.bbox_xyxy[3] - r.bbox_xyxy[1] >= min_height for r in rows)
    }
    if not eligible_frames:
        return [], {}

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"Cannot decode crop-sampling video: {video}")
    candidates = defaultdict(dict)
    try:
        for frame_idx in range(max(eligible_frames) + 1):
            ok, image = capture.read()
            if not ok:
                raise ValueError("Video ended before candidate tracking frames")
            if frame_idx not in eligible_frames:
                continue
            rows = by_frame[frame_idx]
            for record in rows:
                if track_ids is not None and record.track_id not in track_ids:
                    continue
                box = record.bbox_xyxy
                if (record.det_score < min_score or box[2] - box[0] < min_width
                        or box[3] - box[1] < min_height):
                    continue
                roi, completeness = crop_region(box, image.shape, purpose)
                left, top, right, bottom = roi
                required = (32, 48) if purpose == "jersey" else (min_width, min_height)
                if (right - left < required[0] or bottom - top < required[1]
                        or completeness < .5):
                    continue
                gray = cv2.cvtColor(image[top:bottom, left:right], cv2.COLOR_BGR2GRAY)
                # Downscale large crops so size and sharpness do not double-count
                # resolution. Never upscale small crops for quality scoring.
                scale = min(1., 128 / gray.shape[1], 192 / gray.shape[0])
                normalized = cv2.resize(gray, (max(1, round(gray.shape[1] * scale)),
                                              max(1, round(gray.shape[0] * scale))),
                                        interpolation=cv2.INTER_AREA)
                sharpness = float(cv2.Laplacian(normalized, cv2.CV_64F).var())
                exposure = float(np.mean((gray > 5) & (gray < 250)))
                if sharpness < 5 or exposure < .1:
                    continue
                overlap = _overlap_fraction(roi, record, rows)
                components = {
                    "sharpness": min(1., math.log1p(sharpness) / math.log1p(1000.)),
                    "size": min(1., math.sqrt(gray.size / (128 * 192))),
                    "detection": record.det_score,
                    "completeness": completeness,
                    "exposure": exposure,
                    "stability": stability[(record.track_id, frame_idx)],
                    "overlap_fraction": overlap,
                }
                quality = (.30 * components["sharpness"] + .25 * components["size"]
                           + .15 * record.det_score + .15 * completeness
                           + .10 * exposure + .05 * components["stability"])
                quality *= 1. - .5 * overlap
                info = {"sampling_strategy": SAMPLING_STRATEGY,
                        "sampling_quality": quality, "sampling_components": components,
                        "sampling_roi_xyxy": list(roi),
                        "sampling_sharpness_laplacian_var": sharpness}
                thumbnail = cv2.resize(gray, (16, 24), interpolation=cv2.INTER_AREA)
                bucket = math.floor(record.timestamp_s / min_gap_s)
                pool = candidates[record.track_id]
                candidate = (record, info, thumbnail.astype(np.float32) / 255.)
                if bucket not in pool or quality > pool[bucket][1]["sampling_quality"]:
                    pool[bucket] = candidate
                if len(pool) > sample_count * 8:
                    worst = min(pool, key=lambda k: (pool[k][1]["sampling_quality"],
                                                     -pool[k][0].frame_idx))
                    del pool[worst]
    finally:
        capture.release()

    selected, metadata = [], {}
    for tid, buckets in sorted(candidates.items()):
        remaining, chosen = list(buckets.values()), []
        while remaining and len(chosen) < sample_count:
            def rank(candidate):
                # Prefer visual variety among similarly good frames. Similarity
                # is a soft penalty, so a stationary player can still contribute.
                similarity = max((1. - min(1., float(np.mean(np.abs(
                    candidate[2] - old[2]))) / .15) for old in chosen), default=0.)
                return (candidate[1]["sampling_quality"] * (1. - .15 * similarity),
                        -candidate[0].frame_idx)
            best = max(remaining, key=rank)
            chosen.append(best)
            remaining = [c for c in remaining if
                         abs(c[0].timestamp_s - best[0].timestamp_s) >= min_gap_s]
        for record, info, _ in chosen:
            selected.append(record)
            metadata[(tid, record.frame_idx)] = info
    return sorted(selected, key=lambda r: (r.frame_idx, r.track_id)), metadata
