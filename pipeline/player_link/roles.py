"""Associate cached RF-DETR holder/scorer boxes with observed MOTIP boxes.

These are correlated detector cues, not independent votes or player identities.
Duplicate detections in one source frame cannot create temporal support.
"""

import math
from collections import defaultdict

from contracts.tracks import box_iou

ROLE_POLICY = {
    "score_threshold": 0.35,
    "min_iou": 0.5,
    "min_iou_margin": 0.1,
    "min_frames": 3,
    "min_span_s": 0.05,
    "max_gap_s": 0.1,
}


def match_roles(ball_rows, by_frame, event, fps, eligible_tracks, lookback_s=3.0):
    matched = {}
    start = event["goal_time_s"] - lookback_s
    for row in ball_rows:
        time = row["source_timestamp_s"]
        if not math.isfinite(time) or abs(time - row["source_frame_idx"] / fps) > 1e-6:
            raise ValueError("Role source frame clock mismatch")
        if not start <= time <= event["goal_time_s"]:
            continue
        frame = row["source_frame_idx"]
        local = frame - event["window_start_frame"]
        people = [
            r
            for r in by_frame.get(local, [])
            if r.track_id in eligible_tracks and r.det_score >= 0.3
        ]
        for detection in row.get("detections", []):
            label = detection["label"]
            if label not in ("holder", "scorer"):
                continue
            score = float(detection["score"])
            box = detection["bbox_xyxy"]
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("Invalid role detector confidence")
            if score < ROLE_POLICY["score_threshold"]:
                continue
            matches = sorted(
                ((box_iou(box, r.bbox_xyxy), r.track_id) for r in people), reverse=True
            )
            if not matches or matches[0][0] < ROLE_POLICY["min_iou"]:
                continue
            margin = matches[0][0] - (matches[1][0] if len(matches) > 1 else 0)
            if margin < ROLE_POLICY["min_iou_margin"]:
                continue
            iou, tid = matches[0]
            key = (tid, label, frame)
            value = {
                "source_frame_idx": frame,
                "source_time_s": time,
                "score": score,
                "person_iou": iou,
                "person_iou_margin": margin,
            }
            if key not in matched or score > matched[key]["score"]:
                matched[key] = value
    indexed = defaultdict(list)
    for (tid, label, _), value in matched.items():
        indexed[(tid, label)].append(value)
    return {
        key: sorted(values, key=lambda v: v["source_frame_idx"])
        for key, values in indexed.items()
    }


def role_support(index, tid, start_s, end_s):
    result = {}
    for label in ("holder", "scorer"):
        points = [
            p
            for p in index.get((tid, label), [])
            if start_s <= p["source_time_s"] <= end_s
        ]
        runs = []
        for p in points:
            if (
                not runs
                or p["source_time_s"] - runs[-1][-1]["source_time_s"]
                > ROLE_POLICY["max_gap_s"] + 1e-6
            ):
                runs.append([])
            runs[-1].append(p)
        supported = [
            run
            for run in runs
            if len(run) >= ROLE_POLICY["min_frames"]
            and run[-1]["source_time_s"] - run[0]["source_time_s"]
            >= ROLE_POLICY["min_span_s"] - 1e-6
        ]
        best = max(
            supported,
            key=lambda run: (sum(p["score"] for p in run), len(run)),
            default=[],
        )
        result[label] = {
            "supported": bool(best),
            "matched_frames": len(points),
            "support_frames": len(best),
            "mean_score": sum(p["score"] for p in best) / len(best) if best else None,
            "interval_source_s": [best[0]["source_time_s"], best[-1]["source_time_s"]]
            if best
            else None,
            "observations": best,
        }
    return result
