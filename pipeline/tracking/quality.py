"""CPU-only, auditable track quality checks; never mutate tracker output."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path

from contracts.schema import write_json, write_jsonl_line
from contracts.tracks import box_iou, load_tracks


@dataclass(frozen=True)
class QualityOptions:
    tracks: Path
    video_meta: Path
    output_dir: Path
    min_observations: int = 3
    min_observed_seconds: float = 0.1
    duplicate_iou: float = 0.9
    jump_speed: float = 60.0
    suppress_duplicates: bool = False
    overwrite: bool = False


def check_track_quality(options: QualityOptions) -> dict:
    if options.min_observations < 1 or options.min_observed_seconds < 0:
        raise ValueError("Minimum support must be nonnegative; observations >= 1")
    if (
        not 0 < options.duplicate_iou <= 1
        or not math.isfinite(options.jump_speed)
        or options.jump_speed <= 0
    ):
        raise ValueError("Invalid duplicate IoU or normalized jump speed")
    if not math.isfinite(options.min_observed_seconds):
        raise ValueError("Minimum observed seconds must be finite")
    meta = json.loads(options.video_meta.read_text(encoding="utf-8"))
    grouped = load_tracks(options.tracks, meta)
    fps = float(meta["fps"])
    output = options.output_dir.resolve()
    files = [
        output / name
        for name in (
            "tracks.jsonl",
            "quality_tracks.jsonl",
            "quality_observations.jsonl",
            "quality_summary.json",
        )
    ]
    if options.tracks.resolve() in files or options.video_meta.resolve() in files:
        raise ValueError("Quality output cannot overwrite input")
    if not options.overwrite and any(path.exists() for path in files):
        raise FileExistsError(
            "Quality output exists; use a new directory or --overwrite"
        )
    by_frame = defaultdict(list)
    summaries = {}
    decisions = {}
    for track_id, rows in grouped.items():
        jumps = []
        gaps = []
        for left, right in pairwise(rows):
            delta = (right.frame_idx - left.frame_idx) / fps
            gap = max(0, right.frame_idx - left.frame_idx - 1) / fps
            if gap:
                gaps.append(gap)
            a, b = left.bbox_xyxy, right.bbox_xyxy
            distance = math.hypot(
                (a[0] + a[2] - b[0] - b[2]) / 2, (a[1] + a[3] - b[1] - b[3]) / 2
            )
            scale = max(1.0, math.hypot(a[2] - a[0], a[3] - a[1]))
            if distance / scale / delta > options.jump_speed:
                jumps.append(right.frame_idx)
        reasons = []
        if (
            len(rows) < options.min_observations
            or len(rows) / fps < options.min_observed_seconds
        ):
            reasons.append("insufficient_temporal_support")
        if jumps:
            reasons.append("abrupt_motion_review")
        status = (
            "tentative" if "insufficient_temporal_support" in reasons else "accepted"
        )
        if jumps and status == "accepted":
            status = "needs_review"
        summaries[track_id] = {
            "track_id": track_id,
            "status": status,
            "reasons": reasons,
            "observations": len(rows),
            "observed_seconds": len(rows) / fps,
            "start_frame": rows[0].frame_idx,
            "end_frame": rows[-1].frame_idx,
            "span_seconds": (rows[-1].frame_idx - rows[0].frame_idx + 1) / fps,
            "duration_s": (rows[-1].frame_idx - rows[0].frame_idx + 1) / fps,
            "observation_coverage": len(rows)
            / (rows[-1].frame_idx - rows[0].frame_idx + 1),
            "max_gap_seconds": max(gaps, default=0.0),
            "jump_frames": jumps,
            "mean_det_score": sum(r.det_score for r in rows) / len(rows),
        }
        for row in rows:
            by_frame[row.frame_idx].append(row)
            decisions[(track_id, row.frame_idx)] = {
                "track_id": track_id,
                "frame_idx": row.frame_idx,
                "kept": status in ("accepted", "needs_review"),
                "duplicate_of": None,
                "duplicate_iou": None,
            }
    duplicate_pairs = defaultdict(list)
    for frame, rows in sorted(by_frame.items()):
        # Prefer established, supported tracks, then a higher detection score.
        ranked = sorted(
            rows,
            key=lambda r: (
                summaries[r.track_id]["status"] != "accepted",
                -len(grouped[r.track_id]),
                -r.det_score,
                r.track_id,
            ),
        )
        survivors = []
        for row in ranked:
            candidates = [(box_iou(row.bbox_xyxy, s.bbox_xyxy), s) for s in survivors]
            candidates = [
                (iou, s) for iou, s in candidates if iou >= options.duplicate_iou
            ]
            if candidates:
                iou, survivor = max(candidates, key=lambda item: item[0])
                decision = decisions[(row.track_id, frame)]
                decision.update(duplicate_of=survivor.track_id, duplicate_iou=iou)
                duplicate_pairs[
                    tuple(sorted((row.track_id, survivor.track_id)))
                ].append(frame)
                if (
                    options.suppress_duplicates
                    and decision["kept"]
                    and decisions[(survivor.track_id, frame)]["kept"]
                ):
                    decision["kept"] = False
            else:
                survivors.append(row)
    output.mkdir(parents=True, exist_ok=True)
    retained = 0
    retained_ids = set()
    with (
        files[0].open("w", encoding="utf-8") as tracks_handle,
        files[2].open("w", encoding="utf-8") as audit_handle,
    ):
        for frame, rows in sorted(by_frame.items()):
            for row in sorted(rows, key=lambda r: r.track_id):
                decision = decisions[(row.track_id, frame)]
                write_jsonl_line(audit_handle, decision)
                if decision["kept"]:
                    write_jsonl_line(tracks_handle, row.to_dict())
                    retained += 1
                    retained_ids.add(row.track_id)
    with files[1].open("w", encoding="utf-8") as handle:
        for track_id, summary in summaries.items():
            summary["retained_observations"] = sum(
                decisions[(track_id, r.frame_idx)]["kept"] for r in grouped[track_id]
            )
            write_jsonl_line(handle, summary)
    summary = {
        "video_id": meta["video_id"],
        "raw_track_count": len(grouped),
        "retained_track_count": len(retained_ids),
        "retained_observations": retained,
        "tentative_track_count": sum(
            s["status"] == "tentative" for s in summaries.values()
        ),
        "review_track_count": sum(
            s["status"] == "needs_review" for s in summaries.values()
        ),
        "duplicate_candidates": [
            {"track_ids": list(pair), "frames": frames, "support": len(frames)}
            for pair, frames in sorted(duplicate_pairs.items())
        ],
        "settings": {
            k: v
            for k, v in asdict(options).items()
            if k not in ("tracks", "video_meta", "output_dir")
        },
        "warning": "Quality rules are heuristics, not accuracy or calibrated confidence. Raw files are unchanged; inspect rejected observations for false rejection.",
    }
    write_json(files[3], summary)
    return summary
