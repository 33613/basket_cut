"""Core MOTIP tracking service with no command-line or visualization concerns."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
from tqdm import tqdm

from contracts.schema import (
    TrackRecord,
    VideoMeta,
    normalize_xyxy,
    write_json,
    write_jsonl_line,
)
from pipeline.tracking.motip_backend import MotipBackend, MotipRuntimeConfig


@dataclass(frozen=True)
class TrackingOptions:
    input: Path
    checkpoint: Path
    output_dir: Path
    motip_root: Path
    config: str = "configs/r50_deformable_detr_motip_sportsmot.yaml"
    device: str = "cuda:0"
    assignment_protocol: str = "object-max"
    miss_tolerance: int = 60
    det_thresh: float = 0.3
    newborn_thresh: float = 0.6
    id_thresh: float = 0.2
    area_thresh: int = 0
    max_shorter: int = 800
    max_longer: int = 1440
    fp32: bool = False
    max_frames: int | None = None
    overwrite: bool = False


def video_id_for(path: Path) -> str:
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:10]
    return f"{path.stem}-{digest}"


def update_summary(
    summaries: dict[int, dict[str, Any]], record: TrackRecord
) -> None:
    item = summaries.setdefault(
        record.track_id,
        {
            "track_id": record.track_id,
            "start_frame": record.frame_idx,
            "end_frame": record.frame_idx,
            "observations": 0,
            "score_sum": 0.0,
            "min_det_score": record.det_score,
            "max_det_score": record.det_score,
        },
    )
    item["start_frame"] = min(item["start_frame"], record.frame_idx)
    item["end_frame"] = max(item["end_frame"], record.frame_idx)
    item["observations"] += 1
    item["score_sum"] += record.det_score
    item["min_det_score"] = min(item["min_det_score"], record.det_score)
    item["max_det_score"] = max(item["max_det_score"], record.det_score)


def finalize_summary(
    summaries: dict[int, dict[str, Any]], fps: float
) -> dict[str, Any]:
    tracks = []
    for track_id in sorted(summaries):
        item = dict(summaries[track_id])
        span_frames = item["end_frame"] - item["start_frame"] + 1
        item["mean_det_score"] = item.pop("score_sum") / item["observations"]
        item["span_frames"] = span_frames
        item["duration_s"] = span_frames / fps
        item["observation_coverage"] = item["observations"] / span_frames
        tracks.append(item)
    return {"track_count": len(tracks), "tracks": tracks}


def track_video(options: TrackingOptions) -> dict[str, Any]:
    input_path = options.input.expanduser().resolve()
    output_dir = options.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tracks_path = output_dir / "tracks.jsonl"
    if tracks_path.exists() and not options.overwrite:
        raise FileExistsError(
            f"{tracks_path} already exists; pass --overwrite to replace this run"
        )

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open input video: {input_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    declared_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_limit = declared_frames
    if options.max_frames is not None:
        frame_limit = (
            min(frame_limit, options.max_frames)
            if frame_limit
            else options.max_frames
        )

    runtime_config = MotipRuntimeConfig(
        assignment_protocol=options.assignment_protocol,
        miss_tolerance=options.miss_tolerance,
        det_thresh=options.det_thresh,
        newborn_thresh=options.newborn_thresh,
        id_thresh=options.id_thresh,
        area_thresh=options.area_thresh,
        max_shorter=options.max_shorter,
        max_longer=options.max_longer,
        fp16=not options.fp32,
    )
    backend = MotipBackend(
        motip_root=options.motip_root,
        config_path=options.config,
        checkpoint_path=options.checkpoint,
        runtime_config=runtime_config,
        device=options.device,
    )
    tracker = backend.create_tracker(height=height, width=width)
    video_id = video_id_for(input_path)
    summaries: dict[int, dict[str, Any]] = {}
    processed_frames = 0

    try:
        with tracks_path.open("w", encoding="utf-8") as tracks_file:
            progress = tqdm(total=frame_limit or None, desc=input_path.stem, unit="frame")
            while (
                options.max_frames is None
                or processed_frames < options.max_frames
            ):
                ok, frame = capture.read()
                if not ok:
                    break
                detections = backend.infer_frame(tracker, frame)
                for detection in detections:
                    try:
                        bbox = normalize_xyxy(
                            detection["bbox_xyxy"],
                            frame_width=width,
                            frame_height=height,
                        )
                    except ValueError:
                        continue
                    record = TrackRecord(
                        video_id=video_id,
                        frame_idx=processed_frames,
                        timestamp_s=processed_frames / fps,
                        track_id=detection["track_id"],
                        bbox_xyxy=bbox,
                        det_score=detection["det_score"],
                        category_id=detection["category_id"],
                    )
                    write_jsonl_line(tracks_file, record.to_dict())
                    update_summary(summaries, record)
                processed_frames += 1
                progress.update(1)
            progress.close()
    finally:
        capture.release()

    meta = VideoMeta(
        video_id=video_id,
        input_path=str(input_path),
        fps=fps,
        width=width,
        height=height,
        frame_count=declared_frames,
        processed_frames=processed_frames,
        tracker="MOTIP SportsMOT",
        tracker_config=runtime_config.to_dict(),
    )
    write_json(output_dir / "video_meta.json", meta.to_dict())
    track_summary = finalize_summary(summaries, fps)
    write_json(output_dir / "track_summary.json", track_summary)
    return {
        "output_dir": str(output_dir),
        "tracks": str(tracks_path),
        "video_meta": str(output_dir / "video_meta.json"),
        **track_summary,
    }
