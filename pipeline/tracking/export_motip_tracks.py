"""Run MOTIP on one video and export a framework-independent track file."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
from tqdm import tqdm

from pipeline.common.schema import (
    TrackRecord,
    VideoMeta,
    normalize_xyxy,
    write_json,
    write_jsonl_line,
)
from pipeline.tracking.motip_backend import MotipBackend, MotipRuntimeConfig


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export MOTIP person tracks from one video"
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--motip-root", type=Path, default=REPOSITORY_ROOT / "MOTIP")
    parser.add_argument(
        "--config",
        default="configs/r50_deformable_detr_motip_sportsmot.yaml",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--assignment-protocol", default="object-max")
    parser.add_argument("--miss-tolerance", type=int, default=60)
    parser.add_argument("--det-thresh", type=float, default=0.3)
    parser.add_argument("--newborn-thresh", type=float, default=0.6)
    parser.add_argument("--id-thresh", type=float, default=0.2)
    parser.add_argument("--area-thresh", type=int, default=0)
    parser.add_argument("--max-shorter", type=int, default=800)
    parser.add_argument("--max-longer", type=int, default=1440)
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--no-visualization", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def video_id_for(path: Path) -> str:
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:10]
    return f"{path.stem}-{digest}"


def color_for(track_id: int) -> tuple[int, int, int]:
    return (
        64 + (track_id * 47) % 192,
        64 + (track_id * 89) % 192,
        64 + (track_id * 131) % 192,
    )


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


def run(args: argparse.Namespace) -> None:
    input_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tracks_path = output_dir / "tracks.jsonl"
    if tracks_path.exists() and not args.overwrite:
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
    if args.max_frames is not None:
        frame_limit = min(frame_limit, args.max_frames) if frame_limit else args.max_frames

    runtime_config = MotipRuntimeConfig(
        assignment_protocol=args.assignment_protocol,
        miss_tolerance=args.miss_tolerance,
        det_thresh=args.det_thresh,
        newborn_thresh=args.newborn_thresh,
        id_thresh=args.id_thresh,
        area_thresh=args.area_thresh,
        max_shorter=args.max_shorter,
        max_longer=args.max_longer,
        fp16=not args.fp32,
    )
    backend = MotipBackend(
        motip_root=args.motip_root,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        runtime_config=runtime_config,
        device=args.device,
    )
    tracker = backend.create_tracker(height=height, width=width)
    video_id = video_id_for(input_path)
    summaries: dict[int, dict[str, Any]] = {}
    writer = None
    processed_frames = 0

    if not args.no_visualization:
        visualization_path = output_dir / "tracks_vis.mp4"
        writer = cv2.VideoWriter(
            str(visualization_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not open video writer: {visualization_path}")

    try:
        with tracks_path.open("w", encoding="utf-8") as tracks_file:
            progress = tqdm(total=frame_limit or None, desc=input_path.stem, unit="frame")
            while args.max_frames is None or processed_frames < args.max_frames:
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
                    if writer is not None:
                        x1, y1, x2, y2 = (int(value) for value in bbox)
                        color = color_for(record.track_id)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        label = f"ID {record.track_id} {record.det_score:.2f}"
                        cv2.putText(
                            frame,
                            label,
                            (x1, max(15, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            color,
                            2,
                        )
                if writer is not None:
                    writer.write(frame)
                processed_frames += 1
                progress.update(1)
            progress.close()
    finally:
        capture.release()
        if writer is not None:
            writer.release()

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
    write_json(
        output_dir / "track_summary.json", finalize_summary(summaries, fps)
    )
    print(json.dumps({"output_dir": str(output_dir), **finalize_summary(summaries, fps)}, indent=2))


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
