"""RF-DETR ball/rim observations and local ball tracks; no made-shot decision."""

import argparse
import json
import math
import subprocess
from collections import Counter, deque
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

from pipeline.ball.tracker import BallTracker
from pipeline.media import atomic_json, probe_video, sha256_file, verify_frame_clock


@dataclass(frozen=True)
class Options:
    start_s: float = 0.0
    end_s: float | None = None
    goal_time_s: float | None = None
    ball_threshold: float = 0.15
    rim_threshold: float = 0.30
    short_side: int = 768
    max_long_side: int = 1344
    max_gap_s: float = 0.12
    device: str = "cuda"


def run(source, checkpoint, output, options, *, model=None):
    import cv2

    source, checkpoint, output = (
        Path(p).expanduser().resolve() for p in (source, checkpoint, output)
    )
    if (
        output == source
        or output == checkpoint
        or output in source.parents
        or output in checkpoint.parents
    ):
        raise ValueError("Output must not contain input video/checkpoint")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory is not empty; use a new run directory")
    meta = probe_video(source)
    total = verify_frame_clock(source, meta["fps"])
    end = meta["duration_s"] if options.end_s is None else options.end_s
    values = [
        options.start_s,
        end,
        options.ball_threshold,
        options.rim_threshold,
        options.max_gap_s,
    ]
    if any(not math.isfinite(v) for v in values):
        raise ValueError("Options must be finite")
    if not 0 <= options.start_s < end <= meta["duration_s"] + 1e-6:
        raise ValueError("Requested interval must lie inside the supplied video")
    if not all(0 <= t <= 1 for t in (options.ball_threshold, options.rim_threshold)):
        raise ValueError("Thresholds must be in [0,1]")
    if min(options.short_side, options.max_long_side) < 32 or options.max_gap_s <= 0:
        raise ValueError("Invalid resolution/gap")
    if options.goal_time_s is not None and not (
        math.isfinite(options.goal_time_s)
        and 0 <= options.goal_time_s < meta["duration_s"]
    ):
        raise ValueError("Invalid goal timestamp")
    start_frame = max(0, math.floor(options.start_s * meta["fps"]))
    end_frame = min(total, math.ceil(end * meta["fps"]))
    if end_frame <= start_frame:
        raise ValueError("Empty frame interval")
    checkpoint_hash = sha256_file(checkpoint)
    source_hash = sha256_file(source)
    if model is None:
        from rfdetr import from_checkpoint

        model = from_checkpoint(str(checkpoint), device=options.device)
    entries = model.class_names
    names = dict(entries) if isinstance(entries, dict) else dict(enumerate(entries))
    names = {int(k): str(v).lower().replace("_", "-").strip() for k, v in names.items()}
    if not {"ball", "rim"}.issubset(names.values()):
        raise ValueError(f"Checkpoint needs ball and rim classes; got {names}")
    h, w = meta["height"], meta["width"]
    scale = min(options.short_side / min(h, w), options.max_long_side / max(h, w))
    shape = (max(32, int(h * scale) // 32 * 32), max(32, int(w * scale) // 32 * 32))
    output.mkdir(parents=True, exist_ok=True)
    tracker = BallTracker(options.max_gap_s)
    trail = deque(maxlen=max(2, round(meta["fps"])))
    previous_track_id = None
    cap = cv2.VideoCapture(str(source))
    temporary = output / "preview.partial.avi"
    writer = cv2.VideoWriter(
        str(temporary), cv2.VideoWriter_fourcc(*"MJPG"), meta["fps"], (w, h)
    )
    if not cap.isOpened() or not writer.isOpened():
        cap.release()
        writer.release()
        raise RuntimeError("Cannot open source or preview encoder")
    counts, processed = Counter(), 0
    try:
        with (output / "ball.jsonl").open("w") as stream:
            # Sequential decoding avoids inaccurate seeking at GOP boundaries.
            for frame_idx in range(end_frame):
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError(
                        f"Unexpected decode end at source frame {frame_idx}"
                    )
                if frame_idx < start_frame:
                    continue
                prediction = model.predict(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                    threshold=0.1,
                    shape=shape,
                    include_source_image=False,
                )
                detections = []
                for box, score, cid in zip(
                    prediction.xyxy, prediction.confidence, prediction.class_id
                ):
                    box = [float(x) for x in box]
                    score = float(score)
                    if (
                        any(not math.isfinite(x) for x in box + [score])
                        or not 0 <= score <= 1
                    ):
                        raise ValueError("Invalid detector output")
                    if box[2] <= box[0] or box[3] <= box[1]:
                        continue
                    detections.append(
                        {
                            "label": names[int(cid)],
                            "class_id": int(cid),
                            "bbox_xyxy": box,
                            "score": score,
                        }
                    )
                balls = [
                    d
                    for d in detections
                    if d["label"] == "ball" and d["score"] >= options.ball_threshold
                ]
                rims = [
                    d
                    for d in detections
                    if d["label"] == "rim" and d["score"] >= options.rim_threshold
                ]
                track = tracker.update(balls, frame_idx / meta["fps"], w, h)
                row = {
                    "source_frame_idx": frame_idx,
                    "source_timestamp_s": frame_idx / meta["fps"],
                    "window_frame_idx": frame_idx - start_frame,
                    "detections": detections,
                    "rims": rims,
                    "ball_track": track,
                }
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                counts["frames_with_ball_detection"] += bool(balls)
                counts["frames_with_rim_detection"] += bool(rims)
                counts["observed_track_frames"] += (
                    track is not None and track["observed"]
                )
                counts["predicted_track_frames"] += (
                    track is not None and not track["observed"]
                )
                for detection in balls + rims:
                    x1, y1, x2, y2 = map(round, detection["bbox_xyxy"])
                    color = (
                        (0, 220, 0) if detection["label"] == "ball" else (255, 180, 0)
                    )
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        frame,
                        f"{detection['label']} {detection['score']:.2f}",
                        (x1, max(15, y1 - 5)),
                        0,
                        0.55,
                        color,
                        1,
                    )
                if track:
                    if track["ball_track_id"] != previous_track_id:
                        trail.clear()
                    previous_track_id = track["ball_track_id"]
                    center = tuple(round(x) for x in track["center_xy"])
                    color = (0, 255, 255) if track["observed"] else (0, 0, 255)
                    trail.append((center, track["observed"]))
                    for (a, observed_a), (b, observed_b) in zip(trail, list(trail)[1:]):
                        cv2.line(
                            frame,
                            a,
                            b,
                            (0, 255, 255) if observed_a and observed_b else (0, 0, 255),
                            2,
                        )
                    cv2.circle(frame, center, 6, color, 2)
                else:
                    trail.clear()
                text = f"F{frame_idx} / {frame_idx / meta['fps']:.3f}s"
                if options.goal_time_s is not None:
                    text += f" | external goal={options.goal_time_s:.3f}s"
                cv2.putText(frame, text, (15, 30), 0, 0.7, (255, 255, 255), 2)
                writer.write(frame)
                processed += 1
                if processed % 60 == 0:
                    print(f"[ball] {processed}/{end_frame - start_frame}", flush=True)
    finally:
        cap.release()
        writer.release()
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-i",
            str(temporary),
            "-an",
            "-c:v",
            "libx264",
            "-threads",
            "2",
            "-preset",
            "fast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output / "ball_vis.mp4"),
        ],
        check=True,
    )
    temporary.unlink()
    summary = {
        "schema_version": 1,
        "status": "completed_not_accuracy_evaluated",
        "source": meta,
        "source_sha256": source_hash,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "class_names": names,
        "options": asdict(options),
        "detector_shape": list(shape),
        "source_start_frame": start_frame,
        "source_end_frame_exclusive": end_frame,
        "processed_frames": processed,
        "counts": dict(counts),
        "detector_version": version("rfdetr")
        if model.__class__.__module__.startswith("rfdetr")
        else "test-double",
        "tracking": "motion_gated_single_ball_v1",
        "made_shot_evaluated": False,
        "shooter_attributed": False,
        "limitations": [
            "Predicted points are not visual evidence",
            "Ball association may follow a false detection",
            "Does not bridge camera cuts; use a single-shot window",
            "No shooter/player ID is produced",
        ],
    }
    atomic_json(output / "summary.json", summary)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    for name, kind, default in [
        ("start-s", float, 0.0),
        ("end-s", float, None),
        ("goal-time-s", float, None),
        ("ball-threshold", float, 0.15),
        ("rim-threshold", float, 0.30),
        ("short-side", int, 768),
        ("max-long-side", int, 1344),
        ("max-gap-s", float, 0.12),
        ("device", str, "cuda"),
    ]:
        p.add_argument("--" + name, type=kind, default=default)
    args = p.parse_args()
    kwargs = {k: getattr(args, k) for k in Options.__dataclass_fields__}
    print(
        json.dumps(
            run(args.input, args.checkpoint, args.output_dir, Options(**kwargs)),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
