"""RF-DETR ball/rim observations and local ball tracks; no made-shot decision."""

import argparse
import json
import math
import subprocess
from collections import Counter, deque
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

from pipeline.ball.tracker import associate_rows
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
    min_confirm_frames: int = 3
    max_size_ratio: float = 2.5


def run(source, checkpoint, output, options, *, model=None, reuse_detections=None):
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
        options.max_size_ratio,
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
    if options.min_confirm_frames < 2 or options.max_size_ratio <= 1:
        raise ValueError("Invalid confirmation count or size ratio")
    checkpoint_hash = sha256_file(checkpoint)
    source_hash = sha256_file(source)
    cached = None
    if reuse_detections is not None:
        cached = json.loads((Path(reuse_detections) / "summary.json").read_text())
        if (
            cached["source_sha256"] != source_hash
            or cached["checkpoint_sha256"] != checkpoint_hash
            or not cached["status"].startswith("completed")
        ):
            raise ValueError(
                "Cached detections have different source/checkpoint or incomplete status"
            )
        names = {int(k): v for k, v in cached["class_names"].items()}
        shape = tuple(cached["detector_shape"])
        if any(
            getattr(options, k) != cached["options"][k]
            for k in ("short_side", "max_long_side")
        ):
            raise ValueError("Changing detector resolution requires fresh inference")
        detector_threshold = cached.get("detector_threshold", 0.1)
        if min(options.ball_threshold, options.rim_threshold) < detector_threshold:
            raise ValueError(
                "Requested threshold is below cached detector threshold; rerun inference"
            )
        source_rows = [
            json.loads(line)
            for line in (Path(reuse_detections) / "ball.jsonl").read_text().splitlines()
            if line.strip()
        ]
        if len(source_rows) != cached["processed_frames"]:
            raise ValueError("Cached detection row count mismatch")
        expected = list(
            range(cached["source_start_frame"], cached["source_end_frame_exclusive"])
        )
        if [r["source_frame_idx"] for r in source_rows] != expected:
            raise ValueError(
                "Cached detections have missing/duplicate/reordered frame IDs"
            )
        if any(
            not math.isfinite(r["source_timestamp_s"])
            or abs(r["source_timestamp_s"] - r["source_frame_idx"] / meta["fps"]) > 1e-6
            for r in source_rows
        ):
            raise ValueError("Cached timestamps do not match the source frame clock")
        source_rows = [
            r for r in source_rows if start_frame <= r["source_frame_idx"] < end_frame
        ]
        if [r["source_frame_idx"] for r in source_rows] != list(
            range(start_frame, end_frame)
        ):
            raise ValueError("Requested window is outside cached detections")
    else:
        if model is None:
            from rfdetr import from_checkpoint

            model = from_checkpoint(str(checkpoint), device=options.device)
        entries = model.class_names
        names = dict(entries) if isinstance(entries, dict) else dict(enumerate(entries))
        names = {
            int(k): str(v).lower().replace("_", "-").strip() for k, v in names.items()
        }
        h, w = meta["height"], meta["width"]
        scale = min(options.short_side / min(h, w), options.max_long_side / max(h, w))
        shape = (max(32, int(h * scale) // 32 * 32), max(32, int(w * scale) // 32 * 32))
        source_rows = []
        detector_threshold = min(0.1, options.ball_threshold, options.rim_threshold)
        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            raise RuntimeError("Cannot open source video")
        try:
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
                    threshold=detector_threshold,
                    shape=shape,
                    include_source_image=False,
                )
                detections = [
                    {
                        "label": names[int(cid)],
                        "class_id": int(cid),
                        "bbox_xyxy": [float(x) for x in box],
                        "score": float(score),
                    }
                    for box, score, cid in zip(
                        prediction.xyxy, prediction.confidence, prediction.class_id
                    )
                ]
                source_rows.append(
                    {
                        "source_frame_idx": frame_idx,
                        "source_timestamp_s": frame_idx / meta["fps"],
                        "detections": detections,
                    }
                )
                if len(source_rows) % 60 == 0:
                    print(
                        f"[ball] {len(source_rows)}/{end_frame - start_frame}",
                        flush=True,
                    )
        finally:
            cap.release()
    if not {"ball", "rim"}.issubset(names.values()):
        raise ValueError(f"Checkpoint needs ball and rim classes; got {names}")
    h, w = meta["height"], meta["width"]
    rows = []
    for original in source_rows:
        row = {
            k: original[k]
            for k in ("source_frame_idx", "source_timestamp_s", "detections")
        }
        for d in row["detections"]:
            box = d["bbox_xyxy"]
            if (
                len(box) != 4
                or any(not math.isfinite(x) for x in box + [d["score"]])
                or not 0 <= d["score"] <= 1
                or box[2] <= box[0]
                or box[3] <= box[1]
            ):
                raise ValueError("Invalid cached/model detection")
        row["window_frame_idx"] = row["source_frame_idx"] - start_frame
        row["rims"] = [
            d
            for d in row["detections"]
            if d["label"] == "rim" and d["score"] >= options.rim_threshold
        ]
        rows.append(row)
    rows = associate_rows(
        rows,
        w,
        h,
        threshold=options.ball_threshold,
        max_gap_s=options.max_gap_s,
        min_confirm_frames=options.min_confirm_frames,
        max_size_ratio=options.max_size_ratio,
    )
    output.mkdir(parents=True, exist_ok=True)
    with (output / "ball.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    _render(source, output, meta, rows, options)
    counts = Counter()
    for row in rows:
        track = row["ball_track"]
        balls = [
            d
            for d in row["detections"]
            if d["label"] == "ball" and d["score"] >= options.ball_threshold
        ]
        counts["frames_with_ball_detection"] += bool(balls)
        counts["frames_with_rim_detection"] += bool(row["rims"])
        counts["observed_track_frames"] += track is not None and track["observed"]
        counts["predicted_track_frames"] += track is not None and not track["observed"]
        counts["unconfirmed_segment_frames"] += (
            row["association_status"] == "unconfirmed_segment"
        )
        counts["candidate_frames_without_accepted_observation"] += bool(balls) and not (
            track and track["observed"]
        )
    summary = {
        "schema_version": 2,
        "status": "completed_not_accuracy_evaluated",
        "source": meta,
        "source_sha256": source_hash,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "class_names": names,
        "options": asdict(options),
        "detector_shape": list(shape),
        "detector_threshold": detector_threshold,
        "source_start_frame": start_frame,
        "source_end_frame_exclusive": end_frame,
        "processed_frames": len(rows),
        "counts": dict(counts),
        "detector_version": cached["detector_version"]
        if cached
        else (
            version("rfdetr")
            if model.__class__.__module__.startswith("rfdetr")
            else "test-double"
        ),
        "tracking": "motion_size_confirmation_v2",
        "made_shot_evaluated": False,
        "shooter_attributed": False,
        "reuse_detections": str(Path(reuse_detections).resolve()) if cached else None,
        "reuse_jsonl_sha256": sha256_file(Path(reuse_detections) / "ball.jsonl")
        if cached
        else None,
        "limitations": [
            "Predictions are not visual evidence",
            "Confirmed means temporal support, not ground truth",
            "Persistent plausible false detections can survive filtering",
            "Does not bridge camera cuts; use a single-shot window",
            "No shooter/player ID is produced",
        ],
    }
    atomic_json(output / "summary.json", summary)
    return summary


def _render(source, output, meta, rows, options):
    import cv2

    h, w = meta["height"], meta["width"]
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
    try:
        start_frame = rows[0]["source_frame_idx"]
        for _ in range(start_frame):
            if not cap.read()[0]:
                raise RuntimeError("Unexpected decode end before render window")
        for row in rows:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Unexpected decode end while rendering")
            frame_idx = row["source_frame_idx"]
            track = row["ball_track"]
            balls = [
                d
                for d in row["detections"]
                if d["label"] == "ball" and d["score"] >= options.ball_threshold
            ]
            rims = row["rims"]
            for detection in balls + rims:
                x1, y1, x2, y2 = map(round, detection["bbox_xyxy"])
                color = (
                    (150, 150, 150) if detection["label"] == "ball" else (255, 180, 0)
                )
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    f"candidate {detection['score']:.2f}"
                    if detection["label"] == "ball"
                    else f"rim {detection['score']:.2f}",
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
                if track["observed"]:
                    x1, y1, x2, y2 = map(round, track["bbox_xyxy"])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument(
        "--reuse-detections",
        type=Path,
        help="Completed prior ball run; no RF-DETR inference",
    )
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
        ("min-confirm-frames", int, 3),
        ("max-size-ratio", float, 2.5),
    ]:
        p.add_argument("--" + name, type=kind, default=None)
    args = p.parse_args()
    kwargs = asdict(Options())
    if args.reuse_detections:
        saved = json.loads((args.reuse_detections / "summary.json").read_text())
        kwargs.update({k: v for k, v in saved["options"].items() if k in kwargs})
        if args.checkpoint is None:
            args.checkpoint = Path(saved["checkpoint"])
    if args.checkpoint is None:
        p.error("--checkpoint is required without --reuse-detections")
    kwargs.update({k: getattr(args, k) for k in kwargs if getattr(args, k) is not None})
    print(
        json.dumps(
            run(
                args.input,
                args.checkpoint,
                args.output_dir,
                Options(**kwargs),
                reuse_detections=args.reuse_detections,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
