"""Make chronological raw-track evidence, including filtered tracklets."""

import json
from pathlib import Path

from analysis.evaluation.track_review import review_index
from contracts.schema import normalize_xyxy, write_json
from contracts.tracks import load_tracks


def sample_positions(count: int, samples: int) -> list[int]:
    if count < 1 or samples < 2:
        raise ValueError("Require observations and at least two chronological samples")
    return sorted({round(i * (count - 1) / (samples - 1)) for i in range(samples)})


def prepare_review(input_video: Path, tracks: Path, video_meta: Path,
                   output_dir: Path, quality: Path | None = None,
                   samples: int = 5, overwrite: bool = False) -> dict:
    import cv2

    if not 2 <= samples <= 16:
        raise ValueError("samples must be between 2 and 16")
    output_dir = output_dir.resolve()
    for path in (input_video, tracks, video_meta, quality):
        if path and (path.resolve() == output_dir or output_dir in path.resolve().parents):
            raise ValueError("Review evidence cannot overwrite its inputs")
    if (output_dir / "index.json").exists() and not overwrite:
        raise FileExistsError("Review evidence already exists; pass --overwrite")
    index = review_index(tracks, video_meta, quality)
    meta = json.loads(video_meta.read_text(encoding="utf-8"))
    grouped = load_tracks(tracks, meta)
    capture = cv2.VideoCapture(str(input_video))
    if not capture.isOpened():
        raise ValueError("Cannot open review source video")
    requested = {}
    for info in index["tracks"]:
        tid = info["raw_track_id"]
        rows = grouped[tid]
        positions = sample_positions(len(rows), samples)
        # Include anomaly frames and their preceding observation, not only a nice cover.
        jumps = set()
        if quality and quality.is_file():
            from contracts.schema import read_jsonl
            jumps = set(next((r.get("jump_frames", []) for r in read_jsonl(quality)
                              if int(r["track_id"]) == tid), []))
        anomaly_positions = {j for i, row in enumerate(rows) if row.frame_idx in jumps
                             for j in (max(0, i - 1), i)}
        positions = sorted(set(positions) | set(sorted(anomaly_positions)[:16]))
        for position in positions:
            row = rows[position]
            requested.setdefault(row.frame_idx, []).append((info, row))
    output_dir.mkdir(parents=True, exist_ok=True)
    warnings, decoded = [], 0
    try:
        for frame_idx in range(max(requested, default=-1) + 1):
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"Source video ends before reviewed frame {frame_idx}")
            decoded += 1
            if frame_idx not in requested:
                continue
            height, width = frame.shape[:2]
            if (width, height) != (int(meta["width"]), int(meta["height"])):
                raise ValueError("Review source dimensions do not match tracking metadata")
            for info, row in requested[frame_idx]:
                tid = info["raw_track_id"]
                try:
                    x1, y1, x2, y2 = normalize_xyxy(row.bbox_xyxy, frame_width=width, frame_height=height)
                except ValueError:
                    warnings.append(f"T{tid} frame {frame_idx}: box outside frame")
                    continue
                x1, y1 = int(x1), int(y1)
                x2, y2 = min(width, max(x1 + 1, int(x2))), min(height, max(y1 + 1, int(y2)))
                crop = frame[y1:y2, x1:x2]
                if crop.shape[0] > 320:
                    crop = cv2.resize(crop, (max(1, round(crop.shape[1] * 320 / crop.shape[0])), 320))
                context = frame.copy()
                cv2.rectangle(context, (x1, y1), (x2, y2), (40, 210, 120), 3)
                cv2.putText(context, f"T{tid} / frame {frame_idx}", (x1, max(20, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, .65, (40, 210, 120), 2)
                if width > 960:
                    context = cv2.resize(context, (960, max(1, round(height * 960 / width))))
                prefix = f"images/T{tid}_f{frame_idx}"
                paths = {"crop_path": prefix + "_crop.jpg", "context_path": prefix + "_context.jpg"}
                for kind, image in (("crop_path", crop), ("context_path", context)):
                    target = output_dir / paths[kind]
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if not cv2.imwrite(str(target), image):
                        raise OSError("Cannot save review evidence image")
                info["samples"].append({"frame_idx": frame_idx, "timestamp_s": row.timestamp_s,
                                        "det_score": row.det_score, **paths})
    finally:
        capture.release()
    index.update(warnings=warnings, decoded_frames=decoded, samples_per_track=samples,
                 warning="Chronological samples can reveal a switch, but cannot certify purity. Watch the whole tracked interval before marking full_track.")
    write_json(output_dir / "index.json", index)
    write_json(output_dir / "review.template.json", {
        "schema_version": 1, "video_id": index["video_id"], "fingerprint": index["fingerprint"],
        "tracks": [{"raw_track_id": row["raw_track_id"], "verdict": "unreviewed",
                    "scope": "sampled", "identity_label": None, "note": ""} for row in index["tracks"]],
    })
    return {"video_id": index["video_id"], "track_count": len(index["tracks"]),
            "evidence_images": sum(len(r["samples"]) for r in index["tracks"]), "warnings": warnings}
