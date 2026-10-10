"""Independent torso-frame sampling and cached local Qwen number evidence."""

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

from contracts.schema import read_jsonl, write_json, write_jsonl_line
from pipeline.identity.qwen_backend import QwenBackend, parse_reading


def qwen_consensus(readings, *, min_support=2, min_gap_s=.25):
    if min_support < 2 or not math.isfinite(min_gap_s) or min_gap_s <= 0:
        raise ValueError("Require two or more temporally separated supporting frames")
    by_number = defaultdict(dict)
    for row in readings:
        number = row.get("number")
        if number is not None and not row.get("rejection_reasons"):
            checked = parse_reading(json.dumps({"number": number, "readable": True}))
            if checked["number"] is None:
                raise ValueError("Invalid cached Qwen number")
            timestamp = float(row["timestamp_s"])
            if not math.isfinite(timestamp) or timestamp < 0:
                raise ValueError("Invalid evidence timestamp")
            by_number[number][int(row["frame_idx"])] = row
    candidates = []
    for number, evidence in sorted(by_number.items()):
        independent, last = [], -math.inf
        for row in sorted(evidence.values(), key=lambda r: r["timestamp_s"]):
            if row["timestamp_s"] - last >= min_gap_s:
                independent.append(row)
                last = row["timestamp_s"]
        candidates.append({"number": number, "support_frames": len(independent), "evidence": list(evidence.values())})
    accepted = len(candidates) == 1 and candidates[0]["support_frames"] >= min_support
    return {"number": candidates[0]["number"] if accepted else None,
        "status": "candidate_consensus" if accepted else "conflict" if len(candidates) > 1 else "insufficient_evidence" if candidates else "unreadable",
        "candidates": candidates, "readings": readings, "verified": False, "backend": "qwen_vl"}


def sample_torsos(video, tracks, output_dir, *, max_samples=6, min_gap_s=.25):
    import cv2
    if max_samples < 2 or not math.isfinite(min_gap_s) or min_gap_s <= 0:
        raise ValueError("Invalid number sampling parameters")
    by_frame, by_track = defaultdict(list), defaultdict(dict)
    for row in read_jsonl(tracks):
        tid, frame = int(row["track_id"]), int(row["frame_idx"])
        if frame < 0 or not math.isfinite(float(row["timestamp_s"])) or row["timestamp_s"] < 0:
            raise ValueError("Invalid track timestamp/frame")
        if float(row["det_score"]) >= .5:
            by_frame[frame].append(row)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError("Cannot decode number-sampling video")
    try:
        frame_idx = 0
        while by_frame:
            ok, image = cap.read()
            if not ok:
                raise ValueError("Video ended before selected tracking frames")
            for row in by_frame.pop(frame_idx, []):
                x1, y1, x2, y2 = map(float, row["bbox_xyxy"])
                height, width = image.shape[:2]
                # Generous torso region; its legibility is decided by Qwen, not assumed.
                left, right = max(0, round(x1)), min(width, round(x2))
                top, bottom = max(0, round(y1 + .12 * (y2-y1))), min(height, round(y1 + .7 * (y2-y1)))
                if right-left < 32 or bottom-top < 48:
                    continue
                crop = image[top:bottom, left:right]
                sharpness = float(cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
                if sharpness < 5:
                    continue
                tid = int(row["track_id"])
                bucket = int(float(row["timestamp_s"]) / min_gap_s)
                quality = sharpness * math.sqrt(crop.shape[0] * crop.shape[1])
                old = by_track[tid].get(bucket)
                if old is None or quality > old[0]:
                    by_track[tid][bucket] = (quality, dict(row), crop.copy(), [left, top, right, bottom])
                # Bound image memory for every track while keeping diverse temporal buckets.
                if len(by_track[tid]) > max_samples * 3:
                    worst = min(by_track[tid], key=lambda k: by_track[tid][k][0])
                    del by_track[tid][worst]
            frame_idx += 1
    finally:
        cap.release()
    output = Path(output_dir)
    samples = []
    for tid, buckets in sorted(by_track.items()):
        selected = []
        for quality, row, crop, roi in sorted(buckets.values(), key=lambda r: -r[0]):
            if any(abs(row["timestamp_s"] - s["timestamp_s"]) < min_gap_s for s in selected):
                continue
            relative = f"jersey_media/T{tid:04d}/f{row['frame_idx']:08d}.png"
            path = output / relative
            if path.is_symlink() or not path.resolve().is_relative_to(output.resolve()):
                raise ValueError("Number crop path escapes output")
            path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(path), crop):
                raise OSError("Cannot save torso crop")
            selected.append({"raw_track_id": tid, "frame_idx": row["frame_idx"],
                "timestamp_s": row["timestamp_s"], "crop_path": relative, "torso_roi_xyxy": roi,
                "sampling_quality": quality})
            if len(selected) >= max_samples:
                break
        samples.extend(sorted(selected, key=lambda r: r["frame_idx"]))
    return samples


def recognize_jerseys_qwen(video, tracks, identity_map, output_dir, model_dir, *,
                           max_samples=6, min_support=2, min_gap_s=.25, device="cuda", backend=None):
    from PIL import Image
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    mappings = list(read_jsonl(identity_map))
    if len({r["raw_track_id"] for r in mappings}) != len(mappings):
        raise ValueError("Duplicate identity mapping")
    ids = {int(r["raw_track_id"]) for r in mappings}
    qwen_consensus([], min_support=min_support, min_gap_s=min_gap_s)
    samples = sample_torsos(video, tracks, output, max_samples=max_samples, min_gap_s=min_gap_s) if ids else []
    samples = [s for s in samples if s["raw_track_id"] in ids]
    if samples and backend is None:
        backend = QwenBackend(model_dir, device=device)
    provenance = backend.provenance if backend else {"backend": "qwen_vl", "empty_input": True}
    by_track = defaultdict(list)
    for sample in samples:
        path = output / sample["crop_path"]
        cache_id = hashlib.sha256(path.read_bytes() + json.dumps(provenance, sort_keys=True).encode()).hexdigest()
        cache = output / "jersey_cache" / (cache_id + ".json")
        if not cache.resolve().is_relative_to(output):
            raise ValueError("Number cache escapes output")
        if cache.is_file():
            result = json.loads(cache.read_text())
        else:
            with Image.open(path) as image:
                raw = backend.read(image.convert("RGB"))
            result = {**parse_reading(raw), "raw_response": raw}
            write_json(cache, result)
        by_track[sample["raw_track_id"]].append({**sample, **result, "text": result["number"]})
    evidence = {tid: {"raw_track_id": tid, **qwen_consensus(by_track[tid], min_support=min_support, min_gap_s=min_gap_s)} for tid in sorted(ids)}
    from pipeline.identity.jersey import group_numbers
    groups = defaultdict(list)
    for row in mappings:
        groups[row["person_id"]].append(int(row["raw_track_id"]))
    people = [{"person_id": pid, **group_numbers(tids, evidence)} for pid, tids in sorted(groups.items())]
    for name, rows in (("tracks", evidence.values()), ("people", people), ("samples", samples)):
        path = output / f"jersey_{name}.jsonl"
        if path.is_symlink():
            raise ValueError("Number output cannot be a symlink")
        with path.open("w") as handle:
            for row in rows:
                write_jsonl_line(handle, row)
    summary = {"backend": "qwen_vl", "provenance": provenance,
        "thresholds": {"min_support": min_support, "min_gap_s": min_gap_s},
        "track_count": len(ids), "sample_count": len(samples), "model_invoked": bool(samples),
        "warning": "Uncalibrated number evidence; matching digits do not establish player identity."}
    write_json(output / "jersey_summary.json", summary)
    return summary
