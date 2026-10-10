"""Independent torso-frame sampling and cached local Qwen number evidence."""

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

from contracts.schema import read_jsonl, write_json, write_jsonl_line
from pipeline.identity.qwen_backend import QwenBackend, model_provenance, parse_reading
from pipeline.identity.sampling import SAMPLING_STRATEGY


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


def sample_torsos(video, tracks, output_dir, *, max_samples=6, min_gap_s=.25,
                  track_ids=None):
    import cv2
    from contracts.schema import TrackRecord
    from pipeline.identity.sampling import select_quality_samples

    if max_samples < 2:
        raise ValueError("Require at least two number samples")
    records = [TrackRecord.from_dict({"category_id": 0, **row}) for row in read_jsonl(tracks)]
    selected, quality = select_quality_samples(
        video, records, purpose="jersey", sample_count=max_samples,
        min_gap_s=min_gap_s, track_ids=track_ids,
    )
    if not selected:
        return []
    output = Path(output_dir).resolve()
    by_frame = defaultdict(list)
    for record in selected:
        by_frame[record.frame_idx].append(record)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        capture.release()
        raise ValueError("Cannot decode number-sampling video")
    samples = []
    try:
        for frame_idx in range(max(by_frame) + 1):
            ok, image = capture.read()
            if not ok:
                raise ValueError("Video ended before selected number frames")
            for record in by_frame.get(frame_idx, []):
                info = quality[(record.track_id, frame_idx)]
                left, top, right, bottom = info["sampling_roi_xyxy"]
                relative = f"jersey_media/T{record.track_id:04d}/f{frame_idx:08d}.png"
                path = output / relative
                if path.is_symlink() or not path.resolve().is_relative_to(output):
                    raise ValueError("Number crop path escapes output")
                path.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(path), image[top:bottom, left:right]):
                    raise OSError("Cannot save torso crop")
                samples.append({"raw_track_id": record.track_id,
                    "frame_idx": frame_idx, "timestamp_s": record.timestamp_s,
                    "crop_path": relative, "torso_roi_xyxy": [left, top, right, bottom],
                    **info})
    finally:
        capture.release()
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
    samples = sample_torsos(video, tracks, output, max_samples=max_samples,
                            min_gap_s=min_gap_s, track_ids=ids) if ids else []
    if samples and backend is None:
        backend = QwenBackend(model_dir, device=device)
    # A track with no readable crops still belongs to the same model contract
    # as other clips. Resolve provenance without loading a GPU model.
    provenance = (backend.provenance if backend else model_provenance(model_dir)
                  if ids else {"backend": "qwen_vl", "empty_input": True})
    by_track = defaultdict(list)
    inference_count, cache_hit_count = 0, 0
    for sample in samples:
        path = output / sample["crop_path"]
        cache_id = hashlib.sha256(path.read_bytes() + json.dumps(provenance, sort_keys=True).encode()).hexdigest()
        cache = output / "jersey_cache" / (cache_id + ".json")
        if not cache.resolve().is_relative_to(output):
            raise ValueError("Number cache escapes output")
        if cache.is_file():
            result = json.loads(cache.read_text())
            cache_hit_count += 1
        else:
            with Image.open(path) as image:
                raw = backend.read(image.convert("RGB"))
            inference_count += 1
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
        "thresholds": {"min_support": min_support, "min_gap_s": min_gap_s,
                       "max_samples": max_samples, "sampling_strategy": SAMPLING_STRATEGY},
        "track_count": len(ids), "sample_count": len(samples),
        "model_invoked": inference_count > 0, "inference_count": inference_count,
        "cache_hit_count": cache_hit_count,
        "warning": "Uncalibrated number evidence; matching digits do not establish player identity."}
    write_json(output / "jersey_summary.json", summary)
    return summary
