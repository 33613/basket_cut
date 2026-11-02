"""Portable shooter handoff and visual evidence. No identity models or library writes."""

import csv
import html
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

from contracts.schema import read_jsonl, write_jsonl_line
from contracts.tracks import load_tracks
from pipeline.media import atomic_json, sha256_file
from pipeline.player_link.evidence import track_segment


def read_reviews(path, protocol):
    if path is None:
        return {}
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    events = {e["event_id"] for e in protocol["events"]}
    reviews = {}
    for row in rows:
        eid = row["event_id"]
        if eid not in events or eid in reviews:
            raise ValueError(f"Unknown/duplicate review event {eid}")
        row["decision"] = row.get("decision", "").strip()
        if row["decision"] not in ("", "confirm", "reject", "reset"):
            raise ValueError("Review decision must be blank, confirm, reject or reset")
        if row.get("decision") == "confirm" and not row.get("track_id", "").strip():
            raise ValueError("Confirmed review requires track_id")
        reviews[eid] = row
    return reviews


def resolve(result, tracks, event, source, options, review):
    """Preserve the automatic hypothesis; human decisions are a separate revision."""
    chosen = result["shooter_track_id"]
    trajectory = result.get("trajectory")
    status = result["status"]
    if review and review.get("decision") == "reject":
        return None, None, "rejected_by_review"
    if review and review.get("decision") == "confirm":
        chosen = int(review["track_id"])
        rows = sorted((r for r in tracks if r.track_id == chosen), key=lambda r: r.frame_idx)
        if not rows:
            raise ValueError("Reviewed shooter track does not exist in this event window")
        candidate = next((c for c in result["candidates"] if c["track_id"] == chosen), None)
        trajectory = track_segment(candidate, tracks, source["fps"], event, options) if candidate else dict(
            window_start_frame=rows[0].frame_idx, window_end_frame_inclusive=rows[-1].frame_idx)
        start = int(review.get("start_frame") or trajectory["window_start_frame"])
        end = int(review.get("end_frame") or trajectory["window_end_frame_inclusive"])
        if not 0 <= start <= end < event["window_end_frame_exclusive"]-event["window_start_frame"]:
            raise ValueError("Review frame interval must use zero-based EVENT WINDOW frame indices")
        if not any(start <= r.frame_idx <= end for r in rows):
            raise ValueError("Review interval contains no shooter observations")
        trajectory = dict(window_start_frame=start, window_end_frame_inclusive=end,
            source_start_frame=event["window_start_frame"]+start,
            source_end_frame_inclusive=event["window_start_frame"]+end,
            identity_continuity="human selected; requires visual verification")
        status = "reviewed"
    return chosen, trajectory, status


def export_actor(directory, tracks, event, source, result, chosen, trajectory, attribution_status, review, options, ffmpeg):
    import cv2
    import numpy as np
    directory = Path(directory)
    target = directory / "handoff"
    temporary = directory / ".handoff.partial"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()
    record = dict(schema_version=1, event_id=event["event_id"], video_id=event["video_id"],
        source_sha256=source["sha256"], goal_time_s=event["goal_time_s"],
        window_start_source_frame=event["window_start_frame"], fps=source["fps"],
        shooter_track_id=chosen, track_uid=f"{event['event_id']}:T{chosen:04d}" if chosen is not None else None,
        player_id=None, attribution_status=attribution_status, ball_verified=False, accuracy_evaluated=False,
        requires_review=attribution_status != "reviewed", trajectory=trajectory,
        automatic_result="../link.json", window_video="../window.mkv", window_metadata="../window.json",
        source_video_path=source["path"],
        samples=[], notes="Raw local track identity is not a cross-video player identity.")
    if review:
        record["manual_review"] = review
    if chosen is not None:
        start, end = trajectory["window_start_frame"], trajectory["window_end_frame_inclusive"]
        rows = sorted((r for r in tracks if r.track_id == chosen and start <= r.frame_idx <= end), key=lambda r: r.frame_idx)
        if not rows:
            raise ValueError("Empty shooter segment")
        (temporary / "crops").mkdir()
        selected = set(round(i*(len(rows)-1)/max(1, min(options.samples_per_track, len(rows))-1))
                       for i in range(min(options.samples_per_track, len(rows))))
        sample_frames = {rows[i].frame_idx for i in selected}
        by_frame = {r.frame_idx: r for r in rows}
        capture = cv2.VideoCapture(str(directory / "window.mkv"))
        # Real time is preserved, including black frames for missing observations.
        command = [ffmpeg, "-nostdin", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", "256x512", "-r", str(source["fps"]), "-i", "pipe:0", "-an", "-c:v", "libx264",
            "-threads", "2", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temporary / "trajectory.mp4")]
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            with (temporary / "frames.jsonl").open("w") as frame_file:
                for idx in range(end+1):
                    ok, image = capture.read()
                    if not ok:
                        raise ValueError("Decode ended before shooter segment")
                    if idx < start:
                        continue
                    canvas = np.zeros((512, 256, 3), dtype=np.uint8)
                    r = by_frame.get(idx)
                    if r:
                        import math
                        x1, y1, x2, y2 = (math.floor(r.bbox_xyxy[0]), math.floor(r.bbox_xyxy[1]),
                                          math.ceil(r.bbox_xyxy[2]), math.ceil(r.bbox_xyxy[3]))
                        x1, y1, x2, y2 = max(0,x1), max(0,y1), min(image.shape[1],x2), min(image.shape[0],y2)
                        crop = image[y1:y2, x1:x2]
                        if crop.size == 0:
                            raise ValueError("Invalid shooter crop")
                        image_path = f"crops/F{idx:07d}.png"
                        if not cv2.imwrite(str(temporary / image_path), crop):
                            raise RuntimeError("Cannot write raw shooter crop")
                        if idx in sample_frames:
                            record["samples"].append(dict(sample_id=f"{event['event_id']}-T{chosen}-F{idx}",
                                crop_path=image_path, window_frame_idx=idx,
                                source_frame_idx=idx+event["window_start_frame"],
                                source_timestamp_s=(idx+event["window_start_frame"])/source["fps"],
                                bbox_xyxy=list(r.bbox_xyxy), crop_sha256=sha256_file(temporary / image_path)))
                        scale = min(256/crop.shape[1], 512/crop.shape[0])
                        w, h = max(1,round(crop.shape[1]*scale)), max(1,round(crop.shape[0]*scale))
                        canvas[(512-h)//2:(512-h)//2+h, (256-w)//2:(256-w)//2+w] = cv2.resize(crop, (w,h))
                    process.stdin.write(canvas.tobytes())
                    write_jsonl_line(frame_file, dict(window_frame_idx=idx,
                        source_frame_idx=idx+event["window_start_frame"],
                        source_timestamp_s=(idx+event["window_start_frame"])/source["fps"],
                        preview_frame_idx=idx-start, observed=r is not None))
            process.stdin.close()
            error = process.stderr.read().decode(errors="replace")
            if process.wait():
                raise RuntimeError(f"FFmpeg shooter export failed: {error[-2000:]}")
        finally:
            capture.release()
            if process.poll() is None:
                process.kill()
                process.wait()
            process.stderr.close()
            if not process.stdin.closed:
                process.stdin.close()
        with (temporary / "tracks.jsonl").open("w") as stream:
            for r in rows:
                value = r.to_dict()
                value.update(source_video_id=event["video_id"], source_frame_idx=r.frame_idx+event["window_start_frame"],
                             source_timestamp_s=(r.frame_idx+event["window_start_frame"])/source["fps"])
                write_jsonl_line(stream, value)
    atomic_json(temporary / "manifest.json", record)
    if target.exists():
        shutil.rmtree(target)
    temporary.replace(target)
    return record


def export_run(root, options, settings, review_file=None):
    from workflows import link_player as flow
    from analysis.visualization.render import RenderOptions, render_results
    root = Path(root)
    protocol = flow.saved(root)
    persisted = root / "reviews.json"
    state = flow.read(persisted) if persisted.exists() else {}
    reviews = state.get("decisions", {})
    applied_inputs = state.get("applied_inputs", {})
    requested = review_file or (root / "review.csv" if (root / "review.csv").exists() else None)
    changed = requested is not None and applied_inputs.get(str(Path(requested).resolve())) != sha256_file(requested)
    for eid, row in (read_reviews(requested, protocol) if changed else {}).items():
        if row.get("decision") == "reset":
            reviews.pop(eid, None)
        elif row.get("decision"):
            reviews[eid] = row
    if changed:
        applied_inputs[str(Path(requested).resolve())] = sha256_file(requested)
    jobs = []
    for event in protocol["events"]:
        directory = flow.folder(root, event)
        flow.require_stage(directory, "link")
        result = flow.read(directory / "link.json")
        meta = flow.read(directory / "video_meta.json")
        grouped = load_tracks(directory / "tracks.jsonl", meta)
        tracks = [r for rs in grouped.values() for r in rs]
        source = protocol["sources"][event["video_id"]]
        review = reviews.get(event["event_id"])
        jobs.append((event, directory, result, tracks, source, review, resolve(result, tracks, event, source, options, review)))
    records = []
    for event, directory, result, tracks, source, review, (chosen, trajectory, attribution_status) in jobs:
        signature = dict(link=sha256_file(directory / "link.json"), tracks=sha256_file(directory / "tracks.jsonl"),
            window=sha256_file(directory / "window.mkv"), actions=sha256_file(directory / "actions.jsonl"),
            review=review, samples=options.samples_per_track)
        marker = directory / ".export.json"
        if marker.exists() and flow.read(marker)["signature"] == signature:
            old = flow.read(marker)
            for name, checksum in old["outputs"].items():
                if sha256_file(directory / name) != checksum:
                    raise ValueError("Exported evidence changed")
            records.append(flow.read(directory / "handoff/manifest.json"))
            continue
        print(f"[export] {event['event_id']} / {attribution_status}", flush=True)
        record = export_actor(directory, tracks, event, source, result, chosen, trajectory,
                              attribution_status, review, options, settings.ffmpeg)
        raw = directory / ".preview.mp4"
        render_results(RenderOptions(input=directory / "window.mkv", tracks=directory / "tracks.jsonl",
            actions=directory / "actions.jsonl", output=raw, show_top_candidate=True, overwrite=True))
        converted = directory / ".preview.h264.mp4"
        subprocess.run([settings.ffmpeg, "-nostdin", "-v", "error", "-y", "-i", str(raw), "-an", "-c:v", "libx264",
            "-threads", "2", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(converted)], check=True)
        converted.replace(directory / "evidence.mp4")
        raw.unlink()
        names = [p.relative_to(directory).as_posix() for p in (directory / "handoff").rglob("*") if p.is_file()]
        flow.mark(directory, "export", signature, sorted(names)+["evidence.mp4"])
        records.append(record)
    with (root / "handoff.jsonl").open("w") as stream:
        for record in records:
            write_jsonl_line(stream, dict(record, handoff_dir=f"events/{record['event_id']}/handoff"))
    atomic_json(persisted, dict(decisions=reviews, applied_inputs=applied_inputs))
    template = root / "review.csv"
    if not template.exists():
        with template.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["event_id", "decision", "track_id", "start_frame", "end_frame", "notes"])
            writer.writeheader()
            writer.writerows(dict(event_id=r["event_id"], decision="", track_id="", start_frame="", end_frame="", notes="") for r in records)
    sections = []
    for event in protocol["events"]:
        result = flow.read(flow.folder(root,event) / "link.json")
        cards = "".join(f'<li>T{c["track_id"]}: action score {c["score"]:.3f}, '
                        f'candidate interval {c["action_interval_source_s"][0]:.2f}–{c["action_interval_source_s"][1]:.2f}s '
                        f'(source clip)</li>' for c in result["candidates"])
        eid = event["event_id"]
        record = next(r for r in records if r["event_id"] == eid)
        imgs = "".join(f'<img height="180" src="events/{eid}/handoff/{s["crop_path"]}">' for s in record["samples"])
        sections.append(f'<section><h2>Video {html.escape(event["video_id"])} / {eid}</h2>'
            f'<p>Goal {event["goal_time_s"]}s; {record["attribution_status"]}; selected T={record["shooter_track_id"]}</p>'
            f'<video controls width="720" src="events/{eid}/evidence.mp4"></video><ul>{cards}</ul>{imgs}</section>')
    (root / "review.html").write_text('<!doctype html><meta charset="utf-8"><title>link_player evidence</title>'
        '<style>body{font:16px sans-serif;margin:32px}section{margin-bottom:48px}img{margin:5px}</style>'
        '<h1>Goal → shooter track</h1><p>Action-only hypotheses; ball flight is not verified. '
        'Review frame indices are local to each EVENT WINDOW. Player identity is downstream.</p>' + "".join(sections))
    atomic_json(root / "status.json", flow.status(root))


def pack_run(root):
    from workflows import link_player as flow
    root = Path(root).resolve()
    if flow.status(root)["status"] != "completed_not_accuracy_evaluated":
        raise ValueError("Finish all stages before packing")
    for event in flow.saved(root)["events"]:
        directory = flow.folder(root, event)
        for stage in ("prepare", "track", "action", "link", "export"):
            marker = flow.read(directory / f".{stage}.json")
            for name, checksum in marker["outputs"].items():
                if sha256_file(directory / name) != checksum:
                    raise ValueError(f"Changed {stage} evidence; refusing to pack")
    paths = sorted(p for p in root.rglob("*") if p.is_file() and p.name != "checksums.sha256"
                   and not any(part.startswith(".") for part in p.relative_to(root).parts))
    if any(p.is_symlink() for p in paths):
        raise ValueError("Portable package cannot contain symlinks")
    (root / "checksums.sha256").write_text("".join(f"{sha256_file(p)}  {p.relative_to(root).as_posix()}\n" for p in paths))
    archive = root.with_name(root.name+".tar")
    temp = archive.with_suffix(".partial.tar")
    with tarfile.open(temp, "w") as tar:
        for p in paths+[root / "checksums.sha256"]:
            tar.add(p, arcname=str(Path(root.name)/p.relative_to(root)), recursive=False)
    temp.replace(archive)
    archive.with_suffix(".tar.sha256").write_text(f"{sha256_file(archive)}  {archive.name}\n")
    return dict(archive=str(archive), size_mib=archive.stat().st_size/1024**2)
