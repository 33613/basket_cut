"""Isolated, portable MOTIP/KPR research exports; never updates a player library."""

import csv
import html
import json
import math
from collections import defaultdict
from pathlib import Path
import re
import shutil
import subprocess
import tarfile

from contracts.schema import read_jsonl, write_jsonl_line
from contracts.tracks import load_tracks
from pipeline.media import atomic_json, probe_video, sha256_file, verify_frame_clock


# User annotations: seconds from the beginning of each supplied clip.
DEMO_EVENTS = [(3, 10, "27"), (6, 12, "27"), (20, 3.5, "27"),
               (43, 10, "27"), (50, 5, "27"), (68, 27, "27"),
               (11, 7, "15"), (30, 13, "15"), (37, 10, "15"), (63, 12, "15")]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, fields):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def manifest(root):
    return read_json(Path(root) / "manifest.json")


def seed_annotations(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(output, [dict(video_id=str(v), filename=f"{v}.mp4", goal_time_s=t,
                            jersey_number=n, team_id="", person_gt_id="")
                       for v, t, n in DEMO_EVENTS],
              ["video_id", "filename", "goal_time_s", "jersey_number", "team_id", "person_gt_id"])


def prepare(input_dir, annotations, root, match_id, *, ffprobe="ffprobe"):
    input_dir, root = Path(input_dir).resolve(), Path(root).resolve()
    if input_dir == root or input_dir in root.parents or root in input_dir.parents:
        raise ValueError("Input and export directories must be separate, non-nested folders")
    rows = rows_csv(annotations)
    if not rows or not match_id.strip():
        raise ValueError("A nonempty annotation table and match_id are required")
    videos, seen = [], set()
    for row in rows:
        vid = row["video_id"].strip()
        if not re.fullmatch(r"[A-Za-z0-9_-]+", vid) or vid in seen:
            raise ValueError(f"Invalid/duplicate video_id: {vid}")
        seen.add(vid)
        source = (input_dir / row["filename"]).resolve()
        if not source.is_relative_to(input_dir) or not source.is_file():
            raise ValueError(f"Missing/unsafe input: {row['filename']}")
        meta = probe_video(source, ffprobe=ffprobe)
        meta["decoded_frame_count"] = verify_frame_clock(source, meta["fps"], ffprobe)
        goal = float(row["goal_time_s"])
        if not math.isfinite(goal) or not 0 <= goal < meta["duration_s"]:
            raise ValueError(f"{vid}: goal_time_s must lie inside the supplied clip")
        destination = f"video/{vid}{source.suffix.lower()}"
        videos.append(dict(video_id=vid, video_path=destination, source_sha256=sha256_file(source),
                           source_filename=row["filename"], metadata=meta,
                           event_id=f"EV-{vid}-001", goal_time_s=goal,
                           jersey_number=row.get("jersey_number", "").strip(),
                           team_id=row.get("team_id", "").strip(),
                           person_gt_id=row.get("person_gt_id", "").strip()))
    protocol = dict(schema_version=1, match_id=match_id, videos=videos,
                    time_convention="zero-based decoded frame_idx; timestamp_s=frame_idx/fps; goal seconds clip-local",
                    annotation_source="manual; jersey is an event label, not an inferred track identity")
    # Absolute input path is provenance only. Freeze the complete request on resume.
    if (root / "manifest.json").exists():
        if manifest(root) != protocol:
            raise ValueError("Input/annotations changed; use a new output directory")
        validate_videos(root)
        return protocol
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Unprepared output is not empty: {root}")
    (root / "video").mkdir(parents=True)
    for video in videos:
        shutil.copy2(input_dir / video["source_filename"], root / video["video_path"])
    shutil.copy2(annotations, root / "annotations.csv")
    write_csv(root / "shooter_selection.csv", [dict(video_id=v["video_id"], track_id="",
              start_frame="", end_frame="", notes="") for v in videos],
              ["video_id", "track_id", "start_frame", "end_frame", "notes"])
    atomic_json(root / "manifest.json", protocol)
    (root / "README.txt").write_text(
        "Research export, not an accuracy report. Start with review.html.\n"
        "video/: unchanged input files. track/<video_id>/: zero-based XYXY tracks and previews.\n"
        "track/<video_id>/samples/: native-size PNGs; samples.jsonl maps each to its source.\n"
        "identity/<video_id>/features.npz: embeddings, visibility_scores, sample_ids in identical order.\n"
        "identity/<video_id>/feature_contract.json: model/config hashes and feature-space definition.\n"
        "Track IDs are LOCAL to a video, may contain switches, and are NOT player identities.\n"
        "shooter_selection.csv is a MANUAL selection, not an automatic shooter prediction.\n"
        "Fill track_id after inspecting previews; optional inclusive start/end_frame restrict mixed tracks.\n"
        "Run --stage shooter afterwards. shooter crops are raw PNGs; crop videos are resized previews.\n"
        "annotation jersey/person_gt_id are evaluation labels. No automatic cross-video merge is performed.\n",
        encoding="utf-8")
    return protocol


def validate_videos(root):
    root = Path(root)
    for video in manifest(root)["videos"]:
        if sha256_file(root / video["video_path"]) != video["source_sha256"]:
            raise ValueError(f"Copied source changed: {video['video_id']}")


def track_signature(root, vid):
    folder = Path(root) / "track" / vid
    return {name: sha256_file(folder / name) for name in ("tracks.jsonl", "video_meta.json")}


def validated_tracks(root, video):
    folder = Path(root) / "track" / video["video_id"]
    meta = read_json(folder / "video_meta.json")
    if (meta["video_id"] != video["video_id"]
            or meta["processed_frames"] != video["metadata"]["decoded_frame_count"]
            or abs(meta["fps"] - video["metadata"]["fps"]) > 1e-4):
        raise ValueError("Tracking must cover the full clip with the requested video_id")
    if meta["processed_frames"] <= 0:
        raise ValueError("No decoded frames")
    return meta, load_tracks(folder / "tracks.jsonl", meta)


def sample_tracks(root, video, *, samples_per_track=12):
    """Temporal bins, best sharpness/size/detection candidate in each bin; no quality gate."""
    import cv2
    root = Path(root)
    folder = root / "track" / video["video_id"]
    meta, grouped = validated_tracks(root, video)
    protocol = dict(tracking=track_signature(root, video["video_id"]),
                    samples_per_track=samples_per_track, policy="temporal_bins_quality_v1")
    marker = folder / "samples_status.json"
    if marker.exists():
        old = read_json(marker)
        if old["protocol"] != protocol:
            raise ValueError("Sampling inputs changed; use a new output directory")
        if sha256_file(folder / "samples.jsonl") != old["manifest_sha256"]:
            raise ValueError("Sample manifest changed")
        for row in read_jsonl(folder / "samples.jsonl"):
            if sha256_file(root / row["crop_path"]) != row["crop_sha256"]:
                raise ValueError("Sample image changed")
        return
    destination = folder / "samples"
    if destination.exists():
        shutil.rmtree(destination)  # Incomplete derived stage only; raw evidence is untouched.
    destination.mkdir()
    by_frame = defaultdict(list)
    for tid, records in grouped.items():
        for r in records:
            span = records[-1].frame_idx - records[0].frame_idx + 1
            bucket = min(samples_per_track - 1,
                         (r.frame_idx - records[0].frame_idx) * samples_per_track // span)
            by_frame[r.frame_idx].append((r, bucket))
    capture = cv2.VideoCapture(str(root / video["video_path"]))
    best = {}
    try:
        for frame_idx in range(meta["processed_frames"]):
            ok, image = capture.read()
            if not ok:
                raise ValueError("Decode ended before tracking frames")
            for r, bucket in by_frame.get(frame_idx, []):
                x1, y1, x2, y2 = crop_bounds(r.bbox_xyxy, image.shape)
                if x2 <= x1 or y2 <= y1:
                    continue
                crop = image[y1:y2, x1:x2]
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                quality = math.log1p(sharpness) + math.log1p(crop.size) / 4 + r.det_score
                key = (r.track_id, bucket)
                if key in best and best[key][0] >= quality:
                    continue
                path = destination / f"T{r.track_id:05d}-B{bucket:02d}.png"
                if not cv2.imwrite(str(path), crop):
                    raise RuntimeError(f"Cannot write {path}")
                sample = dict(sample_id=f"{video['video_id']}-T{r.track_id:05d}-F{frame_idx:07d}",
                              video_id=video["video_id"], track_id=r.track_id, frame_idx=frame_idx,
                              timestamp_s=r.timestamp_s, bbox_xyxy=list(r.bbox_xyxy),
                              crop_xyxy=[x1, y1, x2, y2], det_score=r.det_score,
                              sharpness_laplacian_var=sharpness, sampling_quality=quality,
                              crop_path=path.relative_to(root).as_posix())
                best[key] = quality, sample
    finally:
        capture.release()
    samples = sorted((s for _, s in best.values()), key=lambda s: (s["track_id"], s["frame_idx"]))
    with (folder / "samples.jsonl").open("w", encoding="utf-8") as stream:
        for sample in samples:
            sample["crop_sha256"] = sha256_file(root / sample["crop_path"])
            write_jsonl_line(stream, sample)
    write_csv(folder / "track_review.csv", [dict(video_id=video["video_id"], track_id=tid,
              start_frame=rs[0].frame_idx, end_frame=rs[-1].frame_idx, observations=len(rs),
              track_status="unchecked", person_gt_id="", team_id="", jersey_number="", notes="")
              for tid, rs in grouped.items()],
              ["video_id", "track_id", "start_frame", "end_frame", "observations", "track_status",
               "person_gt_id", "team_id", "jersey_number", "notes"])
    atomic_json(marker, dict(protocol=protocol, sample_count=len(samples),
                             manifest_sha256=sha256_file(folder / "samples.jsonl")))


def crop_bounds(box, shape):
    h, w = shape[:2]
    return max(0, math.floor(box[0])), max(0, math.floor(box[1])), min(w, math.ceil(box[2])), min(h, math.ceil(box[3]))


def feature_protocol(root, video, checkpoint, config):
    folder = Path(root) / "track" / video["video_id"]
    if not (folder / "samples_status.json").is_file():
        raise ValueError("Run crops stage before extracting features")
    saved = read_json(folder / "samples_status.json")
    if (saved["manifest_sha256"] != sha256_file(folder / "samples.jsonl")
            or saved["protocol"]["tracking"] != track_signature(root, video["video_id"])):
        raise ValueError("Tracking/sample inputs changed after sampling")
    return dict(checkpoint_sha256=sha256_file(checkpoint), config_sha256=sha256_file(config),
                samples_sha256=sha256_file(folder / "samples.jsonl"), prompt_mode="none")


def features(root, video, backend, protocol, *, batch_size=16):
    import cv2
    import numpy as np
    root = Path(root)
    samples = list(read_jsonl(root / "track" / video["video_id"] / "samples.jsonl"))
    target = root / "identity" / video["video_id"]
    target.mkdir(parents=True, exist_ok=True)
    images = []
    for sample in samples:
        path = root / sample["crop_path"]
        if sha256_file(path) != sample["crop_sha256"]:
            raise ValueError("Sample image changed before feature extraction")
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Cannot read {path}")
        images.append(dict(image=image))  # Same OpenCV BGR convention as existing KPR adapter.
    if not images:
        raise ValueError(f"{video['video_id']}: no crops; inspect tracking before KPR")
    embeddings, visibility = backend.extract(images, batch_size=batch_size)
    embeddings, visibility = embeddings.numpy(), visibility.numpy()
    if (embeddings.ndim != 3 or visibility.shape != embeddings.shape[:2]
            or embeddings.shape[0] != len(samples)
            or not np.isfinite(embeddings).all() or not np.isfinite(visibility).all()):
        raise ValueError("Invalid/nonfinite KPR sample feature contract")
    temporary = target / ".features.tmp.npz"
    np.savez_compressed(temporary, embeddings=embeddings, visibility_scores=visibility,
                        sample_ids=np.asarray([s["sample_id"] for s in samples]),
                        track_ids=np.asarray([s["track_id"] for s in samples]),
                        frame_indices=np.asarray([s["frame_idx"] for s in samples]))
    temporary.replace(target / "features.npz")
    shutil.copy2(root / "track" / video["video_id"] / "samples.jsonl", target / "samples.jsonl")
    atomic_json(target / "feature_contract.json", dict(
        schema_version=1, protocol=protocol, status="extracted_not_accuracy_evaluated",
        test_embeddings=list(backend.cfg.model.kpr.test_embeddings),
        normalize_feature=bool(backend.cfg.test.normalize_feature),
        part_count=embeddings.shape[1], feature_dimension=embeddings.shape[2],
        sample_count=len(samples), color_convention="OpenCV BGR, existing KPR adapter",
        distance="euclidean_sqrt_visibility_mean_div2", min_common_parts=2,
        aggregation="per-sample; no cross-track or cross-video identity merge",
        feature_sha256=sha256_file(target / "features.npz")))


def features_complete(root, video, protocol):
    folder = Path(root) / "identity" / video["video_id"]
    path = folder / "feature_contract.json"
    if not path.exists():
        return False
    old = read_json(path)
    if old["protocol"] != protocol:
        raise ValueError("KPR inputs/config/weights changed; use a new export directory")
    if sha256_file(folder / "features.npz") != old["feature_sha256"]:
        raise ValueError("Saved KPR features changed")
    if sha256_file(folder / "samples.jsonl") != protocol["samples_sha256"]:
        raise ValueError("Saved feature sample index changed")
    return True


def selections(root, path):
    """Explicit human track selections. Jersey/time alone never chooses a track."""
    videos = {v["video_id"]: v for v in manifest(root)["videos"]}
    rows, seen, result = rows_csv(path), set(), []
    for row in rows:
        vid = row["video_id"]
        if vid not in videos or vid in seen:
            raise ValueError(f"Unknown/duplicate selection video: {vid}")
        seen.add(vid)
        if not row["track_id"].strip():
            continue
        tid = int(row["track_id"])
        meta, grouped = validated_tracks(root, videos[vid])
        if tid not in grouped:
            raise ValueError(f"{vid}: missing shooter track {tid}")
        start = int(row.get("start_frame") or grouped[tid][0].frame_idx)
        end = int(row.get("end_frame") or grouped[tid][-1].frame_idx)
        if not 0 <= start <= end < meta["processed_frames"]:
            raise ValueError(f"{vid}: invalid inclusive shooter frame interval")
        records = [r for r in grouped[tid] if start <= r.frame_idx <= end]
        if not records:
            raise ValueError(f"{vid}: selection contains no observed shooter frames")
        result.append((videos[vid], records, row))
    return result


def export_shooters(root, selection_path, *, ffmpeg="ffmpeg"):
    import cv2
    import numpy as np
    root = Path(root)
    selected = selections(root, selection_path)  # Validate all rows before modifying derived exports.
    for video, _, _ in selected:
        folder = root / "identity" / video["video_id"]
        contract = read_json(folder / "feature_contract.json")
        if not features_complete(root, video, contract["protocol"]):
            raise ValueError("Extract features before selecting shooters")
    for video in manifest(root)["videos"]:
        folder = root / "track" / video["video_id"] / "shooter"
        if folder.exists():
            shutil.rmtree(folder)
        for filename in ("shooter_features.npz", "shooter_samples.jsonl"):
            (root / "identity" / video["video_id"] / filename).unlink(missing_ok=True)
    for video, records, row in selected:
        folder = root / "track" / video["video_id"] / "shooter"
        temporary = folder.with_name(".shooter.partial")
        if temporary.exists():
            shutil.rmtree(temporary)
        (temporary / "crops").mkdir(parents=True)
        fps = read_json(folder.parent / "video_meta.json")["fps"]
        by_frame = {r.frame_idx: r for r in records}
        capture = cv2.VideoCapture(str(root / video["video_path"]))
        command = [ffmpeg, "-nostdin", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                   "-s", "256x512", "-r", str(fps), "-i", "pipe:0", "-an", "-c:v", "libx264",
                   "-threads", "2", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                   str(temporary / "trajectory.mp4")]
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            with (temporary / "frames.jsonl").open("w", encoding="utf-8") as stream:
                for idx in range(records[-1].frame_idx + 1):
                    ok, image = capture.read()
                    if not ok:
                        raise ValueError("Shooter decode ended early")
                    if idx < records[0].frame_idx:
                        continue
                    canvas = np.zeros((512, 256, 3), dtype=np.uint8)
                    r = by_frame.get(idx)
                    crop_path = None
                    if r:
                        x1, y1, x2, y2 = crop_bounds(r.bbox_xyxy, image.shape)
                        crop = image[y1:y2, x1:x2]
                        crop_path = f"crops/F{idx:07d}.png"
                        if not cv2.imwrite(str(temporary / crop_path), crop):
                            raise RuntimeError("Cannot save shooter crop")
                        ratio = min(256 / crop.shape[1], 512 / crop.shape[0])
                        w, h = max(1, round(crop.shape[1] * ratio)), max(1, round(crop.shape[0] * ratio))
                        resized = cv2.resize(crop, (w, h))
                        canvas[(512-h)//2:(512-h)//2+h, (256-w)//2:(256-w)//2+w] = resized
                    # Missing observations are black frames: preserve the original timeline.
                    process.stdin.write(canvas.tobytes())
                    write_jsonl_line(stream, dict(frame_idx=idx, timestamp_s=idx/fps,
                        preview_frame_idx=idx-records[0].frame_idx, observed=r is not None,
                        crop_path=crop_path, bbox_xyxy=list(r.bbox_xyxy) if r else None))
            process.stdin.close()
            error = process.stderr.read().decode(errors="replace")
            if process.wait():
                raise RuntimeError(f"Shooter FFmpeg failed: {error[-2000:]}")
            with (temporary / "tracks.jsonl").open("w", encoding="utf-8") as stream:
                for r in records:
                    write_jsonl_line(stream, r.to_dict())
            atomic_json(temporary / "selection.json", dict(event_id=video["event_id"],
                video_id=video["video_id"], track_id=records[0].track_id,
                selection_method="manual_track_selection", notes=row.get("notes", ""),
                goal_time_s=video["goal_time_s"], jersey_number_gt=video["jersey_number"],
                team_id_gt=video["team_id"], person_gt_id=video["person_gt_id"],
                start_frame=records[0].frame_idx, end_frame=records[-1].frame_idx,
                observations=len(records), accuracy_evaluated=False,
                tracking_signature=track_signature(root, video["video_id"])))
            temporary.replace(folder)
            identity = root / "identity" / video["video_id"]
            selected_keys = {(r.track_id, r.frame_idx) for r in records}
            samples = list(read_jsonl(identity / "samples.jsonl"))
            indices = [i for i, s in enumerate(samples) if (s["track_id"], s["frame_idx"]) in selected_keys]
            with np.load(identity / "features.npz", allow_pickle=False) as data:
                np.savez_compressed(identity / "shooter_features.npz", **{
                    key: data[key][indices] for key in data.files})
            with (identity / "shooter_samples.jsonl").open("w", encoding="utf-8") as stream:
                for i in indices:
                    write_jsonl_line(stream, samples[i])
        finally:
            capture.release()
            if process.poll() is None:
                process.kill()
                process.wait()
            process.stderr.close()
            if process.stdin and not process.stdin.closed:
                process.stdin.close()
    return len(selected)


def review(root):
    root = Path(root)
    sections = []
    for video in manifest(root)["videos"]:
        vid = video["video_id"]
        folder = root / "track" / vid
        groups = defaultdict(list)
        if (folder / "samples.jsonl").exists():
            for sample in read_jsonl(folder / "samples.jsonl"):
                groups[sample["track_id"]].append(sample)
        cards = []
        for tid, samples in groups.items():
            images = "".join(f'<figure><img src="{html.escape(s["crop_path"], quote=True)}">'
                             f'<figcaption>F{s["frame_idx"]} / {s["timestamp_s"]:.2f}s</figcaption></figure>'
                             for s in samples)
            cards.append(f'<details><summary>T{tid} ({len(samples)} samples)</summary><div>{images}</div></details>')
        vis = f"track/{vid}/tracks_vis.mp4"
        sections.append(f'<section><h2>Video {vid} / goal {video["goal_time_s"]}s / annotated jersey '
                        f'{html.escape(video["jersey_number"])}</h2><p><a href="{vis}">Tracking video</a> '
                        f'— inspect T IDs and fill shooter_selection.csv</p>' + "".join(cards) + '</section>')
    (root / "review.html").write_text('<!doctype html><meta charset="utf-8"><title>Goal test review</title>'
        '<style>body{font:16px sans-serif;margin:32px}section{margin:32px 0}details div{display:flex;flex-wrap:wrap}'
        'figure{margin:8px}img{height:220px;max-width:180px;object-fit:contain}</style>'
        '<h1>Goal test / manual shooter review</h1><p>Jersey labels are annotations, not predictions. '
        'Track IDs are local and unchecked. No automatic identity merge.</p>' + "".join(sections), encoding="utf-8")


def status(root):
    root = Path(root)
    data = manifest(root)
    rows = []
    for video in data["videos"]:
        vid = video["video_id"]
        folder = root / "track" / vid
        rows.append(dict(video_id=vid, goal_time_s=video["goal_time_s"], jersey_number=video["jersey_number"],
            tracked=(folder / "tracking_complete.json").exists(),
            samples=(folder / "samples_status.json").exists(),
            features=(root / "identity" / vid / "feature_contract.json").exists(),
            shooter_selected=(folder / "shooter" / "selection.json").exists()))
    return dict(match_id=data["match_id"], videos=rows, accuracy_evaluated=False,
                selected_shooters=sum(r["shooter_selected"] for r in rows), total=len(rows))


def pack(root):
    root = Path(root).resolve()
    validate_videos(root)
    report = status(root)
    if not all(r["tracked"] and r["samples"] and r["features"] for r in report["videos"]):
        raise ValueError("Tracking/crops/features must finish before packing")
    for video in manifest(root)["videos"]:
        vid = video["video_id"]
        folder = root / "track" / vid
        if read_json(folder / "tracking_complete.json")["outputs"] != track_signature(root, vid):
            raise ValueError("Tracking evidence changed before packing")
        protocol = read_json(root / "identity" / vid / "feature_contract.json")["protocol"]
        if protocol["samples_sha256"] != sha256_file(folder / "samples.jsonl"):
            raise ValueError("Sample index changed before packing")
        features_complete(root, video, protocol)
        for sample in read_jsonl(folder / "samples.jsonl"):
            if sha256_file(root / sample["crop_path"]) != sample["crop_sha256"]:
                raise ValueError("Crop changed before packing")
        selection_path = folder / "shooter" / "selection.json"
        if selection_path.exists() and read_json(selection_path)["tracking_signature"] != track_signature(root, vid):
            raise ValueError("Shooter selection refers to changed tracking")
    review(root)
    atomic_json(root / "export_status.json", dict(**report,
        status="exported_not_accuracy_evaluated", shooter_status=("all_manually_selected"
            if report["selected_shooters"] == report["total"] else "pending_manual_selection")))
    paths = sorted(p for p in root.rglob("*") if p.is_file() and not p.name.startswith(".")
                   and p.name != "checksums.sha256" and not any(x.startswith(".") for x in p.relative_to(root).parts))
    if any(p.is_symlink() for p in paths):
        raise ValueError("Portable export cannot contain symlinks")
    (root / "checksums.sha256").write_text("".join(
        f"{sha256_file(p)}  {p.relative_to(root).as_posix()}\n" for p in paths), encoding="utf-8")
    archive = root.with_name(root.name + ".tar")
    temporary = archive.with_suffix(".partial.tar")
    with tarfile.open(temporary, "w") as tar:
        for path in paths + [root / "checksums.sha256"]:
            tar.add(path, arcname=str(Path(root.name) / path.relative_to(root)), recursive=False)
    temporary.replace(archive)
    archive.with_suffix(".tar.sha256").write_text(f"{sha256_file(archive)}  {archive.name}\n")
    return dict(archive=str(archive), size_mib=archive.stat().st_size / 1024**2, **report)
