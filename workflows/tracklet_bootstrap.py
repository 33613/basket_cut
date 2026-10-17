"""Six clear, native-resolution tracking windows seed a match-scoped gallery.

Sparse detector probes choose windows only. Every identity observation comes
from a continuous track inside one window; no state crosses window boundaries.
"""

from collections import defaultdict
from dataclasses import asdict, dataclass
import fcntl
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from pipeline.media import atomic_json, cut_video, probe_video, sha256_file
from pipeline.identity.player_evidence import (
    COLORS,
    assessment_provenance,
    infer_team_map,
    parse_assessment,
    summarize,
)


@dataclass(frozen=True)
class WindowPolicy:
    windows: int = 6
    window_s: float = 4.0
    candidates_per_window: int = 4
    min_height: int = 128
    min_width: int = 40
    min_score: float = 0.65
    max_overlap: float = 0.15
    min_sharpness: float = 30.0
    min_gap_s: float = 0.5
    samples_per_track: int = 4
    min_track_s: float = 1.0
    min_free_gb: float = 2.0

    def validate(self):
        if (
            self.windows != 6
            or not 1 <= self.candidates_per_window <= 12
            or not 2 <= self.samples_per_track <= 12
            or self.min_height < 64
            or self.min_width < 24
            or not 1 <= self.window_s <= 15
            or not 0 < self.min_gap_s < self.window_s
            or not 0 < self.min_track_s < self.window_s
            or not 0 <= self.max_overlap <= 0.3
            or not 0.5 <= self.min_score <= 1
            or self.min_sharpness < 0
            or self.min_free_gb < 0
            or not all(math.isfinite(v) for v in asdict(self).values())
        ):
            raise ValueError("Invalid six-window initialization policy")
        return self


def candidate_starts(duration, policy):
    policy.validate()
    if duration < policy.windows * policy.window_s:
        raise ValueError("Source is too short for six independent windows")
    span = duration / policy.windows
    return [
        [
            i * span
            + (span - policy.window_s) * (j + 0.5) / policy.candidates_per_window
            for j in range(policy.candidates_per_window)
        ]
        for i in range(policy.windows)
    ]


def manual_starts(path, duration, policy):
    values = json.loads(Path(path).read_text())
    if not isinstance(values, list) or len(values) != 6:
        raise ValueError("Window file must be a JSON list of six source start seconds")
    starts = sorted(float(v) for v in values)
    if any(
        not math.isfinite(v) or v < 0 or v + policy.window_s > duration for v in starts
    ):
        raise ValueError("Manual window lies outside source")
    if any(b < a + policy.window_s for a, b in zip(starts, starts[1:])):
        raise ValueError("Initialization windows must not overlap")
    return [[v] for v in starts]


def choose_windows(selection, policy):
    """Prefer temporal coverage; fill a blurry stratum from other clear areas."""
    selected = []
    for candidates in selection:
        clear = [row for row in candidates if row["probe_clear_people"] > 0]
        if clear:
            selected.append(max(clear, key=lambda r: r["selection_score"]))
    for row in sorted(
        (r for group in selection for r in group), key=lambda r: -r["selection_score"]
    ):
        if len(selected) >= policy.windows:
            break
        if row["probe_clear_people"] > 0 and all(
            row["end_s"] <= old["start_s"] or old["end_s"] <= row["start_s"]
            for old in selected
        ):
            selected.append(row)
    if len(selected) != policy.windows:
        raise ValueError(
            "Fewer than six clear non-overlapping windows; supply --windows-file or use a clearer source"
        )
    return sorted(selected, key=lambda row: row["start_s"])


def clear_crop(frame, detection, others, policy):
    """Hard size/completeness/overlap gates before any model assessment."""
    import cv2
    from pipeline.identity.gallery_memory import box_overlap

    box = list(map(float, detection["bbox_xyxy"]))
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    area = max(0, x2 - x1) * max(0, y2 - y1)
    left, top, right, bottom = (
        max(0, int(x1)),
        max(0, int(y1)),
        min(w, int(x2)),
        min(h, int(y2)),
    )
    if (
        detection["det_score"] < policy.min_score
        or right - left < policy.min_width
        or bottom - top < policy.min_height
        or (right - left) * (bottom - top) < 0.97 * area
    ):
        return None
    # Border-cut bodies are unsuitable initial templates even if DETR's box
    # itself is clipped to the frame and has misleading completeness=1.
    if left <= 1 or top <= 1 or right >= w - 1 or bottom >= h - 1:
        return None
    overlap = max(
        (
            box_overlap(box, o["bbox_xyxy"])[1]
            for o in others
            if o is not detection and o["det_score"] >= 0.3
        ),
        default=0.0,
    )
    if overlap > policy.max_overlap:
        return None
    crop = frame[top:bottom, left:right]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (min(128, right - left), min(192, bottom - top)))
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    exposure = float(np.mean((gray > 5) & (gray < 250)))
    if sharpness < policy.min_sharpness or exposure < 0.5:
        return None
    quality = (
        0.45 * min(1.0, math.log1p(sharpness) / math.log1p(1000))
        + 0.35 * min(1.0, math.sqrt((right - left) * (bottom - top) / (96 * 192)))
        + 0.2 * exposure
    )
    return crop, {
        "bbox_xyxy": [left, top, right, bottom],
        "quality": quality,
        "sharpness": sharpness,
        "overlap_fraction": overlap,
        "det_score": float(detection["det_score"]),
    }


def folder(run):
    return Path(run).resolve() / "bootstrap_tracklets"


def load_manifest(run, *, verify=True):
    root = folder(run)
    data = json.loads((root / "manifest.json").read_text())
    if data.get("status") != "tracked":
        raise ValueError("Six-window tracking is incomplete")
    if verify:
        source = data["protocol"]["source"]
        if sha256_file(source["path"]) != source["sha256"]:
            raise ValueError("Initialization source changed")
        for relative, digest in data["artifacts"].items():
            path = (root / relative).resolve()
            if not path.is_relative_to(root) or sha256_file(path) != digest:
                raise ValueError(
                    "Initialization window/crop evidence changed: " + relative
                )
    return data


def prepare(
    source,
    run,
    backend,
    *,
    policy=None,
    ffmpeg="ffmpeg",
    ffprobe="ffprobe",
    detector=None,
    windows_file=None,
):
    import cv2
    import shutil

    policy = (policy or WindowPolicy()).validate()
    source = Path(source).resolve()
    root = folder(run)
    if source.is_relative_to(Path(run).resolve()):
        raise ValueError("Source must be outside initialization results")
    if (Path(run) / "bootstrap/photos.json").exists():
        raise ValueError(
            "Legacy photo initialization exists; use a fresh output directory"
        )
    meta = probe_video(source, ffprobe=ffprobe)
    digest = sha256_file(source)
    source_meta = {**meta, "sha256": digest, "source_video_id": "SRC-" + digest[:20]}
    groups = (
        manual_starts(windows_file, meta["duration_s"], policy)
        if windows_file
        else candidate_starts(meta["duration_s"], policy)
    )
    protocol = {
        "version": 1,
        "mode": "six_clear_tracking_windows",
        "source": source_meta,
        "policy": asdict(policy),
        "detector": detector,
        "candidate_starts": groups,
        "media": "native_resolution_ffv1_cfr_no_spatial_compression",
    }
    old = root / "manifest.json"
    if old.exists():
        if json.loads(old.read_text())["protocol"] != protocol:
            raise ValueError(
                "Initialization protocol changed; use a new output directory"
            )
        return load_manifest(run)
    root.mkdir(parents=True, exist_ok=True)
    protocol_path = root / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("Incomplete initialization has a different protocol")
    atomic_json(protocol_path, protocol)
    atomic_json(
        root / "status.json",
        {"stage": "window_selection", "windows_completed": 0, "windows_total": 6},
    )
    cached_selection = root / "selection.json"
    saved = (
        json.loads(cached_selection.read_text()) if cached_selection.is_file() else {}
    )
    selection = saved.get("candidates", []) if saved.get("status") == "selected" else []
    if selection:
        if len(selection) != 6 or any(
            [r["start_s"] for r in rows] != group
            for rows, group in zip(selection, groups)
        ):
            raise ValueError(
                "Saved window selection differs from initialization protocol"
            )
        selected = saved["windows"]
    else:
        capture = cv2.VideoCapture(str(source))
        try:
            for group in groups:
                scored = []
                for start in group:
                    center = start + policy.window_s / 2
                    capture.set(cv2.CAP_PROP_POS_MSEC, center * 1000)
                    ok, frame = capture.read()
                    if not ok:
                        raise ValueError("Cannot decode selection probe")
                    tracker = backend.create_tracker(
                        height=frame.shape[0], width=frame.shape[1], only_detr=True
                    )
                    detections = backend.infer_frame(tracker, frame)
                    clear = [
                        v
                        for d in detections
                        if (v := clear_crop(frame, d, detections, policy)) is not None
                    ]
                    scores = sorted((v[1]["quality"] for v in clear), reverse=True)
                    scored.append(
                        {
                            "start_s": start,
                            "end_s": start + policy.window_s,
                            "probe_clear_people": len(clear),
                            "selection_score": sum(scores[:10]),
                        }
                    )
                selection.append(scored)
            try:
                selected = choose_windows(selection, policy)
            except ValueError:
                atomic_json(
                    root / "selection.json",
                    {"status": "needs_clearer_windows", "candidates": selection},
                )
                raise
            atomic_json(
                root / "selection.json",
                {"status": "selected", "candidates": selection, "windows": selected},
            )
        finally:
            capture.release()
    if selected != choose_windows(selection, policy):
        raise ValueError("Saved selected windows changed")
    windows, all_samples, artifacts = [], [], {}
    for number, chosen in enumerate(selected):
        wid = f"WIN-{number + 1:02d}"
        done = root / f"{wid}.json"
        # Window checkpoints allow a killed run to retain completed tracking.
        if done.is_file():
            prior = json.loads(done.read_text())
            if any(
                prior["window"][k] != chosen[k]
                for k in ("start_s", "end_s", "selection_score")
            ):
                raise ValueError("Completed window differs from saved selection")
            for relative, expected in prior["artifacts"].items():
                if sha256_file(root / relative) != expected:
                    raise ValueError("Completed window changed")
            windows.append(prior["window"])
            all_samples.extend(prior["samples"])
            artifacts.update(prior["artifacts"])
            continue
        if shutil.disk_usage(root).free < policy.min_free_gb * 1024**3:
            raise OSError("Insufficient free space for lossless initialization windows")
        path = root / "windows" / (wid + ".mkv")
        media = cut_video(
            source,
            path,
            chosen["start_s"],
            chosen["end_s"],
            ffmpeg=ffmpeg,
            ffprobe=ffprobe,
            metadata=meta,
            lossless=True,
        )
        cap = cv2.VideoCapture(str(path))
        tracker = backend.create_tracker(
            height=meta["height"], width=meta["width"], only_detr=False
        )
        pools, spans = defaultdict(list), {}
        track_path = root / (wid + "-tracks.jsonl")
        frame_idx = 0
        try:
            with track_path.open("w") as stream:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    detections = backend.infer_frame(tracker, frame)
                    for detection in detections:
                        tid = int(detection["track_id"])
                        t = frame_idx / media["fps"]
                        span = spans.setdefault(tid, [t, t])
                        span[1] = t + 1 / media["fps"]
                        stream.write(
                            json.dumps(
                                {**detection, "frame_idx": frame_idx, "timestamp_s": t}
                            )
                            + "\n"
                        )
                        candidate = clear_crop(frame, detection, detections, policy)
                        if candidate is None:
                            continue
                        crop, info = candidate
                        bank = pools[tid]
                        # Retain the best clear crop in each temporal bin;
                        # independent support is checked again after Qwen.
                        bucket = int(t / policy.min_gap_s)
                        existing = next(
                            (i for i, x in enumerate(bank) if x[0] == bucket), None
                        )
                        item = (
                            bucket,
                            {**info, "frame_idx": frame_idx, "timestamp_s": t},
                            crop.copy(),
                        )
                        if existing is None:
                            bank.append(item)
                        elif info["quality"] > bank[existing][1]["quality"]:
                            bank[existing] = item
                        bank.sort(key=lambda x: -x[1]["quality"])
                        del bank[24:]
                    frame_idx += 1
                    if frame_idx % 50 == 0:
                        print(f"[initialize] {wid}: {frame_idx} frames", flush=True)
        finally:
            cap.release()
        if frame_idx < max(2, int(policy.window_s * media["fps"]) - 2):
            raise ValueError("Initialization window decoding stopped early")
        samples = []
        for tid, bank in sorted(pools.items()):
            if spans[tid][1] - spans[tid][0] < policy.min_track_s:
                continue
            picked = []
            for _, info, crop in bank:
                if all(
                    abs(info["timestamp_s"] - p[0]["timestamp_s"]) >= policy.min_gap_s
                    for p in picked
                ):
                    picked.append((info, crop))
                if len(picked) >= policy.samples_per_track:
                    break
            if len(picked) < 2:
                continue
            for info, crop in sorted(picked, key=lambda x: x[0]["frame_idx"]):
                sid = f"{wid}-T{tid:05d}-F{info['frame_idx']:06d}"
                image = root / "media" / (sid + ".png")
                image.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(image), crop):
                    raise OSError("Cannot write native person crop")
                samples.append(
                    {
                        **info,
                        "sample_id": sid,
                        "window_id": wid,
                        "raw_track_id": tid,
                        "track_key": f"{wid}-T{tid:05d}",
                        "source_timestamp_s": chosen["start_s"] + info["timestamp_s"],
                        "interval": spans[tid],
                        "crop_path": str(image.relative_to(root)),
                        "crop_sha256": sha256_file(image),
                    }
                )
        window = {
            **chosen,
            "window_id": wid,
            "path": str(path.relative_to(root)),
            "frames": frame_idx,
            "media": media,
            "qualified_tracks": len({s["track_key"] for s in samples}),
        }
        hashes = {str(p.relative_to(root)): sha256_file(p) for p in [path, track_path]}
        hashes.update({s["crop_path"]: s["crop_sha256"] for s in samples})
        atomic_json(done, {"window": window, "samples": samples, "artifacts": hashes})
        artifacts.update(hashes)
        windows.append(window)
        all_samples.extend(samples)
        atomic_json(
            root / "status.json",
            {
                "stage": "tracking",
                "windows_completed": len(windows),
                "windows_total": 6,
            },
        )
        print(
            f"[initialize] {wid} complete; {window['qualified_tracks']} clear tracklets",
            flush=True,
        )
    artifacts["selection.json"] = sha256_file(root / "selection.json")
    result = {
        "status": "tracked",
        "protocol": protocol,
        "windows": windows,
        "samples": all_samples,
        "artifacts": artifacts,
    }
    atomic_json(old, result)
    return result


def assess(run, backend, *, team_map=None):
    from PIL import Image

    root = folder(run)
    manifest = load_manifest(run)
    signature = {
        "manifest": sha256_file(root / "manifest.json"),
        "model": assessment_provenance(backend.provenance),
        "team_map": team_map,
    }
    path = root / "assessments.json"
    previous = json.loads(path.read_text()) if path.exists() else {}
    if previous and previous["signature"] != signature:
        raise ValueError("Qwen evidence changed; use a new run")
    if previous.get("status") == "completed":
        palette_path = root / "team_map.json"
        if (
            palette_path.exists()
            and json.loads(palette_path.read_text()) != previous["team_map"]
        ):
            raise ValueError("Registered team palette changed; use a new run")
        atomic_json(palette_path, previous["team_map"])
        return previous
    readings = previous.get("readings", {})

    def progress(stage):
        atomic_json(
            root / "status.json",
            {
                "stage": stage,
                "windows_completed": 6,
                "windows_total": 6,
                "samples_completed": len(readings),
                "samples_total": len(manifest["samples"]),
            },
        )

    progress("player_assessment")
    for row in manifest["samples"]:
        key = row["sample_id"]
        if key not in readings:
            with Image.open(root / row["crop_path"]) as image:
                raw = backend.assess_player(image.convert("RGB"))
            readings[key] = {
                **parse_assessment(raw),
                "raw_response": raw,
                "frame_idx": row["frame_idx"],
                "timestamp_s": row["timestamp_s"],
                "source_timestamp_s": row["source_timestamp_s"],
                "quality_score": row["quality"],
                "crop_path": row["crop_path"],
            }
            atomic_json(
                path,
                {"status": "running", "signature": signature, "readings": readings},
            )
            progress("player_assessment")
            print(
                f"[initialize] assessed {len(readings)}/{len(manifest['samples'])} clear samples",
                flush=True,
            )
    grouped = defaultdict(list)
    for row in manifest["samples"]:
        grouped[row["track_key"]].append(readings[row["sample_id"]])
    palette = team_map if team_map is not None else infer_team_map(grouped.values())
    if not isinstance(palette, dict) or any(
        k not in COLORS or not isinstance(v, str) or not v for k, v in palette.items()
    ):
        raise ValueError("Team map must map known shirt colors to nonempty team IDs")
    tracks = {
        key: summarize(
            rows, team_map=palette, gap=manifest["protocol"]["policy"]["min_gap_s"]
        )
        for key, rows in grouped.items()
    }
    result = {
        "status": "completed",
        "signature": signature,
        "readings": readings,
        "tracks": tracks,
        "team_map": palette,
    }
    atomic_json(path, result)
    atomic_json(root / "team_map.json", palette)
    progress("player_assessment_completed")
    return result


def accepted_samples(manifest, assessments):
    return [
        row
        for row in manifest["samples"]
        if assessments["readings"][row["sample_id"]]["usable"]
        and assessments["tracks"][row["track_key"]]["clear_support"] >= 2
        and assessments["tracks"][row["track_key"]]["jersey"]["status"] != "conflict"
    ]


def template_evidence(run):
    """Regenerate the same immutable image banks for maintenance and rollback."""
    root = folder(run)
    manifest = load_manifest(run)
    assessments = json.loads((root / "assessments.json").read_text())
    contract = json.loads((root / "feature_contract.json").read_text())
    if (
        contract["signature"]["manifest"] != sha256_file(root / "manifest.json")
        or contract["signature"]["assessments"]
        != sha256_file(root / "assessments.json")
        or contract["features_sha256"] != sha256_file(root / "features.npz")
        or contract["signature"]["team_map"] != sha256_file(root / "team_map.json")
    ):
        raise ValueError("Initialization feature evidence changed")
    rows = accepted_samples(manifest, assessments)
    if [r["sample_id"] for r in rows] != contract["sample_ids"]:
        raise ValueError("KPR sample order changed")
    with np.load(root / "features.npz", allow_pickle=False) as data:
        vectors, visible = data["embeddings"], data["visibility_scores"]
    if len(vectors) != len(rows):
        raise ValueError("KPR sample count changed")
    banks = defaultdict(list)
    for i, row in enumerate(rows):
        banks[row["track_key"]].append(
            {
                "key": row["sample_id"],
                "feature": vectors[i].tolist(),
                "visibility": visible[i].tolist(),
                "quality": row["quality"],
                "crop_path": row["crop_path"],
                "media_kind": "bootstrap_tracklet",
                "kind": "clear_track_image",
                "view": assessments["readings"][row["sample_id"]]["view"],
                "source_frame": f"{row['window_id']}:{row['frame_idx']}",
                "timestamp_s": row["source_timestamp_s"],
            }
        )
    return {
        key: {
            "templates": bank,
            "source_fingerprint": hashlib.sha256(
                json.dumps(bank, sort_keys=True).encode()
            ).hexdigest(),
        }
        for key, bank in banks.items()
    }


def extract_features(run, backend, checkpoint):
    import cv2

    root = folder(run)
    manifest = load_manifest(run)
    assessments = json.loads((root / "assessments.json").read_text())
    if assessments["status"] != "completed" or assessments["signature"][
        "manifest"
    ] != sha256_file(root / "manifest.json"):
        raise ValueError("Clear-player assessment incomplete or changed")
    if json.loads((root / "team_map.json").read_text()) != assessments["team_map"]:
        raise ValueError("Team palette differs from assessed evidence")
    signature = {
        "manifest": sha256_file(root / "manifest.json"),
        "assessments": sha256_file(root / "assessments.json"),
        "checkpoint": sha256_file(checkpoint),
        "team_map": sha256_file(root / "team_map.json"),
    }
    path = root / "feature_contract.json"
    if path.exists():
        prior = json.loads(path.read_text())
        if prior["signature"] != signature or prior["features_sha256"] != sha256_file(
            root / "features.npz"
        ):
            raise ValueError("KPR evidence changed")
        return prior
    rows = accepted_samples(manifest, assessments)
    images = [{"image": cv2.imread(str(root / r["crop_path"]))} for r in rows]
    atomic_json(
        root / "status.json",
        {
            "stage": "kpr_features",
            "windows_completed": 6,
            "windows_total": 6,
            "clear_samples": len(rows),
        },
    )
    if any(s["image"] is None for s in images):
        raise ValueError("Cannot decode clear crop")
    vectors, visible = (
        backend.extract(images, batch_size=16)
        if images
        else (np.empty((0, 0, 0)), np.empty((0, 0)))
    )
    vectors, visible = np.asarray(vectors), np.asarray(visible)
    if (
        vectors.ndim != 3
        or visible.shape != vectors.shape[:2]
        or len(vectors) != len(rows)
    ):
        raise ValueError("Invalid KPR sample layout")
    vectors = vectors / np.maximum(
        np.linalg.norm(vectors, axis=-1, keepdims=True), 1e-12
    )
    from pipeline.identity.cross_clip import part_distances

    if len(vectors):
        part_distances(vectors, visible)
    temporary = root / ".features.tmp.npz"
    np.savez_compressed(temporary, embeddings=vectors, visibility_scores=visible)
    temporary.replace(root / "features.npz")
    metadata = {
        "schema_version": 1,
        "checkpoint_sha256": signature["checkpoint"],
        "prompt_mode": "none",
        "test_embeddings": list(backend.cfg.model.kpr.test_embeddings),
        "part_count": int(vectors.shape[1]),
        "feature_dimension": int(vectors.shape[2]),
        "distance": "euclidean_sqrt_visibility_mean_div2",
    }
    contract = {
        "signature": signature,
        "metadata": metadata,
        "shape": list(vectors.shape[1:]),
        "sample_ids": [r["sample_id"] for r in rows],
        "features_sha256": sha256_file(root / "features.npz"),
    }
    atomic_json(path, contract)
    atomic_json(
        root / "status.json",
        {
            "stage": "kpr_features_completed",
            "windows_completed": 6,
            "windows_total": 6,
            "clear_samples": len(rows),
        },
    )
    return contract


def register(run, match_id):
    from pipeline.identity.gallery_memory import MemoryPolicy
    from pipeline.identity.player_registry import PlayerRegistry
    from workflows.person_library import export_library

    root = folder(run)
    manifest = load_manifest(run)
    assessments = json.loads((root / "assessments.json").read_text())
    contract = json.loads((root / "feature_contract.json").read_text())
    if (
        contract["signature"]["manifest"] != sha256_file(root / "manifest.json")
        or contract["signature"]["assessments"]
        != sha256_file(root / "assessments.json")
        or contract["features_sha256"] != sha256_file(root / "features.npz")
    ):
        raise ValueError("Initialization model evidence changed")
    rows = accepted_samples(manifest, assessments)
    if [r["sample_id"] for r in rows] != contract["sample_ids"]:
        raise ValueError("KPR sample ordering changed")
    with np.load(root / "features.npz", allow_pickle=False) as data:
        vectors, visible = data["embeddings"], data["visibility_scores"]
    if len(vectors) != len(rows):
        raise ValueError("KPR feature count changed")
    grouped = defaultdict(list)
    for i, row in enumerate(rows):
        grouped[row["track_key"]].append(i)
    source = manifest["protocol"]["source"]
    windows = {w["window_id"]: w for w in manifest["windows"]}
    nodes = []
    curation = template_evidence(run)
    signatures = {
        str(p.relative_to(Path(run).resolve())): [
            p.stat().st_size,
            p.stat().st_mtime_ns,
        ]
        for p in [
            root / "manifest.json",
            root / "assessments.json",
            root / "feature_contract.json",
            root / "features.npz",
            root / "team_map.json",
        ]
    }
    for key, indices in sorted(grouped.items()):
        sample = rows[indices[0]]
        window = windows[sample["window_id"]]
        evidence = assessments["tracks"][key]
        evidence = {
            **evidence,
            "jersey": {
                **evidence["jersey"],
                "identity_evidence_contract": {
                    "model": assessments["signature"]["model"],
                    "team_map": assessments["team_map"],
                },
            },
        }
        cover = max((rows[i] for i in indices), key=lambda r: r["quality"])
        node = {
            "node_id": key,
            "clip_name": sample["window_id"],
            "video_id": sample["window_id"],
            "evidence_kind": "bootstrap_tracklet",
            "filename": Path(source["path"]).name,
            "local_person_id": key,
            "source_video_id": source["source_video_id"],
            "source_sha256": source["sha256"],
            "source_clip_start_s": window["start_s"],
            "source_clip_end_s": window["end_s"],
            "raw_track_ids": [sample["raw_track_id"]],
            "intervals": [sample["interval"]],
            "sample_count": len(indices),
            "observation_count": len(indices),
            "hold_reasons": [],
            "cover": {
                "crop_path": cover["crop_path"],
                "archive_quality_score": cover["quality"],
            },
            "exemplars": [],
            "feature_indices": indices,
            **evidence,
        }
        nodes.append(node)
        for i in indices:
            row = rows[i]
            path = root / row["crop_path"]
            signatures[str(path.relative_to(Path(run).resolve()))] = [
                path.stat().st_size,
                path.stat().st_mtime_ns,
            ]
    library = Path(run).resolve() / "library"
    library.mkdir(parents=True, exist_ok=True)
    if (
        (library / "library_manifest.json").exists()
        or (root / "registration.json").exists()
    ) and not (library / "players.sqlite3").exists():
        raise ValueError("Player database missing; restore stable IDs")
    with (library / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with PlayerRegistry(library / "players.sqlite3", match_id) as registry:
            policy = (
                MemoryPolicy(**json.loads(registry.meta("memory_policy")))
                if registry.meta("memory_policy")
                else MemoryPolicy(gallery_limit=6)
            )
            registry.ingest(
                nodes,
                vectors,
                visible,
                [],
                {k: contract[k] for k in ("metadata", "shape")} if len(rows) else None,
                max_distance=policy.supported_distance,
                novelty_distance=policy.novelty_distance,
                min_margin=policy.min_margin,
                memory_policy=policy,
                curation=curation,
            )
            result = export_library(
                registry, Path(run).resolve(), library, signatures=signatures
            )
    registration = {
        "status": "completed_not_accuracy_evaluated",
        "mode": "six_clear_tracking_windows",
        "match_id": match_id,
        "window_count": 6,
        "tracked_samples": len(manifest["samples"]),
        "accepted_clear_samples": len(rows),
        "tracklet_count": len(nodes),
        "discarded_samples": len(manifest["samples"]) - len(rows),
        "team_map": assessments["team_map"],
        "revision": result["review_revision"],
        "candidate_players": result["candidate_person_count"],
        "ready_for_incremental": result["candidate_person_count"] > 0,
        "diagnostics": (
            []
            if result["candidate_person_count"]
            else [
                "no_clear_team_number_anchor; inspect windows, assessments and team map"
            ]
        ),
    }
    atomic_json(root / "registration.json", registration)
    atomic_json(root / "status.json", registration)
    return registration
