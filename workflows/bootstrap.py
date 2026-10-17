"""Legacy photo-run reproduction and audits; new initialization uses tracklets."""

import fcntl
import json
from pathlib import Path
import random

import numpy as np

from pipeline.media import atomic_json, probe_video, sha256_file
from workflows.person_library import export_library
from pipeline.identity.player_registry import PlayerRegistry


def sample_indices(duration, fps, count=24, seed=42):
    """One seeded random frame per time stratum, without duplicate frames."""
    if not 1 <= count <= 256 or count > int(duration * fps):
        raise ValueError("Choose 1–256 photos, no more than the source frames")
    rng = random.Random(seed)
    length = int(duration * fps)
    return [rng.randrange(i * length // count, (i + 1) * length // count) for i in range(count)]


def verified_photos(run):
    root = Path(run).resolve() / "bootstrap"
    photos = json.loads((root / "photos.json").read_text())
    source = photos["protocol"]["source"]
    if photos["status"] != "completed" or sha256_file(source["path"]) != source["sha256"]:
        raise ValueError("Bootstrap source/detection evidence is incomplete or changed")
    for row in photos["samples"]:
        for key in ("crop_path", "torso_path"):
            path = (root / row[key]).resolve()
            if not path.is_relative_to(root) or sha256_file(path) != row[key + "_sha256"]:
                raise ValueError("Bootstrap photo evidence changed")
    return photos


def prepare(source, run, backend, *, count=24, seed=42, ffprobe="ffprobe", detector_contract=None):
    import cv2
    source, run = Path(source).resolve(), Path(run).resolve()
    if source.is_relative_to(run):
        raise ValueError("Bootstrap results must not contain the source")
    meta = probe_video(source, ffprobe=ffprobe)
    checksum = sha256_file(source)
    protocol = {"source": {**meta, "sha256": checksum, "source_video_id": "SRC-" + checksum[:20]},
        "count": count, "seed": seed, "sampling": "stratified_random_frames_v1",
        "detector": detector_contract, "code_sha256": sha256_file(Path(__file__)),
        "policy": "single_photo_candidate; no cross-frame MOTIP association; not verified players"}
    output = run / "bootstrap"
    path = output / "photos.json"
    if path.is_file():
        previous = json.loads(path.read_text())
        if previous["protocol"] != protocol:
            raise ValueError("Bootstrap source/settings changed; use a new match run")
        if previous["status"] == "completed":
            return verified_photos(run)
    indices = sample_indices(meta["duration_s"], meta["fps"], count, seed)
    output.mkdir(parents=True, exist_ok=True)
    result = {"protocol": protocol, "status": "running", "frames": [], "samples": []}
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError("Cannot decode bootstrap source")
    try:
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            timestamp = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if not ok or abs(timestamp - index / meta["fps"]) > 2 / meta["fps"]:
                raise ValueError("Bootstrap seek/time alignment failed; use a CFR source")
            frame_id = f"SEED-{checksum[:12]}-{index:08d}"
            # Reset the tracker for EVERY independent photograph.
            tracker = backend.create_tracker(height=frame.shape[0], width=frame.shape[1])
            detections = backend.infer_frame(tracker, frame)
            result["frames"].append({"frame_id": frame_id, "frame_idx": index,
                "timestamp_s": timestamp, "detection_count": len(detections)})
            for number, detection in enumerate(detections):
                box = detection["bbox_xyxy"]
                left, top = max(0, int(box[0])), max(0, int(box[1]))
                right, bottom = min(frame.shape[1], int(box[2])), min(frame.shape[0], int(box[3]))
                if detection["det_score"] < .5 or right - left < 24 or bottom - top < 64:
                    continue
                crop = frame[top:bottom, left:right]
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                small = cv2.resize(gray, (min(128, gray.shape[1]), min(192, gray.shape[0])))
                sharpness = float(cv2.Laplacian(small, cv2.CV_64F).var())
                exposure = float(np.mean((gray > 5) & (gray < 250)))
                if sharpness < 5 or exposure < .1:
                    continue
                sid = f"{frame_id}-D{number:03d}"
                crops = {"crop_path": (f"media/{sid}.png", crop),
                         "torso_path": (f"media/{sid}-torso.png", crop[int(.12*len(crop)):int(.7*len(crop)),
                                                    int(.08*crop.shape[1]):int(.92*crop.shape[1])])}
                row = {"sample_id": sid, "frame_id": frame_id, "frame_idx": index,
                    "timestamp_s": timestamp, "bbox_xyxy": [left, top, right, bottom],
                    "det_score": detection["det_score"], "sharpness": sharpness,
                    "quality_score": min(1., sharpness / 1000) * exposure}
                for key, (relative, image) in crops.items():
                    target = output / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if not cv2.imwrite(str(target), image):
                        raise OSError("Cannot save bootstrap crop")
                    row[key], row[key + "_sha256"] = relative, sha256_file(target)
                result["samples"].append(row)
            atomic_json(path, result)
            print(f"[bootstrap] photo {index} at {timestamp:.2f}s; crops {len(result['samples'])}", flush=True)
        result["status"] = "completed"
        atomic_json(path, result)
        return result
    finally:
        capture.release()


def extract_features(run, backend, checkpoint):
    import cv2
    root = Path(run) / "bootstrap"
    photos = verified_photos(run)
    if (root / "registration.json").is_file():
        raise ValueError("Registered bootstrap features are immutable")
    path = root / "features.npz"
    samples = []
    for row in photos["samples"]:
        image = cv2.imread(str(root / row["crop_path"]))
        if image is None:
            raise ValueError("Cannot decode bootstrap person crop")
        # Same direct OpenCV BGR interface as track archive extraction.
        samples.append({"image": image})
    if samples:
        vectors, visibility = backend.extract(samples, batch_size=16)
        vectors, visibility = np.asarray(vectors), np.asarray(visibility)
        if vectors.ndim != 3 or visibility.shape != vectors.shape[:2] or len(vectors) != len(samples):
            raise ValueError("KPR photo feature layout/count mismatch")
        # Match the track archive's per-part prototype normalization even if
        # upstream cfg.test.normalize_feature is disabled.
        norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
        vectors = vectors / np.maximum(norms, 1e-12)
        from pipeline.identity.cross_clip import part_distances
        part_distances(vectors, visibility)
    else:
        vectors, visibility = np.empty((0, 0, 0)), np.empty((0, 0))
    contract = {"schema_version": 1, "checkpoint_sha256": sha256_file(checkpoint),
        "prompt_mode": "none", "test_embeddings": list(backend.cfg.model.kpr.test_embeddings),
        "part_count": int(vectors.shape[1]), "feature_dimension": int(vectors.shape[2]),
        "distance": "euclidean_sqrt_visibility_mean_div2"}
    temporary = root / ".features.tmp.npz"
    np.savez_compressed(temporary, embeddings=vectors, visibility_scores=visibility)
    temporary.replace(path)
    atomic_json(root / "feature_contract.json", {"metadata": contract, "shape": list(vectors.shape[1:]),
        "photos_sha256": sha256_file(root / "photos.json"), "features_sha256": sha256_file(path)})


def read_numbers(run, backend):
    from PIL import Image
    from pipeline.identity.qwen_backend import parse_reading
    root = Path(run) / "bootstrap"
    photos = verified_photos(run)
    if (root / "registration.json").is_file():
        raise ValueError("Registered bootstrap number evidence is immutable")
    path = root / "numbers.json"
    previous = json.loads(path.read_text()) if path.is_file() else {}
    signature = {"photos_sha256": sha256_file(root / "photos.json"), "model": backend.provenance}
    if previous and previous["signature"] != signature:
        raise ValueError("Bootstrap Qwen evidence changed; use a new run")
    readings = previous.get("readings", {})
    for row in photos["samples"]:
        if row["sample_id"] in readings:
            continue
        with Image.open(root / row["torso_path"]) as image:
            raw = backend.read(image.convert("RGB"))
        readings[row["sample_id"]] = {**parse_reading(raw), "raw_response": raw,
            "timestamp_s": row["timestamp_s"], "frame_idx": row["frame_idx"],
            "crop_path": row["torso_path"]}
        atomic_json(path, {"signature": signature, "readings": readings, "status": "running"})
    atomic_json(path, {"signature": signature, "readings": readings, "status": "completed"})


def register(run, match_id, *, max_distance=.2, novelty_distance=.5, min_margin=.05):
    root = Path(run).resolve()
    seed = root / "bootstrap"
    photos = verified_photos(root)
    features = json.loads((seed / "feature_contract.json").read_text())
    numbers = json.loads((seed / "numbers.json").read_text())
    if (photos["status"] != "completed" or numbers["status"] != "completed"
            or features["photos_sha256"] != sha256_file(seed / "photos.json")
            or numbers["signature"]["photos_sha256"] != features["photos_sha256"]
            or features["features_sha256"] != sha256_file(seed / "features.npz")):
        raise ValueError("Bootstrap evidence is incomplete or changed")
    with np.load(seed / "features.npz", allow_pickle=False) as arrays:
        vectors, visibility = arrays["embeddings"], arrays["visibility_scores"]
    if len(vectors) != len(photos["samples"]):
        raise ValueError("Bootstrap feature/photo count differs")
    contract = {k: features[k] for k in ("metadata", "shape")} if len(vectors) else None
    nodes = []
    for i, row in enumerate(photos["samples"]):
        reading = numbers["readings"][row["sample_id"]]
        # One image is never a multi-frame number consensus.
        jersey = {"number": None, "status": "insufficient_evidence", "verified": False,
            "candidate_numbers": [reading["number"]] if reading["number"] is not None else [],
            "readings": [reading], "tracks": []}
        nodes.append({"node_id": row["sample_id"], "clip_name": row["frame_id"],
            "evidence_kind": "bootstrap_photo", "video_id": photos["protocol"]["source"]["source_video_id"],
            "filename": Path(photos["protocol"]["source"]["path"]).name,
            "local_person_id": row["sample_id"], "raw_track_ids": [], "intervals": [],
            "source_video_id": photos["protocol"]["source"]["source_video_id"],
            "source_sha256": photos["protocol"]["source"]["sha256"], "timestamp_s": row["timestamp_s"],
            "sample_count": 1, "observation_count": 1,
            "hold_reasons": [] if np.count_nonzero(visibility[i]) >= 2 else ["insufficient_visible_parts"],
            "cover": {"crop_path": row["crop_path"], "archive_quality_score": row["quality_score"]},
            "exemplars": [], "jersey": jersey, "feature_indices": [i]})
    library = root / "library"
    library.mkdir(parents=True, exist_ok=True)
    if ((library / "library_manifest.json").is_file() or (seed / "registration.json").is_file()) and not (library / "players.sqlite3").is_file():
        raise FileNotFoundError("Player database missing; restore it to preserve stable IDs")
    with (library / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with PlayerRegistry(library / "players.sqlite3", match_id) as registry:
            from pipeline.identity.gallery_memory import MemoryPolicy
            from workflows.gallery_curation import curate_records
            policy = (MemoryPolicy(**json.loads(registry.meta("memory_policy"))) if registry.meta("memory_policy") else
                MemoryPolicy(strong_distance=min(.12, max_distance), supported_distance=max_distance,
                    novelty_distance=novelty_distance, min_margin=min_margin))
            records = [{"node": n, "features": [vectors[i].tolist() for i in n["feature_indices"]],
                "visibility": [visibility[i].tolist() for i in n["feature_indices"]]} for n in nodes]
            registry.ingest(nodes, vectors, visibility, [], contract,
                max_distance=max_distance, novelty_distance=novelty_distance, min_margin=min_margin,
                memory_policy=policy, curation=curate_records(root, records))
            signatures = {"bootstrap/" + name: [p.stat().st_size, p.stat().st_mtime_ns]
                for name in ("photos.json", "features.npz", "feature_contract.json", "numbers.json")
                for p in [seed / name]}
            for row in photos["samples"]:
                for key in ("crop_path", "torso_path"):
                    p = seed / row[key]
                    signatures["bootstrap/" + row[key]] = [p.stat().st_size, p.stat().st_mtime_ns]
            result = export_library(registry, root, library, signatures=signatures)
            atomic_json(seed / "registration.json", {"status": "completed_not_accuracy_evaluated",
                "photo_count": len(photos["frames"]), "crop_count": len(nodes),
                "match_id": match_id, "revision": result["review_revision"]})
            return result
