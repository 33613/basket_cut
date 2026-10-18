"""CPU-only, bounded template evidence; immutable model artifacts are retained."""
from collections import defaultdict
import json
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

from contracts.schema import read_jsonl
from pipeline.identity.cross_clip import part_distances
from pipeline.identity.gallery_memory import select_templates
from pipeline.media import sha256_file
from workflows.library_inputs import safe_relative


def image_quality(path):
    with Image.open(path) as source:
        source.load()
        width, height = source.size
        gray = source.convert("L")
        gray.thumbnail((128, 192))
        a = np.asarray(gray, dtype=np.float32)
    sharpness = float(np.var(4*a[1:-1,1:-1]-a[:-2,1:-1]-a[2:,1:-1]-a[1:-1,:-2]-a[1:-1,2:])) if min(a.shape) > 2 else 0.
    exposure = float(np.mean((a > 5) & (a < 250)))
    quality = (.45*min(1., np.log1p(sharpness)/np.log1p(1000))
               + .35*min(1., np.sqrt(width*height/(96*192))) + .2*exposure)
    if width < 24 or height < 64 or sharpness < 5 or exposure < .1:
        quality = 0.
    return {"quality": float(quality), "width": width, "height": height,
            "sharpness": sharpness, "exposure": exposure, "image_sha256": sha256_file(path)}


def curate_records(root, records, *, per_observation=8):
    """Collect image samples when available; explicitly label prototype fallback.

    Existing clip databases keep their original fingerprints: selected sample
    features and image quality live in a separate derived evidence table.
    """
    root = Path(root).resolve()
    photos_path = root / "bootstrap/photos.json"
    photos = json.loads(photos_path.read_text()) if photos_path.is_file() else {}
    photo_rows = {s["sample_id"]: s for s in photos.get("samples", [])}
    evidence, clip_samples = {}, {}
    tracklet_banks = None
    for record in records:
        node = record["node"]
        key = node["node_id"]
        if node.get("evidence_kind") == "bootstrap_tracklet":
            if tracklet_banks is None:
                from workflows.tracklet_bootstrap import template_evidence
                tracklet_banks = template_evidence(root)
            if key not in tracklet_banks:raise ValueError("Missing clear tracklet templates")
            evidence[key] = tracklet_banks[key]
            continue
        if node.get("evidence_kind") == "bootstrap_photo":
            if key not in photo_rows:
                raise ValueError("Missing bootstrap photo evidence")
            sample = photo_rows[key]
            path = safe_relative(root / "bootstrap", sample["crop_path"])
            if sha256_file(path) != sample["crop_path_sha256"]:
                raise ValueError("Bootstrap crop changed during memory curation")
            evidence[key] = {"source_fingerprint": sample["crop_path_sha256"], "bbox_xyxy": sample["bbox_xyxy"], "templates": [{
                **image_quality(path), "key": key, "crop_path": sample["crop_path"],
                "media_kind": "bootstrap", "kind": "photo", "timestamp_s": sample["timestamp_s"],
                "source_frame": sample["frame_id"], "view": sample.get("view", "unknown"),
                "feature": record["features"][0], "visibility": record["visibility"][0]}]}
            continue
        name = node["clip_name"]
        if name not in clip_samples:
            raw = safe_relative(root, name) / "identity_raw"
            array_path, meta_path = raw / "kpr_samples.npz", raw / "kpr_samples.jsonl"
            grouped = defaultdict(list)
            source_hashes, source_stamps = {}, {}
            if array_path.is_file() and meta_path.is_file():
                for file in (array_path,meta_path):
                    relative = str(file.relative_to(root))
                    source_hashes[relative] = sha256_file(file)
                    source_stamps[relative] = [file.stat().st_size,file.stat().st_mtime_ns]
                rows = list(read_jsonl(meta_path))
                with np.load(array_path, allow_pickle=False) as data:
                    vectors, visible = data["embeddings"], data["visibility_scores"]
                    tids, frames = data["track_ids"], data["frame_indices"]
                if not (len(rows) == len(vectors) == len(visible) == len(tids) == len(frames)):
                    raise ValueError("KPR sample evidence length differs")
                part_distances(vectors[:1], visible[:1]) if len(vectors) else None
                for row, feature, vis, tid, frame in zip(rows, vectors, visible, tids, frames):
                    if int(row["track_id"]) != int(tid) or int(row["frame_idx"]) != int(frame):
                        raise ValueError("KPR sample metadata/feature ordering differs")
                    path = safe_relative(raw, row["crop_path"])
                    relative = str(path.relative_to(root))
                    source_hashes[relative] = sha256_file(path)
                    source_stamps[relative] = [path.stat().st_size,path.stat().st_mtime_ns]
                    grouped[int(tid)].append({**image_quality(path),
                        "key": row["sample_id"], "crop_path": row["crop_path"], "media_kind": "identity_raw",
                        "source_frame": (f"{node['source_video_id']}:{float(node['source_clip_start_s']) + float(row['timestamp_s']):.6f}"
                            if node.get("source_video_id") and "timestamp_s" in row else f"{name}:{int(frame)}"),
                        "observation_id": row.get("observation_id"), "view": row.get("view", "unknown"),
                        "kind": "image_sample", "frame_idx": int(frame), "feature": feature.tolist(), "visibility": vis.tolist()})
            clip_samples[name] = grouped, source_hashes, source_stamps
        grouped, source_hashes, source_stamps = clip_samples[name]
        bank = [item for tid in node["raw_track_ids"] for item in grouped.get(int(tid), [])]
        clear_assessment = (node.get("jersey") or {}).get("method") == "clear_player_multiframe_v1"
        if clear_assessment:
            approved={(int(t["raw_track_id"]),int(r["frame_idx"])): r
                for t in node["jersey"].get("tracks",[]) for r in t.get("readings",[]) if r.get("usable") is True}
            bank=[item for tid in node["raw_track_ids"] for item in grouped.get(int(tid),[])
                if (int(tid),int(item["frame_idx"])) in approved]
            for item in bank:
                reading = next((r for r in approved.values() if r.get("observation_id") == item.get("observation_id")), None) if item.get("observation_id") else None
                if item.get("observation_id") and reading is None:
                    raise ValueError("Qwen/KPR observation ID mismatch")
                if reading is not None:
                    if reading.get("crop_sha256") != item["image_sha256"]:
                        raise ValueError("Qwen/KPR observation image mismatch")
                    item["view"] = reading.get("view", "unknown")
        if bank:
            # Remove unusable images before diversity selection; no source file
            # is unlinked. A larger existing sample archive need not fill DB.
            bank = sorted(bank, key=lambda t: (-t["quality"], t["key"]))[:64]
            matrix = part_distances([t["feature"] for t in bank], [t["visibility"] for t in bank])
            selected = select_templates(dict(enumerate(bank)), matrix, per_observation)
            evidence[key] = {"templates": [bank[i] for i in selected], "source_stamps": source_stamps,
                "source_fingerprint": hashlib.sha256(json.dumps(source_hashes,sort_keys=True).encode()).hexdigest()}
        else:
            # Old/synthetic exports may only contain prototypes. Never pretend
            # these are actual multi-view image features.
            evidence[key] = ({"templates":[],"hold_reasons":["no_clear_assessed_kpr_images"]} if clear_assessment
                             else {"feature_kind": "track_prototype", "hold_reasons": []})
    return evidence
