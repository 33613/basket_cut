"""Read-only KPR distances on bounded initialization track pairs; no identity labels."""

from collections import Counter
import csv
import json
from pathlib import Path
import shutil
import tarfile

import numpy as np

from pipeline.identity.cross_clip import part_distances
from pipeline.media import atomic_json, sha256_file
from workflows.tracklet_bootstrap import folder, load_manifest


def select_pairs(manifest, report):
    by_id = {r["sample_id"]: r for r in manifest["samples"]}
    seeds = [by_id[r["sample_id"]] for r in report["samples"]]
    if not 1 <= len(seeds) <= 18 or len({s["sample_id"] for s in seeds}) != len(seeds):
        raise ValueError("Feature probe requires 1–18 distinct quality-probe samples")
    gap = manifest["protocol"]["policy"]["min_gap_s"]
    selected = {s["sample_id"]: s for s in seeds}
    for seed in seeds:
        peers = [
            r
            for r in manifest["samples"]
            if r["track_key"] == seed["track_key"]
            and abs(r["timestamp_s"] - seed["timestamp_s"]) >= gap
        ]
        if peers:
            peer = max(peers, key=lambda r: (r["quality"], r["sample_id"]))
            selected[peer["sample_id"]] = peer
    return list(selected.values())


def probe_features(run_dir, quality_probe, output_dir, backend, checkpoint):
    import cv2

    run, output = Path(run_dir).resolve(), Path(output_dir).resolve()
    if output.is_relative_to(run) or run.is_relative_to(output):
        raise ValueError("Use a separate diagnostic directory")
    archive = output.with_name(output.name + ".tar.gz")
    if archive.exists() or (output.exists() and any(output.iterdir())):
        raise FileExistsError("Use a new feature-probe output directory")
    manifest = load_manifest(run)
    report = json.loads(Path(quality_probe).read_text())
    if report.get("status") != "completed_not_accuracy_evaluated" or report.get(
        "manifest_sha256"
    ) != sha256_file(folder(run) / "manifest.json"):
        raise ValueError("Quality probe differs from completed tracking")
    samples = select_pairs(manifest, report)
    output.mkdir(parents=True, exist_ok=True)
    (output / "media").mkdir()
    images, records = [], []
    assessed = {r["sample_id"]: r["reading"] for r in report["samples"]}
    for i, row in enumerate(samples):
        source = (folder(run) / row["crop_path"]).resolve()
        if (
            not source.is_relative_to(folder(run) / "media")
            or sha256_file(source) != row["crop_sha256"]
        ):
            raise ValueError("Feature probe crop changed")
        image = cv2.imread(str(source))
        if image is None:
            raise ValueError("Cannot decode probe crop")
        images.append({"image": image})
        relative = f"media/{i:03d}.png"
        shutil.copyfile(source, output / relative)
        records.append(
            {
                **row,
                "probe_image": relative,
                "qwen_observation": assessed.get(row["sample_id"]),
            }
        )
    vectors, visible = backend.extract(images, batch_size=16)
    vectors, visible = np.asarray(vectors), np.asarray(visible)
    if vectors.ndim != 3 or len(vectors) != len(records):
        raise ValueError("Invalid KPR probe features")
    vectors = vectors / np.maximum(
        np.linalg.norm(vectors, axis=-1, keepdims=True), 1e-12
    )
    distances = part_distances(vectors, visible)
    rows = []
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            a, b = records[i], records[j]
            relation = (
                "same_track_candidate"
                if a["track_key"] == b["track_key"]
                else "different_track_same_window"
                if a["window_id"] == b["window_id"]
                else "cross_window_unknown"
            )
            value = float(distances[i, j])
            rows.append(
                {
                    "sample_a": a["sample_id"],
                    "sample_b": b["sample_id"],
                    "image_a": a["probe_image"],
                    "image_b": b["probe_image"],
                    "relation": relation,
                    "common_parts": int(
                        np.count_nonzero((visible[i] > 0) & (visible[j] > 0))
                    ),
                    "distance": value if np.isfinite(value) else None,
                    "comparable": bool(np.isfinite(value)),
                }
            )
    summary = {
        "status": "completed_not_accuracy_evaluated",
        "production_modified": False,
        "accuracy_evaluated": False,
        "sample_count": len(records),
        "pair_count": len(rows),
        "pair_relations": dict(Counter(r["relation"] for r in rows)),
        "manifest_sha256": sha256_file(folder(run) / "manifest.json"),
        "quality_probe_sha256": sha256_file(quality_probe),
        "checkpoint_sha256": sha256_file(checkpoint),
        "distance": "euclidean_sqrt_visibility_mean_div2",
        "min_common_parts": 2,
        "warning": "Track IDs and model jersey readings are not ground truth. No thresholds fitted or identities merged.",
        "samples": records,
        "pairs": rows,
    }
    np.savez_compressed(
        output / "features.npz", embeddings=vectors, visibility_scores=visible
    )
    atomic_json(output / "probe.json", summary)
    with (output / "pairs.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]) if rows else ["sample_a", "sample_b"]
        )
        writer.writeheader()
        writer.writerows(rows)
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(output, arcname=output.name)
    return {k: v for k, v in summary.items() if k not in {"samples", "pairs"}}
