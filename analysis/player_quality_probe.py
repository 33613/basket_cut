"""Small, read-only native-crop check before full initialization assessment."""

from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import shutil

from PIL import Image

from pipeline.identity.player_evidence import assessment_provenance, parse_assessment
from pipeline.media import atomic_json, sha256_file
from workflows.tracklet_bootstrap import folder, load_manifest


def select_samples(manifest, per_window=2):
    if type(per_window) is not int or not 1 <= per_window <= 3:
        raise ValueError("Choose 1–3 probe images per window")
    chosen = []
    for window in manifest["windows"]:
        candidates = [
            s for s in manifest["samples"] if s["window_id"] == window["window_id"]
        ]
        candidates.sort(
            key=lambda s: (
                -(s["bbox_xyxy"][2] - s["bbox_xyxy"][0])
                * (s["bbox_xyxy"][3] - s["bbox_xyxy"][1]),
                -s["quality"],
                s["sample_id"],
            )
        )
        tracks = set()
        for row in candidates:
            if row["track_key"] not in tracks:
                chosen.append(row)
                tracks.add(row["track_key"])
            if len(tracks) >= per_window:
                break
    return chosen


def probe(run_dir, output_dir, *, backend, per_window=2):
    run, output = Path(run_dir).resolve(), Path(output_dir).resolve()
    if output.is_relative_to(run) or run.is_relative_to(output):
        raise ValueError("Use a separate diagnostic output directory")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new probe output directory")
    manifest = load_manifest(run)
    samples = select_samples(manifest, per_window)
    if not samples:
        raise ValueError(
            "No quality-selected track crops; inspect window selection first"
        )
    root = folder(run)
    output.mkdir(parents=True, exist_ok=True)
    (output / "media").mkdir()
    report = {
        "status": "running",
        "accuracy_evaluated": False,
        "production_modified": False,
        "sampling": "largest_native_crop_per_track_per_window; diagnostic selection, not population accuracy",
        "manifest_sha256": sha256_file(root / "manifest.json"),
        "provenance": assessment_provenance(backend.provenance),
        "sample_count": len(samples),
        "samples": [],
    }
    old_path = root / "assessments.json"
    previous = json.loads(old_path.read_text()) if old_path.exists() else {}
    if previous and previous["signature"]["manifest"] != report["manifest_sha256"]:
        raise ValueError("Old assessments refer to another tracking manifest")
    report["previous_provenance"] = previous.get("signature", {}).get("model")
    old_weights = (report["previous_provenance"] or {}).get("model_sha256")
    new_weights = report["provenance"].get("model_sha256")
    report["same_model_weights"] = (
        old_weights == new_weights if old_weights and new_weights else None
    )
    for sample in samples:
        source = (root / sample["crop_path"]).resolve()
        if (
            not source.is_relative_to(root / "media")
            or sha256_file(source) != sample["crop_sha256"]
        ):
            raise ValueError("Probe crop changed")
        name = hashlib.sha256(sample["sample_id"].encode()).hexdigest()[:24] + ".png"
        shutil.copyfile(source, output / "media" / name)
        with Image.open(source) as image:
            size = image.size
            raw = backend.assess_player(image.convert("RGB"))
        report["samples"].append(
            {
                "sample_id": sample["sample_id"],
                "window_id": sample["window_id"],
                "source_timestamp_s": sample["source_timestamp_s"],
                "image": "media/" + name,
                "native_size": list(size),
                "previous": previous.get("readings", {}).get(sample["sample_id"]),
                "reading": parse_assessment(raw),
                "raw_response": raw,
            }
        )
        atomic_json(output / "probe.json", report)
        print(
            f"[quality-probe] {len(report['samples'])}/{len(samples)} {sample['sample_id']}: {report['samples'][-1]['reading']['rejection_reasons']}",
            flush=True,
        )
    report.update(
        status="completed_not_accuracy_evaluated",
        usable_images=sum(r["reading"]["usable"] for r in report["samples"]),
        rejection_reasons=dict(
            Counter(
                reason
                for r in report["samples"]
                for reason in r["reading"]["rejection_reasons"]
            )
        ),
    )
    atomic_json(output / "probe.json", report)
    cards = []
    for row in report["samples"]:
        text = {
            "sample_id": row["sample_id"],
            "native_size": row["native_size"],
            "reading": row["reading"],
            "previous": row["previous"],
            "raw_response": row["raw_response"],
        }
        cards.append(
            f'<article><a href="{row["image"]}" target="_blank"><img src="{row["image"]}"></a><pre>{html.escape(json.dumps(text, ensure_ascii=False, indent=2))}</pre></article>'
        )
    (output / "index.html").write_text(
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>人物画质小样本测试</title><style>body{font:16px sans-serif;margin:24px}article{display:inline-block;vertical-align:top;width:340px;padding:12px;margin:8px;background:#f4f6f8}img{max-width:100%;max-height:300px;width:auto;height:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style><h1>原尺寸人物画质测试</h1><p>只读诊断，不登记球员。已挑选较大人物图，不能当作全视频准确率。</p>'
        + "".join(cards)
        + "</html>",
        encoding="utf-8",
    )
    return report
