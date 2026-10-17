"""Small, read-only native-crop check before full initialization assessment."""

from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import shutil

from PIL import Image

from pipeline.identity.player_evidence import assessment_provenance, parse_assessment
from pipeline.identity.qwen_backend import parse_reading
from pipeline.media import atomic_json, sha256_file
from workflows.tracklet_bootstrap import folder, load_manifest


def save_image_inputs(processor, inputs, output):
    """Invert Qwen patch packing to audit the actual resized, normalized input."""
    import numpy as np
    from analysis.segmentation_diagnostic import array, reconstruct_video

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    grids = array(inputs["image_grid_thw"]).astype(int).tolist()
    if len(grids) != 1 or grids[0][0] != 1:
        raise ValueError("Expected one static image in the Qwen observation request")
    ip = processor.image_processor
    frames = reconstruct_video(
        inputs["pixel_values"],
        grids[0],
        patch_size=ip.patch_size,
        temporal_patch_size=ip.temporal_patch_size,
        merge_size=ip.merge_size,
    )
    values = frames[0].transpose(1, 2, 0)
    if ip.do_normalize:
        values = values * np.asarray(ip.image_std) + np.asarray(ip.image_mean)
    if ip.do_rescale:
        values = values / ip.rescale_factor
    Image.fromarray(np.clip(np.rint(values), 0, 255).astype(np.uint8)).save(
        output / "processed.png"
    )
    summary = {
        "image_grid_thw": grids,
        "processed_width": int(values.shape[1]),
        "processed_height": int(values.shape[0]),
        "input_tokens": int(inputs["input_ids"].shape[-1]),
        "normalized_pixel_std": float(frames.std()),
        "processor_size": dict(ip.size),
    }
    atomic_json(output / "input-summary.json", summary)
    (output / "input-text.txt").write_text(
        processor.batch_decode(
            inputs["input_ids"],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )[0],
        encoding="utf-8",
    )
    return summary


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


def probe(
    run_dir,
    output_dir,
    *,
    backend,
    per_window=2,
    baseline=None,
    compare_number=False,
    capture_inputs=False,
):
    run, output = Path(run_dir).resolve(), Path(output_dir).resolve()
    if output.is_relative_to(run) or run.is_relative_to(output):
        raise ValueError("Use a separate diagnostic output directory")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new probe output directory")
    archive = output.with_name(output.name + ".tar.gz")
    if archive.exists():
        raise FileExistsError("Probe archive already exists")
    manifest = load_manifest(run)
    samples = select_samples(manifest, per_window)
    baseline_report = json.loads(Path(baseline).read_text()) if baseline else None
    if baseline_report is not None:
        if not isinstance(baseline_report, dict):
            raise ValueError("Baseline must be a JSON object")
        if baseline_report.get(
            "status"
        ) != "completed_not_accuracy_evaluated" or baseline_report.get(
            "manifest_sha256"
        ) != sha256_file(folder(run) / "manifest.json"):
            raise ValueError(
                "Baseline must be a completed probe of the same tracking manifest"
            )
        by_id = {s["sample_id"]: s for s in manifest["samples"]}
        ids = [s["sample_id"] for s in baseline_report["samples"]]
        if (
            not 1 <= len(ids) <= 18
            or len(ids) != len(set(ids))
            or not set(ids) <= by_id.keys()
        ):
            raise ValueError("Invalid baseline sample selection")
        samples = [by_id[sid] for sid in ids]
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
        "sampling": (
            "exact_baseline_images; diagnostic selection, not population accuracy"
            if baseline_report
            else "largest_native_crop_per_track_per_window; diagnostic selection, not population accuracy"
        ),
        "manifest_sha256": sha256_file(root / "manifest.json"),
        "provenance": assessment_provenance(backend.provenance),
        "sample_count": len(samples),
        "samples": [],
        "compare_number": compare_number,
        "capture_inputs": capture_inputs,
        "baseline_sha256": sha256_file(baseline) if baseline else None,
    }
    old_path = root / "assessments.json"
    previous = json.loads(old_path.read_text()) if old_path.exists() else {}
    if previous and previous["signature"]["manifest"] != report["manifest_sha256"]:
        raise ValueError("Old assessments refer to another tracking manifest")
    report["previous_provenance"] = (
        baseline_report["provenance"]
        if baseline_report
        else previous.get("signature", {}).get("model")
    )
    old_weights = (report["previous_provenance"] or {}).get("model_sha256")
    new_weights = report["provenance"].get("model_sha256")
    report["same_model_weights"] = (
        old_weights == new_weights if old_weights and new_weights else None
    )
    old_samples = (
        {r["sample_id"]: r for r in baseline_report["samples"]}
        if baseline_report
        else {}
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
            rgb = image.convert("RGB")
            input_dir = output / "inputs" / name.removesuffix(".png")
            raw = (
                backend.assess_player(rgb, diagnostic_dir=input_dir)
                if capture_inputs
                else backend.assess_player(rgb)
            )
            number_raw = backend.read(rgb) if compare_number else None
        reading = parse_assessment(raw, require_observations=True)
        independent = parse_reading(number_raw) if compare_number else None
        disagreement = (
            (independent["number"] != reading["number"]) if compare_number else None
        )
        report["samples"].append(
            {
                "sample_id": sample["sample_id"],
                "window_id": sample["window_id"],
                "source_timestamp_s": sample["source_timestamp_s"],
                "image": "media/" + name,
                "native_size": list(size),
                "previous": old_samples.get(sample["sample_id"], {}).get(
                    "reading", previous.get("readings", {}).get(sample["sample_id"])
                ),
                "previous_raw_response": old_samples.get(sample["sample_id"], {}).get(
                    "raw_response"
                ),
                "independent_number": independent,
                "independent_number_raw": number_raw,
                "number_disagreement": disagreement,
                "processed_image": str(
                    (input_dir / "processed.png").relative_to(output)
                )
                if capture_inputs
                else None,
                "reading": reading,
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
        identity_evidence_images=sum(
            r["reading"].get("evidence_eligible", False) for r in report["samples"]
        ),
        number_evidence_images=sum(
            r["reading"].get("number") is not None for r in report["samples"]
        ),
        number_disagreements=sum(
            r["number_disagreement"] is True for r in report["samples"]
        ),
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
            "previous_raw_response": row["previous_raw_response"],
            "independent_number": row["independent_number"],
            "independent_number_raw": row["independent_number_raw"],
            "number_disagreement": row["number_disagreement"],
        }
        processed = (
            f'<p>实际模型输入（处理器缩放后）</p><img src="{row["processed_image"]}">'
            if row["processed_image"]
            else ""
        )
        cards.append(
            f'<article><a href="{row["image"]}" target="_blank"><img src="{row["image"]}"></a>{processed}<pre>{html.escape(json.dumps(text, ensure_ascii=False, indent=2))}</pre></article>'
        )
    (output / "index.html").write_text(
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>人物画质小样本测试</title><style>body{font:16px sans-serif;margin:24px}article{display:inline-block;vertical-align:top;width:340px;padding:12px;margin:8px;background:#f4f6f8}img{max-width:100%;max-height:300px;width:auto;height:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style><h1>原尺寸人物画质测试</h1><p>只读诊断，不登记球员。已挑选较大人物图，不能当作全视频准确率。</p>'
        + "".join(cards)
        + "</html>",
        encoding="utf-8",
    )
    import tarfile

    archive = output.with_name(output.name + ".tar.gz")
    if archive.exists():
        raise FileExistsError("Probe archive already exists")
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(output, arcname=output.name)
    return report
