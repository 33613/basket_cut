"""Read-only visual audit before any change to initialization acceptance rules."""

from collections import Counter, defaultdict
import csv
import hashlib
import html
import json
from pathlib import Path
import shutil
import tarfile

from PIL import Image

from pipeline.media import atomic_json, sha256_file
from workflows.tracklet_bootstrap import folder, load_manifest


def raw_object(text):
    if not isinstance(text, str):
        return {}
    try:
        text = text.strip()
        if text.startswith("```json") and text.endswith("```"):
            text = text[7:-3].strip()
        value = json.loads(text)
        return value if isinstance(value, dict) else {}
    except (TypeError, ValueError):
        return {}


def rejection_stage(sample, readings, tracks):
    reading, track = readings[sample["sample_id"]], tracks[sample["track_key"]]
    if reading.get("usable") is not True:
        return "player_assessment_rejected"
    from pipeline.identity.player_evidence import strong_identity
    if track["clear_support"] < 2 and strong_identity(track) is None:
        return "insufficient_separated_clear_frames"
    if track["jersey"]["status"] == "conflict":
        return "track_number_conflict"
    return "accepted_for_kpr"


def write_csv(path, rows):
    fields = list(rows[0]) if rows else ["sample_id"]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)


def audit(run_dir, output_dir, *, context_samples=6):
    run, output = Path(run_dir).resolve(), Path(output_dir).resolve()
    if output.is_relative_to(run) or run.is_relative_to(output):
        raise ValueError("Use a new diagnostic directory outside the result tree")
    if type(context_samples) is not int or not 0 <= context_samples <= 12:
        raise ValueError("Choose 0–12 source context frames")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new empty diagnostic directory")
    archive = output.with_name(output.name + ".tar.gz")
    if archive.exists():
        raise FileExistsError("Diagnostic archive already exists")
    root = folder(run)
    manifest = load_manifest(run)
    assessment_path = root / "assessments.json"
    assessments = json.loads(assessment_path.read_text())
    if assessments.get("status") != "completed" or assessments["signature"][
        "manifest"
    ] != sha256_file(root / "manifest.json"):
        raise ValueError("Completed, matching Qwen assessments required")
    samples = manifest["samples"]
    readings, tracks = assessments["readings"], assessments["tracks"]
    if set(readings) != {s["sample_id"] for s in samples} or set(tracks) != {
        s["track_key"] for s in samples
    }:
        raise ValueError("Qwen evidence and initialization samples differ")
    raw = {sid: raw_object(r.get("raw_response", "")) for sid, r in readings.items()}

    def raw_counts(key):
        return dict(
            Counter(
                json.dumps(r.get(key, "<missing>"), ensure_ascii=False, sort_keys=True)
                for r in raw.values()
            )
        )

    stages = Counter(rejection_stage(s, readings, tracks) for s in samples)
    report = {
        "schema_version": 1,
        "accuracy_evaluated": False,
        "production_modified": False,
        "source": manifest["protocol"]["source"],
        "policy": manifest["protocol"]["policy"],
        "assessment_signature": assessments["signature"],
        "window_count": len(manifest["windows"]),
        "sample_count": len(samples),
        "tracklet_count": len(tracks),
        "usable_image_count": sum(r.get("usable") is True for r in readings.values()),
        "accepted_for_kpr": stages.get("accepted_for_kpr", 0),
        "rejection_stages": dict(stages),
        "rejection_reasons": dict(
            Counter(
                reason
                for r in readings.values()
                for reason in r.get("rejection_reasons", [])
            )
        ),
        "raw_fields": {
            key: raw_counts(key)
            for key in (
                "role",
                "single_target",
                "unoccluded",
                "clear",
                "jersey_color",
                "number",
                "readable",
            )
        },
        "clear_support_distribution": dict(
            Counter(t["clear_support"] for t in tracks.values())
        ),
        "jersey_status_counts": dict(
            Counter(t["jersey"]["status"] for t in tracks.values())
        ),
        "team_map": assessments["team_map"],
        "warning": "Raw VLM answers, heuristic quality and completion counts are not accuracy. No thresholds or model evidence are changed.",
    }
    output.mkdir(parents=True, exist_ok=True)
    media, evidence = output / "media", output / "evidence"
    media.mkdir()
    evidence.mkdir()
    for name in (
        "manifest.json",
        "assessments.json",
        "selection.json",
        "feature_contract.json",
        "registration.json",
    ):
        path = root / name
        if path.is_file():
            shutil.copyfile(path, evidence / name)
    rows, by_window = [], defaultdict(list)
    for sample in samples:
        sid = sample["sample_id"]
        reading = readings[sid]
        path = (root / sample["crop_path"]).resolve()
        if (
            not path.is_relative_to(root / "media")
            or sha256_file(path) != sample["crop_sha256"]
        ):
            raise ValueError("Native crop changed or escapes media root")
        name = hashlib.sha256(sid.encode()).hexdigest()[:24] + ".png"
        shutil.copyfile(path, media / name)
        with Image.open(path) as image:
            image.load()
            width, height = image.size
        row = {
            "sample_id": sid,
            "track_key": sample["track_key"],
            "window_id": sample["window_id"],
            "source_timestamp_s": sample["source_timestamp_s"],
            "frame_idx": sample["frame_idx"],
            "width": width,
            "height": height,
            "quality": sample["quality"],
            "sharpness": sample["sharpness"],
            "overlap_fraction": sample["overlap_fraction"],
            "stage": rejection_stage(sample, readings, tracks),
            "rejection_reasons": ";".join(reading.get("rejection_reasons", [])),
            "raw_clear": json.dumps(raw[sid].get("clear")),
            "raw_unoccluded": json.dumps(raw[sid].get("unoccluded")),
            "raw_color": json.dumps(raw[sid].get("jersey_color"), ensure_ascii=False),
            "raw_number": json.dumps(raw[sid].get("number"), ensure_ascii=False),
            "raw_readable": json.dumps(raw[sid].get("readable")),
            "raw_response": reading.get("raw_response", ""),
            "image": "media/" + name,
            "image_sha256": sample["crop_sha256"],
        }
        rows.append(row)
        by_window[sample["window_id"]].append((sample, row))
    write_csv(output / "samples.csv", rows)
    track_rows = [
        {
            "track_key": key,
            "clear_support": t["clear_support"],
            "usable_count": t["usable_count"],
            "team_id": t["team_id"],
            "jersey_status": t["jersey"]["status"],
            "jersey_number": t["jersey"]["number"],
            "candidate_numbers": json.dumps(t["jersey"].get("candidate_numbers", [])),
        }
        for key, t in sorted(tracks.items())
    ]
    write_csv(output / "tracks.csv", track_rows)

    # At most one highest-scoring crop per window gets a full, native frame.
    # Crops are copied verbatim; the highlighted box is an HTML overlay.
    contexts = {}
    candidates = [
        max(values, key=lambda item: item[0]["quality"])
        for values in by_window.values()
    ]
    candidates.sort(key=lambda item: (-item[0]["quality"], item[0]["sample_id"]))
    windows = {w["window_id"]: w for w in manifest["windows"]}
    if context_samples:
        import cv2

        for sample, row in candidates[:context_samples]:
            window = windows[sample["window_id"]]
            video = (root / window["path"]).resolve()
            if not video.is_relative_to(root / "windows"):
                raise ValueError("Window path escapes initialization root")
            capture = cv2.VideoCapture(str(video))
            try:
                capture.set(cv2.CAP_PROP_POS_FRAMES, sample["frame_idx"])
                ok, frame = capture.read()
                if not ok:
                    raise ValueError("Cannot decode context frame")
            finally:
                capture.release()
            name = (
                "context-"
                + hashlib.sha256(sample["sample_id"].encode()).hexdigest()[:24]
                + ".png"
            )
            if not cv2.imwrite(str(media / name), frame):
                raise OSError("Cannot save native context frame")
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = sample["bbox_xyxy"]
            contexts[sample["window_id"]] = {
                "image": "media/" + name,
                "sample_id": sample["sample_id"],
                "box_percent": [
                    100 * x1 / w,
                    100 * y1 / h,
                    100 * (x2 - x1) / w,
                    100 * (y2 - y1) / h,
                ],
            }
    report["contexts"] = contexts
    atomic_json(output / "report.json", report)
    escaped = lambda value: html.escape(str(value), quote=True)
    sections = []
    for wid, values in sorted(by_window.items()):
        context = contexts.get(wid)
        scene = ""
        if context:
            left, top, width, height = context["box_percent"]
            scene = (
                f"<p>场景帧对应 {escaped(context['sample_id'])}，红框在网页叠加，原图未画框。</p>"
                f'<div class="scene"><a href="{context["image"]}" target="_blank"><img src="{context["image"]}"></a>'
                f'<span style="left:{left}%;top:{top}%;width:{width}%;height:{height}%"></span></div>'
            )
        cards = []
        for sample, row in sorted(
            values, key=lambda value: (value[0]["raw_track_id"], -value[0]["quality"])
        ):
            cards.append(
                f'<article><a href="{row["image"]}" target="_blank"><img src="{row["image"]}" loading="lazy"></a>'
                f"<h3>{escaped(row['sample_id'])}</h3><p>{row['width']}×{row['height']} px · 原视频 {row['source_timestamp_s']:.2f}s</p>"
                f"<p>画质分 {row['quality']:.3f} · 锐度 {row['sharpness']:.1f} · 框重叠 {row['overlap_fraction']:.3f}</p>"
                f"<p>{escaped(row['stage'])}<br>{escaped(row['rejection_reasons'])}</p>"
                f"<details><summary>Qwen 原始回答</summary><pre>{escaped(row['raw_response'])}</pre></details></article>"
            )
        sections.append(
            f'<section><h2>{escaped(wid)}</h2>{scene}<div class="cards">{"".join(cards)}</div></section>'
        )
    page = (
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>初始化清晰度复核</title>'
        "<style>body{font:16px sans-serif;margin:24px;background:#f4f6f8}section,header{padding:20px;margin:16px 0;background:white}"
        ".cards{display:flex;flex-wrap:wrap;gap:16px}article{width:270px;padding:12px;background:#f4f6f8;overflow-wrap:anywhere}"
        "article img{max-width:100%;max-height:300px;width:auto;height:auto}h3{font-size:14px}pre{white-space:pre-wrap;overflow-wrap:anywhere}"
        ".scene{position:relative;width:fit-content;max-width:100%}.scene img{display:block;max-width:100%;max-height:540px}"
        ".scene span{position:absolute;box-sizing:border-box;border:2px solid red;pointer-events:none}</style>"
        "<header><h1>初始化清晰度复核</h1><p>原尺寸 PNG 点击可打开。展示尺寸只会缩小，不放大、不增强原图。</p>"
        "<p>画质启发分、Qwen 判断与号码可读性分别查看。尚无人工作为参考，不能据此计算准确率。</p>"
        f"<pre>{escaped(json.dumps({k: report[k] for k in ('sample_count', 'tracklet_count', 'usable_image_count', 'accepted_for_kpr', 'rejection_stages', 'rejection_reasons', 'raw_fields')}, ensure_ascii=False, indent=2))}</pre></header>"
        + "".join(sections)
        + "</html>"
    )
    (output / "index.html").write_text(page, encoding="utf-8")
    checksums = [
        (sha256_file(path), path.relative_to(output).as_posix())
        for path in sorted(output.rglob("*"))
        if path.is_file()
    ]
    (output / "checksums.sha256").write_text(
        "".join(f"{digest}  {name}\n" for digest, name in checksums)
    )
    with tarfile.open(archive, "w:gz") as bundle:
        bundle.add(output, arcname=output.name)
    return report, {
        "directory": str(output),
        "archive": str(archive),
        "archive_mib": round(archive.stat().st_size / 1024**2, 2),
    }
