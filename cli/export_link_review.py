"""Export a portable, read-only review ZIP without lossless windows/all crops."""

import argparse
import csv
import html
import io
import json
import zipfile
from pathlib import Path

from pipeline.media import sha256_file
from workflows import link_player as flow


def export_review(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output == root or root in output.parents:
        raise ValueError("Review ZIP must be outside the source run directory")
    if output.exists():
        raise FileExistsError(output)
    if flow.status(root)["status"] != "completed_not_accuracy_evaluated":
        raise ValueError("Finish the run before exporting its review bundle")
    protocol = flow.saved(root)
    files = {}
    sections = []
    checklist = []

    def include(path):
        path = Path(path).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"Missing/unsafe review asset: {path}")
        name = path.relative_to(root).as_posix()
        files[name] = path
        return html.escape(name, quote=True)

    if (root / "comparison.csv").exists():
        include(root / "comparison.csv")
    for event in protocol["events"]:
        directory = flow.folder(root, event)
        flow.require_stage(directory, "export")
        flow.read(directory / "link.json")
        record = flow.read(directory / "handoff/manifest.json")
        include(directory / "link.json")
        include(directory / "handoff/manifest.json")
        evidence = include(directory / "evidence.mp4")
        chosen = record["shooter_track_id"]
        trajectory = record.get("trajectory") or {}
        offset = event["window_start_frame"]
        fps = protocol["sources"][event["video_id"]]["fps"]
        start = (
            trajectory.get(
                "source_start_frame", offset + trajectory.get("window_start_frame", 0)
            )
            if trajectory
            else None
        )
        end = (
            trajectory.get(
                "source_end_frame_inclusive",
                offset + trajectory.get("window_end_frame_inclusive", 0),
            )
            if trajectory
            else None
        )
        interval = (
            f"原视频 F{start}–F{end}（含末帧），{start / fps:.3f}–{end / fps:.3f} 秒"
            if trajectory
            else "尚未确定目标区间"
        )
        media = f'<div><p>人物与球叠加视频</p><video controls preload="none" width="720" src="{evidence}"></video></div>'
        actor = directory / "handoff/trajectory.mp4"
        if actor.exists():
            path = include(actor)
            media += f'<div><p>目标人物局部轨迹</p><video controls preload="none" height="360" src="{path}"></video></div>'
        ball = directory / "ball/ball_vis.mp4"
        links = ""
        if ball.exists():
            links = f'<a href="{include(ball)}">单独查看球轨迹视频</a>'
        images = []
        for sample in record["samples"]:
            path = include(directory / "handoff" / sample["crop_path"])
            images.append(
                f'<figure><a href="{path}"><img height="160" src="{path}"></a>'
                f"<figcaption>F{sample['source_frame_idx']} / {sample['source_timestamp_s']:.3f}s</figcaption></figure>"
            )
        vid = html.escape(event["video_id"])
        tid = f"T{chosen}" if chosen is not None else "未选择"
        timing = (f'近筐搜索 {event["goal_interval_start_s"]:.2f}–{event["goal_interval_end_s"]:.2f}s（非精确进球时刻）'
                  if 'goal_interval_start_s' in event else f'进球 {event["goal_time_s"]}s')
        sections.append(
            f'<section id="video-{vid}"><h2>视频 {vid} · {timing} · {tid}</h2>'
            f"<p>{html.escape(record['attribution_status'])} · {interval}</p>"
            f'<div class="media">{media}</div><p>{links}</p><div class="images">{"".join(images)}</div></section>'
        )
        checklist.append(
            {
                "video_id": event["video_id"],
                "goal_time_s": event["goal_time_s"],
                "goal_time_semantics": event.get("goal_time_semantics", "external_timestamp"),
                "goal_interval_start_s": event.get("goal_interval_start_s", ""),
                "goal_interval_end_s": event.get("goal_interval_end_s", ""),
                "estimated_rim_time_s": record.get("estimated_rim_time_s", ""),
                "predicted_track_id": chosen,
                "source_start_frame": start,
                "source_end_frame_inclusive": end,
                "shooter_correct": "",
                "track_pure": "",
                "images_usable": "",
                "highlight_complete": "",
                "notes": "",
            }
        )
    page = (
        """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>出手者结果复核</title>
<style>body{font:16px sans-serif;margin:28px;color:#17343d}section{border-top:1px solid #ccd9dc;padding:20px 0}
.media,.images{display:flex;gap:20px;flex-wrap:wrap}figure{margin:8px}figcaption{font-size:13px}video{max-width:100%}</style>
<h1>进球出手者 · 人工复核</h1><p>先看出手者是否正确，再看目标区间内是否混人、图片是否清楚、剪辑是否覆盖动作。
叠加视频高亮的是局部轨迹 ID；同一 ID 可能换人，不能据此确认整段属于同一个人。</p>
<p>图片标注原视频帧号和秒数，点击可打开原尺寸 PNG。候选和几何分数均不等于身份确认。</p>
<p>请在 review_checklist.csv 填写复核结论；本包不修改服务器的人工复核状态。</p>"""
        + "".join(sections)
        + "</html>"
    )
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=list(checklist[0]))
    writer.writeheader()
    writer.writerows(checklist)
    readme = """这是轻量人工复核包，不是完整推理/身份交接数据。
解压后打开 review.html，先看人物与球视频，再看目标人物局部轨迹和原尺寸样本图。
填写 review_checklist.csv：出手者正确、局部轨迹纯净、图片可用、动作剪辑完整、备注。
视频 3 的 T6 曾有换人问题，需逐帧检查目标区间；视频 68 应检查进球附近 T13。
本包只包含所选样本图，未包含全部人物裁剪、无损 window.mkv、模型或完整缓存。
handoff/manifest.json 保留原始来源信息，部分来源文件仅存在于服务器完整结果中。
自动候选状态不表示已经人工确认，球员身份与跨片段匹配仍需后续处理。
"""
    prefix = root.name + "-review"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".partial.zip")
    try:
        with zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=3
        ) as archive:
            for name, path in sorted(files.items()):
                archive.write(path, f"{prefix}/{name}")
            for name, value in {
                "review.html": page,
                "review_checklist.csv": text.getvalue(),
                "README.txt": readme,
                "checksums.json": json.dumps(
                    {n: sha256_file(p) for n, p in files.items()}, indent=2
                ),
            }.items():
                archive.writestr(f"{prefix}/{name}", value)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{sha256_file(output)}  {output.name}\n"
    )
    return {
        "archive": str(output),
        "size_mib": output.stat().st_size / 1024**2,
        "events": len(checklist),
        "review_only": True,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output-dir", required=True, type=Path, help="Completed link_player run"
    )
    p.add_argument("--archive", type=Path)
    a = p.parse_args()
    archive = a.archive or a.output_dir.with_name(a.output_dir.name + "-review.zip")
    print(
        json.dumps(export_review(a.output_dir, archive), ensure_ascii=False, indent=2)
    )


if __name__ == "__main__":
    main()
