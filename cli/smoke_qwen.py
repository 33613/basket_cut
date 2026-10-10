"""Exercise quality sampling, local Qwen and evidence output without accuracy evaluation."""

import argparse
import json
from pathlib import Path

from contracts.execution import ExecutionSettings
from contracts.schema import TrackRecord, read_jsonl, write_json


def prepare_smoke_clip(output):
    """A tiny generated clip lets deployment run before real MOTIP tracks exist."""
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    video = output / "input.avi"
    tracks = output / "tracks.jsonl"
    mapping = output / "identity_map.jsonl"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (320, 384))
    if not writer.isOpened():
        writer.release()
        raise RuntimeError("Cannot create Qwen smoke video")
    # Embedded font works without system fonts, including Pillow 10.0.
    mask = Image.new("L", (32, 20))
    ImageDraw.Draw(mask).text((0, 0), "23", font=ImageFont.load_default(), fill=255)
    number = mask.crop(mask.getbbox()).resize((88, 80), Image.Resampling.LANCZOS)
    observations = []
    try:
        for frame_idx in range(16):
            offset = frame_idx // 5 * 6
            image = Image.new("RGB", (320, 384), (130, 130, 130))
            draw = ImageDraw.Draw(image)
            draw.ellipse((132 + offset, 12, 188 + offset, 76), fill=(200, 160, 130))
            draw.polygon([(100 + offset, 82), (60 + offset, 115),
                          (82 + offset, 157), (100 + offset, 142),
                          (100 + offset, 280), (220 + offset, 280),
                          (220 + offset, 142), (238 + offset, 157),
                          (260 + offset, 115), (220 + offset, 82)], fill=(30 + frame_idx * 3, 65, 150))
            image.paste((245, 245, 245), (116 + offset, 140), number)
            draw.rectangle((106 + offset, 282, 150 + offset, 374), fill=(40, 40, 50))
            draw.rectangle((170 + offset, 282, 214 + offset, 374), fill=(40, 40, 50))
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            # Include unusable imagery so the real sampler is exercised.
            if frame_idx < 5:
                frame[:] = 130
            writer.write(frame)
            observations.append(TrackRecord(
                video_id="qwen-smoke", frame_idx=frame_idx, timestamp_s=frame_idx / 10,
                track_id=0, bbox_xyxy=(60 + offset, 12, 260 + offset, 376),
                det_score=.9, category_id=0,
            ).to_dict())
    finally:
        writer.release()
    tracks.write_text("".join(json.dumps(r) + "\n" for r in observations), encoding="utf-8")
    mapping.write_text(json.dumps({"raw_track_id": 0, "person_id": "P0000"}) + "\n",
                       encoding="utf-8")
    return video, tracks, mapping


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=ExecutionSettings().qwen_model_dir)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new, empty smoke output directory")
    if not (args.model_dir / "config.json").is_file():
        raise FileNotFoundError("Local Qwen config.json is missing")
    output.mkdir(parents=True, exist_ok=True)

    import torch
    from pipeline.identity.jersey_qwen import recognize_jerseys_qwen
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen smoke test requires CUDA")
    video, tracks, mapping = prepare_smoke_clip(output)
    torch.cuda.reset_peak_memory_stats()
    print("开始质量抽样、模型文件校验和本地 Qwen 加载；文件校验需要读取全部权重。", flush=True)
    summary = recognize_jerseys_qwen(
        video, tracks, mapping, output / "identity", args.model_dir,
        max_samples=3, min_gap_s=.25,
    )
    torch.cuda.synchronize()
    evidence = list(read_jsonl(output / "identity/jersey_tracks.jsonl"))
    readings = [r for row in evidence for r in row["readings"]]
    result = {
        "status": "completed_not_accuracy_evaluated",
        "gpu": torch.cuda.get_device_name(0),
        "sample_count": summary["sample_count"],
        "model_invoked": summary["model_invoked"],
        "inference_count": summary["inference_count"],
        "cache_hit_count": summary["cache_hit_count"],
        "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
        "raw_responses": [r["raw_response"] for r in readings],
        "output_dir": str(output),
    }
    write_json(output / "qwen-smoke.json", result)
    if len(readings) < 2 or not summary["model_invoked"]:
        raise RuntimeError("Qwen smoke did not exercise multiple sampled frames")
    if any(set(r["rejection_reasons"]) & {"invalid_model_json", "invalid_number"} for r in readings):
        raise RuntimeError("Qwen response did not satisfy the number JSON contract; inspect qwen-smoke.json")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("Qwen 质量抽样 → GPU 推理 → 缓存与号码证据输出检查通过")


if __name__ == "__main__":
    main()
