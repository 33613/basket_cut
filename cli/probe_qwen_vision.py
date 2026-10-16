"""Four inference calls to compare a real sampled frame/video and SDPA/eager.

This diagnoses visual understanding; it neither scores accuracy nor cuts video.
"""

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import time

from analysis.segmentation_diagnostic import array, save_video_inputs, video_rgb_frames
from contracts.execution import ExecutionSettings, normalize_omp_threads
from pipeline.media import atomic_json, sha256_file


PROBE_PROMPT = ("用中文简要描述画面中最主要的比赛状态、球员站位和裁判所在位置。"
                "只说能够看见的内容，无法确认就说明不确定。"
                "不要编写逐秒解说，也不要推断画面之外的比赛过程或投篮结果。")


def attention_state(model):
    return {name: getattr(config, "_attn_implementation", None) for name, config in (
        ("model", model.config), ("vision", model.config.vision_config),
        ("text", model.config.text_config))}


def compare(backend, image_inputs, video_inputs, output):
    """Reuse the exact prepared tensors within each modality across attention modes."""
    output = Path(output)
    before = attention_state(backend.model)
    if set(before.values()) != {"sdpa"}:
        raise ValueError(f"Expected the original all-SDPA configuration, got {before}")
    result = {"status": "running", "accuracy_evaluated": False, "prompt": PROBE_PROMPT,
              "max_new_tokens": 384, "original_attention": before, "queries": []}
    path = output / "attention-comparison.json"
    atomic_json(path, result)
    try:
        for mode in ("sdpa", "eager"):
            backend.model.set_attn_implementation(mode)
            actual = attention_state(backend.model)
            if set(actual.values()) != {mode}:
                raise RuntimeError(f"Attention implementation did not switch: {actual}")
            for modality, inputs in (("image", image_inputs), ("video", video_inputs)):
                row = {"attention": actual, "modality": modality}
                result["queries"].append(row)
                atomic_json(path, result)
                started = time.monotonic()
                try:
                    row["raw_response"] = backend._generate(inputs, max_new_tokens=384)
                except Exception as exc:
                    row["error"] = str(exc)
                    raise
                finally:
                    row["elapsed_s"] = round(time.monotonic() - started, 3)
                    atomic_json(path, result)
                print(f"[{mode}/{modality}] {row['raw_response']}", flush=True)
        result["status"] = "completed_not_accuracy_evaluated"
    except Exception as exc:
        result.update(status="failed", error=str(exc))
        raise
    finally:
        try:
            backend.model.set_attn_implementation("sdpa")
            result["restored_attention"] = attention_state(backend.model)
            if result["restored_attention"] != before:
                result.update(status="failed", restore_error="Original attention was not restored")
                raise RuntimeError(result["restore_error"])
        finally:
            atomic_json(path, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic-dir", type=Path, required=True)
    parser.add_argument("--window-id", default="coarse-000010")
    parser.add_argument("--frame-index", type=int, default=8,
                        help="Actual sampled frame index, not seconds (default 8 of 16)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--env-file", type=Path)
    args = parser.parse_args()
    if args.window_id != Path(args.window_id).name:
        parser.error("--window-id must be a directory name")
    original = args.diagnostic_dir.resolve()
    diagnostic = json.loads((original / "diagnostic.json").read_text())
    window = next((row for row in diagnostic["results"] if row["window_id"] == args.window_id), None)
    if window is None:
        parser.error("Requested window is not in the completed diagnostic")
    video = original / args.window_id / "window.mp4"
    if not video.is_file():
        parser.error(f"Retained local diagnostic video is missing: {video}")
    frames = diagnostic["requested_frames"]
    if not 0 <= args.frame_index < frames:
        parser.error("--frame-index must be within the original sampled frame count")
    output = args.output_dir.resolve()
    if output == original or output.is_relative_to(original) or original.is_relative_to(output):
        parser.error("Use a separate probe output directory")
    if output.exists() and any(output.iterdir()):
        parser.error("Use a new, empty probe output directory")
    if args.env_file:
        from cli.serve_web import load_environment
        load_environment(args.env_file)
    settings = ExecutionSettings()
    normalize_omp_threads()
    from pipeline.identity.qwen_backend import QwenBackend, ensure_gpu_idle
    ensure_gpu_idle()
    print("Loading local Qwen for four image/video and attention comparisons", flush=True)
    backend = QwenBackend(settings.qwen_model_dir)
    if backend.provenance != diagnostic["model"]:
        raise ValueError("Probe model differs from the audited model")
    output.mkdir(parents=True, exist_ok=True)
    messages = [{"role": "user", "content": [
        {"type": "video", "video": str(video)}, {"type": "text", "text": PROBE_PROMPT}]}]
    inputs = backend.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt", num_frames=frames, fps=None)
    summary = save_video_inputs(backend.processor, inputs, output / "video-input")
    if summary["processed_frames"] != frames:
        raise ValueError("Actual video frame count differs from the audited sampling")
    from PIL import Image
    image = Image.fromarray(video_rgb_frames(backend.processor, inputs)[args.frame_index])
    image.save(output / "sampled-frame.png")
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image}, {"type": "text", "text": PROBE_PROMPT}]}]
    text = backend.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # Preserve spatial resolution: do not enlarge the image separately from the video.
    image_inputs = backend.processor(text=[text], images=[image], do_resize=False,
        padding=True, return_tensors="pt")
    image_grid = array(image_inputs["image_grid_thw"]).astype(int).tolist()
    video_grid = summary["video_grid_thw"]
    if len(image_grid) != 1 or image_grid[0][1:] != video_grid[0][1:]:
        raise ValueError("Image/video spatial grids differ; cannot make a controlled comparison")
    packages = {name: version(name) for name in ("torch", "torchvision", "transformers", "safetensors")}
    atomic_json(output / "probe-protocol.json", {"source_diagnostic": str(original),
        "window_id": args.window_id, "source_range_s": [window["start_s"], window["end_s"]],
        "sampled_frame_index": args.frame_index, "window_sha256": sha256_file(video),
        "model": backend.provenance, "packages": packages, "video_input": summary,
        "image_grid_thw": image_grid, "prompt": PROBE_PROMPT})
    print("Input prepared: " + json.dumps(summary), flush=True)
    compare(backend, image_inputs, inputs, output)
    print(f"Probe saved: {output}", flush=True)


if __name__ == "__main__":
    main()
