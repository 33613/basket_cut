"""Inspect actual Qwen3 video tensors and compare description with boundary detection.

This diagnostic never edits the segmentation run or treats model answers as GT.
"""

import json
import math
from pathlib import Path
import re
import time

import numpy as np

from analysis.possession_states import state_prompt, parse_states, state_candidates
from pipeline.media import atomic_json, cut_video, sha256_file
from pipeline.segmentation.prompts import coarse_prompt
from pipeline.segmentation.service import SegmentOptions, parse_coarse
from pipeline.identity.qwen_backend import set_video_budget


DESCRIPTION_PROMPT = """Describe only what is visibly happening in this basketball video,
in chronological order. Identify teams by their visible jersey colors, without
assuming team names. Describe visible ball control, attack direction, free throws,
inbounds, rebounds and stoppages when identifiable. Explain which details are
unclear at this camera distance. Do not invent events or rely on the scoreboard.
Use approximate seconds relative to this local video, not the game clock.
Do not decide clip boundaries yet. Answer in concise Chinese prose.
"""


def array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().float().numpy()
    return np.asarray(value)


def reconstruct_video(pixels, grid, *, patch_size, temporal_patch_size, merge_size):
    """Invert the pinned Qwen3VLVideoProcessor patch layout, before normalization.

    See Transformers v4.57.1 video_processing_qwen3_vl.py::_preprocess.
    Inputs here must contain exactly one video, as in QwenBackend.read_video.
    """
    t, h, w = (int(v) for v in grid)
    p, tp, m = patch_size, temporal_patch_size, merge_size
    if min(t, h, w, p, tp, m) <= 0 or h % m or w % m:
        raise ValueError("Invalid Qwen3 video grid or patch settings")
    values = array(pixels)
    if values.shape != (t * h * w, 3 * tp * p * p):
        raise ValueError(f"Unexpected Qwen3 video tensor shape: {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("Video input contains nonfinite pixels")
    return values.reshape(t, h // m, w // m, m, m, 3, tp, p, p).transpose(
        0, 6, 5, 1, 3, 7, 2, 4, 8).reshape(t * tp, 3, h * p, w * p)


def video_rgb_frames(processor, inputs):
    """Recover full-resolution RGB frames after the processor's resize."""
    grids = array(inputs["video_grid_thw"]).astype(int).tolist()
    if len(grids) != 1:
        raise ValueError("Diagnostic requires exactly one video per model request")
    vp = processor.video_processor
    frames = reconstruct_video(inputs["pixel_values_videos"], grids[0],
        patch_size=vp.patch_size, temporal_patch_size=vp.temporal_patch_size, merge_size=vp.merge_size)
    values = frames.transpose(0, 2, 3, 1)
    if vp.do_normalize:
        values = values * np.asarray(vp.image_std) + np.asarray(vp.image_mean)
    if vp.do_rescale:
        values = values / vp.rescale_factor
    return np.clip(np.rint(values), 0, 255).astype(np.uint8)


def save_video_inputs(processor, inputs, output):
    """Save the actual processed frames sent to the model, not a second sampler."""
    from PIL import Image, ImageDraw

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    grids = array(inputs["video_grid_thw"]).astype(int).tolist()
    if len(grids) != 1:
        raise ValueError("Diagnostic requires exactly one video per model request")
    vp = processor.video_processor
    frames = reconstruct_video(inputs["pixel_values_videos"], grids[0],
        patch_size=vp.patch_size, temporal_patch_size=vp.temporal_patch_size, merge_size=vp.merge_size)
    summary = {"video_grid_thw": grids, "processed_frames": len(frames),
        "processed_height": frames.shape[2], "processed_width": frames.shape[3],
        "input_tokens": int(inputs["input_ids"].shape[-1]),
        "normalized_pixel_std": float(frames.std()),
        "adjacent_frame_mean_absolute_difference": float(np.abs(np.diff(frames, axis=0)).mean())}
    # Qwen3's processor carries timestamps in text between temporal patch groups.
    text = processor.batch_decode(inputs["input_ids"], skip_special_tokens=False,
                                  clean_up_tokenization_spaces=False)[0]
    text = re.sub(r"(?:<\|(?:image|video)_pad\|>)+", "[visual tokens omitted]", text)
    (output / "input-text.txt").write_text(text, encoding="utf-8")
    values = video_rgb_frames(processor, inputs)
    sheet = Image.new("RGB", (4 * 320, math.ceil(len(values) / 4) * 204), "white")
    draw = ImageDraw.Draw(sheet)
    for index, value in enumerate(values):
        picture = Image.fromarray(value)
        picture.thumbnail((320, 180))
        x, y = index % 4 * 320, index // 4 * 204
        sheet.paste(picture, (x, y))
        draw.text((x + 4, y + 182), f"sample {index:02d}", fill="black")
    sheet.save(output / "model-input.jpg", quality=90)
    atomic_json(output / "input-summary.json", summary)
    return summary


def diagnostic_plan(segmentation_dir, start_s, end_s):
    root = Path(segmentation_dir).resolve()
    protocol = json.loads((root / "protocol.json").read_text())
    plan = json.loads((root / "segmentation.json").read_text())
    source = protocol["source"]
    if not all(math.isfinite(v) for v in (start_s, end_s)) or not 0 <= start_s < end_s <= source["duration_s"]:
        raise ValueError("Diagnostic interval must lie within the source video")
    if plan["source"] != source:
        raise ValueError("Segmentation source does not match its protocol")
    windows = [{"window_id": f"coarse-{i:06d}", **row}
        for i, row in enumerate(plan["windows"])
        if row["end_s"] > start_s and row["start_s"] < end_s]
    if not windows:
        raise ValueError("No existing coarse windows overlap this interval")
    return protocol, windows


def diagnose(segmentation_dir, output, windows, protocol, backend, *, frames=None,
             ffmpeg="ffmpeg", ffprobe="ffprobe", capture_inputs=True,
             video_tokens=None, query_types=("description", "boundary")):
    root, output = Path(segmentation_dir).resolve(), Path(output).resolve()
    source = protocol["source"]
    if backend.provenance != protocol["model"]:
        raise ValueError("Diagnostic model differs from the original segmentation model")
    if output == root or output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError("Use a separate diagnostic output directory")
    if Path(source["path"]).resolve().is_relative_to(output):
        raise ValueError("Diagnostic output cannot contain the source video")
    if sha256_file(source["path"]) != source["sha256"]:
        raise ValueError("Original video changed after segmentation")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new, empty diagnostic output directory")
    options = SegmentOptions(**protocol["options"])
    options.validate()
    selected_frames = frames if frames is not None else options.coarse_frames
    if isinstance(selected_frames, bool) or not isinstance(selected_frames, int) or selected_frames % 2 or not 4 <= selected_frames <= 64:
        raise ValueError("Frame count must be an even integer from 4 to 64")
    if not query_types or len(set(query_types)) != len(query_types) or any(
            name not in {"description", "boundary", "state"} for name in query_types):
        raise ValueError("Use distinct query types selected from description, boundary and state")
    processor = getattr(backend, "processor", None)
    if video_tokens is not None and processor is None:
        raise ValueError("Changing the video budget requires a real processor")
    original_budget = set_video_budget(processor, video_tokens if video_tokens is not None else options.video_tokens) if processor else None
    effective_budget = dict(processor.video_processor.size) if processor else None
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"status": "running", "accuracy_evaluated": False, "source": source,
        "model": backend.provenance, "requested_frames": selected_frames,
        "same_frame_count_as_original": selected_frames == options.coarse_frames,
        "requested_video_tokens": video_tokens, "query_types": list(query_types),
        "original_video_budget": original_budget, "effective_video_budget": effective_budget,
        "state_probe_sha256": sha256_file(Path(__file__).with_name("possession_states.py")),
        "diagnostic_sha256": sha256_file(Path(__file__)), "results": []}
    result_path = output / "diagnostic.json"
    atomic_json(result_path, manifest)
    try:
        for window in windows:
            key = window["window_id"]
            folder = output / key
            folder.mkdir()
            clip = folder / "window.mp4"
            start, end = window["start_s"], window["end_s"]
            print(f"[diagnostic] {key} source {start:.2f}–{end:.2f}s; {selected_frames} frames", flush=True)
            cut_video(source["path"], clip, start, end, metadata=source,
                      ffmpeg=ffmpeg, ffprobe=ffprobe, preview=True)
            baseline_path = root / "responses" / (key + ".json")
            previous = json.loads(baseline_path.read_text()) if baseline_path.is_file() else {}
            # Reuse the exact saved original prompt when available, including context.
            prompt = previous.get("request", {}).get("prompt") or coarse_prompt(end - start, options.context)
            row = {**window, "previous_raw_response": previous.get("raw_response"), "queries": {}}
            questions = {"description": DESCRIPTION_PROMPT, "boundary": prompt,
                         "state": state_prompt(end - start, options.context)}
            for name in query_types:
                question = questions[name]
                request_dir = folder / name
                request_dir.mkdir()
                record = {"prompt": question, "num_frames": selected_frames, "max_new_tokens": 2048,
                          "video_budget": effective_budget}
                path = request_dir / "response.json"
                atomic_json(path, record)
                original_generate = backend._generate if capture_inputs else None
                if capture_inputs:
                    def generate(inputs, max_new_tokens=128):
                        record["video_input"] = save_video_inputs(backend.processor, inputs, request_dir)
                        atomic_json(path, record)
                        return original_generate(inputs, max_new_tokens=max_new_tokens)
                    backend._generate = generate
                started = time.monotonic()
                try:
                    record["raw_response"] = backend.read_video(clip, question,
                        num_frames=selected_frames, max_new_tokens=2048)
                except Exception as exc:
                    record["error"] = str(exc)
                    raise
                finally:
                    record["elapsed_s"] = round(time.monotonic() - started, 3)
                    atomic_json(path, record)
                    if capture_inputs:
                        backend._generate = original_generate
                if name in {"boundary", "state"}:
                    try:
                        if name == "boundary":
                            if options.coarse_method == "states":
                                from pipeline.segmentation.possession import parse_states as parse_production_states
                                observed = parse_production_states(record["raw_response"], end - start)
                                record["parsed_states"] = observed["states"]
                                record["parsed_boundaries"] = observed["candidates"]
                            else:
                                record["parsed_boundaries"] = parse_coarse(record["raw_response"], end - start)
                        else:
                            record["parsed_states"] = parse_states(record["raw_response"], end - start)
                            record["derived_candidates"] = [{**candidate,
                                "source_time_range_s": [start + t for t in candidate["time_range_s"]]}
                                for candidate in state_candidates(record["parsed_states"])]
                    except (ValueError, TypeError, KeyError) as exc:
                        record["parse_error"] = str(exc)
                    atomic_json(path, record)
                row["queries"][name] = record
                print(f"[{key}/{name}] {record['raw_response']}", flush=True)
                if "parse_error" in record:
                    print(f"[{key}/{name}] parse error: {record['parse_error']}", flush=True)
                if "derived_candidates" in record:
                    print("[state candidates] " + json.dumps(record["derived_candidates"], ensure_ascii=False), flush=True)
                if "video_input" in record:
                    print("[model input] " + json.dumps(record["video_input"]), flush=True)
            manifest["results"].append(row)
            atomic_json(result_path, manifest)
        manifest["status"] = "completed_not_accuracy_evaluated"
    except Exception as exc:
        manifest.update(status="failed", error=str(exc))
        raise
    finally:
        if processor:
            processor.video_processor.size = original_budget
        atomic_json(result_path, manifest)
    print(f"Diagnostic saved: {result_path}", flush=True)
    return manifest
