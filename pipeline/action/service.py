"""Core MultiSports action-recognition service."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import mmcv
import mmengine
import numpy as np
import torch
from mmengine.runner import load_checkpoint
from mmengine.structures import InstanceData

from mmaction.registry import MODELS
from mmaction.structures import ActionDataSample

from pipeline.action.proposals import group_by_track, nearest_proposals
from pipeline.action.sampling import sample_frame_indices
from contracts.schema import TrackRecord, read_jsonl, write_json, write_jsonl_line


@dataclass(frozen=True)
class ActionOptions:
    input: Path
    tracks: Path
    config: Path
    checkpoint: Path
    label_map: Path
    output: Path
    device: str = "cuda:0"
    short_side: int = 256
    predict_stepsize: int = 8
    proposal_max_gap: int = 2
    min_det_score: float = 0.3
    action_threshold: float = 0.2
    top_k: int = 3
    label_prefix: str = "basketball_"
    overwrite: bool = False


def load_labels(path: Path, prefix: str) -> dict[int, str]:
    labels: dict[int, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                index_text, name = line.strip().split(": ", maxsplit=1)
            except ValueError as exc:
                raise ValueError(f"Invalid label map at {path}:{line_number}") from exc
            if not prefix or name.startswith(prefix):
                labels[int(index_text)] = name
    if not labels:
        raise ValueError(f"No labels with prefix {prefix!r} found in {path}")
    return labels


def read_video(path: Path, short_side: int) -> tuple[list[np.ndarray], dict[str, Any]]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open input video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    new_width, new_height = mmcv.rescale_size(
        (width, height), (short_side, float("inf"))
    )
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(mmcv.imresize(frame, (new_width, new_height)))
    finally:
        capture.release()
    return frames, {
        "fps": fps,
        "width": width,
        "height": height,
        "resized_width": new_width,
        "resized_height": new_height,
    }


def sampler_settings(config: mmengine.Config) -> tuple[int, int]:
    for transform in config.val_pipeline:
        transform_type = str(transform["type"])
        if transform_type.endswith("SampleAVAFrames"):
            return int(transform["clip_len"]), int(transform["frame_interval"])
    raise KeyError("The MMAction2 config has no SampleAVAFrames transform")


def recognize_actions(options: ActionOptions) -> dict[str, Any]:
    output_path = options.output.expanduser().resolve()
    if output_path.exists() and not options.overwrite:
        raise FileExistsError(
            f"{output_path} already exists; pass --overwrite to replace it"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = [TrackRecord.from_dict(value) for value in read_jsonl(options.tracks)]
    if not records:
        raise ValueError(f"No track observations found in {options.tracks}")
    video_ids = {record.video_id for record in records}
    if len(video_ids) != 1:
        raise ValueError("One action run must contain tracks from exactly one video")
    tracks = group_by_track(records)
    labels = load_labels(options.label_map, options.label_prefix)
    frames, video_meta = read_video(options.input, options.short_side)

    config = mmengine.Config.fromfile(str(options.config))
    clip_len, frame_interval = sampler_settings(config)
    window_size = clip_len * frame_interval
    if not frames:
        raise ValueError("Video contains no decoded frames")
    # Predict across the whole source clip. Boundary context repeats edge frames;
    # timestamps always refer to real frames, including clips shorter than a window.
    timestamps = np.arange(1, len(frames) + 1, options.predict_stepsize)

    try:
        config["model"]["test_cfg"]["rcnn"] = dict(action_thr=0)
    except KeyError:
        pass
    config.model.backbone.pretrained = None
    model = MODELS.build(config.model)
    load_checkpoint(model, str(options.checkpoint), map_location="cpu")
    model.to(options.device).eval()

    mean = np.array(config.model.data_preprocessor.mean)
    std = np.array(config.model.data_preprocessor.std)
    width_ratio = video_meta["resized_width"] / video_meta["width"]
    height_ratio = video_meta["resized_height"] / video_meta["height"]
    action_counts: Counter[str] = Counter()
    prediction_count = 0

    with output_path.open("w", encoding="utf-8") as output_file:
        progress = mmengine.ProgressBar(len(timestamps))
        for timestamp in timestamps:
            # MMAction's timestamp is one-based; our track frame indices are zero-based.
            center_frame_idx = int(timestamp) - 1
            proposals = nearest_proposals(
                tracks,
                center_frame_idx=center_frame_idx,
                max_frame_gap=options.proposal_max_gap,
                min_det_score=options.min_det_score,
            )
            if not proposals:
                progress.update()
                continue

            frame_indices, temporal_padding = sample_frame_indices(
                center_frame_idx, len(frames), clip_len, frame_interval)
            images = [frames[int(index)].astype(np.float32).copy() for index in frame_indices]
            for image in images:
                mmcv.imnormalize_(image, mean=mean, std=std, to_rgb=False)
            input_array = np.stack(images).transpose((3, 0, 1, 2))[np.newaxis]
            input_tensor = torch.from_numpy(input_array).to(options.device)

            boxes = []
            for proposal in proposals:
                x1, y1, x2, y2 = proposal.track.bbox_xyxy
                boxes.append(
                    [
                        x1 * width_ratio,
                        y1 * height_ratio,
                        x2 * width_ratio,
                        y2 * height_ratio,
                    ]
                )
            box_tensor = torch.tensor(
                boxes, dtype=torch.float32, device=options.device
            )
            data_sample = ActionDataSample()
            data_sample.proposals = InstanceData(bboxes=box_tensor)
            data_sample.set_metainfo(
                dict(
                    img_shape=(
                        video_meta["resized_height"],
                        video_meta["resized_width"],
                    )
                )
            )
            with torch.inference_mode():
                prediction = model(input_tensor, [data_sample], mode="predict")
            scores = prediction[0].pred_instances.scores.detach().float().cpu()

            for proposal_index, proposal in enumerate(proposals):
                ranked_actions = sorted(
                    (
                        {
                            "class_id": class_id,
                            "label": label,
                            "score": float(scores[proposal_index, class_id]),
                        }
                        for class_id, label in labels.items()
                    ),
                    key=lambda item: item["score"],
                    reverse=True,
                )
                candidates = ranked_actions[: options.top_k]
                selected = [
                    item
                    for item in ranked_actions
                    if item["score"] >= options.action_threshold
                ]
                for item in selected:
                    action_counts[item["label"]] += 1
                result = {
                    "video_id": proposal.track.video_id,
                    "frame_idx": center_frame_idx,
                    "timestamp_s": center_frame_idx / video_meta["fps"],
                    "temporal_padding": temporal_padding,
                    "track_id": proposal.track.track_id,
                    "bbox_xyxy": list(proposal.track.bbox_xyxy),
                    "det_score": proposal.track.det_score,
                    "proposal_frame_idx": proposal.track.frame_idx,
                    "proposal_frame_offset": proposal.frame_offset,
                    "action_candidates": candidates,
                    "selected_actions": selected,
                }
                write_jsonl_line(output_file, result)
                prediction_count += 1
            progress.update()

    summary = {
        "input": str(options.input.expanduser().resolve()),
        "tracks": str(options.tracks.expanduser().resolve()),
        "output": str(output_path),
        "video_id": next(iter(video_ids)),
        "frame_count": len(frames),
        "sample_count": len(timestamps),
        "person_prediction_count": prediction_count,
        "selected_action_counts": dict(action_counts.most_common()),
        "settings": {
            "clip_len": clip_len,
            "frame_interval": frame_interval,
            "predict_stepsize": options.predict_stepsize,
            "proposal_max_gap": options.proposal_max_gap,
            "min_det_score": options.min_det_score,
            "action_threshold": options.action_threshold,
            "top_k": options.top_k,
            "label_prefix": options.label_prefix,
        },
    }
    write_json(output_path.with_suffix(".summary.json"), summary)
    return summary
