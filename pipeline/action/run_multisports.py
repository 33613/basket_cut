"""Run MultiSports SlowFast using MOTIP tracks as person proposals."""

from __future__ import annotations

import argparse
import json
from collections import Counter
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
from pipeline.common.schema import TrackRecord, read_jsonl, write_json, write_jsonl_line


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply MultiSports SlowFast to MOTIP person tracks"
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--tracks", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--label-map", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--short-side", type=int, default=256)
    parser.add_argument("--predict-stepsize", type=int, default=8)
    parser.add_argument("--proposal-max-gap", type=int, default=2)
    parser.add_argument("--min-det-score", type=float, default=0.3)
    parser.add_argument("--action-threshold", type=float, default=0.2)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--label-prefix", default="basketball_")
    parser.add_argument("--overwrite", action="store_true")
    return parser


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


def run(args: argparse.Namespace) -> None:
    output_path = args.output.expanduser().resolve()
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"{output_path} already exists; pass --overwrite to replace it"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = [TrackRecord.from_dict(value) for value in read_jsonl(args.tracks)]
    if not records:
        raise ValueError(f"No track observations found in {args.tracks}")
    video_ids = {record.video_id for record in records}
    if len(video_ids) != 1:
        raise ValueError("One action run must contain tracks from exactly one video")
    tracks = group_by_track(records)
    labels = load_labels(args.label_map, args.label_prefix)
    frames, video_meta = read_video(args.input, args.short_side)

    config = mmengine.Config.fromfile(str(args.config))
    clip_len, frame_interval = sampler_settings(config)
    window_size = clip_len * frame_interval
    if len(frames) < window_size:
        raise ValueError(
            f"Video has {len(frames)} frames, but this model needs at least "
            f"{window_size} frames per clip"
        )
    timestamps = np.arange(
        window_size // 2,
        len(frames) + 1 - window_size // 2,
        args.predict_stepsize,
    )

    try:
        config["model"]["test_cfg"]["rcnn"] = dict(action_thr=0)
    except KeyError:
        pass
    config.model.backbone.pretrained = None
    model = MODELS.build(config.model)
    load_checkpoint(model, str(args.checkpoint), map_location="cpu")
    model.to(args.device).eval()

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
                max_frame_gap=args.proposal_max_gap,
                min_det_score=args.min_det_score,
            )
            if not proposals:
                progress.update()
                continue

            start_frame = int(timestamp) - (clip_len // 2 - 1) * frame_interval
            frame_indices = start_frame + np.arange(0, window_size, frame_interval) - 1
            images = [frames[int(index)].astype(np.float32).copy() for index in frame_indices]
            for image in images:
                mmcv.imnormalize_(image, mean=mean, std=std, to_rgb=False)
            input_array = np.stack(images).transpose((3, 0, 1, 2))[np.newaxis]
            input_tensor = torch.from_numpy(input_array).to(args.device)

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
            box_tensor = torch.tensor(boxes, dtype=torch.float32, device=args.device)
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
                candidates = ranked_actions[: args.top_k]
                selected = [
                    item
                    for item in ranked_actions
                    if item["score"] >= args.action_threshold
                ]
                for item in selected:
                    action_counts[item["label"]] += 1
                result = {
                    "video_id": proposal.track.video_id,
                    "frame_idx": center_frame_idx,
                    "timestamp_s": center_frame_idx / video_meta["fps"],
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
        "input": str(args.input.expanduser().resolve()),
        "tracks": str(args.tracks.expanduser().resolve()),
        "output": str(output_path),
        "video_id": next(iter(video_ids)),
        "frame_count": len(frames),
        "sample_count": len(timestamps),
        "person_prediction_count": prediction_count,
        "selected_action_counts": dict(action_counts.most_common()),
        "settings": {
            "clip_len": clip_len,
            "frame_interval": frame_interval,
            "predict_stepsize": args.predict_stepsize,
            "proposal_max_gap": args.proposal_max_gap,
            "min_det_score": args.min_det_score,
            "action_threshold": args.action_threshold,
            "top_k": args.top_k,
            "label_prefix": args.label_prefix,
        },
    }
    write_json(output_path.with_suffix(".summary.json"), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
