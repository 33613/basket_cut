"""Render track and action JSONL files onto the original video."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2

from analysis.visualization.timeline import action_text, color_for, nearest_action
from contracts.schema import TrackRecord, read_jsonl


@dataclass(frozen=True)
class RenderOptions:
    input: Path
    tracks: Path
    output: Path
    actions: Path | None = None
    identity_map: Path | None = None
    max_action_gap: int = 4
    show_top_candidate: bool = False
    max_frames: int | None = None
    overwrite: bool = False


def load_tracks(path: Path) -> dict[int, list[TrackRecord]]:
    indexed: dict[int, list[TrackRecord]] = defaultdict(list)
    for value in read_jsonl(path):
        record = TrackRecord.from_dict(value)
        indexed[record.frame_idx].append(record)
    for records in indexed.values():
        records.sort(key=lambda item: item.track_id)
    return dict(indexed)


def load_actions(path: Path | None) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    if path is None:
        return {}
    for value in read_jsonl(path):
        grouped[int(value["track_id"])].append(value)
    for values in grouped.values():
        values.sort(key=lambda item: int(item["frame_idx"]))
    return dict(grouped)


def load_identity_map(path: Path | None) -> dict[int, dict[str, Any]]:
    if path is None:
        return {}
    mappings: dict[int, dict[str, Any]] = {}
    for value in read_jsonl(path):
        track_id = int(value["raw_track_id"])
        if track_id in mappings:
            raise ValueError(f"Duplicate identity mapping for track {track_id}")
        mappings[track_id] = value
    return mappings


def render_results(options: RenderOptions) -> dict[str, Any]:
    input_path = options.input.expanduser().resolve()
    output_path = options.output.expanduser().resolve()
    if output_path.exists() and not options.overwrite:
        raise FileExistsError(
            f"{output_path} already exists; pass --overwrite to replace it"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tracks = load_tracks(options.tracks)
    actions = load_actions(options.actions)
    identities = load_identity_map(options.identity_map)

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open input video: {input_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open output video: {output_path}")

    frame_idx = 0
    try:
        while options.max_frames is None or frame_idx < options.max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            for track in tracks.get(frame_idx, []):
                x1, y1, x2, y2 = (int(value) for value in track.bbox_xyxy)
                color = color_for(track.track_id)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                identity = identities.get(track.track_id)
                if identity is None:
                    identity_text = f"T{track.track_id}"
                else:
                    person_id = str(identity["person_id"])
                    identity_label = identity.get("identity_label")
                    identity_text = (
                        f"{person_id} {identity_label} / T{track.track_id}"
                        if identity_label
                        else f"{person_id} / T{track.track_id}"
                    )
                labels = [f"{identity_text} det {track.det_score:.2f}"]
                action = nearest_action(
                    actions.get(track.track_id, []),
                    frame_idx,
                    options.max_action_gap,
                )
                predicted_action = action_text(
                    action, show_top_candidate=options.show_top_candidate
                )
                if predicted_action:
                    labels.append(predicted_action)
                for line_index, label in enumerate(labels):
                    y = max(16, y1 - 8 - 18 * (len(labels) - line_index - 1))
                    cv2.putText(
                        frame,
                        label,
                        (x1, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2,
                    )
            writer.write(frame)
            frame_idx += 1
    finally:
        capture.release()
        writer.release()
    return {"output": str(output_path), "rendered_frames": frame_idx}
