"""Render track and action JSONL files onto the original video."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2

from pipeline.common.schema import TrackRecord, read_jsonl
from pipeline.tracking.export_motip_tracks import color_for
from pipeline.visualization.timeline import action_text, nearest_action


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render modular tracking/action outputs without rerunning models"
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--tracks", required=True, type=Path)
    parser.add_argument("--actions", type=Path)
    parser.add_argument(
        "--identity-map",
        type=Path,
        help="Optional identity_map.jsonl produced by KPR archive mode",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-action-gap", type=int, default=4)
    parser.add_argument("--show-top-candidate", action="store_true")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser


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


def run(args: argparse.Namespace) -> None:
    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"{output_path} already exists; pass --overwrite to replace it"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tracks = load_tracks(args.tracks)
    actions = load_actions(args.actions)
    identities = load_identity_map(args.identity_map)

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
        while args.max_frames is None or frame_idx < args.max_frames:
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
                    args.max_action_gap,
                )
                predicted_action = action_text(
                    action, show_top_candidate=args.show_top_candidate
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
    print(f"Rendered {frame_idx} frames to {output_path}")


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
