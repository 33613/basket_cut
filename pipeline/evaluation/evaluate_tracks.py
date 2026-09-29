"""Evaluate a ``tracks.jsonl`` file against SHOT reference tracks.

Matching is performed independently in every frame with the Hungarian
algorithm over IoU distances.  ``motmetrics`` then measures both detection
errors and the temporal consistency of the matched identities.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from pipeline.common.schema import TrackRecord, read_jsonl, write_json


@dataclass(frozen=True)
class MotObservation:
    frame_idx: int
    track_id: int
    bbox_xywh: tuple[float, float, float, float]
    score: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate MOTIP JSONL tracks against SHOT reference tracks"
    )
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--prediction", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--events-output", type=Path)
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="Minimum box IoU for a frame-level match (default: 0.5)",
    )
    parser.add_argument(
        "--reference-frame-base",
        type=int,
        default=1,
        help="SHOT MOT files start at frame 1; predictions start at frame 0",
    )
    parser.add_argument("--min-pred-score", type=float, default=0.0)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--sequence-name")
    return parser


def read_reference(
    path: Path,
    *,
    frame_base: int,
    max_frames: int | None,
) -> list[MotObservation]:
    observations: list[MotObservation] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for line_number, row in enumerate(csv.reader(handle), start=1):
            if not row or row[0].strip().startswith("#"):
                continue
            if len(row) < 6:
                raise ValueError(
                    f"Invalid MOT row at {path}:{line_number}; expected >= 6 columns"
                )
            frame_idx = int(float(row[0])) - frame_base
            if frame_idx < 0:
                raise ValueError(
                    f"Negative normalized frame at {path}:{line_number}: {frame_idx}"
                )
            if max_frames is not None and frame_idx >= max_frames:
                continue
            confidence = float(row[6]) if len(row) > 6 else 1.0
            if confidence <= 0:
                continue
            x, y, width, height = (float(value) for value in row[2:6])
            if width <= 0 or height <= 0:
                continue
            observations.append(
                MotObservation(
                    frame_idx=frame_idx,
                    track_id=int(float(row[1])),
                    bbox_xywh=(x, y, width, height),
                    score=confidence,
                )
            )
    return observations


def read_predictions(
    path: Path,
    *,
    min_score: float,
    max_frames: int | None,
) -> list[MotObservation]:
    observations: list[MotObservation] = []
    for value in read_jsonl(path):
        record = TrackRecord.from_dict(value)
        if record.det_score < min_score:
            continue
        if max_frames is not None and record.frame_idx >= max_frames:
            continue
        x1, y1, x2, y2 = record.bbox_xyxy
        observations.append(
            MotObservation(
                frame_idx=record.frame_idx,
                track_id=record.track_id,
                bbox_xywh=(x1, y1, x2 - x1, y2 - y1),
                score=record.det_score,
            )
        )
    return observations


def group_by_frame(
    observations: Iterable[MotObservation],
) -> dict[int, list[MotObservation]]:
    grouped: dict[int, list[MotObservation]] = defaultdict(list)
    for observation in observations:
        grouped[observation.frame_idx].append(observation)
    for values in grouped.values():
        values.sort(key=lambda item: item.track_id)
    return dict(grouped)


def json_value(value: Any) -> Any:
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def iou_distance_matrix(
    reference_boxes: list[tuple[float, float, float, float]],
    prediction_boxes: list[tuple[float, float, float, float]],
    *,
    iou_threshold: float,
):
    """Return MOT distance ``1-IoU`` with invalid pairs set to NaN.

    This small implementation avoids ``motmetrics.distances.iou_matrix``,
    which uses a NumPy API removed in NumPy 2.  The accumulator and metric
    definitions still come from motmetrics.
    """
    import numpy as np

    reference = np.asarray(reference_boxes, dtype=float).reshape((-1, 4))
    prediction = np.asarray(prediction_boxes, dtype=float).reshape((-1, 4))
    if len(reference) == 0 or len(prediction) == 0:
        return np.empty((len(reference), len(prediction)), dtype=float)

    ref_x1 = reference[:, 0][:, None]
    ref_y1 = reference[:, 1][:, None]
    ref_x2 = ref_x1 + reference[:, 2][:, None]
    ref_y2 = ref_y1 + reference[:, 3][:, None]
    pred_x1 = prediction[:, 0][None, :]
    pred_y1 = prediction[:, 1][None, :]
    pred_x2 = pred_x1 + prediction[:, 2][None, :]
    pred_y2 = pred_y1 + prediction[:, 3][None, :]

    intersection_width = np.maximum(
        0.0, np.minimum(ref_x2, pred_x2) - np.maximum(ref_x1, pred_x1)
    )
    intersection_height = np.maximum(
        0.0, np.minimum(ref_y2, pred_y2) - np.maximum(ref_y1, pred_y1)
    )
    intersection = intersection_width * intersection_height
    reference_area = reference[:, 2][:, None] * reference[:, 3][:, None]
    prediction_area = prediction[:, 2][None, :] * prediction[:, 3][None, :]
    union = reference_area + prediction_area - intersection
    iou = np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union > 0,
    )
    distances = 1.0 - iou
    distances[iou < iou_threshold] = np.nan
    return distances


def run(args: argparse.Namespace) -> None:
    if not 0.0 < args.iou_threshold <= 1.0:
        raise ValueError("--iou-threshold must be in (0, 1]")
    if not 0.0 <= args.min_pred_score <= 1.0:
        raise ValueError("--min-pred-score must be in [0, 1]")

    try:
        import motmetrics as mm
    except ImportError as exc:
        raise RuntimeError(
            "motmetrics is required: pip install 'motmetrics>=1.4,<2'"
        ) from exc

    reference_path = args.reference.expanduser().resolve()
    prediction_path = args.prediction.expanduser().resolve()
    output_path = (
        args.output.expanduser().resolve()
        if args.output
        else prediction_path.with_name("tracking_metrics.json")
    )
    events_path = (
        args.events_output.expanduser().resolve()
        if args.events_output
        else output_path.with_name("tracking_events.csv")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    events_path.parent.mkdir(parents=True, exist_ok=True)

    reference = read_reference(
        reference_path,
        frame_base=args.reference_frame_base,
        max_frames=args.max_frames,
    )
    prediction = read_predictions(
        prediction_path,
        min_score=args.min_pred_score,
        max_frames=args.max_frames,
    )
    if not reference:
        raise ValueError(f"No valid reference observations found in {reference_path}")
    if not prediction:
        raise ValueError(f"No valid predictions found in {prediction_path}")

    reference_frames = group_by_frame(reference)
    prediction_frames = group_by_frame(prediction)
    frame_indices = sorted(set(reference_frames) | set(prediction_frames))
    accumulator = mm.MOTAccumulator(auto_id=True)
    for frame_idx in frame_indices:
        gt = reference_frames.get(frame_idx, [])
        pred = prediction_frames.get(frame_idx, [])
        distances = iou_distance_matrix(
            [item.bbox_xywh for item in gt],
            [item.bbox_xywh for item in pred],
            iou_threshold=args.iou_threshold,
        )
        accumulator.update(
            [item.track_id for item in gt],
            [item.track_id for item in pred],
            distances,
        )

    metrics_host = mm.metrics.create()
    metric_names = list(mm.metrics.motchallenge_metrics)
    for extra in ("num_frames", "num_objects", "num_predictions", "num_matches"):
        if extra in metrics_host.metrics and extra not in metric_names:
            metric_names.append(extra)
    sequence_name = args.sequence_name or prediction_path.parent.name
    summary = metrics_host.compute(
        accumulator,
        metrics=metric_names,
        name=sequence_name,
    )
    row = summary.loc[sequence_name]
    metrics = {name: json_value(row[name]) for name in summary.columns}
    motp_distance = metrics.get("motp")
    metrics["matched_mean_iou"] = (
        None if motp_distance is None else 1.0 - float(motp_distance)
    )

    result = {
        "sequence": sequence_name,
        "reference": {
            "path": str(reference_path),
            "frame_base": args.reference_frame_base,
            "observation_count": len(reference),
            "track_count": len({item.track_id for item in reference}),
        },
        "prediction": {
            "path": str(prediction_path),
            "observation_count": len(prediction),
            "track_count": len({item.track_id for item in prediction}),
            "min_score": args.min_pred_score,
        },
        "evaluation": {
            "frame_count": len(frame_indices),
            "iou_threshold": args.iou_threshold,
            "max_frames": args.max_frames,
        },
        "metrics": metrics,
        "metric_notes": {
            "idf1": "Identity F1; higher is better",
            "mota": "Detection errors plus ID switches; higher is better",
            "motp": "Mean matched distance (1-IoU); lower is better",
            "matched_mean_iou": "Mean IoU of matched boxes; higher is better",
            "num_switches": "Matched reference identities that changed predicted ID",
            "num_fragmentations": "Reference trajectories interrupted and later recovered",
        },
    }
    write_json(output_path, result)
    accumulator.events.reset_index().to_csv(events_path, index=False)

    rendered = mm.io.render_summary(
        summary,
        formatters=metrics_host.formatters,
        namemap=mm.io.motchallenge_metric_names,
    )
    print(rendered)
    print(
        json.dumps(
            {
                "metrics_json": str(output_path),
                "events_csv": str(events_path),
                "matched_mean_iou": metrics["matched_mean_iou"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
