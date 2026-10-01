"""Product-oriented evaluation for temporal person events.

The headline is intentionally small and interpretable: a prediction is usable
only when its class, temporal interval and actor are all correct.  A second
event-only score ignores the actor so failures can be attributed quickly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from contracts.schema import (
    EventRecord,
    TrackRecord,
    read_jsonl,
    write_json,
    write_jsonl_line,
)


ACTOR_MODES = ("auto", "exact", "tube", "ignore")


@dataclass(frozen=True)
class EventEvaluationOptions:
    reference: Path
    prediction: Path
    output: Path
    errors_output: Path | None = None
    tracks: Path | None = None
    temporal_iou_threshold: float = 0.5
    actor_mode: str = "auto"
    actor_iou_threshold: float = 0.3
    min_raw_score: float = 0.0


@dataclass(frozen=True)
class EvaluationEvent:
    record: EventRecord
    actor_tube: tuple[tuple[int, float, float, float, float], ...] = ()


def canonical_label(value: str) -> str:
    return "_".join(value.strip().lower().replace("_", " ").split())


def read_events(path: Path, *, reference: bool) -> list[EvaluationEvent]:
    events = []
    for value in read_jsonl(path):
        data = dict(value)
        if reference and "raw_score" not in data:
            data["raw_score"] = 1.0
        record = EventRecord.from_dict(data)
        actor_tube = tuple(
            (
                int(row[0]),
                float(row[1]),
                float(row[2]),
                float(row[3]),
                float(row[4]),
            )
            for row in value.get("actor_tube", ())
        )
        events.append(EvaluationEvent(record=record, actor_tube=actor_tube))
    return events


def temporal_iou(left: EventRecord, right: EventRecord) -> float:
    intersection = max(0.0, min(left.end, right.end) - max(left.start, right.start))
    union = max(left.end, right.end) - min(left.start, right.start)
    return intersection / union if union > 0 else 0.0


def box_iou(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    left_x1, left_y1, left_x2, left_y2 = left
    right_x1, right_y1, right_x2, right_y2 = right
    width = max(0.0, min(left_x2, right_x2) - max(left_x1, right_x1))
    height = max(0.0, min(left_y2, right_y2) - max(left_y1, right_y1))
    intersection = width * height
    left_area = max(0.0, left_x2 - left_x1) * max(0.0, left_y2 - left_y1)
    right_area = max(0.0, right_x2 - right_x1) * max(0.0, right_y2 - right_y1)
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def load_track_boxes(
    path: Path,
) -> dict[tuple[int, int], tuple[float, float, float, float]]:
    boxes = {}
    for value in read_jsonl(path):
        record = TrackRecord.from_dict(value)
        boxes[(record.frame_idx, record.track_id)] = record.bbox_xyxy
    return boxes


def tube_actor_score(
    prediction: EvaluationEvent,
    reference: EvaluationEvent,
    track_boxes: dict[tuple[int, int], tuple[float, float, float, float]],
) -> tuple[float, float]:
    lower = max(prediction.record.start_frame, reference.record.start_frame)
    upper = min(prediction.record.end_frame, reference.record.end_frame)
    reference_boxes = {
        frame_idx: (x1, y1, x2, y2)
        for frame_idx, x1, y1, x2, y2 in reference.actor_tube
        if lower <= frame_idx <= upper
    }
    if not reference_boxes or not prediction.record.raw_track_ids:
        return 0.0, 0.0
    total_iou = 0.0
    observed_frames = 0
    for frame_idx, reference_box in reference_boxes.items():
        candidates = [
            track_boxes[(frame_idx, track_id)]
            for track_id in prediction.record.raw_track_ids
            if (frame_idx, track_id) in track_boxes
        ]
        if candidates:
            observed_frames += 1
            total_iou += max(box_iou(box, reference_box) for box in candidates)
    denominator = len(reference_boxes)
    return total_iou / denominator, observed_frames / denominator


def count_metrics(
    true_positive: int, predictions: int, references: int
) -> dict[str, Any]:
    false_positive = predictions - true_positive
    false_negative = references - true_positive
    precision = true_positive / predictions if predictions else 0.0
    recall = true_positive / references if references else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def greedy_match(
    predictions: list[EvaluationEvent],
    references: list[EvaluationEvent],
    *,
    temporal_threshold: float,
    actor_gate: Callable[[int, int], bool] | None = None,
) -> list[tuple[int, int, float]]:
    candidates = []
    for prediction_index, prediction in enumerate(predictions):
        for reference_index, reference in enumerate(references):
            if canonical_label(prediction.record.event) != canonical_label(
                reference.record.event
            ):
                continue
            overlap = temporal_iou(prediction.record, reference.record)
            if overlap < temporal_threshold:
                continue
            if actor_gate is not None and not actor_gate(
                prediction_index, reference_index
            ):
                continue
            candidates.append(
                (
                    -overlap,
                    -prediction.record.raw_score,
                    prediction_index,
                    reference_index,
                )
            )
    candidates.sort()
    used_predictions: set[int] = set()
    used_references: set[int] = set()
    matches = []
    for negative_overlap, _, prediction_index, reference_index in candidates:
        if prediction_index in used_predictions or reference_index in used_references:
            continue
        used_predictions.add(prediction_index)
        used_references.add(reference_index)
        matches.append((prediction_index, reference_index, -negative_overlap))
    return matches


def evaluate_events(options: EventEvaluationOptions) -> dict[str, Any]:
    if not 0.0 < options.temporal_iou_threshold <= 1.0:
        raise ValueError("--temporal-iou-threshold must be in (0, 1]")
    if not 0.0 <= options.actor_iou_threshold <= 1.0:
        raise ValueError("--actor-iou-threshold must be in [0, 1]")
    if not 0.0 <= options.min_raw_score <= 1.0:
        raise ValueError("--min-raw-score must be in [0, 1]")
    if options.actor_mode not in ACTOR_MODES:
        raise ValueError(f"--actor-mode must be one of {ACTOR_MODES}")

    reference_path = options.reference.expanduser().resolve()
    prediction_path = options.prediction.expanduser().resolve()
    output_path = options.output.expanduser().resolve()
    errors_path = (
        options.errors_output.expanduser().resolve()
        if options.errors_output
        else output_path.with_name("event_errors.jsonl")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    errors_path.parent.mkdir(parents=True, exist_ok=True)

    references = read_events(reference_path, reference=True)
    predictions = [
        event
        for event in read_events(prediction_path, reference=False)
        if event.record.raw_score >= options.min_raw_score
    ]
    if not references:
        raise ValueError(f"Reference event file is empty: {reference_path}")
    reference_video_ids = {event.record.video_id for event in references}
    prediction_video_ids = {event.record.video_id for event in predictions}
    if len(reference_video_ids) != 1:
        raise ValueError("Reference JSONL must contain exactly one video")
    if len(prediction_video_ids) > 1:
        raise ValueError("Prediction JSONL must contain at most one video")

    mode = options.actor_mode
    if mode == "auto":
        mode = "tube" if any(event.actor_tube for event in references) else "exact"

    track_boxes: dict[tuple[int, int], tuple[float, float, float, float]] = {}
    if mode == "tube":
        if options.tracks is None:
            raise ValueError(
                "MultiSports actor evaluation needs --tracks so predicted IDs "
                "can be compared with the reference action tube"
            )
        track_boxes = load_track_boxes(options.tracks.expanduser().resolve())
        if not track_boxes:
            raise ValueError(f"No track boxes found in {options.tracks}")

    actor_evidence: dict[tuple[int, int], dict[str, Any]] = {}

    def actor_gate(prediction_index: int, reference_index: int) -> bool:
        key = (prediction_index, reference_index)
        if key not in actor_evidence:
            prediction = predictions[prediction_index]
            reference = references[reference_index]
            if mode == "ignore":
                score, coverage, passed = 1.0, 1.0, True
            elif mode == "exact":
                passed = (
                    prediction.record.identity_id == reference.record.identity_id
                )
                score, coverage = (1.0 if passed else 0.0), 1.0
            else:
                score, coverage = tube_actor_score(
                    prediction, reference, track_boxes
                )
                passed = score >= options.actor_iou_threshold
            actor_evidence[key] = {
                "actor_score": score,
                "actor_coverage": coverage,
                "actor_pass": passed,
            }
        return bool(actor_evidence[key]["actor_pass"])

    event_only_matches = greedy_match(
        predictions,
        references,
        temporal_threshold=options.temporal_iou_threshold,
    )
    usable_matches = greedy_match(
        predictions,
        references,
        temporal_threshold=options.temporal_iou_threshold,
        actor_gate=actor_gate,
    )
    event_only = count_metrics(
        len(event_only_matches), len(predictions), len(references)
    )
    end_to_end = count_metrics(
        len(usable_matches), len(predictions), len(references)
    )

    usable_by_prediction = {
        prediction: (reference, overlap)
        for prediction, reference, overlap in usable_matches
    }
    event_only_by_prediction = {
        prediction: (reference, overlap)
        for prediction, reference, overlap in event_only_matches
    }
    usable_reference_indices = {reference for _, reference, _ in usable_matches}

    error_records = []
    for prediction_index, prediction in enumerate(predictions):
        base = {
            "prediction_event_id": prediction.record.event_id,
            "prediction_id": prediction.record.identity_id,
            "event": prediction.record.event,
            "start": prediction.record.start,
            "end": prediction.record.end,
            "raw_score": prediction.record.raw_score,
        }
        if prediction_index in usable_by_prediction:
            reference_index, overlap = usable_by_prediction[prediction_index]
            evidence = actor_evidence.get((prediction_index, reference_index), {})
            error_records.append(
                {
                    **base,
                    "status": "usable",
                    "reference_event_id": references[reference_index].record.event_id,
                    "temporal_iou": overlap,
                    **evidence,
                }
            )
        elif prediction_index in event_only_by_prediction:
            reference_index, overlap = event_only_by_prediction[prediction_index]
            actor_gate(prediction_index, reference_index)
            error_records.append(
                {
                    **base,
                    "status": "wrong_actor",
                    "reference_event_id": references[reference_index].record.event_id,
                    "temporal_iou": overlap,
                    **actor_evidence[(prediction_index, reference_index)],
                }
            )
        else:
            error_records.append({**base, "status": "false_positive"})
    for reference_index, reference in enumerate(references):
        if reference_index not in usable_reference_indices:
            error_records.append(
                {
                    "status": "missed_reference",
                    "reference_event_id": reference.record.event_id,
                    "reference_id": reference.record.identity_id,
                    "event": reference.record.event,
                    "start": reference.record.start,
                    "end": reference.record.end,
                }
            )
    with errors_path.open("w", encoding="utf-8") as handle:
        for value in error_records:
            write_jsonl_line(handle, value)

    summary = {
        "headline": {
            "name": "usable_event_f1",
            "value": end_to_end["f1"],
            "plain_language": (
                f"{end_to_end['true_positive']} usable events from "
                f"{len(predictions)} outputs; {len(references)} reference events"
            ),
        },
        "end_to_end": end_to_end,
        "event_and_interval_only": event_only,
        "diagnostics": {
            "actor_failures_after_event_match": max(
                0,
                event_only["true_positive"] - end_to_end["true_positive"],
            ),
            "prediction_count": len(predictions),
            "reference_count": len(references),
        },
        "settings": {
            "temporal_iou_threshold": options.temporal_iou_threshold,
            "actor_mode": mode,
            "actor_iou_threshold": options.actor_iou_threshold,
            "min_raw_score": options.min_raw_score,
        },
        "inputs": {
            "reference": str(reference_path),
            "prediction": str(prediction_path),
            "tracks": (
                str(options.tracks.expanduser().resolve())
                if options.tracks
                else None
            ),
            "reference_video_id": next(iter(reference_video_ids)),
            "prediction_video_id": (
                next(iter(prediction_video_ids)) if prediction_video_ids else None
            ),
        },
        "errors": str(errors_path),
    }
    write_json(output_path, summary)
    return summary
