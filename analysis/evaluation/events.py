"""Fixed-protocol, CPU-only product event evaluation, not official AP or ReID.

Class, temporal interval and actor localisation must pass together. Matching
maximises valid one-to-one pair count. Missing files remain execution errors;
existing empty prediction/GT files represent misses/negative videos.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import Counter, deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from contracts.schema import (
    EventRecord,
    TrackRecord,
    normalize_xyxy,
    read_jsonl,
    write_json,
    write_jsonl_line,
)

ACTOR_MODES = ("auto", "exact", "tube", "ignore")
PROFILES = ("generic", "multisports")
PROTOCOL_VERSION = "person_events_v2"
MULTISPORTS_IGNORED = ("basketball_save", "basketball_jump_ball")


@dataclass(frozen=True)
class EventEvaluationOptions:
    reference: Path
    prediction: Path
    output: Path
    errors_output: Path | None = None
    tracks: Path | None = None
    temporal_iou_threshold: float = 0.5
    actor_mode: str = "auto"
    actor_iou_threshold: float = 0.5
    min_raw_score: float = 0.0
    actor_min_coverage: float = 0.8
    profile: str = "generic"
    exclude_labels: tuple[str, ...] = ()
    reference_video_id: str | None = None
    prediction_video_id: str | None = None
    video_meta: Path | None = None


@dataclass(frozen=True)
class EvaluationEvent:
    record: EventRecord
    actor_tube: tuple[tuple[int, float, float, float, float], ...] = ()
    reference_video_meta: dict[str, Any] | None = None


def canonical_label(value: str) -> str:
    return "_".join(value.strip().lower().replace("_", " ").split())


def read_events(path: Path, *, reference: bool) -> list[EvaluationEvent]:
    events = []
    for value in read_jsonl(path):
        data = dict(value)
        if reference and "raw_score" not in data:
            data["raw_score"] = 1.0
        record = EventRecord.from_dict(data)
        rows, seen = [], set()
        for row in value.get("actor_tube", ()):
            if len(row) != 5 or int(row[0]) != row[0]:
                raise ValueError(f"Invalid actor tube row in {path}: {row!r}")
            frame = int(row[0])
            if frame in seen or not record.start_frame <= frame <= record.end_frame:
                raise ValueError(
                    f"Duplicate/out-of-range actor frame in {path}: {frame}"
                )
            seen.add(frame)
            rows.append((frame, *normalize_xyxy(row[1:])))
        events.append(
            EvaluationEvent(
                record, tuple(sorted(rows)), value.get("reference_video_meta")
            )
        )
    if reference and len({v.record.event_id for v in events}) != len(events):
        raise ValueError(f"Duplicate reference event_id in {path}")
    return events


def temporal_iou(left: EventRecord, right: EventRecord) -> float:
    intersection = max(0.0, min(left.end, right.end) - max(left.start, right.start))
    union = max(left.end, right.end) - min(left.start, right.start)
    return intersection / union if union > 0 else 0.0


def box_iou(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    intersection = max(0.0, min(left[2], right[2]) - max(left[0], right[0])) * max(
        0.0, min(left[3], right[3]) - max(left[1], right[1])
    )
    union = (
        (left[2] - left[0]) * (left[3] - left[1])
        + (right[2] - right[0]) * (right[3] - right[1])
        - intersection
    )
    return intersection / union if union > 0 else 0.0


def load_track_boxes(
    path: Path,
    *,
    expected_video_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict:
    boxes, video_ids = {}, set()
    for value in read_jsonl(path):
        record = TrackRecord.from_dict(value)
        video_ids.add(record.video_id)
        if metadata is not None:
            if record.frame_idx >= int(metadata["frame_count"]):
                raise ValueError("Track frame exceeds video_meta.frame_count")
            if not math.isclose(
                record.timestamp_s,
                record.frame_idx / float(metadata["fps"]),
                abs_tol=1e-4,
            ):
                raise ValueError("Track timestamp/frame differs from video_meta.fps")
        key = (record.frame_idx, record.track_id)
        if key in boxes and boxes[key] != record.bbox_xyxy:
            raise ValueError(f"Conflicting boxes for frame/track {key}")
        boxes[key] = record.bbox_xyxy
    if len(video_ids) > 1 or (
        video_ids and expected_video_id is not None and video_ids != {expected_video_id}
    ):
        raise ValueError("Track video_id must match the prediction video_id")
    return boxes


def tube_actor_evidence(
    prediction: EvaluationEvent,
    reference: EvaluationEvent,
    track_boxes: dict,
    frame_iou_threshold: float,
    minimum_coverage: float = 0.8,
) -> dict:
    lower = max(prediction.record.start_frame, reference.record.start_frame)
    upper = min(prediction.record.end_frame, reference.record.end_frame)
    rows = [row for row in reference.actor_tube if lower <= row[0] <= upper]
    best = (False, 0.0, 0.0, 0.0, None)
    # A single track must carry the evidence; no per-frame identity cherry-picking.
    # Repaired sequential tracklets need a separately defined future protocol.
    for track_id in prediction.record.raw_track_ids:
        overlaps, observed = [], 0
        for frame, *ref_box in rows:
            box = track_boxes.get((frame, track_id))
            observed += box is not None
            overlaps.append(box_iou(box, tuple(ref_box)) if box is not None else 0.0)
        if rows:
            coverage = sum(v >= frame_iou_threshold for v in overlaps) / len(rows)
            score = sum(overlaps) / len(rows)
            candidate = (
                coverage >= minimum_coverage and score >= frame_iou_threshold,
                coverage,
                score,
                observed / len(rows),
                track_id,
            )
            if candidate[:3] > best[:3] or best[4] is None:
                best = candidate
    return {
        "actor_score": best[2],
        "actor_coverage": best[1],
        "actor_observed_coverage": best[3],
        "actor_raw_track_id": best[4],
    }


def count_metrics(true_positive: int, predictions: int, references: int) -> dict:
    if not 0 <= true_positive <= min(predictions, references):
        raise ValueError("Invalid TP/prediction/reference counts")
    return {
        "true_positive": true_positive,
        "false_positive": predictions - true_positive,
        "false_negative": references - true_positive,
        "precision": true_positive / predictions if predictions else None,
        "recall": true_positive / references if references else None,
        "f1": 2 * true_positive / (predictions + references)
        if predictions + references
        else None,
    }


def match_events(
    predictions: list[EvaluationEvent],
    references: list[EvaluationEvent],
    *,
    temporal_threshold: float,
    actor_gate: Callable[[int, int], bool] | None = None,
) -> list:
    """Maximum-cardinality matching, deterministic IoU-first edge traversal.

    Count-based product F1, not official confidence-ranked AP. Iterative
    augmenting paths avoid recursion limits; total IoU is not optimised.
    """
    neighbours, overlaps = {}, {}
    for pi, pred in enumerate(predictions):
        edges = []
        for ri, ref in enumerate(references):
            if canonical_label(pred.record.event) != canonical_label(ref.record.event):
                continue
            iou = temporal_iou(pred.record, ref.record)
            if iou < temporal_threshold or (
                actor_gate is not None and not actor_gate(pi, ri)
            ):
                continue
            overlaps[pi, ri] = iou
            edges.append(ri)
        neighbours[pi] = sorted(edges, key=lambda ri: (-overlaps[pi, ri], ri))
    pred_to_ref, ref_to_pred = {}, {}
    for start in sorted(
        neighbours, key=lambda pi: (-predictions[pi].record.raw_score, pi)
    ):
        queue, seen_preds, parent_ref, endpoint = deque([start]), {start}, {}, None
        while queue and endpoint is None:
            pi = queue.popleft()
            for ri in neighbours[pi]:
                if ri in parent_ref:
                    continue
                parent_ref[ri] = pi
                if ri not in ref_to_pred:
                    endpoint = ri
                    break
                next_pred = ref_to_pred[ri]
                if next_pred not in seen_preds:
                    seen_preds.add(next_pred)
                    queue.append(next_pred)
        while endpoint is not None:
            pi = parent_ref[endpoint]
            previous = pred_to_ref.get(pi)
            pred_to_ref[pi], ref_to_pred[endpoint] = endpoint, pi
            endpoint = previous
    return [(pi, ri, overlaps[pi, ri]) for pi, ri in sorted(pred_to_ref.items())]


def _video_id(events: list[EvaluationEvent], description: str) -> str | None:
    ids = {event.record.video_id for event in events}
    if len(ids) > 1:
        raise ValueError(f"{description} must contain at most one video")
    return next(iter(ids), None)


def evaluate_events(options: EventEvaluationOptions) -> dict[str, Any]:
    for name, value in (
        ("temporal-iou-threshold", options.temporal_iou_threshold),
        ("actor-iou-threshold", options.actor_iou_threshold),
        ("actor-min-coverage", options.actor_min_coverage),
    ):
        if not 0 < value <= 1:
            raise ValueError(f"--{name} must be in (0, 1]")
    if not 0 <= options.min_raw_score <= 1:
        raise ValueError("--min-raw-score must be in [0, 1]")
    if options.actor_mode not in ACTOR_MODES or options.profile not in PROFILES:
        raise ValueError("Invalid actor mode or dataset profile")
    if bool(options.reference_video_id) != bool(options.prediction_video_id):
        raise ValueError(
            "Explicit video mapping requires BOTH --reference-video-id and --prediction-video-id"
        )
    reference_path, prediction_path = (
        options.reference.expanduser().resolve(),
        options.prediction.expanduser().resolve(),
    )
    output_path = options.output.expanduser().resolve()
    errors_path = (
        options.errors_output.expanduser().resolve()
        if options.errors_output
        else output_path.with_name("event_errors.jsonl")
    )
    protected = {reference_path, prediction_path}
    protected.update(
        p.expanduser().resolve()
        for p in (options.tracks, options.video_meta)
        if p is not None
    )
    if (
        output_path in protected
        or errors_path in protected
        or output_path == errors_path
    ):
        raise ValueError(
            "Evaluation outputs must not overwrite input files or each other"
        )
    all_references = read_events(reference_path, reference=True)
    all_predictions = read_events(prediction_path, reference=False)
    ref_id, pred_id = (
        _video_id(all_references, "Reference JSONL"),
        _video_id(all_predictions, "Prediction JSONL"),
    )
    if options.reference_video_id:
        if ref_id not in (None, options.reference_video_id) or pred_id not in (
            None,
            options.prediction_video_id,
        ):
            raise ValueError("Explicit video mapping does not match the event files")
        ref_id, pred_id = options.reference_video_id, options.prediction_video_id
    elif ref_id and pred_id and ref_id != pred_id:
        raise ValueError(
            "Different video IDs: verify the files and supply an explicit video mapping"
        )
    warnings, metadata = [], None
    if options.video_meta is not None:
        metadata = json.loads(
            options.video_meta.expanduser().read_text(encoding="utf-8")
        )
        fps = float(metadata["fps"])
        frame_count, processed = (
            int(metadata["frame_count"]),
            int(metadata["processed_frames"]),
        )
        if (
            not math.isfinite(fps)
            or fps <= 0
            or frame_count <= 0
            or processed != frame_count
        ):
            raise ValueError(
                "Full-video evaluation requires processed_frames == positive frame_count"
            )
        meta_id = str(metadata["video_id"])
        if pred_id is not None and pred_id != meta_id:
            raise ValueError("video_meta.video_id differs from prediction video_id")
        pred_id = meta_id
        if options.reference_video_id is None and ref_id and ref_id != pred_id:
            raise ValueError(
                "GT and metadata video IDs differ; an explicit mapping is required"
            )
        for event in all_references + all_predictions:
            record = event.record
            if record.end_frame >= frame_count:
                raise ValueError("Event frame bounds exceed video_meta.frame_count")
            if not (
                math.isclose(record.start, record.start_frame / fps, abs_tol=1e-4)
                and math.isclose(record.end, (record.end_frame + 1) / fps, abs_tol=1e-4)
            ):
                raise ValueError("Event times/frame bounds differ from video_meta.fps")
        for event in all_references:
            expected = event.reference_video_meta
            if expected is not None:
                for key in ("fps", "frame_count", "height", "width"):
                    if key not in metadata or not math.isclose(
                        float(expected[key]), float(metadata[key]), abs_tol=1e-4
                    ):
                        raise ValueError(
                            f"Video metadata differs from official GT {key}; do not resize/trim/re-time the source"
                        )
    else:
        warnings.append(
            "No video_meta: full processing range and FPS/frame alignment are unverified."
        )
    excluded = {canonical_label(v) for v in options.exclude_labels}
    if options.profile == "multisports":
        excluded.update(MULTISPORTS_IGNORED)
    references = [
        v for v in all_references if canonical_label(v.record.event) not in excluded
    ]
    predictions = [
        v
        for v in all_predictions
        if canonical_label(v.record.event) not in excluded
        and v.record.raw_score >= options.min_raw_score
    ]
    mode = options.actor_mode
    if mode == "auto":
        mode = "tube" if any(v.actor_tube for v in references) else "exact"
    if mode == "tube" and any(not v.actor_tube for v in references):
        raise ValueError("Every evaluated reference needs an actor_tube in tube mode")
    if mode == "exact":
        warnings.append(
            "Exact actor mode requires aligned GT identity labels, not arbitrary tracker/ReID numbers."
        )
    if mode == "ignore":
        warnings.append(
            "Actor correctness is not evaluated; headline is event_interval_f1."
        )
    track_boxes = {}
    if mode == "tube":
        if options.tracks is None:
            raise ValueError(
                "Tube mode requires --tracks (an existing empty file is allowed)"
            )
        track_boxes = load_track_boxes(
            options.tracks.expanduser().resolve(),
            expected_video_id=pred_id,
            metadata=metadata,
        )
    actor_evidence = {}

    def actor_gate(pi: int, ri: int) -> bool:
        if (pi, ri) not in actor_evidence:
            if mode == "tube":
                evidence = tube_actor_evidence(
                    predictions[pi],
                    references[ri],
                    track_boxes,
                    options.actor_iou_threshold,
                    options.actor_min_coverage,
                )
                passed = (
                    evidence["actor_score"] >= options.actor_iou_threshold
                    and evidence["actor_coverage"] >= options.actor_min_coverage
                )
            else:
                passed = (
                    mode == "ignore"
                    or predictions[pi].record.identity_id
                    == references[ri].record.identity_id
                )
                evidence = {
                    "actor_score": float(passed),
                    "actor_coverage": float(passed),
                }
            actor_evidence[pi, ri] = {**evidence, "actor_pass": passed}
        return actor_evidence[pi, ri]["actor_pass"]

    event_matches = match_events(
        predictions, references, temporal_threshold=options.temporal_iou_threshold
    )
    usable_matches = match_events(
        predictions,
        references,
        temporal_threshold=options.temporal_iou_threshold,
        actor_gate=actor_gate,
    )
    event_only = count_metrics(len(event_matches), len(predictions), len(references))
    end_to_end = count_metrics(len(usable_matches), len(predictions), len(references))
    usable_by_pred = {pi: (ri, iou) for pi, ri, iou in usable_matches}
    matched_refs = {ri for _, ri, _ in usable_matches}
    error_records = []
    for pi, pred in enumerate(predictions):
        base = {
            "prediction_event_id": pred.record.event_id,
            "prediction_id": pred.record.identity_id,
            "event": pred.record.event,
            "start": pred.record.start,
            "end": pred.record.end,
            "raw_score": pred.record.raw_score,
        }
        if pi in usable_by_pred:
            ri, overlap = usable_by_pred[pi]
            status = "usable"
        else:
            same = [
                ri
                for ri, ref in enumerate(references)
                if canonical_label(pred.record.event)
                == canonical_label(ref.record.event)
            ]
            temporal = [
                ri
                for ri in same
                if temporal_iou(pred.record, references[ri].record)
                >= options.temporal_iou_threshold
            ]
            valid = [ri for ri in temporal if actor_gate(pi, ri)]
            if valid:
                status, candidates = "duplicate", valid
            elif temporal:
                status, candidates = "wrong_actor", temporal
            else:
                wrong_class = [
                    ri
                    for ri, ref in enumerate(references)
                    if temporal_iou(pred.record, ref.record)
                    >= options.temporal_iou_threshold
                    and actor_gate(pi, ri)
                ]
                if wrong_class:
                    status, candidates = "wrong_class", wrong_class
                else:
                    nearby = [
                        ri
                        for ri in same
                        if temporal_iou(pred.record, references[ri].record) > 0
                        and actor_gate(pi, ri)
                    ]
                    status, candidates = (
                        ("wrong_interval", nearby) if nearby else ("false_positive", [])
                    )
            ri = max(
                candidates,
                key=lambda ri: temporal_iou(pred.record, references[ri].record),
                default=None,
            )
            overlap = (
                temporal_iou(pred.record, references[ri].record)
                if ri is not None
                else 0.0
            )
        details = {"status": status}
        if ri is not None:
            actor_gate(pi, ri)
            details.update(
                reference_event_id=references[ri].record.event_id,
                temporal_iou=overlap,
                **actor_evidence[pi, ri],
            )
        error_records.append({**base, **details})
    for ri, ref in enumerate(references):
        if ri not in matched_refs:
            error_records.append(
                {
                    "status": "missed_reference",
                    "reference_event_id": ref.record.event_id,
                    "reference_id": ref.record.identity_id,
                    "event": ref.record.event,
                    "start": ref.record.start,
                    "end": ref.record.end,
                }
            )
    per_class = {}
    for label in sorted(
        {canonical_label(v.record.event) for v in references + predictions}
    ):
        per_class[label] = count_metrics(
            sum(
                canonical_label(predictions[pi].record.event) == label
                for pi, _, _ in usable_matches
            ),
            sum(canonical_label(v.record.event) == label for v in predictions),
            sum(canonical_label(v.record.event) == label for v in references),
        )
    starts = [
        abs(predictions[pi].record.start - references[ri].record.start)
        for pi, ri, _ in usable_matches
    ]
    ends = [
        abs(predictions[pi].record.end - references[ri].record.end)
        for pi, ri, _ in usable_matches
    ]
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "headline": {
            "name": "event_interval_f1" if mode == "ignore" else "usable_event_f1",
            "value": end_to_end["f1"],
            "plain_language": f"{len(usable_matches)} valid events from {len(predictions)} outputs; {len(references)} GT events",
        },
        "end_to_end": end_to_end,
        "event_and_interval_only": event_only,
        "per_class": per_class,
        "diagnostics": {
            "error_counts": dict(Counter(v["status"] for v in error_records)),
            "prediction_count": len(predictions),
            "reference_count": len(references),
            "ignored_reference_count": len(all_references) - len(references),
            "ignored_prediction_count": sum(
                canonical_label(v.record.event) in excluded for v in all_predictions
            ),
            "score_filtered_prediction_count": sum(
                canonical_label(v.record.event) not in excluded
                and v.record.raw_score < options.min_raw_score
                for v in all_predictions
            ),
            "matched_boundary_errors_s": {
                "count": len(starts),
                "median_start": statistics.median(starts) if starts else None,
                "median_end": statistics.median(ends) if ends else None,
                "max_start": max(starts, default=None),
                "max_end": max(ends, default=None),
            },
        },
        "settings": {
            "profile": options.profile,
            "excluded_labels": sorted(excluded),
            "temporal_iou_threshold": options.temporal_iou_threshold,
            "actor_mode": mode,
            "actor_iou_threshold": options.actor_iou_threshold,
            "actor_min_coverage": options.actor_min_coverage,
            "actor_policy": "best_single_raw_track_in_temporal_overlap"
            if mode == "tube"
            else mode,
            "min_raw_score": options.min_raw_score,
            "matching": "maximum_cardinality_one_to_one",
        },
        "scope": {
            "actor_localisation_evaluated": mode == "tube",
            "actor_label_evaluated": mode == "exact",
            "reid_accuracy_evaluated": False,
            "is_official_multisports_metric": False,
            "full_video_verified": metadata is not None,
            "reference_video_properties_verified": metadata is not None
            and bool(all_references)
            and all(event.reference_video_meta is not None for event in all_references),
        },
        "inputs": {
            "reference": str(reference_path),
            "prediction": str(prediction_path),
            "tracks": str(options.tracks.expanduser().resolve())
            if options.tracks
            else None,
            "video_meta": str(options.video_meta.expanduser().resolve())
            if options.video_meta
            else None,
            "reference_video_id": ref_id,
            "prediction_video_id": pred_id,
            "explicit_video_mapping": bool(options.reference_video_id),
        },
        "warnings": warnings,
        "errors": str(errors_path),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    errors_path.parent.mkdir(parents=True, exist_ok=True)
    with errors_path.open("w", encoding="utf-8") as handle:
        for value in error_records:
            write_jsonl_line(handle, value)
    write_json(output_path, summary)
    return summary
