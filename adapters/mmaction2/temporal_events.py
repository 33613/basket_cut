"""Convert sparse MMAction2 person predictions into temporal events.

This is deliberately an adapter, not part of the action baseline.  MMAction2's
MultiSports detector emits a score for each person at regularly sampled center
frames.  Product events need an interval, so this module joins positive samples
of the same actor and class, expands them to the sampling cells they represent,
and writes the stable :class:`contracts.schema.EventRecord` contract.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from contracts.schema import EventRecord, read_jsonl, write_json, write_jsonl_line


SCORE_REDUCERS = ("mean", "median", "max")


@dataclass(frozen=True)
class TemporalAggregationOptions:
    input: Path
    output: Path
    video_meta: Path | None = None
    fps: float | None = None
    cadence_frames: int | None = None
    score_threshold: float = 0.2
    max_missing_steps: int = 1
    min_support: int = 1
    min_duration_s: float = 0.0
    score_reducer: str = "mean"
    allow_track_id_fallback: bool = True
    overwrite: bool = False


@dataclass(frozen=True)
class ActionPoint:
    video_id: str
    identity_id: str
    event: str
    frame_idx: int
    timestamp_s: float
    score: float
    raw_track_id: int


def _identity_for(record: dict[str, Any], allow_fallback: bool) -> str:
    person_id = record.get("person_id")
    if person_id not in (None, ""):
        return str(person_id)
    if not allow_fallback:
        raise ValueError(
            "Action record has no person_id. Run cli.link_events first or pass "
            "--allow-track-id-fallback."
        )
    raw_track_id = int(record.get("raw_track_id", record["track_id"]))
    return f"T{raw_track_id:04d}"


def _ranked_actions(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Return one maximum score per label from selected and top-k evidence."""
    by_label: dict[str, dict[str, Any]] = {}
    for key in ("selected_actions", "action_candidates"):
        for value in record.get(key, ()) or ():
            label = str(value["label"])
            score = float(value["score"])
            previous = by_label.get(label)
            if previous is None or score > float(previous["score"]):
                by_label[label] = {"label": label, "score": score}
    return list(by_label.values())


def action_points(
    records: Iterable[dict[str, Any]],
    *,
    score_threshold: float,
    allow_track_id_fallback: bool,
) -> tuple[list[ActionPoint], int]:
    points: list[ActionPoint] = []
    input_records = 0
    for record in records:
        input_records += 1
        identity_id = _identity_for(record, allow_track_id_fallback)
        raw_track_id = int(record.get("raw_track_id", record["track_id"]))
        for action in _ranked_actions(record):
            score = float(action["score"])
            if score < score_threshold:
                continue
            points.append(
                ActionPoint(
                    video_id=str(record["video_id"]),
                    identity_id=identity_id,
                    event=str(action["label"]),
                    frame_idx=int(record["frame_idx"]),
                    timestamp_s=float(record["timestamp_s"]),
                    score=score,
                    raw_track_id=raw_track_id,
                )
            )
    return points, input_records


def infer_cadence(points: list[ActionPoint]) -> int:
    frames = sorted({point.frame_idx for point in points})
    differences = [
        right - left
        for left, right in zip(frames, frames[1:])
        if right > left
    ]
    if not differences:
        raise ValueError(
            "Cannot infer prediction cadence from one sampled frame; pass "
            "--cadence-frames explicitly."
        )
    # Positive predictions can skip one or more inference timestamps.  Their
    # median therefore over-estimates the real cadence; the greatest common
    # divisor recovers the sampling step from gaps such as 8, 16 and 24.
    cadence = differences[0]
    for difference in differences[1:]:
        cadence = math.gcd(cadence, difference)
    return max(1, cadence)


def resolve_cadence(
    options: TemporalAggregationOptions,
    input_path: Path,
    points: list[ActionPoint],
) -> tuple[int, str]:
    if options.cadence_frames is not None:
        return options.cadence_frames, "--cadence-frames"
    candidates = [
        input_path.with_suffix(".summary.json"),
        input_path.parent / "actions.summary.json",
    ]
    for path in dict.fromkeys(candidates):
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        value = (summary.get("settings") or {}).get("predict_stepsize")
        if value is not None and int(value) > 0:
            return int(value), str(path)
    return infer_cadence(points), "positive action point frame GCD"


def infer_fps(points: list[ActionPoint]) -> float:
    samples = sorted({(point.frame_idx, point.timestamp_s) for point in points})
    estimates = []
    for (left_frame, left_time), (right_frame, right_time) in zip(samples, samples[1:]):
        if right_frame > left_frame and right_time > left_time:
            estimates.append((right_frame - left_frame) / (right_time - left_time))
    if not estimates:
        raise ValueError("Cannot infer fps; pass --video-meta or --fps")
    return float(statistics.median(estimates))


def load_timing(
    options: TemporalAggregationOptions,
    points: list[ActionPoint],
) -> tuple[float, int | None, str]:
    if options.video_meta is not None:
        metadata_path = options.video_meta.expanduser().resolve()
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        fps = float(metadata["fps"])
        frame_count = int(
            metadata.get("processed_frames")
            or metadata.get("frame_count")
            or 0
        )
        return fps, frame_count or None, str(metadata_path)
    if options.fps is not None:
        return float(options.fps), None, "--fps"
    return infer_fps(points), None, "action timestamps"


def resolve_video_id(
    records: list[dict[str, Any]], options: TemporalAggregationOptions
) -> str:
    video_ids = {
        str(record["video_id"])
        for record in records
        if record.get("video_id") not in (None, "")
    }
    if len(video_ids) > 1:
        raise ValueError("One aggregation run must contain exactly one video_id")
    if video_ids:
        return next(iter(video_ids))
    if options.video_meta is not None:
        metadata_path = options.video_meta.expanduser().resolve()
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        video_id = metadata.get("video_id")
        if video_id not in (None, ""):
            return str(video_id)
    raise ValueError(
        "Cannot resolve video_id from an empty action file; pass --video-meta "
        "containing video_id."
    )


def reduce_score(values: list[float], reducer: str) -> float:
    if reducer == "mean":
        return float(statistics.fmean(values))
    if reducer == "median":
        return float(statistics.median(values))
    if reducer == "max":
        return float(max(values))
    raise ValueError(f"Unknown score reducer {reducer!r}; expected {SCORE_REDUCERS}")


def split_groups(
    points: list[ActionPoint], *, cadence_frames: int, max_missing_steps: int
) -> list[list[ActionPoint]]:
    if not points:
        return []
    maximum_gap = cadence_frames * (max_missing_steps + 1)
    groups = [[points[0]]]
    for point in points[1:]:
        if point.frame_idx - groups[-1][-1].frame_idx <= maximum_gap:
            groups[-1].append(point)
        else:
            groups.append([point])
    return groups


def aggregate_action_points(options: TemporalAggregationOptions) -> dict[str, Any]:
    if not 0.0 <= options.score_threshold <= 1.0:
        raise ValueError("--score-threshold must be in [0, 1]")
    if options.cadence_frames is not None and options.cadence_frames <= 0:
        raise ValueError("--cadence-frames must be positive")
    if options.fps is not None and options.fps <= 0:
        raise ValueError("--fps must be positive")
    if options.max_missing_steps < 0:
        raise ValueError("--max-missing-steps must be non-negative")
    if options.min_support <= 0:
        raise ValueError("--min-support must be positive")
    if options.min_duration_s < 0:
        raise ValueError("--min-duration-s must be non-negative")
    if options.score_reducer not in SCORE_REDUCERS:
        raise ValueError(
            f"--score-reducer must be one of {SCORE_REDUCERS}, "
            f"got {options.score_reducer!r}"
        )

    input_path = options.input.expanduser().resolve()
    output_path = options.output.expanduser().resolve()
    if output_path.exists() and not options.overwrite:
        raise FileExistsError(
            f"{output_path} already exists; pass --overwrite to replace it"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = list(read_jsonl(input_path))
    points, input_records = action_points(
        records,
        score_threshold=options.score_threshold,
        allow_track_id_fallback=options.allow_track_id_fallback,
    )
    video_id = resolve_video_id(records, options)
    cadence, cadence_source = resolve_cadence(options, input_path, points)
    fps, frame_count, timing_source = load_timing(options, points)
    if fps <= 0:
        raise ValueError(f"Resolved invalid fps: {fps}")

    # Multiple upstream lists may repeat one label at one sample.  Keep only
    # the strongest evidence before forming temporal runs.
    deduplicated: dict[tuple[str, str, int], ActionPoint] = {}
    for point in points:
        key = (point.identity_id, point.event, point.frame_idx)
        previous = deduplicated.get(key)
        if previous is None or point.score > previous.score:
            deduplicated[key] = point

    grouped: dict[tuple[str, str], list[ActionPoint]] = defaultdict(list)
    for point in deduplicated.values():
        grouped[(point.identity_id, point.event)].append(point)
    for values in grouped.values():
        values.sort(key=lambda item: item.frame_idx)

    left_width = cadence // 2
    right_width = cadence - left_width - 1
    candidates: list[dict[str, Any]] = []
    rejected_support = 0
    rejected_duration = 0
    for (identity_id, event), values in sorted(grouped.items()):
        for support in split_groups(
            values,
            cadence_frames=cadence,
            max_missing_steps=options.max_missing_steps,
        ):
            if len(support) < options.min_support:
                rejected_support += 1
                continue
            start_frame = max(0, support[0].frame_idx - left_width)
            end_frame = support[-1].frame_idx + right_width
            if frame_count is not None:
                end_frame = min(end_frame, frame_count - 1)
            start_s = start_frame / fps
            end_s = (end_frame + 1) / fps
            if end_s - start_s < options.min_duration_s:
                rejected_duration += 1
                continue
            candidates.append(
                {
                    "identity_id": identity_id,
                    "event": event,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "start": start_s,
                    "end": end_s,
                    "raw_score": reduce_score(
                        [point.score for point in support], options.score_reducer
                    ),
                    "raw_track_ids": tuple(
                        sorted({point.raw_track_id for point in support})
                    ),
                    "support_count": len(support),
                }
            )

    candidates.sort(
        key=lambda item: (
            item["start_frame"],
            item["end_frame"],
            item["identity_id"],
            item["event"],
        )
    )
    events = [
        EventRecord(
            video_id=video_id,
            event_id=f"{video_id}:E{index:05d}",
            identity_id=value["identity_id"],
            event=value["event"],
            start=value["start"],
            end=value["end"],
            raw_score=value["raw_score"],
            start_frame=value["start_frame"],
            end_frame=value["end_frame"],
            raw_track_ids=value["raw_track_ids"],
            support_count=value["support_count"],
        )
        for index, value in enumerate(candidates, start=1)
    ]
    with output_path.open("w", encoding="utf-8") as handle:
        for event in events:
            write_jsonl_line(handle, event.to_dict())

    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "video_id": video_id,
        "input_action_records": input_records,
        "positive_action_points": len(points),
        "deduplicated_action_points": len(deduplicated),
        "event_count": len(events),
        "rejected_for_support": rejected_support,
        "rejected_for_duration": rejected_duration,
        "timing": {
            "fps": fps,
            "fps_source": timing_source,
            "frame_count": frame_count,
            "cadence_frames": cadence,
            "cadence_source": cadence_source,
            "start_inclusive_end_exclusive": True,
        },
        "settings": {
            "score_threshold": options.score_threshold,
            "max_missing_steps": options.max_missing_steps,
            "min_support": options.min_support,
            "min_duration_s": options.min_duration_s,
            "score_reducer": options.score_reducer,
            "allow_track_id_fallback": options.allow_track_id_fallback,
        },
        "warning": (
            "raw_score is aggregated MMAction2 evidence, not a calibrated "
            "probability that the complete product event is correct."
        ),
    }
    write_json(output_path.with_suffix(".summary.json"), summary)
    return summary
