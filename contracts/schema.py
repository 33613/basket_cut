"""Stable, framework-independent data contracts for the pipeline."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


def _finite_number(value: Any, field_name: str) -> float:
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        raise ValueError(f"{field_name} must be finite, got {value!r}")
    return number


def normalize_xyxy(
    bbox: Sequence[float],
    *,
    frame_width: int | None = None,
    frame_height: int | None = None,
) -> tuple[float, float, float, float]:
    """Validate an XYXY box and optionally clip it to the video frame."""
    if len(bbox) != 4:
        raise ValueError(f"bbox_xyxy must contain four values, got {bbox!r}")
    x1, y1, x2, y2 = (
        _finite_number(value, "bbox_xyxy") for value in bbox
    )
    if frame_width is not None:
        x1 = min(max(x1, 0.0), float(frame_width))
        x2 = min(max(x2, 0.0), float(frame_width))
    if frame_height is not None:
        y1 = min(max(y1, 0.0), float(frame_height))
        y2 = min(max(y2, 0.0), float(frame_height))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"bbox_xyxy has non-positive area: {(x1, y1, x2, y2)}")
    return x1, y1, x2, y2


@dataclass(frozen=True)
class VideoMeta:
    video_id: str
    input_path: str
    fps: float
    width: int
    height: int
    frame_count: int
    processed_frames: int
    tracker: str
    tracker_config: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrackRecord:
    """One observed person in one source-video frame."""

    video_id: str
    frame_idx: int
    timestamp_s: float
    track_id: int
    bbox_xyxy: tuple[float, float, float, float]
    det_score: float
    category_id: int
    source: str = "motip_sportsmot"
    association_score: float | None = None

    def __post_init__(self) -> None:
        if self.frame_idx < 0:
            raise ValueError("frame_idx must be non-negative")
        if self.timestamp_s < 0:
            raise ValueError("timestamp_s must be non-negative")
        if self.track_id < 0:
            raise ValueError("track_id must be non-negative")
        normalized = normalize_xyxy(self.bbox_xyxy)
        object.__setattr__(self, "bbox_xyxy", normalized)
        score = _finite_number(self.det_score, "det_score")
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"det_score must be in [0, 1], got {score}")
        object.__setattr__(self, "det_score", score)
        if self.association_score is not None:
            association_score = _finite_number(
                self.association_score, "association_score"
            )
            if not 0.0 <= association_score <= 1.0:
                raise ValueError(
                    "association_score must be in [0, 1], "
                    f"got {association_score}"
                )
            object.__setattr__(self, "association_score", association_score)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TrackRecord":
        data = dict(value)
        data["bbox_xyxy"] = tuple(data["bbox_xyxy"])
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["bbox_xyxy"] = list(self.bbox_xyxy)
        return data


@dataclass(frozen=True)
class EventRecord:
    """One product-facing temporal event.

    ``start`` is inclusive and ``end`` is exclusive, both measured in source
    video seconds.  The public contract intentionally keeps the five fields
    requested by the product (``id``, ``event``, ``start``, ``end`` and
    ``raw_score``); the remaining fields are traceability evidence used by the
    evaluator and the research dashboard.
    """

    video_id: str
    event_id: str
    identity_id: str
    event: str
    start: float
    end: float
    raw_score: float
    start_frame: int
    end_frame: int
    raw_track_ids: tuple[int, ...] = ()
    support_count: int = 0
    source: str = "mmaction2_temporal_aggregation"

    def __post_init__(self) -> None:
        if not self.video_id:
            raise ValueError("video_id must not be empty")
        if not self.event_id:
            raise ValueError("event_id must not be empty")
        if not self.identity_id:
            raise ValueError("id must not be empty")
        if not self.event:
            raise ValueError("event must not be empty")
        start = _finite_number(self.start, "start")
        end = _finite_number(self.end, "end")
        if start < 0 or end <= start:
            raise ValueError(
                f"Expected a positive half-open interval, got [{start}, {end})"
            )
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)
        score = _finite_number(self.raw_score, "raw_score")
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"raw_score must be in [0, 1], got {score}")
        object.__setattr__(self, "raw_score", score)
        if self.start_frame < 0 or self.end_frame < self.start_frame:
            raise ValueError(
                "Expected inclusive frame bounds with 0 <= start_frame <= "
                f"end_frame, got {self.start_frame}, {self.end_frame}"
            )
        if self.support_count < 0:
            raise ValueError("support_count must be non-negative")
        normalized_track_ids = tuple(
            sorted({int(value) for value in self.raw_track_ids})
        )
        if any(value < 0 for value in normalized_track_ids):
            raise ValueError("raw_track_ids must be non-negative")
        object.__setattr__(self, "raw_track_ids", normalized_track_ids)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EventRecord":
        data = dict(value)
        if "identity_id" not in data:
            identity_id = data.pop("id", None)
            if identity_id is None:
                identity_id = data.pop("person_id", None)
            data["identity_id"] = identity_id
        data["raw_track_ids"] = tuple(data.get("raw_track_ids", ()))
        allowed = {
            "video_id",
            "event_id",
            "identity_id",
            "event",
            "start",
            "end",
            "raw_score",
            "start_frame",
            "end_frame",
            "raw_track_ids",
            "support_count",
            "source",
        }
        return cls(**{key: item for key, item in data.items() if key in allowed})

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "event_id": self.event_id,
            "id": self.identity_id,
            "event": self.event,
            "start": self.start,
            "end": self.end,
            "raw_score": self.raw_score,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "raw_track_ids": list(self.raw_track_ids),
            "support_count": self.support_count,
            "source": self.source,
        }


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL at {path}:{line_number}: {exc}"
                ) from exc


def write_jsonl_line(handle: Any, value: Mapping[str, Any]) -> None:
    handle.write(json.dumps(dict(value), ensure_ascii=False, sort_keys=True))
    handle.write("\n")


def write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(dict(value), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def index_track_records(
    values: Iterable[Mapping[str, Any]],
) -> dict[int, list[TrackRecord]]:
    indexed: dict[int, list[TrackRecord]] = {}
    for value in values:
        record = TrackRecord.from_dict(value)
        indexed.setdefault(record.frame_idx, []).append(record)
    for records in indexed.values():
        records.sort(key=lambda item: item.track_id)
    return indexed
