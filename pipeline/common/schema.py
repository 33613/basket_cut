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
