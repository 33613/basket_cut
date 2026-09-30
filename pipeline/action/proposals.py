"""Convert sparse track observations into action-model proposals."""

from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from contracts.schema import TrackRecord


@dataclass(frozen=True)
class TrackProposal:
    track: TrackRecord
    center_frame_idx: int

    @property
    def frame_offset(self) -> int:
        return self.track.frame_idx - self.center_frame_idx


def group_by_track(records: Iterable[TrackRecord]) -> dict[int, list[TrackRecord]]:
    grouped: dict[int, list[TrackRecord]] = defaultdict(list)
    for record in records:
        grouped[record.track_id].append(record)
    for observations in grouped.values():
        observations.sort(key=lambda item: item.frame_idx)
    return dict(grouped)


def nearest_proposals(
    tracks: dict[int, list[TrackRecord]],
    *,
    center_frame_idx: int,
    max_frame_gap: int,
    min_det_score: float,
) -> list[TrackProposal]:
    """Return at most one nearby observation for every track.

    Exact center-frame observations win.  Otherwise the nearest observation is
    used within ``max_frame_gap``; ties prefer the earlier frame.
    """
    selected: list[TrackProposal] = []
    for track_id in sorted(tracks):
        observations = tracks[track_id]
        frame_indices = [item.frame_idx for item in observations]
        position = bisect_left(frame_indices, center_frame_idx)
        candidates = []
        if position < len(observations):
            candidates.append(observations[position])
        if position > 0:
            candidates.append(observations[position - 1])
        if not candidates:
            continue
        observation = min(
            candidates,
            key=lambda item: (abs(item.frame_idx - center_frame_idx), item.frame_idx),
        )
        if abs(observation.frame_idx - center_frame_idx) > max_frame_gap:
            continue
        if observation.det_score < min_det_score:
            continue
        selected.append(
            TrackProposal(track=observation, center_frame_idx=center_frame_idx)
        )
    return selected
