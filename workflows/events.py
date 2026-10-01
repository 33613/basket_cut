"""Application workflow for constructing temporal product events."""

from __future__ import annotations

from typing import Any

from adapters.mmaction2.temporal_events import (
    TemporalAggregationOptions,
    aggregate_action_points,
)


def build_temporal_events(options: TemporalAggregationOptions) -> dict[str, Any]:
    return aggregate_action_points(options)
