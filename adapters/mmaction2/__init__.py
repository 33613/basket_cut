"""MMAction2-specific output adapters."""

from adapters.mmaction2.temporal_events import (
    TemporalAggregationOptions,
    aggregate_action_points,
)

__all__ = ["TemporalAggregationOptions", "aggregate_action_points"]
