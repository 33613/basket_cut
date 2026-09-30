"""Tracking application workflow."""

from __future__ import annotations

from typing import Any

from analysis.visualization.render import RenderOptions, render_results
from pipeline.tracking.service import TrackingOptions, track_video


def run_tracking(
    options: TrackingOptions,
    *,
    create_visualization: bool = True,
) -> dict[str, Any]:
    result = track_video(options)
    if create_visualization:
        visualization = render_results(
            RenderOptions(
                input=options.input,
                tracks=options.output_dir / "tracks.jsonl",
                output=options.output_dir / "tracks_vis.mp4",
                max_frames=options.max_frames,
                overwrite=options.overwrite,
            )
        )
        result["visualization"] = visualization["output"]
    else:
        result["visualization"] = None
    return result
