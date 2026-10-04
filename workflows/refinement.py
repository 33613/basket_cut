"""Reprocess existing artifacts without model inference or baseline mutations."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from adapters.mmaction2.temporal_events import (
    TemporalAggregationOptions,
    aggregate_action_points,
)
from contracts.schema import read_jsonl, write_json, write_jsonl_line
from pipeline.identity.resolution import ResolutionOptions, resolve_identities
from pipeline.tracking.quality import QualityOptions, check_track_quality
from workflows.link_events import LinkEventsOptions, link_actions_to_people
from analysis.evaluation.track_review import review_index


@dataclass(frozen=True)
class RefinementOptions:
    input_dir: Path
    output_dir: Path
    review: Path | None = None
    max_distance: float | None = None
    min_observations: int = 3
    min_observed_seconds: float = 0.1
    suppress_duplicates: bool = False
    score_threshold: float = 0.2
    min_support: int = 1


def refine_existing_clip(options: RefinementOptions) -> dict:
    source = options.input_dir.resolve()
    output = options.output_dir.resolve()
    if (
        source == output
        or source.is_relative_to(output)
        or output.is_relative_to(source)
    ):
        raise ValueError(
            "Use an independent output directory; baseline must remain unchanged"
        )
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new, empty output directory for refinement")
    meta_path = source / "tracking/video_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    raw_identity = (
        source / "identity_raw"
        if (source / "identity_raw/identities.jsonl").is_file()
        else source / "identity"
    )
    # Validate mandatory inputs before creating any output.
    for name in (
        "tracking/tracks.jsonl",
        "action/actions.jsonl",
        "action/actions.summary.json",
    ):
        if not (source / name).is_file():
            raise FileNotFoundError(f"Missing required artifact: {name}")
    if not (raw_identity / "identities.jsonl").is_file():
        raise FileNotFoundError("Missing raw identity archive")
    if options.review:
        review = json.loads(options.review.read_text(encoding="utf-8"))
        if review.get("review_fingerprint"):
            current = review_index(source / "tracking/tracks.jsonl", meta_path)
            if current["fingerprint"] != review["review_fingerprint"]:
                raise ValueError("Merge review does not match the raw source tracks")
    quality = check_track_quality(
        QualityOptions(
            source / "tracking/tracks.jsonl",
            meta_path,
            output / "quality",
            min_observations=options.min_observations,
            min_observed_seconds=options.min_observed_seconds,
            suppress_duplicates=options.suppress_duplicates,
        )
    )
    tracking = output / "tracking"
    tracking.mkdir(parents=True, exist_ok=True)
    # Raw observations must survive refinement for independent review.
    shutil.copy2(source / "tracking/tracks.jsonl", tracking / "tracks.jsonl")
    shutil.copy2(meta_path, tracking / "video_meta.json")
    retained_summaries = [
        r
        for r in read_jsonl(output / "quality/quality_tracks.jsonl")
        if r["retained_observations"]
    ]
    write_json(
        tracking / "track_summary.json",
        {"track_count": len(retained_summaries), "tracks": retained_summaries},
    )
    resolution = resolve_identities(
        ResolutionOptions(
            output / "quality/tracks.jsonl",
            meta_path,
            raw_identity,
            output / "identity",
            review=options.review,
            quality_dir=output / "quality",
            max_distance=options.max_distance,
        )
    )
    observations = {
        (r["track_id"], r["frame_idx"]): r
        for r in read_jsonl(output / "quality/quality_observations.jsonl")
    }
    action = output / "action"
    action.mkdir(parents=True, exist_ok=True)
    total = kept = 0
    with (action / "actions.jsonl").open("w", encoding="utf-8") as handle:
        for record in read_jsonl(source / "action/actions.jsonl"):
            total += 1
            if record["video_id"] != meta["video_id"]:
                raise ValueError("Action video_id does not match this clip")
            key = (
                int(record["track_id"]),
                int(record.get("proposal_frame_idx", record["frame_idx"])),
            )
            decision = observations.get(key)
            if decision is None:
                raise ValueError(
                    f"Action references an unknown track observation: {key}"
                )
            if decision["kept"]:
                write_jsonl_line(handle, record)
                kept += 1
    action_summary = json.loads(
        (source / "action/actions.summary.json").read_text(encoding="utf-8")
    )
    action_summary.update(
        input_action_records=total,
        retained_action_records=kept,
        refinement="cached_predictions_quality_filter",
    )
    write_json(action / "actions.summary.json", action_summary)
    link = link_actions_to_people(
        LinkEventsOptions(
            action / "actions.jsonl",
            output / "identity/identity_map.jsonl",
            action / "actions_with_identity.jsonl",
        )
    )
    events = aggregate_action_points(
        TemporalAggregationOptions(
            action / "actions_with_identity.jsonl",
            action / "events.jsonl",
            video_meta=meta_path,
            score_threshold=options.score_threshold,
            min_support=options.min_support,
            allow_track_id_fallback=False,
        )
    )
    summary = {
        "video_id": meta["video_id"],
        "quality": quality,
        "resolution": resolution,
        "actions": {
            "input": total,
            "retained": kept,
            "linked": link["written_action_records"],
        },
        "events": events["event_count"],
        "models_rerun": False,
        "warning": "Cached action predictions were filtered and regrouped, not recomputed. Quality counts are not accuracy. Render a new video; do not copy the baseline overlay.",
    }
    write_json(output / "refinement_summary.json", summary)
    return summary
