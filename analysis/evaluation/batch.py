"""Manifest-based event evaluation. Never hide failed videos in an average."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from analysis.evaluation.events import (
    PROTOCOL_VERSION,
    EventEvaluationOptions,
    count_metrics,
    evaluate_events,
)
from contracts.schema import write_json


@dataclass(frozen=True)
class BatchEventEvaluationOptions:
    manifest: Path
    output_dir: Path
    temporal_iou_threshold: float = 0.5
    actor_mode: str = "tube"
    actor_iou_threshold: float = 0.5
    actor_min_coverage: float = 0.8
    min_raw_score: float = 0.0
    profile: str = "multisports"
    exclude_labels: tuple[str, ...] = ()


def _sum_metrics(results: list[dict], key: str) -> dict:
    tp = sum(v[key]["true_positive"] for v in results)
    fp = sum(v[key]["false_positive"] for v in results)
    fn = sum(v[key]["false_negative"] for v in results)
    return count_metrics(tp, tp + fp, tp + fn)


def evaluate_event_batch(options: BatchEventEvaluationOptions) -> dict[str, Any]:
    manifest_path = options.manifest.expanduser().resolve()
    output_dir = options.output_dir.expanduser().resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    videos = payload.get("videos") if isinstance(payload, dict) else None
    if not isinstance(videos, list) or not videos:
        raise ValueError(
            "Manifest needs a non-empty videos list of objects, not the download key list"
        )
    if any(not isinstance(v, dict) for v in videos):
        raise ValueError("Each manifest video must be an object containing paths")
    names = [str(v.get("name", "")) for v in videos]
    if any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("Manifest video names must be non-empty and unique")
    # Check ALL destinations against ALL inputs before writing any report.
    input_paths = {manifest_path}
    for item in videos:
        for key in ("reference", "prediction", "tracks", "video_meta"):
            if item.get(key):
                path = Path(item[key]).expanduser()
                input_paths.add((manifest_path.parent / path).resolve())
    destinations = {output_dir / "summary.json"}
    for index in range(len(videos)):
        destinations.update(
            {
                output_dir / f"{index:04d}" / "event_metrics.json",
                output_dir / f"{index:04d}" / "event_errors.jsonl",
            }
        )
    if input_paths & destinations:
        raise ValueError("Batch outputs would overwrite an input file")
    results, entries, error_counts = [], [], Counter()
    for index, item in enumerate(videos):

        def path(key: str, required: bool = False, item=item) -> Path | None:
            if not item.get(key):
                if required:
                    raise ValueError(f"Manifest entry needs {key}")
                return None
            return (manifest_path.parent / Path(item[key]).expanduser()).resolve()

        try:
            result = evaluate_events(
                EventEvaluationOptions(
                    reference=path("reference", True),
                    prediction=path("prediction", True),
                    tracks=path("tracks"),
                    video_meta=path("video_meta", True),
                    output=output_dir / f"{index:04d}" / "event_metrics.json",
                    temporal_iou_threshold=options.temporal_iou_threshold,
                    actor_mode=options.actor_mode,
                    actor_iou_threshold=options.actor_iou_threshold,
                    actor_min_coverage=options.actor_min_coverage,
                    min_raw_score=options.min_raw_score,
                    profile=options.profile,
                    exclude_labels=options.exclude_labels,
                    reference_video_id=item.get("reference_video_id"),
                    prediction_video_id=item.get("prediction_video_id"),
                )
            )
            if (
                options.profile == "multisports"
                and result["diagnostics"]["reference_count"] > 0
                and not result["scope"]["reference_video_properties_verified"]
            ):
                raise ValueError(
                    "MultiSports batch requires GT source video metadata; re-export the reference with cli.prepare_multisports_reference"
                )
            results.append(result)
            error_counts.update(result["diagnostics"]["error_counts"])
            entries.append(
                {
                    "name": names[index],
                    "status": "completed",
                    "headline": result["headline"],
                    "metrics": str(output_dir / f"{index:04d}" / "event_metrics.json"),
                }
            )
        except (ValueError, OSError, KeyError, TypeError) as exc:
            entries.append(
                {"name": names[index], "status": "failed", "error": str(exc)}
            )
    aggregate = _sum_metrics(results, "end_to_end")
    event_only = _sum_metrics(results, "event_and_interval_only")
    per_class = {}
    labels = sorted({label for result in results for label in result["per_class"]})
    for label in labels:
        values = [
            result["per_class"][label]
            for result in results
            if label in result["per_class"]
        ]
        tp = sum(v["true_positive"] for v in values)
        per_class[label] = count_metrics(
            tp,
            tp + sum(v["false_positive"] for v in values),
            tp + sum(v["false_negative"] for v in values),
        )
    failed = len(videos) - len(results)
    # auto must not silently mix tube and exact actor protocols in a batch.
    modes = {result["settings"]["actor_mode"] for result in results}
    incompatible = len(modes) > 1
    complete = failed == 0 and not incompatible
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "manifest": str(manifest_path),
        "provenance": payload.get("provenance", {}),
        "status": "complete" if complete else "incomplete",
        "planned_videos": len(videos),
        "completed_videos": len(results),
        "failed_videos": failed,
        "headline": {
            "name": "event_interval_f1" if modes == {"ignore"} else "usable_event_f1",
            "value": aggregate["f1"] if complete else None,
        },
        "end_to_end": aggregate if complete else None,
        "event_and_interval_only": event_only if complete else None,
        "completed_subset": {
            "end_to_end": aggregate,
            "event_and_interval_only": event_only,
            "per_class": per_class,
            "error_counts": dict(error_counts),
        },
        "settings": {
            "profile": options.profile,
            "actor_mode": options.actor_mode,
            "actor_iou_threshold": options.actor_iou_threshold,
            "actor_min_coverage": options.actor_min_coverage,
            "temporal_iou_threshold": options.temporal_iou_threshold,
            "min_raw_score": options.min_raw_score,
            "exclude_labels": list(options.exclude_labels),
        },
        "warnings": (
            ["Actor modes differ; choose one fixed batch protocol."]
            if incompatible
            else []
        )
        + (
            ["Failed videos are listed; partial results are NOT a final score."]
            if failed
            else []
        ),
        "videos": entries,
    }
    write_json(output_dir / "summary.json", summary)
    return summary
