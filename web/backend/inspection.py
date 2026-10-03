"""Small, explicit diagnostic bundles; no model weights, tensors or videos."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Any

from web.backend.artifacts import collect_video_artifacts
from web.backend.imports import within_root

DIAGNOSTIC_FILES = (
    "tracking/video_meta.json",
    "tracking/track_summary.json",
    "tracking/tracks.jsonl",
    "identity/identity_archive_manifest.json",
    "identity/kpr_summary.json",
    "identity/identity_map.jsonl",
    "identity/identities.jsonl",
    "identity/kpr_samples.jsonl",
    "identity/kpr_track_sampling.jsonl",
    "identity/kpr_track_pairs.jsonl",
    "action/actions.jsonl",
    "action/actions_with_identity.jsonl",
    "action/events.jsonl",
    "action/actions.summary.json",
    "action/actions_with_identity.summary.json",
    "action/events.summary.json",
    "analysis/tracking_metrics.json",
    "analysis/tracking_events.csv",
    "analysis/event_metrics.json",
    "pipeline.log",
)


def build_inspection_bundle(video: dict[str, Any]) -> bytes:
    output = Path(video["output_dir"]).resolve()
    paths = [output / name for name in DIAGNOSTIC_FILES if (output / name).is_file()]
    artifacts = collect_video_artifacts(video)
    covers = []
    for person in artifacts["identity"]["people"][:24]:
        crop = (person.get("cover") or {}).get("crop_path")
        if crop:
            candidate = within_root(output / "identity" / crop, output)
            if candidate.is_file():
                covers.append(candidate)
    paths.extend(covers)
    paths = [within_root(path, output) for path in paths]
    if sum(path.stat().st_size for path in paths) > 64 * 1024**2:
        raise ValueError(
            "Diagnostic files exceed 64 MiB; download individual artifacts instead"
        )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "inspection.json",
            json.dumps(
                {
                    "video": video,
                    "available": artifacts["available"],
                    "event_count": artifacts["action"]["index"]["event_count"],
                    "warnings": artifacts["warnings"],
                    "scope": "clip-local archives; scores are not calibrated probabilities",
                    "note": "Includes at most 24 identity covers. Videos and tensor features are excluded. Upload a video separately for visual review.",
                },
                ensure_ascii=False,
                indent=2,
            ),
        )
        for path in paths:
            archive.write(path, path.relative_to(output).as_posix())
    return buffer.getvalue()
