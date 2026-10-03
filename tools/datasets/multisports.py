"""Convert official MultiSports action tubes to the project event contract."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from contracts.schema import EventRecord, write_json, write_jsonl_line


OFFICIALLY_UNEVALUATED_LABELS = {
    "basketball_save",
    "basketball_jump_ball",
}
VIDEO_SPLITS = ("all", "train", "validation")

# The newer split-specific Hugging Face annotations (for example
# ``data/test/multisports_test.pkl``) omit the label vocabulary that was
# embedded in the original ``multisports_GT.pkl``.  Tube dictionaries still
# use the same official zero-based class indices, so keep the published
# vocabulary here as a deterministic fallback.
OFFICIAL_MULTISPORTS_LABELS = (
    "aerobic push up",
    "aerobic explosive push up",
    "aerobic explosive support",
    "aerobic leg circle",
    "aerobic helicopter",
    "aerobic support",
    "aerobic v support",
    "aerobic horizontal support",
    "aerobic straight jump",
    "aerobic illusion",
    "aerobic bent leg(s) jump",
    "aerobic pike jump",
    "aerobic straddle jump",
    "aerobic split jump",
    "aerobic scissors leap",
    "aerobic kick jump",
    "aerobic off axis jump",
    "aerobic butterfly jump",
    "aerobic split",
    "aerobic turn",
    "aerobic balance turn",
    "volleyball serve",
    "volleyball block",
    "volleyball first pass",
    "volleyball defend",
    "volleyball protect",
    "volleyball second pass",
    "volleyball adjust",
    "volleyball save",
    "volleyball second attack",
    "volleyball spike",
    "volleyball dink",
    "volleyball no offensive attack",
    "football shoot",
    "football long pass",
    "football short pass",
    "football through pass",
    "football cross",
    "football dribble",
    "football trap",
    "football throw",
    "football diving",
    "football tackle",
    "football steal",
    "football clearance",
    "football block",
    "football press",
    "football aerial duels",
    "basketball pass",
    "basketball drive",
    "basketball dribble",
    "basketball 3-point shot",
    "basketball 2-point shot",
    "basketball free throw",
    "basketball block",
    "basketball offensive rebound",
    "basketball defensive rebound",
    "basketball pass steal",
    "basketball dribble steal",
    "basketball interfere shot",
    "basketball pick-and-roll defensive",
    "basketball sag",
    "basketball screen",
    "basketball pass-inbound",
    "basketball save",
    "basketball jump ball",
)


@dataclass(frozen=True)
class MultiSportsReferenceOptions:
    annotation: Path
    output: Path
    videos: tuple[str, ...]
    fps: float = 25.0
    label_prefix: str = "basketball"
    include_unevaluated_labels: bool = False
    overwrite: bool = False


def canonical_label(value: str) -> str:
    """Match the label spelling used by MMAction2's MultiSports label map."""
    return "_".join(value.strip().lower().split())


def _load_pickle(path: Path) -> dict[str, Any]:
    # MultiSports publishes its annotations as a Python pickle.  Pickles can
    # execute code while loading, so this adapter is intentionally limited to
    # official MultiSports pickle files downloaded from MCG-NJU.
    with path.open("rb") as handle:
        value = pickle.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a dictionary in {path}, got {type(value)!r}")
    return value


def _resolve_labels(ground_truth: dict[str, Any]) -> tuple[list[str], str]:
    embedded = ground_truth.get("labels")
    if embedded is not None:
        labels = [str(value) for value in embedded]
        source = "annotation"
    else:
        labels = list(OFFICIAL_MULTISPORTS_LABELS)
        source = "official_fallback"

    tube_indices = {
        int(label_index)
        for video_tubes in ground_truth.get("gttubes", {}).values()
        for label_index in video_tubes
    }
    if tube_indices and max(tube_indices) >= len(labels):
        raise ValueError(
            "Annotation contains a class index outside the known MultiSports "
            f"vocabulary: max={max(tube_indices)}, labels={len(labels)}"
        )
    return labels, source


def _require_tubes(
    ground_truth: dict[str, Any], annotation_path: Path
) -> dict[str, Any]:
    tubes = ground_truth.get("gttubes")
    if not isinstance(tubes, dict):
        keys = ", ".join(sorted(str(key) for key in ground_truth))
        raise ValueError(
            f"{annotation_path} has no gttubes ground truth (keys: {keys}). "
            "The official data/test annotation contains no public action GT; "
            "download data/trainval/multisports_GT.pkl instead."
        )
    return tubes


def _flatten_video_names(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        return [str(value)]
    names: list[str] = []
    for item in value:
        names.extend(_flatten_video_names(item))
    return names


def _videos_for_split(
    ground_truth: dict[str, Any],
    tubes_by_video: dict[str, Any],
    split: str,
) -> list[str]:
    if split not in VIDEO_SPLITS:
        raise ValueError(f"split must be one of {VIDEO_SPLITS}, got {split!r}")
    if split == "all":
        return sorted(str(name) for name in tubes_by_video)
    key = "train_videos" if split == "train" else "test_videos"
    names = _flatten_video_names(ground_truth.get(key))
    if not names:
        raise ValueError(f"Annotation does not contain a non-empty {key!r} split")
    return sorted(name for name in names if name in tubes_by_video)


def list_multisports_videos(
    annotation: Path,
    *,
    label_prefix: str = "basketball",
    split: str = "all",
) -> list[str]:
    """List videos containing at least one action with the selected prefix."""
    annotation_path = annotation.expanduser().resolve()
    ground_truth = _load_pickle(annotation_path)
    tubes_by_video = _require_tubes(ground_truth, annotation_path)
    resolved_labels, _ = _resolve_labels(ground_truth)
    labels = [canonical_label(value) for value in resolved_labels]
    prefix = canonical_label(label_prefix)
    selected = []
    for video_name in _videos_for_split(ground_truth, tubes_by_video, split):
        video_tubes = tubes_by_video[video_name]
        if any(
            labels[int(label_index)].startswith(prefix) and len(tubes) > 0
            for label_index, tubes in video_tubes.items()
        ):
            selected.append(str(video_name))
    return sorted(selected)


def prepare_multisports_reference(
    options: MultiSportsReferenceOptions,
) -> dict[str, Any]:
    if options.fps <= 0:
        raise ValueError("--fps must be positive")
    if not options.videos:
        raise ValueError(
            "Pass at least one --video. Explicit selection prevents accidentally "
            "exporting the complete annotation set."
        )

    annotation_path = options.annotation.expanduser().resolve()
    output_path = options.output.expanduser().resolve()
    if output_path.exists() and not options.overwrite:
        raise FileExistsError(
            f"{output_path} already exists; pass --overwrite to replace it"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ground_truth = _load_pickle(annotation_path)
    tubes_by_video = _require_tubes(ground_truth, annotation_path)
    labels, labels_source = _resolve_labels(ground_truth)

    missing = [video for video in options.videos if video not in tubes_by_video]
    if missing:
        preview = ", ".join(missing[:10])
        raise KeyError(f"Video(s) absent from MultiSports annotations: {preview}")

    records: list[dict[str, Any]] = []
    skipped_labels = 0
    skipped_unevaluated_tubes = 0
    for video_name in options.videos:
        video_tubes = tubes_by_video[video_name]
        for raw_label_index, tubes in video_tubes.items():
            label_index = int(raw_label_index)
            label = canonical_label(labels[label_index])
            if options.label_prefix and not label.startswith(
                canonical_label(options.label_prefix)
            ):
                skipped_labels += len(tubes)
                continue
            if (
                label in OFFICIALLY_UNEVALUATED_LABELS
                and not options.include_unevaluated_labels
            ):
                skipped_unevaluated_tubes += len(tubes)
                continue
            for tube_index, tube in enumerate(tubes):
                if len(tube) == 0:
                    continue
                actor_tube = []
                for row in tube:
                    if len(row) < 5:
                        raise ValueError(
                            f"Invalid tube row for {video_name}/{label}: {row!r}"
                        )
                    # Official MultiSports frame indices start at one.  Every
                    # project contract and MOTIP artifact starts at zero.
                    frame_idx = int(row[0]) - 1
                    actor_tube.append(
                        [frame_idx, *(float(value) for value in row[1:5])]
                    )
                actor_tube.sort(key=lambda item: item[0])
                start_frame = int(actor_tube[0][0])
                end_frame = int(actor_tube[-1][0])
                event_id = (
                    f"{video_name}:L{label_index:02d}:T{tube_index:04d}"
                )
                event = EventRecord(
                    video_id=video_name,
                    event_id=event_id,
                    identity_id=f"{event_id}:actor",
                    event=label,
                    start=start_frame / options.fps,
                    end=(end_frame + 1) / options.fps,
                    raw_score=1.0,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    support_count=len(actor_tube),
                    source="multisports_gt",
                ).to_dict()
                event.update(
                    {
                        "actor_tube": actor_tube,
                        "identity_scope": "event_tube",
                        "label_index": label_index,
                        "reference_video_meta": {
                            "fps": options.fps,
                            "frame_count": int(ground_truth["nframes"][video_name]),
                            "height": int(ground_truth["resolution"][video_name][0]),
                            "width": int(ground_truth["resolution"][video_name][1]),
                        },
                    }
                )
                records.append(event)

    records.sort(
        key=lambda item: (
            item["video_id"],
            item["start_frame"],
            item["end_frame"],
            item["event"],
            item["event_id"],
        )
    )
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            write_jsonl_line(handle, record)

    summary = {
        "annotation": str(annotation_path),
        "output": str(output_path),
        "videos": list(options.videos),
        "fps": options.fps,
        "label_prefix": options.label_prefix,
        "labels_source": labels_source,
        "event_count": len(records),
        "skipped_non_target_tubes": skipped_labels,
        "skipped_officially_unevaluated_tubes": skipped_unevaluated_tubes,
        "include_unevaluated_labels": options.include_unevaluated_labels,
        "time_definition": {
            "start": "first annotated tube frame, inclusive",
            "end": "one frame after the last annotated tube frame, exclusive",
        },
        "identity_scope": "event_tube",
        "warning": (
            "MultiSports supplies the actor tube for every action instance but "
            "does not publish a stable named-player ID across separate tubes. "
            "Actor correctness is therefore evaluated by spatial tube overlap."
        ),
    }
    write_json(output_path.with_suffix(".summary.json"), summary)
    return summary
