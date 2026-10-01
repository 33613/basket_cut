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
    # the official multisports_GT.pkl downloaded from MCG-NJU.
    with path.open("rb") as handle:
        value = pickle.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a dictionary in {path}, got {type(value)!r}")
    return value


def list_multisports_videos(
    annotation: Path, *, label_prefix: str = "basketball"
) -> list[str]:
    """List videos containing at least one action with the selected prefix."""
    ground_truth = _load_pickle(annotation.expanduser().resolve())
    labels = [canonical_label(str(value)) for value in ground_truth["labels"]]
    prefix = canonical_label(label_prefix)
    selected = []
    for video_name, video_tubes in ground_truth["gttubes"].items():
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
    labels = [str(value) for value in ground_truth["labels"]]
    tubes_by_video = ground_truth["gttubes"]

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
