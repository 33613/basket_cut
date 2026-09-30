"""Core KPR identity archive service.

This first integration is deliberately conservative: it samples high-quality
person crops, builds one visibility-aware KPR prototype per track, and ranks
temporally compatible track pairs.  It never rewrites MOTIP IDs automatically.
"""

from __future__ import annotations

import itertools
import math
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from contracts.schema import (
    TrackRecord,
    read_jsonl,
    write_json,
    write_jsonl_line,
)
from pipeline.identity.kpr_backend import KPRBackend, PROMPT_MODES


DEFAULT_KPR_ROOT = Path("KPR")
DEFAULT_KPR_CONFIG = Path("configs/kpr/multidataset_sports_test.yaml")
DEFAULT_KPR_CHECKPOINT = Path(
    "/root/autodl-tmp/models/kpr/"
    "kpr_dancetrack_sportsmot_posetrack21_occludedduke_market_split0.pth.tar"
)
GENERATED_FILES = (
    "kpr_track_sampling.jsonl",
    "kpr_sampling_manifest.jsonl",
    "kpr_prepare_summary.json",
    "kpr_samples.jsonl",
    "kpr_samples.npz",
    "kpr_track_prototypes.npz",
    "kpr_track_pairs.jsonl",
    "kpr_summary.json",
    "identity_map.jsonl",
    "identities.jsonl",
    "identity_archive_manifest.json",
)


@dataclass(frozen=True)
class IdentityArchiveOptions:
    input: Path
    tracks: Path
    output_dir: Path
    kpr_root: Path = DEFAULT_KPR_ROOT
    config: Path = DEFAULT_KPR_CONFIG
    checkpoint: Path = DEFAULT_KPR_CHECKPOINT
    prompt_mode: str = "none"
    keypoints: Path | None = None
    samples_per_track: int = 8
    archive_exemplars: int = 4
    batch_size: int = 16
    min_det_score: float = 0.5
    min_box_width: float = 16.0
    min_box_height: float = 40.0
    crop_padding: float = 0.05
    context_padding: float = 0.5
    max_gap_frames: int = 90
    max_overlap_frames: int = 0
    candidate_threshold: float | None = None
    cpu: bool = False
    prepare_only: bool = False
    overwrite: bool = False


def load_keypoint_prompts(path: Path | None) -> dict[tuple[int, int], dict[str, Any]]:
    """Load strict per-track/per-frame COCO-17 prompts from JSONL."""
    if path is None:
        return {}
    prompts: dict[tuple[int, int], dict[str, Any]] = {}
    for value in read_jsonl(path):
        try:
            key = (int(value["track_id"]), int(value["frame_idx"]))
        except KeyError as exc:
            raise ValueError(
                f"Keypoint prompt is missing required field {exc.args[0]!r}"
            ) from exc
        if key in prompts:
            raise ValueError(f"Duplicate keypoint prompt for track/frame {key}")
        coordinate_space = value.get("coordinate_space", "frame")
        if coordinate_space not in ("frame", "crop"):
            raise ValueError(
                f"Prompt {key} has invalid coordinate_space={coordinate_space!r}"
            )
        prompts[key] = value
    return prompts


def keypoints_for_crop(
    value: Any,
    *,
    crop_xyxy: tuple[int, int, int, int],
    coordinate_space: str,
    allow_many: bool,
):
    import numpy as np

    array = np.asarray(value, dtype=np.float32)
    if allow_many and array.size == 0:
        return np.empty((0, 17, 3), dtype=np.float32)
    expected = (None, 17, 3) if allow_many else (17, 3)
    valid = (
        array.ndim == 3 and array.shape[1:] == (17, 3)
        if allow_many
        else array.shape == (17, 3)
    )
    if not valid:
        raise ValueError(
            f"Expected {'N x ' if allow_many else ''}17 x 3 COCO keypoints, "
            f"got shape {array.shape}; expected {expected}"
        )
    array = array.copy()
    left, top, right, bottom = crop_xyxy
    if coordinate_space == "frame":
        array[..., 0] -= left
        array[..., 1] -= top
    crop_width = right - left
    crop_height = bottom - top
    outside = (
        (array[..., 0] < 0)
        | (array[..., 0] >= crop_width)
        | (array[..., 1] < 0)
        | (array[..., 1] >= crop_height)
    )
    array[..., 2][outside] = 0
    return array


def choose_track_samples(
    records: Iterable[TrackRecord],
    *,
    sample_count: int,
    min_score: float,
    min_width: float,
    min_height: float,
) -> list[TrackRecord]:
    """Select temporally diverse, high-confidence observations for one track."""
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    eligible = []
    for record in sorted(records, key=lambda item: item.frame_idx):
        x1, y1, x2, y2 = record.bbox_xyxy
        if (
            record.det_score >= min_score
            and x2 - x1 >= min_width
            and y2 - y1 >= min_height
        ):
            eligible.append(record)
    if len(eligible) <= sample_count:
        return eligible

    # Pick the highest-confidence observation from each temporal bin.  This is
    # less redundant than simply taking the global top-N scores.
    selected = []
    for bin_idx in range(sample_count):
        start = math.floor(bin_idx * len(eligible) / sample_count)
        end = math.floor((bin_idx + 1) * len(eligible) / sample_count)
        selected.append(
            max(
                eligible[start:end],
                key=lambda item: (item.det_score, -item.frame_idx),
            )
        )
    return sorted(selected, key=lambda item: item.frame_idx)


def build_track_sampling_diagnostics(
    records_by_track: dict[int, list[TrackRecord]],
    selected: list[TrackRecord],
    *,
    min_score: float,
    min_width: float,
    min_height: float,
) -> list[dict[str, Any]]:
    """Explain which raw tracks entered KPR and why others were excluded."""
    selected_counts: dict[int, int] = defaultdict(int)
    for record in selected:
        selected_counts[record.track_id] += 1

    diagnostics = []
    for track_id in sorted(records_by_track):
        records = records_by_track[track_id]
        score_pass_count = sum(record.det_score >= min_score for record in records)
        size_pass_count = sum(
            record.bbox_xyxy[2] - record.bbox_xyxy[0] >= min_width
            and record.bbox_xyxy[3] - record.bbox_xyxy[1] >= min_height
            for record in records
        )
        eligible_count = sum(
            record.det_score >= min_score
            and record.bbox_xyxy[2] - record.bbox_xyxy[0] >= min_width
            and record.bbox_xyxy[3] - record.bbox_xyxy[1] >= min_height
            for record in records
        )
        exclusion_reasons = []
        if eligible_count == 0:
            if score_pass_count == 0:
                exclusion_reasons.append("all_detection_scores_below_threshold")
            if size_pass_count == 0:
                exclusion_reasons.append("all_boxes_below_minimum_size")
            if not exclusion_reasons:
                exclusion_reasons.append("score_and_size_filters_never_pass_together")
        selected_count = selected_counts.get(track_id, 0)
        diagnostics.append(
            {
                "track_id": track_id,
                "start_frame": records[0].frame_idx,
                "end_frame": records[-1].frame_idx,
                "observation_count": len(records),
                "score_pass_count": score_pass_count,
                "size_pass_count": size_pass_count,
                "eligible_observation_count": eligible_count,
                "selected_sample_count": selected_count,
                "status": "selected" if selected_count else "excluded",
                "exclusion_reasons": exclusion_reasons,
                "filters": {
                    "min_det_score": min_score,
                    "min_box_width": min_width,
                    "min_box_height": min_height,
                },
            }
        )
    return diagnostics


def archive_person_id(track_id: int) -> str:
    """Return a deterministic archive ID for no-merge mode."""
    return f"P{track_id:04d}"


def padded_crop(
    frame,
    bbox_xyxy: tuple[float, float, float, float],
    *,
    padding: float,
):
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = bbox_xyxy
    pad_x = (x2 - x1) * padding
    pad_y = (y2 - y1) * padding
    left = max(0, math.floor(x1 - pad_x))
    top = max(0, math.floor(y1 - pad_y))
    right = min(width, math.ceil(x2 + pad_x))
    bottom = min(height, math.ceil(y2 + pad_y))
    if right <= left or bottom <= top:
        raise ValueError(f"Invalid clipped crop: {(left, top, right, bottom)}")
    return frame[top:bottom, left:right].copy(), (left, top, right, bottom)


def save_context_image(
    frame,
    bbox_xyxy: tuple[float, float, float, float],
    destination: Path,
    *,
    padding: float,
) -> None:
    import cv2

    context, context_xyxy = padded_crop(frame, bbox_xyxy, padding=padding)
    left, top, _, _ = context_xyxy
    x1, y1, x2, y2 = bbox_xyxy
    cv2.rectangle(
        context,
        (int(x1 - left), int(y1 - top)),
        (int(x2 - left), int(y2 - top)),
        (0, 255, 255),
        2,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), context, [cv2.IMWRITE_JPEG_QUALITY, 90]):
        raise RuntimeError(f"Failed to write context image: {destination}")


def load_selected_crops(
    video_path: Path,
    selected: list[TrackRecord],
    *,
    padding: float,
    context_padding: float,
    output_dir: Path,
    prompt_mode: str,
    prompts: dict[tuple[int, int], dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required: pip install opencv-python") from exc

    by_frame: dict[int, list[TrackRecord]] = defaultdict(list)
    for record in selected:
        by_frame[record.frame_idx].append(record)
    if not by_frame:
        raise ValueError("No track observations passed the crop filters")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    samples: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    media_dir = output_dir / "identity_media" / "samples"
    media_dir.mkdir(parents=True, exist_ok=True)
    missing_prompts: list[tuple[int, int]] = []
    final_frame = max(by_frame)
    frame_idx = 0
    while frame_idx <= final_frame:
        ok, frame = capture.read()
        if not ok:
            break
        for record in sorted(by_frame.get(frame_idx, []), key=lambda item: item.track_id):
            crop, crop_xyxy = padded_crop(
                frame, record.bbox_xyxy, padding=padding
            )
            sample_id = f"track_{record.track_id:04d}_frame_{record.frame_idx:06d}"
            crop_path = media_dir / f"{sample_id}_crop.jpg"
            context_path = media_dir / f"{sample_id}_context.jpg"
            if not cv2.imwrite(
                str(crop_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95]
            ):
                raise RuntimeError(f"Failed to write KPR crop: {crop_path}")
            save_context_image(
                frame,
                record.bbox_xyxy,
                context_path,
                padding=context_padding,
            )
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            # KPR's official demo also passes OpenCV BGR arrays directly.
            sample: dict[str, Any] = {"image": crop}
            prompt = prompts.get((record.track_id, record.frame_idx))
            if prompt_mode == "keypoints":
                if prompt is None:
                    missing_prompts.append((record.track_id, record.frame_idx))
                else:
                    coordinate_space = prompt.get("coordinate_space", "frame")
                    sample["keypoints_xyc"] = keypoints_for_crop(
                        prompt["keypoints_xyc"],
                        crop_xyxy=crop_xyxy,
                        coordinate_space=coordinate_space,
                        allow_many=False,
                    )
                    sample["negative_kps"] = keypoints_for_crop(
                        prompt.get("negative_kps", []),
                        crop_xyxy=crop_xyxy,
                        coordinate_space=coordinate_space,
                        allow_many=True,
                    )
            samples.append(sample)
            crop_height, crop_width = crop.shape[:2]
            metadata.append(
                {
                    "sample_id": sample_id,
                    "track_id": record.track_id,
                    "frame_idx": record.frame_idx,
                    "timestamp_s": record.timestamp_s,
                    "det_score": record.det_score,
                    "bbox_xyxy": list(record.bbox_xyxy),
                    "crop_xyxy": list(crop_xyxy),
                    "crop_width": crop_width,
                    "crop_height": crop_height,
                    "sharpness_laplacian_var": sharpness,
                    "crop_path": str(crop_path.relative_to(output_dir)),
                    "context_path": str(context_path.relative_to(output_dir)),
                    "prompt_mode": prompt_mode,
                    "prompt_present": prompt is not None,
                }
            )
        frame_idx += 1
    capture.release()

    missing = len(selected) - len(samples)
    if missing:
        raise RuntimeError(
            f"Video ended before {missing} selected track crops could be decoded"
        )
    if missing_prompts:
        preview = ", ".join(str(item) for item in missing_prompts[:10])
        raise ValueError(
            "--prompt-mode keypoints requires prompts for every selected "
            f"track/frame; missing {len(missing_prompts)}: {preview}"
        )
    return samples, metadata


def make_track_prototypes(embeddings, visibility, sample_metadata, torch):
    track_to_indices: dict[int, list[int]] = defaultdict(list)
    for index, value in enumerate(sample_metadata):
        track_to_indices[int(value["track_id"])].append(index)

    track_ids = sorted(track_to_indices)
    prototypes = []
    prototype_visibility = []
    for track_id in track_ids:
        indices = track_to_indices[track_id]
        track_embeddings = embeddings[indices]
        track_visibility = visibility[indices].to(
            dtype=track_embeddings.dtype
        ).clamp_min(0)
        weights = track_visibility.unsqueeze(-1)
        denominator = weights.sum(dim=0).clamp_min(1e-12)
        prototype = (track_embeddings * weights).sum(dim=0) / denominator
        prototype = torch.nn.functional.normalize(prototype, p=2, dim=-1)
        prototypes.append(prototype)
        prototype_visibility.append(track_visibility.mean(dim=0))

    return (
        track_ids,
        track_to_indices,
        torch.stack(prototypes, dim=0),
        torch.stack(prototype_visibility, dim=0),
    )


def off_diagonal_stats(matrix) -> dict[str, float | None]:
    import numpy as np

    values = matrix.detach().cpu().numpy()
    if values.shape[0] < 2:
        return {
            "mean_distance": None,
            "p90_distance": None,
            "max_distance": None,
        }
    off_diagonal = values[~np.eye(values.shape[0], dtype=bool)]
    return {
        "mean_distance": float(off_diagonal.mean()),
        "p90_distance": float(np.quantile(off_diagonal, 0.9)),
        "max_distance": float(off_diagonal.max()),
    }


def archive_quality_score(sample: dict[str, Any]) -> tuple[float, dict[str, float]]:
    """Compute a transparent display-quality heuristic, not confidence."""
    sharpness = max(0.0, float(sample["sharpness_laplacian_var"]))
    sharpness_score = min(1.0, math.log1p(sharpness) / math.log1p(1000.0))
    size_score = min(1.0, float(sample["crop_height"]) / 256.0)
    components = {
        "det_score": float(sample["det_score"]),
        "visibility_fraction": float(sample["visibility_fraction"]),
        "sharpness_score": sharpness_score,
        "size_score": size_score,
    }
    quality = (
        0.35 * components["det_score"]
        + 0.30 * components["visibility_fraction"]
        + 0.20 * components["sharpness_score"]
        + 0.15 * components["size_score"]
    )
    return quality, components


def write_identity_archive(
    *,
    output_dir: Path,
    track_ids: list[int],
    track_to_indices: dict[int, list[int]],
    sample_metadata: list[dict[str, Any]],
    track_summaries: list[dict[str, Any]],
    exemplar_count: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Archive one canonical person per raw track without merging IDs."""
    if exemplar_count <= 0:
        raise ValueError("--archive-exemplars must be positive")
    summaries = {int(item["track_id"]): item for item in track_summaries}
    identities: list[dict[str, Any]] = []
    mappings: list[dict[str, Any]] = []
    for track_id in track_ids:
        person_id = archive_person_id(track_id)
        indices = track_to_indices[track_id]
        ranked = sorted(
            (sample_metadata[index] for index in indices),
            key=lambda item: (
                -float(item["representative_score"]),
                float(item["prototype_distance"]),
                int(item["frame_idx"]),
            ),
        )
        cover = ranked[0]
        exemplars = ranked[: min(exemplar_count, len(ranked))]
        person_dir = output_dir / "identity_media" / "identities" / person_id
        person_dir.mkdir(parents=True, exist_ok=True)
        cover_crop = person_dir / "cover.jpg"
        cover_context = person_dir / "cover_context.jpg"
        shutil.copy2(output_dir / cover["crop_path"], cover_crop)
        shutil.copy2(output_dir / cover["context_path"], cover_context)

        exemplar_records = []
        for rank, sample in enumerate(exemplars, start=1):
            destination = person_dir / f"exemplar_{rank:02d}.jpg"
            shutil.copy2(output_dir / sample["crop_path"], destination)
            exemplar_records.append(
                {
                    "rank": rank,
                    "sample_id": sample["sample_id"],
                    "frame_idx": sample["frame_idx"],
                    "crop_path": str(destination.relative_to(output_dir)),
                    "prototype_distance": sample["prototype_distance"],
                    "archive_quality_score": sample["archive_quality_score"],
                }
            )

        summary = summaries[track_id]
        mapping = {
            "raw_track_id": track_id,
            "person_id": person_id,
            "identity_label": None,
            "resolution_mode": "archive_no_merge",
            "status": "unlabeled",
            "confidence": None,
        }
        mappings.append(mapping)
        identities.append(
            {
                "person_id": person_id,
                "identity_label": None,
                "status": "unlabeled",
                "resolution_mode": "archive_no_merge",
                "raw_track_ids": [track_id],
                "start_frame": summary["start_frame"],
                "end_frame": summary["end_frame"],
                "observation_count": summary["observation_count"],
                "sample_count": summary["sample_count"],
                "mean_visible_parts": summary["mean_visible_parts"],
                "within_track": summary["within_track"],
                "cover": {
                    "sample_id": cover["sample_id"],
                    "frame_idx": cover["frame_idx"],
                    "crop_path": str(cover_crop.relative_to(output_dir)),
                    "context_path": str(cover_context.relative_to(output_dir)),
                    "prototype_distance": cover["prototype_distance"],
                    "archive_quality_score": cover["archive_quality_score"],
                },
                "exemplars": exemplar_records,
                "prototype": {
                    "file": "kpr_track_prototypes.npz",
                    "track_id": track_id,
                    "index": track_ids.index(track_id),
                },
            }
        )

    with (output_dir / "identity_map.jsonl").open("w", encoding="utf-8") as handle:
        for value in mappings:
            write_jsonl_line(handle, value)
    with (output_dir / "identities.jsonl").open("w", encoding="utf-8") as handle:
        for value in identities:
            write_jsonl_line(handle, value)
    return identities, mappings


def temporal_relation(
    first: list[TrackRecord], second: list[TrackRecord]
) -> tuple[int, int]:
    first_start, first_end = first[0].frame_idx, first[-1].frame_idx
    second_start, second_end = second[0].frame_idx, second[-1].frame_idx
    overlap = max(0, min(first_end, second_end) - max(first_start, second_start) + 1)
    if overlap:
        return overlap, 0
    if first_end < second_start:
        return 0, second_start - first_end - 1
    return 0, first_start - second_end - 1


def build_identity_archive(options: IdentityArchiveOptions) -> dict[str, Any]:
    if options.prompt_mode not in PROMPT_MODES:
        raise ValueError(
            f"Unsupported --prompt-mode {options.prompt_mode!r}; "
            f"expected one of {PROMPT_MODES}"
        )
    if not 0.0 <= options.min_det_score <= 1.0:
        raise ValueError("--min-det-score must be in [0, 1]")
    if not 0.0 <= options.crop_padding <= 1.0:
        raise ValueError("--crop-padding must be in [0, 1]")
    if not 0.0 <= options.context_padding <= 2.0:
        raise ValueError("--context-padding must be in [0, 2]")
    if options.archive_exemplars <= 0:
        raise ValueError("--archive-exemplars must be positive")
    if options.prompt_mode == "none" and options.keypoints is not None:
        raise ValueError("--keypoints requires --prompt-mode keypoints")
    if options.prompt_mode == "keypoints" and options.keypoints is None:
        raise ValueError("--prompt-mode keypoints requires --keypoints JSONL")
    if options.candidate_threshold is not None and not (
        0.0 <= options.candidate_threshold <= 1.0
    ):
        raise ValueError("--candidate-threshold must be in [0, 1]")

    video_path = options.input.expanduser().resolve()
    tracks_path = options.tracks.expanduser().resolve()
    output_dir = options.output_dir.expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"Input video not found: {video_path}")
    if not tracks_path.is_file():
        raise FileNotFoundError(f"Tracks JSONL not found: {tracks_path}")
    if output_dir.exists() and any(output_dir.iterdir()) and not options.overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}; pass --overwrite"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    media_root = output_dir / "identity_media"
    if options.overwrite:
        if media_root.exists():
            shutil.rmtree(media_root)
        # A failed rerun must not leave apparently valid archive metadata from
        # an older run beside new sampling/debug files.
        for filename in GENERATED_FILES:
            path = output_dir / filename
            if path.exists():
                path.unlink()

    prompts = load_keypoint_prompts(
        options.keypoints.expanduser().resolve() if options.keypoints else None
    )

    all_records = [TrackRecord.from_dict(value) for value in read_jsonl(tracks_path)]
    records_by_track: dict[int, list[TrackRecord]] = defaultdict(list)
    for record in all_records:
        records_by_track[record.track_id].append(record)
    for values in records_by_track.values():
        values.sort(key=lambda item: item.frame_idx)

    selected = []
    for track_id in sorted(records_by_track):
        selected.extend(
            choose_track_samples(
                records_by_track[track_id],
                sample_count=options.samples_per_track,
                min_score=options.min_det_score,
                min_width=options.min_box_width,
                min_height=options.min_box_height,
            )
        )
    track_sampling = build_track_sampling_diagnostics(
        records_by_track,
        selected,
        min_score=options.min_det_score,
        min_width=options.min_box_width,
        min_height=options.min_box_height,
    )
    with (output_dir / "kpr_track_sampling.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for value in track_sampling:
            write_jsonl_line(handle, value)
    if not selected:
        raise ValueError("No tracks passed the KPR sampling filters")

    samples, sample_metadata = load_selected_crops(
        video_path,
        selected,
        padding=options.crop_padding,
        context_padding=options.context_padding,
        output_dir=output_dir,
        prompt_mode=options.prompt_mode,
        prompts=prompts,
    )
    with (output_dir / "kpr_sampling_manifest.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for value in sample_metadata:
            write_jsonl_line(handle, value)
    if options.prepare_only:
        prepare_summary = {
            "status": "prepared_not_inferred",
            "input": str(video_path),
            "tracks": str(tracks_path),
            "prompt_mode": options.prompt_mode,
            "sample_count": len(sample_metadata),
            "input_track_count": len(records_by_track),
            "selected_track_count": len(
                {int(item["track_id"]) for item in sample_metadata}
            ),
            "excluded_track_count": sum(
                item["status"] == "excluded" for item in track_sampling
            ),
            "next_step": (
                "Run the same command without --prepare-only on a GPU instance."
            ),
        }
        write_json(output_dir / "kpr_prepare_summary.json", prepare_summary)
        return prepare_summary
    backend = KPRBackend(
        kpr_root=options.kpr_root,
        config_path=options.config,
        checkpoint_path=options.checkpoint,
        prompt_mode=options.prompt_mode,
        use_gpu=not options.cpu,
    )
    embeddings, visibility = backend.extract(samples, batch_size=options.batch_size)
    track_ids, track_to_indices, prototypes, prototype_visibility = (
        make_track_prototypes(
            embeddings, visibility, sample_metadata, backend.torch
        )
    )
    prototype_distances = backend.distances(
        prototypes, prototypes, prototype_visibility, prototype_visibility
    )

    import numpy as np

    np.savez_compressed(
        output_dir / "kpr_samples.npz",
        embeddings=embeddings.numpy(),
        visibility_scores=visibility.numpy(),
        track_ids=np.asarray([item["track_id"] for item in sample_metadata]),
        frame_indices=np.asarray([item["frame_idx"] for item in sample_metadata]),
        det_scores=np.asarray([item["det_score"] for item in sample_metadata]),
    )
    np.savez_compressed(
        output_dir / "kpr_track_prototypes.npz",
        embeddings=prototypes.numpy(),
        visibility_scores=prototype_visibility.numpy(),
        track_ids=np.asarray(track_ids),
        person_ids=np.asarray([archive_person_id(track_id) for track_id in track_ids]),
        distances=prototype_distances.numpy(),
    )

    track_summaries = []
    for track_index, track_id in enumerate(track_ids):
        indices = track_to_indices[track_id]
        sample_distances = backend.distances(
            embeddings[indices],
            embeddings[indices],
            visibility[indices],
            visibility[indices],
        )
        stats = off_diagonal_stats(sample_distances)
        prototype_distance = backend.distances(
            embeddings[indices],
            prototypes[track_index : track_index + 1],
            visibility[indices],
            prototype_visibility[track_index : track_index + 1],
        ).reshape(-1)
        for local_index, sample_index in enumerate(indices):
            metadata = sample_metadata[sample_index]
            visible = visibility[sample_index].clamp_min(0)
            metadata["visible_parts"] = float(visible.sum().item())
            metadata["visibility_fraction"] = float(visible.mean().item())
            metadata["prototype_distance"] = float(
                prototype_distance[local_index].item()
            )
            quality, components = archive_quality_score(metadata)
            metadata["archive_quality_components"] = components
            metadata["archive_quality_score"] = quality
            metadata["representative_score"] = (
                quality - 0.35 * metadata["prototype_distance"]
            )
        values = records_by_track[track_id]
        track_summaries.append(
            {
                "track_id": track_id,
                "start_frame": values[0].frame_idx,
                "end_frame": values[-1].frame_idx,
                "observation_count": len(values),
                "sample_count": len(indices),
                "mean_visible_parts": float(
                    prototype_visibility[track_ids.index(track_id)].sum().item()
                ),
                "within_track": stats,
            }
        )

    sample_metadata_path = output_dir / "kpr_samples.jsonl"
    with sample_metadata_path.open("w", encoding="utf-8") as handle:
        for value in sample_metadata:
            write_jsonl_line(handle, value)

    identities, identity_mappings = write_identity_archive(
        output_dir=output_dir,
        track_ids=track_ids,
        track_to_indices=track_to_indices,
        sample_metadata=sample_metadata,
        track_summaries=track_summaries,
        exemplar_count=options.archive_exemplars,
    )

    pairs_path = output_dir / "kpr_track_pairs.jsonl"
    candidate_count = 0
    pairs = []
    for left_idx, right_idx in itertools.combinations(range(len(track_ids)), 2):
        left_id, right_id = track_ids[left_idx], track_ids[right_idx]
        overlap, gap = temporal_relation(
            records_by_track[left_id], records_by_track[right_id]
        )
        compatible = (
            overlap <= options.max_overlap_frames and gap <= options.max_gap_frames
        )
        distance = float(prototype_distances[left_idx, right_idx].item())
        is_candidate = (
            None
            if options.candidate_threshold is None
            else compatible and distance <= options.candidate_threshold
        )
        if is_candidate:
            candidate_count += 1
        pairs.append(
            {
                "track_id_a": left_id,
                "track_id_b": right_id,
                "distance": distance,
                "similarity": 1.0 - distance,
                "overlap_frames": overlap,
                "gap_frames": gap,
                "temporally_compatible": compatible,
                "candidate_threshold": options.candidate_threshold,
                "is_merge_candidate": is_candidate,
            }
        )
    pairs.sort(
        key=lambda item: (
            not item["temporally_compatible"],
            item["distance"],
            item["track_id_a"],
            item["track_id_b"],
        )
    )
    with pairs_path.open("w", encoding="utf-8") as handle:
        for value in pairs:
            write_jsonl_line(
                handle,
                value,
            )

    summary = {
        "input": str(video_path),
        "tracks": str(tracks_path),
        "backend": "KPR",
        "prompt_mode": options.prompt_mode,
        "keypoints": str(options.keypoints.expanduser().resolve()) if options.keypoints else None,
        "config": str(options.config.expanduser().resolve()),
        "checkpoint": str(options.checkpoint.expanduser().resolve()),
        "sample_count": len(sample_metadata),
        "input_track_count": len(records_by_track),
        "track_count": len(track_ids),
        "excluded_track_count": sum(
            item["status"] == "excluded" for item in track_sampling
        ),
        "identity_count": len(identities),
        "identity_resolution_mode": "archive_no_merge",
        "candidate_count": candidate_count,
        "candidate_threshold": options.candidate_threshold,
        "warning": (
            "KPR distances are cross-domain evidence, not calibrated identity "
            "probabilities. No MOTIP track IDs were modified."
        ),
        "tracks_summary": track_summaries,
    }
    write_json(output_dir / "kpr_summary.json", summary)
    write_json(
        output_dir / "identity_archive_manifest.json",
        {
            "schema_version": 1,
            "identity_resolution_mode": "archive_no_merge",
            "prompt_mode": options.prompt_mode,
            "artifacts": {
                "track_sampling": "kpr_track_sampling.jsonl",
                "sampling_manifest": "kpr_sampling_manifest.jsonl",
                "samples": "kpr_samples.jsonl",
                "sample_features": "kpr_samples.npz",
                "track_prototypes": "kpr_track_prototypes.npz",
                "track_pairs": "kpr_track_pairs.jsonl",
                "identity_map": "identity_map.jsonl",
                "identities": "identities.jsonl",
                "media_root": "identity_media",
            },
            "counts": {
                "samples": len(sample_metadata),
                "input_raw_tracks": len(records_by_track),
                "raw_tracks": len(track_ids),
                "excluded_raw_tracks": sum(
                    item["status"] == "excluded" for item in track_sampling
                ),
                "canonical_people": len(identity_mappings),
            },
            "person_id_policy": "P + zero-padded raw_track_id",
            "warning": (
                "Each raw track is archived as one canonical person in this "
                "mode. No automatic merging or identity naming is performed."
            ),
        },
    )
    return summary
