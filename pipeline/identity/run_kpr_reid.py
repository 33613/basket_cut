"""Extract KPR identity evidence from MOTIP tracks.

This first integration is deliberately conservative: it samples high-quality
person crops, builds one visibility-aware KPR prototype per track, and ranks
temporally compatible track pairs.  It never rewrites MOTIP IDs automatically.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from pipeline.common.schema import (
    TrackRecord,
    read_jsonl,
    write_json,
    write_jsonl_line,
)
from pipeline.identity.kpr_backend import KPRBackend


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract KPR track prototypes and ReID pair distances"
    )
    parser.add_argument("--input", required=True, type=Path, help="Source video")
    parser.add_argument("--tracks", required=True, type=Path, help="MOTIP tracks.jsonl")
    parser.add_argument("--kpr-root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--samples-per-track", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--min-det-score", type=float, default=0.5)
    parser.add_argument("--min-box-width", type=float, default=16.0)
    parser.add_argument("--min-box-height", type=float, default=40.0)
    parser.add_argument("--crop-padding", type=float, default=0.05)
    parser.add_argument(
        "--max-gap-frames",
        type=int,
        default=90,
        help="Only propose merging non-overlapping tracklets separated by at most N frames",
    )
    parser.add_argument(
        "--max-overlap-frames",
        type=int,
        default=0,
        help="Maximum temporal overlap for a merge candidate (default: 0)",
    )
    parser.add_argument(
        "--candidate-threshold",
        type=float,
        help=(
            "Optional normalized KPR distance threshold. Omit until calibrated; "
            "all pair distances are still written."
        ),
    )
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


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


def load_selected_crops(
    video_path: Path,
    selected: list[TrackRecord],
    *,
    padding: float,
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
            # KPR's official demo also passes OpenCV BGR arrays directly.
            samples.append({"image": crop})
            metadata.append(
                {
                    "track_id": record.track_id,
                    "frame_idx": record.frame_idx,
                    "timestamp_s": record.timestamp_s,
                    "det_score": record.det_score,
                    "bbox_xyxy": list(record.bbox_xyxy),
                    "crop_xyxy": list(crop_xyxy),
                }
            )
        frame_idx += 1
    capture.release()

    missing = len(selected) - len(samples)
    if missing:
        raise RuntimeError(
            f"Video ended before {missing} selected track crops could be decoded"
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
        track_visibility = visibility[indices].clamp_min(0)
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


def run(args: argparse.Namespace) -> None:
    if not 0.0 <= args.min_det_score <= 1.0:
        raise ValueError("--min-det-score must be in [0, 1]")
    if not 0.0 <= args.crop_padding <= 1.0:
        raise ValueError("--crop-padding must be in [0, 1]")
    if args.candidate_threshold is not None and not (
        0.0 <= args.candidate_threshold <= 1.0
    ):
        raise ValueError("--candidate-threshold must be in [0, 1]")

    video_path = args.input.expanduser().resolve()
    tracks_path = args.tracks.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}; pass --overwrite"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

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
                sample_count=args.samples_per_track,
                min_score=args.min_det_score,
                min_width=args.min_box_width,
                min_height=args.min_box_height,
            )
        )
    if not selected:
        raise ValueError("No tracks passed the KPR sampling filters")

    samples, sample_metadata = load_selected_crops(
        video_path, selected, padding=args.crop_padding
    )
    backend = KPRBackend(
        kpr_root=args.kpr_root,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        use_gpu=not args.cpu,
    )
    embeddings, visibility = backend.extract(samples, batch_size=args.batch_size)
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
        distances=prototype_distances.numpy(),
    )

    sample_metadata_path = output_dir / "kpr_samples.jsonl"
    with sample_metadata_path.open("w", encoding="utf-8") as handle:
        for value in sample_metadata:
            write_jsonl_line(handle, value)

    track_summaries = []
    for track_id in track_ids:
        indices = track_to_indices[track_id]
        sample_distances = backend.distances(
            embeddings[indices],
            embeddings[indices],
            visibility[indices],
            visibility[indices],
        )
        stats = off_diagonal_stats(sample_distances)
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

    pairs_path = output_dir / "kpr_track_pairs.jsonl"
    candidate_count = 0
    pairs = []
    for left_idx, right_idx in itertools.combinations(range(len(track_ids)), 2):
        left_id, right_id = track_ids[left_idx], track_ids[right_idx]
        overlap, gap = temporal_relation(
            records_by_track[left_id], records_by_track[right_id]
        )
        compatible = (
            overlap <= args.max_overlap_frames and gap <= args.max_gap_frames
        )
        distance = float(prototype_distances[left_idx, right_idx].item())
        is_candidate = (
            None
            if args.candidate_threshold is None
            else compatible and distance <= args.candidate_threshold
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
                "candidate_threshold": args.candidate_threshold,
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
        "prompt_mode": "none",
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "sample_count": len(sample_metadata),
        "track_count": len(track_ids),
        "candidate_count": candidate_count,
        "candidate_threshold": args.candidate_threshold,
        "warning": (
            "KPR distances are cross-domain evidence, not calibrated identity "
            "probabilities. No MOTIP track IDs were modified."
        ),
        "tracks_summary": track_summaries,
    }
    write_json(output_dir / "kpr_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
