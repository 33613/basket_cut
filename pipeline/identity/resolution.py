"""Conservative clip-local identity resolution, separate from KPR extraction."""

from __future__ import annotations

import copy
import json
import math
import shutil
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path

from contracts.schema import read_jsonl, write_json, write_jsonl_line
from contracts.tracks import box_iou, load_tracks


@dataclass(frozen=True)
class ResolutionOptions:
    tracks: Path
    video_meta: Path
    archive_dir: Path
    output_dir: Path
    review: Path | None = None
    quality_dir: Path | None = None
    max_distance: float | None = None
    max_gap_seconds: float = 1.5
    overlap_iou: float = 0.9
    max_within_distance: float = 0.4
    min_samples: int = 2
    overwrite: bool = False


def resolve_identities(options: ResolutionOptions) -> dict:
    """Complete-link clustering with spatial/label conflicts and explicit review."""
    if options.min_samples < 1:
        raise ValueError("min_samples must be positive")
    for key, number in (
        ("max_gap_seconds", options.max_gap_seconds),
        ("max_within_distance", options.max_within_distance),
    ):
        if not math.isfinite(number) or number < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    if not 0 < options.overlap_iou <= 1:
        raise ValueError("overlap_iou must be in (0, 1]")
    if options.max_distance is not None and (
        not math.isfinite(options.max_distance) or options.max_distance < 0
    ):
        raise ValueError("max_distance must be finite and nonnegative")
    meta = json.loads(options.video_meta.read_text(encoding="utf-8"))
    grouped = load_tracks(options.tracks, meta)
    archive = options.archive_dir.resolve()
    output = options.output_dir.resolve()
    if output == archive or archive.is_relative_to(output):
        raise ValueError("Resolved output must not overwrite the raw identity archive")
    manifest_path = archive / "identity_archive_manifest.json"
    warnings = []
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("video_id") and manifest["video_id"] != meta["video_id"]:
            raise ValueError("Identity archive video_id does not match this clip")
        if not manifest.get("video_id"):
            warnings.append("legacy_archive_without_video_id")
    source_people = {}
    for item in read_jsonl(archive / "identities.jsonl"):
        ids = item.get("raw_track_ids", [])
        if len(ids) != 1 or int(ids[0]) in source_people:
            raise ValueError(
                "Resolution input must be a raw one-track-per-archive KPR archive"
            )
        source_people[int(ids[0])] = item
    active = sorted(set(grouped) & set(source_people))
    missing = sorted(set(grouped) - set(source_people))
    quality = {}
    if options.quality_dir:
        quality_summary = json.loads(
            (options.quality_dir / "quality_summary.json").read_text(encoding="utf-8")
        )
        if quality_summary["video_id"] != meta["video_id"]:
            raise ValueError("Quality report video_id does not match this clip")
        quality = {
            int(x["track_id"]): x
            for x in read_jsonl(options.quality_dir / "quality_tracks.jsonl")
        }
    held = set()
    held_reasons = {}
    for tid in active:
        within = source_people[tid].get("within_track") or {}
        spread = within.get("max_distance")
        if spread is not None and (
            not math.isfinite(float(spread)) or float(spread) < 0
        ):
            raise ValueError("Invalid within-track KPR distance")
        reasons = []
        if quality.get(tid, {}).get("status") == "needs_review":
            reasons.append("motion_anomaly")
        if spread is not None and float(spread) > options.max_within_distance:
            reasons.append("appearance_inconsistent")
        if int(source_people[tid].get("sample_count", 0)) < options.min_samples:
            reasons.append("insufficient_kpr_samples")
        if reasons:
            held.add(tid)
            held_reasons[tid] = reasons
    distances = {}
    pair_file = archive / "kpr_track_pairs.jsonl"
    if pair_file.is_file():
        for pair in read_jsonl(pair_file):
            key = tuple(sorted((int(pair["track_id_a"]), int(pair["track_id_b"]))))
            number = float(pair["distance"])
            if (
                key[0] == key[1]
                or key in distances
                or not math.isfinite(number)
                or number < 0
            ):
                raise ValueError("Invalid or duplicate KPR pair distance")
            distances[key] = number
    review = {"video_id": meta["video_id"], "assignments": [], "cannot_link": []}
    if options.review:
        review = json.loads(options.review.read_text(encoding="utf-8"))
        if review.get("video_id") != meta["video_id"]:
            raise ValueError("Review video_id does not match this clip")
    forbidden = set()
    for pair in review.get("cannot_link", []):
        if (
            len(pair) != 2
            or pair[0] == pair[1]
            or any(int(t) not in active for t in pair)
        ):
            raise ValueError("cannot_link requires two distinct active tracks")
        forbidden.add(tuple(sorted(map(int, pair))))
    labels = {}
    manual_groups = {}
    for assignment in review.get("assignments", []):
        ids = list(map(int, assignment["raw_track_ids"]))
        if not isinstance(assignment["identity_label"], str):
            raise TypeError("Manual identity label must be a string")
        label = assignment["identity_label"].strip()
        if not label or not ids or len(ids) != len(set(ids)):
            raise ValueError("Manual assignment requires a label and unique track IDs")
        for tid in ids:
            if tid not in active or tid in labels:
                raise ValueError(
                    f"Unknown, excluded, or multiply assigned track: {tid}"
                )
            labels[tid] = label
        manual_groups.setdefault(label, []).extend(ids)
    observations = {tid: {r.frame_idx: r for r in grouped[tid]} for tid in active}
    relation_cache = {}

    def relation(a, b):
        key = tuple(sorted((a, b)))
        if key in relation_cache:
            return relation_cache[key]
        common = sorted(set(observations[a]) & set(observations[b]))
        ious = [
            box_iou(observations[a][f].bbox_xyxy, observations[b][f].bbox_xyxy)
            for f in common
        ]
        left, right = grouped[a], grouped[b]
        gap = max(
            0,
            max(left[0].frame_idx, right[0].frame_idx)
            - min(left[-1].frame_idx, right[-1].frame_idx)
            - 1,
        ) / float(meta["fps"])
        blocked = None
        if key in forbidden:
            blocked = "manual_cannot_link"
        elif a in labels and b in labels and labels[a] != labels[b]:
            blocked = "different_confirmed_people"
        elif ious and min(ious) < options.overlap_iou:
            blocked = "simultaneous_distinct_boxes"
        result = {
            "track_id_a": key[0],
            "track_id_b": key[1],
            "distance": distances.get(key),
            "coobserved_frames": len(common),
            "min_overlap_iou": min(ious, default=None),
            "gap_seconds": gap,
            "blocked_reason": blocked,
        }
        relation_cache[key] = result
        return result

    groups = [{tid} for tid in active]
    merges = []

    def combine(left, right, reason):
        groups.remove(left)
        groups.remove(right)
        merged = left | right
        groups.append(merged)
        merges.append({"raw_track_ids": sorted(merged), "reason": reason})

    for label, ids in sorted(manual_groups.items()):
        for a, b in combinations(ids, 2):
            blocked = relation(a, b)["blocked_reason"]
            if blocked:
                raise ValueError(f"Manual merge conflicts with {blocked}: {a}, {b}")
        for tid in ids[1:]:
            left = next(g for g in groups if ids[0] in g)
            right = next(g for g in groups if tid in g)
            if left is not right:
                combine(left, right, "human_confirmed")
    if options.max_distance is not None:
        for (a, b), distance in sorted(
            distances.items(), key=lambda item: (item[1], item[0])
        ):
            if a not in active or b not in active or distance > options.max_distance:
                continue
            left = next(g for g in groups if a in g)
            right = next(g for g in groups if b in g)
            if left is right:
                continue
            valid = True
            for x in left:
                for y in right:
                    evidence = relation(x, y)
                    if (
                        evidence["blocked_reason"]
                        or x in held
                        or y in held
                        or evidence["distance"] is None
                        or evidence["distance"] > options.max_distance
                        or evidence["gap_seconds"] > options.max_gap_seconds
                    ):
                        valid = False
            if valid:
                combine(left, right, "kpr_complete_link")
    filenames = (
        "identity_map.jsonl",
        "identities.jsonl",
        "resolution_pairs.jsonl",
        "resolution_summary.json",
        "identity_archive_manifest.json",
    )
    input_files = {options.tracks.resolve(), options.video_meta.resolve()}
    if options.review:
        input_files.add(options.review.resolve())
    if input_files & {(output / name).resolve() for name in filenames}:
        raise ValueError("Resolution output cannot overwrite its inputs")
    if not options.overwrite and any((output / name).exists() for name in filenames):
        raise FileExistsError(
            "Resolution output exists; use a new directory or --overwrite"
        )
    output.mkdir(parents=True, exist_ok=True)

    def copy_media(source, destination):
        path = (archive / source).resolve()
        if not path.is_relative_to(archive) or not path.is_file():
            raise ValueError("Archive media missing or outside the archive")
        target = (output / destination).resolve()
        if not target.is_relative_to(output):
            raise ValueError("Destination media path escapes resolution output")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        return destination

    people = []
    mappings = []
    for members in sorted(groups, key=min):
        ids = sorted(members)
        confirmed = {labels[t] for t in ids if t in labels}
        label = next(iter(confirmed), None)
        status = (
            "human_confirmed"
            if label and all(t in labels for t in ids)
            else "auto_assigned"
            if label
            else "needs_review"
            if members & held
            else "auto_merged"
            if len(ids) > 1
            else "unresolved"
        )
        pid = f"P{min(ids):04d}"
        best = max(
            ids,
            key=lambda t: (
                float(
                    (source_people[t].get("cover") or {}).get(
                        "archive_quality_score", 0
                    )
                ),
                len(grouped[t]),
                -t,
            ),
        )
        person = copy.deepcopy(source_people[best])
        person.update(
            person_id=pid,
            video_id=meta["video_id"],
            identity_label=label,
            status=status,
            resolution_mode="clip_local_resolution",
            raw_track_ids=ids,
            start_frame=min(grouped[t][0].frame_idx for t in ids),
            end_frame=max(grouped[t][-1].frame_idx for t in ids),
            observation_count=sum(len(grouped[t]) for t in ids),
            sample_count=sum(source_people[t].get("sample_count", 0) for t in ids),
            source_archives=[source_people[t]["person_id"] for t in ids],
            representative_raw_track_id=best,
            within_track=source_people[best].get("within_track")
            if len(ids) == 1
            else None,
            unique_observed_frames=len(
                set().union(*(set(observations[t]) for t in ids))
            ),
            review_reasons=sorted(
                {reason for t in members & held for reason in held_reasons[t]}
            ),
        )
        # Avoid presenting one constituent prototype/spread as the merged person's.
        person.pop("prototype", None)
        person["cover"]["raw_track_id"] = best
        for kind in ("crop_path", "context_path"):
            if person.get("cover", {}).get(kind):
                person["cover"][kind] = copy_media(
                    person["cover"][kind], f"identity_media/{pid}/{kind}.jpg"
                )
        exemplars = []
        for tid in ids:
            for index, sample in enumerate(source_people[tid].get("exemplars", [])[:2]):
                if len(exemplars) >= 8:
                    break
                item = dict(sample)
                item["crop_path"] = copy_media(
                    item["crop_path"], f"identity_media/{pid}/T{tid}_{index}.jpg"
                )
                item["raw_track_id"] = tid
                exemplars.append(item)
        person["exemplars"] = exemplars
        people.append(person)
        for tid in ids:
            mappings.append(
                {
                    "raw_track_id": tid,
                    "person_id": pid,
                    "video_id": meta["video_id"],
                    "identity_label": label,
                    "status": "human_confirmed" if tid in labels else status,
                    "resolution_mode": "clip_local_resolution",
                    "confidence": None,
                }
            )
    pair_evidence = []
    for a, b in combinations(active, 2):
        evidence = dict(relation(a, b))
        evidence["same_resolved_person"] = any(
            a in group and b in group for group in groups
        )
        evidence["requires_review"] = a in held or b in held
        reasons = []
        if evidence["blocked_reason"]:
            reasons.append(evidence["blocked_reason"])
        if evidence["requires_review"]:
            reasons.extend(
                sorted(set(held_reasons.get(a, []) + held_reasons.get(b, [])))
            )
        if evidence["distance"] is None:
            reasons.append("missing_kpr_distance")
        elif (
            options.max_distance is not None
            and evidence["distance"] > options.max_distance
        ):
            reasons.append("kpr_distance_above_threshold")
        if evidence["gap_seconds"] > options.max_gap_seconds:
            reasons.append("temporal_gap_above_threshold")
        if options.max_distance is None:
            reasons.append("automatic_merge_disabled")
        if not reasons and not evidence["same_resolved_person"]:
            reasons.append("cluster_complete_link_conflict")
        evidence["auto_block_reasons"] = reasons
        pair_evidence.append(evidence)
    for name, rows in (
        ("identity_map.jsonl", mappings),
        ("identities.jsonl", people),
        ("resolution_pairs.jsonl", pair_evidence),
    ):
        with (output / name).open("w", encoding="utf-8") as handle:
            for row in rows:
                write_jsonl_line(handle, row)
    summary = {
        "video_id": meta["video_id"],
        "identity_count": len(people),
        "active_track_count": len(active),
        "warnings": warnings,
        "missing_archive_tracks": missing,
        "excluded_archive_tracks": sorted(set(source_people) - set(active)),
        "review_track_ids": sorted(held),
        "merges": merges,
        "identity_resolution_mode": "clip_local_resolution",
        "human_confirmed_people": sum(p["status"] == "human_confirmed" for p in people),
        "settings": {
            k: v
            for k, v in asdict(options).items()
            if k
            not in (
                "tracks",
                "video_meta",
                "archive_dir",
                "output_dir",
                "review",
                "quality_dir",
            )
        },
        "warning": "Clip-local identities, not verified real-world names. KPR distances are not probabilities. No automatic merge without an explicit threshold; mixed tracks require review, not an automatic identity rewrite.",
    }
    write_json(output / "resolution_summary.json", summary)
    write_json(
        output / "identity_archive_manifest.json",
        {
            "video_id": meta["video_id"],
            "identity_resolution_mode": "clip_local_resolution",
            "identity_count": len(people),
            "scope": "clip_local",
        },
    )
    return summary
