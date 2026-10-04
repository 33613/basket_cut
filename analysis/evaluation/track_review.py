"""Human-audited track purity and fragmentation, not detector accuracy.

Review labels are clip-local. Sampling can disprove purity but cannot prove it;
sampled and full-track verdicts therefore have separate denominators.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

from contracts.schema import read_jsonl, write_json
from contracts.tracks import load_tracks

VERDICTS = {"pure", "mixed", "non_player", "uncertain", "unreviewed"}
SCOPES = {"sampled", "full_track"}


def review_index(tracks: Path, video_meta: Path, quality: Path | None = None) -> dict:
    meta = json.loads(video_meta.read_text(encoding="utf-8"))
    grouped = load_tracks(tracks, meta)
    clock = {key: meta.get(key) for key in (
        "video_id", "fps", "processed_frames", "frame_count", "width", "height"
    )}
    digest = hashlib.sha256(json.dumps(clock, sort_keys=True).encode())
    with tracks.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    quality_rows = {}
    if quality and quality.is_file():
        values = list(read_jsonl(quality))
        quality_rows = {int(row["track_id"]): row for row in values}
        if len(values) != len(quality_rows) or set(quality_rows) != set(grouped):
            raise ValueError("Quality report must cover every raw track exactly once; review the original baseline, not a legacy filtered copy")
    rows = []
    for tid, observations in grouped.items():
        info = quality_rows.get(tid, {})
        rows.append({
            "raw_track_id": tid,
            "start_frame": observations[0].frame_idx,
            "end_frame": observations[-1].frame_idx,
            "start": observations[0].timestamp_s,
            "end": (observations[-1].frame_idx + 1) / float(meta["fps"]),
            "observation_count": len(observations),
            "retained": bool(info.get("retained_observations", len(observations))),
            "quality_status": info.get("status", "not_checked"),
            "quality_reasons": info.get("reasons", []),
            "samples": [],
        })
    return {
        "schema_version": 1, "video_id": meta["video_id"],
        "fingerprint": digest.hexdigest(), "fps": meta["fps"],
        "processed_frames": meta.get("processed_frames", meta.get("frame_count")),
        "quality_available": bool(quality and quality.is_file()),
        "tracks": rows,
    }


def validate_review(index: dict, review: dict | None) -> dict:
    """Reject stale reviews, contradictory identities and duplicate entries."""
    if review is None:
        return {"schema_version": 1, "video_id": index["video_id"],
                "fingerprint": index["fingerprint"], "tracks": []}
    if not isinstance(review, dict) or review.get("schema_version") != 1:
        raise ValueError("Review requires schema_version=1")
    if (review.get("video_id"), review.get("fingerprint")) != (
        index["video_id"], index["fingerprint"]
    ):
        raise ValueError("Review belongs to different/stale tracks; prepare a new review")
    valid = {row["raw_track_id"] for row in index["tracks"]}
    seen, clean = set(), []
    if not isinstance(review.get("tracks"), list):
        raise ValueError("Review tracks must be a list")
    for row in review["tracks"]:
        if not isinstance(row, dict):
            raise ValueError("Each review track must be an object")
        tid = row.get("raw_track_id")
        if type(tid) is not int or tid not in valid or tid in seen:
            raise ValueError("Unknown or duplicate raw_track_id in review")
        verdict = row.get("verdict", "unreviewed")
        scope = row.get("scope", "sampled")
        if verdict not in VERDICTS or scope not in SCOPES:
            raise ValueError("Invalid verdict or review scope")
        label = row.get("identity_label") or None
        if label is not None and (not isinstance(label, str) or len(label) > 120):
            raise ValueError("identity_label must be a short clip-local label")
        if label is not None:
            label = label.strip() or None
        if label and verdict != "pure":
            raise ValueError("Only pure tracks may receive one identity_label")
        note = row.get("note", "")
        if not isinstance(note, str) or len(note) > 1000:
            raise ValueError("Review note must be a string of at most 1000 characters")
        seen.add(tid)
        clean.append({"raw_track_id": tid, "verdict": verdict, "scope": scope,
                      "identity_label": label, "note": note})
    return {"schema_version": 1, "video_id": index["video_id"],
            "fingerprint": index["fingerprint"],
            "tracks": sorted(clean, key=lambda row: row["raw_track_id"])}


def _scope_counts(rows: list[dict], verdicts: dict, scope: str) -> dict:
    counts = Counter(verdicts.get(row["raw_track_id"], {}).get("verdict", "unreviewed")
                     if verdicts.get(row["raw_track_id"], {}).get("scope") == scope
                     else "unreviewed" for row in rows)
    player_tracks = counts["pure"] + counts["mixed"]
    return {
        "total_tracks": len(rows), "pure": counts["pure"], "mixed": counts["mixed"],
        "non_player": counts["non_player"], "uncertain": counts["uncertain"],
        "unreviewed_in_scope": counts["unreviewed"],
        "assessed_tracks": player_tracks + counts["non_player"],
        "player_tracks_assessed": player_tracks,
        "review_coverage": (player_tracks + counts["non_player"]) / len(rows) if rows else None,
        "pure_track_rate": counts["pure"] / player_tracks if player_tracks else None,
        "mixed_track_rate": counts["mixed"] / player_tracks if player_tracks else None,
    }


def evaluate_review(index: dict, review: dict | None, identity_map: Path | None = None) -> dict:
    clean = validate_review(index, review)
    verdicts = {row["raw_track_id"]: row for row in clean["tracks"]}
    raw = index["tracks"]
    retained = [row for row in raw if row["retained"]]
    scopes = {scope: {"raw": _scope_counts(raw, verdicts, scope),
                       "retained": _scope_counts(retained, verdicts, scope)}
              for scope in sorted(SCOPES)}
    # Grouping is based on independently reviewed full-track identities only.
    labeled = defaultdict(list)
    for row in raw:
        verdict = verdicts.get(row["raw_track_id"], {})
        if (verdict.get("verdict") == "pure" and verdict.get("scope") == "full_track"
                and verdict.get("identity_label")):
            labeled[verdict["identity_label"]].append(row)
    groups = []
    for label, rows in sorted(labeled.items()):
        overlapping = [[a["raw_track_id"], b["raw_track_id"]]
                       for a, b in combinations(rows, 2)
                       if max(a["start"], b["start"]) < min(a["end"], b["end"])]
        groups.append({"identity_label": label,
                       "raw_track_ids": [row["raw_track_id"] for row in rows],
                       "retained_track_ids": [row["raw_track_id"] for row in rows if row["retained"]],
                       "track_count": len(rows), "extra_track_ids": len(rows) - 1,
                       "overlapping_span_pairs": overlapping})
    labeled_count = sum(group["track_count"] for group in groups)
    pure_count = scopes["full_track"]["raw"]["pure"]
    fragmentation = {
        "labeled_pure_tracks": labeled_count, "full_track_pure_tracks": pure_count,
        "identity_label_coverage": labeled_count / pure_count if pure_count else None,
        "reviewed_people": len(groups),
        "people_with_multiple_track_ids": sum(g["track_count"] > 1 for g in groups),
        "extra_track_ids": sum(g["extra_track_ids"] for g in groups),
        "tracks_per_reviewed_person": labeled_count / len(groups) if groups else None,
        "retained_labeled_pure_tracks": sum(len(g["retained_track_ids"]) for g in groups),
        "retained_reviewed_people": sum(bool(g["retained_track_ids"]) for g in groups),
        "retained_extra_track_ids": sum(max(0, len(g["retained_track_ids"]) - 1) for g in groups),
        "groups": groups,
        "warning": "Counts cover labeled pure tracks only. Overlapping spans flag possible duplicate detections; they are not proven sequential fragments. Missing players and mixed-track segments are not counted.",
    }
    archives = None
    if identity_map is not None:
        predicted, seen = defaultdict(list), set()
        for row in read_jsonl(identity_map):
            tid = int(row["raw_track_id"])
            if tid not in {r["raw_track_id"] for r in raw} or tid in seen:
                raise ValueError("Identity map contains unknown/duplicate tracks")
            if row.get("video_id") != index["video_id"] or not row.get("person_id"):
                raise ValueError("Identity map has wrong video_id or empty person_id")
            seen.add(tid)
            predicted[str(row["person_id"])].append(tid)
        checked = []
        for pid, ids in sorted(predicted.items()):
            labels = set()
            unsafe, unknown = [], []
            for tid in ids:
                row = verdicts.get(tid, {})
                if row.get("verdict") in {"mixed", "non_player"}:
                    unsafe.append(tid)  # A sampled counterexample is still a real failure.
                elif row.get("verdict") == "pure" and row.get("scope") == "full_track" and row.get("identity_label"):
                    labels.add(row["identity_label"])
                else:
                    unknown.append(tid)
            status = "conflict" if unsafe or len(labels) > 1 else "unverified" if unknown else "consistent"
            checked.append({"person_id": pid, "raw_track_ids": ids,
                            "status": status, "identity_labels": sorted(labels),
                            "unsafe_track_ids": unsafe, "unverified_track_ids": unknown})
        archives = {"groups": checked,
                    "consistent": sum(g["status"] == "consistent" for g in checked),
                    "conflict": sum(g["status"] == "conflict" for g in checked),
                    "unverified": sum(g["status"] == "unverified" for g in checked),
                    "warning": "If review labels were used to correct this mapping, this is an acceptance check, not independent model accuracy."}
    return {
        "schema_version": 1, "video_id": index["video_id"],
        "fingerprint": index["fingerprint"], "scopes": scopes,
        "quality_available": index.get("quality_available", False),
        "fragmentation": fragmentation, "archives": archives,
        "warning": "Human track audit only: no missing-player recall, box accuracy, action accuracy or cross-clip identity accuracy. Sampled purity is not full-track purity.",
    }


def merge_review(index: dict, review: dict) -> dict:
    """Export explicit corrections; never label or merge an entire mixed track."""
    clean = validate_review(index, review)
    retained = {row["raw_track_id"] for row in index["tracks"] if row["retained"]}
    assignments = [{"raw_track_ids": [row["raw_track_id"]],
                    "identity_label": row["identity_label"]}
                   for row in clean["tracks"] if row["verdict"] == "pure"
                   and row["scope"] == "full_track" and row["identity_label"]
                   and row["raw_track_id"] in retained]
    return {"video_id": index["video_id"], "assignments": assignments,
            "cannot_link": [], "review_fingerprint": index["fingerprint"],
            "blocked_track_ids": [row["raw_track_id"] for row in clean["tracks"]
                                  if row["raw_track_id"] in retained
                                  and row["verdict"] in {"mixed", "non_player", "uncertain"}]}


def summarize_reviews(results: list[dict], *, expected_clips: int, failed: list[dict] | None = None) -> dict:
    failed = failed or []
    scopes = {}
    for scope in sorted(SCOPES):
        scopes[scope] = {}
        for stage in ("raw", "retained"):
            names = ("total_tracks", "pure", "mixed", "non_player", "uncertain",
                     "unreviewed_in_scope", "assessed_tracks", "player_tracks_assessed")
            pooled = {name: sum(r["scopes"][scope][stage][name] for r in results) for name in names}
            denominator = pooled["player_tracks_assessed"]
            pooled.update(
                pure_track_rate=pooled["pure"] / denominator if denominator else None,
                mixed_track_rate=pooled["mixed"] / denominator if denominator else None,
                review_coverage=pooled["assessed_tracks"] / pooled["total_tracks"] if pooled["total_tracks"] else None,
            )
            scopes[scope][stage] = pooled
    tracks = sum(r["fragmentation"]["labeled_pure_tracks"] for r in results)
    people = sum(r["fragmentation"]["reviewed_people"] for r in results)
    return {"expected_clips": expected_clips, "completed_clips": len(results),
            "failed_clips": failed, "complete": not failed and len(results) == expected_clips,
            "completed_subset": {"quality_available": bool(results) and all(r["quality_available"] for r in results), "scopes": scopes, "fragmentation": {
                "labeled_pure_tracks": tracks, "reviewed_people": people,
                "tracks_per_reviewed_person": tracks / people if people else None,
                "people_with_multiple_track_ids": sum(r["fragmentation"]["people_with_multiple_track_ids"] for r in results),
                "extra_track_ids": sum(r["fragmentation"]["extra_track_ids"] for r in results),
                **{key: sum(r["fragmentation"][key] for r in results) for key in (
                    "retained_labeled_pure_tracks", "retained_reviewed_people", "retained_extra_track_ids"
                )},
            }},
            "warning": "Pooled counts, not a mean of clip percentages. Incomplete runs are a completed subset only. A 30-clip development set is not evidence of broad generalization."}


def evaluate_manifest(manifest: Path, output: Path) -> dict:
    entries = json.loads(manifest.read_text(encoding="utf-8"))["clips"]
    if not entries or len({e["name"] for e in entries}) != len(entries):
        raise ValueError("Manifest requires nonempty clips with unique names")
    inputs = {manifest.resolve()}
    for entry in entries:
        inputs.update((manifest.parent / entry[key]).resolve() for key in (
            "tracks", "video_meta", "quality", "review", "identity_map"
        ) if entry.get(key))
    if output.resolve() in inputs:
        raise ValueError("Metrics cannot overwrite the manifest or any evaluation input")
    results, failures, seen = [], [], set()
    for entry in entries:
        def path(key):
            value = entry.get(key)
            return (manifest.parent / value).resolve() if value else None
        try:
            index = review_index(path("tracks"), path("video_meta"), path("quality"))
            if index["fingerprint"] in seen:
                raise ValueError("Duplicate reviewed clip fingerprint in batch")
            seen.add(index["fingerprint"])
            review_path = path("review")
            review = json.loads(review_path.read_text(encoding="utf-8")) if review_path else None
            result = evaluate_review(index, review, path("identity_map"))
            results.append({"name": entry["name"], **result})
        except (OSError, ValueError, KeyError, TypeError) as exc:
            failures.append({"name": entry["name"], "error": str(exc)})
    summary = summarize_reviews(results, expected_clips=len(entries), failed=failures)
    write_json(output, {**summary, "clips": results})
    return summary
