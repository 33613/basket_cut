"""Build a match-scoped person/event library from completed clip artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from contracts.schema import EventRecord, read_jsonl
from pipeline.identity.jersey import group_numbers


@dataclass(frozen=True)
class PersonLibraryOptions:
    run_dir: Path
    output_dir: Path | None = None
    max_distance: float | None = .2
    min_samples: int = 2
    max_within_distance: float = 0.4
    min_common_parts: int = 2
    use_jersey_evidence: bool = False
    match_id: str | None = None
    clip_names: tuple[str, ...] | None = None
    novelty_distance: float = 0.5
    min_margin: float = 0.05
    gallery_limit: int = 12


def safe_relative(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Artifact path escapes its result directory")
    return path


def file_stamp(path: Path) -> list[int] | None:
    if not path.is_file():
        return None
    stat = path.stat()
    return [stat.st_size, stat.st_mtime_ns]


def load_library_inputs(options: PersonLibraryOptions):
    if options.min_samples < 1 or options.min_common_parts < 1:
        raise ValueError("Sample and visible-part minima must be positive")
    if not math.isfinite(options.max_within_distance) or options.max_within_distance < 0:
        raise ValueError("Invalid within-track distance threshold")
    root = options.run_dir.expanduser().resolve()
    manifest = json.loads((root / "batch_manifest.json").read_text())
    names = list(options.clip_names) if options.clip_names is not None else [row["name"] for row in manifest["clips"]]
    if not names or len(names) != len(set(names)) or any(Path(name).name != name for name in names):
        raise ValueError("Batch must contain unique, safe clip names")
    nodes, features, visibility, events, signatures, warnings = [], [], [], [], {}, []
    contract, jersey_contract = None, None

    def record(relative):
        path = safe_relative(root, relative)
        signatures[relative] = file_stamp(path)
        return path

    for name in names:
        clip = safe_relative(root, name)
        # Operational timestamps change on a no-op resume; they are not features.
        status_path = safe_relative(root, f"{name}/batch_status.json")
        if options.clip_names is None and status_path.is_file() and json.loads(status_path.read_text()).get("status") != "completed":
            raise ValueError(f"Clip is not complete: {name}; finish/retry the batch first")
        meta = json.loads(record(f"{name}/tracking/video_meta.json").read_text())
        raw = list(read_jsonl(record(f"{name}/identity_raw/identities.jsonl")))
        local = list(read_jsonl(record(f"{name}/identity/identities.jsonl")))
        resolution = json.loads(record(f"{name}/identity/resolution_summary.json").read_text())
        if resolution.get("missing_archive_tracks"):
            warnings.append(f"{name}: tracks without KPR archives are not represented")
        event_path = record(f"{name}/action/events.jsonl")
        if options.clip_names is None and manifest.get("target", "full") == "full" and not event_path.is_file():
            raise FileNotFoundError(f"Missing required events: {event_path}")
        raw_by_track = {}
        for person in raw:
            ids = person["raw_track_ids"]
            if len(ids) != 1 or int(ids[0]) in raw_by_track:
                raise ValueError("Expected one raw KPR archive per track")
            raw_by_track[int(ids[0])] = person
        jersey_tracks = {}
        if options.use_jersey_evidence:
            jersey_summary = json.loads(record(f"{name}/identity/jersey_summary.json").read_text())
            if jersey_summary.get("backend") != "qwen_vl":
                raise ValueError("Jersey constraints require local Qwen VL evidence")
            for row in read_jsonl(record(f"{name}/identity/jersey_tracks.jsonl")):
                tid = int(row["raw_track_id"])
                if tid in jersey_tracks or tid not in raw_by_track:
                    raise ValueError("Duplicate or foreign Qwen raw track")
                if row.get("number") is not None:
                    number = row["number"]
                    if (not isinstance(number, str) or not number.isascii() or not number.isdigit()
                            or not 1 <= len(number) <= 2 or row.get("status") != "candidate_consensus"):
                        raise ValueError("Invalid Qwen number candidate")
                for reading in row.get("readings", []):
                    if reading.get("crop_path"):
                        safe_relative(clip / "identity", reading["crop_path"])
                jersey_tracks[tid] = row
            if set(jersey_tracks) != set(raw_by_track):
                raise ValueError("Qwen evidence does not cover the current raw KPR archives")
            # Refuse mixtures of models or different acceptance rules across this match.
            if raw:
                signature = {"provenance": jersey_summary["provenance"], "thresholds": jersey_summary["thresholds"]}
                if jersey_contract is not None and signature != jersey_contract:
                    raise ValueError("Incompatible Qwen models or thresholds across clips")
                jersey_contract = signature
        prototype_path = record(f"{name}/identity_raw/kpr_track_prototypes.npz")
        summary_path = record(f"{name}/identity_raw/kpr_summary.json")
        track_features = {}
        if raw:
            summary = json.loads(summary_path.read_text())
            with np.load(prototype_path, allow_pickle=False) as data:
                vectors = data["embeddings"].astype(np.float32)
                weights = data["visibility_scores"].astype(np.float32)
                track_ids = [int(value) for value in data["track_ids"]]
            if vectors.ndim != 3 or len(track_ids) != len(vectors) or len(set(track_ids)) != len(track_ids):
                raise ValueError(f"Invalid KPR prototype arrays: {name}")
            if weights.shape != vectors.shape[:2] or set(track_ids) != set(raw_by_track):
                raise ValueError(f"KPR prototype/identity mismatch: {name}")
            signature = summary.get("feature_contract")
            if not signature:
                raise ValueError(f"{name}: KPR feature provenance is missing; re-extract into a new run")
            signature = {"metadata": signature, "shape": list(vectors.shape[1:])}
            if contract is not None and signature != contract:
                raise ValueError("Incompatible KPR checkpoints, prompts or feature layouts across clips")
            contract = signature
            for index, tid in enumerate(track_ids):
                track_features[tid] = len(features)
                features.append(vectors[index])
                visibility.append(weights[index])
        track_path = clip / "quality/tracks.jsonl"
        if not track_path.is_file():
            track_path = clip / "tracking/tracks.jsonl"
        track_times = {}
        for observation in read_jsonl(record(str(track_path.relative_to(root)))):
            tid = int(observation["track_id"])
            track_times.setdefault(tid, []).append(float(observation["timestamp_s"]))
        seen_person_ids, seen_tracks = set(), set()
        for person in local:
            pid = str(person["person_id"])
            tids = list(map(int, person["raw_track_ids"]))
            if not pid or pid in seen_person_ids or not tids or len(set(tids)) != len(tids) or seen_tracks.intersection(tids):
                raise ValueError(f"Invalid or duplicate local identity: {name}")
            seen_person_ids.add(pid)
            seen_tracks.update(tids)
            reasons = set(person.get("review_reasons", []))
            if person.get("status") == "needs_review":
                reasons.add("local_archive_needs_review")
            for tid in tids:
                item = raw_by_track.get(tid)
                if item is None or tid not in track_features:
                    raise ValueError(f"Resolved identity references missing KPR track: {name}/{tid}")
                if int(item.get("sample_count", 0)) < options.min_samples:
                    reasons.add("insufficient_kpr_samples")
                spread = (item.get("within_track") or {}).get("max_distance")
                if spread is not None and (not math.isfinite(float(spread)) or float(spread) < 0):
                    raise ValueError("Invalid within-track spread")
                if spread is not None and float(spread) > options.max_within_distance:
                    reasons.add("appearance_inconsistent")
                if int(np.count_nonzero(visibility[track_features[tid]])) < options.min_common_parts:
                    reasons.add("insufficient_visible_parts")
            for sample in [person.get("cover") or {}, *person.get("exemplars", [])]:
                for key in ("crop_path", "context_path"):
                    if sample.get(key):
                        safe_relative(clip / "identity", sample[key])
            jersey = group_numbers(tids, jersey_tracks) if options.use_jersey_evidence else None
            if jersey is not None:
                jersey["backend"] = "qwen_vl"
                jersey["provenance"] = jersey_contract
                jersey["tracks"] = [jersey_tracks[tid] for tid in tids]
                if jersey["status"] == "conflict":
                    reasons.add("jersey_number_conflict")
            node_id = "N-" + hashlib.sha256(f"{name}\0{pid}".encode()).hexdigest()[:16]
            nodes.append({"node_id": node_id, "clip_name": name, "video_id": meta["video_id"],
                          "filename": Path(meta["input_path"]).name, "local_person_id": pid,
                          "raw_track_ids": tids,
                          "intervals": [[min(track_times[t]), max(track_times[t]) + 1 / float(meta["fps"])]
                                        for t in tids if t in track_times], "sample_count": int(person.get("sample_count", 0)),
                          "observation_count": int(person.get("observation_count", 0)),
                          "hold_reasons": sorted(reasons), "cover": person.get("cover") or {},
                          "jersey": jersey,
                          "exemplars": person.get("exemplars", [])[:4],
                          "feature_indices": [track_features[tid] for tid in tids]})
        if event_path.is_file():
            seen_events = set()
            for row in read_jsonl(event_path):
                event = EventRecord.from_dict(row)
                if event.video_id != meta["video_id"]:
                    raise ValueError(f"Event references a foreign/missing identity: {name}")
                if event.event_id in seen_events:
                    raise ValueError(f"Duplicate event id in {name}")
                seen_events.add(event.event_id)
                events.append({**event.to_dict(), "clip_name": name,
                               "local_person_id": event.identity_id})
    fingerprint = hashlib.sha256(json.dumps({"artifacts": signatures, "clips": names,
                                             "target": manifest.get("target", "full"),
                                             "use_jersey_evidence": options.use_jersey_evidence}, sort_keys=True).encode()).hexdigest()
    return nodes, features, visibility, events, signatures, fingerprint, sorted(set(warnings)), names, contract
