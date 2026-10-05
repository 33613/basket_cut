"""Build a match-scoped person/event library from completed clip artifacts."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile

import numpy as np

from contracts.schema import EventRecord, read_jsonl, write_json, write_jsonl_line
from pipeline.identity.cross_clip import group_nodes, part_distances
from pipeline.identity.jersey import group_numbers


@dataclass(frozen=True)
class PersonLibraryOptions:
    run_dir: Path
    output_dir: Path | None = None
    max_distance: float | None = None
    min_samples: int = 2
    max_within_distance: float = 0.4
    min_common_parts: int = 2
    review: Path | None = None
    allow_legacy_features: bool = False
    use_jersey_evidence: bool = False
    overwrite: bool = False


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
    names = [row["name"] for row in manifest["clips"]]
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
        if status_path.is_file() and json.loads(status_path.read_text()).get("status") != "completed":
            raise ValueError(f"Clip is not complete: {name}; finish/retry the batch first")
        meta = json.loads(record(f"{name}/tracking/video_meta.json").read_text())
        raw = list(read_jsonl(record(f"{name}/identity_raw/identities.jsonl")))
        local = list(read_jsonl(record(f"{name}/identity/identities.jsonl")))
        resolution = json.loads(record(f"{name}/identity/resolution_summary.json").read_text())
        if resolution.get("missing_archive_tracks"):
            warnings.append(f"{name}: tracks without KPR archives are not represented")
        event_path = record(f"{name}/action/events.jsonl")
        if manifest.get("target", "full") == "full" and not event_path.is_file():
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
            if jersey_summary.get("backend") != "uncertainty_jnr":
                raise ValueError("Jersey constraints require the uncertainty-jnr baseline, not generic OCR")
            for row in read_jsonl(record(f"{name}/identity/jersey_tracks.jsonl")):
                tid = int(row["raw_track_id"])
                if tid in jersey_tracks or tid not in raw_by_track:
                    raise ValueError("Duplicate or foreign JNR raw track")
                if row.get("number") is not None:
                    number = row["number"]
                    if (not isinstance(number, str) or not number.isascii() or not number.isdigit()
                            or not 1 <= int(number) <= 99 or str(int(number)) != number
                            or row.get("status") != "candidate_consensus" or row.get("verified") is not False):
                        raise ValueError("Invalid or unverifiable JNR number candidate")
                for reading in row.get("readings", []):
                    if reading.get("crop_path"):
                        safe_relative(clip / "identity_raw", reading["crop_path"])
                jersey_tracks[tid] = row
            if set(jersey_tracks) != set(raw_by_track):
                raise ValueError("JNR evidence does not cover the current raw KPR archives")
            # Refuse mixtures of models or different acceptance rules across this match.
            if raw:
                signature = {"provenance": {key: value for key, value in jersey_summary["provenance"].items()
                                           if key not in {"device", "trusted_pickle_loading"}},
                             "thresholds": jersey_summary["thresholds"]}
                if jersey_contract is not None and signature != jersey_contract:
                    raise ValueError("Incompatible JNR models or thresholds across clips")
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
                if not options.allow_legacy_features:
                    raise ValueError(f"{name}: legacy KPR metadata; explicitly use --allow-legacy-features only for known compatible weights")
                signature = {"legacy_checkpoint": Path(summary.get("checkpoint", "")).name,
                             "prompt_mode": summary.get("prompt_mode"),
                             "shape": list(vectors.shape[1:])}
                if not signature["legacy_checkpoint"] or signature["prompt_mode"] is None:
                    raise ValueError("Legacy features lack checkpoint/prompt provenance")
                warnings.append("Legacy feature compatibility is based on filenames, not a checkpoint checksum")
            signature = {"metadata": signature, "shape": list(vectors.shape[1:])}
            if contract is not None and signature != contract:
                raise ValueError("Incompatible KPR checkpoints, prompts or feature layouts across clips")
            contract = signature
            for index, tid in enumerate(track_ids):
                track_features[tid] = len(features)
                features.append(vectors[index])
                visibility.append(weights[index])
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
                jersey["backend"] = "uncertainty_jnr"
                jersey["tracks"] = [jersey_tracks[tid] for tid in tids]
                if jersey["status"] == "conflict":
                    reasons.add("jersey_number_conflict")
            node_id = "N-" + hashlib.sha256(f"{name}\0{pid}".encode()).hexdigest()[:16]
            nodes.append({"node_id": node_id, "clip_name": name, "video_id": meta["video_id"],
                          "filename": Path(meta["input_path"]).name, "local_person_id": pid,
                          "raw_track_ids": tids, "sample_count": int(person.get("sample_count", 0)),
                          "observation_count": int(person.get("observation_count", 0)),
                          "hold_reasons": sorted(reasons), "cover": person.get("cover") or {},
                          "jersey": jersey,
                          "exemplars": person.get("exemplars", [])[:4],
                          "feature_indices": [track_features[tid] for tid in tids]})
        if event_path.is_file():
            seen_events = set()
            for row in read_jsonl(event_path):
                event = EventRecord.from_dict(row)
                if event.video_id != meta["video_id"] or event.identity_id not in seen_person_ids:
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


def build_person_library(options: PersonLibraryOptions, *, review_value=None) -> dict:
    requested_output = (options.output_dir or options.run_dir / "library").expanduser()
    if requested_output.is_symlink():
        raise ValueError("Library output cannot be a symlink")
    output = requested_output.resolve()
    root = options.run_dir.expanduser().resolve()
    if output == root or root.is_relative_to(output):
        raise ValueError("Person library must not overwrite a run or its parent")
    if output.is_relative_to(root) and output != root / "library":
        raise ValueError("Inside a run, only the dedicated library directory is allowed")
    if output.exists() and any(output.iterdir()) and not options.overwrite:
        raise FileExistsError("Library exists; use --overwrite to rebuild this derived library only")
    nodes, features, visible, events, signatures, fingerprint, warnings, clips, contract = load_library_inputs(options)
    review = review_value if review_value is not None else (
        json.loads(options.review.read_text()) if options.review else {})
    if review and review.get("input_fingerprint") != fingerprint:
        raise ValueError("Library review is stale for these clip artifacts")
    assignments = review.get("assignments", {})
    blocked = review.get("blocked_node_ids", [])
    if features:
        matrix = part_distances(np.stack(features), np.stack(visible), min_common_parts=options.min_common_parts)
    else:
        matrix = np.empty((0, 0), dtype=np.float32)
    groups, pairs, merges = group_nodes(nodes, matrix, max_distance=options.max_distance,
                                       assignments=assignments, blocked_node_ids=blocked)
    people, mappings, node_people = [], [], {}
    for group in groups:
        members = sorted(group["members"], key=lambda node: node["node_id"])
        gid = "G-" + hashlib.sha256("|".join(n["node_id"] for n in members).encode()).hexdigest()[:12]
        covers = sorted(members, key=lambda n: (-float(n["cover"].get("archive_quality_score", 0)), n["node_id"]))
        clean_members = [{key: value for key, value in node.items() if key != "feature_indices"} for node in members]
        for member in clean_members:
            member["manually_detached"] = member["node_id"] in blocked
            node_people[(member["clip_name"], member["local_person_id"])] = gid
            mappings.append({"global_person_id": gid, "node_id": member["node_id"],
                             "clip_name": member["clip_name"], "video_id": member["video_id"],
                             "local_person_id": member["local_person_id"], "raw_track_ids": member["raw_track_ids"]})
        global_jersey = None
        if options.use_jersey_evidence:
            global_jersey = group_numbers(list(range(len(members))),
                {i: n.get("jersey") or {"number": None, "status": "unreadable"}
                 for i, n in enumerate(members)})
            global_jersey.pop("raw_track_ids")
            global_jersey["node_ids"] = [n["node_id"] for n in members]
        people.append({"global_person_id": gid, "identity_label": group["identity_label"],
                       "jersey": global_jersey,
                       "status": group["status"], "manual_member_count": group["manual_member_count"],
                       "clip_count": len({n["clip_name"] for n in members}),
                       "local_archive_count": len(members), "members": clean_members,
                       "cover_node_id": covers[0]["node_id"], "event_count": 0, "event_types": {}})
    global_events = []
    event_counts, event_types = Counter(), {}
    for event in events:
        gid = node_people[(event["clip_name"], event["local_person_id"])]
        event_counts[gid] += 1
        event_types.setdefault(gid, Counter())[event["event"]] += 1
        global_events.append({**event, "id": gid, "global_person_id": gid,
                              "source_event_id": event["event_id"],
                              "event_id": "GE-" + hashlib.sha256(f'{event["clip_name"]}\0{event["event_id"]}'.encode()).hexdigest()[:20]})
    for person in people:
        gid = person["global_person_id"]
        person["event_count"] = event_counts[gid]
        person["event_types"] = dict(event_types.get(gid, {}))
    summary = {"schema_version": 1, "scope": "match_candidate", "expected_clips": len(clips),
               "local_archive_count": len(nodes), "global_person_count": len(people),
               "merged_group_count": sum(p["local_archive_count"] > 1 for p in people),
               "held_archive_count": sum(bool(n["hold_reasons"]) or n["node_id"] in blocked for n in nodes),
               "event_count": len(global_events), "event_types": dict(Counter(e["event"] for e in events)),
               "input_fingerprint": fingerprint, "feature_contract": contract,
               "review_revision": review.get("revision"),
               "batch_clip_names": clips,
               "batch_target": json.loads((root / "batch_manifest.json").read_text()).get("target", "full"),
               "settings": {k: v for k, v in asdict(options).items() if k not in {"run_dir", "output_dir", "review", "overwrite"}},
               "jersey_constraints": options.use_jersey_evidence,
               "candidate_number_archives": sum(bool(n.get("jersey", {}).get("number")) for n in nodes) if options.use_jersey_evidence else 0,
               "jersey_blocked_pairs": sum("different_reliable_jersey_numbers" in p["auto_block_reasons"] for p in pairs),
               "jersey_pair_count_scope": "bounded nearest review candidates; not all pair comparisons",
               "warnings": warnings, "merges": merges,
               "warning": "Candidate identities, not verified player names or accuracy. Distances and event raw_scores are uncalibrated. Event times are relative to each source clip; local artifacts are unchanged."}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".person-library-", dir=output.parent))
    backup = temporary.with_name(temporary.name + "-previous")
    try:
        for filename, rows in (("people.jsonl", people), ("identity_map.jsonl", mappings),
                               ("events.jsonl", global_events), ("match_pairs.jsonl", pairs)):
            with (temporary / filename).open("w", encoding="utf-8") as handle:
                for row in rows:
                    write_jsonl_line(handle, row)
        write_json(temporary / "library_manifest.json", {**summary, "source_artifacts": signatures})
        if output.exists():
            output.rename(backup)
        try:
            temporary.rename(output)
        except BaseException:
            if backup.exists():
                backup.rename(output)
            raise
        if backup.exists():
            shutil.rmtree(backup)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return summary


def library_is_current(run_dir: Path, manifest: dict) -> bool:
    artifacts = manifest.get("source_artifacts", {})
    current = json.loads((run_dir / "batch_manifest.json").read_text())
    for clip in current["clips"]:
        status = safe_relative(run_dir, f'{clip["name"]}/batch_status.json')
        if status.is_file() and json.loads(status.read_text()).get("status") != "completed":
            return False
    return (bool(artifacts) and [clip["name"] for clip in current["clips"]] == manifest.get("batch_clip_names")
            and current.get("target", "full") == manifest.get("batch_target")
            and all(file_stamp(safe_relative(run_dir, name)) == stamp for name, stamp in artifacts.items()))
