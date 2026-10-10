"""Incremental clip registration and dashboard-compatible indexes."""
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import shutil
import fcntl

from contracts.schema import write_json, write_jsonl_line
from pipeline.identity.player_registry import PlayerRegistry
from workflows.library_inputs import PersonLibraryOptions, load_library_inputs, safe_relative, file_stamp


def export_library(registry, root, output, *, signatures=None, options=None, clips=None, warnings=None):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output == root or root.is_relative_to(output):
        raise ValueError("Library cannot overwrite source result trees")
    output.mkdir(parents=True, exist_ok=True)
    previous_path = output / "library_manifest.json"
    previous = json.loads(previous_path.read_text()) if previous_path.is_file() else {}
    sources = dict(previous.get("source_artifacts", {}))
    sources.update(signatures or {})
    snapshot = registry.snapshot()
    people, events = snapshot["people"], snapshot["events"]
    names = sorted(set(previous.get("batch_clip_names", [])) | set(clips or []) |
                   {m["clip_name"] for p in people for m in p["members"]} |
                   {e["clip_name"] for e in events})
    event_types = {}
    for event in events:
        event_types[event["event"]] = event_types.get(event["event"], 0) + 1
    manifest = {"schema_version": 2, "scope": "match_incremental", "match_id": snapshot["match_id"],
        "expected_clips": len(names), "batch_clip_names": names, "batch_target": "full",
        "global_person_count": len(people), "local_archive_count": len(snapshot["mappings"]),
        "merged_group_count": sum(p["local_archive_count"] > 1 for p in people),
        "held_archive_count": sum(bool(m["hold_reasons"]) or m["manually_detached"] for p in people for m in p["members"]),
        "event_count": len(events), "unmapped_event_count": sum(e["global_person_id"] is None for e in events),
        "event_types": event_types, "review_revision": snapshot["revision"],
        "input_fingerprint": snapshot["revision"], "feature_contract": snapshot["feature_contract"],
        "index_dir": "indexes/" + (snapshot["revision"] or "empty"),
        "source_artifacts": sources, "player_aliases": snapshot["aliases"], "settings": options or previous.get("settings", {}),
        "jersey_constraints": (options or previous.get("settings", {})).get("use_jersey_evidence", False),
        "warnings": warnings or [], "warning": "Stable match-scoped IDs; candidate identities and action scores are not verified accuracy."}
    indexes = output / "indexes"
    indexes.mkdir(exist_ok=True)
    generation = safe_relative(output, manifest["index_dir"])
    temporary = Path(tempfile.mkdtemp(prefix=".export-", dir=indexes))
    try:
        for filename, rows in (("people.jsonl", people), ("identity_map.jsonl", snapshot["mappings"]),
                               ("events.jsonl", events), ("match_pairs.jsonl", [])):
            with (temporary / filename).open("w", encoding="utf-8") as handle:
                for row in rows:
                    write_jsonl_line(handle, row)
        if not generation.exists():
            temporary.replace(generation)
        # Compatibility files for CLI consumers are derived, atomically replaced.
        for filename in ("people.jsonl", "identity_map.jsonl", "events.jsonl", "match_pairs.jsonl"):
            target = output / filename
            if target.is_symlink():
                raise ValueError("Library index cannot be a symlink")
            temporary_file = output / ("." + filename + ".tmp")
            shutil.copyfile(generation / filename, temporary_file)
            temporary_file.replace(target)
        manifest_temp = output / ".library-manifest.tmp.json"
        write_json(manifest_temp, manifest)
        manifest_temp.replace(previous_path)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    return manifest


def build_person_library(options: PersonLibraryOptions):
    root = options.run_dir.resolve()
    requested = options.output_dir or root / "library"
    if requested.is_symlink() or requested.resolve() != root / "library":
        raise ValueError("Persistent library must use the run's dedicated library directory")
    nodes, features, visible, events, signatures, _, warnings, clips, contract = load_library_inputs(options)
    batch = json.loads((root / "batch_manifest.json").read_text())
    match_id = options.match_id or batch.get("match_id")
    if not match_id:
        raise ValueError("An explicit match_id is required")
    settings = {k: v for k, v in asdict(options).items() if k not in
                {"run_dir", "output_dir", "match_id", "clip_names"}}
    requested.mkdir(parents=True, exist_ok=True)
    if (requested / "library_manifest.json").is_file() and not (requested / "players.sqlite3").is_file():
        raise FileNotFoundError("Player database missing; restore it from backup to preserve stable IDs")
    with (requested / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with PlayerRegistry(requested / "players.sqlite3", match_id) as registry:
            registry.ingest(nodes, features, visible, events, contract,
                max_distance=options.max_distance, novelty_distance=options.novelty_distance,
                min_margin=options.min_margin, min_common_parts=options.min_common_parts,
                gallery_limit=options.gallery_limit)
            return export_library(registry, root, requested, signatures=signatures,
                                  options=settings, clips=clips, warnings=warnings)


def library_is_current(run_dir, manifest):
    root = Path(run_dir).resolve()
    try:
        return all(file_stamp(safe_relative(root, relative)) == stamp
                   for relative, stamp in manifest.get("source_artifacts", {}).items())
    except (OSError, ValueError):
        return False
