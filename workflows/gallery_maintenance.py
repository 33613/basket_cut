"""Review/export memory on CPU, with a backup and no mutation of model files."""
import fcntl
import json
from pathlib import Path
import sqlite3

from pipeline.identity.player_registry import PlayerRegistry
from pipeline.media import atomic_json
from workflows.gallery_curation import curate_records
from workflows.person_library import export_library, library_is_current


def maintain(root, *, operation, expected_revision=None, output_dir=None, policy=None, node_id=None):
    root = Path(root).resolve()
    library = root / "library"
    if library.is_symlink() or not (library / "players.sqlite3").is_file():
        raise FileNotFoundError("Existing dedicated player library required")
    manifest = json.loads((library / "library_manifest.json").read_text())
    if not library_is_current(root, manifest):
        raise ValueError("Registered source artifacts changed; verify original evidence before maintaining the library")
    # Cooperate with process_match as well as per-library writers. A paused
    # parent still holds its lock, so do not edit the active match in place.
    match_lock = (root.parent / ".match.lock").open("a")
    try:
        if operation != "preview":
            try: fcntl.flock(match_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("process_match still holds this match lock; stop its process before editing this library") from exc
        with (library / ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX if operation != "preview" else fcntl.LOCK_SH)
            if operation == "preview":
                if output_dir is None:
                    raise ValueError("Preview requires a new output directory")
                output = Path(output_dir).resolve()
                if output.is_relative_to(root) or root.is_relative_to(output):
                    raise ValueError("Preview must be separate from original results")
                output.mkdir(parents=True, exist_ok=False)
                database = output / "players.sqlite3"
                with sqlite3.connect(f"file:{library / 'players.sqlite3'}?mode=ro", uri=True) as source, sqlite3.connect(database) as dest:
                    source.backup(dest)
            else:
                output, database = library, library / "players.sqlite3"
            with PlayerRegistry(database, manifest["match_id"]) as registry:
                revision = registry.meta("revision")
                if operation != "preview" and revision != expected_revision:
                    raise RuntimeError("Player library changed; inspect its latest revision")
                before = registry.snapshot()
                if operation != "preview":
                    backups = library / "backups"
                    backups.mkdir(exist_ok=True)
                    backup = backups / f"{revision or 'empty'}.sqlite3"
                    if not backup.exists():
                        with sqlite3.connect(backup) as dest: registry.db.backup(dest)
                if operation in {"preview", "reprocess"}:
                    evidence = curate_records(root, registry.memory_records())
                    registry.refresh_memory(policy, evidence=evidence, expected_revision=revision)
                else:
                    registry.review(operation, node_id=node_id, expected_revision=revision)
                after = registry.snapshot()
                # Preview is a detached export: all source references still
                # belong to the original result tree; do not import as a match.
                export_library(registry, root, output)
                report = {"operation": operation, "accuracy_evaluated": False,
                    "before": memory_summary(before), "after": memory_summary(after),
                    "production_modified": operation != "preview", "revision": after["revision"],
                    "source_results": str(root), "policy": after["memory_policy"],
                    "warning": "Counts and similarity are not identity precision/recall."}
                atomic_json(output / "memory_report.json", report)
                from analysis.gallery_report import write_report
                write_report(root, after, output)
                return report
    finally:
        match_lock.close()


def memory_summary(snapshot):
    people = snapshot["people"]
    members = [m for p in people for m in p["members"]]
    return {"identity_records": len(people),
        "candidate_players": sum(p["status"] in {"candidate", "manual_grouping"} for p in people),
        "gallery_observations": sum(m["gallery_eligible"] for m in members),
        "gallery_templates": sum(p.get("gallery_template_count", 0) for p in people),
        "memory_states": {s: sum(m.get("memory", {}).get("status") == s for m in members)
                          for s in ("accepted", "pending", "rejected", "duplicate")},
        "events": len(snapshot["events"]), "unmapped_events": sum(e["player_id"] is None for e in snapshot["events"])}
