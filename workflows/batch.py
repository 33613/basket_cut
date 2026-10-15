"""Frozen, serial, resumable subprocess runs, independent of the dashboard."""

import hashlib
import json
import re
import shlex
import shutil
import subprocess
from pathlib import Path

from contracts.schema import write_json
from workflows.commands import CommandBuilder
from pipeline.media import sha256_file
from contracts.lineage import clip_id_for

VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm"}


def select_inputs(input_dir: Path, limit: int, selection: Path | None = None) -> list[Path]:
    if limit < 1:
        raise ValueError("limit must be positive")
    root = input_dir.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    if selection:
        value = json.loads(selection.read_text(encoding="utf-8"))
        names = value.get("videos", []) if isinstance(value, dict) else value
        paths = [root / (name["path"] if isinstance(name, dict) else name) for name in names]
    else:
        paths = sorted(p for p in root.rglob("*") if p.suffix.lower() in VIDEO_SUFFIXES and p.is_file())
    resolved = [path.resolve() for path in paths[:limit]]
    if not resolved or len(set(resolved)) != len(resolved):
        raise ValueError("Selection is empty or contains duplicate videos")
    for path in resolved:
        if not path.is_relative_to(root) or not path.is_file() or path.suffix.lower() not in VIDEO_SUFFIXES:
            raise ValueError(f"Missing/unsafe video: {path}")
    return resolved


def stages_for(target: str, jersey: bool = True) -> list[str]:
    stages = ["tracking", "quality"]
    if target == "full":
        stages.extend(["action", "aggregate"])
    stages.extend(["identity", "resolution"])
    if jersey:
        stages.append("jersey")
    stages.append("players")
    if target == "full":
        stages.append("link")
    stages.extend(["render", "review"])
    return stages


def run_batch(settings, input_dir: Path, output_dir: Path, *, limit: int = 30,
              selection: Path | None = None, target: str = "full", options: dict | None = None,
              force: bool = False, dry_run: bool = False, execute=None,
              clips_manifest: Path | None = None) -> dict:
    if target not in {"identity", "full"}:
        raise ValueError("target must be identity or full")
    options = dict(options or {})
    options.setdefault("match_id", output_dir.resolve().name)
    options.setdefault("jersey_qwen", True)
    import math
    if (not math.isfinite(options.get("player_match_distance", .2))
            or options.get("player_match_distance", .2) < 0
            or not math.isfinite(options.get("player_novelty_distance", .5))
            or options.get("player_novelty_distance", .5) <= options.get("player_match_distance", .2)
            or not math.isfinite(options.get("player_min_margin", .05))
            or options.get("player_min_margin", .05) < 0):
        raise ValueError("Invalid player distance/margin parameters")
    lineage_by_input = {}
    if clips_manifest:
        clips_manifest = clips_manifest.resolve()
        if clips_manifest.parent != input_dir.resolve():
            raise ValueError("Input directory must be the clips manifest directory")
        segmented = json.loads(clips_manifest.read_text())
        for row in segmented["videos"]:
            if row["path"] in lineage_by_input or row["clip_id"] in {r["clip_id"] for r in lineage_by_input.values()}:
                raise ValueError("Duplicate exported clip path or ID")
            if not re.fullmatch(r"CL-[0-9a-f]{20}", row["clip_id"]):
                raise ValueError("Invalid segmentation clip ID")
            if (row["source_video_id"] != segmented["source"]["source_video_id"]
                    or row["source_sha256"] != segmented["source"]["sha256"]
                    or row["source_path"] != segmented["source"]["path"]
                    or row["clip_id"] != clip_id_for(row["source_video_id"], row["start_s"], row["end_s"])):
                raise ValueError("Clip source/ID differs from its export provenance")
            lineage_by_input[row["path"]] = row
        selection = selection or clips_manifest
    sources = select_inputs(input_dir, limit, selection)
    output = output_dir.resolve()
    # Outputs must never be mixed into the data tree or overlap source inputs.
    data = input_dir.resolve()
    if output == data or output.is_relative_to(data) or data.is_relative_to(output):
        raise ValueError("Batch output and input trees must not overlap")
    builder = CommandBuilder(settings, validate_resources=not dry_run)
    def command_for(stage, paths, clip_options):
        command = builder.command(stage, paths, clip_options)
        if stage == 'render':
            for flag in ('--actions', '--identity-map'):
                if flag in command:
                    position = command.index(flag)
                    del command[position:position + 2]
            command.extend(['--identity-map', str(paths['identity_map'])])
            if target == 'full':
                command.extend(['--actions', str(paths['linked_actions'])])
        return command
    entries = []
    code_files = {p.relative_to(settings.repository_root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                  for folder in ('cli', 'workflows', 'pipeline', 'contracts', 'adapters', 'analysis')
                  for p in (settings.repository_root / folder).rglob('*.py')}
    for source in sources:
        relative = source.relative_to(data).as_posix()
        slug = "".join(c if c.isalnum() or c in "-_" else "-" for c in source.stem)[:70]
        name = f"{slug}-{hashlib.sha256(relative.encode()).hexdigest()[:8]}"
        lineage = lineage_by_input.get(relative)
        if clips_manifest and lineage is None:
            raise ValueError("Selected clip is absent from the segmentation manifest")
        if lineage:
            if sha256_file(source) != lineage["sha256"]:
                raise ValueError(f"Exported clip checksum mismatch: {source}")
            name = lineage["clip_id"]
        paths = builder.paths({"source_path": str(source), "output_dir": str(output / name)})
        stages = stages_for(target, bool(options["jersey_qwen"]))
        clip_options = {**options, **({"clip_id": name} if lineage else {})}
        commands = {stage: command_for(stage, paths, clip_options) for stage in stages}
        # Separate clip identity from stage configuration so identity-only runs
        # can gain actions later without re-extracting registered KPR evidence.
        identity = {"input": relative, "size": source.stat().st_size,
                    "mtime_ns": source.stat().st_mtime_ns, "options": options, "code": code_files}
        if lineage:
            identity["lineage"] = lineage
        fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        stage_fingerprints = {}
        for stage, command in commands.items():
            resources = {str(p): [p.stat().st_size, p.stat().st_mtime_ns]
                for word in command if (p := Path(word)).is_file() and not p.resolve().is_relative_to(output)}
            stage_fingerprints[stage] = hashlib.sha256(json.dumps(
                [fingerprint, command, resources], sort_keys=True).encode()).hexdigest()
        entries.append({"name": name, "input": relative, "fingerprint": fingerprint,
                        **({"clip_id": name, "lineage": lineage} if lineage else {}),
                        "commands": commands, "stage_fingerprints": stage_fingerprints, "paths": paths})
    plan = {"schema_version": 2, "match_id": options["match_id"], "target": target, "options": options,
            "clips": [{k: e[k] for k in ("name", "input", "fingerprint", "clip_id", "lineage") if k in e} for e in entries]}
    if clips_manifest:
        plan["segmentation"] = {"clips_manifest": str(clips_manifest),
                                "plan_path": str(clips_manifest.parent / "segmentation.json")}
    manifest = output / "batch_manifest.json"
    if manifest.is_file():
        previous_plan = json.loads(manifest.read_text())
        if previous_plan.get("match_id") != options["match_id"]:
            raise ValueError("Output directory belongs to another match")
        prior = {row["name"]: row for row in previous_plan["clips"]}
        for row in plan["clips"]:
            if row["name"] in prior and prior[row["name"]] != row and not force:
                raise ValueError("Registered clip inputs/config changed; use a new run or --force for unchanged evidence")
            prior[row["name"]] = row
        plan["clips"] = list(prior.values())
    if dry_run:
        return {"dry_run": True, "selected_videos": len(entries), "plan": plan,
                "commands": {e["name"]: e["commands"] for e in entries}}
    output.mkdir(parents=True, exist_ok=True)
    write_json(manifest, plan)
    results = []
    for entry in entries:
        paths = entry["paths"]
        paths["output"].mkdir(parents=True, exist_ok=True)
        status_path = paths["output"] / "batch_status.json"
        try:
            previous = json.loads(status_path.read_text()) if status_path.is_file() else {}
        except (OSError, json.JSONDecodeError):
            previous = {}  # A damaged status cannot authorize reuse of outputs.
        status = {"name": entry["name"], "input": entry["input"],
                  "fingerprint": entry["fingerprint"], "status": "running", "stages": {},
                  "stage_fingerprints": entry["stage_fingerprints"]}
        try:
            for stage, command in entry["commands"].items():
                if (not force and previous.get("stages", {}).get(stage) == "completed"
                        and previous.get("stage_fingerprints", {}).get(stage) == entry["stage_fingerprints"][stage]
                        and builder.artifact_exists(stage, paths)):
                    status["stages"][stage] = "completed"
                    continue
                if shutil.disk_usage(output).free < float(options.get('min_free_gb', 2)) * 1024**3:
                    raise RuntimeError('Disk reserve reached; free or expand storage before resuming. Existing artifacts were preserved.')
                if builder.write_empty_model_artifacts(stage, paths):
                    status['stages'][stage] = 'completed'
                    write_json(status_path, status)
                    continue
                status["stages"][stage] = "running"
                write_json(status_path, status)
                print(f"[{entry['name']}] {stage}", flush=True)
                with paths["log"].open("a", encoding="utf-8") as log:
                    log.write(f"\n$ {shlex.join(command)}\n"); log.flush()
                    code = (execute(command, paths) if execute else subprocess.run(
                        command, cwd=settings.repository_root, stdout=log, stderr=subprocess.STDOUT).returncode)
                if code:
                    status["stages"][stage] = "failed"
                    raise RuntimeError(f"{stage} exited {code}; see pipeline.log")
                if not builder.artifact_exists(stage, paths):
                    status["stages"][stage] = "failed"
                    raise RuntimeError(f"{stage} returned success without fresh required artifacts")
                status["stages"][stage] = "completed"
                write_json(status_path, status)
            status["status"] = "completed"
        except Exception as exc:
            status["status"] = "failed"
            status["error"] = str(exc)
            print(f"FAILED {entry['name']}: {exc}", flush=True)
        write_json(status_path, status)
        results.append(status)
        write_json(output / "batch_summary.json", {"expected_clips": len(entries),
                   "completed_clips": sum(r["status"] == "completed" for r in results),
                   "processed_clips": len(results), "clips": results})
    return {"expected_clips": len(entries), "completed_clips": sum(r["status"] == "completed" for r in results),
            "failed_clips": [r for r in results if r["status"] == "failed"],
            "output_dir": str(output), "warning": "Processing success and archive reduction are not accuracy."}
