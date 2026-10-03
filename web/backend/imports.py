"""Register existing CLI results without copying or mutating their artifacts."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from web.backend.artifacts import load_json
from web.backend.settings import WebSettings
from web.backend.store import STAGE_NAMES, ProjectStore, safe_slug, utc_now


def within_root(path: Path, root: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"Path must be inside {root}: {resolved}")
    return resolved


def import_results(
    settings: WebSettings,
    store: ProjectStore,
    result_dir: Path,
    name: str | None = None,
) -> dict[str, Any]:
    """Validate all paths before committing one deterministic project manifest."""
    root = within_root(result_dir, settings.import_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Result directory not found: {root}")
    directories = (
        [root]
        if (root / "tracking/video_meta.json").is_file()
        else sorted(path for path in root.iterdir() if path.is_dir())
    )
    now = utc_now()
    project_id = "import-" + hashlib.sha256(str(root).encode()).hexdigest()[:16]
    videos = []
    for output in directories:
        output = within_root(output, root)
        meta_path = output / "tracking/video_meta.json"
        if not meta_path.is_file():
            continue
        # Refuse symlinks to artifacts outside this imported directory, including
        # image symlinks. Routes repeat containment checks when serving files.
        for path in output.rglob("*"):
            within_root(path, output)
        meta = load_json(meta_path)
        if not meta or not meta.get("input_path"):
            raise ValueError(f"Missing input_path in {meta_path}")
        source = within_root(Path(meta["input_path"]), settings.source_root)
        video_id = (
            safe_slug(output.name)
            + "-"
            + hashlib.sha256(str(output).encode()).hexdigest()[:8]
        )
        cache = settings.output_root / project_id / "media" / video_id
        expected = {
            "tracking": output / "tracking/tracks.jsonl",
            "identity": output / "identity/identity_archive_manifest.json",
            "action": output / "action/actions.jsonl",
            "link": output / "action/actions_with_identity.jsonl",
            "aggregate": output / "action/events.jsonl",
            "render": next(
                (
                    output / p
                    for p in (
                        "visualization/result_web.mp4",
                        "visualization/result.mp4",
                        "result.mp4",
                    )
                    if (output / p).is_file()
                ),
                output / "result.mp4",
            ),
        }
        stages = {
            stage: {
                "status": "completed" if expected[stage].is_file() else "pending",
                "started_at": None,
                "finished_at": None,
                "command": None,
                "return_code": None,
            }
            for stage in STAGE_NAMES
        }
        complete = all(value["status"] == "completed" for value in stages.values())
        videos.append(
            {
                "video_id": video_id,
                "filename": source.name,
                "source_path": str(source),
                "output_dir": str(output),
                "media_cache_dir": str(cache),
                "size_bytes": source.stat().st_size if source.is_file() else 0,
                "created_at": now,
                "updated_at": now,
                "status": "completed" if complete else "partial",
                "active_stage": None,
                "error": None,
                "run_options": {},
                "stages": stages,
                "read_only": True,
            }
        )
    if not videos:
        raise ValueError(
            "No tracking/video_meta.json found. Select a run directory or one clip directory."
        )
    manifest = {
        "schema_version": 1,
        "project_id": project_id,
        "name": name or root.name,
        "created_at": now,
        "updated_at": now,
        "read_only": True,
        "imported_from": str(root),
        "videos": videos,
    }
    store.register_import(manifest)
    return manifest


def prepare_browser_media(
    settings: WebSettings, manifest: dict[str, Any]
) -> list[dict[str, Any]]:
    """CPU-only H.264 previews in a separate cache, never in original outputs."""
    executable = shutil.which(settings.ffmpeg)
    if executable is None:
        raise FileNotFoundError("ffmpeg not found; install it or set BASKET_FFMPEG")
    prepared = []
    for video in manifest["videos"]:
        output = Path(video["output_dir"])
        cache = within_root(Path(video["media_cache_dir"]), settings.output_root)
        for kind, original in {
            "source": Path(video["source_path"]),
            "tracking": output / "tracking/tracks_vis.mp4",
            "final": next(
                (
                    output / p
                    for p in (
                        "visualization/result_web.mp4",
                        "visualization/result.mp4",
                        "result.mp4",
                    )
                    if (output / p).is_file()
                ),
                output / "result.mp4",
            ),
        }.items():
            if not original.is_file():
                continue
            original = within_root(
                original, settings.source_root if kind == "source" else output
            )
            cache.mkdir(parents=True, exist_ok=True)
            target = cache / f"{kind}.mp4"
            stamp_path = cache / f"{kind}.json"
            stamp = {
                "source": str(original),
                "size": original.stat().st_size,
                "mtime_ns": original.stat().st_mtime_ns,
            }
            if target.is_file() and load_json(stamp_path) == stamp:
                prepared.append(
                    {"video": video["filename"], "kind": kind, "cached": True}
                )
                continue
            temporary = cache / f"{kind}.partial.mp4"
            result = subprocess.run(
                [
                    executable,
                    "-y",
                    "-i",
                    str(original),
                    "-map",
                    "0:v:0",
                    "-an",
                    "-c:v",
                    "libx264",
                    "-threads",
                    "2",
                    "-preset",
                    "veryfast",
                    "-crf",
                    "22",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(temporary),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode:
                temporary.unlink(missing_ok=True)
                raise RuntimeError(
                    f"Video conversion failed for {original}: {result.stderr[-2000:]}"
                )
            temporary.replace(target)
            stamp_path.write_text(json.dumps(stamp), encoding="utf-8")
            prepared.append({"video": video["filename"], "kind": kind, "cached": False})
    return prepared
