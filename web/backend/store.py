"""Persistent project manifests used by the Web dashboard."""

from __future__ import annotations

import json
import re
import threading
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

STAGE_NAMES = (
    "tracking",
    "quality",
    "identity",
    "resolution",
    "action",
    "link",
    "aggregate",
    "render",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_slug(value: str, fallback: str = "item") -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._")
    return normalized[:80] or fallback


class ProjectStore:
    def __init__(self, data_root: Path, output_root: Path) -> None:
        self.data_root = data_root
        self.output_root = output_root
        self._lock = threading.RLock()

    def _manifest_path(self, project_id: str) -> Path:
        return self.output_root / project_id / "project.json"

    def _write(self, manifest: dict[str, Any]) -> None:
        destination = self._manifest_path(manifest["project_id"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)

    def create_project(self, name: str | None = None) -> dict[str, Any]:
        with self._lock:
            project_id = (
                datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
                + "-"
                + uuid.uuid4().hex[:6]
            )
            now = utc_now()
            manifest = {
                "schema_version": 1,
                "project_id": project_id,
                "name": (name or f"Basketball run {project_id[:15]}").strip()
                or f"Basketball run {project_id[:15]}",
                "created_at": now,
                "updated_at": now,
                "videos": [],
            }
            (self.data_root / project_id / "inputs").mkdir(parents=True, exist_ok=True)
            self._write(manifest)
            return manifest

    def list_projects(self) -> list[dict[str, Any]]:
        projects = []
        if not self.output_root.exists():
            return projects
        for path in self.output_root.glob("*/project.json"):
            try:
                projects.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue
        projects.sort(key=lambda item: item.get("updated_at", ""), reverse=True)
        return projects

    def register_import(self, manifest: dict[str, Any]) -> None:
        """Only store a Web index; referenced source and result files stay put."""
        with self._lock:
            path = self._manifest_path(manifest["project_id"])
            if path.is_file():
                existing = self.get_project(manifest["project_id"])
                if not existing.get("read_only"):
                    raise ValueError("Cannot replace a writable project with an import")
                manifest["created_at"] = existing["created_at"]
            self._write(manifest)

    def get_project(self, project_id: str) -> dict[str, Any]:
        path = self._manifest_path(safe_slug(project_id))
        if not path.is_file():
            raise FileNotFoundError(f"Project not found: {project_id}")
        return json.loads(path.read_text(encoding="utf-8"))

    def add_video(
        self,
        project_id: str,
        *,
        original_filename: str,
        source_path: Path,
        size_bytes: int,
    ) -> dict[str, Any]:
        with self._lock:
            manifest = self.get_project(project_id)
            stem = safe_slug(Path(original_filename).stem, "video")
            video_id = f"{stem}-{uuid.uuid4().hex[:7]}"
            output_dir = self.output_root / safe_slug(project_id) / "videos" / video_id
            output_dir.mkdir(parents=True, exist_ok=True)
            video = {
                "video_id": video_id,
                "filename": original_filename,
                "source_path": str(source_path.resolve()),
                "output_dir": str(output_dir.resolve()),
                "size_bytes": size_bytes,
                "created_at": utc_now(),
                "updated_at": utc_now(),
                "status": "uploaded",
                "active_stage": None,
                "error": None,
                "run_options": {},
                "stages": {
                    stage: {
                        "status": "pending",
                        "started_at": None,
                        "finished_at": None,
                        "command": None,
                        "return_code": None,
                    }
                    for stage in STAGE_NAMES
                },
            }
            manifest["videos"].append(video)
            manifest["updated_at"] = utc_now()
            self._write(manifest)
            return video

    def get_video(self, project_id: str, video_id: str) -> dict[str, Any]:
        manifest = self.get_project(project_id)
        for video in manifest["videos"]:
            if video["video_id"] == video_id:
                return video
        raise FileNotFoundError(f"Video not found: {project_id}/{video_id}")

    def update_video(
        self,
        project_id: str,
        video_id: str,
        update: Callable[[dict[str, Any]], None],
    ) -> dict[str, Any]:
        with self._lock:
            manifest = self.get_project(project_id)
            for video in manifest["videos"]:
                if video["video_id"] == video_id:
                    update(video)
                    video["updated_at"] = utc_now()
                    manifest["updated_at"] = video["updated_at"]
                    self._write(manifest)
                    return video
            raise FileNotFoundError(f"Video not found: {project_id}/{video_id}")

    def recover_interrupted_jobs(self) -> None:
        """Mark in-memory jobs left behind by a Web process restart."""
        for project in self.list_projects():
            project_id = project["project_id"]
            for item in project.get("videos", []):
                if item.get("status") not in {"queued", "running"}:
                    continue

                def mark_interrupted(video: dict[str, Any]) -> None:
                    active = video.get("active_stage")
                    if active and active in video.get("stages", {}):
                        video["stages"][active]["status"] = "failed"
                        video["stages"][active]["finished_at"] = utc_now()
                        video["stages"][active]["return_code"] = -1
                    video["status"] = "failed"
                    video["active_stage"] = None
                    video["error"] = "Web process restarted while this job was active"

                self.update_video(project_id, item["video_id"], mark_interrupted)

    def set_stage(
        self,
        project_id: str,
        video_id: str,
        stage: str,
        status: str,
        **values: Any,
    ) -> dict[str, Any]:
        if stage not in STAGE_NAMES:
            raise ValueError(f"Unknown stage: {stage}")

        def apply(video: dict[str, Any]) -> None:
            state = video["stages"].setdefault(
                stage,
                {
                    "status": "pending",
                    "started_at": None,
                    "finished_at": None,
                    "command": None,
                    "return_code": None,
                },
            )
            state["status"] = status
            state.update(values)
            if status == "running":
                state["started_at"] = utc_now()
                state["finished_at"] = None
                video["active_stage"] = stage
                video["status"] = "running"
                video["error"] = None
            elif status in {"completed", "failed", "skipped"}:
                state["finished_at"] = utc_now()
                if video.get("active_stage") == stage:
                    video["active_stage"] = None

        return self.update_video(project_id, video_id, apply)
