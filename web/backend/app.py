"""FastAPI application for the basketball pipeline research dashboard."""

from __future__ import annotations

import mimetypes
import re
import uuid
from pathlib import Path
from typing import Any, Literal, Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from web.backend.artifacts import (
    collect_video_artifacts,
    project_people_index,
    read_jsonl,
)
from web.backend.runner import PipelineRunner
from web.backend.settings import WebSettings
from web.backend.store import ProjectStore, safe_slug

VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}
settings = WebSettings()
settings.ensure_directories()
store = ProjectStore(settings.data_root, settings.output_root)
store.recover_interrupted_jobs()
runner = PipelineRunner(settings, store)

app = FastAPI(
    title="CourtVision Lab",
    description="Visual inspection dashboard for tracking, identity and actions",
    version="0.1.0",
)
app.mount(
    "/assets",
    StaticFiles(directory=settings.frontend_root),
    name="assets",
)


class ProjectCreate(BaseModel):
    name: Optional[str] = Field(default=None, max_length=120)


class RunOptions(BaseModel):
    target: Literal["tracking", "identity", "action", "final", "full"] = "full"
    force: bool = False
    max_frames: Optional[int] = Field(default=None, ge=1, le=100000)
    tracking_det_threshold: float = Field(default=0.3, ge=0, le=1)
    identity_samples: int = Field(default=8, ge=1, le=64)
    identity_min_det_score: float = Field(default=0.5, ge=0, le=1)
    prompt_mode: Literal["none"] = "none"
    action_threshold: float = Field(default=0.2, ge=0, le=1)
    action_min_det_score: float = Field(default=0.3, ge=0, le=1)


def not_found(exc: FileNotFoundError) -> HTTPException:
    return HTTPException(status_code=404, detail=str(exc))


def get_video(project_id: str, video_id: str) -> dict[str, Any]:
    try:
        return store.get_video(project_id, video_id)
    except FileNotFoundError as exc:
        raise not_found(exc) from exc


def safe_child(base: Path, relative: str) -> Path:
    base = base.resolve()
    candidate = (base / relative).resolve()
    try:
        candidate.relative_to(base)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid artifact path") from exc
    if not candidate.is_file():
        raise HTTPException(status_code=404, detail="Artifact not found")
    return candidate


def artifact_url(project_id: str, video_id: str, relative: str) -> str:
    return (
        f"/api/projects/{project_id}/videos/{video_id}/artifacts/"
        f"{relative}"
    )


def decorate_artifacts(
    project_id: str, video: dict[str, Any], artifacts: dict[str, Any]
) -> dict[str, Any]:
    video_id = video["video_id"]
    available = artifacts["available"]
    media = {
        "source": f"/api/projects/{project_id}/videos/{video_id}/source",
        "tracking": (
            artifact_url(project_id, video_id, "tracking/tracks_vis.mp4")
            if available["tracks_video"]
            else None
        ),
        "final": None,
    }
    output = Path(video["output_dir"])
    if (output / "visualization/result_web.mp4").is_file():
        media["final"] = artifact_url(
            project_id, video_id, "visualization/result_web.mp4"
        )
    elif (output / "visualization/result.mp4").is_file():
        media["final"] = artifact_url(
            project_id, video_id, "visualization/result.mp4"
        )

    def decorate_identity(identity: dict[str, Any] | None) -> None:
        if not identity:
            return
        cover = identity.get("cover") or {}
        crop = cover.get("crop_path")
        context = cover.get("context_path")
        if crop:
            cover["crop_url"] = artifact_url(
                project_id, video_id, f"identity/{crop}"
            )
        if context:
            cover["context_url"] = artifact_url(
                project_id, video_id, f"identity/{context}"
            )
        for exemplar in identity.get("exemplars") or []:
            path = exemplar.get("crop_path")
            if path:
                exemplar["crop_url"] = artifact_url(
                    project_id, video_id, f"identity/{path}"
                )

    for identity in artifacts["identity"]["people"]:
        decorate_identity(identity)
    for person in artifacts["action"]["index"]["people"]:
        decorate_identity(person.get("identity"))
    artifacts["media"] = media
    return artifacts


@app.on_event("shutdown")
def shutdown_runner() -> None:
    runner.shutdown()


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(settings.frontend_root / "index.html")


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "configuration": settings.readiness()}


@app.get("/api/configuration")
def configuration() -> dict[str, Any]:
    return {
        "readiness": settings.readiness(),
        "settings": settings.public_dict(),
        "max_upload_bytes": settings.max_upload_bytes,
    }


@app.post("/api/projects")
def create_project(request: ProjectCreate) -> dict[str, Any]:
    return store.create_project(request.name)


@app.get("/api/projects")
def list_projects() -> list[dict[str, Any]]:
    return store.list_projects()


@app.get("/api/projects/{project_id}")
def project_detail(project_id: str) -> dict[str, Any]:
    try:
        manifest = store.get_project(project_id)
    except FileNotFoundError as exc:
        raise not_found(exc) from exc
    result = dict(manifest)
    result["people_index"] = project_people_index(manifest)
    return result


@app.post("/api/projects/{project_id}/videos")
async def upload_videos(
    project_id: str,
    files: list[UploadFile] = File(...),
) -> dict[str, Any]:
    try:
        store.get_project(project_id)
    except FileNotFoundError as exc:
        raise not_found(exc) from exc
    if not files:
        raise HTTPException(status_code=400, detail="No video files supplied")

    input_dir = settings.data_root / safe_slug(project_id) / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    uploaded = []
    for upload in files:
        original_name = Path(upload.filename or "video.mp4").name
        extension = Path(original_name).suffix.lower()
        if extension not in VIDEO_EXTENSIONS:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported video extension: {extension or '(none)'}",
            )
        stored_name = (
            safe_slug(Path(original_name).stem, "video")
            + "-"
            + uuid.uuid4().hex[:8]
            + extension
        )
        destination = input_dir / stored_name
        size = 0
        try:
            with destination.open("wb") as handle:
                while chunk := await upload.read(8 * 1024 * 1024):
                    size += len(chunk)
                    if size > settings.max_upload_bytes:
                        raise HTTPException(
                            status_code=413,
                            detail=f"{original_name} exceeds upload size limit",
                        )
                    handle.write(chunk)
        except Exception:
            destination.unlink(missing_ok=True)
            raise
        finally:
            await upload.close()
        uploaded.append(
            store.add_video(
                project_id,
                original_filename=original_name,
                source_path=destination,
                size_bytes=size,
            )
        )
    return {"project_id": project_id, "videos": uploaded}


@app.get("/api/projects/{project_id}/videos/{video_id}")
def video_detail(project_id: str, video_id: str) -> dict[str, Any]:
    video = get_video(project_id, video_id)
    artifacts = decorate_artifacts(
        project_id, video, collect_video_artifacts(video)
    )
    return {
        "video": video,
        "artifacts": artifacts,
        "job_active": runner.is_active(project_id, video_id),
    }


@app.post("/api/projects/{project_id}/videos/{video_id}/run")
def run_video(
    project_id: str,
    video_id: str,
    request: RunOptions,
) -> dict[str, Any]:
    get_video(project_id, video_id)
    payload = request.model_dump() if hasattr(request, "model_dump") else request.dict()
    target = payload.pop("target")
    force = bool(payload.pop("force"))
    try:
        video = runner.submit(
            project_id,
            video_id,
            target=target,
            options=payload,
            force=force,
        )
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"queued": True, "video": video}


@app.get("/api/projects/{project_id}/videos/{video_id}/log")
def video_log(
    project_id: str,
    video_id: str,
    max_bytes: int = Query(default=120000, ge=1000, le=2_000_000),
) -> PlainTextResponse:
    video = get_video(project_id, video_id)
    log_path = Path(video["output_dir"]) / "pipeline.log"
    if not log_path.is_file():
        return PlainTextResponse("No pipeline log yet.\n")
    with log_path.open("rb") as handle:
        handle.seek(0, 2)
        size = handle.tell()
        handle.seek(max(0, size - max_bytes))
        content = handle.read().decode("utf-8", errors="replace")
    content = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", content)
    return PlainTextResponse(content)


@app.get("/api/projects/{project_id}/videos/{video_id}/source")
def source_video(project_id: str, video_id: str) -> FileResponse:
    video = get_video(project_id, video_id)
    path = Path(video["source_path"])
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Source video not found")
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return FileResponse(path, media_type=media_type, filename=None)


@app.get("/api/projects/{project_id}/videos/{video_id}/artifacts/{relative:path}")
def artifact_file(project_id: str, video_id: str, relative: str) -> FileResponse:
    video = get_video(project_id, video_id)
    path = safe_child(Path(video["output_dir"]), relative)
    return FileResponse(path, filename=None)


@app.get("/api/projects/{project_id}/videos/{video_id}/records/{kind}")
def artifact_records(
    project_id: str,
    video_id: str,
    kind: Literal["tracks", "identities", "actions", "linked_actions", "pairs"],
    limit: int = Query(default=200, ge=1, le=2000),
) -> dict[str, Any]:
    video = get_video(project_id, video_id)
    output = Path(video["output_dir"])
    mapping = {
        "tracks": output / "tracking/tracks.jsonl",
        "identities": output / "identity/identities.jsonl",
        "actions": output / "action/actions.jsonl",
        "linked_actions": output / "action/actions_with_identity.jsonl",
        "pairs": output / "identity/kpr_track_pairs.jsonl",
    }
    path = mapping[kind]
    return {
        "kind": kind,
        "path": str(path),
        "records": read_jsonl(path, limit=limit),
        "limit": limit,
    }
