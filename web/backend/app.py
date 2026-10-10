"""FastAPI application for the basketball pipeline research dashboard."""

from __future__ import annotations

import mimetypes
import re
import uuid
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from web.backend.artifacts import (
    collect_video_artifacts,
    final_video_path,
    project_people_index,
    read_jsonl,
)
from web.backend.imports import import_results, within_root
from web.backend.inspection import build_inspection_bundle
from web.backend.runner import PipelineRunner
from web.backend.evidence import EvidenceJobs, evidence_root
from web.backend.reviews import collect_review, save_review, project_reviews, review_bundle
from analysis.evaluation.track_review import merge_review
from contracts.tracks import load_tracks
from web.backend.reviews import review_paths
from web.backend.artifacts import load_json
from web.backend.settings import WebSettings
from web.backend.store import ProjectStore, safe_slug
from web.backend.catalog import catalog, material_catalog, next_review_video, review_page, video_catalog
from web.backend.person_library import library_catalog, library_person, library_events, edit_library, clip_player_assignments
from web.backend.workflow import evidence_page

VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}
settings = WebSettings()
settings.ensure_directories()
store = ProjectStore(settings.data_root, settings.output_root)
store.recover_interrupted_jobs()
runner = PipelineRunner(settings, store)
evidence_jobs = EvidenceJobs(settings, store)

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
    name: str | None = Field(default=None, max_length=120)


class ResultsImport(BaseModel):
    result_dir: str = Field(min_length=1, max_length=2048)
    name: str | None = Field(default=None, max_length=120)


class RunOptions(BaseModel):
    target: Literal["tracking", "identity", "action", "final", "full"] = "full"
    force: bool = False
    max_frames: int | None = Field(default=None, ge=1, le=100000)
    tracking_det_threshold: float = Field(default=0.3, ge=0, le=1)
    identity_samples: int = Field(default=8, ge=1, le=64)
    identity_min_det_score: float = Field(default=0.5, ge=0, le=1)
    player_match_distance: float = Field(default=.2, ge=0, allow_inf_nan=False)
    player_novelty_distance: float = Field(default=.5, gt=0, allow_inf_nan=False)
    player_min_margin: float = Field(default=.05, ge=0, allow_inf_nan=False)
    quality_min_observations: int = Field(default=3, ge=1)
    quality_min_observed_seconds: float = Field(default=0.1, ge=0, allow_inf_nan=False)
    prompt_mode: Literal["none"] = "none"
    action_threshold: float = Field(default=0.2, ge=0, le=1)
    action_min_det_score: float = Field(default=0.3, ge=0, le=1)
    jersey_qwen: bool = True


class ReviewSave(BaseModel):
    review: dict[str, Any]
    expected_revision: str | None = None


class LibraryEdit(BaseModel):
    operation: Literal["label_group", "confirm", "exclude", "detach", "restore"]
    expected_revision: str | None = None
    person_id: str | None = Field(default=None, max_length=80)
    node_id: str | None = Field(default=None, max_length=80)
    label: str | None = Field(default=None, max_length=120)


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
    return f"/api/projects/{project_id}/videos/{video_id}/artifacts/{relative}"


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
    final = final_video_path(output)
    if final:
        media["final"] = artifact_url(
            project_id, video_id, final.relative_to(output).as_posix()
        )
    cache = Path(video["media_cache_dir"]) if video.get("media_cache_dir") else None
    if cache:
        for kind in media:
            if (cache / f"{kind}.mp4").is_file():
                media[kind] = (
                    f"/api/projects/{project_id}/videos/{video_id}/media/{kind}"
                )
    if not available["source"]:
        media["source"] = None

    def decorate_identity(identity: dict[str, Any] | None, prefix: str = 'identity') -> None:
        if not identity:
            return
        cover = identity.get("cover") or {}
        crop = cover.get("crop_path")
        context = cover.get("context_path")
        if crop:
            cover["crop_url"] = artifact_url(project_id, video_id, f"{prefix}/{crop}")
        if context:
            cover["context_url"] = artifact_url(
                project_id, video_id, f"{prefix}/{context}"
            )
        for exemplar in identity.get("exemplars") or []:
            path = exemplar.get("crop_path")
            if path:
                exemplar["crop_url"] = artifact_url(
                    project_id, video_id, f"{prefix}/{path}"
                )
        for source in identity.get("source_tracks") or []:
            decorate_identity(source, prefix)
        for candidate in (identity.get('jersey') or {}).get('candidates', []):
            for evidence in candidate.get('evidence', []):
                path = evidence.get('crop_path')
                if path:
                    evidence['crop_url'] = artifact_url(project_id, video_id,
                        f"identity/{path}")

        for track in (identity.get('jersey') or {}).get('tracks', []):
            for reading in track.get('readings', []):
                path = reading.get('crop_path')
                if path:
                    reading['crop_url'] = artifact_url(project_id, video_id,
                        f"identity/{path}")
        for reading in (identity.get('jersey') or {}).get('readings', []):
            path = reading.get('crop_path')
            if path:
                reading['crop_url'] = artifact_url(project_id, video_id,
                    f"identity/{path}")

    for identity in artifacts["identity"]["people"]:
        decorate_identity(identity)
    for identity in artifacts['identity']['raw_people']:
        decorate_identity(identity, artifacts['identity']['raw_media_prefix'])
    for key in ("index", "points"):
        for person in artifacts["action"][key]["people"]:
            decorate_identity(person.get("identity"))
    artifacts["media"] = media
    return artifacts


@app.on_event("shutdown")
def shutdown_runner() -> None:
    runner.shutdown()
    evidence_jobs.executor.shutdown(wait=False)


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


@app.post("/api/import-results")
def import_existing_results(request: ResultsImport) -> dict[str, Any]:
    try:
        return import_results(settings, store, Path(request.result_dir), request.name)
    except FileNotFoundError as exc:
        raise not_found(exc) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/projects/{project_id}")
def project_detail(project_id: str, include_people: bool = True) -> dict[str, Any]:
    try:
        manifest = store.get_project(project_id)
    except FileNotFoundError as exc:
        raise not_found(exc) from exc
    result = dict(manifest)
    if include_people:
        result["people_index"] = project_people_index(manifest)
    return result


@app.get('/api/projects/{project_id}/review-queue')
def get_review_queue(project_id: str, offset: int = Query(default=0, ge=0),
                     limit: int = Query(default=12, ge=1, le=50),
                     status: Literal['all', 'remaining', 'complete', 'risk', 'uncertain', 'failed'] = 'all',
                     q: str = Query(default='', max_length=120)) -> dict:
    try:
        return video_catalog(store, store.get_project(project_id), offset=offset, limit=limit, status=status, q=q)
    except FileNotFoundError as exc:
        raise not_found(exc) from exc


@app.get('/api/projects/{project_id}/next-review')
def get_next_review(project_id: str, after: str | None = None,
                    status: Literal['remaining', 'risk', 'uncertain'] = 'remaining',
                    q: str = Query(default='', max_length=120)) -> dict:
    try:
        return next_review_video(store, store.get_project(project_id), after, q=q, status=status)
    except FileNotFoundError as exc:
        raise not_found(exc) from exc


@app.get('/api/projects/{project_id}/materials')
def get_materials(project_id: str, offset: int = Query(default=0, ge=0),
                  limit: int = Query(default=18, ge=1, le=50), view: Literal['raw', 'resolved'] = 'resolved',
                  q: str = Query(default='', max_length=120)) -> dict:
    try:
        return material_catalog(store.get_project(project_id), offset=offset, limit=limit, view=view, q=q)
    except FileNotFoundError as exc:
        raise not_found(exc) from exc


@app.get("/api/projects/{project_id}/person-library")
def get_person_library(project_id: str, offset: int = Query(default=0, ge=0),
                       limit: int = Query(default=12, ge=1, le=50),
                       q: str = Query(default="", max_length=120),
                       status: Literal["all", "candidate", "merged", "singleton", "needs_review", "manual_grouping", "non_player"] = "all"):
    try:
        return library_catalog(settings, store, project_id, offset=offset, limit=limit, q=q, status=status)
    except FileNotFoundError as exc:
        raise not_found(exc) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/api/projects/{project_id}/person-library/events")
def get_library_events(project_id: str, person_id: str = Query(default="", max_length=80),
                       event: str = Query(default="", max_length=120), q: str = Query(default="", max_length=120),
                       min_score: float = Query(default=0, ge=0, le=1, allow_inf_nan=False),
                       offset: int = Query(default=0, ge=0), limit: int = Query(default=20, ge=1, le=100)):
    try:
        return library_events(settings, store, project_id, person_id=person_id, event=event,
                              q=q, min_score=min_score, offset=offset, limit=limit)
    except FileNotFoundError as exc:
        raise not_found(exc) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/api/projects/{project_id}/person-library/people/{person_id}")
def get_library_person(project_id: str, person_id: str):
    try:
        return library_person(settings, store, project_id, person_id)
    except FileNotFoundError as exc:
        raise not_found(exc) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/projects/{project_id}/person-library/review")
def review_person_library(project_id: str, value: LibraryEdit):
    try:
        return edit_library(settings, store, project_id, **value.model_dump())
    except FileNotFoundError as exc:
        raise not_found(exc) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/projects/{project_id}/track-reviews")
def get_project_reviews(project_id: str) -> dict[str, Any]:
    try:
        return project_reviews(store, store.get_project(project_id))
    except FileNotFoundError as exc:
        raise not_found(exc) from exc


@app.get("/api/projects/{project_id}/track-reviews.zip")
def export_project_reviews(project_id: str) -> Response:
    try:
        data = review_bundle(store, store.get_project(project_id))
    except FileNotFoundError as exc:
        raise not_found(exc) from exc
    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return Response(data, media_type="application/zip", headers={
        "Content-Disposition": 'attachment; filename="track_reviews.zip"'})


def decorate_review(project_id: str, video_id: str, result: dict) -> dict:
    for track in result["index"]["tracks"]:
        for sample in track["samples"]:
            for kind in ("crop", "context"):
                relative = sample[kind + '_path']
                sample[f"{kind}_url"] = (f'/api/projects/{project_id}/videos/{video_id}/review-evidence/{relative}'
                    if result.get('evidence_origin') == 'cache' else artifact_url(
                    project_id, video_id, f"analysis/track_review/{relative}"))
    return result


@app.get("/api/projects/{project_id}/videos/{video_id}/track-review")
def get_track_review(project_id: str, video_id: str, offset: int = Query(default=0, ge=0),
                     limit: int | None = Query(default=None, ge=1, le=24),
                     status: Literal['all', 'remaining', 'retained', 'filtered', 'risk', 'uncertain', 'full'] = 'all',
                     q: str = Query(default='', max_length=120)) -> dict:
    video = get_video(project_id, video_id)
    if runner.is_active(project_id, video_id):
        raise HTTPException(status_code=409, detail="Wait for processing to finish before reviewing")
    try:
        data = catalog.review(store, project_id, video)
        if isinstance(limit, int):
            data = review_page(data, offset=offset, limit=limit, status=status, q=q)
        return decorate_review(project_id, video_id, data)
    except FileNotFoundError as exc:
        raise not_found(exc) from exc
    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.patch('/api/projects/{project_id}/videos/{video_id}/track-review')
def patch_track_review(project_id: str, video_id: str, request: ReviewSave) -> dict:
    video = get_video(project_id, video_id)
    if runner.is_active(project_id, video_id):
        raise HTTPException(status_code=409, detail='Cannot review a running clip')
    try:
        data = save_review(store, project_id, video, request.review, request.expected_revision, partial=True)
        # A PATCH response stays compact even for long videos; the client reloads its visible page.
        return {key: data[key] for key in ('metrics', 'revision', 'stale_review', 'warning')}
    except FileNotFoundError as exc:
        raise not_found(exc) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.put("/api/projects/{project_id}/videos/{video_id}/track-review")
def put_track_review(project_id: str, video_id: str, request: ReviewSave) -> dict:
    video = get_video(project_id, video_id)
    if runner.is_active(project_id, video_id):
        raise HTTPException(status_code=409, detail="Cannot review a running clip")
    try:
        return decorate_review(project_id, video_id, save_review(
            store, project_id, video, request.review, request.expected_revision))
    except FileNotFoundError as exc:
        raise not_found(exc) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post('/api/projects/{project_id}/videos/{video_id}/review-evidence')
def prepare_review_evidence(project_id: str, video_id: str) -> dict:
    video = get_video(project_id, video_id)
    if runner.is_active(project_id, video_id):
        raise HTTPException(status_code=409, detail='Wait for model processing to finish')
    try:
        return evidence_jobs.submit(project_id, video)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get('/api/projects/{project_id}/videos/{video_id}/review-evidence/{relative:path}')
def review_evidence_file(project_id: str, video_id: str, relative: str) -> FileResponse:
    if not relative.startswith('images/'):
        raise HTTPException(status_code=400, detail='Only evidence images are served here')
    return FileResponse(safe_child(evidence_root(store, project_id, video_id), relative))


@app.get("/api/projects/{project_id}/videos/{video_id}/track-review/{track_id}/observations")
def get_review_observations(project_id: str, video_id: str, track_id: int) -> dict:
    video = get_video(project_id, video_id)
    if runner.is_active(project_id, video_id):
        raise HTTPException(status_code=409, detail="Cannot inspect changing tracks")
    paths = review_paths(video)
    try:
        meta = load_json(paths["video_meta"])
        grouped = load_tracks(paths["tracks"], meta or {})
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if track_id not in grouped:
        raise HTTPException(status_code=404, detail="Unknown raw track")
    return {"fps": meta["fps"], "raw_track_id": track_id,
            "observations": [{"frame_idx": r.frame_idx, "bbox_xyxy": r.bbox_xyxy}
                             for r in grouped[track_id]]}


@app.get("/api/projects/{project_id}/videos/{video_id}/merge-review")
def export_merge_review(project_id: str, video_id: str) -> Response:
    import json
    data = get_track_review(project_id, video_id)
    return Response(json.dumps(merge_review(data["index"], data["review"]), ensure_ascii=False, indent=2),
                    media_type="application/json", headers={
                        "Content-Disposition": 'attachment; filename="merge_review.json"'})


@app.post("/api/projects/{project_id}/videos")
async def upload_videos(
    project_id: str,
    files: Annotated[list[UploadFile], File()],
) -> dict[str, Any]:
    try:
        project = store.get_project(project_id)
    except FileNotFoundError as exc:
        raise not_found(exc) from exc
    if project.get("read_only"):
        raise HTTPException(
            status_code=409,
            detail="Imported experiments are read-only. Create a new experiment to upload videos.",
        )
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
    artifacts = decorate_artifacts(project_id, video, collect_video_artifacts(video))
    # Audit consistency belongs next to merged archives, not just on the review tab.
    if not runner.is_active(project_id, video_id):
        try:
            artifacts["evaluation"]["track_review"] = collect_review(store, project_id, video)["metrics"]
        except (OSError, ValueError, KeyError, TypeError):
            artifacts["evaluation"]["track_review"] = None
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
    if evidence_jobs.active(project_id, video_id):
        raise HTTPException(status_code=409, detail='Wait for the active evidence job before changing model outputs')
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


@app.get("/api/projects/{project_id}/videos/{video_id}/media/{kind}")
def cached_media(
    project_id: str, video_id: str, kind: Literal["source", "tracking", "final"]
) -> FileResponse:
    video = get_video(project_id, video_id)
    if not video.get("media_cache_dir"):
        raise HTTPException(status_code=404, detail="No preview cache")
    cache = within_root(Path(video["media_cache_dir"]), settings.output_root)
    return FileResponse(safe_child(cache, f"{kind}.mp4"), media_type="video/mp4")


@app.get("/api/projects/{project_id}/videos/{video_id}/inspection.zip")
def inspection_bundle(project_id: str, video_id: str) -> Response:
    try:
        content = build_inspection_bundle(get_video(project_id, video_id))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return Response(
        content,
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{safe_slug(video_id)}-inspection.zip"',
        },
    )


@app.get("/api/projects/{project_id}/videos/{video_id}/identity-evidence")
def get_identity_evidence(project_id: str, video_id: str,
                          purpose: Literal["kpr", "jersey"] = "kpr",
                          track_id: int | None = Query(default=None, ge=0),
                          offset: int = Query(default=0, ge=0),
                          limit: int = Query(default=24, ge=1, le=48)):
    video = get_video(project_id, video_id)
    return evidence_page(video["output_dir"], purpose, track_id=track_id, offset=offset, limit=limit)


@app.get("/api/projects/{project_id}/videos/{video_id}/player-assignments")
def get_clip_players(project_id: str, video_id: str):
    video = get_video(project_id, video_id)
    try:
        return clip_player_assignments(settings, store, project_id, video)
    except FileNotFoundError as exc:
        raise not_found(exc) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/api/projects/{project_id}/videos/{video_id}/records/{kind}")
def artifact_records(
    project_id: str,
    video_id: str,
    kind: Literal[
        "tracks",
        "identities",
        "actions",
        "linked_actions",
        "pairs",
        "events",
        "samples",
        "sampling",
        "quality",
        "quality_observations",
        "resolution",
        "raw_identities",
        "jersey_tracks",
        "jersey_people",
    ],
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
        "events": output / "action/events.jsonl",
        "samples": output / "identity/kpr_samples.jsonl",
        "sampling": output / "identity/kpr_track_sampling.jsonl",
        "quality": output / "quality/quality_tracks.jsonl",
        "quality_observations": output / "quality/quality_observations.jsonl",
        "resolution": output / "identity/resolution_pairs.jsonl",
        "raw_identities": output / "identity_raw/identities.jsonl",
        "jersey_tracks": output / "identity/jersey_tracks.jsonl",
        "jersey_people": output / "identity/jersey_people.jsonl",
    }
    path = mapping[kind]
    if kind in {"pairs", "samples", "sampling"} and not path.is_file():
        path = output / "identity_raw" / path.name
    return {
        "kind": kind,
        "path": str(path),
        "records": read_jsonl(path, limit=limit),
        "limit": limit,
    }
