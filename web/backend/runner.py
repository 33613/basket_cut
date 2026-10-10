"""Single-GPU job queue that composes the existing CLI entry points."""

from __future__ import annotations

import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

from workflows.commands import CommandBuilder
from contracts.schema import write_json
from web.backend.settings import WebSettings
from web.backend.store import ProjectStore, utc_now

TARGETS = ("tracking", "identity", "action", "final", "full")


class PipelineRunner:
    """Serialize GPU work while keeping the HTTP server responsive."""

    def __init__(self, settings: WebSettings, store: ProjectStore) -> None:
        self.settings = settings
        self.store = store
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="basket-pipeline"
        )
        self._futures: dict[tuple[str, str], Future[Any]] = {}
        self._lock = threading.RLock()

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=False)

    def is_active(self, project_id: str, video_id: str) -> bool:
        with self._lock:
            future = self._futures.get((project_id, video_id))
            return future is not None and not future.done()

    def submit(
        self,
        project_id: str,
        video_id: str,
        *,
        target: str,
        options: dict[str, Any],
        force: bool,
    ) -> dict[str, Any]:
        if target not in TARGETS:
            raise ValueError(f"Unknown target {target!r}; expected one of {TARGETS}")
        distance = options.get("player_match_distance", .2)
        novelty = options.get("player_novelty_distance", .5)
        margin = options.get("player_min_margin", .05)
        if (not math.isfinite(distance) or distance < 0 or not math.isfinite(novelty)
                or novelty <= distance or not math.isfinite(margin) or margin < 0):
            raise ValueError("Require novelty distance > match distance and nonnegative margin")
        key = (project_id, video_id)
        with self._lock:
            if self.store.get_video(project_id, video_id).get("read_only"):
                raise ValueError(
                    "Imported results are read-only. Upload a video to a new experiment to rerun it."
                )
            if self.is_active(project_id, video_id):
                raise RuntimeError("This video already has a queued or running job")

            def mark_queued(video: dict[str, Any]) -> None:
                video["status"] = "queued"
                video["active_stage"] = None
                video["error"] = None
                video["run_options"] = {
                    "target": target,
                    "force": force,
                    **options,
                }

            queued = self.store.update_video(project_id, video_id, mark_queued)
            future = self._executor.submit(
                self._run_job,
                project_id,
                video_id,
                target,
                dict(options),
                force,
            )
            self._futures[key] = future
            future.add_done_callback(lambda _: self._discard_future(key))
            return queued

    def _discard_future(self, key: tuple[str, str]) -> None:
        with self._lock:
            self._futures.pop(key, None)

    @staticmethod
    def _paths(video):
        return CommandBuilder.paths(video)

    def _artifact_exists(self, stage, paths):
        return CommandBuilder(self.settings).artifact_exists(stage, paths)

    def _plan(self, target: str, paths: dict[str, Path], force: bool, jersey: bool = False) -> list[str]:
        from workflows.batch import stages_for
        requested = {
            "tracking": ["tracking", "quality", "review"],
            "identity": stages_for("identity", jersey),
            "action": ["tracking", "quality", "action", "aggregate", "review"],
            "final": ["players", "link", "render", "review"],
            "full": stages_for("full", jersey),
        }[target]
        dependencies = {
            "quality": ["tracking"], "action": ["quality"], "aggregate": ["action"],
            "identity": ["quality"], "resolution": ["identity"], "jersey": ["resolution"],
            "players": ["resolution"] + (["aggregate"] if "aggregate" in requested else []) +
                       (["jersey"] if jersey else []),
            "link": ["action", "players"], "render": ["quality", "link", "resolution"],
            "review": ["tracking", "quality"],
        }
        plan = []
        for stage in requested:
            if (force or not self._artifact_exists(stage, paths)
                    or any(parent in plan for parent in dependencies.get(stage, []))):
                plan.append(stage)
        return plan

    def _run_job(
        self,
        project_id: str,
        video_id: str,
        target: str,
        options: dict[str, Any],
        force: bool,
    ) -> None:
        video = self.store.get_video(project_id, video_id)
        paths = self._paths(video)
        paths["output"].mkdir(parents=True, exist_ok=True)
        options = {**options, "match_id": project_id}
        project = self.store.get_project(project_id)
        # Uploads are stored under project/videos/<clip>; that parent is the match run.
        manifest_path = paths["match_root"] / "batch_manifest.json"
        write_json(manifest_path, {"schema_version": 2, "match_id": project_id,
            "target": "full" if target in {"full", "action", "final"} else "identity",
            "clips": [{"name": Path(v["output_dir"]).name} for v in project["videos"]]})
        plan = self._plan(target, paths, force, jersey=bool(options.get('jersey_qwen', True)))
        if not options.get('jersey_qwen', True):
            plan = [stage for stage in plan if stage != 'jersey']
            self.store.set_stage(project_id, video_id, 'jersey', 'skipped', return_code=None)
        self._append_log(
            paths["log"],
            f"\n=== Job {utc_now()} target={target} force={force} plan={plan} ===\n",
        )
        try:
            for stage in plan:
                if self._write_empty_model_artifacts(stage, paths):
                    self.store.set_stage(
                        project_id,
                        video_id,
                        stage,
                        "completed",
                        command=None,
                        return_code=0,
                    )
                    self._append_log(
                        paths["log"],
                        f"[{stage}] no stable tracks: empty output, model not invoked\n",
                    )
                    continue
                if stage == "aggregate" and not paths["actions"].is_file():
                    self.store.set_stage(
                        project_id, video_id, stage, "skipped", return_code=None
                    )
                    self._append_log(
                        paths["log"], "[aggregate] skipped: no action predictions\n"
                    )
                    continue
                if stage == "link" and not (
                    paths["actions"].is_file() and paths["identity_map"].is_file()
                ):
                    self.store.set_stage(
                        project_id,
                        video_id,
                        stage,
                        "skipped",
                        return_code=None,
                    )
                    self._append_log(
                        paths["log"],
                        "[link] skipped: action or identity artifact is missing\n",
                    )
                    continue
                if (
                    stage in {"link", "aggregate"}
                    and paths["actions"].is_file()
                    and paths["stable_tracks"].stat().st_mtime_ns
                    > paths["actions"].stat().st_mtime_ns
                ):
                    raise ValueError(
                        "Action predictions predate the current quality result; rerun action/full or use cli.refine_results for cached filtering"
                    )
                try:
                    command = self._command(stage, paths, options)
                except Exception:
                    self.store.set_stage(
                        project_id,
                        video_id,
                        stage,
                        "failed",
                        command=None,
                        return_code=-1,
                    )
                    raise
                self.store.set_stage(
                    project_id,
                    video_id,
                    stage,
                    "running",
                    command=command,
                    return_code=None,
                )
                return_code = self._run_command(command, paths["log"])
                if return_code != 0:
                    self.store.set_stage(
                        project_id,
                        video_id,
                        stage,
                        "failed",
                        return_code=return_code,
                    )
                    raise RuntimeError(
                        f"Stage {stage} exited with status {return_code}. "
                        "Open the log panel for the traceback."
                    )
                if stage == "render":
                    self._make_browser_video(paths)
                self.store.set_stage(
                    project_id,
                    video_id,
                    stage,
                    "completed",
                    return_code=return_code,
                )

            def mark_completed(value: dict[str, Any]) -> None:
                value["status"] = "completed"
                value["active_stage"] = None
                value["error"] = None

            self.store.update_video(project_id, video_id, mark_completed)
            self._append_log(paths["log"], f"=== Completed {utc_now()} ===\n")
        except Exception as exc:  # noqa: BLE001 -- persist worker failures for polling clients
            error_message = str(exc)
            active = self.store.get_video(project_id, video_id).get("active_stage")
            if active:
                self.store.set_stage(
                    project_id,
                    video_id,
                    active,
                    "failed",
                    return_code=-1,
                )

            def mark_failed(value: dict[str, Any]) -> None:
                value["status"] = "failed"
                value["active_stage"] = None
                value["error"] = error_message

            self.store.update_video(project_id, video_id, mark_failed)
            self._append_log(paths["log"], f"ERROR: {error_message}\n")

    def _command(self, stage, paths, options):
        return CommandBuilder(self.settings).command(stage, paths, options)

    @staticmethod
    def _write_empty_model_artifacts(stage, paths):
        return CommandBuilder.write_empty_model_artifacts(stage, paths)

    def _run_command(self, command: list[str], log_path: Path) -> int:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n$ {shlex.join(command)}\n")
            log.flush()
            environment = dict(os.environ)
            environment["PYTHONUNBUFFERED"] = "1"
            process = subprocess.Popen(
                command,
                cwd=self.settings.repository_root,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                env=environment,
            )
            return process.wait()

    def _make_browser_video(self, paths: dict[str, Path]) -> None:
        ffmpeg = shutil.which(self.settings.ffmpeg)
        if ffmpeg is None or not paths["result"].is_file():
            self._append_log(
                paths["log"],
                "[render] ffmpeg unavailable; serving the original MP4 codec\n",
            )
            return
        command = [
            ffmpeg,
            "-y",
            "-i",
            str(paths["result"]),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "22",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(paths["web_result"]),
        ]
        return_code = self._run_command(command, paths["log"])
        if return_code != 0:
            paths["web_result"].unlink(missing_ok=True)
            self._append_log(
                paths["log"],
                "[render] browser transcode failed; using result.mp4 instead\n",
            )

    @staticmethod
    def _require(path: Path, label: str) -> None:
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    @staticmethod
    def _append_log(path: Path, value: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(value)
