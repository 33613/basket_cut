"""Single-GPU job queue that composes the existing CLI entry points."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

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
    def _paths(video: dict[str, Any]) -> dict[str, Path]:
        output = Path(video["output_dir"])
        return {
            "source": Path(video["source_path"]),
            "output": output,
            "tracking": output / "tracking",
            "tracks": output / "tracking/tracks.jsonl",
            "quality": output / "quality",
            "stable_tracks": output / "quality/tracks.jsonl",
            "identity_raw": output / "identity_raw",
            "identity": output / "identity",
            "identity_map": output / "identity/identity_map.jsonl",
            "action": output / "action",
            "actions": output / "action/actions.jsonl",
            "linked_actions": output / "action/actions_with_identity.jsonl",
            "events": output / "action/events.jsonl",
            "video_meta": output / "tracking/video_meta.json",
            "visualization": output / "visualization",
            "result": output / "visualization/result.mp4",
            "web_result": output / "visualization/result_web.mp4",
            "log": output / "pipeline.log",
        }

    def _artifact_exists(self, stage: str, paths: dict[str, Path]) -> bool:
        expected = {
            "tracking": paths["tracks"],
            "quality": paths["quality"] / "quality_summary.json",
            "identity": paths["identity_raw"] / "identity_archive_manifest.json",
            "resolution": paths["identity"] / "resolution_summary.json",
            "action": paths["actions"],
            "link": paths["linked_actions"],
            "aggregate": paths["events"],
            "render": paths["result"],
        }
        artifact = expected[stage]
        dependencies = {
            "quality": [paths["tracks"], paths["video_meta"]],
            "identity": [paths["stable_tracks"]],
            "resolution": [paths["stable_tracks"], expected["identity"]],
            "action": [paths["stable_tracks"]],
            "link": [paths["actions"], paths["identity_map"]],
            "aggregate": [
                paths["linked_actions"]
                if paths["linked_actions"].is_file()
                else paths["actions"]
            ],
            "render": [paths["stable_tracks"]]
            + [
                path
                for path in (paths["linked_actions"], paths["identity_map"])
                if path.is_file()
            ],
        }
        return artifact.is_file() and all(
            dependency.is_file()
            and dependency.stat().st_mtime_ns <= artifact.stat().st_mtime_ns
            for dependency in dependencies.get(stage, [])
        )

    def _plan(self, target: str, paths: dict[str, Path], force: bool) -> list[str]:
        requested = {
            "tracking": ["tracking", "quality"],
            "identity": ["tracking", "quality", "identity", "resolution"],
            "action": ["tracking", "quality", "action"],
            "final": ["tracking", "quality", "link", "aggregate", "render"],
            "full": [
                "tracking",
                "quality",
                "identity",
                "resolution",
                "action",
                "link",
                "aggregate",
                "render",
            ],
        }[target]
        if (
            target == "final"
            and (paths["identity_raw"] / "identity_archive_manifest.json").is_file()
        ):
            requested.insert(requested.index("link"), "resolution")
        plan = []
        stale = force
        for stage in requested:
            if stale or not self._artifact_exists(stage, paths):
                plan.append(stage)
                stale = True
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
        plan = self._plan(target, paths, force)
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

    def _command(
        self, stage: str, paths: dict[str, Path], options: dict[str, Any]
    ) -> list[str]:
        if stage == "tracking":
            self._require(self.settings.motip_python, "MOTIP Python")
            self._require(self.settings.motip_checkpoint, "MOTIP checkpoint")
            command = [
                str(self.settings.motip_python),
                "-m",
                "cli.tracking",
                "--input",
                str(paths["source"]),
                "--checkpoint",
                str(self.settings.motip_checkpoint),
                "--output-dir",
                str(paths["tracking"]),
                "--overwrite",
            ]
            if options.get("max_frames"):
                command.extend(["--max-frames", str(int(options["max_frames"]))])
            if options.get("tracking_det_threshold") is not None:
                command.extend(
                    ["--det-thresh", str(float(options["tracking_det_threshold"]))]
                )
            return command
        if stage == "identity":
            self._require(self.settings.kpr_python, "KPR Python")
            self._require(self.settings.kpr_checkpoint, "KPR checkpoint")
            return [
                str(self.settings.kpr_python),
                "-m",
                "cli.identity",
                "--input",
                str(paths["source"]),
                "--tracks",
                str(paths["stable_tracks"]),
                "--output-dir",
                str(paths["identity_raw"]),
                "--kpr-root",
                str(self.settings.repository_root / "KPR"),
                "--config",
                str(
                    self.settings.repository_root
                    / "configs/kpr/multidataset_sports_test.yaml"
                ),
                "--checkpoint",
                str(self.settings.kpr_checkpoint),
                "--prompt-mode",
                str(options.get("prompt_mode", "none")),
                "--samples-per-track",
                str(int(options.get("identity_samples", 8))),
                "--min-det-score",
                str(float(options.get("identity_min_det_score", 0.5))),
                "--overwrite",
            ]
        if stage == "quality":
            return [
                sys.executable,
                "-m",
                "cli.track_quality",
                "--tracks",
                str(paths["tracks"]),
                "--video-meta",
                str(paths["video_meta"]),
                "--output-dir",
                str(paths["quality"]),
                "--min-observations",
                str(options.get("quality_min_observations", 3)),
                "--min-observed-seconds",
                str(options.get("quality_min_observed_seconds", 0.1)),
                "--overwrite",
            ]
        if stage == "resolution":
            command = [
                sys.executable,
                "-m",
                "cli.resolve_identity",
                "--tracks",
                str(paths["stable_tracks"]),
                "--video-meta",
                str(paths["video_meta"]),
                "--archive-dir",
                str(paths["identity_raw"]),
                "--quality-dir",
                str(paths["quality"]),
                "--output-dir",
                str(paths["identity"]),
                "--overwrite",
            ]
            if options.get("identity_merge_distance") is not None:
                command.extend(
                    ["--max-distance", str(options["identity_merge_distance"])]
                )
            return command
        if stage == "action":
            self._require(self.settings.action_python, "MMAction2 Python")
            self._require(self.settings.action_checkpoint, "action checkpoint")
            self._require(self.settings.action_config, "action config")
            self._require(self.settings.action_label_map, "action label map")
            return [
                str(self.settings.action_python),
                "-m",
                "cli.action",
                "--input",
                str(paths["source"]),
                "--tracks",
                str(paths["stable_tracks"]),
                "--config",
                str(self.settings.action_config),
                "--checkpoint",
                str(self.settings.action_checkpoint),
                "--label-map",
                str(self.settings.action_label_map),
                "--output",
                str(paths["actions"]),
                "--action-threshold",
                str(float(options.get("action_threshold", 0.2))),
                "--min-det-score",
                str(float(options.get("action_min_det_score", 0.3))),
                "--overwrite",
            ]
        if stage == "link":
            self._require(self.settings.motip_python, "MOTIP Python")
            return [
                str(self.settings.motip_python),
                "-m",
                "cli.link_events",
                "--actions",
                str(paths["actions"]),
                "--identity-map",
                str(paths["identity_map"]),
                "--output",
                str(paths["linked_actions"]),
                "--overwrite",
            ]
        if stage == "render":
            self._require(self.settings.motip_python, "MOTIP Python")
            command = [
                str(self.settings.motip_python),
                "-m",
                "cli.render_results",
                "--input",
                str(paths["source"]),
                "--tracks",
                str(paths["stable_tracks"]),
                "--output",
                str(paths["result"]),
                "--show-top-candidate",
                "--overwrite",
            ]
            action_path = (
                paths["linked_actions"]
                if paths["linked_actions"].is_file()
                else paths["actions"]
            )
            if action_path.is_file():
                command.extend(["--actions", str(action_path)])
            if paths["identity_map"].is_file():
                command.extend(["--identity-map", str(paths["identity_map"])])
            if options.get("max_frames"):
                command.extend(["--max-frames", str(int(options["max_frames"]))])
            return command
        if stage == "aggregate":
            self._require(self.settings.motip_python, "MOTIP Python")
            return [
                str(self.settings.motip_python),
                "-m",
                "cli.aggregate_events",
                "--input",
                str(
                    paths["linked_actions"]
                    if paths["linked_actions"].is_file()
                    else paths["actions"]
                ),
                "--video-meta",
                str(paths["video_meta"]),
                "--output",
                str(paths["events"]),
                "--score-threshold",
                str(float(options.get("action_threshold", 0.2))),
                "--max-missing-steps",
                "1",
                "--min-support",
                "1",
                "--score-reducer",
                "mean",
                "--overwrite",
            ]
        raise ValueError(f"Unknown stage: {stage}")

    @staticmethod
    def _write_empty_model_artifacts(stage: str, paths: dict[str, Path]) -> bool:
        """An empty quality result is valid, not a model crash or a perfect score."""
        if (
            stage not in {"identity", "action"}
            or not paths["stable_tracks"].is_file()
            or paths["stable_tracks"].stat().st_size
        ):
            return False
        meta = json.loads(paths["video_meta"].read_text(encoding="utf-8"))
        if stage == "identity":
            destination = paths["identity_raw"]
            destination.mkdir(parents=True, exist_ok=True)
            for name in (
                "identities.jsonl",
                "identity_map.jsonl",
                "kpr_track_pairs.jsonl",
                "kpr_samples.jsonl",
                "kpr_track_sampling.jsonl",
            ):
                (destination / name).write_text("", encoding="utf-8")
            summary = {
                "video_id": meta["video_id"],
                "identity_count": 0,
                "model_invoked": False,
                "reason": "no_stable_tracks",
                "identity_resolution_mode": "archive_no_merge",
            }
            write_json(destination / "kpr_summary.json", summary)
            write_json(destination / "identity_archive_manifest.json", summary)
        else:
            paths["action"].mkdir(parents=True, exist_ok=True)
            paths["actions"].write_text("", encoding="utf-8")
            write_json(
                paths["action"] / "actions.summary.json",
                {
                    "video_id": meta["video_id"],
                    "model_invoked": False,
                    "reason": "no_stable_tracks",
                    "settings": {"predict_stepsize": 8},
                },
            )
        return True

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
