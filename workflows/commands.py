"""Shared subprocess commands for CLI batches and the Web queue; no HTTP dependency."""

import json
from contracts.schema import write_json
from pathlib import Path
import sys
from typing import Any


class CommandBuilder:
    def __init__(self, settings, control_python: str | None = None, *, validate_resources=True):
        self.validate_resources = validate_resources
        self.settings = settings
        self.control_python = control_python or sys.executable

    def _require(self, path: Path, label: str) -> None:
        if self.validate_resources and not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")

    @staticmethod
    def paths(video: dict[str, Any]) -> dict[str, Path]:
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
            "review": output / "analysis/track_review",
            "log": output / "pipeline.log",
            "match_root": output.parent,
            "players": output / "identity/player_registration.json",
            "player_map": output.parent / "library/identity_map.jsonl",
        }

    def artifact_exists(self, stage: str, paths: dict[str, Path]) -> bool:
        expected = {
            "tracking": paths["tracks"],
            "quality": paths["quality"] / "quality_summary.json",
            "identity": paths["identity_raw"] / "identity_archive_manifest.json",
            "resolution": paths["identity"] / "resolution_summary.json",
            "action": paths["actions"],
            "link": paths["linked_actions"],
            "aggregate": paths["events"],
            "render": paths["result"],
            "review": paths["review"] / "index.json",
            "jersey": paths["identity"] / "jersey_summary.json",
            "players": paths["players"],
        }
        artifact = expected[stage]
        if stage == "players" and not (paths["match_root"] / "library/players.sqlite3").is_file():
            return False
        dependencies = {
            "jersey": [paths["stable_tracks"], paths["identity_map"]],
            "players": [paths["identity"] / "resolution_summary.json"] +
                       ([paths["events"]] if paths["events"].is_file() else []) +
                       ([expected["jersey"]] if expected["jersey"].is_file() else []),
            "review": [paths["tracks"], paths["video_meta"], paths["quality"] / "quality_tracks.jsonl"],
            "quality": [paths["tracks"], paths["video_meta"]],
            "identity": [paths["stable_tracks"]],
            "resolution": [paths["stable_tracks"], expected["identity"]],
            "action": [paths["stable_tracks"]],
            "link": [paths["actions"], paths["identity_map"]],
            "aggregate": [paths["actions"]],
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

    def command(
        self, stage: str, paths: dict[str, Path], options: dict[str, Any]
    ) -> list[str]:
        if stage == "jersey":
            self._require(self.settings.qwen_python, "Qwen Python")
            return [str(self.settings.qwen_python), "-m", "cli.jersey_qwen",
                    "--video", str(paths["source"]), "--tracks", str(paths["stable_tracks"]),
                    "--identity-map", str(paths["identity_map"]), "--output-dir", str(paths["identity"]),
                    "--model-dir", str(self.settings.qwen_model_dir)]
        if stage == "players":
            command = [self.control_python, "-m", "cli.build_person_library",
                       "--run-dir", str(paths["match_root"]), "--clip-name", paths["output"].name,
                       "--match-id", str(options.get("match_id") or paths["match_root"].name),
                       "--max-distance", str(options.get("player_match_distance", .2)),
                       "--novelty-distance", str(options.get("player_novelty_distance", .5)),
                       "--min-margin", str(options.get("player_min_margin", .05)),
                       "--registration-output", str(paths["players"])]
            if options.get("jersey_qwen", True):
                command.append("--use-jersey-evidence")
            return command
        if stage == "review":
            self._require(self.settings.review_python, "Review Python (OpenCV)")
            return [str(self.settings.review_python), "-m", "cli.prepare_track_review",
                    "--input", str(paths["source"]), "--tracks", str(paths["tracks"]),
                    "--video-meta", str(paths["video_meta"]),
                    "--quality", str(paths["quality"] / "quality_tracks.jsonl"),
                    "--output-dir", str(paths["review"]), "--overwrite"]
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
                "--sample-min-gap-s",
                str(float(options.get("identity_sample_min_gap_s", 0.25))),
                "--min-det-score",
                str(float(options.get("identity_min_det_score", 0.5))),
                "--overwrite",
            ]
        if stage == "quality":
            return [
                self.control_python,
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
                self.control_python,
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
                "--allow-unmapped",
                "--player-map", str(paths["player_map"]),
                "--clip-name", paths["output"].name,
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
            return [
                self.control_python,
                "-m",
                "cli.aggregate_events",
                "--input",
                str(paths["actions"]),
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
    def write_empty_model_artifacts(stage: str, paths: dict[str, Path]) -> bool:
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
