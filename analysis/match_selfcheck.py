"""CPU integration checks with real video cuts and synthetic model responses."""

from dataclasses import replace
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

from analysis.person_library_selfcheck import make_library_fixture, rows
from contracts.schema import read_jsonl, write_json
from contracts.lineage import track_uid
from pipeline.media import probe_video, cut_video, sha256_file, frame_sync_args, preflight_video
from pipeline.segmentation.service import (SegmentOptions, analyze_video, export_segments,
    make_segments, parse_boundary, parse_coarse, plan_windows)
from pipeline.identity.player_registry import PlayerRegistry
from workflows.person_library import PersonLibraryOptions, build_person_library
from workflows.player_events import query_player_events, export_player_events


class SyntheticQwen:
    provenance = {"architecture": "synthetic_cpu_fixture", "model_sha256": "not_a_model"}

    def __init__(self):
        self.coarse, self.calls = 0, 0

    def read_video(self, path, prompt, *, num_frames, max_new_tokens):
        assert path.is_file() and max_new_tokens == 2048
        self.calls += 1
        if "Find candidate boundary ranges" in prompt:
            interval = ([2.8, 3.2], [0, .2], [.8, 1.2])[self.coarse]
            self.coarse += 1
            return json.dumps({"boundaries": [{"time_range_s": interval,
                "kind": "possession_change", "evidence": "synthetic fixture", "needs_review": False}]})
        return json.dumps({"confirmed": True, "boundary_s": 1.0,
            "kind": "possession_change", "evidence": "synthetic verification", "needs_review": False})


class MatchChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
            raise RuntimeError("ffmpeg and ffprobe are required for match integration checks")
        cls.media_temp = tempfile.TemporaryDirectory(prefix="basket-match-media-")
        cls.source_fixture = Path(cls.media_temp.name) / "source.mp4"
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
            "testsrc2=size=96x64:rate=10:duration=8", "-c:v", "libx264", "-threads", "1",
            "-pix_fmt", "yuv420p", str(cls.source_fixture)], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.media_temp.cleanup()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="basket-match-check-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source.mp4"
        shutil.copyfile(self.source_fixture, self.source)
        self.output = self.root / "segmentation"
        self.options = SegmentOptions(window_s=4, overlap_s=1, review_radius_s=1, min_free_gb=0)

    def analyze(self, backend=None):
        return analyze_video(self.source, self.output, backend or SyntheticQwen(), options=self.options)

    def library(self):
        run = make_library_fixture(self.root, angles=(0, .05))
        checksum = sha256_file(self.source)
        manifest = json.loads((run / "batch_manifest.json").read_text())
        for i, row in enumerate(manifest["clips"]):
            row.update(clip_id=row["name"], lineage={"start_s": i * 4, "end_s": (i + 1) * 4,
                "source_video_id": "SRC-synthetic", "source_path": str(self.source),
                "source_sha256": checksum, "needs_review": False})
            meta_path = run / row["name"] / "tracking/video_meta.json"
            meta = json.loads(meta_path.read_text())
            Path(meta["input_path"]).unlink()
            cut_video(self.source, meta["input_path"], i * 4, (i + 1) * 4)
            meta.update(fps=10, frame_count=40, processed_frames=40, width=96, height=64)
            write_json(meta_path, meta)
        write_json(run / "batch_manifest.json", manifest)
        build_person_library(PersonLibraryOptions(run))
        with PlayerRegistry(run / "library/players.sqlite3", "synthetic-match") as registry:
            player = registry.snapshot()["people"][0]["player_id"]
        return run, player

    def test_windows_cover_entire_video_and_overlap(self):
        self.assertEqual(plan_windows(8, self.options), [
            {"start_s": 0, "end_s": 4}, {"start_s": 3, "end_s": 7}, {"start_s": 6, "end_s": 8}])
        self.assertEqual(len(plan_windows(.2, self.options)), 1)
        for kwargs in ({"overlap_s": 4}, {"window_s": float("nan")}, {"coarse_frames": 5}, {"min_free_gb": -1}):
            with self.assertRaises(ValueError): replace(self.options, **kwargs).validate()

    def test_qwen_diagnostic_reconstructs_actual_processed_pixels(self):
        from analysis.segmentation_diagnostic import reconstruct_video, save_video_inputs
        frames = np.linspace(-1, 1, 4 * 3 * 64 * 96, dtype=np.float32).reshape(4, 3, 64, 96)
        # The independent forward layout from the pinned HF Qwen3 processor.
        patches = frames.reshape(2, 2, 3, 2, 2, 16, 3, 2, 16).transpose(
            0, 3, 6, 4, 7, 2, 1, 5, 8).reshape(2 * 4 * 6, 3 * 2 * 16 * 16)
        decoded = reconstruct_video(patches, [2, 4, 6], patch_size=16, temporal_patch_size=2, merge_size=2)
        np.testing.assert_array_equal(decoded, frames)
        processor = SimpleNamespace(video_processor=SimpleNamespace(patch_size=16,
            temporal_patch_size=2, merge_size=2, do_normalize=True, image_std=[.5] * 3,
            image_mean=[.5] * 3, do_rescale=True, rescale_factor=1 / 255),
            batch_decode=lambda *args, **kwargs: ["0.04<|image_pad|><|image_pad|>0.20"])
        inputs = {"pixel_values_videos": patches, "video_grid_thw": np.array([[2, 4, 6]]),
                  "input_ids": np.ones((1, 100))}
        output = self.root / "diagnostic-tensors"
        summary = save_video_inputs(processor, inputs, output)
        self.assertEqual((summary["processed_frames"], summary["processed_height"], summary["processed_width"]), (4, 64, 96))
        self.assertGreater(summary["adjacent_frame_mean_absolute_difference"], 0)
        self.assertTrue((output / "model-input.jpg").is_file())
        self.assertEqual((output / "input-text.txt").read_text(), "0.04[visual tokens omitted]0.20")
        with self.assertRaises(ValueError):
            reconstruct_video(patches[:1], [2, 4, 6], patch_size=16, temporal_patch_size=2, merge_size=2)

    def test_segmentation_diagnostic_preserves_original_evidence_and_errors(self):
        from analysis.segmentation_diagnostic import diagnose, diagnostic_plan
        self.analyze()
        protocol, windows = diagnostic_plan(self.output, 3.5, 6.5)
        self.assertEqual([w["window_id"] for w in windows], ["coarse-000000", "coarse-000001", "coarse-000002"])
        original = {str(p): p.read_bytes() for p in self.output.rglob("*.json")}
        class Backend:
            provenance = protocol["model"]
            def read_video(self, path, prompt, **kwargs):
                assert path.is_file() and kwargs["num_frames"] == 16
                return "visible basketball play" if "chronological order" in prompt else "null"
        backend = Backend()
        output = self.root / "diagnostic"
        result = diagnose(self.output, output, windows, protocol, backend, capture_inputs=False)
        self.assertEqual(result["status"], "completed_not_accuracy_evaluated")
        self.assertEqual(len(result["results"]), 3)
        for row in result["results"]:
            self.assertIn("parse_error", row["queries"]["boundary"])
            self.assertNotIn("parsed_boundaries", row["queries"]["boundary"])
            self.assertTrue((output / row["window_id"] / "window.mp4").is_file())
        self.assertEqual(original, {str(p): p.read_bytes() for p in self.output.rglob("*.json")})
        from pipeline.identity.qwen_backend import VIDEO_BUDGET
        processor = SimpleNamespace(video_processor=SimpleNamespace(size=dict(VIDEO_BUDGET),
            patch_size=16, merge_size=2, temporal_patch_size=2))
        backend.processor = processor
        single = diagnose(self.output, self.root / "boundary-only", windows, protocol, backend,
                          capture_inputs=False, query_types=("boundary",), video_tokens=8192)
        self.assertEqual(single["original_video_budget"], VIDEO_BUDGET)
        self.assertEqual(single["effective_video_budget"]["longest_edge"], 16777216)
        self.assertTrue(all(set(row["queries"]) == {"boundary"} for row in single["results"]))
        self.assertEqual(processor.video_processor.size, VIDEO_BUDGET)
        with patch.object(backend, "read_video", side_effect=RuntimeError("synthetic inference failure")):
            with self.assertRaisesRegex(RuntimeError, "synthetic inference failure"):
                diagnose(self.output, self.root / "failed-budget", windows, protocol, backend,
                         capture_inputs=False, query_types=("boundary",), video_tokens=8192)
        self.assertEqual(processor.video_processor.size, VIDEO_BUDGET)
        failed = json.loads((self.root / "failed-budget/diagnostic.json").read_text())
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["effective_video_budget"]["longest_edge"], 16777216)
        import sys
        planned = subprocess.run([sys.executable, "-m", "cli.diagnose_segmentation",
            "--segmentation-dir", str(self.output), "--output-dir", str(self.root / "dry"),
            "--start-s", "3.5", "--end-s", "6.5", "--queries", "boundary",
            "--video-tokens", "8192", "--dry-run"], capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(planned.stdout)["model_queries"], 3)
        self.assertFalse((self.root / "dry").exists())
        with self.assertRaises(FileExistsError):
            diagnose(self.output, output, windows, protocol, backend, capture_inputs=False)
        with self.assertRaises(ValueError):
            diagnose(self.output, self.output / "wrong", windows, protocol, backend, capture_inputs=False)
        backend.provenance = {"model_sha256": "different"}
        with self.assertRaisesRegex(ValueError, "model differs"):
            diagnose(self.output, self.root / "different", windows, protocol, backend, capture_inputs=False)
        with self.assertRaises(ValueError):
            diagnostic_plan(self.output, -1, 3)

    def test_parser_rejects_out_of_window_and_false_precision(self):
        valid = {"boundaries": [{"time_range_s": [1, 2], "kind": "uncertain", "evidence": "occluded", "needs_review": True}]}
        self.assertEqual(len(parse_coarse(json.dumps(valid), 3)), 1)
        for interval in ([True, 2], [-1, 2], [2, 1], [1, 4], [0, float("nan")]):
            valid["boundaries"][0]["time_range_s"] = interval
            with self.assertRaises(ValueError): parse_coarse(json.dumps(valid), 3)
        with self.assertRaises(ValueError): parse_boundary('{"confirmed":true,"boundary_s":null,"kind":"uncertain","evidence":"none","needs_review":false}', 3)
        with self.assertRaises(ValueError): parse_coarse('A basketball game.', 3)

    def test_empty_qwen_array_and_valid_boundary_arrays_are_accepted(self):
        for raw in ('[]', '{"boundaries": []}', '```json\n[]\n```', '```JSON\n[]\n```'):
            self.assertEqual(parse_coarse(raw, 20), [])
        candidate = {"time_range_s": [1, 2], "kind": "possession_change",
                     "evidence": "opponent controls rebound", "needs_review": False}
        self.assertEqual(parse_coarse(json.dumps([candidate]), 20),
                         parse_coarse(json.dumps({"boundaries": [candidate]}), 20))
        for raw in ('null', 'true', '0', '"no boundaries"', '{}', '[{}]', '{"boundaries":null}'):
            with self.assertRaises(ValueError): parse_coarse(raw, 20)
        with self.assertRaises(ValueError): parse_coarse(json.dumps([{**candidate, "time_range_s": [1, 21]}]), 20)

    def test_format_retry_preserves_all_answers_and_changes_only_prompt(self):
        class Backend(SyntheticQwen):
            def read_video(self, path, prompt, **kwargs):
                if not self.calls:
                    self.calls += 1
                    return "null"
                return super().read_video(path, prompt, **kwargs)
        backend = Backend()
        result = self.analyze(backend)
        self.assertEqual(len(result["segments"]), 3)
        self.assertEqual(backend.calls, 6)
        saved = json.loads((self.output / "responses/coarse-000000.json").read_text())
        self.assertEqual(len(saved["attempts"]), 2)
        self.assertEqual(saved["attempts"][0]["raw_response"], "null")
        self.assertIn("error", saved["attempts"][0])
        self.assertIn("OUTPUT FORMAT RETRY", saved["attempts"][1]["prompt"])
        self.assertNotIn("error", saved)
        self.assertEqual(len(saved["normalized_response"]), 1)
        backend.read_video = lambda *a, **kw: self.fail("Successful retry must be cached")
        self.analyze(backend)

    def test_all_empty_arrays_finish_without_boundary_review(self):
        class Backend:
            provenance = SyntheticQwen.provenance
            def read_video(self, *args, **kwargs): return "[]"
        backend = Backend()
        result = self.analyze(backend)
        self.assertEqual(result["boundaries"], [])
        self.assertEqual(len(result["segments"]), 1)
        self.assertFalse(result["needs_review"])
        # Revalidate a cached response marked failed by an older parser, rather
        # than spending another GPU invocation or losing the raw evidence.
        path = self.output / "responses/coarse-000001.json"
        saved = json.loads(path.read_text()); saved["error"] = "Model response must be a JSON object"
        write_json(path, saved)
        backend.read_video = lambda *a, **kw: self.fail("Valid empty arrays must be reused")
        self.analyze(backend)
        self.assertNotIn("error", json.loads(path.read_text()))

    def test_real_window_cuts_review_merge_source_offsets_and_resume(self):
        backend = SyntheticQwen()
        result = self.analyze(backend)
        self.assertEqual(backend.calls, 5)
        self.assertEqual([r["source_boundary_s"] for r in result["boundaries"]], [3, 7])
        self.assertEqual(len(result["boundaries"][0]["support"]), 2)
        self.assertEqual([(r["start_s"], r["end_s"]) for r in result["segments"]], [(0, 3), (3, 7), (7, 8)])
        backend.read_video = lambda *a, **kw: self.fail("Cache should prevent another inference")
        self.assertEqual(self.analyze(backend)["segments"], result["segments"])
        self.assertFalse(list((self.output / "windows").glob("*.mp4")))

    def test_exported_clips_have_real_durations_hashes_and_stable_mtimes(self):
        self.analyze()
        manifest = export_segments(self.output, min_free_gb=0)
        times = {}
        for row, duration in zip(manifest["videos"], (3, 4, 1)):
            path = self.output / row["path"]
            self.assertAlmostEqual(probe_video(path)["duration_s"], duration, places=2)
            self.assertEqual(row["sha256"], sha256_file(path))
            times[path] = path.stat().st_mtime_ns
        export_segments(self.output, min_free_gb=0)
        self.assertTrue(all(p.stat().st_mtime_ns == t for p, t in times.items()))
        self.assertEqual(len({r["clip_id"] for r in manifest["videos"]}), 3)

    def test_failed_response_preserves_raw_evidence_and_failed_status(self):
        backend = SyntheticQwen(); backend.read_video = lambda *a, **kw: 'not JSON'
        with self.assertRaises(ValueError): self.analyze(backend)
        self.assertEqual(json.loads((self.output / "status.json").read_text())["status"], "failed")
        response = json.loads((self.output / "responses/coarse-000000.json").read_text())
        self.assertEqual(response["raw_response"], "not JSON")
        self.assertIn("error", response)
        self.assertEqual(len(response["attempts"]), 3)
        self.assertTrue(all(r["raw_response"] == "not JSON" and r["error"] for r in response["attempts"]))
        self.assertFalse((self.output / "segmentation.json").exists())

    def test_uncertain_boundaries_are_retained_without_inventing_cuts(self):
        review = {"confirmed": True, "needs_review": True, "kind": "uncertain",
                  "source_boundary_s": None, "candidate_id": "R1", "source_range_s": [2, 4]}
        segments, pending = make_segments("SRC-fixture", 8, [review], .5)
        self.assertEqual(len(segments), 1); self.assertEqual(len(pending), 1)
        self.assertTrue(segments[0]["needs_review"])

    def test_changed_inputs_and_models_refuse_stale_cache(self):
        self.analyze()
        backend = SyntheticQwen(); backend.provenance = {"model_sha256": "changed"}
        with self.assertRaises(ValueError): self.analyze(backend)
        with self.source.open("ab") as stream: stream.write(b"changed")
        with self.assertRaises(ValueError): export_segments(self.output, min_free_gb=0)

    def test_manual_review_before_export_and_no_overlapping_cuts(self):
        self.analyze(); review = self.root / "review.json"
        write_json(review, {"segments": [{"start_s": 0, "end_s": 4}, {"start_s": 4, "end_s": 8}]})
        value = export_segments(self.output, review_file=review, min_free_gb=0)
        self.assertTrue(value["manually_reviewed"]); self.assertEqual(len(value["videos"]), 2)
        with self.assertRaises(ValueError): export_segments(self.output, min_free_gb=0)
        write_json(review, {"segments": [{"start_s": 0, "end_s": 5}, {"start_s": 4, "end_s": 8}]})
        with self.assertRaises(ValueError): export_segments(self.output, review_file=review, min_free_gb=0)

    def test_clip_checksums_and_ids_flow_into_batch_commands(self):
        from contracts.execution import ExecutionSettings
        from workflows.batch import run_batch
        self.analyze(); manifest = export_segments(self.output, min_free_gb=0)
        args = {"clips_manifest": self.output / "clips_manifest.json", "dry_run": True, "options": {"match_id": "fixture"}}
        plan = run_batch(ExecutionSettings(), self.output, self.root / "results", **args)
        for row in manifest["videos"]:
            command = plan["commands"][row["clip_id"]]["tracking"]
            self.assertEqual(command[command.index("--video-id") + 1], row["clip_id"])
        with (self.output / manifest["videos"][0]["path"]).open("ab") as stream: stream.write(b"corrupt")
        with self.assertRaises(ValueError): run_batch(ExecutionSettings(), self.output, self.root / "results", **args)

    def test_identity_payload_contains_clip_track_and_original_times(self):
        run, player = self.library(); result = query_player_events(run, player)
        self.assertEqual(result["total"], 2)
        self.assertEqual([e["source_start_s"] for e in result["items"]], [.4, 4.4])
        self.assertEqual(len({e["track_uids"][0] for e in result["items"]}), 2)
        self.assertEqual(track_uid("clip1", 0), "clip1:T0000")
        self.assertTrue(all(e["player_id"] == player and e["raw_track_ids"] == [0] for e in result["items"]))

    def test_query_filters_and_real_original_and_clip_exports(self):
        run, player = self.library()
        self.assertEqual(query_player_events(run, player, clip_id="clip2", min_score=.5)["total"], 1)
        self.assertEqual(query_player_events(run, player, event="nonexistent")["total"], 0)
        output = self.root / "exports"
        result = export_player_events(run, player, output, clip_id="clip2", before_s=.1, after_s=.2, media_source="original")
        row = result["videos"][0]
        self.assertAlmostEqual(row["start_s"], 4.3); self.assertAlmostEqual(row["end_s"], 5.4)
        self.assertAlmostEqual(probe_video(row["path"])["duration_s"], 1.1, places=2)
        result = export_player_events(run, player, output, clip_id="clip2", scope="clip")
        self.assertAlmostEqual(probe_video(result["videos"][0]["path"])["duration_s"], 4, places=2)

    def test_export_uses_current_assignments_and_checks_revision(self):
        run, player = self.library(); before = query_player_events(run, player)
        with PlayerRegistry(run / "library/players.sqlite3", "synthetic-match") as registry:
            node = registry.snapshot()["people"][0]["members"][0]["node_id"]
            registry.review("detach", expected_revision=before["revision"], node_id=node)
        with self.assertRaises(RuntimeError):
            export_player_events(run, player, self.root / "exports", expected_revision=before["revision"])
        self.assertEqual(query_player_events(run, player)["total"], 1)

    def test_partly_unmapped_event_does_not_acquire_wrong_player(self):
        run = make_library_fixture(self.root, angles=(0,)); path = run / "clip1/action/events.jsonl"
        row = list(read_jsonl(path))[0]; row["raw_track_ids"] = [0, 999]; rows(path, [row])
        build_person_library(PersonLibraryOptions(run))
        with PlayerRegistry(run / "library/players.sqlite3", "synthetic-match") as registry:
            self.assertIsNone(registry.snapshot()["events"][0]["player_id"])

    def test_import_allows_generated_media_and_shows_segmentation(self):
        from web.backend.exports import segmentation_page
        from web.backend.imports import import_results
        from web.backend.settings import WebSettings
        from web.backend.store import ProjectStore
        run, _ = self.library(); self.analyze()
        batch = json.loads((run / "batch_manifest.json").read_text())
        batch["segmentation"] = {"plan_path": str(self.output / "segmentation.json"),
                                 "clips_manifest": str(self.output / "clips_manifest.json")}
        exported = []
        for row in batch["clips"]:
            path = run / row["name"] / "tracking/video_meta.json"; meta = json.loads(path.read_text())
            media = self.output / (row["name"] + ".mp4")
            shutil.copyfile(meta["input_path"], media); meta["input_path"] = str(media); write_json(path, meta)
            exported.append({"clip_id": row["clip_id"], "path": media.name})
        write_json(self.output / "clips_manifest.json", {"videos": exported})
        write_json(run / "batch_manifest.json", batch)
        settings = WebSettings(source_root=self.root / "data", import_root=self.root,
            data_root=self.root / "web-data", output_root=self.root / "web-output")
        project_store = ProjectStore(settings.data_root, settings.output_root)
        project = import_results(settings, project_store, run)
        self.assertEqual(project["videos"][0]["lineage"]["source_video_id"], "SRC-synthetic")
        self.assertEqual(segmentation_page(settings, project_store, project["project_id"])["total"], 2)

    def test_web_event_export_download_and_stale_identity_rejection(self):
        from fastapi.testclient import TestClient
        from web.backend import app as api_module
        from web.backend.imports import import_results
        from web.backend.settings import WebSettings
        from web.backend.store import ProjectStore
        # This uses a real FFmpeg encode and HTTP file response, no model.
        run, player = self.library()
        settings = WebSettings(source_root=self.root / "data", import_root=self.root / "outputs",
            data_root=self.root / "web-data", output_root=self.root / "web-output")
        project_store = ProjectStore(settings.data_root, settings.output_root)
        project = import_results(settings, project_store, run)
        query = query_player_events(run, player)
        request = {"player_id": player, "event_id": query["items"][0]["event_id"],
                   "expected_revision": query["revision"]}
        with patch.object(api_module, "settings", settings), patch.object(api_module, "store", project_store):
            with TestClient(api_module.app) as client:
                url = f"/api/projects/{project['project_id']}/person-library/export"
                response = client.post(url, json=request)
                self.assertEqual(response.status_code, 200, response.text)
                download = client.get(response.json()["download_url"])
                self.assertEqual(download.status_code, 200)
                self.assertEqual(download.headers["content-type"], "video/mp4")
                self.assertGreater(len(download.content), 1000)
                self.assertEqual(client.post(url, json={**request, "expected_revision": "stale"}).status_code, 409)

    def test_nonzero_video_pts_keep_source_relative_cut_time(self):
        shifted = self.root / "shifted.mp4"
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(self.source),
            "-c", "copy", "-output_ts_offset", "5", str(shifted)], check=True)
        meta = probe_video(shifted)
        self.assertAlmostEqual(meta["video_start_pts_s"], 5)
        self.assertAlmostEqual(meta["duration_s"], 8)
        target = self.root / "cut.mp4"
        cut_video(shifted, target, 2, 4)
        self.assertAlmostEqual(probe_video(target)["duration_s"], 2, places=2)
        import cv2
        import numpy as np
        original, cut = cv2.VideoCapture(str(self.source)), cv2.VideoCapture(str(target))
        try:
            original.set(cv2.CAP_PROP_POS_MSEC, 2000)
            ok1, expected = original.read(); ok2, actual = cut.read()
            self.assertTrue(ok1 and ok2)
            self.assertLess(float(np.abs(expected.astype(float) - actual).mean()), 5)
        finally:
            original.release(); cut.release()

    def test_segmented_batch_joins_real_clip_ids_to_one_player(self):
        from contracts.execution import ExecutionSettings
        from workflows.batch import run_batch
        from workflows.link_events import LinkEventsOptions, link_actions_to_people
        from workflows.events import build_temporal_events
        from adapters.mmaction2.temporal_events import TemporalAggregationOptions
        self.analyze(); exported = export_segments(self.output, min_free_gb=0)
        fixture = make_library_fixture(self.root / "identity-fixture", angles=(0,)) / "clip1"
        weight = self.root / "synthetic-model.pth"; weight.write_bytes(b"never loaded")
        import sys
        settings = replace(ExecutionSettings(), motip_python=Path(sys.executable), kpr_python=Path(sys.executable),
            action_python=Path(sys.executable), review_python=Path(sys.executable), qwen_python=Path(sys.executable),
            motip_checkpoint=weight, kpr_checkpoint=weight, action_checkpoint=weight,
            action_config=weight, action_label_map=weight)

        def execute(command, paths):
            module = command[2]
            clip_id = paths["output"].name
            if module == "cli.tracking":
                meta = probe_video(paths["source"])
                write_json(paths["video_meta"], {"video_id": clip_id, "input_path": str(paths["source"]),
                    "fps": meta["fps"], "frame_count": round(meta["duration_s"] * meta["fps"]),
                    "processed_frames": round(meta["duration_s"] * meta["fps"]), "width": 96, "height": 64})
                rows(paths["tracks"], [{"video_id": clip_id, "track_id": 0, "frame_idx": f,
                    "timestamp_s": f / 10, "bbox_xyxy": [0, 0, 96, 64], "det_score": .9, "category_id": 0}
                    for f in (0, 3, 6)])
            elif module == "cli.track_quality":
                paths["quality"].mkdir(exist_ok=True)
                shutil.copyfile(paths["tracks"], paths["stable_tracks"])
                write_json(paths["quality"] / "quality_summary.json", {})
                rows(paths["quality"] / "quality_tracks.jsonl", [])
            elif module == "cli.action":
                rows(paths["actions"], [{"video_id": clip_id, "track_id": 0, "frame_idx": f,
                    "timestamp_s": f / 10, "selected_actions": [{"label": "basketball_2point_shot", "score": .8}]}
                    for f in (0, 3, 6)])
                write_json(paths["action"] / "actions.summary.json", {"settings": {"predict_stepsize": 3}})
            elif module == "cli.aggregate_events":
                build_temporal_events(TemporalAggregationOptions(paths["actions"], paths["events"],
                    video_meta=paths["video_meta"], overwrite=True))
            elif module == "cli.identity":
                shutil.copytree(fixture / "identity_raw", paths["identity_raw"])
                write_json(paths["identity_raw"] / "identity_archive_manifest.json", {})
            elif module == "cli.resolve_identity":
                shutil.copytree(fixture / "identity", paths["identity"])
                rows(paths["identity_map"], [{"video_id": clip_id, "raw_track_id": 0, "person_id": "P0000"}])
                write_json(paths["identity"] / "resolution_summary.json", {"missing_archive_tracks": []})
            elif module == "cli.jersey_qwen":
                write_json(paths["identity"] / "jersey_summary.json", {"backend": "qwen_vl",
                    "provenance": {"model_sha256": "synthetic-only"}, "thresholds": {}})
                rows(paths["identity"] / "jersey_tracks.jsonl", [{"raw_track_id": 0, "number": "23",
                    "status": "candidate_consensus", "readings": []}])
            elif module == "cli.build_person_library":
                build_person_library(PersonLibraryOptions(paths["match_root"], match_id="integration",
                    clip_names=(clip_id,), use_jersey_evidence=True))
                write_json(paths["players"], {})
            elif module == "cli.link_events":
                link_actions_to_people(LinkEventsOptions(paths["actions"], paths["identity_map"],
                    paths["linked_actions"], player_map=paths["player_map"], clip_name=clip_id))
            elif module == "cli.render_results":
                paths["visualization"].mkdir(exist_ok=True)
                shutil.copyfile(paths["source"], paths["result"])
            elif module == "cli.prepare_track_review":
                write_json(paths["review"] / "index.json", {})
            else:
                self.fail(module)
            return 0

        run = self.root / "results"
        result = run_batch(settings, self.output, run, clips_manifest=self.output / "clips_manifest.json",
            options={"match_id": "integration", "min_free_gb": 0}, execute=execute)
        self.assertEqual(result["completed_clips"], 3, result)
        with PlayerRegistry(run / "library/players.sqlite3", "integration") as registry:
            snapshot = registry.snapshot()
        self.assertEqual(len(snapshot["people"]), 1)
        player = snapshot["people"][0]["player_id"]
        self.assertEqual(snapshot["people"][0]["jersey"]["number"], "23")
        query = query_player_events(run, player)
        self.assertEqual(query["total"], 3)
        self.assertEqual({e["clip_id"] for e in query["items"]}, {e["clip_id"] for e in exported["videos"]})
        for row in query["items"]:
            linked = list(read_jsonl(run / row["clip_id"] / "action/actions_with_identity.jsonl"))
            self.assertTrue(all(a["player_id"] == player and a["clip_id"] == row["clip_id"] for a in linked))

    def test_vfr_input_is_normalized_without_losing_source_offset(self):
        vfr = self.root / "vfr.mp4"
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(self.source),
            "-vf", "setpts='if(lt(N,40),N/(10*TB),(4+(N-40)/5)/TB)'", *frame_sync_args(mode="vfr"),
            "-c:v", "libx264", "-threads", "1", str(vfr)], check=True)
        target = self.root / "vfr-cut.mp4"
        metadata = cut_video(vfr, target, 6, 8)
        self.assertAlmostEqual(metadata["source_start_s"], 6)
        self.assertLessEqual(abs(probe_video(target)["duration_s"] - 2), 2 / metadata["fps"])
        import cv2
        import numpy as np
        expected_reader = cv2.VideoCapture(str(self.source))
        actual_reader = cv2.VideoCapture(str(target))
        try:
            expected_reader.set(cv2.CAP_PROP_POS_FRAMES, 50)
            ok1, expected = expected_reader.read(); ok2, actual = actual_reader.read()
            self.assertTrue(ok1 and ok2)
            self.assertLess(float(np.abs(expected.astype(float) - actual).mean()), 6)
        finally:
            expected_reader.release(); actual_reader.release()

    def ffmpeg_wrapper(self, option, *, broken_encode=False):
        import sys
        path = self.root / ("modern-ffmpeg" if option == "-fps_mode" else "legacy-ffmpeg")
        path.write_text(f"#!{sys.executable}\n" +
            "import os, sys\nargs = sys.argv[1:]\n" +
            f"if args == ['-hide_banner', '-h', 'full']:\n    print({option!r} + ' <> frame synchronization')\n    sys.exit(0)\n" +
            f"if {('-vsync' if option == '-fps_mode' else '-fps_mode')!r} in args:\n    sys.exit('Unrecognized option')\n" +
            ("sys.exit('synthetic decoder failure')\n" if broken_encode else
             f"os.execv({shutil.which('ffmpeg')!r}, [{shutil.which('ffmpeg')!r}] + args)\n"))
        path.chmod(0o755)
        return str(path)

    def test_modern_ffmpeg_without_vsync_can_cut_and_preflight(self):
        ffmpeg = self.ffmpeg_wrapper("-fps_mode")
        self.assertEqual(frame_sync_args(ffmpeg), ["-fps_mode", "cfr"])
        self.assertEqual(preflight_video(self.source, ffmpeg=ffmpeg)["status"], "passed")
        target = self.root / "modern.mp4"
        cut_video(self.source, target, 1, 3, ffmpeg=ffmpeg)
        self.assertAlmostEqual(probe_video(target)["duration_s"], 2, places=2)

    def test_legacy_ffmpeg_falls_back_to_vsync(self):
        ffmpeg = self.ffmpeg_wrapper("-vsync")
        self.assertEqual(frame_sync_args(ffmpeg), ["-vsync", "cfr"])
        self.assertEqual(preflight_video(self.source, ffmpeg=ffmpeg)["status"], "passed")

    def test_failed_media_preflight_never_loads_qwen_or_queries_gpu(self):
        import sys
        from types import SimpleNamespace
        from cli.segment_video import main as segment_main
        from pipeline.identity import qwen_backend
        ffmpeg = self.ffmpeg_wrapper("-fps_mode", broken_encode=True)
        settings = SimpleNamespace(ffmpeg=ffmpeg, ffprobe=shutil.which("ffprobe"),
                                   qwen_model_dir=self.root / "no-model")
        argv = ["segment_video", "--input", str(self.source), "--output-dir", str(self.output), "--stage", "analyze"]
        with patch.object(sys, "argv", argv), patch("cli.segment_video.ExecutionSettings", return_value=settings), \
                patch.object(qwen_backend, "QwenBackend") as model, patch.object(qwen_backend, "ensure_gpu_idle") as gpu:
            with self.assertRaisesRegex(RuntimeError, "synthetic decoder failure"):
                segment_main()
            model.assert_not_called(); gpu.assert_not_called()

    def test_invalid_omp_values_are_repaired_and_valid_lists_preserved(self):
        from contracts.execution import normalize_omp_threads
        for value in ("", "0", "-1", "None", "auto", "4,", "2.5", "99999999999999999999999"):
            env = {"OMP_NUM_THREADS": value}
            self.assertTrue(normalize_omp_threads(env))
            self.assertGreater(int(env["OMP_NUM_THREADS"]), 0)
        for value in ("1", "4", "4,2", " 4 , 2 "):
            env = {"OMP_NUM_THREADS": value}
            self.assertFalse(normalize_omp_threads(env))
            self.assertEqual(env["OMP_NUM_THREADS"], value)
        self.assertFalse(normalize_omp_threads({}))


def main():
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(MatchChecks))
    if not result.wasSuccessful(): raise SystemExit(1)
