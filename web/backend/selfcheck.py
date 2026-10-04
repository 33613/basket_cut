"""CPU-only checks for dashboard imports, event semantics and read-only safety."""

from __future__ import annotations

import base64
import io
import json
import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from contracts.schema import EventRecord
from web.backend.artifacts import collect_video_artifacts
from web.backend.imports import import_results, prepare_browser_media
from web.backend.inspection import build_inspection_bundle
from web.backend.runner import PipelineRunner
from web.backend.settings import WebSettings
from web.backend.store import ProjectStore


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(value) + "\n" for value in values), encoding="utf-8"
    )


def make_fixture(root: Path, count: int = 5) -> tuple[WebSettings, Path]:
    """Synthetic data for local UI/API QA, never passed off as model results."""
    settings = WebSettings(
        data_root=root / "data/web",
        source_root=root / "data",
        output_root=root / "outputs/web",
        import_root=root / "outputs",
    )
    settings.ensure_directories()
    run = root / "outputs/synthetic-ui-check"
    for index in range(1, count + 1):
        output = run / f"{index}_example(black)"
        source = settings.source_root / "clips" / f"{output.name}.mp4"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(b"synthetic-video")
        vid = output.name
        write_json(
            output / "tracking/video_meta.json",
            {
                "video_id": vid,
                "input_path": str(source),
                "fps": 25,
                "width": 1280,
                "height": 720,
                "processed_frames": 64,
                "frame_count": 64,
            },
        )
        write_json(
            output / "tracking/track_summary.json",
            {
                "track_count": 1,
                "tracks": [
                    {
                        "track_id": 0,
                        "start_frame": 0,
                        "end_frame": 63,
                        "observations": 64,
                        "duration_s": 2.56,
                        "observation_coverage": 1,
                        "mean_det_score": 0.93,
                    }
                ],
            },
        )
        write_jsonl(
            output / "tracking/tracks.jsonl",
            [
                {
                    "video_id": vid,
                    "frame_idx": 0,
                    "timestamp_s": 0,
                    "track_id": 0,
                    "bbox_xyxy": [100, 100, 200, 300],
                    "det_score": 0.93,
                    "category_id": 0,
                }
            ],
        )
        cover = "identity_media/identities/P0000/cover.png"
        crop = output / "identity" / cover
        crop.parent.mkdir(parents=True, exist_ok=True)
        crop.write_bytes(
            base64.b64decode(
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aJ9sAAAAASUVORK5CYII="
            )
        )
        write_jsonl(
            output / "identity/identities.jsonl",
            [
                {
                    "person_id": "P0000",
                    "raw_track_ids": [0],
                    "sample_count": 8,
                    "observation_count": 64,
                    "status": "unlabeled",
                    "mean_visible_parts": 4.5,
                    "within_track": {"mean_distance": 0.21},
                    "cover": {"crop_path": cover},
                    "exemplars": [{"crop_path": cover, "frame_idx": 0}],
                }
            ],
        )
        write_json(
            output / "identity/identity_archive_manifest.json",
            {"identity_resolution_mode": "archive_no_merge"},
        )
        write_json(
            output / "identity/kpr_summary.json",
            {"identity_count": 1, "sample_count": 8, "prompt_mode": "none"},
        )
        write_jsonl(
            output / "identity/identity_map.jsonl",
            [{"video_id": vid, "raw_track_id": 0, "person_id": "P0000", "status": "unlabeled"}],
        )
        actions = [
            {
                "video_id": vid,
                "track_id": 0,
                "person_id": "P0000",
                "frame_idx": 24,
                "timestamp_s": 0.96,
                "selected_actions": [],
                "action_candidates": [{"label": "basketball_pass", "score": 0.1}],
            }
        ]
        write_jsonl(output / "action/actions.jsonl", actions)
        write_jsonl(output / "action/actions_with_identity.jsonl", actions)
        events = [
            EventRecord(
                video_id=vid,
                event_id=f"{vid}:E{j}",
                identity_id="P0000",
                event=label,
                start=start,
                end=end,
                raw_score=score,
                start_frame=int(start * 25),
                end_frame=int(end * 25) - 1,
                raw_track_ids=(0,),
                support_count=3,
            ).to_dict()
            for j, (label, start, end, score) in enumerate(
                [
                    ("basketball_shoot", 0.4, 1.5, 0.68),
                    ("basketball_dribble", 1.6, 2.3, 0.41),
                ]
            )
        ]
        write_jsonl(output / "action/events.jsonl", events)
        (output / "result.mp4").write_bytes(b"synthetic-result")
    return settings, run


class DashboardSelfCheck(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.settings, self.run = make_fixture(Path(self.temporary.name))
        self.store = ProjectStore(self.settings.data_root, self.settings.output_root)

    def imported(self) -> dict:
        return import_results(self.settings, self.store, self.run, "Synthetic UI check")

    def test_five_clips_and_root_result_supported(self) -> None:
        manifest = self.imported()
        self.assertEqual(len(manifest["videos"]), 5)
        self.assertTrue(
            all(
                v["read_only"] and v["status"] == "completed"
                for v in manifest["videos"]
            )
        )
        self.assertTrue(
            collect_video_artifacts(manifest["videos"][0])["available"]["final_video"]
        )

    def test_single_clip_import(self) -> None:
        result = import_results(
            self.settings, self.store, self.run / "1_example(black)"
        )
        self.assertEqual(len(result["videos"]), 1)

    def test_batch_manifest_limits_import_and_preserves_failure_counts(self) -> None:
        write_json(self.run / 'batch_manifest.json', {
            'clips': [{'name': '1_example(black)'}, {'name': 'missing-video'}]})
        write_json(self.run / '1_example(black)/batch_status.json', {
            'status': 'failed', 'error': 'synthetic failure'})
        manifest = self.imported()
        self.assertEqual(len(manifest['videos']), 1)  # Ignore old leftovers outside the fixed selection.
        self.assertEqual(manifest['expected_video_count'], 2)
        self.assertEqual(manifest['videos'][0]['status'], 'failed')
        self.assertEqual(len(manifest['import_warnings']), 2)

    def test_stale_jersey_group_not_attached_to_reassigned_archive(self) -> None:
        video = self.imported()['videos'][0]
        write_jsonl(Path(video['output_dir']) / 'identity/jersey_people.jsonl', [
            {'person_id': 'P0000', 'raw_track_ids': [1], 'number': '15'}])
        artifacts = collect_video_artifacts(video)
        self.assertIsNone(artifacts['identity']['people'][0]['jersey'])
        self.assertIn('stale jersey', artifacts['warnings'][0])

    def test_idempotent_and_no_original_writes(self) -> None:
        before = {str(p): p.read_bytes() for p in self.run.rglob("*") if p.is_file()}
        first = self.imported()
        second = self.imported()
        self.assertEqual(first["project_id"], second["project_id"])
        self.assertEqual(len(self.store.list_projects()), 1)
        after = {str(p): p.read_bytes() for p in self.run.rglob("*") if p.is_file()}
        self.assertEqual(before, after)

    def test_import_path_boundary(self) -> None:
        with self.assertRaises(ValueError):
            import_results(self.settings, self.store, self.settings.source_root)

    def test_source_path_boundary_and_atomic_rejection(self) -> None:
        path = self.run / "1_example(black)/tracking/video_meta.json"
        value = json.loads(path.read_text())
        value["input_path"] = str(self.settings.import_root / "secret.mp4")
        write_json(path, value)
        with self.assertRaises(ValueError):
            self.imported()
        self.assertEqual(self.store.list_projects(), [])

    def test_symlink_escape_rejected(self) -> None:
        secret = Path(self.temporary.name) / "outside.json"
        write_json(secret, {"secret": True})
        (self.run / "1_example(black)/leak.json").symlink_to(secret)
        with self.assertRaises(ValueError):
            self.imported()

    def test_empty_events_are_zero_not_candidates(self) -> None:
        video = self.imported()["videos"][0]
        write_jsonl(Path(video["output_dir"]) / "action/events.jsonl", [])
        artifacts = collect_video_artifacts(video)
        self.assertTrue(artifacts["available"]["events"])
        self.assertEqual(artifacts["action"]["index"]["event_count"], 0)
        self.assertEqual(artifacts["action"]["points"]["event_count"], 1)

    def test_interval_index_and_identity_cover(self) -> None:
        artifacts = collect_video_artifacts(self.imported()["videos"][0])
        self.assertEqual(artifacts["action"]["index"]["event_count"], 2)
        person = artifacts["action"]["index"]["people"][0]
        self.assertEqual(person["events"][0]["start"], 0.4)
        self.assertEqual(person["identity"]["person_id"], "P0000")

    def test_invalid_or_wrong_video_event_is_reported(self) -> None:
        video = self.imported()["videos"][0]
        path = Path(video["output_dir"]) / "action/events.jsonl"
        values = [json.loads(line) for line in path.read_text().splitlines()]
        values[0]["video_id"] = "wrong-video"
        write_jsonl(path, values)
        with path.open("a") as handle:
            handle.write("not-json\n")
        artifacts = collect_video_artifacts(video)
        self.assertEqual(len(artifacts["warnings"]), 2)
        self.assertEqual(artifacts["action"]["index"]["event_count"], 1)

    def test_read_only_cannot_submit(self) -> None:
        manifest = self.imported()
        runner = PipelineRunner(self.settings, self.store)
        self.addCleanup(runner.shutdown)
        with self.assertRaisesRegex(ValueError, "read-only"):
            runner.submit(
                manifest["project_id"],
                manifest["videos"][0]["video_id"],
                target="full",
                options={},
                force=True,
            )

    def test_browser_transcode_missing_ffmpeg_is_clear(self) -> None:
        with (
            patch("web.backend.imports.shutil.which", return_value=None),
            self.assertRaisesRegex(FileNotFoundError, "ffmpeg"),
        ):
            prepare_browser_media(self.settings, self.imported())

    def test_raw_and_resolved_archives_are_separate(self):
        video = self.imported()['videos'][0]
        output = Path(video['output_dir'])
        write_jsonl(output / 'identity_raw/identities.jsonl', [
            {'person_id': 'P0', 'raw_track_ids': [0]}, {'person_id': 'P1', 'raw_track_ids': [1]}])
        result = collect_video_artifacts(video)['identity']
        self.assertEqual(result['raw_count'], 2)
        self.assertEqual(result['resolved_count'], 1)
        self.assertEqual(result['reduction'], 1)
        self.assertEqual(result['raw_media_prefix'], 'identity_raw')

    def test_evidence_cache_does_not_write_original_results(self):
        from web.backend.evidence import EvidenceJobs
        from web.backend.reviews import collect_review
        video = self.imported()['videos'][0]
        project = self.imported()['project_id']
        before = {str(p): p.read_bytes() for p in self.run.rglob('*') if p.is_file()}
        jobs = EvidenceJobs(self.settings, self.store)
        self.addCleanup(lambda: jobs.executor.shutdown(wait=True))
        def fake(command, **kwargs):
            destination = Path(command[command.index('--output-dir') + 1])
            from analysis.evaluation.track_review import review_index
            from web.backend.reviews import review_paths
            paths = review_paths(video)
            index = review_index(paths['tracks'], paths['video_meta'])
            write_json(destination / 'index.json', index)
            from subprocess import CompletedProcess
            return CompletedProcess(command, 0)
        with patch('web.backend.evidence.subprocess.run', side_effect=fake):
            jobs.submit(project, video)
            jobs.futures[(project, video['video_id'])].result(timeout=5)
        result = collect_review(self.store, project, video)
        self.assertTrue(result['evidence_available'])
        self.assertEqual(result['evidence_origin'], 'cache')
        self.assertEqual(result['evidence_job']['status'], 'completed')
        self.assertEqual(before, {str(p): p.read_bytes() for p in self.run.rglob('*') if p.is_file()})

    def test_merged_archive_without_events_stays_in_project_index(self) -> None:
        from web.backend.artifacts import project_people_index
        manifest = self.imported()
        output = Path(manifest["videos"][0]["output_dir"])
        (output / "action/events.jsonl").write_text("", encoding="utf-8")
        entries = project_people_index(manifest)["entries"]
        self.assertEqual(len(entries), 5)
        self.assertEqual(entries[0]["event_count"], 0)

    def test_changed_identity_map_does_not_show_stale_event_owner(self) -> None:
        video = self.imported()["videos"][0]
        output = Path(video["output_dir"])
        write_jsonl(output / "identity/identity_map.jsonl", [
            {"video_id": output.name, "raw_track_id": 0, "person_id": "P0099"}])
        artifacts = collect_video_artifacts(video)
        self.assertEqual(artifacts["action"]["index"]["event_count"], 0)
        self.assertIn("ownership", artifacts["warnings"][0])

    def test_browser_transcode_cache_only_and_repeat_skip(self) -> None:
        manifest = self.imported()
        before = {str(p): p.read_bytes() for p in self.run.rglob("*") if p.is_file()}

        def fake_convert(command, **kwargs):
            Path(command[-1]).write_bytes(b"h264-preview")
            from subprocess import CompletedProcess

            return CompletedProcess(command, 0)

        with (
            patch("web.backend.imports.shutil.which", return_value="ffmpeg"),
            patch(
                "web.backend.imports.subprocess.run", side_effect=fake_convert
            ) as converted,
        ):
            prepare_browser_media(self.settings, manifest)
            self.assertEqual(
                converted.call_count, 10
            )  # source + final; no tracking preview fixture
            second = prepare_browser_media(self.settings, manifest)
            self.assertTrue(all(item["cached"] for item in second))
            self.assertEqual(converted.call_count, 10)
        self.assertEqual(
            before, {str(p): p.read_bytes() for p in self.run.rglob("*") if p.is_file()}
        )

    def test_diagnostic_bundle_no_video_or_tensor(self) -> None:
        bundle = build_inspection_bundle(self.imported()["videos"][0])
        with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
            self.assertIn("action/events.jsonl", archive.namelist())
            self.assertIn("inspection.json", archive.namelist())
            self.assertTrue(
                any(name.endswith("cover.png") for name in archive.namelist())
            )
            self.assertFalse(
                any(
                    name.endswith((".mp4", ".npz", ".pth"))
                    for name in archive.namelist()
                )
            )

    def test_web_runner_has_aggregation_stage(self) -> None:
        runner = PipelineRunner(self.settings, self.store)
        self.addCleanup(runner.shutdown)
        paths = runner._paths(self.imported()["videos"][0])
        plan = runner._plan("full", paths, True)
        self.assertLess(plan.index("link"), plan.index("aggregate"))
        self.assertLess(plan.index("aggregate"), plan.index("render"))
        self.assertLess(plan.index("quality"), plan.index("identity"))
        self.assertLess(plan.index("resolution"), plan.index("link"))
        numbered = runner._plan('full', paths, True, jersey=True)
        self.assertLess(numbered.index('resolution'), numbered.index('jersey'))
        self.assertLess(numbered.index('jersey'), numbered.index('action'))
        command = runner._command("resolution", paths, {})
        self.assertNotIn("--max-distance", command)
        self.assertIn(
            "--max-distance",
            runner._command("resolution", paths, {"identity_merge_distance": 0.2}),
        )

    def test_new_artifacts_and_quality_reach_dashboard(self) -> None:
        video = self.imported()["videos"][0]
        output = Path(video["output_dir"])
        write_json(
            output / "quality/quality_summary.json",
            {"raw_track_count": 2, "retained_track_count": 1},
        )
        write_jsonl(
            output / "quality/quality_tracks.jsonl",
            [{"track_id": 0, "status": "accepted"}],
        )
        write_json(
            output / "identity/resolution_summary.json",
            {"identity_count": 1, "human_confirmed_people": 1},
        )
        artifacts = collect_video_artifacts(video)
        self.assertEqual(artifacts["tracking"]["quality"]["raw_track_count"], 2)
        self.assertEqual(artifacts["identity"]["summary"]["human_confirmed_people"], 1)
        self.assertIn(
            "quality/quality_summary.json",
            zipfile.ZipFile(io.BytesIO(build_inspection_bundle(video))).namelist(),
        )

    def test_empty_quality_result_does_not_invoke_models(self) -> None:
        from adapters.mmaction2.temporal_events import (
            TemporalAggregationOptions,
            aggregate_action_points,
        )
        from pipeline.identity.resolution import ResolutionOptions, resolve_identities
        from workflows.link_events import LinkEventsOptions, link_actions_to_people

        video = self.imported()["videos"][0]
        paths = PipelineRunner._paths(video)
        paths["quality"].mkdir()
        paths["stable_tracks"].write_text("")
        self.assertTrue(PipelineRunner._write_empty_model_artifacts("identity", paths))
        self.assertTrue(PipelineRunner._write_empty_model_artifacts("action", paths))
        resolve_identities(
            ResolutionOptions(
                paths["stable_tracks"],
                paths["video_meta"],
                paths["identity_raw"],
                paths["identity"],
                overwrite=True,
            )
        )
        link_actions_to_people(
            LinkEventsOptions(
                paths["actions"],
                paths["identity_map"],
                paths["linked_actions"],
                overwrite=True,
            )
        )
        result = aggregate_action_points(
            TemporalAggregationOptions(
                paths["linked_actions"],
                paths["events"],
                video_meta=paths["video_meta"],
                overwrite=True,
            )
        )
        self.assertEqual(result["event_count"], 0)
        self.assertFalse(
            json.loads((paths["action"] / "actions.summary.json").read_text())[
                "model_invoked"
            ]
        )

    def test_http_import_detail_raw_media_export_and_read_only(self) -> None:
        from fastapi.testclient import TestClient

        env = {
            "BASKET_WEB_DATA_ROOT": str(self.settings.data_root),
            "BASKET_WEB_OUTPUT_ROOT": str(self.settings.output_root),
            "BASKET_WEB_IMPORT_ROOT": str(self.settings.import_root),
            "BASKET_WEB_SOURCE_ROOT": str(self.settings.source_root),
        }
        with patch.dict(os.environ, env):
            from web.backend import app as module
        with (
            patch.object(module, "settings", self.settings),
            patch.object(module, "store", self.store),
        ):
            runner = PipelineRunner(self.settings, self.store)
            with (
                patch.object(module, "runner", runner),
                TestClient(module.app) as client,
            ):
                response = client.post(
                    "/api/import-results", json={"result_dir": str(self.run)}
                )
                self.assertEqual(response.status_code, 200)
                manifest = response.json()
                base = f"/api/projects/{manifest['project_id']}/videos/{manifest['videos'][0]['video_id']}"
                detail = client.get(base).json()
                self.assertEqual(
                    detail["artifacts"]["action"]["index"]["event_count"], 2
                )
                self.assertEqual(
                    client.get(detail["artifacts"]["media"]["final"]).status_code, 200
                )
                cover = detail["artifacts"]["identity"]["people"][0]["cover"][
                    "crop_url"
                ]
                self.assertEqual(client.get(cover).status_code, 200)
                self.assertEqual(
                    len(client.get(base + "/records/events").json()["records"]), 2
                )
                self.assertEqual(client.get(base + "/inspection.zip").status_code, 200)
                # Read-only imports can have a separate persistent audit sidecar.
                before = (Path(manifest["videos"][0]["output_dir"]) / "tracking/tracks.jsonl").read_bytes()
                audit = client.get(base + "/track-review")
                self.assertEqual(audit.status_code, 200)
                value = audit.json()
                observations = client.get(base + "/track-review/0/observations")
                self.assertEqual(observations.status_code, 200)
                self.assertEqual(observations.json()["observations"][0]["frame_idx"], 0)
                self.assertEqual(client.get(base + "/track-review/999/observations").status_code, 404)
                value["review"]["tracks"] = [{"raw_track_id": 0, "verdict": "pure", "scope": "full_track", "identity_label": "white#15"}]
                saved = client.put(base + "/track-review", json={"review": value["review"], "expected_revision": None})
                self.assertEqual(saved.status_code, 200)
                self.assertEqual(client.get(base + "/track-review").json()["metrics"]["scopes"]["full_track"]["raw"]["pure"], 1)
                self.assertEqual(client.put(base + "/track-review", json={"review": value["review"]}).status_code, 409)
                self.assertEqual((Path(manifest["videos"][0]["output_dir"]) / "tracking/tracks.jsonl").read_bytes(), before)
                bundle = client.get(f"/api/projects/{manifest['project_id']}/track-reviews.zip")
                self.assertEqual(bundle.status_code, 200)
                from analysis.evaluation.track_review import evaluate_manifest
                with zipfile.ZipFile(io.BytesIO(bundle.content)) as archive:
                    target = self.settings.output_root / "audit-export"
                    archive.extractall(target)
                metrics = evaluate_manifest(target / "manifest.json", target / "metrics.json")
                self.assertTrue(metrics["complete"])
                self.assertEqual(metrics["expected_clips"], 5)
                self.assertEqual(metrics["completed_subset"]["scopes"]["full_track"]["raw"]["pure"], 1)
                self.assertEqual(
                    client.post(base + "/run", json={"target": "full"}).status_code, 409
                )
                self.assertEqual(
                    client.post(
                        f"/api/projects/{manifest['project_id']}/videos",
                        files={"files": ("new.mp4", b"video", "video/mp4")},
                    ).status_code,
                    409,
                )
                self.assertEqual(
                    client.post(
                        "/api/import-results",
                        json={"result_dir": str(self.settings.source_root)},
                    ).status_code,
                    400,
                )
                cache = Path(manifest["videos"][0]["media_cache_dir"])
                cache.mkdir(parents=True)
                (cache / "final.mp4").write_bytes(b"0123456789")
                response = client.get(
                    base + "/media/final", headers={"Range": "bytes=2-5"}
                )
                self.assertEqual(response.status_code, 206)
                self.assertEqual(response.content, b"2345")

    def many_tracks(self, video: dict, count: int = 17) -> None:
        output = Path(video['output_dir'])
        base = json.loads((output / 'tracking/tracks.jsonl').read_text().splitlines()[0])
        write_jsonl(output / 'tracking/tracks.jsonl', [{**base, 'track_id': tid} for tid in range(count)])

    def test_review_page_limits_cards_without_promoting_unreviewed(self) -> None:
        from web.backend.catalog import catalog, review_page
        video = self.imported()['videos'][0]
        self.many_tracks(video)
        data = catalog.review(self.store, self.imported()['project_id'], video)
        page = review_page(data, offset=6, limit=6)
        self.assertEqual([track['raw_track_id'] for track in page['index']['tracks']], list(range(6, 12)))
        self.assertEqual(page['pagination']['total'], 17)
        self.assertEqual(page['review']['tracks'], [])
        self.assertEqual(page['metrics']['scopes']['full_track']['raw']['assessed_tracks'], 0)

    def test_partial_review_save_preserves_hidden_pages_and_revision(self) -> None:
        from web.backend.reviews import collect_review, save_review
        video = self.imported()['videos'][0]
        project = self.imported()['project_id']
        self.many_tracks(video)
        data = collect_review(self.store, project, video)
        first = {**data['review'], 'tracks': [{'raw_track_id': 0, 'verdict': 'pure', 'scope': 'full_track', 'identity_label': 'white#15'}]}
        saved = save_review(self.store, project, video, first, None, partial=True)
        next_page = {**data['review'], 'tracks': [{'raw_track_id': 12, 'verdict': 'uncertain', 'scope': 'full_track'}]}
        second = save_review(self.store, project, video, next_page, saved['revision'], partial=True)
        self.assertEqual([row['raw_track_id'] for row in second['review']['tracks']], [0, 12])
        self.assertEqual(second['metrics']['scopes']['full_track']['raw']['assessed_tracks'], 1)
        self.assertEqual(second['metrics']['scopes']['full_track']['raw']['uncertain'], 1)
        with self.assertRaisesRegex(RuntimeError, 'another window'):
            save_review(self.store, project, video, first, saved['revision'], partial=True)

    def test_review_catalog_invalidates_after_save_and_changed_artifacts(self) -> None:
        from web.backend.catalog import ReviewCatalog
        from web.backend.reviews import save_review
        video = self.imported()['videos'][0]
        project = self.imported()['project_id']
        index = ReviewCatalog(capacity=2)
        original = index.review(self.store, project, video)
        value = {**original['review'], 'tracks': [{'raw_track_id': 0, 'verdict': 'pure', 'scope': 'full_track'}]}
        saved = save_review(self.store, project, video, value, None, partial=True)
        self.assertEqual(index.review(self.store, project, video)['revision'], saved['revision'])
        path = Path(video['output_dir']) / 'tracking/tracks.jsonl'
        record = json.loads(path.read_text().splitlines()[0])
        record['bbox_xyxy'] = [110, 100, 210, 300]
        write_jsonl(path, [record])
        changed = index.review(self.store, project, video)
        self.assertTrue(changed['stale_review'])
        self.assertEqual(changed['review']['tracks'], [])

    def test_queue_uncertain_and_sampled_remain_incomplete(self) -> None:
        from web.backend.catalog import video_catalog, next_review_video
        from web.backend.reviews import collect_review, save_review
        manifest = self.imported()
        for number, video in enumerate(manifest['videos'][:3]):
            data = collect_review(self.store, manifest['project_id'], video)
            row = {'raw_track_id': 0, 'verdict': ['pure', 'pure', 'uncertain'][number],
                   'scope': ['full_track', 'sampled', 'full_track'][number]}
            save_review(self.store, manifest['project_id'], video, {**data['review'], 'tracks':[row]}, None)
        result = video_catalog(self.store, manifest, limit=2)
        self.assertEqual(len(result['items']), 2)
        self.assertEqual(result['summary']['review_statuses']['complete'], 1)
        self.assertEqual(result['summary']['full_assessed'], 1)
        self.assertEqual(result['summary']['sampled_tracks'], 1)
        self.assertEqual(result['summary']['uncertain_tracks'], 1)
        self.assertEqual(result['summary']['remaining_tracks'], 4)
        remaining = video_catalog(self.store, manifest, status='remaining', limit=20)
        self.assertEqual(remaining['total'], 4)
        self.assertEqual(next_review_video(self.store, manifest, manifest['videos'][0]['video_id'])['video_id'], manifest['videos'][1]['video_id'])
        self.assertEqual(next_review_video(self.store, manifest, status='uncertain')['video_id'], manifest['videos'][2]['video_id'])
        self.assertEqual(video_catalog(self.store, manifest, q='2_example')['total'], 1)

    def test_queue_missing_results_not_counted_complete_and_empty_not_accurate(self) -> None:
        from web.backend.catalog import video_catalog
        manifest = self.imported()
        manifest['expected_video_count'] = 7
        output = Path(manifest['videos'][0]['output_dir'])
        write_jsonl(output / 'tracking/tracks.jsonl', [])
        write_jsonl(output / 'identity/identity_map.jsonl', [])
        result = video_catalog(self.store, manifest)
        self.assertEqual(result['summary']['missing_clips'], 2)
        self.assertEqual(result['items'][0]['review_status'], 'empty')
        self.assertNotIn('complete', result['summary']['review_statuses'])
        self.assertEqual(video_catalog(self.store, manifest, status='failed')['total'], 1)

    def test_unreviewed_saved_row_does_not_count_as_partial_judgment(self) -> None:
        from web.backend.catalog import video_catalog
        from web.backend.reviews import collect_review, save_review
        manifest = self.imported()
        video = manifest['videos'][0]
        data = collect_review(self.store, manifest['project_id'], video)
        save_review(self.store, manifest['project_id'], video, {**data['review'], 'tracks': [
            {'raw_track_id': 0, 'verdict': 'unreviewed', 'scope': 'sampled', 'note': 'needs check'}]}, None)
        self.assertEqual(video_catalog(self.store, manifest)['items'][0]['review_status'], 'unreviewed')

    def test_hundred_clip_queue_is_paged_and_warm_cache_skips_track_rescan(self) -> None:
        from web.backend.catalog import video_catalog, material_catalog
        settings, run = make_fixture(Path(self.temporary.name) / 'hundred', count=100)
        store = ProjectStore(settings.data_root, settings.output_root)
        manifest = import_results(settings, store, run, 'Synthetic 100-clip check')
        first = video_catalog(store, manifest)
        self.assertEqual(len(first['items']), 12)
        self.assertEqual(first['total'], 100)
        self.assertEqual(first['summary']['available_clips'], 100)
        self.assertEqual(first['summary']['full_assessed'], 0)
        with patch('web.backend.catalog.collect_review', side_effect=AssertionError('warm queue must use cache')):
            second = video_catalog(store, manifest, offset=96)
        self.assertEqual(len(second['items']), 4)
        self.assertNotEqual(first['items'][0]['video_id'], second['items'][0]['video_id'])
        gallery = material_catalog(manifest)
        self.assertEqual(len(gallery['items']), 18)
        self.assertEqual(gallery['total'], 100)

    def test_material_catalog_no_event_load_and_clip_local_scope(self) -> None:
        from web.backend.catalog import material_catalog
        manifest = self.imported()
        output = Path(manifest['videos'][0]['output_dir'])
        write_jsonl(output / 'identity_raw/identities.jsonl', [
            {'person_id': 'R0000', 'raw_track_ids': [0], 'cover': {'crop_path':'cover.png'}}])
        before = {str(path): path.read_bytes() for path in self.run.rglob('*') if path.is_file()}
        with patch('web.backend.artifacts.temporal_event_index', side_effect=AssertionError('should not load events')):
            result = material_catalog(manifest, limit=2)
        self.assertEqual(len(result['items']), 2)
        self.assertEqual(result['total'], 5)
        self.assertEqual(result['identity_scope'], 'clip_local')
        self.assertEqual(material_catalog(manifest, view='raw')['total'], 1)
        self.assertEqual(material_catalog(manifest, q='1_example')['total'], 1)
        self.assertEqual(before, {str(path): path.read_bytes() for path in self.run.rglob('*') if path.is_file()})

    def test_http_paginated_review_patch_and_compact_project(self) -> None:
        from fastapi.testclient import TestClient
        from web.backend import app as module
        manifest = self.imported()
        video = manifest['videos'][0]
        self.many_tracks(video)
        runner = PipelineRunner(self.settings, self.store)
        self.addCleanup(runner.shutdown)
        with patch.object(module, 'store', self.store), patch.object(module, 'runner', runner), TestClient(module.app) as client:
            project = f"/api/projects/{manifest['project_id']}"
            base = project + '/videos/' + video['video_id']
            with patch.object(module, 'project_people_index', side_effect=AssertionError('expensive project scan')):
                self.assertNotIn('people_index', client.get(project + '?include_people=false').json())
            queue = client.get(project + '/review-queue?limit=2')
            self.assertEqual(queue.status_code, 200)
            self.assertEqual(len(queue.json()['items']), 2)
            self.assertEqual(client.get(project + '/materials?limit=2').json()['total'], 5)
            page = client.get(base + '/track-review?offset=6&limit=6').json()
            self.assertEqual(len(page['index']['tracks']), 6)
            request = {'review': {**page['review'], 'tracks':[{'raw_track_id': 6, 'verdict':'pure', 'scope':'sampled'}]}, 'expected_revision':None}
            saved = client.patch(base + '/track-review', json=request)
            self.assertEqual(saved.status_code, 200)
            self.assertNotIn('index', saved.json())
            self.assertEqual(client.patch(base + '/track-review', json=request).status_code, 409)
            self.assertEqual(client.get(base + '/track-review?status=remaining&limit=6').json()['pagination']['total'], 17)
            self.assertEqual(client.get(project + '/review-queue?limit=1000').status_code, 422)
            self.assertEqual(client.get(base + '/track-review?limit=25').status_code, 422)


def main() -> None:
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(DashboardSelfCheck)
    )
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
