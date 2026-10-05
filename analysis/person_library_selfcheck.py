"""Synthetic CPU checks. Fixtures never represent real tracking/ReID accuracy."""

from dataclasses import replace
import base64
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
import contextlib
import sys
from unittest.mock import patch

import numpy as np

from contracts.schema import EventRecord, read_jsonl, write_json
from pipeline.identity.cross_clip import group_nodes, part_distances
from workflows.person_library import PersonLibraryOptions, build_person_library, library_is_current


def rows(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(value) + "\n" for value in values))


def make_library_fixture(root, angles=(0, 0.05, 1.4), *, extra_same_clip=False):
    run = root / "outputs" / "synthetic-match"
    for index, angle in enumerate(angles):
        name = f"clip{index + 1}"
        clip = run / name
        source = root / "data" / f"{name}.mp4"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(b"synthetic-video")
        write_json(clip / "batch_status.json", {"status": "completed"})
        write_json(clip / "tracking/video_meta.json", {"video_id": name, "input_path": str(source),
                   "fps": 25, "width": 640, "height": 360, "frame_count": 75, "processed_frames": 75})
        count = 2 if index == 0 and extra_same_clip else 1
        people, vectors = [], []
        for tid in range(count):
            path = "media/cover.png" if tid == 0 else "media/other.png"
            person = {"person_id": f"P{tid:04d}", "raw_track_ids": [tid], "sample_count": 4,
                      "status": "unverified", "observation_count": 75, "review_reasons": [],
                      "within_track": {"max_distance": 0.1}, "cover": {"crop_path": path,
                      "archive_quality_score": 0.8}, "exemplars": [{"crop_path": path, "frame_idx": 0}]}
            people.append(person)
            vector = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)
            vectors.append(np.stack([vector, vector]))
            for prefix in ("identity", "identity_raw"):
                destination = clip / prefix / path
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(base64.b64decode(
                    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aJ9sAAAAASUVORK5CYII="))
        rows(clip / "identity/identities.jsonl", people)
        rows(clip / "identity_raw/identities.jsonl", people)
        write_json(clip / "identity/resolution_summary.json", {"identity_count": count, "missing_archive_tracks": []})
        write_json(clip / "identity/identity_archive_manifest.json", {"video_id": name})
        write_json(clip / "identity_raw/kpr_summary.json", {"feature_contract": {
            "checkpoint_sha256": "synthetic-checkpoint", "prompt_mode": "none", "test_embeddings": ["synthetic"]}})
        np.savez_compressed(clip / "identity_raw/kpr_track_prototypes.npz",
                            embeddings=np.stack(vectors), visibility_scores=np.ones((count, 2), dtype=np.float32),
                            track_ids=np.arange(count))
        rows(clip / "tracking/tracks.jsonl", [{"video_id": name, "track_id": tid, "frame_idx": 0,
            "timestamp_s": 0, "bbox_xyxy": [tid * 100, 0, tid * 100 + 50, 100],
            "det_score": 0.9, "category_id": 0} for tid in range(count)])
        rows(clip / "action/actions.jsonl", [])
        rows(clip / "action/events.jsonl", [EventRecord(name, f"E{tid}", f"P{tid:04d}",
             "basketball_2point_shot", 0.4, 1.2, 0.7, 10, 29, (tid,), 3).to_dict() for tid in range(count)])
    write_json(run / "batch_manifest.json", {"target": "full", "clips": [
        {"name": f"clip{index + 1}"} for index in range(len(angles))]})
    return run


class PersonLibraryChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="basket-library-check-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = make_library_fixture(self.root)
        self.options = PersonLibraryOptions(self.run, max_distance=0.2)

    def people(self):
        return list(read_jsonl(self.run / "library/people.jsonl"))

    def test_same_local_id_different_clips_and_global_events(self):
        before = {str(path): path.read_bytes() for path in self.run.rglob("*") if path.is_file()}
        result = build_person_library(self.options)
        self.assertEqual((result["expected_clips"], result["global_person_count"], result["event_count"]), (3, 2, 3))
        mapping = list(read_jsonl(self.run / "library/identity_map.jsonl"))
        self.assertEqual(len({row["node_id"] for row in mapping}), 3)
        events = list(read_jsonl(self.run / "library/events.jsonl"))
        self.assertEqual(len({event["event_id"] for event in events}), 3)
        self.assertEqual({event["local_person_id"] for event in events}, {"P0000"})
        self.assertEqual({event["start"] for event in events}, {0.4})
        self.assertTrue(all(event["id"] == event["global_person_id"] for event in events))
        self.assertTrue(all(Path(path).read_bytes() == value for path, value in before.items()))

    def test_part_distance_matches_visibility_weighted_reference(self):
        random = np.random.default_rng(8)
        vectors = random.normal(size=(7, 3, 4)).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=-1, keepdims=True)
        visibility = random.random((7, 3)).astype(np.float32)
        visibility[0] = 0
        matrix = part_distances(vectors, visibility, min_common_parts=2, block_size=2)
        for a in range(7):
            for b in range(7):
                weight = np.sqrt(visibility[a] * visibility[b])
                if np.count_nonzero(weight) < 2:
                    self.assertTrue(np.isinf(matrix[a, b]))
                else:
                    expected = np.sum(np.linalg.norm(vectors[a] - vectors[b], axis=-1) * weight) / weight.sum() / 2
                    self.assertAlmostEqual(float(matrix[a, b]), float(expected), delta=0.0003)

    def test_complete_link_prevents_transitive_identity_chain(self):
        nodes = [{"node_id": str(i), "clip_name": str(i), "feature_indices": [i]} for i in range(3)]
        matrix = np.array([[0, .1, .4], [.1, 0, .1], [.4, .1, 0]])
        groups, _, _ = group_nodes(nodes, matrix, max_distance=.2)
        self.assertEqual(sorted(len(group["members"]) for group in groups), [1, 2])

    def test_same_clip_cannot_link_is_transitive_across_groups(self):
        nodes = [{"node_id": str(i), "clip_name": "same" if i != 1 else "other",
                  "feature_indices": [i]} for i in range(3)]
        groups, _, _ = group_nodes(nodes, np.zeros((3, 3)), max_distance=.2)
        self.assertEqual(len(groups), 2)
        for group in groups:
            names = [n["clip_name"] for n in group["members"]]
            self.assertEqual(len(names), len(set(names)))

    def test_mixed_archive_abstains_and_disabled_threshold_does_not_merge(self):
        rows(self.run / "clip1/identity/identities.jsonl", [{**next(read_jsonl(self.run / "clip1/identity/identities.jsonl")),
                                                          "status": "needs_review"}])
        result = build_person_library(self.options)
        self.assertEqual(result["held_archive_count"], 1)
        self.assertEqual(result["global_person_count"], 3)
        disabled = build_person_library(replace(self.options, max_distance=None, overwrite=True))
        self.assertEqual(disabled["global_person_count"], 3)

    def test_incompatible_checkpoints_rejected_before_output(self):
        write_json(self.run / "clip2/identity_raw/kpr_summary.json", {"feature_contract": {"checkpoint_sha256": "other"}})
        with self.assertRaisesRegex(ValueError, "Incompatible"):
            build_person_library(self.options)
        self.assertFalse((self.run / "library").exists())

    def test_legacy_requires_explicit_opt_in_and_malformed_features_fail(self):
        for index in range(1, 4):
            write_json(self.run / f"clip{index}/identity_raw/kpr_summary.json",
                       {"checkpoint": "weights/synthetic.pth", "prompt_mode": "none"})
        with self.assertRaisesRegex(ValueError, "legacy KPR metadata"):
            build_person_library(self.options)
        summary = build_person_library(replace(self.options, allow_legacy_features=True))
        self.assertTrue(any("not a checkpoint checksum" in warning for warning in summary["warnings"]))
        np.savez_compressed(self.run / "clip1/identity_raw/kpr_track_prototypes.npz",
                            embeddings=np.ones((1, 2, 2)), visibility_scores=np.ones((1, 2)), track_ids=[0])
        with self.assertRaisesRegex(ValueError, "L2-normalized"):
            build_person_library(replace(self.options, allow_legacy_features=True, overwrite=True))

    def test_output_symlink_and_overlapping_directory_are_rejected(self):
        destination = self.root / "keep"
        destination.mkdir()
        marker = destination / "preserved.txt"
        marker.write_text("preserved")
        (self.run / "library").symlink_to(destination, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink"):
            build_person_library(replace(self.options, overwrite=True))
        self.assertEqual(marker.read_text(), "preserved")
        with self.assertRaisesRegex(ValueError, "overwrite a run"):
            build_person_library(replace(self.options, output_dir=self.run, overwrite=True))

    def test_failed_or_missing_clip_is_not_silently_omitted(self):
        write_json(self.run / "clip2/batch_status.json", {"status": "failed"})
        with self.assertRaisesRegex(ValueError, "not complete"):
            build_person_library(self.options)
        write_json(self.run / "clip2/batch_status.json", {"status": "completed"})
        (self.run / "clip2/action/events.jsonl").unlink()
        with self.assertRaises(FileNotFoundError):
            build_person_library(self.options)

    def test_stale_source_detection_ignores_operational_resume_timestamps(self):
        build_person_library(self.options)
        manifest = json.loads((self.run / "library/library_manifest.json").read_text())
        write_json(self.run / "clip1/batch_status.json", {"status": "completed", "retry": True})
        value = json.loads((self.run / "batch_manifest.json").read_text())
        write_json(self.run / "batch_manifest.json", value)
        self.assertTrue(library_is_current(self.run, manifest))
        write_json(self.run / "clip1/batch_status.json", {"status": "running"})
        self.assertFalse(library_is_current(self.run, manifest))
        write_json(self.run / "clip1/batch_status.json", {"status": "completed"})
        (self.run / "clip1/action/events.jsonl").write_text("")
        self.assertFalse(library_is_current(self.run, manifest))

    def test_foreign_event_and_unsafe_names_rejected(self):
        record = next(read_jsonl(self.run / "clip1/action/events.jsonl"))
        rows(self.run / "clip1/action/events.jsonl", [{**record, "id": "Punknown"}])
        with self.assertRaisesRegex(ValueError, "foreign/missing"):
            build_person_library(self.options)
        write_json(self.run / "batch_manifest.json", {"clips": [{"name": "../escape"}]})
        with self.assertRaisesRegex(ValueError, "safe clip"):
            build_person_library(self.options)

    def test_empty_tracks_are_represented_without_fake_accuracy(self):
        for index in range(1, 4):
            clip = self.run / f"clip{index}"
            for filename in ("identity/identities.jsonl", "identity_raw/identities.jsonl", "action/events.jsonl"):
                rows(clip / filename, [])
        result = build_person_library(self.options)
        self.assertEqual((result["expected_clips"], result["global_person_count"], result["event_count"]), (3, 0, 0))

    def test_review_detach_manual_group_and_stale_fingerprint(self):
        result = build_person_library(self.options)
        merged = next(p for p in self.people() if p["clip_count"] == 2)
        node = merged["members"][0]["node_id"]
        review = {"input_fingerprint": result["input_fingerprint"], "blocked_node_ids": [node]}
        result = build_person_library(replace(self.options, overwrite=True), review_value=review)
        self.assertEqual(result["global_person_count"], 3)
        nodes = [p["members"][0]["node_id"] for p in self.people()]
        review = {"input_fingerprint": result["input_fingerprint"], "assignments": {n: "team#15" for n in nodes}}
        result = build_person_library(replace(self.options, overwrite=True), review_value=review)
        self.assertEqual(result["global_person_count"], 1)
        self.assertEqual(self.people()[0]["status"], "manual_grouping")
        with self.assertRaisesRegex(ValueError, "stale"):
            build_person_library(replace(self.options, overwrite=True), review_value={"input_fingerprint": "old"})

    def test_manual_labels_cannot_override_same_clip_conflict(self):
        run = make_library_fixture(self.root / "conflict", extra_same_clip=True)
        result = build_person_library(PersonLibraryOptions(run, max_distance=.2))
        same_clip_nodes = [n["node_id"] for p in read_jsonl(run / "library/people.jsonl")
                           for n in p["members"] if n["clip_name"] == "clip1"]
        with self.assertRaisesRegex(ValueError, "same-clip"):
            build_person_library(PersonLibraryOptions(run, max_distance=.2, overwrite=True),
                                 review_value={"input_fingerprint": result["input_fingerprint"],
                                               "assignments": {n: "team#15" for n in same_clip_nodes}})

    def test_batch_flag_builds_only_after_complete_and_never_on_dry_run(self):
        from cli import process_batch
        arguments = ["process_batch", "--input-dir", str(self.root / "data"),
                     "--output-dir", str(self.run), "--cross-clip-distance", ".2"]
        with patch.object(sys, "argv", arguments), patch.object(process_batch, "run_batch", return_value={"failed_clips": []}), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            process_batch.main()
        self.assertEqual(json.loads(output.getvalue())["person_library"]["global_person_count"], 2)
        manifest = (self.run / "library/library_manifest.json").read_bytes()
        with patch.object(sys, "argv", arguments + ["--dry-run"]), \
                patch.object(process_batch, "run_batch", return_value={"dry_run": True}), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            process_batch.main()
        self.assertTrue(json.loads(output.getvalue())["person_library_plan"]["runs_after_all_clips_complete"])
        with patch.object(sys, "argv", arguments), \
                patch.object(process_batch, "run_batch", return_value={"failed_clips": ["clip2"]}), \
                contextlib.redirect_stdout(io.StringIO()) as output, self.assertRaises(SystemExit):
            process_batch.main()
        self.assertIn("person_library_skipped", json.loads(output.getvalue()))
        self.assertEqual((self.run / "library/library_manifest.json").read_bytes(), manifest)

    def test_api_person_event_pagination_review_and_readonly_sources(self):
        try:
            from fastapi.testclient import TestClient
            from web.backend.settings import WebSettings
            from web.backend.store import ProjectStore
            from web.backend.imports import import_results
        except ImportError:
            self.skipTest("Install requirements-web-test.txt for API regressions")
        build_person_library(self.options)
        settings = WebSettings(data_root=self.root / "data/web", source_root=self.root / "data",
                               import_root=self.root / "outputs", output_root=self.root / "outputs/web")
        settings.ensure_directories()
        store = ProjectStore(settings.data_root, settings.output_root)
        manifest = import_results(settings, store, self.run)
        baseline = (self.run / "library/identity_map.jsonl").read_bytes()
        with patch.dict("os.environ", {"BASKET_WEB_OUTPUT_ROOT": str(settings.output_root),
                                      "BASKET_WEB_DATA_ROOT": str(settings.data_root)}):
            from web.backend import app as module
        with patch.object(module, "settings", settings), patch.object(module, "store", store), TestClient(module.app) as client:
            base = f'/api/projects/{manifest["project_id"]}/person-library'
            catalog = client.get(base + "?limit=1").json()
            self.assertEqual((catalog["total"], len(catalog["items"])), (2, 1))
            person_id = catalog["items"][0]["global_person_id"]
            person = client.get(base + f"/people/{person_id}").json()
            self.assertEqual(person["clip_count"], 2)
            self.assertEqual(client.get(person["members"][0]["cover"]["crop_url"]).status_code, 200)
            events = client.get(base + f"/events?person_id={person_id}&event=basketball_2point_shot&limit=1").json()
            self.assertEqual((events["total"], len(events["items"])), (2, 1))
            self.assertTrue(events["items"][0]["web_video_id"])
            self.assertEqual(client.get(base + "/events?min_score=0.9").json()["total"], 0)
            detached = client.post(base + "/review", json={"operation": "detach", "node_id": person["members"][0]["node_id"]})
            self.assertEqual(detached.status_code, 200, detached.text)
            self.assertEqual(client.get(base).json()["summary"]["global_person_count"], 3)
            conflict = client.post(base + "/review", json={"operation": "reset"})
            self.assertEqual(conflict.status_code, 409)
            revision = detached.json()["revision"]
            # A fresh store and a reload keep the same independent review sidecar.
            from web.backend.person_library import library_catalog
            fresh_store = ProjectStore(settings.data_root, settings.output_root)
            persisted = library_catalog(settings, fresh_store, manifest["project_id"])
            self.assertEqual(persisted["revision"], revision)
            self.assertEqual(persisted["summary"]["global_person_count"], 3)
            self.assertEqual(client.post(base + "/review", json={"operation": "reset", "expected_revision": revision}).status_code, 200)
            self.assertEqual(client.get(base + "/people/unknown").status_code, 404)
            self.assertEqual(client.get(base + "?limit=1000").status_code, 422)
            self.assertEqual(client.get(base + "/events?min_score=nan").status_code, 422)
        self.assertEqual((self.run / "library/identity_map.jsonl").read_bytes(), baseline)


def main():
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(PersonLibraryChecks))
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
