"""Synthetic CPU checks. Fixtures never represent real tracking/ReID accuracy."""

from dataclasses import replace
import base64
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from contracts.schema import EventRecord, read_jsonl, write_json
from pipeline.identity.cross_clip import part_distances
from pipeline.identity.player_registry import PlayerRegistry
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
    write_json(run / "batch_manifest.json", {"target": "full", "match_id": "synthetic-match", "clips": [
        {"name": f"clip{index + 1}"} for index in range(len(angles))]})
    return run

class PersonLibraryChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='basket-library-check-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = make_library_fixture(self.root)
        self.options = PersonLibraryOptions(self.run, max_distance=.2)

    def snapshot(self):
        with PlayerRegistry(self.run / 'library/players.sqlite3', 'synthetic-match') as registry:
            return registry.snapshot()

    def test_incremental_ids_survive_new_clip_and_repeated_registration(self):
        first = build_person_library(replace(self.options, clip_names=('clip1',)))
        pid = self.snapshot()['people'][0]['global_person_id']
        second = build_person_library(replace(self.options, clip_names=('clip2',)))
        snap = self.snapshot()
        self.assertEqual(len(snap['people']), 1)
        self.assertEqual(snap['people'][0]['global_person_id'], pid)
        self.assertEqual(snap['people'][0]['clip_count'], 2)
        third = build_person_library(replace(self.options, clip_names=('clip2',)))
        self.assertEqual(third['review_revision'], second['review_revision'])
        self.assertNotEqual(first['review_revision'], second['review_revision'])
        build_person_library(replace(self.options, clip_names=('clip3',)))
        self.assertEqual(len(self.snapshot()['people']), 2)
        self.assertIn(pid, {p['global_person_id'] for p in self.snapshot()['people']})

    def test_same_local_ids_are_scoped_and_events_join_without_losing_sources(self):
        originals = {p: p.read_bytes() for p in self.run.rglob('*') if p.is_file()}
        result = build_person_library(self.options)
        self.assertEqual((result['global_person_count'], result['event_count']), (2, 3))
        events = self.snapshot()['events']
        self.assertEqual(len({e['event_id'] for e in events}), 3)
        self.assertTrue(all(e['start'] == .4 for e in events))
        self.assertTrue(all(e['id'] == e['global_person_id'] for e in events))
        self.assertTrue(all(p.read_bytes() == value for p, value in originals.items()))

    def test_foreign_match_refused(self):
        build_person_library(self.options)
        with self.assertRaises(ValueError):
            PlayerRegistry(self.run / 'library/players.sqlite3', 'another-match')

    def test_changed_registered_evidence_rolls_back(self):
        build_person_library(self.options)
        revision = self.snapshot()['revision']
        path = self.run / 'clip1/identity/identities.jsonl'
        value = list(read_jsonl(path)); value[0]['sample_count'] += 1; rows(path, value)
        with self.assertRaises(ValueError):
            build_person_library(self.options)
        self.assertEqual(self.snapshot()['revision'], revision)

    def test_incompatible_feature_contract_refused(self):
        build_person_library(replace(self.options, clip_names=('clip1',)))
        write_json(self.run / 'clip2/identity_raw/kpr_summary.json', {'feature_contract': {'checkpoint_sha256': 'different'}})
        with self.assertRaises(ValueError):
            build_person_library(replace(self.options, clip_names=('clip2',)))
        self.assertEqual(len(self.snapshot()['mappings']), 1)

    def test_low_quality_stays_pending_and_does_not_enter_gallery(self):
        path = self.run / 'clip1/identity_raw/identities.jsonl'
        value = list(read_jsonl(path)); value[0]['sample_count'] = 1; rows(path, value)
        build_person_library(replace(self.options, clip_names=('clip1',)))
        person = self.snapshot()['people'][0]
        self.assertEqual(person['status'], 'needs_review')
        self.assertFalse(person['members'][0]['gallery_eligible'])

    def test_unknown_events_preserved(self):
        event_path = self.run / 'clip1/action/events.jsonl'
        event = list(read_jsonl(event_path))[0]
        event.update(id='T0999', raw_track_ids=[999]); rows(event_path, [event])
        build_person_library(replace(self.options, clip_names=('clip1',)))
        event = self.snapshot()['events'][0]
        self.assertIsNone(event['global_person_id'])
        self.assertEqual(event['identity_status'], 'unmapped')

    def test_review_revision_detach_restore_and_event_reassociation(self):
        build_person_library(self.options)
        snap = self.snapshot()
        person = next(p for p in snap['people'] if p['clip_count'] == 2)
        node = person['members'][0]['node_id']; pid = person['global_person_id']
        with PlayerRegistry(self.run / 'library/players.sqlite3', 'synthetic-match') as registry:
            revision = registry.review('detach', node_id=node, expected_revision=snap['revision'])
            with self.assertRaises(RuntimeError):
                registry.review('restore', node_id=node, expected_revision=snap['revision'])
            detached = next(m for m in registry.snapshot()['mappings'] if m['node_id'] == node)
            self.assertNotEqual(detached['global_person_id'], pid)
            registry.review('restore', node_id=node, expected_revision=revision)
            restored = next(m for m in registry.snapshot()['mappings'] if m['node_id'] == node)
            self.assertEqual(restored['global_person_id'], pid)
            self.assertEqual(len(registry.snapshot()['people']), 2)

    def test_same_clip_simultaneous_people_cannot_merge(self):
        run = make_library_fixture(self.root / 'overlap', angles=(0,), extra_same_clip=True)
        build_person_library(PersonLibraryOptions(run, max_distance=.2))
        with PlayerRegistry(run / 'library/players.sqlite3', 'synthetic-match') as registry:
            snap = registry.snapshot(); left, right = snap['people']
            rev = registry.review('label_group', person_id=left['global_person_id'], label='white#15', expected_revision=snap['revision'])
            with self.assertRaises(ValueError):
                registry.review('label_group', person_id=right['global_person_id'], label='white#15', expected_revision=rev)
            self.assertEqual(registry.snapshot()['revision'], rev)

    def test_disjoint_fragments_same_clip_can_match(self):
        run = make_library_fixture(self.root / 'fragments', angles=(0,), extra_same_clip=True)
        tracks = list(read_jsonl(run / 'clip1/tracking/tracks.jsonl'))
        tracks[1].update(frame_idx=25, timestamp_s=1)
        rows(run / 'clip1/tracking/tracks.jsonl', tracks)
        result = build_person_library(PersonLibraryOptions(run, max_distance=.2))
        self.assertEqual(result['global_person_count'], 1)

    def test_ambiguous_second_candidate_margin_abstains(self):
        run = make_library_fixture(self.root / 'ambiguous', angles=(0, .1, .05))
        opts = PersonLibraryOptions(run, max_distance=.02, novelty_distance=.04, min_margin=.03)
        build_person_library(replace(opts, clip_names=('clip1',)))
        build_person_library(replace(opts, clip_names=('clip2',)))
        build_person_library(replace(opts, clip_names=('clip3',), max_distance=.2, novelty_distance=.5))
        with PlayerRegistry(run / 'library/players.sqlite3', 'synthetic-match') as registry:
            latest = next(p for p in registry.snapshot()['people'] if p['members'][0]['clip_name'] == 'clip3')
            self.assertEqual(latest['status'], 'needs_review')

    def test_source_changes_mark_index_stale_and_symlink_refused(self):
        result = build_person_library(self.options)
        self.assertTrue(library_is_current(self.run, result))
        (self.run / 'clip1/action/events.jsonl').write_text('')
        self.assertFalse(library_is_current(self.run, result))
        other = self.root / 'escape'; other.mkdir()
        with self.assertRaises(ValueError):
            build_person_library(replace(self.options, output_dir=other))

    def test_invalid_prototypes_refused(self):
        path = self.run / 'clip1/identity_raw/kpr_track_prototypes.npz'
        np.savez(path, embeddings=np.ones((1,2,2)), visibility_scores=np.ones((1,2)), track_ids=[0])
        with self.assertRaises(ValueError):
            build_person_library(self.options)

    def test_different_numbers_veto_matching_and_equal_numbers_do_not_force_it(self):
        run = make_library_fixture(self.root / 'numbers', angles=(0, .01, 1.4))
        for name, number in (('clip1', '15'), ('clip2', '32'), ('clip3', '15')):
            rows(run / name / 'identity/jersey_tracks.jsonl', [{'raw_track_id': 0,
                'number': number, 'status': 'candidate_consensus', 'verified': False, 'readings': []}])
            write_json(run / name / 'identity/jersey_summary.json', {'backend': 'qwen_vl',
                'provenance': {'model_sha256': 'synthetic'}, 'thresholds': {'min_support': 2}})
        result = build_person_library(PersonLibraryOptions(run, max_distance=.2, use_jersey_evidence=True))
        self.assertEqual(result['global_person_count'], 3)
        with PlayerRegistry(run / 'library/players.sqlite3', 'synthetic-match') as registry:
            second = next(p for p in registry.snapshot()['people'] if p['members'][0]['clip_name'] == 'clip2')
            self.assertEqual(second['status'], 'needs_review')

    def test_incremental_qwen_model_changes_refused(self):
        for name in ('clip1', 'clip2'):
            rows(self.run / name / 'identity/jersey_tracks.jsonl', [{'raw_track_id': 0,
                'number': '15', 'status': 'candidate_consensus', 'verified': False, 'readings': []}])
            write_json(self.run / name / 'identity/jersey_summary.json', {'backend': 'qwen_vl',
                'provenance': {'model_sha256': name}, 'thresholds': {'min_support': 2}})
        opts = replace(self.options, use_jersey_evidence=True)
        build_person_library(replace(opts, clip_names=('clip1',)))
        with self.assertRaises(ValueError):
            build_person_library(replace(opts, clip_names=('clip2',)))
        self.assertEqual(len(self.snapshot()['mappings']), 1)

    def test_manual_merge_preserves_alias_and_exclusion_stops_gallery(self):
        build_person_library(self.options)
        with PlayerRegistry(self.run / 'library/players.sqlite3', 'synthetic-match') as registry:
            snap = registry.snapshot(); left, right = snap['people']
            revision = registry.review('label_group', person_id=left['global_person_id'], label='confirmed-player', expected_revision=snap['revision'])
            registry.review('label_group', person_id=right['global_person_id'], label='confirmed-player', expected_revision=revision)
            merged = registry.snapshot()
            self.assertEqual(len(merged['people']), 1)
            self.assertEqual(merged['aliases'][right['global_person_id']], left['global_person_id'])
            revision = registry.review('exclude', person_id=left['global_person_id'], expected_revision=merged['revision'])
            self.assertEqual(registry.snapshot()['people'][0]['status'], 'non_player')
            self.assertFalse(any(m['gallery_eligible'] for m in registry.snapshot()['people'][0]['members']))

    def test_missing_database_cannot_silently_regenerate_ids(self):
        build_person_library(self.options)
        (self.run / 'library/players.sqlite3').unlink()
        with self.assertRaises(FileNotFoundError):
            build_person_library(self.options)

    def test_snapshot_manifest_publishes_one_immutable_generation(self):
        result = build_person_library(self.options)
        directory = self.run / 'library' / result['index_dir']
        self.assertTrue((directory / 'people.jsonl').is_file())
        # A partially replaced compatibility file cannot affect the published snapshot.
        (self.run / 'library/people.jsonl').write_text('broken')
        self.assertEqual(len(list(read_jsonl(directory / 'people.jsonl'))), 2)
        build_person_library(self.options)
        self.assertEqual(len(list(read_jsonl(self.run / 'library/people.jsonl'))), 2)

    def test_visibility_distance_matches_independent_reference(self):
        rng = np.random.default_rng(9)
        f = rng.normal(size=(5,3,4)).astype(np.float32)
        f /= np.linalg.norm(f, axis=-1, keepdims=True)
        v = rng.random((5,3)).astype(np.float32)
        actual = part_distances(f,v,min_common_parts=2)
        for a in range(5):
            for b in range(5):
                w = np.sqrt(v[a]*v[b])
                expected = (np.linalg.norm(f[a]-f[b],axis=-1)*w).sum()/w.sum()/2
                self.assertAlmostEqual(float(actual[a,b]), float(expected), places=3)

    def test_http_import_review_and_dynamic_events(self):
        from fastapi.testclient import TestClient
        from web.backend import app as module
        from web.backend.store import ProjectStore
        from web.backend.settings import WebSettings
        from web.backend.imports import import_results
        build_person_library(self.options)
        settings = replace(WebSettings(), data_root=self.root / 'web-data', output_root=self.root / 'web-output',
                           source_root=self.root / 'data', import_root=self.root / 'outputs')
        store = ProjectStore(settings.data_root, settings.output_root)
        project = import_results(settings, store, self.run)
        with patch.object(module, 'settings', settings), patch.object(module, 'store', store), TestClient(module.app) as client:
            base = f"/api/projects/{project['project_id']}/person-library"
            result = client.get(base); self.assertEqual(result.status_code, 200)
            value = result.json(); self.assertEqual(value['total'], 2)
            self.assertEqual(client.get(base + '?status=candidate').json()['total'], 2)
            assignments = f"/api/projects/{project['project_id']}/videos/{project['videos'][0]['video_id']}/player-assignments"
            before = client.get(assignments).json()
            self.assertEqual(before['items'][0]['raw_track_ids'], [0])
            person = next(p for p in value['items'] if p['clip_count'] == 2)
            detail = client.get(base + '/people/' + person['global_person_id']).json()
            request = {'operation': 'detach', 'node_id': detail['members'][0]['node_id'], 'expected_revision': value['revision']}
            # Route is shared with the existing review UI.
            response = client.post(base + '/review', json=request)
            self.assertEqual(response.status_code, 200, response.text)
            after = client.get(assignments).json()
            self.assertNotEqual(after['revision'], before['revision'])
            self.assertNotEqual(after['items'][0]['global_person_id'], before['items'][0]['global_person_id'])
            self.assertTrue(after['items'][0]['manually_detached'])
            self.assertEqual(client.post(base + '/review', json=request).status_code, 409)
            events = client.get(base + '/events').json()
            self.assertEqual(events['total'], 3)
            exclusion = client.post(base + '/review', json={'operation': 'exclude',
                'person_id': person['global_person_id'], 'expected_revision': response.json()['revision']})
            self.assertEqual(exclusion.status_code, 200, exclusion.text)
            self.assertEqual(client.get(base + '?status=non_player').json()['total'], 1)
            self.assertEqual(client.get(base).json()['total'], 2)


def main():
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(PersonLibraryChecks))
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == '__main__':
    main()
