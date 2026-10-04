"""Synthetic CPU regressions for track quality, identity resolution and joins."""

import json
import tempfile
import unittest
from pathlib import Path

from analysis.evaluation.identity import evaluate_identity
from contracts.schema import read_jsonl
from pipeline.identity.resolution import ResolutionOptions, resolve_identities
from pipeline.tracking.quality import QualityOptions, check_track_quality
from workflows.link_events import LinkEventsOptions, link_actions_to_people
from workflows.refinement import RefinementOptions, refine_existing_clip


class QualityChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="basket-quality-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.meta = self.root / "meta.json"
        self.write(
            self.meta,
            {"video_id": "v", "fps": 10, "processed_frames": 100, "frame_count": 100},
        )
        self.tracks = self.root / "tracks.jsonl"
        self.archive = self.root / "archive"
        self.archive.mkdir()

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def rows(self, path, values):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(v) + "\n" for v in values), encoding="utf-8")

    def track(self, tid, frame, box=(0, 0, 10, 20)):
        return {
            "video_id": "v",
            "track_id": tid,
            "frame_idx": frame,
            "timestamp_s": frame / 10,
            "bbox_xyxy": box,
            "det_score": 0.9,
            "category_id": 0,
        }

    def prepare(self, rows, distances=()):
        self.rows(self.tracks, rows)
        ids = sorted({r["track_id"] for r in rows})
        people = []
        for tid in ids:
            media = f"media/{tid}.jpg"
            (self.archive / "media").mkdir(exist_ok=True)
            (self.archive / media).write_bytes(b"synthetic-media")
            people.append(
                {
                    "person_id": f"P{tid:04d}",
                    "raw_track_ids": [tid],
                    "sample_count": 2,
                    "within_track": {"max_distance": 0.1},
                    "cover": {"crop_path": media, "archive_quality_score": 0.8},
                    "exemplars": [],
                }
            )
        self.rows(self.archive / "identities.jsonl", people)
        self.rows(
            self.archive / "kpr_track_pairs.jsonl",
            [
                {"track_id_a": a, "track_id_b": b, "distance": d}
                for a, b, d in distances
            ],
        )
        self.write(self.archive / "identity_archive_manifest.json", {"video_id": "v"})

    def q(self, **kwargs):
        return check_track_quality(
            QualityOptions(self.tracks, self.meta, self.root / "quality", **kwargs)
        )

    def r(self, **kwargs):
        return resolve_identities(
            ResolutionOptions(
                self.tracks, self.meta, self.archive, self.root / "resolved", **kwargs
            )
        )

    def test_short_track_is_audited_not_deleted(self):
        self.prepare([self.track(1, f) for f in range(5)] + [self.track(2, 2)])
        original = self.tracks.read_bytes()
        summary = self.q()
        self.assertEqual(summary["retained_track_count"], 1)
        self.assertEqual(self.tracks.read_bytes(), original)
        self.assertEqual(
            len(list(read_jsonl(self.root / "quality/quality_observations.jsonl"))), 6
        )

    def test_empty_tracks_are_valid(self):
        self.prepare([])
        self.assertEqual(self.q()["retained_track_count"], 0)
        self.assertEqual(self.r()["identity_count"], 0)

    def test_duplicate_reporting_is_not_automatic_suppression(self):
        self.prepare([self.track(t, f) for t in (1, 2) for f in range(3)])
        self.assertEqual(self.q()["retained_observations"], 6)
        self.assertEqual(
            self.q(overwrite=True, suppress_duplicates=True)["retained_observations"], 3
        )

    def test_distinct_boxes_are_not_suppressed(self):
        self.prepare(
            [self.track(1, f) for f in range(3)]
            + [self.track(2, f, (20, 0, 30, 20)) for f in range(3)]
        )
        self.assertFalse(self.q(suppress_duplicates=True)["duplicate_candidates"])

    def test_motion_jump_is_review_not_silent_rewrite(self):
        self.prepare(
            [
                self.track(1, f, (1000, 0, 1010, 20) if f == 2 else (0, 0, 10, 20))
                for f in range(4)
            ]
        )
        self.assertEqual(self.q()["review_track_count"], 1)
        self.assertEqual(len(list(read_jsonl(self.root / "quality/tracks.jsonl"))), 4)

    def test_duplicate_frame_is_rejected(self):
        self.prepare([self.track(1, 0)] * 2)
        with self.assertRaises(ValueError):
            self.q()

    def test_wrong_video_or_clock_or_bounds_is_rejected(self):
        for changes in (
            {"video_id": "other"},
            {"timestamp_s": 50},
            {"frame_idx": 101},
            {"timestamp_s": float("nan")},
        ):
            self.prepare([{**self.track(1, 0), **changes}])
            with self.assertRaises(ValueError):
                self.q()

    def test_quality_cannot_overwrite_input(self):
        self.prepare([self.track(1, 0)])
        with self.assertRaises(ValueError):
            check_track_quality(
                QualityOptions(self.tracks, self.meta, self.root, overwrite=True)
            )

    def test_no_automatic_merge_without_threshold(self):
        self.prepare([self.track(1, 0), self.track(2, 3)], [(1, 2, 0.01)])
        self.assertEqual(self.r()["identity_count"], 2)

    def test_human_mixed_track_is_blocked_even_with_small_kpr_distance(self):
        self.prepare([self.track(1, 0), self.track(2, 3)], [(1, 2, 0.01)])
        review = self.root / "review.json"
        self.write(review, {"video_id": "v", "blocked_track_ids": [1]})
        result = self.r(review=review, max_distance=.2)
        self.assertEqual(result["identity_count"], 2)
        self.assertIn(1, result["review_track_ids"])

    def test_all_merged_constituents_have_traceable_covers(self):
        self.prepare([self.track(tid, tid * 3) for tid in range(1, 7)])
        review = self.root / "review.json"
        self.write(review, {"video_id": "v", "assignments": [
            {"raw_track_ids": list(range(1, 7)), "identity_label": "white#15"}]})
        self.assertEqual(self.r(review=review)["identity_count"], 1)
        person = list(read_jsonl(self.root / "resolved/identities.jsonl"))[0]
        self.assertEqual(len(person["source_tracks"]), 6)
        for source in person["source_tracks"]:
            self.assertTrue((self.root / "resolved" / source["cover"]["crop_path"]).is_file())

    def test_complete_link_prevents_transitive_chain(self):
        self.prepare(
            [self.track(1, 0), self.track(2, 3), self.track(3, 6)],
            [(1, 2, 0.1), (2, 3, 0.15), (1, 3, 0.8)],
        )
        self.assertEqual(self.r(max_distance=0.2)["identity_count"], 2)

    def test_missing_pair_evidence_blocks_merge(self):
        self.prepare(
            [self.track(1, 0), self.track(2, 3), self.track(3, 6)],
            [(1, 2, 0.1), (2, 3, 0.1)],
        )
        self.assertEqual(self.r(max_distance=0.2)["identity_count"], 2)

    def test_simultaneous_distinct_people_cannot_merge(self):
        self.prepare(
            [self.track(1, 0), self.track(2, 0, (50, 0, 60, 20))], [(1, 2, 0.01)]
        )
        self.assertEqual(self.r(max_distance=0.2)["identity_count"], 2)

    def test_confirmed_merge_and_media_are_traceable(self):
        self.prepare([self.track(1, 0), self.track(2, 3)])
        review = self.root / "review.json"
        self.write(
            review,
            {
                "video_id": "v",
                "assignments": [
                    {"raw_track_ids": [1, 2], "identity_label": "team-A:#7"}
                ],
            },
        )
        self.assertEqual(self.r(review=review)["human_confirmed_people"], 1)
        people = list(read_jsonl(self.root / "resolved/identities.jsonl"))
        self.assertEqual(people[0]["raw_track_ids"], [1, 2])
        self.assertTrue(
            (self.root / "resolved" / people[0]["cover"]["crop_path"]).is_file()
        )
        self.assertEqual(len(list(read_jsonl(self.archive / "identities.jsonl"))), 2)

    def test_cannot_link_and_conflicting_review_are_rejected(self):
        self.prepare([self.track(1, 0), self.track(2, 3)], [(1, 2, 0.01)])
        review = self.root / "review.json"
        self.write(
            review,
            {
                "video_id": "v",
                "assignments": [{"raw_track_ids": [1, 2], "identity_label": "A"}],
                "cannot_link": [[1, 2]],
            },
        )
        with self.assertRaises(ValueError):
            self.r(review=review)

    def test_unknown_review_track_and_wrong_video_are_rejected(self):
        self.prepare([self.track(1, 0)])
        review = self.root / "review.json"
        for value in (
            {"video_id": "wrong"},
            {
                "video_id": "v",
                "assignments": [{"raw_track_ids": [99], "identity_label": "A"}],
            },
        ):
            self.write(review, value)
            with self.assertRaises(ValueError):
                self.r(review=review)

    def test_possible_mixed_track_blocks_auto_merge(self):
        self.prepare([self.track(1, 0), self.track(2, 3)], [(1, 2, 0.01)])
        people = list(read_jsonl(self.archive / "identities.jsonl"))
        people[0]["within_track"]["max_distance"] = 0.8
        self.rows(self.archive / "identities.jsonl", people)
        self.assertEqual(self.r(max_distance=0.2)["identity_count"], 2)

    def test_baseline_archive_cannot_be_overwritten(self):
        self.prepare([self.track(1, 0)])
        with self.assertRaises(ValueError):
            resolve_identities(
                ResolutionOptions(
                    self.tracks, self.meta, self.archive, self.archive, overwrite=True
                )
            )

    def test_media_escape_is_rejected(self):
        self.prepare([self.track(1, 0)])
        people = list(read_jsonl(self.archive / "identities.jsonl"))
        people[0]["cover"]["crop_path"] = "../tracks.jsonl"
        self.rows(self.archive / "identities.jsonl", people)
        with self.assertRaises(ValueError):
            self.r()

    def test_identity_pair_metric_penalizes_wrong_and_missing_merges(self):
        truth = self.root / "truth.jsonl"
        pred = self.root / "pred.jsonl"
        self.rows(
            truth,
            [
                {
                    "video_id": "v",
                    "raw_track_id": t,
                    "identity_label": "A" if t < 3 else "B",
                }
                for t in (1, 2, 3)
            ],
        )
        self.rows(
            pred,
            [{"video_id": "v", "raw_track_id": t, "person_id": "P"} for t in (1, 2, 3)],
        )
        result = evaluate_identity(truth, pred, self.root / "metric.json")
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (1, 2, 0))
        self.rows(pred, [{"raw_track_id": 1, "person_id": "P"}])
        self.assertEqual(
            evaluate_identity(truth, pred, self.root / "metric.json")["fn"], 1
        )

    def test_empty_prediction_does_not_fake_perfect_identity(self):
        truth, pred = self.root / "truth.jsonl", self.root / "pred.jsonl"
        self.rows(
            truth,
            [
                {"video_id": "v", "raw_track_id": t, "identity_label": "A"}
                for t in (1, 2)
            ],
        )
        self.rows(pred, [])
        self.assertEqual(
            evaluate_identity(truth, pred, self.root / "metric.json")["headline"][
                "value"
            ],
            0,
        )

    def test_empty_identity_map_yields_empty_join(self):
        actions, mapping = self.root / "actions.jsonl", self.root / "map.jsonl"
        self.rows(actions, [{"track_id": 1, "video_id": "v"}])
        self.rows(mapping, [])
        summary = link_actions_to_people(
            LinkEventsOptions(actions, mapping, self.root / "linked.jsonl")
        )
        self.assertEqual(summary["written_action_records"], 0)

    def test_auto_assignment_is_not_human_confirmation(self):
        self.prepare(
            [self.track(1, f) for f in range(3)]
            + [self.track(2, f) for f in range(3, 6)],
            [(1, 2, 0.1)],
        )
        review = self.root / "review.json"
        self.write(
            review,
            {
                "video_id": "v",
                "assignments": [{"raw_track_ids": [1], "identity_label": "A"}],
            },
        )
        self.r(review=review, max_distance=0.2)
        mapping = list(read_jsonl(self.root / "resolved/identity_map.jsonl"))
        self.assertEqual(
            [m["status"] for m in mapping], ["human_confirmed", "auto_assigned"]
        )
        self.assertTrue(all(m["confidence"] is None for m in mapping))

    def test_different_confirmed_labels_block_automatic_merge(self):
        self.prepare(
            [self.track(1, f) for f in range(3)]
            + [self.track(2, f) for f in range(3, 6)],
            [(1, 2, 0.1)],
        )
        review = self.root / "review.json"
        self.write(
            review,
            {
                "video_id": "v",
                "assignments": [
                    {"raw_track_ids": [1], "identity_label": "A"},
                    {"raw_track_ids": [2], "identity_label": "B"},
                ],
            },
        )
        self.assertEqual(self.r(review=review, max_distance=0.2)["identity_count"], 2)
        pair = next(read_jsonl(self.root / "resolved/resolution_pairs.jsonl"))
        self.assertIn("different_confirmed_people", pair["auto_block_reasons"])

    def test_wrong_archive_video_is_rejected(self):
        self.prepare([self.track(1, f) for f in range(3)])
        self.write(
            self.archive / "identity_archive_manifest.json", {"video_id": "other"}
        )
        with self.assertRaises(ValueError):
            self.r()

    def test_insufficient_appearance_support_blocks_auto_merge(self):
        self.prepare(
            [self.track(1, f) for f in range(3)]
            + [self.track(2, f) for f in range(3, 6)],
            [(1, 2, 0.1)],
        )
        people = list(read_jsonl(self.archive / "identities.jsonl"))
        people[0]["sample_count"] = 1
        self.rows(self.archive / "identities.jsonl", people)
        self.assertEqual(self.r(max_distance=0.2)["identity_count"], 2)
        pairs = list(read_jsonl(self.root / "resolved/resolution_pairs.jsonl"))
        self.assertIn("insufficient_kpr_samples", pairs[0]["auto_block_reasons"])

    def test_invalid_kpr_distances_are_rejected(self):
        self.prepare(
            [self.track(1, f) for f in range(3)]
            + [self.track(2, f) for f in range(3, 6)],
            [(1, 2, float("nan"))],
        )
        with self.assertRaises(ValueError):
            self.r(max_distance=0.2)

    def test_identity_metric_cannot_overwrite_its_inputs(self):
        truth, pred = self.root / "truth.jsonl", self.root / "prediction.jsonl"
        self.rows(
            truth,
            [
                {"video_id": "v", "raw_track_id": t, "identity_label": "A"}
                for t in (1, 2)
            ],
        )
        self.rows(pred, [])
        with self.assertRaises(ValueError):
            evaluate_identity(truth, pred, truth)

    def test_no_positive_identity_pairs_is_not_perfect(self):
        truth, pred = self.root / "truth.jsonl", self.root / "prediction.jsonl"
        self.rows(
            truth,
            [
                {"video_id": "v", "raw_track_id": t, "identity_label": str(t)}
                for t in (1, 2)
            ],
        )
        self.rows(
            pred,
            [{"video_id": "v", "raw_track_id": t, "person_id": str(t)} for t in (1, 2)],
        )
        self.assertIsNone(
            evaluate_identity(truth, pred, self.root / "metric.json")["headline"][
                "value"
            ]
        )

    def test_refinement_reuses_models_and_keeps_baseline(self):
        self.prepare([self.track(1, f) for f in range(6)] + [self.track(2, 2)])
        run = self.root / "baseline"
        (run / "tracking").mkdir(parents=True)
        (run / "identity").mkdir()
        (run / "action").mkdir()
        import shutil

        shutil.copy2(self.tracks, run / "tracking/tracks.jsonl")
        shutil.copy2(self.meta, run / "tracking/video_meta.json")
        shutil.copytree(self.archive, run / "identity", dirs_exist_ok=True)
        self.rows(
            run / "action/actions.jsonl",
            [
                {
                    "video_id": "v",
                    "track_id": t,
                    "frame_idx": 2,
                    "proposal_frame_idx": 2,
                    "timestamp_s": 0.2,
                    "selected_actions": [{"label": "basketball_pass", "score": 0.9}],
                    "action_candidates": [],
                }
                for t in (1, 2)
            ],
        )
        self.write(
            run / "action/actions.summary.json", {"settings": {"predict_stepsize": 2}}
        )
        original = (run / "tracking/tracks.jsonl").read_bytes()
        result = refine_existing_clip(RefinementOptions(run, self.root / "refined"))
        self.assertFalse(result["models_rerun"])
        self.assertEqual(result["actions"]["retained"], 1)
        self.assertEqual((run / "tracking/tracks.jsonl").read_bytes(), original)
        self.assertEqual((self.root / "refined/tracking/tracks.jsonl").read_bytes(), original)
        self.assertEqual(
            len(list(read_jsonl(self.root / "refined/action/events.jsonl"))), 1
        )


def run_selfcheck():
    return unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(QualityChecks)
    )
