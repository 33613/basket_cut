"""Synthetic regressions for product-facing human track audits."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from analysis.evaluation.track_review import (
    evaluate_manifest, evaluate_review, merge_review, review_index, validate_review,
)
from analysis.visualization.track_review import sample_positions, prepare_review
from contracts.schema import write_json


class TrackReviewChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="basket-review-check-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.meta = self.root / "video_meta.json"
        self.tracks = self.root / "tracks.jsonl"
        write_json(self.meta, {"video_id": "clip", "fps": 10, "processed_frames": 30,
                               "frame_count": 30, "width": 64, "height": 48})
        self.rows(self.tracks, [{"video_id": "clip", "frame_idx": frame,
                                "timestamp_s": frame / 10, "track_id": tid,
                                "bbox_xyxy": [10, 5, 30, 40], "det_score": .9,
                                "category_id": 0}
                               for tid, frames in ((1, (0, 1, 2)), (2, (10, 11, 12)),
                                                   (3, (20, 21, 22)), (4, (28,)))
                               for frame in frames])
        self.quality = self.root / "quality.jsonl"
        self.rows(self.quality, [{"track_id": tid, "retained_observations": 0 if tid == 4 else 3}
                                 for tid in (1, 2, 3, 4)])
        self.index = review_index(self.tracks, self.meta, self.quality)

    @staticmethod
    def rows(path, rows):
        path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    def review(self, rows):
        return {"schema_version": 1, "video_id": "clip", "fingerprint": self.index["fingerprint"],
                "tracks": [{"raw_track_id": tid, "verdict": verdict, "scope": scope,
                            "identity_label": label} for tid, verdict, scope, label in rows]}

    def test_no_review_and_empty_tracks_never_fake_accuracy(self):
        self.assertIsNone(evaluate_review(self.index, None)["scopes"]["full_track"]["raw"]["pure_track_rate"])
        self.tracks.write_text("", encoding="utf-8")
        result = evaluate_review(review_index(self.tracks, self.meta), None)
        self.assertIsNone(result["scopes"]["full_track"]["raw"]["review_coverage"])
        self.assertIsNone(result["fragmentation"]["tracks_per_reviewed_person"])

    def test_sampling_cannot_prove_purity_and_denominators_are_visible(self):
        review = self.review([(1, "pure", "sampled", "white#15"),
                              (2, "mixed", "full_track", None),
                              (4, "non_player", "full_track", None)])
        result = evaluate_review(self.index, review)
        self.assertEqual(result["scopes"]["full_track"]["raw"]["pure_track_rate"], 0)
        self.assertEqual(result["scopes"]["full_track"]["raw"]["review_coverage"], .5)
        self.assertEqual(result["scopes"]["full_track"]["retained"]["total_tracks"], 3)
        self.assertEqual(result["scopes"]["sampled"]["raw"]["pure"], 1)
        self.assertEqual(result["fragmentation"]["reviewed_people"], 0)

    def test_same_person_can_have_pure_fragments(self):
        review = self.review([(1, "pure", "full_track", "white#15"),
                              (2, "pure", "full_track", "white#15"),
                              (3, "pure", "full_track", "black#8")])
        result = evaluate_review(self.index, review)
        self.assertEqual(result["scopes"]["full_track"]["raw"]["pure_track_rate"], 1)
        self.assertEqual(result["fragmentation"]["extra_track_ids"], 1)
        self.assertEqual(result["fragmentation"]["tracks_per_reviewed_person"], 1.5)
        self.assertEqual(result["fragmentation"]["groups"][1]["overlapping_span_pairs"], [])

    def test_stale_unknown_duplicate_and_mixed_labels_rejected(self):
        clean = self.review([(1, "pure", "full_track", "white#15")])
        variants = []
        stale = copy.deepcopy(clean); stale["fingerprint"] = "stale"; variants.append(stale)
        unknown = copy.deepcopy(clean); unknown["tracks"][0]["raw_track_id"] = 999; variants.append(unknown)
        duplicate = copy.deepcopy(clean); duplicate["tracks"] *= 2; variants.append(duplicate)
        mixed = copy.deepcopy(clean); mixed["tracks"][0]["verdict"] = "mixed"; variants.append(mixed)
        for value in variants:
            with self.assertRaises(ValueError):
                validate_review(self.index, value)

    def test_archive_conflicts_and_unknowns_are_not_confirmed(self):
        review = self.review([(1, "pure", "full_track", "white#15"),
                              (2, "pure", "full_track", "black#8"),
                              (3, "mixed", "sampled", None)])
        mapping = self.root / "identity_map.jsonl"
        self.rows(mapping, [{"raw_track_id": tid, "person_id": pid, "video_id": "clip"}
                            for tid, pid in ((1, "P1"), (2, "P1"), (3, "P3"), (4, "P4"))])
        result = evaluate_review(self.index, review, mapping)
        self.assertEqual(result["archives"]["conflict"], 2)
        self.assertEqual(result["archives"]["unverified"], 1)
        self.assertEqual(result["archives"]["consistent"], 0)

    def test_merge_export_excludes_sampled_and_filtered_tracks(self):
        review = self.review([(1, "pure", "sampled", "white#15"),
                              (2, "pure", "full_track", "white#15"),
                              (3, "mixed", "sampled", None),
                              (4, "pure", "full_track", "white#15")])
        exported = merge_review(self.index, review)
        self.assertEqual(exported["assignments"], [{"raw_track_ids": [2], "identity_label": "white#15"}])
        self.assertEqual(exported["blocked_track_ids"], [3])

    def test_fingerprint_ignores_deployment_path_but_detects_boxes(self):
        meta = json.loads(self.meta.read_text()); meta["input_path"] = "runtime/clip.mp4"
        write_json(self.meta, meta)
        self.assertEqual(review_index(self.tracks, self.meta)["fingerprint"], self.index["fingerprint"])
        values = [json.loads(line) for line in self.tracks.read_text().splitlines()]
        values[0]["bbox_xyxy"] = [11, 5, 30, 40]
        self.rows(self.tracks, values)
        self.assertNotEqual(review_index(self.tracks, self.meta)["fingerprint"], self.index["fingerprint"])

    def test_batch_counts_and_missing_clip_not_silently_dropped(self):
        review = self.review([(1, "pure", "full_track", "white#15"), (2, "mixed", "full_track", None)])
        write_json(self.root / "review.json", review)
        manifest = self.root / "manifest.json"
        write_json(manifest, {"clips": [{"name": "ok", "tracks": "tracks.jsonl", "video_meta": "video_meta.json", "review": "review.json"},
                                       {"name": "missing", "tracks": "missing.jsonl", "video_meta": "video_meta.json"}]})
        result = evaluate_manifest(manifest, self.root / "metrics.json")
        self.assertFalse(result["complete"])
        self.assertEqual(result["completed_clips"], 1)
        self.assertEqual(len(result["failed_clips"]), 1)
        self.assertEqual(result["completed_subset"]["scopes"]["full_track"]["raw"]["pure_track_rate"], .5)

    def test_chronological_samples_unique_short_tracks(self):
        self.assertEqual(sample_positions(1, 5), [0])
        self.assertEqual(sample_positions(3, 5), [0, 1, 2])
        self.assertEqual(sample_positions(100, 5), [0, 25, 50, 74, 99])

    def test_batch_metrics_cannot_overwrite_review_or_tracks(self):
        manifest = self.root / "manifest.json"
        write_json(manifest, {"clips": [{"name": "clip", "tracks": "tracks.jsonl", "video_meta": "video_meta.json"}]})
        original = self.tracks.read_bytes()
        with self.assertRaises(ValueError):
            evaluate_manifest(manifest, self.tracks)
        self.assertEqual(self.tracks.read_bytes(), original)

    def test_real_video_evidence_includes_filtered_track(self):
        try:
            import cv2
            import numpy as np
        except ImportError:
            self.skipTest("Install OpenCV to check real video evidence")
        video = self.root / "source.avi"
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
        self.assertTrue(writer.isOpened())
        for _ in range(30):
            writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
        writer.release()
        original = self.tracks.read_bytes()
        prepare_review(video, self.tracks, self.meta, self.root / "evidence", self.quality)
        evidence = json.loads((self.root / "evidence/index.json").read_text())
        self.assertEqual(len(evidence["tracks"]), 4)
        self.assertEqual(len(evidence["tracks"][-1]["samples"]), 1)
        self.assertTrue((self.root / "evidence" / evidence["tracks"][0]["samples"][0]["context_path"]).is_file())
        self.assertEqual(original, self.tracks.read_bytes())


def main():
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(TrackReviewChecks))
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
