"""Functional crop-selection regressions; no model accuracy evaluation."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from contracts.schema import TrackRecord, read_jsonl
from pipeline.identity.sampling import select_quality_samples


class SamplingChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def video(self, images):
        path = self.root / "clip.avi"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (80, 120))
        self.assertTrue(writer.isOpened())
        for image in images:
            writer.write(image)
        writer.release()
        return path

    def record(self, frame, **kwargs):
        values = dict(video_id="v", frame_idx=frame, timestamp_s=frame / 10,
                      track_id=0, bbox_xyxy=(0, 0, 80, 120), det_score=.9, category_id=0)
        values.update(kwargs)
        return TrackRecord(**values)

    def noise(self):
        return np.random.default_rng(11).integers(30, 220, (120, 80, 3), dtype=np.uint8)

    def test_clear_frames_win_even_with_lower_detection_score(self):
        images = [np.full((120, 80, 3), 130, dtype=np.uint8) for _ in range(12)]
        for frame in (5, 9):
            images[frame] = self.noise()
        records = [self.record(f, det_score=.6 if f in (5, 9) else .99) for f in range(12)]
        selected, info = select_quality_samples(self.video(images), records,
            purpose="person", sample_count=8)
        self.assertEqual([r.frame_idx for r in selected], [5, 9])
        self.assertEqual(len(info), 2)  # Do not fill remaining slots with unusable imagery.

    def test_jersey_quality_excludes_sharp_legs_and_background(self):
        image = np.full((120, 80, 3), 130, dtype=np.uint8)
        # Keep textured legs below a JPEG block boundary so ringing from the
        # fixture codec does not introduce artificial detail into the torso.
        image[96:] = self.noise()[96:]
        video, records = self.video([image]), [self.record(0)]
        people, _ = select_quality_samples(video, records, purpose="person", sample_count=1)
        torsos, _ = select_quality_samples(video, records, purpose="jersey", sample_count=1)
        self.assertEqual(len(people), 1)
        self.assertEqual(torsos, [])

    def test_overlap_is_a_soft_penalty_even_for_unselected_tracks(self):
        video = self.video([self.noise()] * 11)
        records = [self.record(0), self.record(0, track_id=1), self.record(10)]
        selected, info = select_quality_samples(video, records, purpose="person",
            sample_count=2, track_ids={0})
        self.assertEqual({r.track_id for r in selected}, {0})
        self.assertGreater(info[(0, 10)]["sampling_quality"], info[(0, 0)]["sampling_quality"])
        self.assertEqual(info[(0, 0)]["sampling_components"]["overlap_fraction"], 1.)
        # Overlap alone must not eliminate a legitimate player crop.
        one, _ = select_quality_samples(video, records, purpose="person",
            sample_count=1, track_ids={0})
        self.assertEqual(one[0].frame_idx, 10)

    def test_temporal_gap_and_clipping_are_enforced(self):
        video = self.video([self.noise()] * 12)
        records = [self.record(f) for f in range(12)]
        records.append(self.record(4, track_id=1, bbox_xyxy=(-100, 0, 80, 120)))
        selected, _ = select_quality_samples(video, records, purpose="person",
                                             sample_count=3, min_gap_s=.25)
        self.assertEqual(len(selected), 3)
        self.assertTrue(all(r.track_id == 0 for r in selected))
        for a, b in zip(selected, selected[1:]):
            self.assertGreaterEqual(b.timestamp_s - a.timestamp_s, .25)

    def test_invalid_tracks_and_empty_selection_do_not_decode(self):
        with patch("cv2.VideoCapture", side_effect=AssertionError("No decode")):
            self.assertEqual(select_quality_samples(self.root / "missing", [],
                purpose="person", sample_count=2), ([], {}))
            self.assertEqual(select_quality_samples(self.root / "missing", [self.record(0)],
                purpose="person", sample_count=2, track_ids={1}), ([], {}))
            for records in ([self.record(0), self.record(0)],
                            [self.record(0), self.record(1, video_id="different")],
                            [self.record(0, timestamp_s=float("nan"))]):
                with self.assertRaises(ValueError):
                    select_quality_samples(self.root / "missing", records,
                                           purpose="person", sample_count=2)

    def test_video_ending_before_candidates_is_an_error(self):
        video = self.video([self.noise()])
        with self.assertRaisesRegex(ValueError, "Video ended"):
            select_quality_samples(video, [self.record(10)], purpose="person", sample_count=2)

    def test_kpr_prepare_only_uses_quality_selection_without_loading_model(self):
        from pipeline.identity.archive import IdentityArchiveOptions, build_identity_archive
        images = [np.full((120, 80, 3), 130, dtype=np.uint8) for _ in range(12)]
        images[5], images[9] = self.noise(), self.noise()
        tracks = self.root / "tracks.jsonl"
        tracks.write_text("".join(json.dumps(self.record(f).to_dict()) + "\n" for f in range(12)))
        with patch("pipeline.identity.archive.KPRBackend", side_effect=AssertionError("No GPU")):
            result = build_identity_archive(IdentityArchiveOptions(
                self.video(images), tracks, self.root / "kpr", prepare_only=True))
        self.assertEqual(result["sample_count"], 2)
        manifest = list(read_jsonl(self.root / "kpr/kpr_sampling_manifest.jsonl"))
        self.assertEqual([r["frame_idx"] for r in manifest], [5, 9])
        self.assertTrue(all("sampling_components" in r for r in manifest))
        self.assertTrue(all((self.root / "kpr" / r["crop_path"]).is_file() for r in manifest))

    def test_smoke_fixture_runs_sampling_cache_and_evidence_with_injected_backend(self):
        from cli.smoke_qwen import prepare_smoke_clip
        from pipeline.identity.jersey_qwen import recognize_jerseys_qwen
        video, tracks, mapping = prepare_smoke_clip(self.root)

        class Backend:
            provenance = {"backend": "qwen_vl", "model_sha256": "fixture"}

            def read(self, image):
                return '{"number":null,"readable":false}'

        backend = Backend()
        first = recognize_jerseys_qwen(video, tracks, mapping, self.root / "identity",
            self.root / "missing-model", max_samples=3, backend=backend)
        self.assertGreaterEqual(first["inference_count"], 2)
        samples = list(read_jsonl(self.root / "identity/jersey_samples.jsonl"))
        self.assertTrue(all(r["frame_idx"] >= 5 for r in samples))
        self.assertTrue(all(r["torso_roi_xyxy"] == r["sampling_roi_xyxy"] for r in samples))
        evidence = list(read_jsonl(self.root / "identity/jersey_tracks.jsonl"))
        self.assertEqual(evidence[0]["status"], "unreadable")
        second = recognize_jerseys_qwen(video, tracks, mapping, self.root / "identity",
            self.root / "missing-model", max_samples=3, backend=backend)
        self.assertEqual(second["inference_count"], 0)
        self.assertEqual(second["cache_hit_count"], len(samples))
        self.assertFalse(second["model_invoked"])

    def test_all_bad_crops_keep_empty_identity_outputs_and_unmapped_events(self):
        from pipeline.identity.archive import IdentityArchiveOptions, build_identity_archive
        from pipeline.identity.resolution import ResolutionOptions, resolve_identities
        from workflows.person_library import PersonLibraryOptions, build_person_library
        video = self.video([np.full((120, 80, 3), 130, dtype=np.uint8)] * 5)
        run, name = self.root / "run", "clip"
        clip = run / name
        tracking = clip / "tracking"
        tracking.mkdir(parents=True)
        tracks = tracking / "tracks.jsonl"
        tracks.write_text("".join(json.dumps(self.record(f).to_dict()) + "\n" for f in range(5)))
        meta = tracking / "video_meta.json"
        meta.write_text(json.dumps({"video_id": "v", "fps": 10, "processed_frames": 5}))
        raw = clip / "identity_raw"
        with patch("pipeline.identity.archive.KPRBackend", side_effect=AssertionError("No model")):
            result = build_identity_archive(IdentityArchiveOptions(video, tracks, raw))
        self.assertFalse(result["model_invoked"])
        self.assertEqual(result["sample_count"], 0)
        self.assertEqual(list(read_jsonl(raw / "identities.jsonl")), [])
        resolved = clip / "identity"
        resolution = resolve_identities(ResolutionOptions(tracks, meta, raw, resolved))
        self.assertEqual(resolution["missing_archive_tracks"], [0])
        (clip / "action").mkdir()
        (clip / "action/events.jsonl").write_text(json.dumps({
            "video_id": "v", "event_id": "e0", "identity_id": "T0000", "event": "pass",
            "start": 0, "end": .5, "raw_score": .8, "start_frame": 0, "end_frame": 4,
            "raw_track_ids": [0],
        }) + "\n")
        (run / "batch_manifest.json").write_text(json.dumps({
            "match_id": "game", "target": "full", "clips": [{"name": name}]}))
        build_person_library(PersonLibraryOptions(run, match_id="game"))
        events = list(read_jsonl(run / "library/events.jsonl"))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["identity_status"], "unmapped")

    def test_no_torso_crops_preserve_qwen_model_contract_without_gpu(self):
        from pipeline.identity.jersey_qwen import recognize_jerseys_qwen
        from pipeline.identity.qwen_backend import model_provenance
        video = self.video([np.full((120, 80, 3), 130, dtype=np.uint8)])
        tracks, mapping = self.root / "tracks.jsonl", self.root / "map.jsonl"
        tracks.write_text(json.dumps(self.record(0).to_dict()) + "\n")
        mapping.write_text(json.dumps({"raw_track_id": 0, "person_id": "P0000"}) + "\n")
        model = self.root / "model"
        model.mkdir()
        (model / "config.json").write_text('{}')
        (model / "model.safetensors").write_bytes(b'provenance-only-fixture')
        with patch("pipeline.identity.jersey_qwen.QwenBackend", side_effect=AssertionError("No GPU")):
            result = recognize_jerseys_qwen(video, tracks, mapping, self.root / "identity", model)
        self.assertEqual(result["provenance"], model_provenance(model))
        self.assertFalse(result["model_invoked"])
        evidence = list(read_jsonl(self.root / "identity/jersey_tracks.jsonl"))
        self.assertEqual(evidence[0]["status"], "unreadable")


def main():
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(SamplingChecks))
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
