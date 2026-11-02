"""CPU integration: synthetic tracks/features, real video/crops/portable archive."""

import copy
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest

import cv2
import numpy as np

from contracts.schema import TrackRecord, write_jsonl_line
from pipeline.media import atomic_json, sha256_file
from workflows import goal_test_export as flow


class Tensor:
    def __init__(self, array):
        self.array = array

    def numpy(self):
        return self.array


class FakeKPR:
    cfg = SimpleNamespace(model=SimpleNamespace(kpr=SimpleNamespace(test_embeddings=["parts"])),
                          test=SimpleNamespace(normalize_feature=True))

    def extract(self, images, *, batch_size):
        return Tensor(np.ones((len(images), 2, 4), dtype=np.float32)), Tensor(np.ones((len(images), 2), dtype=np.float32))


class ExportCheck(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.inputs = self.base / "input"
        self.inputs.mkdir()
        self.root = self.base / "test"
        for vid in ("3", "11"):
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=160x120:rate=10:duration=2",
                            "-c:v", "libx264", "-threads", "1", str(self.inputs / f"{vid}.mp4")], check=True)
        self.annotations = self.base / "events.csv"
        flow.write_csv(self.annotations, [dict(video_id=v, filename=f"{v}.mp4", goal_time_s="1.5", jersey_number=n)
                       for v, n in (("3", "27"), ("11", "15"))],
                       ["video_id", "filename", "goal_time_s", "jersey_number"])
        flow.prepare(self.inputs, self.annotations, self.root, "same-game")
        for video in flow.manifest(self.root)["videos"]:
            folder = self.root / "track" / video["video_id"]
            folder.mkdir(parents=True)
            atomic_json(folder / "video_meta.json", dict(video_id=video["video_id"], fps=10,
                        width=160, height=120, frame_count=20, processed_frames=20))
            with (folder / "tracks.jsonl").open("w") as stream:
                for f in range(20):
                    for tid in (0, 1):
                        if tid == 0 and f == 8:
                            continue
                        r = TrackRecord(video["video_id"], f, f/10, tid,
                                        (10+tid*65, 10, 60+tid*65, 110), .9, 0)
                        write_jsonl_line(stream, r.to_dict())
            shutil_source = self.root / video["video_path"]
            import shutil
            shutil.copy2(shutil_source, folder / "tracks_vis.mp4")
            atomic_json(folder / "tracking_complete.json", dict(outputs=flow.track_signature(self.root, video["video_id"])))
            flow.sample_tracks(self.root, video, samples_per_track=4)
            protocol = dict(checkpoint_sha256="fake-for-cpu-test", config_sha256="fake-for-cpu-test",
                            samples_sha256=sha256_file(folder / "samples.jsonl"), prompt_mode="none")
            flow.features(self.root, video, FakeKPR(), protocol)

    def tearDown(self):
        self.temporary.cleanup()

    def test_portable_archive_and_feature_index(self):
        result = flow.pack(self.root)
        self.assertEqual(result["selected_shooters"], 0)
        self.assertEqual(flow.read_json(self.root / "export_status.json")["shooter_status"], "pending_manual_selection")
        moved = self.base / "downloaded"
        moved.mkdir()
        with tarfile.open(result["archive"]) as tar:
            tar.extractall(moved, filter="data")
        portable = moved / "test"
        subprocess.run(["sha256sum", "--quiet", "-c", "checksums.sha256"], cwd=portable, check=True)
        ids = set()
        for video in flow.manifest(portable)["videos"]:
            vid = video["video_id"]
            samples = list(flow.read_jsonl(portable / "identity" / vid / "samples.jsonl"))
            with np.load(portable / "identity" / vid / "features.npz", allow_pickle=False) as data:
                self.assertEqual(list(data["sample_ids"]), [s["sample_id"] for s in samples])
                self.assertEqual(data["embeddings"].shape, (8, 2, 4))
            for s in samples:
                self.assertNotIn(s["sample_id"], ids)
                ids.add(s["sample_id"])
                image = cv2.imread(str(portable / s["crop_path"]))
                self.assertEqual(image.shape[:2], (100, 50))

    def test_shooter_selection_preserves_gaps_and_features(self):
        selection = self.root / "shooter_selection.csv"
        rows = flow.rows_csv(selection)
        rows[0].update(track_id="0", start_frame="3", end_frame="12")
        flow.write_csv(selection, rows, list(rows[0]))
        self.assertEqual(flow.export_shooters(self.root, selection), 1)
        folder = self.root / "track/3/shooter"
        frames = list(flow.read_jsonl(folder / "frames.jsonl"))
        self.assertEqual(len(frames), 10)
        self.assertFalse(frames[5]["observed"])
        self.assertIsNone(frames[5]["crop_path"])
        capture = cv2.VideoCapture(str(folder / "trajectory.mp4"))
        self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 10)
        capture.release()
        self.assertEqual(len(list((folder / "crops").glob("*.png"))), 9)
        with np.load(self.root / "identity/3/shooter_features.npz", allow_pickle=False) as data:
            self.assertTrue((data["track_ids"] == 0).all())
            self.assertTrue(((data["frame_indices"] >= 3) & (data["frame_indices"] <= 12)).all())
        self.assertEqual(flow.read_json(folder / "selection.json")["selection_method"], "manual_track_selection")

    def test_bad_selection_does_not_destroy_existing_exports(self):
        selection = self.root / "shooter_selection.csv"
        rows = flow.rows_csv(selection)
        rows[0]["track_id"] = "0"
        flow.write_csv(selection, rows, list(rows[0]))
        flow.export_shooters(self.root, selection)
        rows[1]["track_id"] = "999"
        flow.write_csv(selection, rows, list(rows[0]))
        with self.assertRaisesRegex(ValueError, "missing shooter"):
            flow.export_shooters(self.root, selection)
        self.assertTrue((self.root / "track/3/shooter/selection.json").exists())

    def test_changed_crop_cannot_be_packed(self):
        video = flow.manifest(self.root)["videos"][0]
        sample = next(flow.read_jsonl(self.root / "track/3/samples.jsonl"))
        (self.root / sample["crop_path"]).write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "Crop changed"):
            flow.pack(self.root)

    def test_resume_rejects_policy_and_annotation_changes(self):
        flow.prepare(self.inputs, self.annotations, self.root, "same-game")
        video = flow.manifest(self.root)["videos"][0]
        flow.sample_tracks(self.root, video, samples_per_track=4)
        with self.assertRaisesRegex(ValueError, "Sampling inputs changed"):
            flow.sample_tracks(self.root, video, samples_per_track=8)
        with self.assertRaisesRegex(ValueError, "Input/annotations changed"):
            flow.prepare(self.inputs, self.annotations, self.root, "other-game")

    def test_bad_timestamps_and_unsafe_files_rejected_before_copy(self):
        rows = flow.rows_csv(self.annotations)
        for key, value in (("goal_time_s", "nan"), ("goal_time_s", "2"),
                           ("filename", "../outside.mp4"), ("video_id", "../bad")):
            bad = copy.deepcopy(rows)
            bad[0][key] = value
            flow.write_csv(self.base / "bad.csv", bad, list(bad[0]))
            with self.assertRaises(ValueError):
                flow.prepare(self.inputs, self.base / "bad.csv", self.base / "bad-export", "match")
            self.assertFalse((self.base / "bad-export").exists())

    def test_all_resumes_in_separate_interpreters_without_gpu_or_lock_conflict(self):
        checkpoint = self.base / "fake-weights"
        checkpoint.write_bytes(b"CPU integration only")
        repo = Path(__file__).resolve().parents[1]
        for video in flow.manifest(self.root)["videos"]:
            folder = self.root / "track" / video["video_id"]
            atomic_json(folder / "tracking_complete.json", dict(
                protocol=dict(checkpoint_sha256=sha256_file(checkpoint),
                    config_sha256=sha256_file(repo / "MOTIP/configs/r50_deformable_detr_motip_sportsmot.yaml"),
                    tracker_options="cli.tracking defaults v1", source_sha256=video["source_sha256"]),
                outputs=flow.track_signature(self.root, video["video_id"]),
                visualization_sha256=sha256_file(folder / "tracks_vis.mp4")))
            path = self.root / "identity" / video["video_id"] / "feature_contract.json"
            contract = flow.read_json(path)
            contract["protocol"] = flow.feature_protocol(self.root, video, checkpoint,
                repo / "configs/kpr/multidataset_sports_test.yaml")
            atomic_json(path, contract)
        env = dict(os.environ, BASKET_MOTIP_PYTHON=sys.executable, BASKET_KPR_PYTHON=sys.executable,
                   BASKET_MOTIP_CHECKPOINT=str(checkpoint), BASKET_KPR_CHECKPOINT=str(checkpoint))
        result = subprocess.run([sys.executable, "-m", "cli.export_goal_test", "--stage", "all",
            "--input-dir", str(self.inputs), "--annotations", str(self.annotations), "--output-dir", str(self.root),
            "--match-id", "same-game", "--samples-per-track", "4"], cwd=repo, env=env,
            capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("verified existing result", result.stdout)
        self.assertTrue(self.root.with_suffix(".tar").is_file())


if __name__ == "__main__":
    unittest.main()
