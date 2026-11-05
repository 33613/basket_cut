"""CPU integration checks; fake detections do not measure RF-DETR accuracy."""

import json
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from cli.ball_track import Options, run
from cli.pl_nba_annotations import inventory, parse_timestamp
from pipeline.ball.tracker import BallTracker


def detection(x, score=0.8):
    return {"bbox_xyxy": [x, 20, x + 4, 24], "score": score}


class BallTests(unittest.TestCase):
    def test_prediction_expiration_and_new_id(self):
        tracker = BallTracker(0.12)
        first = tracker.update([detection(10)], 0, 100, 100)
        next_point = tracker.update([detection(12)], 0.02, 100, 100)
        self.assertEqual(first["ball_track_id"], next_point["ball_track_id"])
        gap = tracker.update([], 0.04, 100, 100)
        self.assertFalse(gap["observed"])
        self.assertIsNone(gap["score"])
        self.assertIsNone(tracker.update([], 0.20, 100, 100))
        new = tracker.update([detection(70)], 0.22, 100, 100)
        self.assertNotEqual(first["ball_track_id"], new["ball_track_id"])
        with self.assertRaises(ValueError):
            tracker.update([], 0.22, 100, 100)

    def test_motion_gate_rejects_distant_high_confidence(self):
        tracker = BallTracker()
        tracker.update([detection(10)], 0, 100, 100)
        point = tracker.update(
            [detection(11, 0.4), detection(80, 0.99)], 0.02, 100, 100
        )
        self.assertEqual(point["center_xy"], [13, 22])
        gap = tracker.update([detection(80, 0.99)], 0.04, 100, 100)
        self.assertFalse(gap["observed"])

    def test_real_media_with_injected_detector(self):
        class Detector:
            class_names = ("ball", "rim", "holder")

            def predict(self, image, **kwargs):
                self_shape = image.shape
                assert self_shape == (96, 96, 3)
                assert kwargs["threshold"] == 0.1
                return SimpleNamespace(
                    xyxy=np.array([[20, 20, 24, 24], [50, 10, 75, 20]]),
                    confidence=np.array([0.8, 0.9]),
                    class_id=np.array([0, 1]),
                )

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "input.mp4"
            weight = root / "fake.pth"
            weight.write_bytes(b"not a real checkpoint")
            subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "testsrc2=size=96x96:rate=10:duration=4",
                    "-c:v",
                    "libx264",
                    "-threads",
                    "1",
                    str(source),
                ],
                check=True,
            )
            summary = run(
                source,
                weight,
                root / "result",
                Options(start_s=1, end_s=3.8, goal_time_s=3.5),
                model=Detector(),
            )
            self.assertEqual(summary["processed_frames"], 28)
            self.assertFalse(summary["shooter_attributed"])
            rows = [
                json.loads(s)
                for s in (root / "result/ball.jsonl").read_text().splitlines()
            ]
            self.assertEqual(rows[0]["source_frame_idx"], 10)
            self.assertEqual(rows[0]["source_timestamp_s"], 1.0)
            self.assertEqual(rows[-1]["source_frame_idx"], 37)
            self.assertTrue(all(r["ball_track"]["observed"] for r in rows))
            self.assertTrue((root / "result/ball_vis.mp4").is_file())
            with self.assertRaisesRegex(ValueError, "not empty"):
                run(source, weight, root / "result", Options(), model=Detector())
            with self.assertRaisesRegex(ValueError, "inside"):
                run(
                    source, weight, root / "invalid", Options(end_s=5), model=Detector()
                )


class AnnotationTests(unittest.TestCase):
    def test_time_semantics_must_be_explicit(self):
        self.assertIsNone(parse_timestamp("03:30", "raw"))
        self.assertEqual(parse_timestamp("03:30", "seconds-frames", 60), 3.5)
        self.assertEqual(parse_timestamp("03:30", "seconds-centiseconds"), 3.3)
        with self.assertRaises(ValueError):
            parse_timestamp("03:60", "seconds-frames", 60)

    def test_made_event_filter_and_game_scoping(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            archive = root / "data.zip"
            event = {
                "event_id": "ev_1",
                "event_label": "PutBack",
                "result": "made",
                "player": "GT name",
                "timestamp_video": {"start": "02:00", "end": "03:30"},
            }
            with zipfile.ZipFile(archive, "w") as z:
                for game in ["GAME_A", "GAME_B"]:
                    document = {
                        "video_id": game,
                        "possession_id": "poss_3",
                        "events": [
                            event,
                            dict(event, event_id="ev_2", result="missed"),
                            dict(
                                event,
                                event_id="ev_3",
                                event_label="PassBall",
                                result="success",
                            ),
                        ],
                    }
                    z.writestr(f"labels/{game}/3_attack.json", json.dumps(document))
            result = inventory(archive, root / "raw", "GAME_A", ["3"])
            self.assertEqual(result["made_events"], 1)
            row = json.loads((root / "raw/made_events.jsonl").read_text())
            self.assertIsNone(row["goal_time_s"])
            self.assertIsNone(row["action_end_s"])
            self.assertEqual(row["timestamp_video_raw"]["end"], "03:30")
            result = inventory(
                archive, root / "converted", "GAME_A", ["3"], "seconds-frames", 60
            )
            row = json.loads((root / "converted/made_events.jsonl").read_text())
            self.assertEqual(row["action_end_s"], 3.5)
            self.assertIsNone(row["goal_time_s"])
            with self.assertRaisesRegex(ValueError, "requires --match-id"):
                inventory(archive, root / "wrong", video_ids=["3"])
            with self.assertRaisesRegex(ValueError, "Missing clip"):
                inventory(archive, root / "missing", "GAME_A", ["100"])


if __name__ == "__main__":
    unittest.main()
