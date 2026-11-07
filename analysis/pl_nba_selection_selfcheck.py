"""Local-file matching checks; video contents need not be decoded for selection."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from tools.datasets.select_pl_nba import select


class SelectionChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.videos = self.root / "videos"
        self.videos.mkdir()
        self.inventory = self.root / "made.jsonl"

    def events(self, ids):
        rows = [
            {
                "event_id": "GAME__" + vid,
                "match_id": "GAME",
                "video_id": vid,
                "original_video_basename": vid + "_attack_successed",
                "result": "made",
                "timestamp_video_raw": {"start": "01:30", "end": "03:50"},
                "action_label": "Layup",
                "evaluation_player_name": "Player",
            }
            for vid in ids
        ]
        self.inventory.write_text("".join(json.dumps(row) + "\n" for row in rows))

    def test_exact_clip_matching_hashes_and_blank_goal_times(self):
        self.events(["3", "30"])
        for name in ("3_attack_successed.mp4", "30.mp4", "31.mp4"):
            (self.videos / name).write_bytes(name.encode())
        before = {p: p.read_bytes() for p in self.videos.iterdir()}
        output = self.root / "selection"
        result = select(self.inventory, self.videos, output)
        self.assertEqual(result["matched_videos"], 2)
        self.assertEqual(
            set((output / "upload_files.txt").read_text().splitlines()),
            {"3_attack_successed.mp4", "30.mp4"},
        )
        with (output / "timing_review.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertTrue(
            all(
                row["goal_time_s"] == "" and row["timing_checked"] == "" for row in rows
            )
        )
        self.assertAlmostEqual(float(rows[0]["frames60_action_end_s"]), 3 + 50 / 60)
        self.assertAlmostEqual(float(rows[0]["centiseconds_action_end_s"]), 3.5)
        self.assertEqual(before, {p: p.read_bytes() for p in self.videos.iterdir()})

    def test_missing_and_ambiguous_files_do_not_emit_partial_upload_lists(self):
        self.events(["3", "30"])
        (self.videos / "30.mp4").write_bytes(b"video")
        with self.assertRaisesRegex(ValueError, "missing/ambiguous"):
            select(self.inventory, self.videos, self.root / "missing")
        self.assertFalse((self.root / "missing/upload_files.txt").exists())
        (self.videos / "3.mp4").write_bytes(b"video")
        (self.videos / "3_attack_successed.mp4").write_bytes(b"duplicate")
        with self.assertRaisesRegex(ValueError, "missing/ambiguous"):
            select(self.inventory, self.videos, self.root / "ambiguous")
        self.assertFalse((self.root / "ambiguous/upload_files.txt").exists())

    def test_external_symlink_is_not_uploaded(self):
        self.events(["3"])
        outside = self.root / "outside.mp4"
        outside.write_bytes(b"not this dataset")
        (self.videos / "3.mp4").symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            select(self.inventory, self.videos, self.root / "unsafe")


if __name__ == "__main__":
    unittest.main()
