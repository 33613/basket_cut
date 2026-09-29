import json
import tempfile
import unittest
from pathlib import Path

from pipeline.common.schema import (
    TrackRecord,
    index_track_records,
    normalize_xyxy,
    read_jsonl,
    write_jsonl_line,
)


class TrackSchemaTest(unittest.TestCase):
    def test_box_is_clipped_to_frame(self):
        self.assertEqual(
            normalize_xyxy((-2, 4, 105, 60), frame_width=100, frame_height=50),
            (0.0, 4.0, 100.0, 50.0),
        )

    def test_invalid_box_is_rejected(self):
        with self.assertRaises(ValueError):
            normalize_xyxy((10, 10, 5, 20))

    def test_jsonl_round_trip_and_index(self):
        records = [
            TrackRecord("video", 2, 0.08, 9, (1, 2, 3, 4), 0.8, 0),
            TrackRecord("video", 1, 0.04, 3, (2, 3, 4, 5), 0.7, 0),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tracks.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                for record in records:
                    write_jsonl_line(handle, record.to_dict())
            loaded = list(read_jsonl(path))
        indexed = index_track_records(loaded)
        self.assertEqual(sorted(indexed), [1, 2])
        self.assertEqual(indexed[2][0].track_id, 9)
        self.assertEqual(loaded[0]["bbox_xyxy"], [1.0, 2.0, 3.0, 4.0])


if __name__ == "__main__":
    unittest.main()

