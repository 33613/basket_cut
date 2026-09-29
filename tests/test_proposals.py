import unittest

from pipeline.action.proposals import group_by_track, nearest_proposals
from pipeline.common.schema import TrackRecord


def record(frame_idx, track_id, score=0.8):
    return TrackRecord(
        video_id="video",
        frame_idx=frame_idx,
        timestamp_s=frame_idx / 25,
        track_id=track_id,
        bbox_xyxy=(1, 2, 10, 20),
        det_score=score,
        category_id=0,
    )


class ProposalTest(unittest.TestCase):
    def test_nearest_observation_is_selected(self):
        tracks = group_by_track([record(8, 1), record(12, 1), record(10, 2)])
        proposals = nearest_proposals(
            tracks,
            center_frame_idx=10,
            max_frame_gap=3,
            min_det_score=0.3,
        )
        self.assertEqual(
            [(item.track.track_id, item.track.frame_idx) for item in proposals],
            [(1, 8), (2, 10)],
        )

    def test_gap_and_score_filters(self):
        tracks = group_by_track([record(1, 1), record(9, 2, score=0.1)])
        proposals = nearest_proposals(
            tracks,
            center_frame_idx=10,
            max_frame_gap=2,
            min_det_score=0.3,
        )
        self.assertEqual(proposals, [])


if __name__ == "__main__":
    unittest.main()

