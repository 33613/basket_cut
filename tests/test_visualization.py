import unittest

from pipeline.visualization.timeline import action_text, nearest_action


class VisualizationTest(unittest.TestCase):
    def test_nearest_action_respects_gap(self):
        actions = [{"frame_idx": 4}, {"frame_idx": 12}]
        self.assertEqual(nearest_action(actions, 10, 3)["frame_idx"], 12)
        self.assertIsNone(nearest_action(actions, 20, 3))

    def test_candidate_is_marked_as_unthresholded(self):
        action = {
            "selected_actions": [],
            "action_candidates": [
                {"label": "basketball_pass", "score": 0.12}
            ],
        }
        self.assertIsNone(action_text(action, show_top_candidate=False))
        self.assertEqual(
            action_text(action, show_top_candidate=True),
            "top? basketball_pass 0.12",
        )


if __name__ == "__main__":
    unittest.main()
