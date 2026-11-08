"""CPU checks for uncertain annotation clocks, interval anchors and compatibility."""

import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from analysis.link_ball_selfcheck import AttributionTests
from cli.pl_nba_link_input import convert, timestamp_bounds
from pipeline.player_link.events import LinkOptions, load_events
from workflows import link_player as flow


class IntervalTests(unittest.TestCase):
    def test_conversion_keeps_uncertainty_and_never_invents_goal_time(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            review, output = folder/'review.csv', folder/'input.json'
            with review.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=[
                    'event_id', 'video_id', 'filename', 'raw_action_start',
                    'raw_action_end', 'goal_time_s', 'evaluation_player_name'])
                writer.writeheader()
                writer.writerow(dict(event_id='e1', video_id='20', filename='20.mp4',
                    raw_action_start='00:00', raw_action_end='03:30', goal_time_s='',
                    evaluation_player_name='evaluation-only'))
            report = convert(review, output)
            event = json.loads(output.read_text())[0]
            self.assertNotIn('goal_time_s', event)
            self.assertEqual(event['goal_interval_start_s'], 0)
            self.assertEqual(event['goal_interval_end_s'], 5)
            self.assertFalse(report['timestamp_units_confirmed'])
            self.assertEqual(convert(review, output), report)
            with self.assertRaisesRegex(ValueError, 'different'):
                convert(review, output, 2)
            for raw in ('bad', '03:99'):
                with self.assertRaises(ValueError):
                    timestamp_bounds(raw)

    def test_interval_window_clips_and_requires_ball_mode(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            videos = folder/'videos'
            videos.mkdir()
            (videos/'1.mp4').write_bytes(b'fixture')
            annotations = folder/'events.json'
            annotations.write_text(json.dumps([dict(video_id='1', filename='1.mp4',
                goal_interval_start_s=2, goal_interval_end_s=20)]))
            meta = dict(fps=20, width=100, height=100, duration_s=10)
            with patch('pipeline.player_link.events.probe_video', return_value=meta), \
                 patch('pipeline.player_link.events.verify_frame_clock', return_value=200):
                protocol = load_events(videos, annotations, 'match', LinkOptions(max_lookback_s=4))
                event = protocol['events'][0]
                self.assertEqual(event['goal_interval_end_s'], 9.95)
                self.assertEqual(event['window_start_frame'], 0)
                self.assertEqual(event['window_end_frame_exclusive'], 200)
                self.assertIn('not_measured', event['goal_time_semantics'])
                with self.assertRaisesRegex(ValueError, 'require ball'):
                    flow.prepare(videos, annotations, folder/'output', 'match',
                                 LinkOptions(), SimpleNamespace(ffprobe='ffprobe'))

    def test_interval_finds_earlier_anchor_without_exact_goal_and_ignores_identity(self):
        fixture = AttributionTests()
        fixture.setUp()
        event = dict(fixture.event, goal_time_s=9., goal_interval_start_s=5.,
                     goal_interval_end_s=9.,
                     goal_time_semantics='interval_end_reference_not_measured_goal')
        result = fixture.rank(event=event)
        self.assertEqual(result['shooter_track_id'], 0)
        self.assertLess(result['estimated_rim_time_s'], 7)
        self.assertFalse(result['made_shot_verified'])
        self.assertEqual(result, fixture.rank(event=dict(event,
            evaluation_labels={'evaluation_player_name': 'wrong', 'jersey_number': '999'})))
        outside = dict(event, goal_interval_start_s=7.)
        self.assertIsNone(fixture.rank(event=outside)['shooter_track_id'])
        predicted = [dict(r, ball_track=dict(r['ball_track'], observed=False))
                     for r in fixture.ball]
        self.assertIsNone(fixture.rank(event=event, ball_rows=predicted)['shooter_track_id'])


if __name__ == '__main__':
    unittest.main()
