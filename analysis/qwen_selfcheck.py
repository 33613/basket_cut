"""CPU checks for number parsing, independent sampling, caching and abstention."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from contracts.schema import write_jsonl_line, read_jsonl
from pipeline.identity.qwen_backend import parse_reading
from pipeline.identity.jersey_qwen import qwen_consensus, recognize_jerseys_qwen


class QwenChecks(unittest.TestCase):
    def test_zero_double_zero_and_strict_json(self):
        for number in ('0', '00', '15', '99'):
            self.assertEqual(parse_reading(json.dumps({'number': number, 'readable': True}))['number'], number)
        for value in (15, '015', '01', '100', '-1', '１５', '15 or 32', None):
            self.assertIsNone(parse_reading(json.dumps({'number': value, 'readable': True}))['number'])
        self.assertIsNone(parse_reading('the number is 15')['number'])
        self.assertIsNone(parse_reading('{"number":"15","readable":false}')['number'])

    def test_temporal_support_duplicates_and_conflicts(self):
        row = {'number': '15', 'frame_idx': 0, 'timestamp_s': 0, 'rejection_reasons': []}
        self.assertIsNone(qwen_consensus([row, row])['number'])
        self.assertIsNone(qwen_consensus([row, {**row, 'frame_idx': 1, 'timestamp_s': .04}])['number'])
        self.assertEqual(qwen_consensus([row, {**row, 'frame_idx': 10, 'timestamp_s': .4}])['number'], '15')
        self.assertEqual(qwen_consensus([row, {**row, 'number': '32', 'frame_idx': 10, 'timestamp_s': .4}])['status'], 'conflict')

    def test_invalid_sampling_settings(self):
        for gap in (0, -1, float('nan')):
            with self.assertRaises(ValueError):
                qwen_consensus([], min_gap_s=gap)

    def test_real_video_sampling_and_cache_without_model_download(self):
        import cv2
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = root / 'clip.avi'
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 25, (80, 120))
            self.assertTrue(writer.isOpened())
            rng = np.random.default_rng(8)
            for i in range(21):
                writer.write(rng.integers(0, 255, (120,80,3), dtype=np.uint8))
            writer.release()
            tracks, mapping = root / 'tracks.jsonl', root / 'map.jsonl'
            with tracks.open('w') as handle:
                for frame in (0,10,20):
                    write_jsonl_line(handle, {'video_id': 'clip', 'track_id': 0, 'frame_idx': frame,
                        'timestamp_s': frame/25, 'bbox_xyxy': [0,0,80,120], 'det_score': .9})
            with mapping.open('w') as handle:
                write_jsonl_line(handle, {'raw_track_id': 0, 'person_id': 'P0000'})
            class Backend:
                provenance = {'backend': 'qwen_vl', 'model_sha256': 'synthetic', 'prompt_sha256': 'synthetic'}
                def __init__(self): self.calls = 0
                def read(self, image):
                    self.calls += 1
                    return '{"number":"00","readable":true}'
            backend = Backend()
            output = root / 'identity'
            with patch('pipeline.identity.jersey_qwen.QwenBackend', side_effect=AssertionError('No GPU model in CPU test')):
                result = recognize_jerseys_qwen(video, tracks, mapping, output, root / 'missing-model', backend=backend)
                self.assertEqual(result['sample_count'], 3)
                self.assertEqual(backend.calls, 3)
                recognize_jerseys_qwen(video, tracks, mapping, output, root / 'missing-model', backend=backend)
                self.assertEqual(backend.calls, 3)
            evidence = list(read_jsonl(output / 'jersey_tracks.jsonl'))[0]
            self.assertEqual(evidence['number'], '00')
            self.assertFalse(evidence['verified'])
            self.assertNotIn('raw_score', evidence['readings'][0])
            self.assertTrue(all((output / r['crop_path']).is_file() for r in evidence['readings']))
            self.assertEqual(json.loads(mapping.read_text())['person_id'], 'P0000')

    def test_short_action_windows_keep_real_source_frame_bounds(self):
        from pipeline.action.sampling import sample_frame_indices
        for count in (1, 8, 100):
            for center in (0, count-1):
                indices, padded = sample_frame_indices(center, count, 32, 2)
                self.assertEqual(len(indices), 32)
                self.assertTrue((indices >= 0).all() and (indices < count).all())
                self.assertTrue(padded)
        indices, padded = sample_frame_indices(50, 100, 32, 2)
        self.assertFalse(padded)

    def test_empty_tracks_do_not_load_gpu_or_decode_video(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); mapping = root / 'map.jsonl'; mapping.write_text('')
            with patch('pipeline.identity.jersey_qwen.QwenBackend', side_effect=AssertionError('GPU called')):
                result = recognize_jerseys_qwen(root / 'no-video', root / 'no-tracks', mapping,
                    root / 'identity', root / 'no-model')
            self.assertFalse(result['model_invoked'])
            self.assertEqual(result['sample_count'], 0)


def main():
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(QwenChecks))
    if not result.wasSuccessful(): raise SystemExit(1)


if __name__ == '__main__': main()
