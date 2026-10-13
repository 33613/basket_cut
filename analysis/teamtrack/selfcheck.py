"""CPU regression fixtures for CSV conversion, TrackEval and resume safeguards."""
import csv
import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path

import cv2
import numpy as np

from .dataset import ATTRS, parse_annotations, prepare, save_json, sha256
from .metrics import evaluate, evaluate_files
from .inference import infer


class TeamTrackChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def annotation(self, path, frames=4):
        # Duplicate jersey-like PlayerID across teams must remain distinct; BALL excluded.
        rows = [['TeamID'] + ['0']*4 + ['1']*4 + ['BALL']*4,
                ['PlayerID'] + ['1']*8 + ['BALL']*4,
                ['Attributes'] + list(ATTRS)*3, ['frame'] + ['']*12]
        rows.extend([[i, 2, 2, 8, 8, 20, 2, 8, 8, 1, 1, 2, 2] for i in range(1, frames+1)])
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', newline='') as stream:
            csv.writer(stream).writerows(rows)
        return rows

    def test_csv_identity_and_ball(self):
        path = self.root / 'a.csv'
        self.annotation(path)
        frames, mapping, absent = parse_annotations(path)
        self.assertEqual(len(frames), 4)
        self.assertEqual(len(mapping), 2)
        self.assertEqual([b[0] for b in frames[0]], [1, 2])
        self.assertEqual(absent, 0)

    def test_missing_box_and_partial_rejection(self):
        path = self.root / 'a.csv'
        rows = self.annotation(path)
        rows[4][1:5] = ['nan']*4
        with path.open('w', newline='') as stream:
            csv.writer(stream).writerows(rows)
        frames, _, absent = parse_annotations(path)
        self.assertEqual((len(frames[0]), absent), (1, 1))
        rows[4][1] = 2
        with path.open('w', newline='') as stream:
            csv.writer(stream).writerows(rows)
        with self.assertRaisesRegex(ValueError, 'invalid/partial'):
            parse_annotations(path)

    def test_wrong_frame_base_rejected(self):
        path = self.root / 'a.csv'
        rows = self.annotation(path)
        rows[4][0] = 0
        with path.open('w', newline='') as stream:
            csv.writer(stream).writerows(rows)
        with self.assertRaisesRegex(ValueError, 'one-based'):
            parse_annotations(path)

    def raw_dataset(self, video_frames=4):
        root = self.root / 'raw'
        folder = root / 'basketball_top/val'
        (folder / 'videos').mkdir(parents=True)
        self.annotation(folder / 'annotations/Q4_top_0-1.csv')
        writer = cv2.VideoWriter(str(folder / 'videos/Q4_top_0-1.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 4, (40, 32))
        self.assertTrue(writer.isOpened())
        for _ in range(video_frames):
            writer.write(np.zeros((32, 40, 3), dtype=np.uint8))
        writer.release()
        (root / 'selection.csv').write_text('view,split,video\nbasketball_top,val,Q4_top_0-1.mp4\n')
        return root

    def test_video_prepare_resume_and_changed_source(self):
        root = self.raw_dataset()
        output = self.root / 'out'
        result = prepare(root, output, 'val')
        self.assertEqual(result['sequences'][0]['gt_boxes'], 8)
        self.assertEqual(result, prepare(root, output, 'val'))
        self.assertEqual(len(list((output / 'val/alignment').rglob('*.jpg'))), 3)
        path = root / 'basketball_top/val/annotations/Q4_top_0-1.csv'
        path.write_text(path.read_text().replace('20,2', '21,2'))
        with self.assertRaisesRegex(ValueError, 'Dataset changed'):
            prepare(root, output, 'val')

    def test_video_frame_mismatch_rejected(self):
        root = self.raw_dataset(video_frames=3)
        with self.assertRaisesRegex(ValueError, 'refusing misaligned'):
            prepare(root, self.root / 'out', 'val')
        self.assertFalse((self.root / 'out/val/dataset.json').exists())

    def mot_sequence(self, name, view, gt, prediction, frames):
        folder = self.root / 'val'
        gt_path = folder / 'gt' / name / 'gt/gt.txt'
        gt_path.parent.mkdir(parents=True, exist_ok=True)
        pred_path = folder / 'predictions/motip/data' / f'{name}.txt'
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        def render(observations, tail):
            return ''.join(f'{frame},{identity},{x},0,10,10,{tail}\n' for frame, identity, x in observations)
        gt_path.write_text(render(gt, '1,1,1'))
        pred_path.write_text(render(prediction, '1,-1,-1,-1'))
        return dict(sequence=name, view=view, frames=frames)

    def run_metrics(self, sequences):
        return evaluate_files(self.root / 'val', sequences)

    def test_perfect_scores(self):
        boxes = [(i, 1, 0) for i in range(1, 5)]
        seq = self.mot_sequence('perfect', 'basketball_top', boxes, boxes, 4)
        rows = self.run_metrics([seq])
        overall = rows[-1]
        for name in ('HOTA', 'DetA', 'AssA', 'MOTA', 'IDF1', 'recall', 'precision'):
            self.assertAlmostEqual(overall[name], 100, places=4)
        self.assertEqual((overall['TP'], overall['FP'], overall['FN']), (4, 0, 0))
        with (self.root / 'val/metrics/hota_thresholds.csv').open() as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 19*3)

    def test_id_fragmentation_and_weighted_grouping(self):
        a = [(i, 1, 0) for i in range(1, 5)]
        b = [(i, 1, 0) for i in range(1, 9)]
        pred_b = [(i, 1 if i <= 4 else 2, 0) for i in range(1, 9)]
        rows = self.run_metrics([self.mot_sequence('short', 'basketball_top', a, a, 4),
                                 self.mot_sequence('long', 'basketball_side', b, pred_b, 8)])
        overall = rows[-1]
        self.assertEqual(overall['IDSW'], 1)
        self.assertAlmostEqual(overall['IDF1'], 100*2/3)
        self.assertAlmostEqual(overall['AssA'], 100*2/3, places=4)
        self.assertAlmostEqual(overall['HOTA'], 100*np.sqrt(2/3), places=4)
        self.assertAlmostEqual(overall['MOTA'], 100*11/12)
        self.assertNotAlmostEqual(overall['IDF1'], (rows[0]['IDF1']+rows[1]['IDF1'])/2)

    def test_frame_trace_misses_false_positives_and_switch(self):
        gt = [(1, 1, 0), (2, 1, 0), (4, 1, 0)]
        pred = [(1, 8, 0), (1, 9, 100), (3, 9, 100), (4, 10, 0)]
        row = self.run_metrics([self.mot_sequence('errors', 'basketball_top', gt, pred, 5)])[-1]
        self.assertEqual((row['TP'], row['FP'], row['FN'], row['IDSW']), (2, 2, 1, 1))
        self.assertAlmostEqual(row['recall'], 200/3)
        self.assertEqual(row['precision'], 50)
        with (self.root / 'val/metrics/sequences/errors/clear_frames.csv').open() as stream:
            frames = list(csv.DictReader(stream))
        self.assertEqual(len(frames), 5)  # Last empty frame is preserved.
        self.assertEqual(frames[-1]['cumulative_FN'], '1')
        self.assertEqual(frames[1]['FN'], '1')

    def test_empty_predictions(self):
        row = self.run_metrics([self.mot_sequence('empty', 'basketball_top', [(1, 1, 0)], [], 3)])[-1]
        self.assertEqual((row['TP'], row['FN'], row['IDFN']), (0, 1, 1))
        self.assertEqual(row['HOTA'], 0)

    def test_empty_ground_truth(self):
        row = self.run_metrics([self.mot_sequence('emptygt', 'basketball_top', [], [(1, 1, 0)], 3)])[-1]
        self.assertEqual((row['FP'], row['IDFP']), (1, 1))

    def test_duplicate_prediction_id_rejected(self):
        seq = self.mot_sequence('dup', 'basketball_top', [(1, 1, 0)], [(1, 1, 0), (1, 1, 20)], 1)
        with self.assertRaises(Exception):
            self.run_metrics([seq])

    def test_evaluation_requires_complete_and_unchanged_files(self):
        root = self.raw_dataset()
        output = self.root / 'out'
        manifest = prepare(root, output, 'val')
        folder = output / 'val'
        protocol = {'dataset_sha256': sha256(folder / 'dataset.json')}
        save_json(folder / 'inference_protocol.json', protocol)
        with self.assertRaises(FileNotFoundError):
            evaluate(output, 'val')
        seq = manifest['sequences'][0]
        pred = folder / 'predictions/motip/data' / (seq['sequence'] + '.txt')
        pred.parent.mkdir(parents=True)
        pred.write_text('')
        save_json(folder / 'completed' / (seq['sequence'] + '.json'), {
            'frames': 4, 'protocol_sha256': sha256(folder / 'inference_protocol.json'),
            'prediction_sha256': sha256(pred)})
        self.assertEqual(evaluate(output, 'val')[-1]['FN'], 8)
        pred.write_text('1,1,0,0,1,1,1,-1,-1,-1\n')
        with self.assertRaisesRegex(ValueError, 'prediction checksum'):
            evaluate(output, 'val')

    def fake_model_modules(self, fail_on_frame=None):
        calls = {'loads': 0, 'trackers': 0}
        class Backend:
            def __init__(self, **kwargs):
                calls['loads'] += 1
                self.assert_settings = kwargs
            def create_tracker(self, **kwargs):
                calls['trackers'] += 1
                return {'frame': 0}
            def infer_frame(self, tracker, frame):
                tracker['frame'] += 1
                if tracker['frame'] == fail_on_frame:
                    raise RuntimeError('simulated interrupted inference')
                return [dict(track_id=0, bbox_xyxy=[2., 2., 10., 10.], det_score=.9)]
        modules = {
            'torch': SimpleNamespace(__version__='fixture', manual_seed=lambda _: None),
            'torchvision': SimpleNamespace(__version__='fixture'),
            'pipeline.tracking.motip_backend': SimpleNamespace(
                MotipBackend=Backend,
                MotipRuntimeConfig=lambda **kwargs: SimpleNamespace(to_dict=lambda: kwargs)),
        }
        modules['torch'].version = SimpleNamespace(cuda=None)
        return modules, calls

    def inference_fixture(self):
        root = self.raw_dataset()
        output = self.root / 'out'
        prepare(root, output, 'val')
        checkpoint = self.root / 'fixture.pth'
        checkpoint.write_bytes(b'fake checkpoint for orchestration checks only')
        return output, checkpoint

    def test_inference_complete_resume_and_tamper(self):
        output, checkpoint = self.inference_fixture()
        modules, calls = self.fake_model_modules()
        with patch.dict('sys.modules', modules):
            infer(output, 'val', checkpoint, device='cpu')
            infer(output, 'val', checkpoint, device='cpu')
            self.assertEqual(calls, {'loads': 1, 'trackers': 1})
            pred = next((output / 'val/predictions/motip/data').glob('*.txt'))
            pred.write_text('corrupt')
            with self.assertRaisesRegex(ValueError, 'modified'):
                infer(output, 'val', checkpoint, device='cpu')

    def test_interrupted_inference_restarts_whole_sequence(self):
        output, checkpoint = self.inference_fixture()
        modules, _ = self.fake_model_modules(fail_on_frame=2)
        with patch.dict('sys.modules', modules):
            with self.assertRaisesRegex(RuntimeError, 'simulated'):
                infer(output, 'val', checkpoint, device='cpu')
        self.assertFalse(list((output / 'val/completed').glob('*.json')))
        self.assertTrue(list((output / 'val/predictions/motip/data').glob('*.part')))
        modules, _ = self.fake_model_modules()
        with patch.dict('sys.modules', modules):
            infer(output, 'val', checkpoint, device='cpu')
        row = evaluate(output, 'val')[-1]
        self.assertEqual((row['TP'], row['FN']), (4, 4))

    def test_checkpoint_change_refuses_mixed_run(self):
        output, checkpoint = self.inference_fixture()
        modules, _ = self.fake_model_modules()
        with patch.dict('sys.modules', modules):
            infer(output, 'val', checkpoint, device='cpu')
            checkpoint.write_bytes(b'changed checkpoint')
            with self.assertRaisesRegex(ValueError, 'protocol changed'):
                infer(output, 'val', checkpoint, device='cpu')


def run_selfcheck():
    return unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(TeamTrackChecks))
