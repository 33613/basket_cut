"""CPU checks for native-frame preparation and parity report failure modes."""
import contextlib
import io
import json
import tempfile
import unittest
import sys
import subprocess
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from analysis.teamtrack.dataset import save_json, sha256
from .dataset import prepare, verify_sources
from .evaluation import compare_predictions, evaluate
from .runner import official_command, prediction_rows


class SportsMotChecks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data = self.root / 'data'
        self.output = self.root / 'out'
        self.data.mkdir()
        (self.data / 'basketball.txt').write_text('v_a_c001\nv_a_c003\nv_train_c001\n')
        (self.data / 'val.txt').write_text('v_a_c001\nv_a_c003\nv_football_c001\n')
        for name in ('v_a_c001', 'v_a_c003'):
            folder = self.data / 'SportsMOT/val' / name
            (folder / 'img1').mkdir(parents=True)
            (folder / 'gt').mkdir()
            for frame in range(1, 13):
                Image.new('RGB', (64, 48), (frame*10, 30, 20)).save(folder / 'img1' / f'{frame:06d}.jpg')
            (folder / 'seqinfo.ini').write_text(f'[Sequence]\nname={name}\nimDir=img1\nframeRate=25\nseqLength=12\nimWidth=64\nimHeight=48\nimExt=.jpg\n')
            (folder / 'gt/gt.txt').write_text(''.join(f'{i},0,2,3,10,20,1,1,1\n' for i in range(1, 13)))

    def test_prepare_original_frames_limit_and_resume(self):
        result = prepare(self.data, self.output, 1)
        self.assertEqual([s['sequence'] for s in result['sequences']], ['v_a_c001'])
        self.assertEqual(result, prepare(self.data, self.output, 1))
        self.assertTrue((self.output / 'reference-input/v_a_c001/SportsMOT/val/v_a_c001').is_symlink())
        seq = result['sequences'][0]
        self.assertEqual(seq['gt_boxes'], 12)
        verify_sources(seq)
        self.assertEqual(sha256(Path(seq['source']) / 'gt/gt.txt'), sha256(self.output / 'official/val/gt/v_a_c001/gt/gt.txt'))

    def test_all_selection_and_nested_archive_root(self):
        result = prepare(self.root, self.output, 0)
        self.assertEqual(len(result['sequences']), 2)
        self.assertEqual(result['official_basketball_val_count'], 2)

    def test_selection_change_rejected(self):
        prepare(self.data, self.output, 1)
        with self.assertRaisesRegex(ValueError, 'selection changed'):
            prepare(self.data, self.output, 0)

    def test_missing_frame_rejected(self):
        (self.data / 'SportsMOT/val/v_a_c001/img1/000002.jpg').unlink()
        with self.assertRaisesRegex(ValueError, 'filenames/count'):
            prepare(self.data, self.output, 1)

    def test_changed_source_rejected(self):
        seq = prepare(self.data, self.output, 1)['sequences'][0]
        Image.new('RGB', (64, 48), 'white').save(Path(seq['source']) / 'img1/000002.jpg')
        with self.assertRaisesRegex(ValueError, 'source frame changed'):
            verify_sources(seq)

    def test_wrong_roster_and_ambiguous_copy(self):
        (self.data / 'other').mkdir()
        (self.data / 'other/basketball.txt').write_text('v_other\n')
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            prepare(self.data, self.output, 1)

    def test_nonbasketball_sequence_rejected(self):
        (self.data / 'SportsMOT/val/v_football_c001').mkdir()
        with self.assertRaisesRegex(ValueError, 'outside the official'):
            prepare(self.data, self.output, 1)

    def test_duplicate_gt_and_ignore_protocol_review(self):
        gt = self.data / 'SportsMOT/val/v_a_c001/gt/gt.txt'
        gt.write_text(gt.read_text() + '1,0,2,3,10,20,1,1,1\n')
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            prepare(self.data, self.output, 1)
        gt.write_text('1,0,2,3,10,20,0,1,1\n')
        with self.assertRaisesRegex(ValueError, 'protocol review'):
            prepare(self.data, self.output, 1)

    def predictions(self, name, ids=(0, 0), x=(2, 2), scores=(1, 1)):
        path = self.root / name
        path.write_text(''.join(f'{i},{identity},{left},3,10,20,{score},-1,-1,-1\n'
                               for i, (identity, left, score) in enumerate(zip(ids, x, scores), 1)))
        return path

    def test_parity_allows_global_id_permutation_and_score_difference(self):
        a = self.predictions('a.txt')
        b = self.predictions('b.txt', ids=(5, 5), scores=(.9, .7))
        result = compare_predictions(a, b, 3)
        self.assertTrue(result['equivalent'])
        self.assertEqual(result['global_id_mapping'], {'0': 5})

    def test_parity_detects_id_switch_and_box_error(self):
        a = self.predictions('a.txt')
        b = self.predictions('b.txt', ids=(5, 6))
        self.assertFalse(compare_predictions(a, b, 2)['equivalent'])
        b = self.predictions('b.txt', x=(2, 3))
        self.assertEqual(compare_predictions(a, b, 2)['geometry_mismatches'], 1)

    def test_parity_detects_missing_prediction_and_duplicate(self):
        a = self.predictions('a.txt')
        b = self.predictions('b.txt', ids=(0,), x=(2,), scores=(1,))
        self.assertEqual(compare_predictions(a, b, 2)['count_mismatch_frames'], 1)
        b.write_text(b.read_text()*2)
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            prediction_rows(b, 2)

    def test_original_command_uses_submit_and_explicit_runtime(self):
        args = official_command('python-native', Path('/models/model.pth'), Path('/data/native'), Path('/outputs/reference'))
        self.assertEqual(args[:3], ['python-native', '-u', 'submit_and_evaluate.py'])
        self.assertIn('--inference-mode', args)
        self.assertIn('submit', args)
        self.assertIn('FP32', args)
        self.assertEqual(args[args.index('--id-thresh')+1], '0.2')
        self.assertEqual(args[args.index('--inference-max-longer')+1], '1440')
        # Parse with the real upstream argparse entry, rather than mirroring flags.
        from MOTIP.runtime_option import runtime_option
        with patch.object(sys, 'argv', args[2:]):
            options = runtime_option()
        self.assertEqual(options.only_detr, 'False')
        self.assertEqual(options.inference_dtype, 'FP32')

    def completed_fixture(self):
        manifest = prepare(self.data, self.output, 1)
        shared = {'dataset_sha256': sha256(self.output / 'dataset.json'), 'checkpoint_sha256': 'fixture-only'}
        save_json(self.output / 'experiment_protocol.json', shared)
        for method in ('official', 'adapter'):
            folder = self.output / method / 'val'
            save_json(folder / 'inference_protocol.json', dict(method=method, **shared))
            pred = folder / 'predictions/motip/data/v_a_c001.txt'
            pred.parent.mkdir(parents=True)
            pred.write_text(''.join(f'{i},4,2,3,10,20,1,-1,-1,-1\n' for i in range(1, 13)))
            save_json(folder / 'completed/v_a_c001.json', dict(
                frames=12, prediction_sha256=sha256(pred), protocol_sha256=sha256(folder / 'inference_protocol.json'),
                preprocessing_probes=[dict(frame=1, pixel_max_abs=0, tensor_max_abs=0., tensor_mean_abs=0.)]))
        return manifest

    def test_real_trackeval_parity_summary_and_completion(self):
        self.completed_fixture()
        result = evaluate(self.output, 'parity')
        self.assertTrue(result['parity_passed'])
        summary = (self.output / 'summary.csv').read_text()
        self.assertIn('100.0', summary)
        self.assertEqual(result['processed_frames_per_method'], 12)
        self.assertTrue((self.output / 'official/val/metrics/sequences/v_a_c001/clear_frames.csv').exists())

    def test_tampered_prediction_refuses_completed_result(self):
        self.completed_fixture()
        (self.output / 'official/val/predictions/motip/data/v_a_c001.txt').write_text('')
        with self.assertRaisesRegex(ValueError, 'changed prediction'):
            evaluate(self.output, 'parity')
        self.assertEqual(json.loads((self.output / 'status.json').read_text())['status'], 'running')

    def test_status_cli_checks_final_summary(self):
        self.completed_fixture()
        evaluate(self.output, 'parity')
        from cli.benchmark_sportsmot import main
        command = ['benchmark_sportsmot', '--output-dir', str(self.output), '--stage', 'status']
        with patch.object(sys, 'argv', command), contextlib.redirect_stdout(io.StringIO()) as stdout:
            main()
        self.assertIn('"parity_passed": true', stdout.getvalue())
        (self.output / 'summary.csv').write_text('changed')
        with patch.object(sys, 'argv', command), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, 'summary checksum'):
                main()

    def test_different_checkpoint_refused(self):
        self.completed_fixture()
        path = self.output / 'adapter/val/inference_protocol.json'
        value = json.loads(path.read_text());value['checkpoint_sha256'] = 'different'
        save_json(path, value)
        with self.assertRaisesRegex(ValueError, 'same checkpoint'):
            evaluate(self.output, 'parity')

    def test_suite_stops_before_full_evaluation_on_parity_failure(self):
        from cli.benchmark_sportsmot_suite import main
        argv = ['suite', '--dataset-root', str(self.data), '--output-root', str(self.output), '--env-file', str(self.root / 'private.env')]
        with patch.object(sys, 'argv', argv), patch('cli.benchmark_sportsmot_suite.subprocess.run', side_effect=subprocess.CalledProcessError(2, 'fixture')) as runner:
            with self.assertRaises(SystemExit):
                main()
        self.assertEqual(runner.call_count, 1)
        self.assertEqual(json.loads((self.output / 'suite_status.json').read_text())['status'], 'stopped')

    def test_suite_orders_parity_before_all_official(self):
        from cli.benchmark_sportsmot_suite import main
        argv = ['suite', '--dataset-root', str(self.data), '--output-root', str(self.output), '--env-file', str(self.root / 'private.env')]
        with patch.object(sys, 'argv', argv), patch('cli.benchmark_sportsmot_suite.subprocess.run') as runner:
            main()
        self.assertEqual(runner.call_count, 2)
        commands = [call.args[0] for call in runner.call_args_list]
        self.assertEqual([c[c.index('--mode')+1] for c in commands], ['parity', 'official'])
        self.assertEqual([c[c.index('--limit')+1] for c in commands], ['5', '0'])
        self.assertEqual(json.loads((self.output / 'suite_status.json').read_text())['status'], 'completed')


def run_selfcheck():
    return unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SportsMotChecks))
