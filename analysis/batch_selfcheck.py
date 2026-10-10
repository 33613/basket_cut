"""Synthetic checks for batch execution, bounded extraction and jersey abstention."""

import io
import json
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from dataclasses import replace

from contracts.execution import ExecutionSettings
from contracts.schema import write_json
from tools.datasets.video_archive import extract_videos
from workflows.batch import run_batch, select_inputs
from analysis.evaluation.run_summary import summarize_run


class BatchChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='basket-batch-check-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / 'data'; self.data.mkdir()
        (self.data / 'a.mp4').write_bytes(b'synthetic-video')
        weight = self.root / 'model.pth'; weight.write_bytes(b'synthetic-weight')
        self.settings = replace(ExecutionSettings(), motip_python=Path(sys.executable), kpr_python=Path(sys.executable),
                                review_python=Path(sys.executable), motip_checkpoint=weight, kpr_checkpoint=weight)
        self.calls = []

    def fake(self, command, paths):
        stage = command[2]
        self.calls.append(stage)
        if stage == 'cli.tracking':
            paths['tracking'].mkdir(parents=True, exist_ok=True)
            paths['tracks'].write_text('')
            write_json(paths['video_meta'], {'video_id': 'synthetic', 'fps': 25, 'processed_frames': 1,
                                           'frame_count': 1, 'width': 32, 'height': 32, 'input_path': str(paths['source'])})
        elif stage == 'cli.track_quality':
            paths['quality'].mkdir(exist_ok=True)
            paths['stable_tracks'].write_text('')
            (paths['quality'] / 'quality_tracks.jsonl').write_text('')
            write_json(paths['quality'] / 'quality_summary.json', {})
        elif stage == 'cli.resolve_identity':
            paths['identity'].mkdir(exist_ok=True)
            paths['identity_map'].write_text('')
            (paths['identity'] / 'identities.jsonl').write_text('')
            write_json(paths['identity'] / 'resolution_summary.json', {})
        elif stage == 'cli.jersey_qwen':
            write_json(paths['identity'] / 'jersey_summary.json', {'backend': 'qwen_vl', 'provenance': {}, 'thresholds': {}})
            (paths['identity'] / 'jersey_tracks.jsonl').write_text('')
        elif stage == 'cli.build_person_library':
            from workflows.person_library import PersonLibraryOptions, build_person_library
            build_person_library(PersonLibraryOptions(paths['match_root'], match_id='outputs',
                clip_names=(paths['output'].name,), use_jersey_evidence=True, max_distance=.2))
            write_json(paths['players'], {})
        elif stage == 'cli.render_results':
            paths['visualization'].mkdir(exist_ok=True)
            paths['result'].write_bytes(b'synthetic-result')
        elif stage == 'cli.link_events':
            paths['linked_actions'].write_text('')
        elif stage == 'cli.aggregate_events':
            self.assertEqual(command[command.index('--input') + 1], str(paths['actions']))
            paths['events'].write_text('')
        elif stage == 'cli.prepare_track_review':
            paths['review'].mkdir(parents=True, exist_ok=True)
            write_json(paths['review'] / 'index.json', {})
        else:
            self.fail(f'Unexpected model invocation: {stage}')
        return 0

    def test_resume_and_empty_tracks_never_invoke_kpr(self):
        run = self.root / 'outputs'
        first = run_batch(self.settings, self.data, run, target='identity', execute=self.fake)
        self.assertEqual(first['completed_clips'], 1)
        original = list(self.calls)
        second = run_batch(self.settings, self.data, run, target='identity', execute=self.fake)
        self.assertEqual(second['completed_clips'], 1)
        self.assertEqual(original, self.calls)
        metrics = summarize_run(run, run / 'run_evaluation.json')
        self.assertIsNone(metrics['headline']['pure_track_rate'])
        self.assertIsNone(metrics['completed_subset']['identity_pairs']['f1'])

    def test_full_plan_keeps_track_events_before_registration_and_render(self):
        settings = replace(self.settings, action_python=Path(sys.executable),
                           action_checkpoint=self.settings.motip_checkpoint,
                           action_config=self.settings.motip_checkpoint,
                           action_label_map=self.settings.motip_checkpoint)
        run = self.root / 'outputs'
        plan = run_batch(settings, self.data, run, target='full', dry_run=True)
        commands = next(iter(plan['commands'].values()))
        aggregation = commands['aggregate']
        self.assertTrue(aggregation[aggregation.index('--input') + 1].endswith('actions.jsonl'))
        self.assertIn('--identity-map', commands['render'])
        self.assertIn('--actions', commands['render'])
        result = run_batch(settings, self.data, run, target='full', execute=self.fake)
        self.assertEqual(result['completed_clips'], 1)
        self.assertLess(self.calls.index('cli.aggregate_events'), self.calls.index('cli.build_person_library'))
        self.assertNotIn('cli.action', self.calls)  # Empty trajectories must not invoke the model.

    def test_failed_clip_is_reported_and_retry_resumes_upstream(self):
        run = self.root / 'outputs'
        def failing(command, paths):
            if command[2] == 'cli.track_quality':
                return 7
            return self.fake(command, paths)
        result = run_batch(self.settings, self.data, run, target='identity', execute=failing)
        self.assertEqual(len(result['failed_clips']), 1)
        run_batch(self.settings, self.data, run, target='identity', execute=self.fake)
        self.assertEqual(self.calls.count('cli.tracking'), 1)

    def test_new_clips_append_without_removing_old_manifest_members(self):
        run = self.root / 'outputs'
        run_batch(self.settings, self.data, run, target='identity', execute=self.fake)
        (self.data / 'b.mp4').write_bytes(b'new-synthetic-video')
        result = run_batch(self.settings, self.data, run, target='identity', execute=self.fake)
        self.assertEqual(result['completed_clips'], 2)
        self.assertEqual(self.calls.count('cli.tracking'), 2)
        manifest = json.loads((run / 'batch_manifest.json').read_text())
        self.assertEqual(len(manifest['clips']), 2)
        run_batch(self.settings, self.data, run, limit=1, target='identity', execute=self.fake)
        self.assertEqual(len(json.loads((run / 'batch_manifest.json').read_text())['clips']), 2)

    def test_identity_run_can_gain_events_without_reextracting_identity(self):
        settings = replace(self.settings, action_python=Path(sys.executable),
            action_checkpoint=self.settings.motip_checkpoint, action_config=self.settings.motip_checkpoint,
            action_label_map=self.settings.motip_checkpoint)
        run = self.root / 'outputs'
        first = run_batch(settings, self.data, run, target='identity', execute=self.fake)
        self.assertEqual(first['completed_clips'], 1)
        second = run_batch(settings, self.data, run, target='full', execute=self.fake)
        self.assertEqual(second['completed_clips'], 1)
        self.assertEqual(self.calls.count('cli.tracking'), 1)
        self.assertEqual(self.calls.count('cli.resolve_identity'), 1)
        self.assertEqual(self.calls.count('cli.jersey_qwen'), 1)
        self.assertEqual(self.calls.count('cli.build_person_library'), 2)

    def test_changed_video_and_option_refuse_silent_resume(self):
        run = self.root / 'outputs'
        run_batch(self.settings, self.data, run, target='identity', execute=self.fake)
        with self.assertRaises(ValueError):
            run_batch(self.settings, self.data, run, target='identity', options={'identity_merge_distance': .2}, execute=self.fake)
        (self.data / 'a.mp4').write_bytes(b'changed-video')
        with self.assertRaises(ValueError):
            run_batch(self.settings, self.data, run, target='identity', execute=self.fake)

    def test_output_input_overlap_and_selection_escape_rejected(self):
        with self.assertRaises(ValueError):
            run_batch(self.settings, self.data, self.data / 'outputs', target='identity', dry_run=True)
        selection = self.root / 'selection.json'
        write_json(selection, {'videos': ['../not-a-video.mp4']})
        with self.assertRaises(ValueError):
            select_inputs(self.data, 30, selection)

    def test_missing_expected_clip_not_dropped_from_summary(self):
        run = self.root / 'outputs'
        run_batch(self.settings, self.data, run, target='identity', execute=self.fake)
        manifest = json.loads((run / 'batch_manifest.json').read_text())
        manifest['clips'].append({'name': 'missing'})
        write_json(run / 'batch_manifest.json', manifest)
        result = summarize_run(run, run / 'run_evaluation.json')
        self.assertEqual(result['expected_clips'], 2)
        self.assertEqual(len(result['failed_clips']), 1)

    def archive(self, names):
        path = self.root / 'videos.tar'
        with tarfile.open(path, 'w') as archive:
            for name in names:
                value = b'synthetic-video'
                entry = tarfile.TarInfo(name); entry.size = len(value)
                archive.addfile(entry, io.BytesIO(value))
        return path

    def test_extraction_is_bounded_and_manifest_is_reusable(self):
        archive = self.archive(['basketball/c.mp4', 'basketball/a.mp4', 'basketball/b.mp4'])
        out = self.root / 'extracted'
        result = extract_videos(archive, out, limit=2)
        self.assertEqual(result['selected_videos'], 2)
        self.assertFalse((out / 'basketball/c.mp4').exists())
        self.assertEqual(len(select_inputs(out, 30, Path(result['manifest']))), 2)
        self.assertEqual(extract_videos(archive, out, limit=2)['selected_videos'], 2)

    def test_unsafe_and_over_budget_archive_rejected(self):
        with self.assertRaises(ValueError):
            extract_videos(self.archive(['../escape.mp4']), self.root / 'out')
        with self.assertRaises(ValueError):
            extract_videos(self.archive(['a.mp4']), self.root / 'out', max_gb=1e-10)

    def test_private_environment_loader_does_not_evaluate_shell(self):
        from cli.serve_web import load_environment
        import os
        from unittest.mock import patch
        env = self.root / 'private.env'
        env.write_text('BASKET_RUNTIME_ROOT="runtime/test path"\nBASKET_FFMPEG=$(not_a_command)\n')
        with patch.dict(os.environ):
            load_environment(env)
            self.assertEqual(os.environ['BASKET_RUNTIME_ROOT'], 'runtime/test path')
            self.assertEqual(os.environ['BASKET_FFMPEG'], '$(not_a_command)')
        env.write_text('PATH=unsafe\n')
        with self.assertRaises(ValueError):
            load_environment(env)

    def test_python_environment_symlink_is_preserved(self):
        from contracts.execution import _python_env
        from unittest.mock import patch
        import os
        link = self.root / 'venv-python'
        link.symlink_to(sys.executable)
        with patch.dict(os.environ, {'BASKET_REVIEW_PYTHON': str(link)}):
            self.assertEqual(_python_env('BASKET_REVIEW_PYTHON'), link)
            self.assertNotEqual(_python_env('BASKET_REVIEW_PYTHON'), link.resolve())


def main():
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(BatchChecks))
    if not result.wasSuccessful():
        raise SystemExit(1)
