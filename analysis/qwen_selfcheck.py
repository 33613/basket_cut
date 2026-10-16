"""CPU checks for number parsing, independent sampling, caching and abstention."""
import json
import contextlib
import hashlib
import os
import subprocess
import sys
from types import SimpleNamespace
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

    def model_fixture(self, root):
        root.mkdir(parents=True, exist_ok=True)
        assets = {
            'config.json': b'{"model_type":"qwen3_vl"}',
            'preprocessor_config.json': b'{}', 'tokenizer_config.json': b'{}',
            'tokenizer.json': b'{}', 'chat_template.jinja': b'fixture template',
            'model.safetensors': b'fixture for hashing, never loaded as a real model',
        }
        for name, data in assets.items():
            (root / name).write_bytes(data)
        return assets

    def test_qwen3_provenance_rejects_old_architecture_and_hashes_template(self):
        from pipeline.identity.qwen_backend import model_provenance
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.model_fixture(root)
            first = model_provenance(root)
            self.assertEqual(first['architecture'], 'qwen3_vl')
            (root / 'chat_template.jinja').write_text('changed template')
            self.assertNotEqual(first['model_sha256'], model_provenance(root)['model_sha256'])
            (root / 'config.json').write_text('{"model_type":"qwen2_5_vl"}')
            with self.assertRaisesRegex(ValueError, 'Expected Qwen3'):
                model_provenance(root)

    def test_qwen3_loader_uses_bfloat16_local_weights_and_bounded_processors(self):
        from unittest.mock import MagicMock
        from pipeline.identity.qwen_backend import QwenBackend, IMAGE_BUDGET, VIDEO_BUDGET
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary);self.model_fixture(root)
            processor = SimpleNamespace(image_processor=SimpleNamespace(size=None), video_processor=SimpleNamespace(size=None))
            model = MagicMock();model.to.return_value = model;model.eval.return_value = model
            loader = MagicMock(return_value=model)
            proc_loader = MagicMock(return_value=processor)
            fake_torch = SimpleNamespace(device=lambda value: SimpleNamespace(type='cuda'),
                cuda=SimpleNamespace(is_available=lambda: True, is_bf16_supported=lambda: True),
                bfloat16='bf16', float16='fp16', float32='fp32')
            fake_transformers = SimpleNamespace(Qwen3VLForConditionalGeneration=SimpleNamespace(from_pretrained=loader),
                AutoProcessor=SimpleNamespace(from_pretrained=proc_loader))
            with patch.dict(sys.modules, {'torch': fake_torch, 'transformers': fake_transformers}):
                backend = QwenBackend(root)
            self.assertEqual(loader.call_args.kwargs['dtype'], 'bf16')
            self.assertTrue(loader.call_args.kwargs['local_files_only'])
            self.assertFalse(loader.call_args.kwargs['trust_remote_code'])
            self.assertEqual(processor.image_processor.size, IMAGE_BUDGET)
            self.assertEqual(processor.video_processor.size, VIDEO_BUDGET)
            self.assertEqual(backend.device, 'cuda')

    def test_video_sampling_passes_explicit_fps_none_and_local_path(self):
        from unittest.mock import MagicMock
        from pipeline.identity.qwen_backend import QwenBackend
        backend = QwenBackend.__new__(QwenBackend)
        backend.processor = SimpleNamespace(apply_chat_template=MagicMock(return_value='inputs'))
        backend._generate = MagicMock(return_value='video response')
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'window.avi';path.write_bytes(b'fixture')
            self.assertEqual(backend.read_video(path, 'describe', num_frames=8), 'video response')
            kwargs = backend.processor.apply_chat_template.call_args.kwargs
            self.assertEqual(kwargs['num_frames'], 8)
            self.assertIsNone(kwargs['fps'])
            self.assertNotIn('video_load_backend', kwargs)
            for count in (2, 5, 66, True):
                with self.assertRaises(ValueError): backend.read_video(path, 'describe', num_frames=count)
            with self.assertRaises(ValueError): backend.read_video(path, '')

    def test_generation_trims_input_and_uses_greedy_decode(self):
        from unittest.mock import MagicMock
        from pipeline.identity.qwen_backend import QwenBackend
        backend = QwenBackend.__new__(QwenBackend);backend.device='cuda'
        inputs = {'input_ids': np.array([[1,2]])}
        class Batch(dict):
            @property
            def input_ids(self): return self['input_ids']
            def to(self, device): return self
        backend.model = SimpleNamespace(generate=MagicMock(return_value=np.array([[1,2,3,4]])))
        backend.processor = SimpleNamespace(batch_decode=MagicMock(return_value=['decoded']))
        with patch.dict(sys.modules, {'torch': SimpleNamespace(inference_mode=contextlib.nullcontext)}):
            self.assertEqual(backend._generate(Batch(inputs)), 'decoded')
        self.assertFalse(backend.model.generate.call_args.kwargs['do_sample'])
        self.assertEqual(backend.processor.batch_decode.call_args.args[0].tolist(), [[3,4]])

    def test_vision_probe_reuses_inputs_checks_switch_and_restores_on_failure(self):
        from cli.probe_qwen_vision import compare, set_video_budget
        from pipeline.identity.qwen_backend import VIDEO_BUDGET
        processor = SimpleNamespace(video_processor=SimpleNamespace(size=dict(VIDEO_BUDGET),
            patch_size=16, merge_size=2, temporal_patch_size=2))
        self.assertEqual(set_video_budget(processor, None), VIDEO_BUDGET)
        self.assertEqual(set_video_budget(processor, 8192), VIDEO_BUDGET)
        self.assertEqual(processor.video_processor.size, {
            'shortest_edge': VIDEO_BUDGET['shortest_edge'], 'longest_edge': 16777216})
        self.assertEqual(VIDEO_BUDGET['longest_edge'], 4194304)
        for invalid in (128, 8193, True):
            with self.assertRaises(ValueError): set_video_budget(processor, invalid)
        class Model:
            def __init__(self):
                self.config = SimpleNamespace(_attn_implementation='sdpa',
                    vision_config=SimpleNamespace(_attn_implementation='sdpa'),
                    text_config=SimpleNamespace(_attn_implementation='sdpa'))
            def set_attn_implementation(self, mode):
                for config in (self.config, self.config.vision_config, self.config.text_config):
                    config._attn_implementation = mode
        with tempfile.TemporaryDirectory() as temporary:
            model = Model()
            calls = []
            image, video = object(), object()
            def generate(inputs, **kwargs):
                calls.append((inputs, model.config._attn_implementation))
                return 'synthetic response, not a visual prediction'
            backend = SimpleNamespace(model=model, _generate=generate)
            result = compare(backend, image, video, Path(temporary) / 'success')
            self.assertEqual(calls, [(image, 'sdpa'), (video, 'sdpa'), (image, 'eager'), (video, 'eager')])
            self.assertEqual(len(result['queries']), 4)
            self.assertEqual(result['restored_attention'], result['original_attention'])
            calls.clear()
            result = compare(backend, image, video, Path(temporary) / 'sdpa-only', modes=('sdpa',))
            self.assertEqual(calls, [(image, 'sdpa'), (video, 'sdpa')])
            self.assertEqual(result['requested_attention_modes'], ['sdpa'])
            with self.assertRaisesRegex(ValueError, 'distinct attention modes'):
                compare(backend, image, video, Path(temporary) / 'invalid', modes=('sdpa', 'sdpa'))
            def failure(inputs, **kwargs):
                if model.config._attn_implementation == 'eager':
                    raise RuntimeError('synthetic eager failure')
                return 'saved baseline'
            backend._generate = failure
            output = Path(temporary) / 'failed'
            with self.assertRaisesRegex(RuntimeError, 'synthetic eager failure'):
                compare(backend, image, video, output)
            saved = json.loads((output / 'attention-comparison.json').read_text())
            self.assertEqual(saved['status'], 'failed')
            self.assertEqual(saved['queries'][0]['raw_response'], 'saved baseline')
            self.assertEqual(saved['queries'][-1]['error'], 'synthetic eager failure')
            self.assertEqual(saved['restored_attention'], saved['original_attention'])

    def test_gpu_smoke_refuses_other_jobs_but_ignores_own_context(self):
        from pipeline.identity.qwen_backend import ensure_gpu_idle
        with patch('subprocess.run', return_value=SimpleNamespace(stdout='6835, motip-python\n')):
            with self.assertRaisesRegex(RuntimeError, 'GPU is busy'):
                ensure_gpu_idle()
        with patch('subprocess.run', return_value=SimpleNamespace(stdout=f'{os.getpid()}, current-python\n')):
            ensure_gpu_idle()
        with patch('subprocess.run', return_value=SimpleNamespace(stdout='unexpected output')):
            with self.assertRaisesRegex(RuntimeError, 'Cannot interpret'):
                ensure_gpu_idle()

    def test_modelscope_records_require_published_hashes_and_complete_assets(self):
        from cli.download_qwen import asset_records
        with tempfile.TemporaryDirectory() as temporary:
            assets = self.model_fixture(Path(temporary))
        records = [dict(Path=name, Type='blob', Size=len(data), Sha256=hashlib.sha256(data).hexdigest()) for name,data in assets.items()]
        self.assertEqual(len(asset_records(records)), len(assets))
        broken = [dict(r) for r in records];broken[0]['Sha256']='invalid'
        with self.assertRaisesRegex(ValueError, 'valid SHA256'): asset_records(broken)
        with self.assertRaisesRegex(ValueError, 'Incomplete remote'):
            asset_records([r for r in records if r['Path']!='tokenizer.json'])

    def test_modelscope_verification_detects_corruption_and_missing_index_shards(self):
        from cli.download_qwen import verify_files
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary);assets=self.model_fixture(root)
            records = [dict(path=name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()) for name,data in assets.items()]
            verify_files(root, records)
            (root / 'model.safetensors').write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'truncated'): verify_files(root, records)
            (root / 'model.safetensors').write_bytes(assets['model.safetensors'])
            (root / 'model.safetensors.index.json').write_text('{"weight_map":{"x":"missing.safetensors"}}')
            with self.assertRaisesRegex(ValueError, 'unverified shards'): verify_files(root, records)


def main():
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(QwenChecks))
    if not result.wasSuccessful(): raise SystemExit(1)


if __name__ == '__main__': main()
