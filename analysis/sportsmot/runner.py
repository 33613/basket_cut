"""Run the unmodified upstream CLI and native-frame adapter as independent jobs."""
from __future__ import annotations

import csv
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

from analysis.teamtrack.dataset import save_json, sha256
from .dataset import verify_sources

ROOT = Path(__file__).resolve().parents[2]
CONFIG = 'configs/r50_deformable_detr_motip_sportsmot.yaml'


def official_command(python, checkpoint, data_root, output):
    return [str(python), '-u', 'submit_and_evaluate.py', '--config-path', CONFIG,
            '--data-root', str(data_root), '--inference-model', str(checkpoint),
            '--outputs-dir', str(output), '--inference-mode', 'submit',
            '--inference-dataset', 'SportsMOT', '--inference-split', 'val',
            '--inference-dtype', 'FP32', '--inference-max-longer', '1440',
            '--assignment-protocol', 'object-max', '--miss-tolerance', '60',
            '--det-thresh', '0.3', '--newborn-thresh', '0.6', '--id-thresh', '0.2',
            '--area-thresh', '0', '--only-detr', 'False']


def preflight(python):
    # Import the real CLI with the same module search path/cwd as the GPU job.
    # This also checks optional logger dependencies before any costly inference.
    command = [str(python), '-c',
               "import submit_and_evaluate; print('Official MOTIP entry imports successfully')"]
    subprocess.run(command, cwd=ROOT / 'MOTIP', check=True)


def prediction_rows(path, frames):
    seen, result = set(), []
    with Path(path).open(newline='') as stream:
        for line, row in enumerate(csv.reader(stream), 1):
            if len(row) < 7:
                raise ValueError(f'{path}:{line}: invalid prediction row')
            values = list(map(float, row[:7]))
            frame, identity, x, y, width, height, score = values
            if (not all(map(math.isfinite, values)) or frame != int(frame) or identity != int(identity)
                    or not 1 <= frame <= frames or identity < 0 or width <= 0 or height <= 0):
                raise ValueError(f'{path}:{line}: invalid prediction geometry or ID')
            key = (int(frame), int(identity))
            if key in seen:
                raise ValueError(f'{path}:{line}: duplicate prediction ID in frame')
            seen.add(key)
            result.append(values)
    return result


def source_code_hashes():
    paths = [ROOT / 'pipeline/tracking/motip_backend.py']
    paths.extend((ROOT / 'analysis/sportsmot').glob('*.py'))
    paths.extend(ROOT / 'MOTIP' / name for name in ('submit_and_evaluate.py', 'runtime_option.py'))
    for pattern in ('models/**/*.py', 'configs/**/*.yaml', 'configs/**/*.py', 'utils/**/*.py', 'data/**/*.py', 'log/**/*.py'):
        paths.extend((ROOT / 'MOTIP').glob(pattern))
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(set(paths))}


def infer(output, checkpoint, method):
    import cv2
    import numpy as np
    import torch
    import torchvision
    from pipeline.tracking.motip_backend import MotipBackend, MotipRuntimeConfig

    output, checkpoint = Path(output).resolve(), Path(checkpoint).resolve()
    manifest = json.loads((output / 'dataset.json').read_text())
    if method not in ('official', 'adapter'):
        raise ValueError(method)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    runtime = MotipRuntimeConfig(fp16=False)
    protocol = dict(dataset_sha256=sha256(output / 'dataset.json'), checkpoint_sha256=sha256(checkpoint),
                    method=method, runtime=runtime.to_dict(), config=CONFIG, torch=torch.__version__,
                    torchvision=torchvision.__version__, opencv=cv2.__version__,
                    allow_tf32_matmul=False, allow_tf32_cudnn=False,
                    code_sha256=source_code_hashes())
    shared = {key: value for key, value in protocol.items() if key != 'method'}
    shared_path = output / 'experiment_protocol.json'
    if shared_path.exists() and json.loads(shared_path.read_text()) != shared:
        raise ValueError('Cross-method experiment protocol changed; use a new output directory')
    save_json(shared_path, shared)
    folder = output / method / 'val'
    protocol_path = folder / 'inference_protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError('Inference inputs/code/protocol changed; use a new output directory')
    save_json(protocol_path, protocol)
    save_json(folder / 'environment.json', dict(cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
              git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()))
    protocol_hash = sha256(protocol_path)
    predictions = folder / 'predictions/motip/data'
    predictions.mkdir(parents=True, exist_ok=True)
    backend = None
    for seq in manifest['sequences']:
        name = seq['sequence']
        pred = predictions / f'{name}.txt'
        complete = folder / 'completed' / f'{name}.json'
        verify_sources(seq)
        if complete.exists():
            value = json.loads(complete.read_text())
            if value['protocol_sha256'] != protocol_hash or value['prediction_sha256'] != sha256(pred):
                raise ValueError(f'{name}: modified completed prediction or protocol')
            print(f'[{method}] Reusing {name}', flush=True)
            continue
        print(f'[{method}] Starting {name}: {seq["frames"]} native frames', flush=True)
        start = time.monotonic()
        extra = {}
        if method == 'official':
            upstream_root = output / 'upstream-output' / name
            command = official_command(sys.executable, checkpoint, output / 'reference-input' / name, upstream_root)
            log = folder / 'logs' / f'{name}.log'
            log.parent.mkdir(parents=True, exist_ok=True)
            save_json(log.with_suffix('.command.json'), dict(cwd=str(ROOT / 'MOTIP'), argv=command))
            print(f'[{method}] Native upstream CLI log: {log}', flush=True)
            environment = dict(os.environ, WANDB_MODE='disabled')
            with log.open('a') as stream:
                subprocess.run(command, cwd=ROOT / 'MOTIP', env=environment,
                               stdout=stream, stderr=subprocess.STDOUT, check=True)
            source = upstream_root / 'submit/default/SportsMOT/val' / checkpoint.stem / 'tracker' / f'{name}.txt'
            prediction_rows(source, seq['frames'])
            temporary = pred.with_suffix('.txt.part')
            shutil.copyfile(source, temporary)
            temporary.replace(pred)
        else:
            from PIL import Image
            if backend is None:
                backend = MotipBackend(motip_root=ROOT / 'MOTIP', config_path=CONFIG,
                                       checkpoint_path=checkpoint, runtime_config=runtime,
                                       device='cuda:0', preprocessing='upstream-pil')
            random.seed(0)
            np.random.seed(0)
            torch.manual_seed(0)
            tracker = backend.create_tracker(height=seq['height'], width=seq['width'])
            # Compare the adapter preprocessing with the actual upstream SeqDataset.
            from data.seq_dataset import SeqDataset
            frames = [str(Path(seq['source']) / 'img1' / item['name']) for item in seq['images']]
            original = SeqDataset(seq_info=dict(height=seq['height'], width=seq['width']),
                                  image_paths=frames, max_shorter=800, max_longer=1440)
            probes = []
            temporary = pred.with_suffix('.txt.part')
            with temporary.open('w') as stream:
                for index, path in enumerate(frames):
                    frame = cv2.imread(path)
                    if frame is None or frame.shape[:2] != (seq['height'], seq['width']):
                        raise ValueError(f'Cannot decode original frame: {path}')
                    if index in {0, len(frames)//2, len(frames)-1}:
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        with Image.open(path) as image:
                            official_pixels = np.asarray(image)
                        adapter_tensor = backend._pil_transform(Image.fromarray(rgb))
                        official_tensor = original[index][0].tensors[0]
                        delta = (adapter_tensor - official_tensor).abs()
                        probes.append(dict(frame=index+1, pixel_max_abs=int(np.abs(rgb.astype(int)-official_pixels.astype(int)).max()),
                                           tensor_max_abs=float(delta.max()), tensor_mean_abs=float(delta.mean())))
                    results = backend.infer_frame(tracker, frame)
                    for record in results:
                        x, y, x2, y2 = record['bbox_xyxy']
                        stream.write(f'{index+1},{record["track_id"]},{x},{y},{x2-x},{y2-y},{record["det_score"]},-1,-1,-1\n')
                    if index == 0 or (index+1) % 100 == 0:
                        print(f'[{method}] {name}: {index+1}/{len(frames)} frames', flush=True)
            prediction_rows(temporary, seq['frames'])
            temporary.replace(pred)
            extra['preprocessing_probes'] = probes
            del tracker
        rows = prediction_rows(pred, seq['frames'])
        save_json(complete, dict(frames=seq['frames'], detections=len(rows), seconds=time.monotonic()-start,
                                prediction_sha256=sha256(pred), protocol_sha256=protocol_hash, **extra))
        print(f'[{method}] Completed {name}: {len(rows)} detections', flush=True)
