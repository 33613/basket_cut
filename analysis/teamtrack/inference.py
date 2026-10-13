"""Full-frame MOTIP inference; model shared, tracker state reset per sequence."""
import json
import math
import random
import subprocess
import time
from pathlib import Path

from .dataset import save_json, sha256

ROOT = Path(__file__).resolve().parents[2]


def infer(output, split, checkpoint, device='cuda:0'):
    import cv2
    import numpy as np
    import torch
    import torchvision
    from pipeline.tracking.motip_backend import MotipBackend, MotipRuntimeConfig

    output, checkpoint = Path(output).resolve(), Path(checkpoint).resolve()
    folder = output / split
    dataset = json.loads((folder / 'dataset.json').read_text())
    runtime = MotipRuntimeConfig(fp16=False)
    config = 'configs/r50_deformable_detr_motip_sportsmot.yaml'
    code_paths = [ROOT / 'pipeline/tracking/motip_backend.py', Path(__file__), ROOT / 'analysis/teamtrack/dataset.py']
    for pattern in ('models/**/*.py', 'configs/**/*.yaml', 'configs/**/*.py', 'utils/**/*.py'):
        code_paths.extend((ROOT / 'MOTIP').glob(pattern))
    protocol = {
        'checkpoint_sha256': sha256(checkpoint), 'config': config,
        'runtime': runtime.to_dict(), 'preprocessing': 'upstream-pil',
        'frame_decoder': 'OpenCV BGR -> RGB PIL', 'device': device, 'seed': 0,
        'torch': torch.__version__, 'torchvision': torchvision.__version__, 'opencv': cv2.__version__,
        'code_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in sorted(set(code_paths))},
        'dataset_sha256': sha256(folder / 'dataset.json'),
    }
    protocol_path = folder / 'inference_protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError('Inference protocol changed; choose a new output directory to avoid mixing runs')
    save_json(protocol_path, protocol)
    save_json(folder / 'inference_environment.json', {
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'gpu': torch.cuda.get_device_name(device) if str(device).startswith('cuda') else None,
        'cuda': torch.version.cuda,
    })
    protocol_hash = sha256(protocol_path)
    prediction_dir = folder / 'predictions/motip/data'
    prediction_dir.mkdir(parents=True, exist_ok=True)
    backend = None
    for seq in dataset['sequences']:
        name = seq['sequence']
        pred = prediction_dir / f'{name}.txt'
        complete = folder / 'completed' / f'{name}.json'
        if complete.exists():
            record = json.loads(complete.read_text())
            if record['protocol_sha256'] != protocol_hash or record['prediction_sha256'] != sha256(pred):
                raise ValueError(f'{name}: completed prediction was modified')
            print(f'[{split}] Reusing complete prediction: {name}', flush=True)
            continue
        if sha256(seq['video_path']) != seq['video_sha256']:
            raise ValueError(f'{name}: source video changed')
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        if backend is None:
            backend = MotipBackend(motip_root=ROOT / 'MOTIP', config_path=config,
                                   checkpoint_path=checkpoint, runtime_config=runtime,
                                   device=device, preprocessing='upstream-pil')
        # Loading the model may consume random state; each tracker starts from the same seed.
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        tracker = backend.create_tracker(height=seq['height'], width=seq['width'])
        capture = cv2.VideoCapture(seq['video_path'])
        if not capture.isOpened():
            raise ValueError(f'Cannot open {seq["video_path"]}')
        start, count, detections = time.monotonic(), 0, 0
        temporary = pred.with_suffix('.txt.part')
        try:
            with temporary.open('w') as stream:
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    count += 1
                    if frame.shape[:2] != (seq['height'], seq['width']):
                        raise ValueError(f'{name}: unexpected frame dimensions')
                    results = backend.infer_frame(tracker, frame)
                    ids = set()
                    for result in results:
                        identity = result['track_id']
                        x, y, x2, y2 = result['bbox_xyxy']
                        score = result['det_score']
                        if identity < 0 or identity in ids or not all(math.isfinite(v) for v in (x, y, x2, y2, score)) or x2 <= x or y2 <= y:
                            raise ValueError(f'{name}: invalid tracker output at frame {count}')
                        ids.add(identity)
                        # Preserve raw boxes/IDs; no quality, ReID, top-k, or GT filtering.
                        stream.write(f'{count},{identity},{x},{y},{x2-x},{y2-y},{score},-1,-1,-1\n')
                    detections += len(results)
                    if count == 1 or count % 100 == 0:
                        print(f'[{split}] {name}: {count}/{seq["frames"]} frames, {count / (time.monotonic()-start):.2f} fps', flush=True)
        finally:
            capture.release()
        if count != seq['frames']:
            raise ValueError(f'{name}: processed {count}, expected {seq["frames"]}; incomplete prediction not evaluated')
        temporary.replace(pred)
        save_json(complete, {'frames': count, 'detections': detections, 'seconds': time.monotonic()-start,
                             'protocol_sha256': protocol_hash, 'prediction_sha256': sha256(pred)})
        del tracker
