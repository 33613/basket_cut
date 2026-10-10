"""Probe the four isolated model environments before a GPU run; never download weights."""
import argparse
import json
import subprocess
from pathlib import Path
from contracts.execution import ExecutionSettings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env-file', type=Path)
    args = parser.parse_args()
    if args.env_file:
        from cli.serve_web import load_environment
        load_environment(args.env_file)
    settings = ExecutionSettings()
    probes = {
        'motip': (settings.motip_python, settings.motip_checkpoint,
            "import cv2,torch,torchvision,MultiScaleDeformableAttention; import sys; sys.path.insert(0,'MOTIP'); from models.motip import build"),
        'kpr': (settings.kpr_python, settings.kpr_checkpoint,
            "import cv2,torch; import sys; sys.path.insert(0,'KPR'); from torchreid.tools.feature_extractor import KPRFeatureExtractor"),
        'action': (settings.action_python, settings.action_checkpoint,
            "import torch,mmcv,mmengine,mmaction; from mmcv.ops import roi_align; import pipeline.action.service"),
        'qwen': (settings.qwen_python, settings.qwen_model_dir / 'config.json',
            "import torch; from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor; import pipeline.identity.jersey_qwen"),
    }
    results = {}
    for name, (python, weight, imports) in probes.items():
        result = {'python_exists': python.is_file(), 'weight_exists': weight.is_file()}
        if python.is_file():
            code = imports + "; assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.__version__,torch.version.cuda,torch.cuda.get_device_name(0))"
            probe = subprocess.run([str(python), '-c', code], cwd=settings.repository_root,
                                   capture_output=True, text=True)
            result.update(exit_code=probe.returncode, output=probe.stdout.strip(), error=probe.stderr.strip())
        results[name] = result
    print(json.dumps(results, ensure_ascii=False, indent=2))
    if not all(r.get('exit_code') == 0 and r['weight_exists'] for r in results.values()):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
