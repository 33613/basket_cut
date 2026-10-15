"""Resume a ModelScope Qwen3-VL snapshot and verify its published SHA256 values."""
import argparse
import json
import re
import shutil
from pathlib import Path

from analysis.teamtrack.dataset import save_json, sha256

MODEL_ID = 'Qwen/Qwen3-VL-8B-Instruct'


def asset_records(files):
    records = []
    for item in files:
        if item.get('Type') == 'tree':
            continue
        name = item['Path']
        path = Path(name)
        if path.name != name or path.is_absolute() or name in ('.', '..'):
            continue
        if path.suffix not in ('.safetensors', '.json', '.txt', '.model', '.jinja') and name not in ('LICENSE', 'README.md'):
            continue
        digest = str(item.get('Sha256', '')).lower()
        if not re.fullmatch(r'[a-f0-9]{64}', digest):
            raise ValueError(f'ModelScope did not provide a valid SHA256 for {name}; verification cannot proceed')
        size = int(item['Size'])
        if size < 0:
            raise ValueError(f'Invalid remote file size: {name}')
        records.append(dict(path=name, bytes=size, sha256=digest))
    names = {r['path'] for r in records}
    required = {'config.json', 'preprocessor_config.json', 'tokenizer_config.json', 'tokenizer.json'}
    if not required <= names or not any(n.endswith('.safetensors') for n in names):
        raise ValueError(f'Incomplete remote snapshot; missing required assets: {sorted(required-names)}')
    if not {'chat_template.json', 'chat_template.jinja'} & names:
        raise ValueError('Remote snapshot has no chat template')
    if len(names) != len(records):
        raise ValueError('Duplicate remote file records')
    return sorted(records, key=lambda r: r['path'])


def verify_files(root, records):
    for index, record in enumerate(records, 1):
        path = root / record['path']
        if not path.is_file() or path.stat().st_size != record['bytes']:
            raise ValueError(f'Missing/truncated model asset: {path}')
        print(f'[{index}/{len(records)}] SHA256: {path.name}', flush=True)
        if sha256(path) != record['sha256']:
            raise ValueError(f'SHA256 mismatch: {path}; do not load this file')
    config = json.loads((root / 'config.json').read_text())
    if config.get('model_type') != 'qwen3_vl':
        raise ValueError('This snapshot is not Qwen3-VL')
    index_path = root / 'model.safetensors.index.json'
    if index_path.exists():
        index = json.loads(index_path.read_text())
        downloaded = {r['path'] for r in records}
        if not set(index['weight_map'].values()) <= downloaded:
            raise ValueError('Weight index references missing or unverified shards')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True, help='Store outside the model directory')
    parser.add_argument('--revision', default='master')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    root, manifest_path = args.output_dir.expanduser().resolve(), args.manifest.expanduser().resolve()
    if manifest_path.is_relative_to(root):
        parser.error('--manifest must be outside the model directory')
    if not 1 <= args.workers <= 8:
        parser.error('--workers must be between 1 and 8')
    root.mkdir(parents=True, exist_ok=True)
    from modelscope import snapshot_download
    from modelscope.hub.api import HubApi
    records = asset_records(HubApi().get_model_files(MODEL_ID, revision=args.revision, recursive=True, use_cookies=False))
    contract = dict(source='ModelScope', model_id=MODEL_ID, revision=args.revision, files=records)
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous['snapshot'] != contract:
            raise ValueError('Remote snapshot changed; select a fixed tag or use a new directory and manifest')
    expected = sum(r['bytes'] for r in records)
    reusable = sum(r['bytes'] for r in records if (root/r['path']).is_file() and (root/r['path']).stat().st_size == r['bytes'])
    needed = expected-reusable + 1024**3
    free = shutil.disk_usage(root).free
    print(f'ModelScope snapshot: {expected/1024**3:.2f} GiB; estimated remaining {max(0,expected-reusable)/1024**3:.2f} GiB; free {free/1024**3:.2f} GiB', flush=True)
    if free < needed:
        raise RuntimeError(f'Insufficient disk space; need approximately {needed/1024**3:.2f} GiB free')
    save_json(manifest_path, dict(status='downloading', snapshot=contract))
    snapshot_download(MODEL_ID, revision=args.revision, local_dir=str(root),
                      allow_patterns=[r['path'] for r in records], max_workers=args.workers)
    verify_files(root, records)
    from pipeline.identity.qwen_backend import load_processor
    processor = load_processor(root)  # CPU only: catch missing tokenizer/image/video configs now.
    if not hasattr(processor, 'chat_template') or not processor.chat_template:
        raise ValueError('Local processor has no chat template')
    save_json(manifest_path, dict(status='verified', snapshot=contract,
              image_processor=type(processor.image_processor).__name__,
              video_processor=type(processor.video_processor).__name__))
    print('Qwen3-VL snapshot SHA256 and local image/video processor checks passed; GPU inference remains to be tested.', flush=True)


if __name__ == '__main__':
    main()
