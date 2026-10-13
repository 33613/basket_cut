"""Validate raw TeamTrack CSV/video pairs and export player-only MOT ground truth."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path

VIEWS = ('basketball_side', 'basketball_side_2', 'basketball_top')
ATTRS = ('bb_left', 'bb_top', 'bb_width', 'bb_height')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.part')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def parse_annotations(path):
    """Return all frames, including empty ones; never silently truncate or fill boxes."""
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.reader(stream))
    if len(rows) < 5 or [r[0] for r in rows[:4]] != ['TeamID', 'PlayerID', 'Attributes', 'frame']:
        raise ValueError(f'{path}: expected four TeamTrack header rows')
    width = len(rows[0])
    if any(len(r) != width for r in rows):
        raise ValueError(f'{path}: inconsistent CSV column counts')
    columns = defaultdict(dict)
    for col in range(1, width):
        team, player, attr = (rows[i][col].strip() for i in range(3))
        if team == 'BALL' or player == 'BALL':
            continue
        if not team or not player or attr not in ATTRS or attr in columns[(team, player)]:
            raise ValueError(f'{path}: invalid/duplicate player attribute in column {col}')
        columns[(team, player)][attr] = col
    if not columns or any(set(attrs) != set(ATTRS) for attrs in columns.values()):
        raise ValueError(f'{path}: incomplete player headers')
    identities = {key: i + 1 for i, key in enumerate(sorted(columns))}
    frames, absent = [], 0
    for expected, row in enumerate(rows[4:], 1):
        if float(row[0]) != expected:
            raise ValueError(f'{path}: frames must be consecutive and one-based; expected {expected}')
        boxes = []
        for key, attrs in columns.items():
            values = [float(row[attrs[a]]) if row[attrs[a]].strip() else math.nan for a in ATTRS]
            if all(math.isnan(v) for v in values):
                absent += 1
                continue
            if not all(math.isfinite(v) for v in values) or values[2] <= 0 or values[3] <= 0:
                raise ValueError(f'{path}: invalid/partial box at frame {expected}, player {key}: {values}')
            boxes.append((identities[key], *values))
        frames.append(boxes)
    mapping = [{'mot_id': value, 'team_id': key[0], 'player_id': key[1]} for key, value in identities.items()]
    return frames, mapping, absent


def read_selection(root, split):
    with (root / 'selection.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    selected, seen = [], set()
    for row in rows:
        view, part, video = row['view'], row['split'], row['video']
        if view not in VIEWS or part not in ('val', 'test') or not re.fullmatch(r'[A-Za-z0-9_-]+\.mp4', video):
            raise ValueError(f'Invalid selection row: {row}')
        key = (view, part, video)
        if key in seen:
            raise ValueError(f'Duplicate selection: {key}')
        seen.add(key)
        if part == split:
            selected.append({'view': view, 'split': part, 'video': video,
                             'sequence': view + '__' + Path(video).stem})
    if not selected:
        raise ValueError(f'No {split} sequences selected')
    return selected


def prepare(root, output, split):
    import cv2
    root, output = Path(root).resolve(), Path(output).resolve()
    folder = output / split
    folder.mkdir(parents=True, exist_ok=True)
    manifest_path = folder / 'dataset.json'
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    rows = read_selection(root, split)
    sources = []
    for row in rows:
        base = root / row['view'] / split
        video = base / 'videos' / row['video']
        annotation = base / 'annotations' / Path(row['video']).with_suffix('.csv')
        sources.append({**row, 'video_path': str(video), 'annotation_path': str(annotation),
                        'video_sha256': sha256(video), 'annotation_sha256': sha256(annotation)})
    if previous:
        if previous['sources'] != sources or previous.get('converter_sha256') != sha256(__file__):
            raise ValueError('Dataset changed or converter updated; use a new output directory')
        for seq in previous['sequences']:
            if sha256(folder / 'gt' / seq['sequence'] / 'gt/gt.txt') != seq['gt_sha256']:
                raise ValueError('Prepared GT changed; use a new output directory')
        print(f'[{split}] Reusing validated dataset: {len(sources)} sequences', flush=True)
        return previous
    sequences = []
    for row in sources:
        frames, mapping, absent = parse_annotations(row['annotation_path'])
        sequence = row['sequence']
        seq_dir = folder / 'gt' / sequence
        (seq_dir / 'gt').mkdir(parents=True, exist_ok=True)
        preview_dir = folder / 'alignment' / sequence
        preview_dir.mkdir(parents=True, exist_ok=True)
        capture = cv2.VideoCapture(row['video_path'])
        if not capture.isOpened():
            raise ValueError(f'Cannot open {row["video_path"]}')
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(fps) or fps <= 0:
            capture.release()
            raise ValueError('Invalid video FPS')
        width, height = int(capture.get(3)), int(capture.get(4))
        declared = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        decoded = 0
        samples = {1, (len(frames) + 1) // 2, len(frames)}
        try:
            while True:
                ok, image = capture.read()
                if not ok:
                    break
                decoded += 1
                if image.shape[:2] != (height, width):
                    raise ValueError('Video changes resolution')
                if decoded in samples:
                    scale = min(1., 1600 / width)
                    image = cv2.resize(image, (round(width * scale), round(height * scale)))
                    for identity, x, y, w, h in frames[decoded - 1]:
                        p1, p2 = (round(x * scale), round(y * scale)), (round((x+w)*scale), round((y+h)*scale))
                        cv2.rectangle(image, p1, p2, (0, 255, 0), 2)
                        cv2.putText(image, str(identity), p1, cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 0), 2)
                    if not cv2.imwrite(str(preview_dir / f'{decoded:06d}.jpg'), image):
                        raise OSError('Could not save alignment preview')
        finally:
            capture.release()
        if decoded != len(frames) or (declared > 0 and decoded != declared):
            raise ValueError(f'{sequence}: CSV={len(frames)}, decoded={decoded}, container={declared}; refusing misaligned evaluation')
        outside = 0
        gt = seq_dir / 'gt/gt.txt'
        with gt.open('w') as stream:
            for frame, boxes in enumerate(frames, 1):
                for identity, x, y, w, h in boxes:
                    outside += int(x < 0 or y < 0 or x+w > width or y+h > height)
                    stream.write(f'{frame},{identity},{x},{y},{w},{h},1,1,1\n')
        (seq_dir / 'seqinfo.ini').write_text(
            f'[Sequence]\nname={sequence}\nimDir=img1\nframeRate={fps}\nseqLength={decoded}\nimWidth={width}\nimHeight={height}\nimExt=.jpg\n')
        save_json(seq_dir / 'identity_map.json', mapping)
        sequences.append({**row, 'frames': decoded, 'fps': fps, 'width': width, 'height': height,
                          'gt_boxes': sum(map(len, frames)), 'absent_boxes': absent,
                          'out_of_bounds_boxes': outside, 'gt_sha256': sha256(gt)})
        print(f'[{split}] {sequence}: {decoded} frames, {sum(map(len, frames))} player boxes, {outside} outside-image boxes (retained)', flush=True)
    manifest = {'schema_version': 1, 'converter_sha256': sha256(__file__), 'sources': sources, 'sequences': sequences,
                'protocol': 'player-only; BALL excluded; frame base 1; raw pixel xywh; no box clipping'}
    save_json(manifest_path, manifest)
    return manifest
