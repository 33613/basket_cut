"""Read official SportsMOT frames/GT without conversion or re-encoding."""
from __future__ import annotations

import configparser
import csv
import json
import math
import re
import shutil
from pathlib import Path

from PIL import Image

from analysis.teamtrack.dataset import save_json, sha256


NAME = re.compile(r'[A-Za-z0-9_-]+')


def roster(root, name):
    candidates = sorted(root.rglob(name))
    if not candidates:
        raise FileNotFoundError(f'Missing official roster {name} below {root}')
    values = []
    for path in candidates:
        lines = {s.strip() for s in path.read_text(encoding='utf-8-sig').splitlines() if s.strip()}
        if not lines or any(not NAME.fullmatch(s) for s in lines):
            raise ValueError(f'Invalid roster: {path}')
        values.append(lines)
    if any(v != values[0] for v in values[1:]):
        raise ValueError(f'Conflicting copies of {name}')
    return values[0], candidates[0]


def locate_val(root):
    candidates = [p for p in (root / 'SportsMOT/val', root / 'val') if p.is_dir()]
    candidates.extend(p for p in root.rglob('val') if p.is_dir() and p.parent.name == 'SportsMOT')
    candidates = sorted(set(p.resolve() for p in candidates))
    if len(candidates) != 1:
        raise ValueError(f'Expected one SportsMOT/val directory, found {candidates}')
    return candidates[0]


def validate_gt(path, frames):
    count, ids, seen = 0, set(), set()
    with path.open(newline='') as stream:
        for line, row in enumerate(csv.reader(stream), 1):
            if len(row) < 9:
                raise ValueError(f'{path}:{line}: expected nine MOT GT columns')
            values = list(map(float, row[:9]))
            f, identity, x, y, w, h, mark, category, visibility = values
            if not all(map(math.isfinite, values)) or f != int(f) or identity != int(identity):
                raise ValueError(f'{path}:{line}: nonfinite value or noninteger frame/ID')
            if not 1 <= f <= frames or identity < 0 or w <= 0 or h <= 0:
                raise ValueError(f'{path}:{line}: invalid frame, ID, or box size')
            if (int(f), int(identity)) in seen:
                raise ValueError(f'{path}:{line}: duplicate frame/ID')
            # SportsMOT official player GT is marked/class=1. Fail explicitly rather
            # than misusing the TeamTrack frame-trace assumptions for ignore classes.
            if mark != 1 or category != 1:
                raise ValueError(f'{path}:{line}: unexpected ignore/class annotation; requires separate protocol review')
            seen.add((int(f), int(identity)))
            ids.add(int(identity))
            count += 1
    if not count:
        raise ValueError(f'Empty SportsMOT player GT: {path}')
    return count, len(ids)


def prepare(root, output, limit):
    root, output = Path(root).expanduser().resolve(), Path(output).expanduser().resolve()
    if limit < 0:
        raise ValueError('--limit must be nonnegative (0 means all available basketball val sequences)')
    basketball, basketball_path = roster(root, 'basketball.txt')
    val_names, val_path = roster(root, 'val.txt')
    source = locate_val(root)
    eligible = basketball & val_names
    available = {p.name for p in source.iterdir() if p.is_dir() and not p.name.startswith('.')}
    unexpected = available - eligible
    if unexpected:
        raise ValueError(f'This basketball-only package contains sequences outside the official intersection: {sorted(unexpected)}')
    selected = sorted(available & eligible)
    if not selected:
        raise ValueError('No downloaded official basketball val sequences found')
    if limit:
        if len(selected) < limit:
            raise ValueError(f'Requested {limit} sequences, only {len(selected)} available')
        selected = selected[:limit]
    output.mkdir(parents=True, exist_ok=True)
    seqs = []
    for name in selected:
        if not NAME.fullmatch(name):
            raise ValueError(f'Invalid sequence name: {name}')
        folder = source / name
        cfg = configparser.ConfigParser()
        cfg.read(folder / 'seqinfo.ini')
        info = cfg['Sequence']
        length, width, height = (int(info[k]) for k in ('seqLength', 'imWidth', 'imHeight'))
        fps = float(info['frameRate'])
        if length <= 10 or width <= 0 or height <= 0 or not math.isfinite(fps) or fps <= 0:
            raise ValueError(f'{name}: invalid metadata or too short for upstream submission (>10 frames required)')
        if info.get('imDir') != 'img1' or info.get('imExt', '.jpg').lower() != '.jpg':
            raise ValueError(f'{name}: expected original img1 six-digit JPEG frames')
        expected = {f'{i:06d}.jpg' for i in range(1, length+1)}
        actual = {p.name for p in (folder / 'img1').iterdir() if p.is_file() and not p.name.startswith('.')}
        if actual != expected:
            raise ValueError(f'{name}: image filenames/count disagree with original seqinfo')
        images = []
        for filename in sorted(expected):
            path = folder / 'img1' / filename
            with Image.open(path) as image:
                image.load()
                if image.size != (width, height) or image.mode != 'RGB':
                    raise ValueError(f'{path}: unexpected dimensions or non-RGB original frame')
            images.append({'name': filename, 'sha256': sha256(path), 'bytes': path.stat().st_size})
        gt = folder / 'gt/gt.txt'
        count, ids = validate_gt(gt, length)
        seqs.append(dict(sequence=name, view='basketball', split='val', source=str(folder), frames=length,
                         width=width, height=height, fps=fps, gt_boxes=count, identities=ids,
                         gt_sha256=sha256(gt), seqinfo_sha256=sha256(folder / 'seqinfo.ini'), images=images))
        print(f'[prepare] {name}: {length} original frames, {count} GT boxes, {ids} sequence-local IDs', flush=True)
    manifest = dict(schema_version=1, dataset='SportsMOT official basketball val', limit=limit,
                    basketball_roster_sha256=sha256(basketball_path), val_roster_sha256=sha256(val_path),
                    official_basketball_val_count=len(eligible), downloaded_count=len(available),
                    missing_from_package=sorted(eligible-available), sequences=seqs)
    target = output / 'dataset.json'
    if target.exists() and json.loads(target.read_text()) != manifest:
        raise ValueError('Dataset or selection changed; use a new output directory')
    # Read-only references for the actual upstream CLI; each job contains exactly one sequence.
    for seq in seqs:
        folder = Path(seq['source'])
        for method in ('official', 'adapter'):
            dest = output / method / 'val/gt' / seq['sequence']
            (dest / 'gt').mkdir(parents=True, exist_ok=True)
            for src, dst in [(folder / 'gt/gt.txt', dest / 'gt/gt.txt'), (folder / 'seqinfo.ini', dest / 'seqinfo.ini')]:
                if dst.exists() and sha256(dst) != sha256(src):
                    raise ValueError(f'Prepared GT was changed: {dst}')
                if not dst.exists():
                    shutil.copyfile(src, dst)
        link = output / 'reference-input' / seq['sequence'] / 'SportsMOT/val' / seq['sequence']
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.is_symlink() and link.resolve() != folder:
            raise ValueError(f'Unexpected existing symlink: {link}')
        if not link.exists():
            link.symlink_to(folder, target_is_directory=True)
        elif not link.is_symlink():
            raise ValueError(f'Expected a reference symlink, not a directory: {link}')
    save_json(target, manifest)
    print(f'Prepared {len(seqs)}/{len(eligible)} official basketball val sequences; source frames and GT retained', flush=True)
    return manifest


def verify_sources(sequence):
    source = Path(sequence['source'])
    if sha256(source / 'gt/gt.txt') != sequence['gt_sha256'] or sha256(source / 'seqinfo.ini') != sequence['seqinfo_sha256']:
        raise ValueError(f'{sequence["sequence"]}: GT or metadata changed')
    for frame in sequence['images']:
        if sha256(source / 'img1' / frame['name']) != frame['sha256']:
            raise ValueError(f'{sequence["sequence"]}: source frame changed: {frame["name"]}')
