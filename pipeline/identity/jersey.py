"""Optional jersey-number evidence. OCR never defines or merges real identities."""

import math
from collections import defaultdict
from pathlib import Path

from contracts.schema import read_jsonl, write_json, write_jsonl_line


def number_consensus(readings: list[dict], min_score: float = .8, min_support: int = 2) -> dict:
    """Count independent frames, preserve 0/00, and abstain on contradictions."""
    if not 0 <= min_score <= 1 or min_support < 1:
        raise ValueError("Invalid OCR consensus thresholds")
    by_number = defaultdict(dict)
    for row in readings:
        text, score = str(row.get('text', '')).strip(), float(row.get('raw_score', 0))
        if not text.isascii() or not text.isdigit() or not 1 <= len(text) <= 2 or not math.isfinite(score) or score < min_score:
            continue
        frame = int(row['frame_idx'])
        if frame not in by_number[text] or score > by_number[text][frame]['raw_score']:
            by_number[text][frame] = row
    candidates = [{'number': number, 'support_frames': len(rows),
                   'mean_raw_score': sum(r['raw_score'] for r in rows.values()) / len(rows),
                   'evidence': list(rows.values())}
                  for number, rows in sorted(by_number.items())]
    reliable = [c for c in candidates if c['support_frames'] >= min_support]
    # Even minority high-confidence digits may reveal mixed people; do not majority-vote them away.
    accepted = reliable[0] if len(reliable) == 1 and len(candidates) == 1 else None
    return {'number': accepted['number'] if accepted else None,
            'status': 'candidate_consensus' if accepted else 'conflict' if len(candidates) > 1 else 'insufficient_evidence' if candidates else 'unreadable',
            'candidates': candidates, 'verified': False}


def group_numbers(ids: list[int], tracks: dict[int, dict]) -> dict:
    rows = [tracks.get(tid, {'number': None, 'status': 'unreadable'}) for tid in ids]
    numbers = {r['number'] for r in rows if r.get('number') is not None}
    status = ('conflict' if len(numbers) > 1 or any(r['status'] == 'conflict' for r in rows)
              else 'candidate_consensus' if len(numbers) == 1 and all(r.get('number') is not None for r in rows)
              else 'partial_evidence' if numbers else 'unreadable')
    return {'number': next(iter(numbers)) if status == 'candidate_consensus' else None,
            'status': status, 'candidate_numbers': sorted(numbers), 'verified': False,
            'raw_track_ids': ids}


def recognize_jerseys(archive_dir: Path, identity_map: Path, output_dir: Path,
                      model_dir: Path, *, allow_download: bool = False,
                      min_score: float = .8, min_support: int = 2, overwrite: bool = False,
                      reader=None) -> dict:
    import cv2
    archive = archive_dir.resolve()
    output = output_dir.resolve()
    targets = [output / f'jersey_{name}' for name in ('tracks.jsonl', 'people.jsonl', 'summary.json')]
    if identity_map.resolve() in targets or (archive / 'kpr_samples.jsonl') in targets:
        raise ValueError("OCR outputs cannot overwrite inputs")
    if not overwrite and any(p.exists() for p in targets):
        raise FileExistsError("OCR output exists; pass --overwrite")
    mappings = list(read_jsonl(identity_map))
    ids = {int(row['raw_track_id']) for row in mappings}
    if len(ids) != len(mappings):
        raise ValueError('Duplicate identity mappings')
    samples_path = archive / 'kpr_samples.jsonl'
    if not samples_path.is_file():
        samples_path = archive / 'kpr_sampling_manifest.jsonl'
    samples = list(read_jsonl(samples_path)) if ids else []
    if reader is None and samples:
        import easyocr
        reader = easyocr.Reader(['en'], gpu=False, model_storage_directory=str(model_dir),
                                download_enabled=allow_download)
    readings = defaultdict(list)
    for sample in samples:
        tid = int(sample['track_id'])
        if tid not in ids:
            continue
        path = (archive / sample['crop_path']).resolve()
        if not path.is_relative_to(archive):
            raise ValueError('OCR crop escapes archive')
        crop = cv2.imread(str(path))
        if crop is None:
            raise ValueError(f'Missing OCR crop: {sample["crop_path"]}')
        height, width = crop.shape[:2]
        # Approximate torso ROI only; crowded/raised-arm crops remain a known limitation.
        left, top, right, bottom = round(width * .2), round(height * .15), round(width * .8), round(height * .62)
        if right <= left or bottom <= top:
            continue
        roi = crop[top:bottom, left:right]
        scale = max(1., 192 / max(1, roi.shape[0]))
        roi = cv2.resize(roi, (max(1, round(roi.shape[1] * scale)), max(1, round(roi.shape[0] * scale))))
        for box, text, score in reader.readtext(roi, allowlist='0123456789', detail=1, paragraph=False):
            readings[tid].append({'frame_idx': int(sample['frame_idx']), 'text': str(text),
                                  'raw_score': float(score), 'crop_path': sample['crop_path'],
                                  'torso_roi_xyxy': [left, top, right, bottom],
                                  'ocr_box_in_roi': [[float(v) for v in point] for point in box]})
    tracks = {tid: {'raw_track_id': tid, **number_consensus(readings[tid], min_score, min_support)} for tid in sorted(ids)}
    groups = defaultdict(list)
    for row in mappings:
        groups[row['person_id']].append(int(row['raw_track_id']))
    people = [{'person_id': pid, **group_numbers(tids, tracks)} for pid, tids in sorted(groups.items())]
    output.mkdir(parents=True, exist_ok=True)
    for path, rows in zip(targets[:2], (tracks.values(), people)):
        with path.open('w', encoding='utf-8') as handle:
            for row in rows:
                write_jsonl_line(handle, row)
    summary = {'track_count': len(tracks), 'person_count': len(people),
               'candidate_number_tracks': sum(r['number'] is not None for r in tracks.values()),
               'conflicting_tracks': sum(r['status'] == 'conflict' for r in tracks.values()),
               'min_raw_score': min_score, 'min_support_frames': min_support,
               'warning': 'Uncalibrated generic OCR on approximate torso crops, not a trained jersey recognizer. Number evidence never changes identity mappings. Same number across teams is not the same person.'}
    write_json(targets[2], summary)
    return summary
