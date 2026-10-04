"""Product-facing batch diagnostics, with explicit absence of accuracy evidence."""

import json
from pathlib import Path

from analysis.evaluation.track_review import evaluate_review, review_index, summarize_reviews
from contracts.schema import read_jsonl, write_json


def saved_reviews(web_root: Path | None) -> dict:
    found = {}
    if web_root:
        for manifest in web_root.glob('*/project.json'):
            value = json.loads(manifest.read_text())
            for video in value.get('videos', []):
                path = manifest.parent / 'reviews' / f'{video["video_id"]}.json'
                if path.is_file():
                    key = str(Path(video['output_dir']).resolve())
                    review = json.loads(path.read_text())
                    if key in found and found[key] != review:
                        raise ValueError('Conflicting saved reviews for the same output; export one frozen review bundle instead')
                    found[key] = review
    return found


def summarize_run(run_dir: Path, output: Path, web_output_root: Path | None = None) -> dict:
    root = run_dir.resolve()
    manifest_path = root / 'batch_manifest.json'
    if manifest_path.is_file():
        names = [row['name'] for row in json.loads(manifest_path.read_text())['clips']]
    else:
        names = sorted(p.name for p in root.iterdir() if p.is_dir() and (p / 'tracking').is_dir())
    if not names or len(set(names)) != len(names):
        raise ValueError('No unique clips found')
    # Only one dedicated summary filename is allowed, avoiding destructive overwrite of artifacts.
    if output.name != 'run_evaluation.json':
        raise ValueError('Use a dedicated run_evaluation.json output')
    inputs = saved_reviews(web_output_root)
    results, failures, audits = [], [], []
    for name in names:
        clip = (root / name).resolve()
        if not clip.is_relative_to(root):
            raise ValueError('Unsafe clip in batch manifest')
        try:
            index = review_index(clip / 'tracking/tracks.jsonl', clip / 'tracking/video_meta.json',
                                 clip / 'quality/quality_tracks.jsonl')
            status_path = clip / 'batch_status.json'
            status = json.loads(status_path.read_text()) if status_path.is_file() else None
            if status and status.get('status') != 'completed':
                raise ValueError('Batch processing is incomplete: ' + status.get('status', 'unknown'))
            mapping = clip / 'identity/identity_map.jsonl'
            if not mapping.is_file():
                raise FileNotFoundError(mapping)
            audit = evaluate_review(index, inputs.get(str(clip)), mapping)
            raw_file = clip / 'identity_raw/identities.jsonl'
            raw = list(read_jsonl(raw_file)) if raw_file.is_file() else None
            resolved = list(read_jsonl(clip / 'identity/identities.jsonl'))
            retained = sum(r['retained'] for r in index['tracks'])
            raw_count = len(raw) if raw is not None else None
            if raw is not None:
                raw_ids = [int(p['raw_track_ids'][0]) for p in raw if len(p.get('raw_track_ids', [])) == 1]
                if len(raw_ids) != len(raw) or len(set(raw_ids)) != len(raw_ids) or raw_count > retained:
                    raise ValueError('Invalid original identity archive: expected unique retained one-track archives')
            resolved_ids = [p['person_id'] for p in resolved]
            mapped_ids = {p['person_id'] for p in read_jsonl(mapping)}
            if len(resolved_ids) != len(set(resolved_ids)) or set(resolved_ids) != mapped_ids:
                raise ValueError('Resolved archive IDs differ from identity mappings')
            numbers_path = clip / 'identity/jersey_people.jsonl'
            numbers = list(read_jsonl(numbers_path)) if numbers_path.is_file() else []
            diagnostic = {'name': name, 'video_id': index['video_id'],
                          'raw_tracks': len(index['tracks']), 'retained_tracks': retained,
                          'raw_identity_archives': raw_count, 'resolved_archives': len(resolved),
                          'archive_reduction': raw_count - len(resolved) if raw_count is not None else None,
                          'tracks_without_raw_archive': retained - raw_count if raw_count is not None else None,
                          'jersey_candidate_archives': sum(r.get('number') is not None for r in numbers),
                          'jersey_conflicts': sum(r.get('status') == 'conflict' for r in numbers),
                          'audit': audit}
            results.append(diagnostic)
            audits.append(audit)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            failures.append({'name': name, 'error': str(exc)})
    pooled = summarize_reviews(audits, expected_clips=len(names), failed=failures)
    full = pooled['completed_subset']['scopes']['full_track']['raw']
    result = {**pooled, 'clips': results,
              'headline': {'full_reviewed_player_tracks': full['player_tracks_assessed'],
                           'pure_track_rate': full['pure_track_rate'], 'mixed_tracks': full['mixed'],
                           'review_coverage': full['review_coverage']},
              'diagnostics_totals': {k: sum(r[k] for r in results if r[k] is not None) for k in (
                  'raw_tracks', 'retained_tracks', 'resolved_archives', 'archive_reduction', 'jersey_candidate_archives', 'jersey_conflicts')},
              'warning': 'Counts, filtering and OCR consensus are not accuracy. Review purity covers observed trajectories, not missed players. Identity pair F1 covers fully reviewed pure labeled retained tracklets only. Corrections evaluated with the same labels are not independent accuracy. MultiSports action tubes do not provide persistent player/jersey ground truth.'}
    write_json(output, result)
    return result
