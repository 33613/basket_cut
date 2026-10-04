"""Bounded review queues and material indexes for large result collections.

Caching is based on artifact and sidecar stat signatures, not clip status alone.
An audit progress count is never promoted to a model accuracy claim.
"""

from __future__ import annotations

import copy
import threading
from collections import Counter, OrderedDict
from pathlib import Path

from web.backend.artifacts import load_json, read_jsonl
from web.backend.reviews import collect_review, review_paths
from web.backend.store import safe_slug


class ReviewCatalog:
    def __init__(self, capacity: int = 256) -> None:
        self.capacity = capacity
        self.cache = OrderedDict()
        self.lock = threading.RLock()

    def review(self, store, project_id: str, video: dict, *, clone: bool = True) -> dict:
        paths = review_paths(video)
        output = Path(video['output_dir'])
        cache_root = store.output_root / project_id / 'evidence' / video['video_id']
        # video is from the validated manifest. Avoid rereading that whole manifest
        # for each clip merely to locate a review sidecar in a 100-item queue.
        review_sidecar = store.output_root / safe_slug(project_id) / 'reviews' / f"{safe_slug(video['video_id'])}.json"
        files = [*paths.values(), review_sidecar,
                 cache_root / 'index.json', cache_root / 'job.json',
                 output / 'identity/identities.jsonl',
                 output / 'identity_raw/identities.jsonl', output / 'quality/quality_summary.json']
        def stamp(path):
            try:
                stat = path.stat()
                return (str(path), stat.st_mtime_ns, stat.st_size, stat.st_ino)
            except FileNotFoundError:
                return (str(path), None)
        signature = tuple(stamp(path) for path in files)
        key = (str(store.output_root), project_id, video['video_id'], video['output_dir'])
        with self.lock:
            cached = self.cache.get(key)
            if cached and cached[0] == signature:
                self.cache.move_to_end(key)
                return copy.deepcopy(cached[1]) if clone else cached[1]
        result = collect_review(store, project_id, video)
        raw_path = output / 'identity_raw/identities.jsonl'
        result['archive_counts'] = {'raw': len(read_jsonl(raw_path)) if raw_path.is_file() else None,
                                    'resolved': len(read_jsonl(output / 'identity/identities.jsonl'))}
        with self.lock:
            self.cache[key] = (signature, result)
            self.cache.move_to_end(key)
            while len(self.cache) > self.capacity:
                self.cache.popitem(last=False)
        return copy.deepcopy(result) if clone else result


catalog = ReviewCatalog()


def review_page(data: dict, *, offset: int = 0, limit: int = 6,
                status: str = 'all', q: str = '') -> dict:
    """Return only visible evidence cards; edits use the revision-checked PATCH API."""
    by_id = {row['raw_track_id']: row for row in data['review']['tracks']}
    selected = []
    for track in data['index']['tracks']:
        row = by_id.get(track['raw_track_id'], {})
        verdict = row.get('verdict', 'unreviewed')
        full = row.get('scope') == 'full_track' and verdict in {'pure', 'mixed', 'non_player'}
        matches = {
            'all': True, 'remaining': not full, 'retained': track['retained'],
            'filtered': not track['retained'], 'risk': bool(track['quality_reasons']) or verdict == 'mixed',
            'uncertain': verdict == 'uncertain', 'full': full,
        }
        text = f"T{track['raw_track_id']} {row.get('identity_label') or ''} {row.get('note') or ''}".casefold()
        if matches.get(status, False) and (not q or q.casefold() in text):
            selected.append(track)
    result = {**data, 'index': {**data['index'], 'tracks': selected[offset:offset + limit]},
              'review': {**data['review'], 'tracks': [by_id[track['raw_track_id']]
                  for track in selected[offset:offset + limit] if track['raw_track_id'] in by_id]},
              'pagination': {'offset': offset, 'limit': limit, 'total': len(selected),
                             'raw_total': len(data['index']['tracks'])}}
    return result


def video_summary(store, project_id: str, video: dict, ordinal: int) -> dict:
    row = {key: video.get(key) for key in ('video_id', 'filename', 'status', 'size_bytes', 'error')}
    row.update(ordinal=ordinal, review_status='unavailable', raw_tracks=0, retained_tracks=0,
               full_assessed=0, remaining_tracks=0, sampled_tracks=0, uncertain_tracks=0,
               mixed_tracks=0, risk_tracks=0, raw_archives=None, resolved_archives=0,
               revision=None, stale_review=False)
    output = Path(video['output_dir'])
    meta = load_json(output / 'tracking/video_meta.json') or {}
    row['duration_s'] = (meta.get('processed_frames', 0) / meta['fps']) if meta.get('fps') else None
    row['source_duration_s'] = (meta.get('frame_count', 0) / meta['fps']) if meta.get('fps') else None
    if video.get('status') in {'queued', 'running'}:
        row['review_status'] = 'processing'
        return row
    try:
        data = catalog.review(store, project_id, video, clone=False)
        tracks = data['index']['tracks']
        reviewed = {item['raw_track_id']: item for item in data['review']['tracks']}
        full = data['metrics']['scopes']['full_track']['raw']
        row.update(raw_tracks=len(tracks), retained_tracks=sum(t['retained'] for t in tracks),
                   full_assessed=full['assessed_tracks'], remaining_tracks=len(tracks) - full['assessed_tracks'],
                   sampled_tracks=sum(item['scope'] == 'sampled' and item['verdict'] != 'unreviewed'
                                      for item in reviewed.values()),
                   uncertain_tracks=sum(item['verdict'] == 'uncertain' for item in reviewed.values()),
                   mixed_tracks=sum(item['verdict'] == 'mixed' for item in reviewed.values()),
                   risk_tracks=sum(bool(t['quality_reasons']) or
                       reviewed.get(t['raw_track_id'], {}).get('verdict') == 'mixed' for t in tracks),
                   revision=data['revision'], stale_review=data['stale_review'])
        row['resolved_archives'] = data['archive_counts']['resolved']
        row['raw_archives'] = data['archive_counts']['raw']
        has_judgment = any(item['verdict'] != 'unreviewed' for item in reviewed.values())
        row['review_status'] = ('empty' if not tracks else 'complete' if row['remaining_tracks'] == 0
                                else 'partial' if has_judgment else 'unreviewed')
        if data['stale_review']:
            row['review_status'] = 'stale'
    except (OSError, ValueError, KeyError, TypeError) as exc:
        row['review_error'] = str(exc)
    return row


def video_catalog(store, manifest: dict, *, offset=0, limit=12, status='all', q='') -> dict:
    rows = [video_summary(store, manifest['project_id'], video, number)
            for number, video in enumerate(manifest['videos'], 1)]
    selected = []
    for row in rows:
        matches = {'all': True,
                   'remaining': row['review_status'] in {'unreviewed', 'partial', 'stale'},
                   'complete': row['review_status'] == 'complete',
                   'risk': row['risk_tracks'] > 0 or row['mixed_tracks'] > 0,
                   'uncertain': row['uncertain_tracks'] > 0,
                   'failed': row['status'] == 'failed' or row['review_status'] in {'unavailable', 'empty'}}
        if matches.get(status, False) and (not q or q.casefold() in str(row['filename']).casefold()):
            selected.append(row)
    expected = manifest.get('expected_video_count') or len(rows)
    return {'items': selected[offset:offset + limit], 'offset': offset, 'limit': limit,
            'total': len(selected), 'summary': {
                'expected_clips': expected, 'available_clips': len(rows),
                'missing_clips': max(0, expected - len(rows)),
                'review_statuses': dict(Counter(row['review_status'] for row in rows)),
                'processing_statuses': dict(Counter(row['status'] for row in rows)),
                'raw_tracks': sum(row['raw_tracks'] for row in rows),
                'full_assessed': sum(row['full_assessed'] for row in rows),
                'remaining_tracks': sum(row['remaining_tracks'] for row in rows),
                'sampled_tracks': sum(row['sampled_tracks'] for row in rows),
                'uncertain_tracks': sum(row['uncertain_tracks'] for row in rows),
                'mixed_tracks': sum(row['mixed_tracks'] for row in rows),
                'warning': 'Review progress is not accuracy. Uncertain and sampled-only tracks remain incomplete.'}}


def next_review_video(store, manifest: dict, after: str | None = None, *, q='', status='remaining') -> dict:
    rows = video_catalog(store, manifest, limit=max(1, len(manifest['videos'])), status=status, q=q)['items']
    rows = [row for row in rows if row['review_status'] in {'unreviewed', 'partial', 'stale'}]
    original = [video['video_id'] for video in manifest['videos']]
    after_index = original.index(after) if after in original else -1
    rows.sort(key=lambda row: ((row['ordinal'] - 1 - after_index - 1) % max(1, len(original))))
    return {'video_id': rows[0]['video_id'] if rows else None, 'remaining_clips': len(rows)}


def material_catalog(manifest: dict, *, offset=0, limit=18, q='', view='resolved') -> dict:
    """Read archive metadata only, not every event/observation in a 100-video run."""
    entries = []
    for video in manifest['videos']:
        output = Path(video['output_dir'])
        prefix = 'identity_raw' if view == 'raw' else 'identity'
        identities = read_jsonl(output / prefix / 'identities.jsonl')
        jerseys = {str(row.get('person_id')): row for row in read_jsonl(output / 'identity/jersey_people.jsonl')} if view != 'raw' else {}
        for person in identities:
            jersey = jerseys.get(str(person.get('person_id')))
            if jersey and set(jersey.get('raw_track_ids', [])) != set(person.get('raw_track_ids', [])):
                jersey = None
            text = f"{video['filename']} {person.get('person_id')} {person.get('identity_label') or ''} {jersey.get('number') if jersey else ''}".casefold()
            if q and q.casefold() not in text:
                continue
            entries.append({'video_id': video['video_id'], 'filename': video['filename'],
                            'person_id': person.get('person_id'), 'raw_track_ids': person.get('raw_track_ids', []),
                            'identity_label': person.get('identity_label'), 'cover': person.get('cover') or {},
                            'jersey': {'number': jersey.get('number'), 'status': jersey.get('status')} if jersey else None,
                            'status': person.get('status'), 'media_prefix': prefix})
    return {'items': entries[offset:offset + limit], 'total': len(entries),
            'offset': offset, 'limit': limit, 'identity_scope': 'clip_local'}
