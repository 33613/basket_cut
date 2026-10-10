"""Combine per-track number evidence without treating matching digits as identity."""


def group_numbers(ids: list[int], tracks: dict[int, dict]) -> dict:
    rows = [tracks.get(tid, {'number': None, 'status': 'unreadable'}) for tid in ids]
    numbers = {r['number'] for r in rows if r.get('number') is not None}
    status = ('conflict' if len(numbers) > 1 or any(r['status'] == 'conflict' for r in rows)
              else 'candidate_consensus' if len(numbers) == 1 and all(r.get('number') is not None for r in rows)
              else 'partial_evidence' if numbers else 'unreadable')
    return {'number': next(iter(numbers)) if status == 'candidate_consensus' else None,
            'status': status, 'candidate_numbers': sorted(numbers), 'verified': False,
            'raw_track_ids': ids}
