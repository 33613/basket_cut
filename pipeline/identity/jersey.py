"""Combine per-track number evidence without treating matching digits as identity."""


def number_policy(signature):
    """Compare acceptance rules, excluding legacy shared-crop sample counts.

    The shared KPR path recorded the observed maximum in ``max_samples``;
    unlike independent sampling it does not cap or select any Qwen inputs.
    Normalize only for comparison, preserving stored evidence/fingerprints.
    """
    result = dict(signature)
    thresholds = dict(result.get('thresholds') or {})
    if thresholds.get('sampling_strategy') == 'shared_kpr_observations_v1':
        thresholds.pop('max_samples', None)
    result['thresholds'] = thresholds
    return result


def group_numbers(ids: list[int], tracks: dict[int, dict]) -> dict:
    rows = [tracks.get(tid, {'number': None, 'status': 'unreadable'}) for tid in ids]
    numbers = {r['number'] for r in rows if r.get('number') is not None}
    status = ('conflict' if len(numbers) > 1 or any(r['status'] == 'conflict' for r in rows)
              else 'candidate_consensus' if len(numbers) == 1 and all(r.get('number') is not None for r in rows)
              else 'partial_evidence' if numbers else 'unreadable')
    return {'number': next(iter(numbers)) if status == 'candidate_consensus' else None,
            'status': status, 'candidate_numbers': sorted(numbers), 'verified': False,
            'raw_track_ids': ids}
