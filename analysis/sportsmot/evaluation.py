"""Evaluate each native-frame path and report geometric and identity equivalence."""
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from analysis.teamtrack.dataset import save_json, sha256
from analysis.teamtrack.metrics import evaluate_files, write_csv
from .runner import prediction_rows

HEADLINE = ('HOTA', 'DetA', 'AssA', 'MOTA', 'IDF1', 'TP', 'FP', 'FN', 'IDSW', 'IDTP', 'IDFP', 'IDFN')


def compare_predictions(official, adapter, frames, box_tolerance=.01):
    a, b = defaultdict(list), defaultdict(list)
    for row in prediction_rows(official, frames):
        a[int(row[0])].append(row)
    for row in prediction_rows(adapter, frames):
        b[int(row[0])].append(row)
    count_mismatch, bad_geometry, bad_identity, checked = 0, 0, 0, 0
    max_abs, mapping, inverse = 0., {}, {}
    for frame in range(1, frames+1):
        left, right = a[frame], b[frame]
        if len(left) != len(right):
            count_mismatch += 1
        if not left or not right:
            continue
        aa, bb = np.asarray(left), np.asarray(right)
        distances = np.abs(aa[:, None, 2:6] - bb[None, :, 2:6]).max(2)
        rows, cols = linear_sum_assignment(distances)
        for i, j in zip(rows, cols):
            delta = float(distances[i, j])
            max_abs = max(max_abs, delta)
            checked += 1
            if delta > box_tolerance:
                bad_geometry += 1
                continue
            source, target = int(aa[i, 1]), int(bb[j, 1])
            if mapping.get(source, target) != target or inverse.get(target, source) != source:
                bad_identity += 1
            else:
                mapping[source], inverse[target] = target, source
    return dict(frames=frames, count_mismatch_frames=count_mismatch, compared_boxes=checked,
                geometry_mismatches=bad_geometry, identity_mapping_conflicts=bad_identity,
                maximum_box_abs_difference_px=max_abs, box_tolerance_px=box_tolerance,
                global_id_mapping={str(k): v for k, v in mapping.items()},
                equivalent=count_mismatch == bad_geometry == bad_identity == 0)


def evaluate(output, mode):
    output = Path(output).resolve()
    save_json(output / 'status.json', dict(status='running', mode=mode))
    manifest = json.loads((output / 'dataset.json').read_text())
    shared = json.loads((output / 'experiment_protocol.json').read_text())
    methods = ('official', 'adapter') if mode == 'parity' else (mode,)
    summaries = {}
    for method in methods:
        folder = output / method / 'val'
        protocol = json.loads((folder / 'inference_protocol.json').read_text())
        if {key: value for key, value in protocol.items() if key != 'method'} != shared:
            raise ValueError('Methods did not use the same checkpoint/code/experiment protocol')
        if protocol['dataset_sha256'] != sha256(output / 'dataset.json'):
            raise ValueError('Inference dataset manifest changed')
        for seq in manifest['sequences']:
            name = seq['sequence']
            record = json.loads((folder / 'completed' / f'{name}.json').read_text())
            if record['frames'] != seq['frames'] or record['protocol_sha256'] != sha256(folder / 'inference_protocol.json'):
                raise ValueError(f'{method}/{name}: incomplete or mismatched run')
            if record['prediction_sha256'] != sha256(folder / 'predictions/motip/data' / f'{name}.txt'):
                raise ValueError(f'{method}/{name}: changed prediction')
            if sha256(folder / 'gt' / name / 'gt/gt.txt') != seq['gt_sha256']:
                raise ValueError(f'{method}/{name}: changed GT')
        summaries[method] = evaluate_files(folder, manifest['sequences'], do_preproc=True)
    rows = [dict(method=method, **r) for method, values in summaries.items() for r in values]
    write_csv(output / 'summary.csv', rows)
    comparison = []
    if mode == 'parity':
        for seq in manifest['sequences']:
            name = seq['sequence']
            paths = [output / m / 'val/predictions/motip/data' / f'{name}.txt' for m in methods]
            result = compare_predictions(*paths, seq['frames'])
            completion = json.loads((output / 'adapter/val/completed' / f'{name}.json').read_text())
            probes = completion['preprocessing_probes']
            result['preprocessing_probes'] = probes
            result['preprocessing_close'] = all(p['tensor_max_abs'] <= 1e-6 for p in probes)
            comparison.append(dict(sequence=name, **result))
        left = {(r['scope'], r['view'], r['sequence']): r for r in summaries['official']}
        metric_deltas = [dict(scope=r['scope'], sequence=r['sequence'],
                             **{key: float(r[key])-float(left[(r['scope'], r['view'], r['sequence'])][key]) for key in HEADLINE})
                         for r in summaries['adapter']]
        write_csv(output / 'metric_deltas.csv', metric_deltas)
        save_json(output / 'parity.json', dict(
            note='Unmodified upstream submit CLI versus native-frame adapter; no accuracy target is assumed.',
            box_tolerance_px=.01, headline_tolerance_percentage_points=.01,
            confidence_column='Ignored for parity: upstream writes 1, adapter retains actual score; TrackEval has no score filtering.',
            sequences=comparison))
        count_fields = {'TP', 'FP', 'FN', 'IDSW', 'IDTP', 'IDFP', 'IDFN'}
        metric_match = all(abs(row[key]) <= (0 if key in count_fields else .01) for row in metric_deltas for key in HEADLINE)
        parity_ok = metric_match and all(r['equivalent'] and r['preprocessing_close'] for r in comparison)
    else:
        parity_ok = None
    status = dict(status='completed', mode=mode, sequences=len(manifest['sequences']),
                  processed_frames_per_method=sum(s['frames'] for s in manifest['sequences']),
                  parity_passed=parity_ok, dataset_sha256=sha256(output / 'dataset.json'),
                  summary_sha256=sha256(output / 'summary.csv'))
    save_json(output / 'status.json', status)
    for row in rows:
        if row['scope'] == 'overall':
            print(json.dumps(row, ensure_ascii=False, indent=2), flush=True)
    print('RUN_COMPLETE', json.dumps(status, ensure_ascii=False), flush=True)
    return status
