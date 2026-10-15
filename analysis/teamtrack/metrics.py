"""TrackEval metrics, count-weighted grouping, and audited CLEAR frame traces."""
from __future__ import annotations

import csv
import importlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from .dataset import save_json, sha256

ROOT = Path(__file__).resolve().parents[2]


def trackeval_module():
    path = ROOT / 'MOTIP/TrackEval'
    sys.path.insert(0, str(path)) if str(path) not in sys.path else None
    module = importlib.import_module('trackeval')
    if not Path(module.__file__).resolve().is_relative_to(path):
        raise RuntimeError('Expected the repository-bundled TrackEval, not another installed version')
    return module


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f'No rows to write: {path}')
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def clear_trace(data, raw, directory, expected):
    """Mirror bundled CLEAR matching (including its empty-frame state handling).

    This is diagnostic only. Published metrics always come from TrackEval.
    Totals must agree with TrackEval or evaluation fails.
    """
    last = np.full(data['num_gt_ids'], np.nan)
    previous = last.copy()
    totals = dict(TP=0, FP=0, FN=0, IDSW=0)
    frames = []
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'clear_matches.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['frame', 'event', 'gt_id', 'tracker_id', 'iou', 'id_switch'])
        writer.writeheader()
        for t, (gt, pred) in enumerate(zip(data['gt_ids'], data['tracker_ids'])):
            similarity = data['similarity_scores'][t]
            rows, cols = np.array([], dtype=int), np.array([], dtype=int)
            switches = np.array([], dtype=bool)
            if len(gt) and len(pred):
                score = 1000 * (pred[None, :] == previous[gt[:, None]]) + similarity
                score[similarity < .5 - np.finfo(float).eps] = 0
                rows, cols = linear_sum_assignment(-score)
                keep = score[rows, cols] > np.finfo(float).eps
                rows, cols = rows[keep], cols[keep]
                switches = ~np.isnan(last[gt[rows]]) & (last[gt[rows]] != pred[cols])
                last[gt[rows]] = pred[cols]
                previous[:] = np.nan
                previous[gt[rows]] = pred[cols]
            # Both supported inputs contain evaluable player GT only, so preprocessing
            # preserves detection ordering. Only ID relabelling differs.
            raw_gt, raw_pred = raw['gt_ids'][t], raw['tracker_ids'][t]
            for i, j, switched in zip(rows, cols, switches):
                writer.writerow(dict(frame=t+1, event='TP', gt_id=int(raw_gt[i]),
                                     tracker_id=int(raw_pred[j]), iou=float(similarity[i, j]), id_switch=int(switched)))
            for i in sorted(set(range(len(gt))) - set(rows)):
                writer.writerow(dict(frame=t+1, event='FN', gt_id=int(raw_gt[i]), tracker_id='', iou='', id_switch=0))
            for j in sorted(set(range(len(pred))) - set(cols)):
                writer.writerow(dict(frame=t+1, event='FP', gt_id='', tracker_id=int(raw_pred[j]), iou='', id_switch=0))
            counts = dict(TP=len(rows), FP=len(pred)-len(rows), FN=len(gt)-len(rows), IDSW=int(sum(switches)))
            for field in totals:
                totals[field] += counts[field]
            frames.append(dict(frame=t+1, GT=len(gt), predictions=len(pred), **counts,
                               recall=100*counts['TP']/max(1, len(gt)),
                               precision=100*counts['TP']/max(1, len(pred)),
                               **{'cumulative_'+k: v for k, v in totals.items()},
                               cumulative_recall=100*totals['TP']/max(1, totals['TP']+totals['FN']),
                               cumulative_precision=100*totals['TP']/max(1, totals['TP']+totals['FP'])))
    for field, official in [('TP', 'CLR_TP'), ('FP', 'CLR_FP'), ('FN', 'CLR_FN'), ('IDSW', 'IDSW')]:
        if totals[field] != expected[official]:
            raise AssertionError(f'CLEAR diagnostic mismatch: {field}: {totals[field]} != {expected[official]}')
    write_csv(directory / 'clear_frames.csv', frames)


def serial(value):
    if isinstance(value, dict):
        return {k: serial(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def evaluate_files(folder, sequences, *, do_preproc=False):
    """Evaluate exactly the declared sequences; missing prediction files are errors."""
    trackeval = trackeval_module()
    metrics = {'HOTA': trackeval.metrics.HOTA(),
               'CLEAR': trackeval.metrics.CLEAR({'PRINT_CONFIG': False, 'THRESHOLD': .5}),
               'Identity': trackeval.metrics.Identity({'PRINT_CONFIG': False, 'THRESHOLD': .5})}
    dataset = trackeval.datasets.MotChallenge2DBox({
        'GT_FOLDER': str(folder / 'gt'), 'TRACKERS_FOLDER': str(folder / 'predictions'),
        'TRACKERS_TO_EVAL': ['motip'], 'SKIP_SPLIT_FOL': True, 'DO_PREPROC': do_preproc,
        'PRINT_CONFIG': False, 'SEQ_INFO': {s['sequence']: s['frames'] for s in sequences},
    })
    results = {}
    destination = folder / 'metrics'
    destination.mkdir(parents=True, exist_ok=True)
    for seq in sequences:
        name = seq['sequence']
        raw = dataset.get_raw_seq_data('motip', name)
        if do_preproc and any(
            np.any(classes != 1) or np.any(extras['zero_marked'] == 0)
            for classes, extras in zip(raw['gt_classes'], raw['gt_extras'])
        ):
            raise ValueError('Frame trace requires player-only GT without ignored rows; review the dataset protocol')
        data = dataset.get_preprocessed_seq_data(raw, 'pedestrian')
        results[name] = {key: metric.eval_sequence(data) for key, metric in metrics.items()}
        clear_trace(data, raw, destination / 'sequences' / name, results[name]['CLEAR'])
        print(f'Evaluated {name}', flush=True)
    groups = []
    for seq in sequences:
        groups.append(('sequence', seq['view'], seq['sequence'], [seq['sequence']]))
    for view in sorted({s['view'] for s in sequences}):
        groups.append(('view', view, 'COMBINED_SEQ', [s['sequence'] for s in sequences if s['view'] == view]))
    groups.append(('overall', 'all', 'COMBINED_SEQ', [s['sequence'] for s in sequences]))
    summaries, thresholds, raw_groups = [], [], []
    by_name = {s['sequence']: s for s in sequences}
    for scope, view, name, members in groups:
        combined = (results[name] if scope == 'sequence' else {
            key: metric.combine_sequences({n: results[n][key] for n in members}) for key, metric in metrics.items()})
        h, c, identity = combined['HOTA'], combined['CLEAR'], combined['Identity']
        context = dict(split=folder.name, scope=scope, view=view, sequence=name)
        summaries.append(dict(**context, sequences=len(members), frames=sum(by_name[n]['frames'] for n in members),
                              HOTA=float(np.mean(h['HOTA']))*100, DetA=float(np.mean(h['DetA']))*100,
                              AssA=float(np.mean(h['AssA']))*100, MOTA=float(c['MOTA'])*100,
                              IDF1=float(identity['IDF1'])*100,
                              TP=int(c['CLR_TP']), FP=int(c['CLR_FP']), FN=int(c['CLR_FN']),
                              recall=float(c['CLR_Re'])*100, precision=float(c['CLR_Pr'])*100,
                              IDSW=int(c['IDSW']), Frag=int(c['Frag']),
                              IDTP=int(identity['IDTP']), IDFP=int(identity['IDFP']), IDFN=int(identity['IDFN']),
                              IDR=float(identity['IDR'])*100, IDP=float(identity['IDP'])*100))
        for i, alpha in enumerate(metrics['HOTA'].array_labels):
            thresholds.append(dict(**context, alpha=round(float(alpha), 2),
                                   **{k: int(h[k][i]) for k in ('HOTA_TP', 'HOTA_FP', 'HOTA_FN')},
                                   **{k: float(h[k][i])*100 for k in ('HOTA', 'DetA', 'AssA', 'DetRe', 'DetPr', 'AssRe', 'AssPr')}))
        raw_groups.append(dict(**context, members=members, metrics=serial(combined)))
    write_csv(destination / 'summary.csv', summaries)
    write_csv(destination / 'hota_thresholds.csv', thresholds)
    save_json(destination / 'trackeval_raw.json', raw_groups)
    save_json(destination / 'metric_protocol.json', {
        'implementation': 'MOTIP/TrackEval', 'do_preproc': do_preproc, 'class': 'player (MOT pedestrian=1)',
        'hota_thresholds': [round(float(a), 2) for a in metrics['HOTA'].array_labels],
        'clear_identity_iou': .5, 'csv_rate_units': 'percent', 'raw_json_rate_units': 'fraction',
        'aggregation': 'TrackEval combine_sequences, never arithmetic mean of sequence scores',
        'frame_trace': 'CLEAR matching only; totals verified against TrackEval',
        'zero_denominator': '0 via max(1, denominator), matching TrackEval convention',
        'code_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in sorted((ROOT / 'MOTIP/TrackEval/trackeval').rglob('*.py'))},
        'reporter_sha256': sha256(__file__),
    })
    return summaries


def evaluate(output, split):
    folder = Path(output).resolve() / split
    save_json(folder / 'evaluation_status.json', {'status': 'running', 'split': split})
    manifest = json.loads((folder / 'dataset.json').read_text())
    protocol_path = folder / 'inference_protocol.json'
    protocol = json.loads(protocol_path.read_text())
    if protocol['dataset_sha256'] != sha256(folder / 'dataset.json'):
        raise ValueError('Dataset manifest differs from inference')
    for seq in manifest['sequences']:
        name = seq['sequence']
        record = json.loads((folder / 'completed' / f'{name}.json').read_text())
        if record['frames'] != seq['frames'] or record['protocol_sha256'] != sha256(protocol_path):
            raise ValueError(f'{name}: incomplete/mismatched inference')
        if sha256(folder / 'predictions/motip/data' / f'{name}.txt') != record['prediction_sha256']:
            raise ValueError(f'{name}: prediction checksum mismatch')
        if sha256(folder / 'gt' / name / 'gt/gt.txt') != seq['gt_sha256']:
            raise ValueError(f'{name}: GT checksum mismatch')
    results = evaluate_files(folder, manifest['sequences'])
    save_json(folder / 'evaluation_status.json', {
        'status': 'completed', 'split': split, 'sequences': len(manifest['sequences']),
        'dataset_sha256': sha256(folder / 'dataset.json'),
        'inference_protocol_sha256': sha256(protocol_path),
        'summary_sha256': sha256(folder / 'metrics/summary.csv'),
    })
    return results
