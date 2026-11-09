"""Reviewed goal tracks -> existing KPR/registry -> per-player demo videos."""

import csv
import hashlib
import json
import math
from pathlib import Path
import re
import posixpath
import xml.etree.ElementTree as ET
import zipfile

from contracts.schema import TrackRecord, read_jsonl, write_jsonl_line
from pipeline.media import atomic_json, sha256_file
from workflows import link_player as flow


def read(path):
    return json.loads(Path(path).read_text())


def truth(value):
    value = str(value or '').strip().lower()
    if value in ('1', '1.0', 'true', 'yes', '是'):
        return True
    if value in ('0', '0.0', 'false', 'no', '否'):
        return False
    if not value:
        return None
    raise ValueError(f'Invalid yes/no value: {value}')


def track_id(value):
    match = re.fullmatch(r'[Tt]?(\d+)(?:\.0)?', str(value).strip())
    if not match:
        raise ValueError(f'Invalid track ID: {value}')
    return int(match[1])


def review_table(path):
    """Read CSV or a review_checklist XLSX without requiring pandas/openpyxl."""
    path = Path(path)
    if path.suffix.lower() != '.xlsx':
        with path.open(encoding='utf-8-sig', newline='') as stream:
            return list(csv.DictReader(stream))
    ns = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    relns = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
    with zipfile.ZipFile(path) as archive:
        strings = []
        if 'xl/sharedStrings.xml' in archive.namelist():
            strings = [''.join(t.text or '' for t in si.findall('.//s:t', ns))
                       for si in ET.fromstring(archive.read('xl/sharedStrings.xml')).findall('s:si', ns)]
        sheets = ET.fromstring(archive.read('xl/workbook.xml')).findall('s:sheets/s:sheet', ns)
        selected = next((s for s in sheets if s.get('name') == 'review_checklist'), None)
        if selected is None:
            if len(sheets) != 1:
                raise ValueError('XLSX must have one sheet or a review_checklist sheet')
            selected = sheets[0]
        relations = ET.fromstring(archive.read('xl/_rels/workbook.xml.rels'))
        relation = next(r for r in relations if r.get('Id') == selected.get('{'+relns+'}id'))
        if relation.get('TargetMode') == 'External':
            raise ValueError('External XLSX worksheet is not supported')
        target = relation.get('Target', '')
        name = posixpath.normpath(target.lstrip('/') if target.startswith('/') else 'xl/'+target)
        if not name.startswith('xl/'):
            raise ValueError('Invalid XLSX worksheet path')
        matrix = []
        for row in ET.fromstring(archive.read(name)).findall('s:sheetData/s:row', ns):
            cells = {}
            for c in row.findall('s:c', ns):
                letters = re.match(r'[A-Z]+', c.get('r', '')).group()
                col = 0
                for letter in letters:
                    col = col*26 + ord(letter)-64
                value = c.findtext('s:v', '', ns)
                if c.get('t') == 's':
                    value = strings[int(value)]
                elif c.get('t') == 'inlineStr':
                    value = ''.join(t.text or '' for t in c.findall('.//s:t', ns))
                if c.find('s:f', ns) is not None:
                    raise ValueError('Review XLSX contains formulas; export values-only first')
                if value:
                    cells[col-1] = value
            if cells:
                matrix.append(cells)
    if not matrix:
        raise ValueError('Empty review worksheet')
    width = max(max(row) for row in matrix)+1
    headers = [matrix[0].get(i, '').strip() for i in range(width)]
    note_column = next((i for i, h in enumerate(headers) if not h and any(row.get(i) for row in matrix[1:])), None)
    headers = [h or ('notes' if i == note_column and 'notes' not in headers else f'unnamed_{i+1}')
               for i, h in enumerate(headers)]
    if len(headers) != len(set(headers)):
        raise ValueError('Duplicate review worksheet headers')
    return [{key: row.get(i, '') for i, key in enumerate(headers)} for row in matrix[1:]]


def reviewed_rows(path, events):
    by_id = {e['event_id']: e for e in events}
    by_video = {}
    for e in events:
        by_video.setdefault(e['video_id'], []).append(e)
    result = {}
    for raw in review_table(path):
        row = {str(k).strip(): str(v or '').strip() for k, v in raw.items() if k}
        if not any(row.values()):
            continue
        eid = row.get('event_id')
        if not eid:
            choices = by_video.get(row.get('video_id', ''), [])
            if len(choices) != 1:
                raise ValueError('video_id must identify one event; otherwise supply event_id')
            eid = choices[0]['event_id']
        if eid not in by_id or eid in result:
            raise ValueError(f'Unknown/duplicate review event: {eid}')
        if row.get('video_id') and row['video_id'] != by_id[eid]['video_id']:
            raise ValueError('Review event_id/video_id mismatch')
        if not row.get('reviewed_track_id'):
            row['reviewed_track_id'] = row.get('corrected_track_id') or row.get('轨迹号码', '')
        if not row.get('player_key'):
            value = next((v for k, v in row.items() if k.startswith('球员号码')), '')
            if re.fullmatch(r'\d{1,2}\s*[A-Za-z]+', value):
                row['player_key'] = re.sub(r'\s+', '', value).lower()
        if not row.get('player_key') and row.get('team_id') and row.get('jersey_number'):
            row['player_key'] = row['team_id'] + ':' + row['jersey_number']
        jersey = re.fullmatch(r'(\d{1,2})([A-Za-z]+)', row.get('player_key', ''))
        if jersey:
            row['player_key'] = jersey[1] + jersey[2].lower()
            row.setdefault('team_id', '')
            row.setdefault('jersey_number', '')
            row['team_id'] = row['team_id'] or jersey[2].lower()
            row['jersey_number'] = row['jersey_number'] or jersey[1]
        result[eid] = row
    return result


def target_segments(event, result, review):
    """Inclusive SOURCE frame bounds. Mixed identities require explicit repair."""
    explicit = review.get('source_segments', '')
    if explicit:
        segments = []
        for part in explicit.split(';'):
            match = re.fullmatch(r'\s*[Tt]?(\d+)\s*:\s*(\d+)\s*-\s*(\d+)\s*', part)
            if not match:
                raise ValueError('source_segments must look like T2:100-130;T4:131-200')
            segments.append(dict(track_id=int(match[1]), start=int(match[2]), end=int(match[3])))
    else:
        corrected = review.get('reviewed_track_id', '')
        tid = track_id(corrected) if corrected else result['shooter_track_id']
        if tid is None:
            return [], 'missing_reviewed_track'
        repaired_bounds = review.get('review_start_frame') and review.get('review_end_frame')
        if truth(review.get('track_pure')) is False and not repaired_bounds:
            return [], 'mixed_track_needs_source_segments_or_review_bounds'
        trajectory = result.get('trajectory') or {}
        use_auto = not corrected or tid == result['shooter_track_id']
        start = (trajectory.get('source_start_frame', event['window_start_frame'])
                 if use_auto else event['window_start_frame'])
        end = (trajectory.get('source_end_frame_inclusive', event['window_end_frame_exclusive']-1)
               if use_auto else event['window_end_frame_exclusive']-1)
        if repaired_bounds:
            start, end = int(review['review_start_frame']), int(review['review_end_frame'])
        elif review.get('source_start_frame') and review.get('source_end_frame_inclusive') and use_auto:
            start, end = int(review['source_start_frame']), int(review['source_end_frame_inclusive'])
        segments = [dict(track_id=tid, start=start, end=end)]
    for s in segments:
        if not event['window_start_frame'] <= s['start'] <= s['end'] < event['window_end_frame_exclusive']:
            raise ValueError(f"{event['event_id']}: target segment outside tracked window")
    return segments, None


def prepare(link_run, review_csv, output, *, identity_mode='reviewed', samples=6):
    from pipeline.identity.sampling import select_quality_samples
    from pipeline.identity.archive import load_selected_crops

    link_run, output = Path(link_run).resolve(), Path(output).resolve()
    if link_run == output or link_run in output.parents or output in link_run.parents:
        raise ValueError('Demo output must be separate from the inference run')
    if identity_mode not in ('reviewed', 'kpr') or samples < 1:
        raise ValueError('Invalid demo options')
    protocol = flow.saved(link_run)
    reviews = reviewed_rows(review_csv, protocol['events'])
    request = dict(version=3, link_run=str(link_run), protocol_sha256=sha256_file(link_run/'protocol.json'),
                   review_sha256=sha256_file(review_csv), identity_mode=identity_mode, samples=samples)
    request_path = output/'request.json'
    if request_path.exists():
        if read(request_path) != request:
            raise ValueError('Review/options changed; use a new demo output directory')
        if (output/'plan.json').is_file():
            return read(output/'plan.json')
    elif output.exists() and any(output.iterdir()):
        raise ValueError('New demo output directory must be empty')
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(request_path, request)
    atomic_json(output/'review_rows.json', reviews)
    plan = dict(match_id=protocol['match_id'], request=request, items=[], skipped=[])
    source_hashes = {}
    for index, event in enumerate(protocol['events'], 1):
        eid = event['event_id']
        row = reviews.get(eid, {})
        corrected = bool(row.get('reviewed_track_id') or row.get('source_segments'))
        include = truth(row.get('include'))
        if include is False or (include is not True and not corrected and truth(row.get('shooter_correct')) is not True):
            plan['skipped'].append(dict(event_id=eid, reason='not_selected_by_review'))
            continue
        directory = flow.folder(link_run, event)
        flow.require_stage(directory, 'link')
        flow.require_stage(directory, 'track')
        result = read(directory/'link.json')
        segments, reason = target_segments(event, result, row)
        if reason:
            plan['skipped'].append(dict(event_id=eid, reason=reason))
            continue
        source = protocol['sources'][event['video_id']]
        source_hashes.setdefault(source['path'], None)
        if source_hashes[source['path']] is None:
            source_hashes[source['path']] = sha256_file(source['path'])
        if source_hashes[source['path']] != source['sha256']:
            raise ValueError('Original source video changed')
        fps, offset = source['fps'], event['window_start_frame']
        records = [TrackRecord.from_dict(r) for r in read_jsonl(directory/'tracks.jsonl')]
        target = [r for r in records if any(r.track_id == s['track_id'] and
                  s['start'] <= r.frame_idx+offset <= s['end'] for s in segments)]
        if not target or len({r.frame_idx for r in target}) != len(target):
            raise ValueError(f'{eid}: target has no observations or multiple people in one frame')
        for r in target:
            if r.video_id != eid or abs(r.timestamp_s-r.frame_idx/fps) > 1e-5:
                raise ValueError('Target observation clock mismatch')
        dest = output/eid
        (dest/'tracking').mkdir(parents=True, exist_ok=True)
        selected, quality = select_quality_samples(directory/'window.mkv', records, purpose='person',
            sample_count=samples, min_score=.5, min_width=24, min_height=64, min_gap_s=.15,
            eligible_keys={(r.track_id, r.frame_idx) for r in target})
        quality_passed = bool(selected)
        if not selected:
            selected = [target[len(target)//2]]
        _, images = load_selected_crops(directory/'window.mkv', selected, padding=0.,
            context_padding=.25, output_dir=dest/'identity', prompt_mode='none', prompts={})
        for image in images:
            image['source_frame_idx'] = image['frame_idx']+offset
            image.update(quality.get((image['track_id'], image['frame_idx']), {}))
            overlap = image.get('sampling_components', {}).get('overlap_fraction', 1.)
            image['identity_eligible'] = quality_passed and overlap <= .45
        images.sort(key=lambda s: (-s['identity_eligible'], -s.get('sampling_quality', 0.), s['frame_idx']))
        with (dest/'target_tracks.jsonl').open('w') as stream:
            for r in target:
                value = r.to_dict()
                value.update(frame_idx=r.frame_idx+offset, timestamp_s=(r.frame_idx+offset)/fps)
                write_jsonl_line(stream, value)
        atomic_json(dest/'tracking/video_meta.json', dict(input_path=source['path'],
            fps=fps, width=source['width'], height=source['height'], frame_count=source['frame_count'],
            tracking_scope='reviewed_goal_segments_only'))
        ball_path = directory/'ball/ball.jsonl'
        if ball_path.is_file():
            expected = read(directory/'.ball.json')['outputs']['ball/ball.jsonl']
            if sha256_file(ball_path) != expected:
                raise ValueError('Ball artifact changed')
        incomplete = truth(row.get('highlight_complete')) is False
        clip_start = float(row.get('clip_start_s') or (0 if incomplete else event['window_start_s']))
        clip_end = float(row.get('clip_end_s') or (source['duration_s'] if incomplete else event['window_end_s']))
        if not all(math.isfinite(v) for v in (clip_start, clip_end)) or not 0 <= clip_start < clip_end <= source['duration_s']:
            raise ValueError('Invalid source clip range')
        item = dict(event_id=eid, video_id=event['video_id'], player_key=row.get('player_key', ''),
            display_name=row.get('player_name') or row.get('player_key', ''),
            team_id=row.get('team_id', ''), jersey_number=row.get('jersey_number', ''),
            segments=segments, fps=fps, source=source,
            window_start_s=event['window_start_s'], window_end_s=event['window_end_s'],
            clip_start_s=clip_start, clip_end_s=clip_end,
            clip_range_policy='full_source_for_incomplete_highlight' if incomplete else 'window_or_review_bounds',
            ball_path=str(ball_path) if ball_path.is_file() else None,
            ball_sha256=sha256_file(ball_path) if ball_path.is_file() else None,
            samples=images, feature_eligible=any(s['identity_eligible'] for s in images) and truth(row.get('images_usable')) is not False,
            sampling_quality_passed=quality_passed,
            automatic_track_id=result['shooter_track_id'], reviewed_target=corrected,
            score=next((c['score'] for c in result.get('candidates', [])
                        if c['track_id'] in {s['track_id'] for s in segments}), 0.))
        atomic_json(dest/'item.json', item)
        plan['items'].append(item)
        print(f'[demo prepare] {index}/{len(protocol["events"])} video {event["video_id"]}', flush=True)
    if not plan['items']:
        raise ValueError('No reviewed targets selected; fill shooter_correct or reviewed_track_id/source_segments')
    atomic_json(output/'plan.json', plan)
    return plan


def template_indices(embeddings, visibility, samples, *, limit=4):
    """Drop isolated appearance outliers, then keep supported, diverse real views."""
    import numpy as np
    from pipeline.identity.cross_clip import part_distances
    from pipeline.identity.gallery_memory import select_templates

    distance = part_distances(embeddings, visibility)
    visible = np.count_nonzero(np.asarray(visibility) > 0, axis=1) >= 2
    supported = visible & ((distance <= .4).sum(axis=1) >= (2 if len(samples) > 1 else 1))
    bank = {i: dict(quality=s.get('sampling_quality', .5), view='unknown',
                    source_frame=s['source_frame_idx']) for i, s in enumerate(samples) if supported[i]}
    ranked = sorted(bank, key=lambda i: (-bank[i]['quality'], samples[i]['source_frame_idx']))
    protected = []
    for i in ranked:
        if all(abs(samples[i]['timestamp_s']-samples[j]['timestamp_s']) >= .15-1e-6 for j in protected):
            protected.append(i)
        if len(protected) == 2:
            break
    selected = select_templates(bank, distance, limit, protected=protected)
    # Track fragments can be adjacent in time despite per-track sample spacing.
    spaced = []
    for i in selected:
        if all(abs(samples[i]['timestamp_s']-samples[j]['timestamp_s']) >= .15-1e-6 for j in spaced):
            spaced.append(i)
    return spaced, [dict(source_frame_idx=s['source_frame_idx'],
        accepted=i in spaced, reason=('selected' if i in spaced else
            'insufficient_visible_parts' if not visible[i] else
            'isolated_appearance' if not supported[i] else 'redundant_or_bank_limit'))
        for i, s in enumerate(samples)]


def verify_sample_images(output, plan):
    from workflows.library_inputs import safe_relative
    paths = []
    for item in plan['items']:
        for sample in item['samples']:
            path = safe_relative(output, f"{item['event_id']}/identity/{sample['crop_path']}")
            if sha256_file(path) != sample['crop_sha256']:
                raise ValueError('Identity input crop changed after preparation')
            paths.append(path)
    return paths


def features(output, *, kpr_root, config, checkpoint, cpu=False, backend=None):
    import numpy as np
    import cv2
    from pipeline.identity.kpr_backend import KPRBackend

    output = Path(output).resolve()
    plan = read(output/'plan.json')
    if plan['request'].get('version') not in (2, 3):
        raise ValueError('KPR sampling policy changed; prepare into a new demo output directory')
    verify_sample_images(output, plan)
    contract = dict(backend='KPR', checkpoint_sha256=sha256_file(checkpoint),
                    config_sha256=sha256_file(config), prompt_mode='none',
                    template_policy='quality_visible_multiframe_v2', template_limit=4)
    signature = dict(plan_sha256=sha256_file(output/'plan.json'), contract=contract)
    marker = output/'features.json'
    if marker.exists():
        previous = read(marker)
        if previous['signature'] != signature or previous['sha256'] != sha256_file(output/'features.npz'):
            raise ValueError('KPR inputs/features changed; use a new output directory')
        return previous
    vectors, weights, ids, event_indices, audits = [], [], [], {}, {}
    for item in plan['items']:
        if not item['feature_eligible']:
            audits[item['event_id']] = dict(reason='no_usable_identity_images', samples=[])
            continue
        samples = [s for s in item['samples'] if s.get('identity_eligible', False)]
        paths = [output/item['event_id']/'identity'/s['crop_path'] for s in samples]
        for path, sample in zip(paths, samples):
            if sha256_file(path) != sample['crop_sha256']:
                raise ValueError('KPR input crop changed after preparation')
        images = [cv2.imread(str(path)) for path in paths]
        if any(image is None for image in images):
            raise ValueError('Missing identity sample image')
        if backend is None:
            backend = KPRBackend(kpr_root=Path(kpr_root), config_path=Path(config),
                checkpoint_path=Path(checkpoint), use_gpu=not cpu)
        embeddings, visibility = backend.extract([{'image': image} for image in images], batch_size=8)
        e, v = np.asarray(embeddings, dtype=np.float32), np.asarray(visibility, dtype=np.float32)
        if e.ndim != 3 or len(e) != len(samples) or v.shape != e.shape[:2]:
            raise ValueError('KPR output/sample layout differs')
        selected, audit = template_indices(e, v, samples)
        audits[item['event_id']] = dict(samples=audit, matching_eligible=len(selected)>=2)
        if not selected:
            continue
        event_indices[item['event_id']] = list(range(len(vectors), len(vectors)+len(selected)))
        vectors.extend(e[selected])
        weights.extend(v[selected])
        ids.append(item['event_id'])
        print(f'[demo KPR] {len(ids)}/{len(plan["items"])} {item["event_id"]}', flush=True)
    np.savez(output/'features.npz', embeddings=np.asarray(vectors), visibility=np.asarray(weights))
    result = dict(signature=signature, event_ids=ids, event_indices=event_indices,
                  template_audits=audits, contract=contract,
                  sha256=sha256_file(output/'features.npz'))
    atomic_json(marker, result)
    return result


def build_library(output, *, max_distance=.2):
    import numpy as np
    from pipeline.identity.player_registry import PlayerRegistry
    from workflows.person_library import export_library
    from workflows.library_inputs import file_stamp

    output = Path(output).resolve()
    plan = read(output/'plan.json')
    image_paths = verify_sample_images(output, plan)
    vectors, weights, indices, contract = [], [], {}, None
    if plan['request']['identity_mode'] == 'kpr':
        feature_info = read(output/'features.json')
        if feature_info['signature']['plan_sha256'] != sha256_file(output/'plan.json') or feature_info['sha256'] != sha256_file(output/'features.npz'):
            raise ValueError('KPR feature provenance changed')
        contract = feature_info['contract']
        if contract.get('template_policy') != 'quality_visible_multiframe_v2':
            raise ValueError('Old averaged KPR features; prepare/extract in a new demo output directory')
        with np.load(output/'features.npz', allow_pickle=False) as data:
            vectors, weights = list(data['embeddings']), list(data['visibility'])
        indices = feature_info.get('event_indices') or {eid: [i] for i, eid in enumerate(feature_info['event_ids'])}
    nodes, events = [], []
    for item in plan['items']:
        eid, source = item['event_id'], item['source']
        tids = sorted({s['track_id'] for s in item['segments']})
        samples = item['samples']
        media = lambda s: dict(s, crop_path=f"{eid}/identity/{s['crop_path']}",
                               context_path=f"{eid}/identity/{s['context_path']}")
        nodes.append(dict(node_id='N-'+hashlib.sha256(eid.encode()).hexdigest()[:20],
            clip_name=eid, video_id=item['video_id'], local_person_id='shooter', raw_track_ids=tids,
            intervals=[[s['start']/item['fps'], (s['end']+1)/item['fps']] for s in item['segments']],
            review_player_key=item['player_key'], team_id=item['team_id'],
            sample_count=len(samples), observation_count=sum(1 for _ in read_jsonl(output/eid/'target_tracks.jsonl')),
            hold_reasons=[] if item['feature_eligible'] else ['images_not_usable_for_gallery'],
            cover=media(samples[0]), exemplars=[media(s) for s in samples[:4]],
            feature_indices=indices.get(eid, []),
            jersey=dict(number=item['jersey_number'] or None,
                        status='candidate_consensus' if item['jersey_number'] else 'unreadable')))
        events.append(dict(clip_name=eid, event_id=eid, video_id=item['video_id'],
            local_person_id='shooter', raw_track_ids=tids, event='made_shot',
            start=item['clip_start_s'], end=item['clip_end_s'],
            raw_score=item['score'], source_path=source['path'], source_sha256=source['sha256'],
            source_clip_start_s=0., source_clip_end_s=source['duration_s'],
            source_start_s=item['clip_start_s'], source_end_s=item['clip_end_s'],
            reviewed_target=item['reviewed_target']))
    base = output/'library'
    base.mkdir(exist_ok=True)
    if (base/'library_manifest.json').is_file() and not (base/'players.sqlite3').is_file():
        raise ValueError('Player database missing; restore it to preserve player IDs')
    registration = dict(plan_sha256=sha256_file(output/'plan.json'),
                        features_sha256=sha256_file(output/'features.npz') if indices else None,
                        max_distance=max_distance)
    if (base/'request.json').exists() and read(base/'request.json') != registration:
        raise ValueError('Library inputs/settings changed; use a new output directory')
    atomic_json(base/'request.json', registration)
    with PlayerRegistry(base/'players.sqlite3', plan['match_id']) as registry:
        # Explicit reviewed identities form the gallery; only unlabelled KPR nodes use automatic matching.
        reviewed = [n for n in nodes if n['review_player_key'] or not n['feature_indices']]
        automatic = [n for n in nodes if not n['review_player_key'] and n['feature_indices']]
        registry.ingest_reviewed(reviewed, features=vectors, visibility=weights, contract=contract)
        if automatic:
            registry.ingest(automatic, vectors, weights, [], contract, max_distance=max_distance,
                            template_matching=True)
        registry.ingest_reviewed([], events)
        signatures = {'plan.json': file_stamp(output/'plan.json')}
        for path in image_paths:
            signatures[path.relative_to(output).as_posix()] = file_stamp(path)
        if plan['request']['identity_mode'] == 'kpr':
            for name in ('features.npz', 'features.json'):
                signatures[name] = file_stamp(output/name)
        for item in plan['items']:
            for name in ('item.json', 'target_tracks.jsonl', 'tracking/video_meta.json'):
                p = output/item['event_id']/name
                signatures[p.relative_to(output).as_posix()] = file_stamp(p)
        return export_library(registry, output, base, signatures=signatures,
            options=dict(demo=True, identity_mode=plan['request']['identity_mode']),
            clips=[i['event_id'] for i in plan['items']])
