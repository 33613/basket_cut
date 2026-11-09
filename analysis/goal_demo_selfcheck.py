"""CPU demo integration using real H.264 media and synthetic reviewed tracks."""

import csv
from collections import deque
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
import zipfile
from unittest.mock import patch

import cv2
import numpy as np

from contracts.schema import TrackRecord, write_jsonl_line, read_jsonl
from pipeline.media import atomic_json, cut_video, sha256_file
from pipeline.identity.player_registry import PlayerRegistry
from pipeline.identity.cross_clip import supported_set_distance
from pipeline.player_link.demo_render import draw, render_player_videos
from workflows import goal_demo as demo, link_player as link
from workflows.player_events import query_player_events


class DemoTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.run, self.output = self.base/'links', self.base/'demo'
        self.run.mkdir()
        original = self.base/'original.mp4'
        subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i',
            'testsrc2=size=160x120:rate=10:duration=4', '-f', 'lavfi', '-i',
            'sine=frequency=440:duration=4', '-c:v', 'libx264', '-c:a', 'aac',
            '-threads', '1', str(original)], check=True)
        self.protocol = dict(match_id='demo-match', sources={}, events=[])
        for vid in ('1', '2'):
            eid = 'EV-'+vid
            source = dict(path=str(original), sha256=sha256_file(original), fps=10,
                width=160, height=120, duration_s=4, frame_count=40)
            self.protocol['sources'][vid] = source
            event = dict(event_id=eid, video_id=vid, window_start_frame=10,
                window_end_frame_exclusive=30, window_start_s=1, window_end_s=3)
            self.protocol['events'].append(event)
            directory = self.run/'events'/eid
            directory.mkdir(parents=True)
            cut_video(original, directory/'window.mkv', 1, 3, lossless=True)
            with (directory/'tracks.jsonl').open('w') as stream:
                for f in range(20):
                    for tid, box in ((0, (10,20,50,100)), (1, (90,20,140,100))):
                        write_jsonl_line(stream, TrackRecord(eid, f, f/10, tid, box, .9, 0).to_dict())
            atomic_json(directory/'video_meta.json', {})
            link.mark(directory, 'track', {}, ['tracks.jsonl', 'video_meta.json'])
            atomic_json(directory/'link.json', dict(status='candidate', shooter_track_id=0,
                trajectory=dict(source_start_frame=12, source_end_frame_inclusive=24),
                candidates=[dict(track_id=0, score=.9)]))
            link.mark(directory, 'link', {}, ['link.json'])
            (directory/'ball').mkdir()
            with (directory/'ball/ball.jsonl').open('w') as stream:
                for sf in range(10, 30):
                    write_jsonl_line(stream, dict(source_frame_idx=sf, source_timestamp_s=sf/10,
                        ball_track=dict(ball_track_id=0, center_xy=[70,15], observed=True, confirmed=True)))
            link.mark(directory, 'ball', {}, ['ball/ball.jsonl'])
        atomic_json(self.run/'protocol.json', self.protocol)
        self.review = self.base/'review.csv'
        fields = ['video_id', 'shooter_correct', 'track_pure', 'images_usable',
                  'player_key', 'source_segments']
        with self.review.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerow(dict(video_id='1', shooter_correct=1, track_pure=1,
                                 images_usable=1, player_key='27g'))
            writer.writerow(dict(video_id='2', shooter_correct=0, track_pure=0,
                images_usable=0, player_key='27g', source_segments='T0:12-15;T1:16-24'))

    def tearDown(self):
        self.temp.cleanup()

    def test_reviewed_end_to_end_registry_query_render_audio_and_resume(self):
        original = sha256_file(self.run/'events/EV-2/link.json')
        plan = demo.prepare(self.run, self.review, self.output)
        self.assertEqual(len(plan['items']), 2)
        self.assertTrue(plan['items'][1]['reviewed_target'])
        manifest = demo.build_library(self.output)
        self.assertEqual(manifest['global_person_count'], 1)
        self.assertEqual(manifest['event_count'], 2)
        pid = next(read_jsonl(self.output/'library/people.jsonl'))['player_id']
        self.assertEqual(query_player_events(self.output, pid)['total'], 2)
        rendered = render_player_videos(self.output, width=160, height=120)
        self.assertEqual(rendered['players'], 1)
        self.assertEqual(rendered['videos'][0]['clips'], 2)
        self.assertTrue((self.output/rendered['combined']).is_file())
        video = self.output/rendered['videos'][0]['video']
        media = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_streams',
            '-show_format', '-of', 'json', str(video)]))
        self.assertEqual({s['codec_type'] for s in media['streams']}, {'video', 'audio'})
        self.assertAlmostEqual(float(media['format']['duration']), 4, delta=.25)
        before = manifest['review_revision']
        self.assertEqual(demo.build_library(self.output)['review_revision'], before)
        self.assertEqual(demo.prepare(self.run, self.review, self.output), plan)
        self.assertEqual(sha256_file(self.run/'events/EV-2/link.json'), original)
        self.assertIn('27g', (self.output/'index.html').read_text())
        with PlayerRegistry(self.output/'library/players.sqlite3', 'demo-match') as registry:
            self.assertTrue(all(json.loads(r[0]) == [] for r in registry.db.execute('SELECT features FROM observations')))
        self.review.write_text(self.review.read_text()+'\n')
        with self.assertRaisesRegex(ValueError, 'changed'):
            demo.prepare(self.run, self.review, self.output)

    def test_mixed_track_guard_and_source_frame_segments(self):
        event = self.protocol['events'][0]
        result = link.read(self.run/'events/EV-1/link.json')
        segments, reason = demo.target_segments(event, result, {'track_pure':'0'})
        self.assertEqual(segments, [])
        self.assertIn('mixed_track', reason)
        with self.assertRaisesRegex(ValueError, 'outside'):
            demo.target_segments(event, result, {'source_segments':'T0:0-5'})
        rows = demo.reviewed_rows(self.review, self.protocol['events'])
        self.assertEqual(rows['EV-1']['team_id'], 'g')
        self.assertEqual(rows['EV-1']['jersey_number'], '27')

    def test_draw_only_target_and_confirmed_ball_never_connects_ids(self):
        image = np.zeros((120,160,3), dtype=np.uint8)
        target = dict(bbox_xyxy=[10,30,40,100])
        trail = deque()
        observed = dict(ball_track=dict(observed=True, confirmed=True,
                        ball_track_id=0, center_xy=[70,80]))
        draw(image, target, observed, trail, frame_idx=10, fps=30, label='27g')
        self.assertTrue(image[50,10].any())
        self.assertFalse(image[50,100].any())
        predicted = dict(ball_track=dict(observed=False, confirmed=True,
                         ball_track_id=0, center_xy=[100,80]))
        image = np.zeros_like(image)
        draw(image, None, predicted, trail, frame_idx=11, fps=30, label='27g')
        self.assertFalse(image[80,100].any())
        observed['ball_track'].update(ball_track_id=1, center_xy=[120,80])
        draw(image, None, observed, trail, frame_idx=12, fps=30, label='27g')
        self.assertEqual(len(trail), 1)
        self.assertFalse(image[80,95].any())

    def test_kpr_worker_and_existing_matcher_use_real_features(self):
        checkpoint, config = self.base/'weights', self.base/'config'
        checkpoint.write_bytes(b'test-checkpoint')
        config.write_text('test')
        self.review.write_text(self.review.read_text().replace('27g', ''))
        demo.prepare(self.run, self.review, self.output, identity_mode='kpr')
        class Backend:
            def extract(self, samples, batch_size):
                return np.tile([[[1.,0.],[0.,1.]]], (len(samples),1,1)), np.ones((len(samples),2))
        demo.features(self.output, kpr_root=self.base, config=config,
                      checkpoint=checkpoint, backend=Backend())
        info = json.loads((self.output/'features.json').read_text())
        self.assertEqual(info['event_ids'], ['EV-1'])
        result = demo.build_library(self.output)
        self.assertEqual(result['event_count'], 2)
        self.assertEqual(result['global_person_count'], 2)
        # The unreadable, unlabelled second event is kept separate, not discarded or force-matched.

    def test_unlabelled_events_can_match_through_existing_kpr_distance(self):
        checkpoint, config = self.base/'weights', self.base/'config'
        checkpoint.write_bytes(b'test-checkpoint')
        config.write_text('test')
        text = self.review.read_text().replace('27g', '').replace('2,0,0,0,', '2,0,0,1,')
        self.review.write_text(text)
        demo.prepare(self.run, self.review, self.output, identity_mode='kpr')
        class Backend:
            def extract(self, samples, batch_size):
                return np.tile([[[1.,0.],[0.,1.]]], (len(samples),1,1)), np.ones((len(samples),2))
        demo.features(self.output, kpr_root=self.base, config=config,
                      checkpoint=checkpoint, backend=Backend())
        result = demo.build_library(self.output)
        self.assertEqual(result['global_person_count'], 1)
        self.assertEqual(result['event_count'], 2)

    def test_occluder_boxes_are_used_but_never_selected_as_the_target(self):
        directory = self.run/'events/EV-1'
        records = list(read_jsonl(directory/'tracks.jsonl'))
        for row in records:
            if row['track_id'] == 1:
                row['bbox_xyxy'] = [10,20,50,100]
        with (directory/'tracks.jsonl').open('w') as stream:
            for row in records:
                write_jsonl_line(stream, row)
        link.mark(directory, 'track', {}, ['tracks.jsonl', 'video_meta.json'])
        plan = demo.prepare(self.run, self.review, self.output)
        item = plan['items'][0]
        self.assertTrue(all(s['track_id'] == 0 for s in item['samples']))
        self.assertTrue(all(s['sampling_components']['overlap_fraction'] == 1 for s in item['samples']))
        self.assertFalse(item['feature_eligible'])
        self.assertEqual(len(plan['items']), 2)  # Preserve the goal and manual identity.

    def test_fallback_preview_never_enters_kpr_or_loads_model(self):
        with patch('pipeline.identity.sampling.select_quality_samples', return_value=([], {})):
            plan = demo.prepare(self.run, self.review, self.output, identity_mode='kpr')
        self.assertFalse(any(i['feature_eligible'] for i in plan['items']))
        checkpoint, config = self.base/'weights', self.base/'config'
        checkpoint.write_bytes(b'test-checkpoint')
        config.write_text('test')
        with patch('pipeline.identity.kpr_backend.KPRBackend') as backend:
            result = demo.features(self.output, kpr_root=self.base, config=config, checkpoint=checkpoint)
            backend.assert_not_called()
        self.assertEqual(result['event_ids'], [])
        self.assertEqual(demo.build_library(self.output)['event_count'], 2)

    def test_real_templates_retain_views_and_remove_isolated_outlier(self):
        def feature(angle):
            return [[np.cos(angle), np.sin(angle)]]*2
        vectors = np.asarray([feature(0), feature(.05), feature(1.6)], dtype=np.float32)
        samples = [dict(source_frame_idx=i*2, timestamp_s=i*.2, sampling_quality=.5+i*.2) for i in range(3)]
        kept, audit = demo.template_indices(vectors, np.ones((3,2)), samples)
        self.assertEqual(set(kept), {0,1})
        self.assertEqual(audit[2]['reason'], 'isolated_appearance')
        # Equal-looking images from different times can still provide two supports.
        kept, _ = demo.template_indices(vectors[[0,0]], np.ones((2,2)), samples[:2])
        self.assertEqual(len(kept), 2)
        samples[1]['timestamp_s'] = .05
        kept, _ = demo.template_indices(vectors[[0,0]], np.ones((2,2)), samples[:2])
        self.assertEqual(len(kept), 1)

    def test_zero_and_double_zero_are_distinct_player_keys(self):
        self.review.write_text(self.review.read_text().replace('27g', '0g', 1).replace('27g', '00g', 1))
        rows = demo.reviewed_rows(self.review, self.protocol['events'])
        self.assertEqual(rows['EV-1']['jersey_number'], '0')
        self.assertEqual(rows['EV-2']['jersey_number'], '00')
        demo.prepare(self.run, self.review, self.output)
        self.assertEqual(demo.build_library(self.output)['global_person_count'], 2)

    def test_changed_crop_is_rejected_before_feature_extraction(self):
        plan = demo.prepare(self.run, self.review, self.output, identity_mode='kpr')
        item = plan['items'][0]
        path = self.output/item['event_id']/'identity'/item['samples'][0]['crop_path']
        path.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'crop changed'):
            demo.build_library(self.output)

    def test_xlsx_sparse_columns_numeric_zero_and_untitled_notes(self):
        path = self.base/'review.xlsx'
        workbook = ('<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            '<sheets><sheet name="review_checklist" sheetId="1" r:id="rel1"/></sheets></workbook>')
        relationships = ('<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rel1" Target="worksheets/sheet1.xml"/></Relationships>')
        sheet = ('<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>'
            '<row r="1"><c r="A1" t="inlineStr"><is><t>video_id</t></is></c>'
            '<c r="N1" t="inlineStr"><is><t>轨迹号码</t></is></c>'
            '<c r="O1" t="inlineStr"><is><t>球员号码(g表示绿，y表示黄)</t></is></c></row>'
            '<row r="2"><c r="A2"><v>1</v></c><c r="N2"><v>0</v></c>'
            '<c r="O2" t="inlineStr"><is><t>00g</t></is></c>'
            '<c r="P2" t="inlineStr"><is><t>补篮备注</t></is></c></row></sheetData></worksheet>')
        with zipfile.ZipFile(path, 'w') as archive:
            archive.writestr('xl/workbook.xml', workbook)
            archive.writestr('xl/_rels/workbook.xml.rels', relationships)
            archive.writestr('xl/worksheets/sheet1.xml', sheet)
        row = demo.reviewed_rows(path, self.protocol['events'])['EV-1']
        self.assertEqual(row['reviewed_track_id'], '0')
        self.assertEqual(row['player_key'], '00g')
        self.assertEqual(row['notes'], '补篮备注')

    def test_incomplete_highlight_uses_entire_original_not_old_shooter_window(self):
        with self.review.open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        rows[0]['highlight_complete'] = '0'
        with self.review.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        plan = demo.prepare(self.run, self.review, self.output)
        self.assertEqual(plan['items'][0]['clip_start_s'], 0)
        self.assertEqual(plan['items'][0]['clip_end_s'], 4)
        demo.build_library(self.output)
        event = next(read_jsonl(self.output/'library/events.jsonl'))
        self.assertGreaterEqual(event['start'], 0)
        self.assertEqual(event['source_start_s'], 0)


class TemplateMatchingTests(unittest.TestCase):
    def test_one_lucky_pair_does_not_certify_two_images(self):
        self.assertGreater(supported_set_distance([[.01,.9],[.9,.39]]), .2)
        self.assertAlmostEqual(supported_set_distance([[.05,.8],[.8,.07]]), .07, places=5)
        self.assertTrue(np.isinf(supported_set_distance([[.01,.02]])))

    def test_automatic_matches_cannot_create_an_identity_drift_chain(self):
        def feature(angle):
            return [[float(np.cos(angle)), float(np.sin(angle))]]*2
        vectors = [feature(a) for a in (0,.04,.28,.32,.56,.60)]
        weights = [[1.,1.]]*6
        def node(name, indices, key='', team='g', number=None):
            return dict(node_id=name, clip_name=name, video_id=name, local_person_id='shooter',
                raw_track_ids=[0], intervals=[[0,1]], hold_reasons=[], cover={}, exemplars=[],
                sample_count=len(indices), feature_indices=indices, review_player_key=key,
                team_id=team, jersey=dict(number=number, status='candidate_consensus' if number else 'unreadable'))
        with tempfile.TemporaryDirectory() as folder:
            with PlayerRegistry(Path(folder)/'players.sqlite3', 'match') as registry:
                contract = {'test':'multi-image-v2'}
                registry.ingest_reviewed([node('seed',[0,1],key='27g',number='27')],
                    features=vectors, visibility=weights, contract=contract)
                registry.ingest([node('near',[2,3])], vectors, weights, [], contract, template_matching=True)
                stored = registry.db.execute("SELECT gallery,payload FROM observations WHERE id='near'").fetchone()
                self.assertEqual(json.loads(stored['payload'])['decision'], 'matched')
                self.assertEqual(stored['gallery'], 0)
                registry.ingest([node('drift',[4,5])], vectors, weights, [], contract, template_matching=True)
                stored = registry.db.execute("SELECT payload FROM observations WHERE id='drift'").fetchone()
                self.assertEqual(json.loads(stored[0])['decision'], 'pending')
                # A close appearance from an explicitly different team cannot match.
                registry.ingest([node('other-team',[0,1],team='y')], vectors, weights, [], contract, template_matching=True)
                other = registry.db.execute("SELECT payload FROM observations WHERE id='other-team'").fetchone()
                self.assertEqual(json.loads(other[0])['decision'], 'new_candidate')

    def test_similar_teammates_need_margin_and_number_conflicts_veto(self):
        vectors = [[[float(np.cos(a)),float(np.sin(a))]]*2 for a in (0,.02,.08,.10,.04,.06)]
        weights = [[1.,1.]]*6
        def node(name, indices, key='', number=None):
            return dict(node_id=name, clip_name=name, video_id=name, local_person_id='shooter',
                raw_track_ids=[0], intervals=[[0,1]], hold_reasons=[], cover={}, exemplars=[],
                feature_indices=indices, review_player_key=key, team_id='g',
                jersey=dict(number=number, status='candidate_consensus' if number else 'unreadable'))
        with tempfile.TemporaryDirectory() as folder:
            with PlayerRegistry(Path(folder)/'db.sqlite3', 'match') as registry:
                contract = {'test':'multi-image-v2'}
                registry.ingest_reviewed([node('a',[0,1],'27g','27'), node('b',[2,3],'15g','15')],
                    features=vectors, visibility=weights, contract=contract)
                registry.ingest([node('unknown',[4,5])], vectors, weights, [], contract, template_matching=True)
                unknown = registry.db.execute("SELECT payload FROM observations WHERE id='unknown'").fetchone()
                self.assertEqual(json.loads(unknown[0])['decision'], 'pending')
                registry.ingest([node('known-number',[4,5],number='27')], vectors, weights, [], contract, template_matching=True)
                matched = registry.db.execute("SELECT player,payload FROM observations WHERE id='known-number'").fetchone()
                seed = registry.db.execute("SELECT player FROM observations WHERE id='a'").fetchone()
                self.assertEqual(json.loads(matched['payload'])['decision'], 'matched')
                self.assertEqual(matched['player'], seed['player'])


if __name__ == '__main__':
    unittest.main()
