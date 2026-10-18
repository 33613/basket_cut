"""CPU integration checks for source scopes and evidence-safe incremental memory."""
import hashlib
import json
from pathlib import Path
import tempfile
import subprocess
import unittest

import numpy as np
from PIL import Image

from analysis.gallery_memory_selfcheck import record
from contracts.execution import ExecutionSettings
from contracts.lineage import clip_id_for, clip_context
from pipeline.identity.gallery_memory import MemoryPolicy, plan_memory
from pipeline.identity.jersey_qwen import shared_samples
from pipeline.identity.player_registry import PlayerRegistry
from pipeline.segmentation.service import SegmentOptions, merge_candidates, make_segments, parse_boundary
from pipeline.segmentation.possession import parse_states
from pipeline.segmentation.timeline import reconcile_states, attach_scopes
from workflows.batch import run_batch
from workflows.commands import CommandBuilder


def strong(key, angle=0):
    result = record(key, angle, track=True, number='23', team='dark')
    common = {'method':'clear_player_multiframe_v1', 'status':'candidate_consensus', 'support_frames':2}
    result['node']['team_evidence'] = {**common,'team_id':'dark'}
    result['node']['jersey'] = {**common,'number':'23','joint_support_frames':2}
    return result


def observation(time, team='dark', phase='live_play', window='w'):
    return {'source_observation_s':time,'attacking_team':team,'phase':phase,'window_id':window,'evidence':'synthetic'}


class FlowChecks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_shared_crop_is_identical_and_never_resampled(self):
        root = self.root/'raw'; root.mkdir()
        crop = root/'crop.png'
        Image.fromarray(np.arange(10000,dtype=np.uint8).reshape(100,100)).save(crop)
        checksum = hashlib.sha256(crop.read_bytes()).hexdigest()
        row = {'observation_id':'OBS-test','sample_id':'s','video_id':'v','track_id':1,
               'frame_idx':13,'timestamp_s':.52,'bbox_xyxy':[1,2,101,102],
               'crop_path':'crop.png','crop_sha256':checksum}
        manifest = root/'kpr_samples.jsonl'; manifest.write_text(json.dumps(row)+'\n')
        tracks = self.root/'tracks.jsonl'; tracks.write_text(json.dumps(row)+'\n')
        output = self.root/'identity'; output.mkdir()
        samples = shared_samples(manifest, tracks, output, {1})
        self.assertEqual(samples[0]['observation_id'],'OBS-test')
        self.assertEqual((output/samples[0]['crop_path']).read_bytes(),crop.read_bytes())
        crop.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'provenance'):
            shared_samples(manifest, tracks, output, {1})

    def test_shared_crop_cannot_reuse_other_video_or_escape_archive(self):
        root = self.root/'raw'; root.mkdir(); (root/'crop.png').write_bytes(b'image')
        row = {'observation_id':'id','video_id':'v','track_id':0,'frame_idx':0,
               'timestamp_s':0,'bbox_xyxy':[0,0,10,20],'crop_path':'crop.png',
               'crop_sha256':hashlib.sha256(b'image').hexdigest()}
        manifest=root/'samples.jsonl'; manifest.write_text(json.dumps(row)+'\n')
        tracks=self.root/'tracks.jsonl'; tracks.write_text(json.dumps({**row,'video_id':'wrong'})+'\n')
        with self.assertRaisesRegex(ValueError,'tracking evidence'):
            shared_samples(manifest, tracks, self.root, {0})
        tracks.write_text(json.dumps(row)+'\n')
        manifest.write_text(json.dumps({**row,'crop_path':'../tracks.jsonl'})+'\n')
        with self.assertRaisesRegex(ValueError,'escapes'):
            shared_samples(manifest, tracks, self.root, {0})

    def test_commands_use_shared_manifest_even_without_bootstrap(self):
        paths=CommandBuilder.paths({'output_dir':str(self.root/'clip'),'source_path':str(self.root/'v.mp4')})
        cmd=CommandBuilder(ExecutionSettings(),validate_resources=False).command('jersey',paths,{})
        self.assertIn('--observations',cmd)
        self.assertIn('--assess-player',cmd)
        self.assertEqual(cmd[cmd.index('--observations')+1],str(paths['identity_raw']/'kpr_samples.jsonl'))

    def test_cpu_scene_scan_returns_real_source_relative_cut(self):
        from pipeline.segmentation.timeline import shot_candidates
        source = self.root/'two-shots.mp4'
        subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','color=c=black:s=64x64:r=10:d=2',
            '-f','lavfi','-i','color=c=white:s=64x64:r=10:d=2',
            '-filter_complex','[0:v][1:v]concat=n=2:v=1:a=0[out]', '-map','[out]',
            '-c:v','libx264',str(source)],check=True,capture_output=True)
        proposals=shot_candidates(source,ffmpeg='ffmpeg',threshold=.45)
        self.assertEqual(len(proposals),1)
        self.assertLessEqual(proposals[0]['time_range_s'][0],2)
        self.assertGreaterEqual(proposals[0]['time_range_s'][1],2)
        self.assertEqual(proposals[0]['kind'],'camera_cut')

    def test_state_mode_is_default_and_replay_cannot_claim_live_possession(self):
        self.assertEqual(SegmentOptions().coarse_method,'states')
        raw={'states':[{'id':i,'attacking_team':'dark','phase':'replay','evidence':'synthetic'} for i in range(4)]}
        self.assertTrue(all(s['attacking_team']=='unknown' for s in parse_states(json.dumps(raw),20)['states']))

    def test_candidates_do_not_chain_or_merge_opposite_transitions(self):
        def candidate(low,high,transition):
            return {'time_range_s':[low,high],'kind':'possession_change','transition':transition,'evidence':'x'}
        groups=merge_candidates([candidate(0,4,['dark','light']),candidate(3,7,['dark','light']),
            candidate(6,10,['dark','light']),candidate(3,5,['light','dark'])],0)
        self.assertEqual(len(groups),3)
        self.assertEqual(sorted(len(g['support']) for g in groups),[1,1,2])

    def test_unknown_conflicting_and_stale_states_schedule_review(self):
        rows=[observation(0),observation(5,'unknown','unknown'),observation(10,'unknown','unknown'),observation(15,'light')]
        _, proposals, _=reconcile_states(rows,20)
        self.assertTrue(any(r['kind']=='uncertain' for r in proposals))
        self.assertFalse(any(r['kind']=='possession_change' for r in proposals))
        timeline,proposals,_=reconcile_states([observation(0),observation(0,'light',window='other')],20)
        self.assertEqual(timeline[0]['phase'],'unknown')
        self.assertTrue(proposals)
        _,proposals,_=reconcile_states([observation(t) for t in range(0,61,5)],65)
        self.assertTrue(any('long unchanged' in r['evidence'] for r in proposals))

    def test_replay_blocks_cross_replay_team_transition(self):
        _,proposals,_=reconcile_states([observation(0),observation(5,'unknown','replay'),observation(10,'light')],15)
        self.assertEqual([p['kind'] for p in proposals],['content_transition','content_transition'])

    def test_shots_split_tracking_but_not_possession_and_replay_is_excluded(self):
        reviews=[]
        for time,kind,after in [(5,'camera_cut',None),(10,'possession_change',None),
                                (15,'content_transition','replay'),(20,'content_transition','game')]:
            reviews.append({'source_boundary_s':time,'kind':kind,'confirmed':True,'needs_review':False,
                            'candidate_id':str(time),'source_range_s':[time-1,time+1],'content_after':after})
        segments,_=make_segments('SRC-test',25,reviews,.5)
        segments=attach_scopes(segments,reviews)
        self.assertEqual(segments[0]['possession_id'],segments[1]['possession_id'])
        self.assertNotEqual(segments[0]['shot_id'],segments[1]['shot_id'])
        self.assertNotEqual(segments[1]['possession_id'],segments[2]['possession_id'])
        self.assertFalse(segments[3]['processing_eligible'])
        self.assertIsNone(segments[3]['possession_id'])
        self.assertTrue(segments[4]['partial_possession_start'])
        self.assertTrue(segments[4]['partial_possession_end'])

    def test_coincident_shot_and_possession_do_not_create_fake_unresolved_cut(self):
        reviews=[{'source_boundary_s':5,'kind':kind,'confirmed':True,'needs_review':False,
                  'candidate_id':kind,'source_range_s':[4,6]} for kind in ('camera_cut','possession_change')]
        segments,pending=make_segments('SRC-test',10,reviews,.5)
        self.assertEqual(len(segments),2); self.assertFalse(pending)
        segments=attach_scopes(segments,reviews)
        self.assertNotEqual(segments[0]['possession_id'],segments[1]['possession_id'])
        self.assertNotEqual(segments[0]['shot_id'],segments[1]['shot_id'])

    def test_verified_content_transition_requires_explicit_label(self):
        row={'kind':'content_transition','confirmed':True,'needs_review':False,'boundary_s':2,'evidence':'replay logo'}
        with self.assertRaises(ValueError):parse_boundary(json.dumps(row),4)
        self.assertEqual(parse_boundary(json.dumps({**row,'content_after':'replay'}),4)['content_after'],'replay')

    def test_batch_skips_replay_before_applying_limit_and_keeps_scopes(self):
        data=self.root/'data'; data.mkdir()
        rows=[]
        for i,content in enumerate(('replay','game')):
            path=data/f'{i}.mp4'; path.write_bytes(bytes([i]))
            rows.append({'clip_id':clip_id_for('SRC-test',i,i+1),'path':path.name,'start_s':i,'end_s':i+1,
                         'source_video_id':'SRC-test','source_sha256':'sourcehash','source_path':'original.mp4',
                         'sha256':hashlib.sha256(path.read_bytes()).hexdigest(), 'needs_review':False,
                         'content_type':content,'processing_eligible':content=='game','possession_id':'POS-test','shot_id':'SHOT-test'})
        manifest=data/'clips_manifest.json'
        manifest.write_text(json.dumps({'source':{'source_video_id':'SRC-test','sha256':'sourcehash','path':'original.mp4'},'videos':rows}))
        result=run_batch(ExecutionSettings(),data,self.root/'results',clips_manifest=manifest,limit=1,dry_run=True)
        batch=json.loads((self.root/'results/batch_manifest.json').read_text()) if (self.root/'results/batch_manifest.json').exists() else result["plan"]
        # Dry runs return the complete planned clip rows without executing models.
        self.assertEqual(len(batch['clips']),1)
        self.assertEqual(batch['clips'][0]['input'],'1.mp4')
        self.assertEqual(clip_context(batch,batch['clips'][0]['name'])['possession_id'],'POS-test')

    def test_status_counts_only_eligible_clips_as_required_work(self):
        from workflows.match_report import match_report
        root=self.root/'match'; (root/'results/game').mkdir(parents=True)
        (root/'segmentation').mkdir()
        (root/'segmentation/clips_manifest.json').write_text(json.dumps({'videos':[
            {'processing_eligible':False},{'processing_eligible':True}]}))
        (root/'results/batch_manifest.json').write_text(json.dumps({'clips':[{'name':'game'}]}))
        (root/'results/game/batch_status.json').write_text(json.dumps({'status':'completed'}))
        report=match_report(root)
        self.assertEqual(report['status'],'completed')
        self.assertEqual(report['planned_processing_clips'],1)
        self.assertEqual(report['segmentation']['excluded_content_clips'],1)

    def test_incremental_batch_does_not_use_new_template_as_bridge(self):
        a=record('a0',0,track=True); a.update(established=True,active_template_keys=['prototype-0'])
        b=record('b2',.18,track=True); c=record('c4',.35,track=True)
        plan=plan_memory([a,b,c],{},MemoryPolicy(min_margin=0),incremental=True)
        group=next(g for g in plan['groups'] if 0 in g['members'])
        self.assertEqual(group['members'],[0,1])
        self.assertEqual(plan['states'][1]['match_dependencies']['observations'],['a0'])
        self.assertEqual(plan['states'][2]['status'],'pending')

    def test_strong_identity_links_without_admitting_bad_image(self):
        a,b=strong('a0'),strong('b2',1)
        a.update(established=True,active_template_keys=['prototype-0'])
        plan=plan_memory([a,b],{'b2':{'templates':[],'hold_reasons':['no_clear_assessed_kpr_images']}},
                         MemoryPolicy(),incremental=True)
        self.assertEqual(plan['states'][1]['status'],'accepted')
        self.assertTrue(plan['states'][1]['association_only'])
        self.assertFalse(plan['states'][1]['templates'])
        self.assertEqual(len(plan['groups']),1)

    def test_conflicting_simultaneous_arrivals_cannot_choose_ordered_winner(self):
        a=record('a0',0,track=True); a.update(established=True,active_template_keys=['prototype-0'])
        b,c=record('b2',.01,track=True,frame='same'),record('c4',.02,track=True,frame='same')
        plan=plan_memory([a,b,c],{},MemoryPolicy(min_margin=0),incremental=True)
        self.assertEqual(next(g['members'] for g in plan['groups'] if 0 in g['members']),[0])
        self.assertEqual([s['status'] for s in plan['states'][1:]],['pending','pending'])

    def test_dependency_rejection_revokes_event_and_rollback_restores(self):
        with PlayerRegistry(self.root/'players.sqlite3','match') as registry:
            def ingest(r,event=()):
                return registry.ingest([{**r['node'],'feature_indices':[0]}],np.array(r['features']),
                    np.array(r['visibility']),event,{'metadata':{'model':'synthetic'}},memory_policy=MemoryPolicy(min_margin=0))
            a,b=record('a0',0,track=True),record('b2',.15,track=True)
            first=ingest(a)
            event={'clip_name':'b2','event_id':'shot','raw_track_ids':[0],'event':'shot','start':0.,'end':1.}
            second=ingest(b,[event])
            member=next(m for p in second['people'] for m in p['members'] if m['node_id']=='b2')
            self.assertEqual(member['memory']['matched_against_revision'],first['revision'])
            self.assertIsNotNone(second['events'][0]['player_id'])
            rev=registry.review('reject_observation',node_id='a0',expected_revision=second['revision'])
            after=registry.snapshot()
            self.assertIsNone(after['events'][0]['player_id'])
            registry.review('rollback_memory',expected_revision=rev)
            self.assertEqual(registry.snapshot()['events'][0]['player_id'],second['events'][0]['player_id'])


def main():
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(FlowChecks))
    if not result.wasSuccessful():raise SystemExit(1)

if __name__=='__main__':main()
