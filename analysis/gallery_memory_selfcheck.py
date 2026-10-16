"""CPU regression checks for abstention, bounded galleries and reversible updates."""
from dataclasses import replace
import json
import math
from pathlib import Path
import tempfile
import unittest
import subprocess
import sys

import numpy as np

from pipeline.identity.gallery_memory import MemoryPolicy, plan_memory
from pipeline.identity.player_registry import PlayerRegistry
from workflows.gallery_maintenance import maintain
from analysis.person_library_selfcheck import make_library_fixture
from workflows.person_library import PersonLibraryOptions, build_person_library


def record(key, angle, *, frame=None, number=None, team=None, track=False, quality=.9):
    node = {"node_id":key, "clip_name":frame or key, "video_id":"source", "filename":"source.mp4",
        "local_person_id":key,"raw_track_ids":[0] if track else [],"intervals":[[0,1]] if track else [],
        "hold_reasons":[],"cover":{"archive_quality_score":quality},"sample_count":4 if track else 1,
        "timestamp_s":float(''.join(c for c in key if c.isdigit()) or 0),
        "jersey":{"number":None,"candidate_numbers":[number] if number else []},"team_id":team}
    if not track: node["evidence_kind"]="bootstrap_photo"
    f = [[math.cos(angle),math.sin(angle)]]*2
    return {"node":node,"player":"p-"+key,"features":[f],"visibility":[[1.,1.]]}


class GalleryMemoryChecks(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix="basket-memory-")
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.registry=PlayerRegistry(self.root/"players.sqlite3","test")
        self.addCleanup(self.registry.db.close)
        self.policy=MemoryPolicy(min_margin=0)
        self.contract={"metadata":{"checkpoint_sha256":"synthetic"},"shape":[2,2]}

    def ingest(self, records, policy=None, evidence=None, events=()):
        nodes=[{**r["node"],"feature_indices":[i]} for i,r in enumerate(records)]
        return self.registry.ingest(nodes,np.asarray([r["features"][0] for r in records]),
            np.asarray([r["visibility"][0] for r in records]),events,self.contract,
            memory_policy=policy or self.policy,curation=evidence)

    def test_cold_start_does_not_require_distance_point_five(self):
        records=[record('a0',0,frame='f0',number='23'),record('b0',.6,frame='f0',number='32'),
                 record('a2',.02,frame='f2',number='23'),record('b2',.62,frame='f2',number='32')]
        snap=self.ingest(records)
        self.assertEqual(len(snap['people']),2)
        self.assertTrue(all(p['status']=='candidate' for p in snap['people']))
        self.assertTrue(all(len(p['members'])==2 for p in snap['people']))
        self.assertEqual(self.ingest(records)['revision'],snap['revision'])

    def test_different_numbers_and_teams_veto_close_uniforms(self):
        records=[record('a0',0,number='41',team='white'),record('b2',.34,number='32',team='white'),
                 record('c4',.01,number='41',team='black')]
        plan=plan_memory(records,{},self.policy)
        self.assertEqual(len(plan['groups']),3)
        self.assertFalse(any(g['promoted'] for g in plan['groups']))

    def test_complete_link_prevents_unknown_transitive_bridge(self):
        records=[record('a0',0),record('b2',.14),record('c4',.28)]
        plan=plan_memory(records,{},replace(self.policy,strong_distance=.08))
        self.assertEqual(sorted(len(g['members']) for g in plan['groups']),[1,2])

    def test_appearance_rivals_abstain_and_single_ocr_does_not_relax_threshold(self):
        records=[record('a0',0,track=True),record('b2',.1,track=True),record('c4',.05,track=True)]
        plan=plan_memory(records,{},replace(self.policy,min_margin=.05))
        self.assertTrue(all(not g['promoted'] for g in plan['groups']))
        weak=[record('a0',0,number='41'),record('b2',.34,number='41')]
        plan=plan_memory(weak,{},self.policy)
        self.assertEqual(len(plan['groups']),2)
        self.assertFalse(any(g['promoted'] for g in plan['groups']))

    def test_source_feature_mutation_is_refused_and_reject_is_sticky(self):
        a,b=record('a0',0,track=True),record('b2',.01,track=True)
        snap=self.ingest([a,b],evidence={'a0':{'source_fingerprint':'source-one'}})
        with self.assertRaisesRegex(ValueError,'Registered image/sample'):
            self.registry.refresh_memory(evidence={'a0':{'source_fingerprint':'changed'}},expected_revision=snap['revision'])
        self.assertEqual(self.registry.meta('revision'),snap['revision'])
        rev=self.registry.review('reject_observation',node_id='a0',expected_revision=snap['revision'])
        self.registry.review('reprocess',expected_revision=rev)
        member=next(m for p in self.registry.snapshot()['people'] for m in p['members'] if m['node_id']=='a0')
        self.assertTrue(member['memory']['manual_rejected'])
        self.assertFalse(member['gallery_eligible'])

    def test_pending_observation_revisited_when_new_track_arrives(self):
        first=self.ingest([record('a0',0)])
        self.assertEqual(first['people'][0]['status'],'needs_review')
        second=self.ingest([record('b2',.01,track=True)])
        person=next(p for p in second['people'] if p['status']=='candidate')
        self.assertEqual(len(person['members']),2)
        self.assertTrue(all(m['gallery_eligible'] for m in person['members']) or person['gallery_template_count']>=1)

    def test_reinstating_an_earlier_sample_preserves_the_established_player_id(self):
        first=self.ingest([record('a0',0,track=True),record('b2',.01,track=True)])
        pid=first['people'][0]['player_id']
        rev=self.registry.review('reject_observation',node_id='a0',expected_revision=first['revision'])
        rev=self.registry.review('reinstate_observation',node_id='a0',expected_revision=rev)
        snap=self.registry.snapshot()
        self.assertEqual(len(snap['people']),1)
        self.assertEqual(snap['people'][0]['player_id'],pid)

    def test_duplicate_detection_not_independent_support(self):
        a,b=record('a0',0,frame='f0'),record('b0',.01,frame='f0')
        evidence={r['node']['node_id']:{'bbox_xyxy':[0,0,100,200]} for r in(a,b)}
        plan=plan_memory([a,b],evidence,self.policy)
        self.assertEqual(len(plan['duplicates']),1)
        self.assertFalse(any(g['promoted'] for g in plan['groups']))

    def test_quality_rejection_and_ambiguous_overlap_do_not_enter_gallery(self):
        records=[record('a0',0,frame='f0'),record('b0',1,frame='f0'),record('c2',0,quality=.1)]
        evidence={'a0':{'bbox_xyxy':[0,0,100,200]},'b0':{'bbox_xyxy':[10,0,100,180]}}
        plan=plan_memory(records,evidence,self.policy)
        self.assertIn('overlapping_person_targets',plan['states'][0]['reasons'])
        self.assertEqual(plan['states'][2]['status'],'rejected')
        self.assertFalse(any(s['templates'] for s in plan['states']))

    def test_template_limit_prefers_quality_and_declared_views(self):
        records=[record(f'a{i*2}',i*.02,number='23',quality=.5+i*.05) for i in range(8)]
        snap=self.ingest(records,replace(self.policy,gallery_limit=3))
        person=next(p for p in snap['people'] if p['status']=='candidate')
        self.assertLessEqual(person['gallery_template_count'],3)
        chosen=[m for m in person['members'] if m['gallery_eligible']]
        self.assertIn('a14',{m['node_id'] for m in chosen})
        self.assertEqual(len(person['members']),8)

    def test_reject_and_rollback_restore_assignments_templates_and_events(self):
        a,b=record('a0',0,track=True),record('b2',.01,track=True)
        event={'clip_name':'b2','event_id':'e1','raw_track_ids':[0],'local_person_id':'b2','event':'shot','start':0.,'end':1.}
        before=self.ingest([a,b],events=[event])
        original=before['events'][0]['player_id']
        rev=self.registry.review('reject_observation',node_id='b2',expected_revision=before['revision'])
        rejected=self.registry.snapshot()
        self.assertIsNone(rejected['events'][0]['player_id'])
        self.assertEqual(next(m for p in rejected['people'] for m in p['members'] if m['node_id']=='b2')['memory']['status'],'rejected')
        with self.assertRaises(RuntimeError):
            self.registry.review('rollback_memory',expected_revision=before['revision'])
        self.registry.review('rollback_memory',expected_revision=rev)
        after=self.registry.snapshot()
        self.assertEqual(after['events'][0]['player_id'],original)
        self.assertEqual([p['gallery_template_count'] for p in after['people']],[p['gallery_template_count'] for p in before['people']])
        self.assertEqual(self.registry.db.execute('SELECT COUNT(*) FROM observations').fetchone()[0],2)

    def test_manual_confirm_exclude_and_detach_survive_reprocess(self):
        snap=self.ingest([record('a0',0,track=True),record('b2',1,track=True)])
        left,right=snap['people']
        self.registry.review('confirm',person_id=left['player_id'],expected_revision=snap['revision'])
        snap=self.registry.snapshot()
        self.registry.review('exclude',person_id=right['player_id'],expected_revision=snap['revision'])
        snap=self.registry.snapshot()
        self.registry.review('reprocess',expected_revision=snap['revision'])
        states={p['player_id']:p['status'] for p in self.registry.snapshot()['people']}
        self.assertEqual(states[left['player_id']],'manual_grouping')
        self.assertEqual(states[right['player_id']],'non_player')

    def test_invalid_policy_or_features_leave_revision_unchanged(self):
        snap=self.ingest([record('a0',0)])
        with self.assertRaises(ValueError):
            self.registry.refresh_memory(replace(self.policy,strong_distance=float('nan')),expected_revision=snap['revision'])
        self.assertEqual(self.registry.snapshot()['revision'],snap['revision'])

    def test_preview_migration_does_not_modify_production_and_mutation_has_backup(self):
        run=make_library_fixture(self.root/'fixture')
        old=build_person_library(PersonLibraryOptions(run,memory_enabled=False))
        database=run/'library/players.sqlite3'
        original=database.read_bytes()
        preview=maintain(run,operation='preview',output_dir=self.root/'preview')
        self.assertFalse(preview['production_modified'])
        self.assertEqual(database.read_bytes(),original)
        self.assertTrue((self.root/'preview/index.html').is_file())
        applied=maintain(run,operation='reprocess',expected_revision=old['review_revision'])
        self.assertTrue(applied['production_modified'])
        self.assertTrue((run/'library/backups'/f'{old["review_revision"]}.sqlite3').is_file())
        maintain(run,operation='rollback_memory',expected_revision=applied['revision'])

    def test_cli_preview_produces_portable_report_without_gpu(self):
        run=make_library_fixture(self.root/'cli-fixture')
        build_person_library(PersonLibraryOptions(run,memory_enabled=False))
        result=subprocess.run([sys.executable,'-m','cli.maintain_players','--run-dir',str(run),
            '--output-dir',str(self.root/'cli-preview')],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        report=json.loads(result.stdout)
        self.assertFalse(report['production_modified'])
        self.assertTrue((self.root/'cli-preview/index.html').is_file())

    def test_tracks_use_image_features_and_changed_sample_archive_is_refused(self):
        from PIL import Image
        run=make_library_fixture(self.root/'image-features',angles=(0,))
        raw=run/'clip1/identity_raw';rows=[]
        for i in range(2):
            gray=np.tile(np.arange(128,dtype=np.uint8),(256,1))
            gray[::4]=255
            path=raw/f'media/sample-{i}.png'
            Image.fromarray(gray).save(path)
            rows.append({'track_id':0,'frame_idx':i*25,'sample_id':f's-{i}','crop_path':f'media/sample-{i}.png'})
        (raw/'kpr_samples.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        vectors=np.array([[[1,0],[1,0]],[[math.cos(.2),math.sin(.2)]]*2],dtype=np.float32)
        np.savez(raw/'kpr_samples.npz',embeddings=vectors,visibility_scores=np.ones((2,2)),track_ids=[0,0],frame_indices=[0,25])
        first=build_person_library(PersonLibraryOptions(run))
        self.assertEqual(first['gallery_template_count'],2)
        with PlayerRegistry(run/'library/players.sqlite3','synthetic-match') as registry:
            templates=registry.snapshot()['people'][0]['members'][0]['memory']['templates']
            self.assertTrue(all(t['kind']=='image_sample' for t in templates))
        self.assertEqual(build_person_library(PersonLibraryOptions(run))['review_revision'],first['review_revision'])
        np.savez(raw/'kpr_samples.npz',embeddings=vectors[::-1],visibility_scores=np.ones((2,2)),track_ids=[0,0],frame_indices=[0,25])
        with self.assertRaisesRegex(ValueError,'Registered image/sample'):
            build_person_library(PersonLibraryOptions(run))


def main():
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(GalleryMemoryChecks))
    if not result.wasSuccessful(): raise SystemExit(1)


if __name__=='__main__': main()
