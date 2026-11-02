"""CPU contracts/integration with real media and synthetic person/action evidence."""

from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest

import cv2

from contracts.execution import ExecutionSettings
from contracts.schema import TrackRecord, read_jsonl, write_jsonl_line
from pipeline.action.sampling import physical_sampling, sample_frame_indices
from pipeline.media import atomic_json, sha256_file
from pipeline.player_link.events import LinkOptions, load_events
from pipeline.player_link.evidence import rank_shooters, track_segment
from pipeline.player_link.export import export_run, pack_run
from workflows import link_player as flow


def action(eid, tid, frame, score=.8, label="basketball_2-point_shot", fps=10):
    return dict(video_id=eid, frame_idx=frame, timestamp_s=frame/fps, track_id=tid,
                proposal_frame_offset=0, action_candidates=[dict(label=label, score=score)], selected_actions=[])


def tracks(eid="EV-test", count=80):
    return [TrackRecord(eid, f, f/10, tid, (10+tid*35, 5, 40+tid*35, 85), .9, 0)
            for tid in (0,1) for f in range(count)]


class RankTests(unittest.TestCase):
    def setUp(self):
        self.event = dict(event_id="EV-test", video_id="3", goal_time_s=8,
            window_start_s=1., window_start_frame=10, window_end_frame_exclusive=90,
            partial_pre_context=False)
        self.options = LinkOptions()
        self.rows = tracks()

    def rank(self, actions, event=None, rows=None):
        return rank_shooters(event or self.event, {"fps":10}, self.rows if rows is None else rows, actions, self.options)

    def test_zero_id_and_source_clock_are_preserved(self):
        result = self.rank([action("EV-test",0,50), action("EV-test",0,52)])
        self.assertEqual(result["status"], "candidate")
        self.assertEqual(result["shooter_track_id"], 0)
        self.assertEqual(result["candidates"][0]["peak_source_time_s"], 6.2)
        self.assertTrue(result["requires_review"])
        self.assertFalse(result["ball_verified"])
        self.assertEqual(result["trajectory"]["source_start_frame"],
                         result["trajectory"]["window_start_frame"]+10)

    def test_ambiguous_and_non_shot_and_post_goal_are_not_forced(self):
        points = [action("EV-test",tid,f,.8) for tid in (0,1) for f in (50,52)]
        self.assertEqual(self.rank(points)["status"], "needs_review")
        self.assertIsNone(self.rank(points)["shooter_track_id"])
        for points in ([action("EV-test",0,50,label="basketball_block"),action("EV-test",0,52,label="basketball_block")],
                       [action("EV-test",0,72),action("EV-test",0,74)],
                       [action("EV-test",0,50)]):
            self.assertEqual(self.rank(points)["status"], "no_candidate")

    def test_expanded_search_and_identity_labels_do_not_leak(self):
        points = [action("EV-test",1,f) for f in (12,14)]
        result = self.rank(points)
        self.assertTrue(result["expanded_search"])
        self.assertEqual(result["shooter_track_id"],1)
        modified = dict(self.event, evaluation_labels={"jersey_number":"999", "person_gt_id":"fake"})
        self.assertEqual(result, self.rank(points, event=modified))

    def test_earlier_strong_attempt_does_not_hide_later_putback_candidate(self):
        result=self.rank([action("EV-test",0,f,.95) for f in (40,42)] +
                         [action("EV-test",1,f,.6) for f in (60,62)])
        self.assertEqual(result["status"],"needs_review")
        self.assertIn("later_supported_shot_by_another_actor",result["diagnostics"])

    def test_gaps_split_and_duplicate_or_mixed_evidence_rejected(self):
        candidate = dict(track_id=0, peak_frame=52, action_interval_source_s=[6.,6.2])
        rows = [r for r in self.rows if r.track_id != 0 or r.frame_idx <= 30 or r.frame_idx >= 50]
        segment = track_segment(candidate, rows, 10, self.event, self.options)
        self.assertGreaterEqual(segment["window_start_frame"],50)
        row = action("EV-test",0,50)
        with self.assertRaisesRegex(ValueError,"Duplicate"):
            self.rank([row,row])
        with self.assertRaisesRegex(ValueError,"different event"):
            self.rank([action("wrong",0,50)])
        with self.assertRaisesRegex(ValueError,"Invalid action"):
            self.rank([action("EV-test",0,50,float("nan"))])

    def test_physical_sampling_60fps_keeps_time_span(self):
        interval,cadence = physical_sampling(60,2,2,25)
        self.assertAlmostEqual(interval,4.8)
        self.assertEqual(cadence,5)
        indices,padded = sample_frame_indices(200,500,32,interval)
        self.assertFalse(padded)
        self.assertAlmostEqual((indices[-1]-indices[0])/60,31*2/25)
        self.assertEqual(physical_sampling(60,2,8), (2,8))


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.inputs = self.base / "input"
        self.inputs.mkdir()
        subprocess.run(["ffmpeg","-v","error","-f","lavfi","-i","testsrc2=size=96x96:rate=10:duration=6",
                        "-c:v","libx264","-threads","1",str(self.inputs/"3.mp4")], check=True)
        self.csv = self.base / "events.csv"
        self.csv.write_text("event_id,video_id,filename,goal_time_s,jersey_number\nEV-A,3,3.mp4,4.5,27\nEV-B,3,3.mp4,5.5,27\n")
        self.root = self.base / "result"
        self.options = LinkOptions(lookback_s=2, max_lookback_s=4)
        self.settings = ExecutionSettings()
        self.protocol = flow.prepare(self.inputs,self.csv,self.root,"match",self.options,self.settings)
        for event in self.protocol["events"]:
            directory = flow.folder(self.root,event)
            count = flow.read(directory/"window.json")["frame_count"]
            rows = [r for r in tracks(event["event_id"],count) if not(r.track_id==0 and r.frame_idx==31)]
            with (directory/"tracks.jsonl").open("w") as stream:
                for row in rows:
                    write_jsonl_line(stream,row.to_dict())
            atomic_json(directory/"video_meta.json",dict(video_id=event["event_id"],fps=10,
                        width=96,height=96,processed_frames=count,frame_count=count))
            flow.mark(directory,"track",{},["tracks.jsonl","video_meta.json"])
            # About one second before each goal, in window-local coordinates.
            peak = round((event["goal_time_s"]-event["window_start_s"]-1)*10)
            with (directory/"actions.jsonl").open("w") as stream:
                for f in (peak-2,peak,peak+2):
                    write_jsonl_line(stream,action(event["event_id"],0,f))
            atomic_json(directory/"actions.summary.json",dict(synthetic=True))
            flow.mark(directory,"action",{},["actions.jsonl","actions.summary.json"])

    def tearDown(self):
        self.temp.cleanup()

    def test_full_cpu_link_export_pack_review_and_portability(self):
        flow.link(self.root,self.options)
        export_run(self.root,self.options,self.settings)
        self.assertEqual(flow.status(self.root)["status"],"completed_not_accuracy_evaluated")
        event=self.protocol["events"][0]
        directory=flow.folder(self.root,event)
        actor=flow.read(directory/"handoff/manifest.json")
        self.assertEqual(actor["shooter_track_id"],0)
        self.assertEqual(actor["attribution_status"],"candidate")
        self.assertIsNone(actor["player_id"])
        frames=list(read_jsonl(directory/"handoff/frames.jsonl"))
        self.assertTrue(any(not row["observed"] for row in frames))
        for sample in actor["samples"]:
            image=cv2.imread(str(directory/"handoff"/sample["crop_path"]))
            self.assertEqual(image.shape[:2],(80,30))
            self.assertEqual(sample["source_frame_idx"],sample["window_frame_idx"]+event["window_start_frame"])
        before=sha256_file(directory/"link.json")
        review=self.base/"review.csv"
        review.write_text("event_id,decision,track_id,start_frame,end_frame,notes\nEV-A,confirm,1,20,35,checked\nEV-B,reject,,,,wrong\n")
        export_run(self.root,self.options,self.settings,review)
        self.assertEqual(sha256_file(directory/"link.json"),before)
        self.assertEqual(flow.read(directory/"handoff/manifest.json")["shooter_track_id"],1)
        self.assertFalse((self.root/"events/EV-B/handoff/trajectory.mp4").exists())
        export_run(self.root,self.options,self.settings)  # Resume must not silently undo reviews.
        self.assertEqual(flow.read(directory/"handoff/manifest.json")["shooter_track_id"],1)
        packaged=pack_run(self.root)
        downloaded=self.base/"download"
        downloaded.mkdir()
        with tarfile.open(packaged["archive"]) as tar:
            tar.extractall(downloaded,filter="data")
        subprocess.run(["sha256sum","--quiet","-c","checksums.sha256"],cwd=downloaded/"result",check=True)
        self.assertTrue((downloaded/"result/events/EV-A/handoff/trajectory.mp4").is_file())

    def test_idempotent_prepare_and_mutation_guards(self):
        flow.prepare(self.inputs,self.csv,self.root,"match",self.options,self.settings)
        with self.assertRaisesRegex(ValueError,"changed"):
            flow.prepare(self.inputs,self.csv,self.root,"match",replace(self.options,min_margin=.3),self.settings)
        directory=flow.folder(self.root,self.protocol["events"][0])
        (directory/"actions.jsonl").write_text("")
        with self.assertRaisesRegex(ValueError,"evidence changed"):
            flow.link(self.root,self.options)

    def test_annotation_validation_and_cli_plan(self):
        for bad in ("nan","6","-1"):
            file=self.base/"bad.csv"
            file.write_text(f"video_id,filename,goal_time_s\n3,3.mp4,{bad}\n")
            with self.assertRaises(ValueError):
                load_events(self.inputs,file,"match",self.options)
        result=subprocess.run([sys.executable,"-m","cli.link_player","--input-dir",str(self.inputs),
            "--annotations",str(self.csv),"--output-dir",str(self.base/"dry"),"--dry-run"],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertFalse((self.base/"dry").exists())
        self.assertTrue(json.loads(result.stdout)["dry_run"])

    def test_verified_reuse_preserves_full_clip_ids_and_source_mapping(self):
        old=self.base/"old"
        directory=old/"track/3"
        directory.mkdir(parents=True)
        source=self.protocol["sources"]["3"]
        atomic_json(old/"manifest.json",dict(videos=[dict(video_id="3",source_sha256=source["sha256"])]))
        atomic_json(directory/"video_meta.json",dict(video_id="3",fps=10,processed_frames=60,frame_count=60))
        with (directory/"tracks.jsonl").open("w") as stream:
            for r in tracks("3",60):
                write_jsonl_line(stream,r.to_dict())
        atomic_json(directory/"tracking_complete.json",dict(protocol=dict(checkpoint_sha256="weights",
            config_sha256="config",tracker_options="cli.tracking defaults v1"),
            outputs={name:sha256_file(directory/name) for name in ("tracks.jsonl","video_meta.json")}))
        event=self.protocol["events"][0]
        destination=self.base/"reuse"
        destination.mkdir()
        before=sha256_file(directory/"tracks.jsonl")
        flow.reuse_tracking(dict(self.protocol,tracking_root=str(old)),event,destination,dict(checkpoint="weights",config="config"))
        rows=list(read_jsonl(destination/"tracks.jsonl"))
        self.assertEqual(rows[0]["frame_idx"],0)
        self.assertEqual(rows[0]["video_id"],event["event_id"])
        self.assertEqual(rows[0]["track_id"],0)
        self.assertEqual(before,sha256_file(directory/"tracks.jsonl"))

    def test_all_subprocess_resume_without_gpu_or_controller_deadlock(self):
        weight=self.base/"weights"
        weight.write_bytes(b"synthetic CPU test only")
        config=self.base/"action.py"
        config.write_text("model = {}\n")
        env=dict(os.environ,BASKET_MOTIP_PYTHON=sys.executable,BASKET_ACTION_PYTHON=sys.executable,
                 BASKET_MOTIP_CHECKPOINT=str(weight),BASKET_ACTION_CHECKPOINT=str(weight),BASKET_ACTION_CONFIG=str(config))
        from unittest.mock import patch
        with patch.dict(os.environ,env):
            settings=ExecutionSettings()
        for event in self.protocol["events"]:
            directory=flow.folder(self.root,event)
            flow.mark(directory,"track",dict(model=flow.model_protocol(settings,"track"),
                window=sha256_file(directory/"window.mkv"),reuse=None),["tracks.jsonl","video_meta.json"])
            flow.mark(directory,"action",dict(model=flow.model_protocol(settings,"action"),
                window=sha256_file(directory/"window.mkv"),tracks=sha256_file(directory/"tracks.jsonl"),
                options=self.protocol["options"]),["actions.jsonl","actions.summary.json"])
        result=subprocess.run([sys.executable,"-m","cli.link_player","--input-dir",str(self.inputs),
            "--annotations",str(self.csv),"--output-dir",str(self.root),"--match-id","match"],
            env=env,capture_output=True,text=True,timeout=40)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertTrue(self.root.with_suffix(".tar").exists())


if __name__=="__main__":
    unittest.main()
