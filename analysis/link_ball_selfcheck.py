"""Ball-conditioned attribution and isolated second-run CPU integration checks."""

import os
import re
import subprocess
import sys
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from analysis import link_player_selfcheck as prior_tests
from cli.ball_track import Options, run
from cli.export_link_review import export_review
from contracts.schema import TrackRecord, read_jsonl, write_jsonl_line
from pipeline.media import atomic_json, sha256_file
from pipeline.player_link.ball_evidence import rank_with_ball
from pipeline.player_link.events import LinkOptions
from pipeline.player_link.export import export_run
from workflows import link_player as flow


class AttributionTests(unittest.TestCase):
    def setUp(self):
        self.event = {
            "event_id": "EV-ball",
            "video_id": "37",
            "goal_time_s": 6.0,
            "window_start_s": 2.0,
            "window_start_frame": 40,
            "window_end_frame_exclusive": 140,
            "partial_pre_context": False,
        }
        self.options = LinkOptions(lookback_s=3, max_lookback_s=4, post_roll_s=1)
        self.tracks = [
            TrackRecord("EV-ball", f, f / 20, tid, box, 0.9, 0)
            for f in range(100)
            for tid, box in [(0, (20, 80, 100, 280)), (1, (430, 180, 490, 300))]
        ]
        self.actions = [
            {
                "video_id": "EV-ball",
                "frame_idx": f,
                "timestamp_s": f / 20,
                "track_id": tid,
                "proposal_frame_offset": 0,
                "action_candidates": [
                    {"label": "basketball_2-point_shot", "score": score}
                ],
            }
            for f in (40, 42, 44, 46, 48, 50)
            for tid, score in [(0, 0.25), (1, 0.95)]
        ]
        self.ball = []
        for source_frame in range(60, 126):
            t = source_frame / 20
            ratio = max(0, min(1, (t - 4.2) / 1.4))
            x, y = 60 + 360 * ratio, 100 - 60 * ratio
            self.ball.append(
                {
                    "source_frame_idx": source_frame,
                    "source_timestamp_s": t,
                    "ball_track": {
                        "ball_track_id": 0,
                        "center_xy": [x, y],
                        "observed": True,
                        "confirmed": True,
                    },
                    "rims": [{"bbox_xyxy": [400, 30, 440, 50]}],
                }
            )

    def rank(self, **kwargs):
        values = {
            "event": self.event,
            "source": {"fps": 20},
            "tracks": self.tracks,
            "actions": self.actions,
            "ball_rows": self.ball,
            "options": self.options,
        }
        values.update(kwargs)
        return rank_with_ball(**values)

    def test_ball_finds_actor_zero_despite_weak_action_and_gt_does_not_leak(self):
        result = self.rank()
        self.assertEqual(result["status"], "candidate")
        self.assertEqual(result["shooter_track_id"], 0)
        self.assertTrue(
            result["candidates"][0]["ball_evidence"]["separation_frames"] >= 2
        )
        self.assertFalse(result["ball_verified"])
        changed = dict(
            self.event,
            evaluation_labels={"jersey_number": "999", "person_gt_id": "wrong"},
        )
        self.assertEqual(result, self.rank(event=changed))
        self.assertEqual(
            result["trajectory"]["source_start_frame"],
            result["trajectory"]["window_start_frame"] + 40,
        )

    def test_predicted_unconfirmed_or_unanchored_ball_cannot_select_actor(self):
        for key, value in [("observed", False), ("confirmed", False)]:
            rows = [
                dict(r, ball_track=dict(r["ball_track"], **{key: value}))
                for r in self.ball
            ]
            result = self.rank(ball_rows=rows)
            self.assertIsNone(result["shooter_track_id"])
            self.assertIn("no_supported_ball_rim_anchor", result["diagnostics"])
        rows = [dict(r, rims=[]) for r in self.ball]
        self.assertIsNone(self.rank(ball_rows=rows)["shooter_track_id"])
        with self.assertRaisesRegex(ValueError, "clock"):
            self.rank(ball_rows=self.ball + [self.ball[0]])

    def test_later_control_prevents_assist_passer_winning_on_action_score(self):
        tracks = [r for r in self.tracks if r.track_id == 0]
        tracks += [
            TrackRecord("EV-ball", f, f / 20, 2, (140, 80, 220, 280), 0.9, 0)
            for f in range(100)
        ]
        balls = []
        for row in self.ball:
            t = row["source_timestamp_s"]
            if t <= 3.4:
                x, y = 60, 100
            elif t <= 3.8:
                x, y = 60 + (t - 3.4) / 0.4 * 120, 100
            elif t <= 4.4:
                x, y = 180, 100
            else:
                ratio = min(1, (t - 4.4) / 1.2)
                x, y = 180 + 240 * ratio, 100 - 60 * ratio
            balls.append(
                dict(row, ball_track=dict(row["ball_track"], center_xy=[x, y]))
            )
        actions = [
            dict(
                a,
                action_candidates=[{"label": "basketball_2-point_shot", "score": 0.99}],
            )
            for a in self.actions
            if a["track_id"] == 0
        ]
        # A former passer's holder tag cannot revive an earlier ball control.
        balls = [
            dict(
                r,
                detections=[
                    {"label": "holder", "score": 0.99, "bbox_xyxy": [20, 80, 100, 280]}
                ],
            )
            for r in balls
        ]
        result = self.rank(tracks=tracks, actions=actions, ball_rows=balls)
        self.assertEqual(result["shooter_track_id"], 2)
        self.assertTrue(
            any(c["track_id"] == 0 for c in result["earlier_ball_controls"])
        )

    def test_dunk_does_not_require_a_flight_arc_and_overlap_can_abstain(self):
        tracks = []
        balls = []
        for f in range(80):
            for tid in (0, 1):
                tracks.append(
                    TrackRecord("EV-ball", f, f / 20, tid, (360, 20, 440, 220), 0.9, 0)
                )
            sf = f + 40
            t = sf / 20
            balls.append(
                {
                    "source_frame_idx": sf,
                    "source_timestamp_s": t,
                    "ball_track": {
                        "ball_track_id": 0,
                        "center_xy": [405, 40],
                        "observed": True,
                        "confirmed": True,
                    },
                    "rims": [{"bbox_xyxy": [400, 30, 440, 50]}],
                }
            )
        # Equal overlapping people remain ambiguous even with an observed ball.
        result = self.rank(tracks=tracks, actions=[], ball_rows=balls)
        self.assertEqual(result["status"], "needs_review")
        self.assertIsNone(result["shooter_track_id"])
        one = self.rank(
            tracks=[r for r in tracks if r.track_id == 0], actions=[], ball_rows=balls
        )
        self.assertEqual(one["shooter_track_id"], 0)
        self.assertEqual(
            one["candidates"][0]["ball_evidence"]["mode"], "hand_to_rim_candidate"
        )

    def test_later_rim_fragment_does_not_discard_earlier_shooter_path(self):
        # A post-score fragment is nearer the rounded goal timestamp, but has no actor.
        rows = [
            dict(r, ball_track=dict(r["ball_track"], ball_track_id=99))
            if r["source_timestamp_s"] >= 5.9
            else r
            for r in self.ball
        ]
        result = self.rank(ball_rows=rows)
        self.assertEqual(result["shooter_track_id"], 0)
        self.assertEqual(
            {a["ball_track_id"] for a in result["ball_path_audits"]}, {0, 99}
        )

    def test_holder_disambiguates_but_scorer_alone_and_duplicate_boxes_do_not(self):
        tracks = [
            TrackRecord("EV-ball", f, f / 20, tid, box, 0.9, 0)
            for f in range(80)
            for tid, box in [(0, (360, 20, 440, 220)), (1, (350, 10, 450, 230))]
        ]
        balls = [
            {
                "source_frame_idx": f + 40,
                "source_timestamp_s": (f + 40) / 20,
                "ball_track": {
                    "ball_track_id": 0,
                    "center_xy": [405, 40],
                    "observed": True,
                    "confirmed": True,
                },
                "rims": [{"bbox_xyxy": [400, 30, 440, 50]}],
                "detections": [],
            }
            for f in range(80)
        ]
        for row in balls:
            row["detections"] = [
                {"label": "holder", "score": 0.8, "bbox_xyxy": [360, 20, 440, 220]},
                {"label": "scorer", "score": 0.99, "bbox_xyxy": [350, 10, 450, 230]},
            ]
        result = self.rank(tracks=tracks, actions=[], ball_rows=balls)
        self.assertEqual(result["shooter_track_id"], 0)
        self.assertEqual(
            result["selection_basis"], "exclusive_stable_holder_on_supported_ball_path"
        )
        conflicting = [
            {
                "video_id": "EV-ball",
                "frame_idx": f,
                "timestamp_s": f / 20,
                "track_id": 1,
                "proposal_frame_offset": 0,
                "action_candidates": [
                    {"label": "basketball_2-point_shot", "score": 1.0}
                ],
            }
            for f in (74, 75)
        ]
        conflict = self.rank(tracks=tracks, actions=conflicting, ball_rows=balls)
        self.assertIsNone(conflict["shooter_track_id"])
        self.assertIn(
            "conflicting_stable_holder_and_ball_action_evidence",
            conflict["diagnostics"],
        )
        scorer_only = [
            dict(r, detections=[d for d in r["detections"] if d["label"] == "scorer"])
            for r in balls
        ]
        self.assertIsNone(
            self.rank(tracks=tracks, actions=[], ball_rows=scorer_only)[
                "shooter_track_id"
            ]
        )
        duplicate = [dict(r, detections=[]) for r in balls]
        duplicate[50]["detections"] = balls[50]["detections"] * 20
        self.assertIsNone(
            self.rank(tracks=tracks, actions=[], ball_rows=duplicate)[
                "shooter_track_id"
            ]
        )
        no_rim = [dict(r, rims=[]) for r in balls]
        self.assertIsNone(
            self.rank(tracks=tracks, actions=[], ball_rows=no_rim)["shooter_track_id"]
        )
        same_boxes = [
            TrackRecord(
                "EV-ball",
                r.frame_idx,
                r.timestamp_s,
                r.track_id,
                (360, 20, 440, 220),
                0.9,
                0,
            )
            for r in tracks
        ]
        self.assertIsNone(
            self.rank(tracks=same_boxes, actions=[], ball_rows=balls)[
                "shooter_track_id"
            ]
        )


class Integration(unittest.TestCase):
    def test_full_second_run_reuses_both_models_and_ball_cache(self):
        fixture = prior_tests.IntegrationTests(
            methodName="test_full_cpu_link_export_pack_review_and_portability"
        )
        fixture.setUp()
        try:
            base = fixture.base
            weights = base / "weights.pth"
            weights.write_bytes(b"fake CPU checkpoint")
            config = base / "action.py"
            config.write_text("model = {}\n")
            cache = base / "full-tracks"
            d = cache / "track/3"
            d.mkdir(parents=True)
            source = fixture.inputs / "3.mp4"
            atomic_json(
                cache / "manifest.json",
                {"videos": [{"video_id": "3", "source_sha256": sha256_file(source)}]},
            )
            with (d / "tracks.jsonl").open("w") as stream:
                for f in range(60):
                    for tid, box in [(0, (10, 5, 40, 85)), (1, (45, 5, 75, 85))]:
                        write_jsonl_line(
                            stream,
                            TrackRecord("3", f, f / 10, tid, box, 0.9, 0).to_dict(),
                        )
            atomic_json(
                d / "video_meta.json",
                {
                    "video_id": "3",
                    "fps": 10,
                    "width": 96,
                    "height": 96,
                    "processed_frames": 60,
                    "frame_count": 60,
                },
            )
            atomic_json(
                d / "tracking_complete.json",
                {
                    "protocol": {
                        "checkpoint_sha256": sha256_file(weights),
                        "config_sha256": sha256_file(
                            Path(__file__).resolve().parents[1]
                            / "MOTIP/configs/r50_deformable_detr_motip_sportsmot.yaml"
                        ),
                        "tracker_options": "cli.tracking defaults v1",
                    },
                    "outputs": {
                        n: sha256_file(d / n)
                        for n in ("tracks.jsonl", "video_meta.json")
                    },
                },
            )

            class Detector:
                class_names = ("ball", "rim")

                def predict(self, image, **kwargs):
                    return SimpleNamespace(
                        xyxy=np.array([[20, 20, 24, 24], [18, 15, 30, 25]]),
                        confidence=np.array([0.8, 0.9]),
                        class_id=np.array([0, 1]),
                    )

            ball_cache = base / "ball-cache"
            run(source, weights, ball_cache, Options(), model=Detector())
            before = {
                str(p): sha256_file(p) for p in fixture.root.rglob("*") if p.is_file()
            }
            # Populate immutable first-round automatic decisions for comparison.
            flow.link(fixture.root, fixture.options)
            export_run(fixture.root, fixture.options, fixture.settings)
            before = {
                str(p): sha256_file(p) for p in fixture.root.rglob("*") if p.is_file()
            }
            env = dict(
                os.environ,
                BASKET_MOTIP_CHECKPOINT=str(weights),
                BASKET_ACTION_CHECKPOINT=str(weights),
                BASKET_ACTION_CONFIG=str(config),
            )
            destination = base / "second"
            command = [
                sys.executable,
                "-m",
                "cli.link_player",
                "--input-dir",
                str(fixture.inputs),
                "--annotations",
                str(fixture.csv),
                "--output-dir",
                str(destination),
                "--match-id",
                "match",
                "--tracking-root",
                str(cache),
                "--reuse-run",
                str(fixture.root),
                "--ball-checkpoint",
                str(weights),
                "--ball-python",
                sys.executable,
                "--ball-cache",
                f"3={ball_cache}",
                "--max-lookback-s",
                "3",
            ]
            first = subprocess.run(
                command,
                env=env,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
            self.assertTrue(destination.with_suffix(".tar").is_file())
            self.assertTrue((destination / "comparison.csv").is_file())
            self.assertEqual(
                flow.status(destination)["status"], "completed_not_accuracy_evaluated"
            )
            protocol = flow.saved(destination)
            for event in protocol["events"]:
                directory = flow.folder(destination, event)
                self.assertTrue(flow.status(destination)["events"][0]["stages"]["ball"])
                old_event = next(
                    e
                    for e in fixture.protocol["events"]
                    if e["event_id"] == event["event_id"]
                )
                old_rows = list(
                    read_jsonl(flow.folder(fixture.root, old_event) / "actions.jsonl")
                )
                for a in read_jsonl(directory / "actions.jsonl"):
                    self.assertTrue(
                        any(
                            a["frame_idx"] + event["window_start_frame"]
                            == b["frame_idx"] + old_event["window_start_frame"]
                            and a["track_id"] == b["track_id"]
                            for b in old_rows
                        )
                    )
                self.assertIn(
                    "ball_contact_rim_action",
                    flow.read(directory / "link.json")["attribution_method"],
                )
                handoff = flow.read(directory / "handoff/manifest.json")
                if handoff["shooter_track_id"] is not None:
                    self.assertEqual(
                        handoff["sample_selection"],
                        "sharpness_size_detection_per_time_bin",
                    )
                    self.assertTrue(
                        all("quality_score" in s for s in handoff["samples"])
                    )
            second = subprocess.run(
                command,
                env=env,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
            snapshot = {
                str(p): sha256_file(p) for p in destination.rglob("*") if p.is_file()
            }
            review_archive = base / "first-review.zip"
            export_review(fixture.root, review_archive)
            with zipfile.ZipFile(review_archive) as archive:
                names = set(archive.namelist())
                prefix = fixture.root.name + "-review/"
                page = archive.read(prefix + "review.html").decode()
                self.assertIn("· T0", page)
                for link in re.findall(r'(?:src|href)="([^"]+)"', page):
                    self.assertIn(prefix + link, names)
                self.assertFalse(any(n.endswith("window.mkv") for n in names))
                self.assertTrue(any(n.endswith("trajectory.mp4") for n in names))
                selected_count = sum(
                    len(
                        flow.read(
                            flow.folder(fixture.root, e) / "handoff/manifest.json"
                        )["samples"]
                    )
                    for e in fixture.protocol["events"]
                )
                self.assertEqual(sum(n.endswith(".png") for n in names), selected_count)
                self.assertIn(prefix + "review_checklist.csv", names)
            self.assertEqual(
                snapshot,
                {str(p): sha256_file(p) for p in destination.rglob("*") if p.is_file()},
            )
            with self.assertRaises(FileExistsError):
                export_review(fixture.root, review_archive)
            third_command = command.copy()
            third_command[third_command.index("--output-dir") + 1] = str(base / "third")
            third_command[third_command.index("--reuse-run") + 1] = str(destination)
            i = third_command.index("--ball-cache")
            third_command[i : i + 2] = ["--ball-cache-run", str(destination)]
            third = subprocess.run(
                third_command,
                env=env,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            self.assertEqual(third.returncode, 0, third.stdout + third.stderr)
            self.assertIn("reused verified artifacts", third.stdout)
            self.assertIn("reused verified window", third.stdout)
            self.assertEqual(
                snapshot,
                {str(p): sha256_file(p) for p in destination.rglob("*") if p.is_file()},
            )
            self.assertEqual(
                before,
                {
                    str(p): sha256_file(p)
                    for p in fixture.root.rglob("*")
                    if p.is_file()
                },
            )
        finally:
            fixture.tearDown()


if __name__ == "__main__":
    unittest.main()
