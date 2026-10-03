"""Tracked CPU regression checks (tests/ intentionally remains gitignored)."""

from __future__ import annotations

import itertools
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from analysis.evaluation.batch import BatchEventEvaluationOptions, evaluate_event_batch
from analysis.evaluation.events import (
    EvaluationEvent,
    EventEvaluationOptions,
    evaluate_events,
    match_events,
)
from contracts.schema import EventRecord


def event(
    key="e",
    actor="P1",
    label="basketball_pass",
    start=0.0,
    end=1.0,
    video="v",
    ids=(1,),
):
    return EventRecord(
        video,
        key,
        actor,
        label,
        start,
        end,
        0.9,
        round(start * 10),
        round(end * 10) - 1,
        tuple(ids),
    ).to_dict()


def track(frame, identity=1, box=(0, 0, 10, 10), video="v"):
    return {
        "video_id": video,
        "frame_idx": frame,
        "timestamp_s": frame / 10,
        "track_id": identity,
        "bbox_xyxy": list(box),
        "det_score": 0.9,
        "category_id": 0,
    }


def tube_event(**kwargs):
    value = event(**kwargs)
    value["actor_tube"] = [
        [i, 0, 0, 10, 10] for i in range(value["start_frame"], value["end_frame"] + 1)
    ]
    return value


class _EvaluationFixtures(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="basket-evaluation-")
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)

    def jsonl(self, name, rows):
        path = self.root / name
        path.write_text("".join(json.dumps(v) + "\n" for v in rows), encoding="utf-8")
        return path

    def json(self, name, value):
        path = self.root / name
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def metadata(self, **changes):
        return self.json(
            "video_meta.json",
            {
                "video_id": "v",
                "fps": 10.0,
                "frame_count": 30,
                "processed_frames": 30,
                **changes,
            },
        )

    def options(self, refs, preds, **kwargs):
        return EventEvaluationOptions(
            reference=self.jsonl("ref.jsonl", refs),
            prediction=self.jsonl("pred.jsonl", preds),
            output=self.root / "metrics.json",
            actor_mode="exact",
            **kwargs,
        )

    def run_events(self, refs, preds, **kwargs):
        return evaluate_events(self.options(refs, preds, **kwargs))


class EventEvaluationSelfCheck(_EvaluationFixtures):
    def test_perfect(self):
        self.assertEqual(self.run_events([event()], [event()])["headline"]["value"], 1)

    def test_all_missed(self):
        value = self.run_events([event()], [])["end_to_end"]
        self.assertEqual(value["false_negative"], 1)
        self.assertEqual(value["f1"], 0)
        self.assertIsNone(value["precision"])

    def test_duplicate(self):
        value = self.run_events([event()], [event("a"), event("b")])
        self.assertAlmostEqual(value["end_to_end"]["f1"], 2 / 3)
        self.assertEqual(value["diagnostics"]["error_counts"]["duplicate"], 1)

    def test_wrong_class(self):
        value = self.run_events([event()], [event(label="basketball_dribble")])
        self.assertEqual(value["headline"]["value"], 0)
        self.assertEqual(value["diagnostics"]["error_counts"]["wrong_class"], 1)

    def test_wrong_interval(self):
        value = self.run_events([event()], [event(start=0.8, end=1.8)])
        self.assertEqual(value["headline"]["value"], 0)
        self.assertEqual(value["diagnostics"]["error_counts"]["wrong_interval"], 1)

    def test_wrong_actor(self):
        value = self.run_events([event()], [event(actor="P2")])
        self.assertEqual(value["headline"]["value"], 0)
        self.assertEqual(value["event_and_interval_only"]["f1"], 1)

    def test_negative_video(self):
        value = self.run_events([], [event()])["end_to_end"]
        self.assertEqual(value["false_positive"], 1)
        self.assertEqual(value["f1"], 0)
        self.assertIsNone(value["recall"])

    def test_both_empty_are_not_perfect_f1(self):
        self.assertIsNone(self.run_events([], [])["headline"]["value"])

    def test_empty_tracks_are_misses(self):
        options = replace(
            self.options([tube_event()], []),
            actor_mode="tube",
            tracks=self.jsonl("tracks.jsonl", []),
        )
        self.assertEqual(evaluate_events(options)["end_to_end"]["false_negative"], 1)

    def test_correct_tube(self):
        options = replace(
            self.options([tube_event()], [event()]),
            actor_mode="tube",
            tracks=self.jsonl("tracks.jsonl", [track(i) for i in range(10)]),
        )
        self.assertEqual(evaluate_events(options)["headline"]["value"], 1)

    def test_wrong_tube(self):
        options = replace(
            self.options([tube_event()], [event()]),
            actor_mode="tube",
            tracks=self.jsonl(
                "tracks.jsonl", [track(i, box=(30, 30, 40, 40)) for i in range(10)]
            ),
        )
        self.assertEqual(evaluate_events(options)["headline"]["value"], 0)

    def test_low_actor_coverage(self):
        options = replace(
            self.options([tube_event()], [event()]),
            actor_mode="tube",
            tracks=self.jsonl("tracks.jsonl", [track(i) for i in range(7)]),
        )
        value = evaluate_events(options)
        self.assertEqual(value["headline"]["value"], 0)

    def test_eligible_track_not_hidden_by_an_ineligible_track(self):
        # Track 1 has greater correct coverage but too low a mean IoU;
        # track 2 satisfies BOTH criteria and must be selected.
        boxes = [track(i, 1, (0, 0, 20, 10)) for i in range(9)]
        boxes += [track(i, 2) for i in range(8)]
        options = replace(
            self.options([tube_event()], [event(ids=(1, 2))]),
            actor_mode="tube",
            tracks=self.jsonl("tracks.jsonl", boxes),
        )
        self.assertEqual(evaluate_events(options)["headline"]["value"], 1)

    def test_multiple_ids_cannot_hide_switches(self):
        boxes = []
        for i in range(10):
            boxes.extend(
                [
                    track(i, 1, (0, 0, 10, 10) if i < 5 else (30, 30, 40, 40)),
                    track(i, 2, (30, 30, 40, 40) if i < 5 else (0, 0, 10, 10)),
                ]
            )
        options = replace(
            self.options([tube_event()], [event(ids=(1, 2))]),
            actor_mode="tube",
            tracks=self.jsonl("tracks.jsonl", boxes),
        )
        self.assertEqual(evaluate_events(options)["headline"]["value"], 0)

    def test_wrong_video_rejected(self):
        with self.assertRaises(ValueError):
            self.run_events([event(video="gt")], [event(video="wrong")])

    def test_explicit_video_mapping(self):
        result = self.run_events(
            [event(video="official")],
            [event()],
            reference_video_id="official",
            prediction_video_id="v",
            video_meta=self.metadata(),
        )
        self.assertEqual(result["headline"]["value"], 1)

    def test_incorrect_mapping_rejected(self):
        with self.assertRaises(ValueError):
            self.run_events(
                [event(video="official")],
                [event()],
                reference_video_id="typo",
                prediction_video_id="v",
            )

    def test_half_mapping_rejected(self):
        with self.assertRaises(ValueError):
            self.run_events([event()], [event()], reference_video_id="v")

    def test_wrong_track_video_rejected(self):
        options = replace(
            self.options([tube_event()], [event()]),
            actor_mode="tube",
            tracks=self.jsonl("tracks.jsonl", [track(0, video="wrong")]),
        )
        with self.assertRaises(ValueError):
            evaluate_events(options)

    def test_metadata_validates_fps(self):
        with self.assertRaises(ValueError):
            self.run_events([event()], [event()], video_meta=self.metadata(fps=25))

    def test_partial_video_rejected(self):
        with self.assertRaises(ValueError):
            self.run_events(
                [event()], [event()], video_meta=self.metadata(processed_frames=10)
            )

    def test_official_reference_video_properties_match(self):
        gt = event()
        gt["reference_video_meta"] = {
            "fps": 10,
            "frame_count": 30,
            "height": 720,
            "width": 1280,
        }
        value = self.run_events(
            [gt], [event()], video_meta=self.metadata(height=720, width=1280)
        )
        self.assertEqual(value["headline"]["value"], 1)

    def test_resized_source_rejected(self):
        gt = event()
        gt["reference_video_meta"] = {
            "fps": 10,
            "frame_count": 30,
            "height": 720,
            "width": 1280,
        }
        with self.assertRaises(ValueError):
            self.run_events(
                [gt], [event()], video_meta=self.metadata(height=360, width=640)
            )

    def test_gt_frame_count_mismatch_even_without_tail_event(self):
        gt = event()
        gt["reference_video_meta"] = {
            "fps": 10,
            "frame_count": 40,
            "height": 720,
            "width": 1280,
        }
        with self.assertRaises(ValueError):
            self.run_events(
                [gt], [event()], video_meta=self.metadata(height=720, width=1280)
            )

    def test_event_exceeds_video_rejected(self):
        with self.assertRaises(ValueError):
            self.run_events([event(end=4)], [event(end=4)], video_meta=self.metadata())

    def test_empty_prediction_metadata_wrong_video_rejected(self):
        with self.assertRaises(ValueError):
            self.run_events([event(video="different")], [], video_meta=self.metadata())

    def test_duplicate_gt_id_rejected(self):
        with self.assertRaises(ValueError):
            self.run_events([event(), event()], [event()])

    def test_ignored_classes_filtered_symmetrically(self):
        value = self.run_events(
            [event()],
            [event(), event("save", label="basketball_save")],
            profile="multisports",
        )
        self.assertEqual(value["headline"]["value"], 1)
        self.assertEqual(value["diagnostics"]["ignored_prediction_count"], 1)

    def test_custom_class_scope(self):
        value = self.run_events(
            [event()], [event()], exclude_labels=("basketball pass",)
        )
        self.assertIsNone(value["headline"]["value"])

    def test_score_filter(self):
        value = self.run_events([event()], [event()], min_raw_score=0.95)
        self.assertEqual(value["end_to_end"]["false_negative"], 1)
        self.assertEqual(value["diagnostics"]["score_filtered_prediction_count"], 1)

    def test_no_actor_mode_is_not_claimed_usable(self):
        options = replace(
            self.options([event()], [event(actor="wrong")]), actor_mode="ignore"
        )
        self.assertEqual(
            evaluate_events(options)["headline"]["name"], "event_interval_f1"
        )

    def test_boundary_error_diagnostic(self):
        value = self.run_events([event()], [event(end=1.2)])
        self.assertAlmostEqual(
            value["diagnostics"]["matched_boundary_errors_s"]["median_end"], 0.2
        )

    def test_unknown_classes_count_as_false_positives(self):
        value = self.run_events(
            [event()], [event(), event("unknown", label="not_supported")]
        )
        self.assertEqual(value["end_to_end"]["false_positive"], 1)

    def test_output_cannot_overwrite_input(self):
        options = self.options([event()], [event()])
        with self.assertRaises(ValueError):
            evaluate_events(replace(options, output=options.reference))

    def test_missing_file_is_not_a_negative_video(self):
        options = self.options([event()], [])
        with self.assertRaises(FileNotFoundError):
            evaluate_events(replace(options, prediction=self.root / "missing.jsonl"))

    def test_maximum_valid_pair_count(self):
        value = self.run_events(
            [event("g1", start=4, end=14), event("g2", start=8, end=18)],
            [event("p1", start=4, end=15), event("p2", start=0, end=11)],
        )
        self.assertEqual(value["end_to_end"]["true_positive"], 2)

    def test_matching_against_exhaustive_small_graph_oracle(self):
        refs = [
            EvaluationEvent(EventRecord.from_dict(event(f"g{i}"))) for i in range(3)
        ]
        preds = [
            EvaluationEvent(EventRecord.from_dict(event(f"p{i}"))) for i in range(3)
        ]
        for bits in range(512):
            allowed = {
                (pi, ri)
                for pi in range(3)
                for ri in range(3)
                if bits & (1 << (pi * 3 + ri))
            }
            expected = 0
            for size in range(1, 4):
                for left in itertools.combinations(range(3), size):
                    for right in itertools.permutations(range(3), size):
                        if all(pair in allowed for pair in zip(left, right)):
                            expected = max(expected, size)
            actual = match_events(
                preds,
                refs,
                temporal_threshold=0.5,
                actor_gate=lambda p, r, allowed=allowed: (p, r) in allowed,
            )
            self.assertEqual(len(actual), expected, f"graph={bits}")

    def batch(self, entries):
        manifest = self.json("manifest.json", {"videos": entries})
        return evaluate_event_batch(
            BatchEventEvaluationOptions(
                manifest, self.root / "batch", actor_mode="exact", profile="generic"
            )
        )

    def entry(self, name, refs, preds):
        return {
            "name": name,
            "reference": str(self.jsonl(name + ".gt.jsonl", refs)),
            "prediction": str(self.jsonl(name + ".pred.jsonl", preds)),
            "video_meta": str(self.metadata()),
        }

    def test_batch_micro_f1_not_mean_of_video_f1(self):
        value = self.batch(
            [
                self.entry("one", [event()], [event()]),
                self.entry("two", [event(f"g{i}") for i in range(9)], []),
            ]
        )
        self.assertEqual(value["status"], "complete")
        self.assertAlmostEqual(value["headline"]["value"], 2 / 11)

    def test_batch_failure_invalidates_final_score(self):
        entry = self.entry("missing", [event()], [])
        entry["prediction"] = "does-not-exist.jsonl"
        value = self.batch([self.entry("ok", [event()], [event()]), entry])
        self.assertEqual(value["status"], "incomplete")
        self.assertIsNone(value["headline"]["value"])
        self.assertEqual(value["failed_videos"], 1)

    def test_batch_negative_video_penalises_false_positive(self):
        value = self.batch(
            [
                self.entry("ok", [event()], [event()]),
                self.entry("negative", [], [event()]),
            ]
        )
        self.assertAlmostEqual(value["headline"]["value"], 2 / 3)

    def test_multisports_batch_rejects_old_reference(self):
        manifest = self.json(
            "manifest.json", {"videos": [self.entry("old", [tube_event()], [event()])]}
        )
        result = evaluate_event_batch(
            BatchEventEvaluationOptions(
                manifest, self.root / "batch", actor_mode="exact"
            )
        )
        self.assertEqual(result["status"], "incomplete")

    def test_multisports_batch_accepts_aligned_tube_reference(self):
        gt = tube_event()
        gt["reference_video_meta"] = {
            "fps": 10,
            "frame_count": 30,
            "height": 720,
            "width": 1280,
        }
        entry = self.entry("new", [gt], [event()])
        self.metadata(height=720, width=1280)
        entry["tracks"] = str(self.jsonl("tracks.jsonl", [track(i) for i in range(10)]))
        manifest = self.json("manifest.json", {"videos": [entry]})
        result = evaluate_event_batch(
            BatchEventEvaluationOptions(manifest, self.root / "batch")
        )
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["headline"]["value"], 1)

    def test_batch_requires_metadata(self):
        entry = self.entry("ok", [event()], [event()])
        del entry["video_meta"]
        self.assertEqual(self.batch([entry])["failed_videos"], 1)

    def test_batch_does_not_overwrite_other_video_input(self):
        dest = self.root / "batch" / "0000" / "event_metrics.json"
        dest.parent.mkdir(parents=True)
        dest.write_text("input", encoding="utf-8")
        entry = self.entry("ok", [event()], [event()])
        entry["prediction"] = str(dest)
        with self.assertRaises(ValueError):
            self.batch([entry])
        self.assertEqual(dest.read_text(), "input")

    def test_action_aggregation_to_event_evaluation(self):
        from adapters.mmaction2.temporal_events import (
            TemporalAggregationOptions,
            aggregate_action_points,
        )

        actions = self.jsonl(
            "actions.jsonl",
            [
                {
                    "video_id": "v",
                    "track_id": 1,
                    "person_id": "P1",
                    "frame_idx": frame,
                    "timestamp_s": frame / 10,
                    "selected_actions": [{"label": "basketball_pass", "score": 0.9}],
                }
                for frame in (2, 6)
            ],
        )
        prediction = self.root / "events.jsonl"
        aggregate_action_points(
            TemporalAggregationOptions(
                actions, prediction, video_meta=self.metadata(), cadence_frames=4
            )
        )
        gt = event(end=0.8)
        options = EventEvaluationOptions(
            self.jsonl("gt.jsonl", [gt]),
            prediction,
            self.root / "integration_metrics.json",
            actor_mode="exact",
            video_meta=self.metadata(),
        )
        self.assertEqual(evaluate_events(options)["headline"]["value"], 1)


class TrackingEvaluationSelfCheck(_EvaluationFixtures):
    def test_multisports_converter_carries_alignment_metadata(self):
        import pickle

        import numpy as np

        from tools.datasets.multisports import (
            MultiSportsReferenceOptions,
            prepare_multisports_reference,
        )

        annotation = self.root / "trusted_synthetic_GT.pkl"
        with annotation.open("wb") as handle:
            pickle.dump(
                {
                    "labels": ["basketball pass"],
                    "nframes": {"v": 30},
                    "resolution": {"v": (720, 1280)},
                    "gttubes": {
                        "v": {0: [np.array([[1, 0, 0, 10, 10], [2, 0, 0, 10, 10]])]}
                    },
                },
                handle,
            )
        output = self.root / "reference.jsonl"
        prepare_multisports_reference(
            MultiSportsReferenceOptions(annotation, output, ("v",), fps=10)
        )
        row = json.loads(output.read_text())
        self.assertEqual(row["start_frame"], 0)
        self.assertEqual(row["end_frame"], 1)
        self.assertEqual(row["reference_video_meta"]["frame_count"], 30)

    def tracking(self, reference_rows, predictions, **kwargs):
        from analysis.evaluation.tracking import (
            TrackingEvaluationOptions,
            evaluate_tracks,
        )

        reference = self.root / "reference.txt"
        reference.write_text(reference_rows, encoding="utf-8")
        options = TrackingEvaluationOptions(
            reference=reference,
            prediction=self.jsonl("tracks.jsonl", predictions),
            output=self.root / "tracking.json",
            events_output=self.root / "tracking_events.csv",
            **kwargs,
        )
        return evaluate_tracks(options)

    def test_perfect_tracks(self):
        value = self.tracking(
            "1,7,0,0,10,10,1,-1,-1,-1\n2,7,0,0,10,10,1,-1,-1,-1\n", [track(0), track(1)]
        )
        self.assertEqual(value["metrics"]["idf1"], 1)
        self.assertEqual(value["metrics"]["num_switches"], 0)

    def test_empty_predictions_count_as_misses(self):
        value = self.tracking("1,7,0,0,10,10,1,-1,-1,-1\n", [])
        self.assertEqual(value["metrics"]["idf1"], 0)
        self.assertEqual(value["metrics"]["num_misses"], 1)

    def test_negative_tracking_video(self):
        value = self.tracking("", [track(0)])
        self.assertEqual(value["metrics"]["num_false_positives"], 1)

    def test_empty_tracking_range_is_not_perfect_identity(self):
        value = self.tracking("", [], max_frames=5)
        self.assertIsNone(value["headline"]["value"])
        self.assertEqual(value["metrics"]["num_frames"], 5)

    def test_identity_switch_is_penalised(self):
        value = self.tracking(
            "1,7,0,0,10,10,1,-1,-1,-1\n2,7,0,0,10,10,1,-1,-1,-1\n",
            [track(0, 1), track(1, 2)],
        )
        self.assertLess(value["metrics"]["idf1"], 1)
        self.assertEqual(value["metrics"]["num_switches"], 1)

    def test_missing_tail_is_not_hidden(self):
        gt = "".join(f"{i},7,0,0,10,10,1,-1,-1,-1\n" for i in range(1, 5))
        value = self.tracking(gt, [track(0), track(1)])
        self.assertEqual(value["metrics"]["num_misses"], 2)
        self.assertAlmostEqual(value["metrics"]["idf1"], 2 / 3)

    def test_partial_scope_is_reported(self):
        gt = "".join(f"{i},7,0,0,10,10,1,-1,-1,-1\n" for i in range(1, 5))
        value = self.tracking(gt, [track(0), track(1)], max_frames=2)
        self.assertTrue(value["evaluation"]["range_only"])
        self.assertEqual(
            value["evaluation"]["reference_observations_ignored_after_limit"], 2
        )


def run_selfcheck(verbosity=2, *, include_tracking=False) -> unittest.TestResult:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(EventEvaluationSelfCheck)
    if include_tracking:
        suite.addTests(
            unittest.defaultTestLoader.loadTestsFromTestCase(
                TrackingEvaluationSelfCheck
            )
        )
    return unittest.TextTestRunner(verbosity=verbosity).run(suite)
