"""Synthetic tests, not a claim of model accuracy or a GPU weight test."""

from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from analysis.person_library_selfcheck import make_library_fixture, rows
from contracts.execution import ExecutionSettings
from contracts.schema import read_jsonl, write_json
from pipeline.identity.cross_clip import group_nodes
from pipeline.identity.jnr_backend import normalize_state_dict
from pipeline.identity.jersey_jnr import jnr_consensus, prediction_reading, recognize_jerseys_jnr
from workflows.commands import CommandBuilder
from workflows.person_library import PersonLibraryOptions, build_person_library, library_is_current


def reading(number="15", frame=0, time=0, score=.95, uncertainty=.02, margin=.9, reasons=None):
    return {"raw_track_id": 0, "frame_idx": frame, "timestamp_s": time, "text": number,
            "raw_score": score, "uncertainty": uncertainty, "margin": margin,
            "crop_path": "media/cover.png", "rejection_reasons": reasons or []}


class FakeJNR:
    provenance = {"backend": "uncertainty_jnr", "checkpoint_sha256": "synthetic"}

    def predict(self, images):
        values = np.zeros((len(images), 100), dtype=np.float32)
        values[:, 15], values[:, 8] = .95, .05
        return values, np.full(len(images), .02)


class JNRChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="basket-jnr-check-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_temporal_support_not_adjacent_frame_vote(self):
        close = [reading(frame=0, time=0), reading(frame=1, time=.04)]
        self.assertIsNone(jnr_consensus(close)["number"])
        result = jnr_consensus(close + [reading(frame=9, time=.36)])
        self.assertEqual(result["number"], "15")
        self.assertEqual(result["candidates"][0]["support_frames"], 2)

    def test_high_uncertainty_and_small_margin_abstain(self):
        for value in (reading(uncertainty=.4), reading(margin=.01), reading(score=.3)):
            self.assertIsNone(jnr_consensus([value, {**value, "frame_idx": 10, "timestamp_s": .4}])["number"])

    def test_minority_conflict_not_majority_voted_away(self):
        result = jnr_consensus([reading(frame=i, time=i) for i in range(5)] + [reading("32", 20, 6)])
        self.assertEqual(result["status"], "conflict")
        self.assertIsNone(result["number"])

    def test_zero_and_double_zero_not_claimed(self):
        sample = {"track_id": 0, "frame_idx": 0, "timestamp_s": 0, "crop_path": "x.jpg"}
        probs = np.zeros(100); probs[0] = 1
        result = prediction_reading(sample, probs, .001, min_score=.8, max_uncertainty=.2, min_margin=.2)
        self.assertIn("ambiguous_0_or_00", result["rejection_reasons"])
        self.assertIsNone(jnr_consensus([result, {**result, "frame_idx": 10, "timestamp_s": 1}])["number"])

    def test_nonfinite_and_bad_shapes_fail(self):
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            jnr_consensus([reading(uncertainty=float("nan"))])
        for probs in (np.zeros(99), np.full(100, float("nan")), np.full(100, -1)):
            with self.assertRaises(ValueError):
                prediction_reading({}, probs, .1, min_score=.8, max_uncertainty=.2, min_margin=.2)
        with self.assertRaises(ValueError):
            jnr_consensus([], min_support=1)

    def test_checkpoint_compile_wrappers_and_collision(self):
        self.assertEqual(normalize_state_dict({"_orig_mod.backbone._orig_mod.weight": 1}), {"backbone.weight": 1})
        with self.assertRaises(ValueError):
            normalize_state_dict({"_orig_mod.weight": 1, "weight": 2})

    def run_recognition(self, samples, ids=(0,), **kwargs):
        import cv2
        archive = self.root / "raw"
        archive.mkdir(exist_ok=True)
        for sample in samples:
            path = archive / sample["crop_path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(path), np.zeros((100, 50, 3), dtype=np.uint8))
        rows(archive / "kpr_samples.jsonl", samples)
        mapping = self.root / "mapping.jsonl"
        rows(mapping, [{"raw_track_id": tid, "person_id": f"P{tid}"} for tid in ids])
        return recognize_jerseys_jnr(archive, mapping, self.root / "number-output",
                                    self.root, self.root / "cfg", self.root / "weights",
                                    backend=FakeJNR(), **kwargs)

    def test_cached_crop_prediction_and_full_scores_preserved(self):
        samples = [{"track_id": 0, "frame_idx": i * 10, "timestamp_s": i * .4,
                    "crop_path": f"media/{i}.png"} for i in range(3)]
        summary = self.run_recognition(samples, batch_size=2)
        self.assertEqual(summary["candidate_number_tracks"], 1)
        self.assertEqual(next(read_jsonl(self.root / "number-output/jersey_tracks.jsonl"))["number"], "15")
        with np.load(self.root / "number-output/jersey_predictions.npz", allow_pickle=False) as data:
            self.assertEqual(data["probabilities"].shape, (3, 100))
        self.assertEqual(len(list(read_jsonl(self.root / "number-output/jersey_readings.jsonl"))), 3)
        with self.assertRaises(FileExistsError):
            self.run_recognition(samples)

    def test_empty_input_writes_explicit_empty_summary(self):
        summary = self.run_recognition([], ids=())
        self.assertEqual(summary["sample_count"], 0)
        self.assertEqual(summary["candidate_number_tracks"], 0)
        self.assertFalse("accuracy" in summary)

    def test_duplicate_sample_cannot_inflate_support(self):
        sample = {"track_id": 0, "frame_idx": 0, "timestamp_s": 0, "crop_path": "x.png"}
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.run_recognition([sample, sample])

    def jersey_fixture(self, numbers=("15", "32", "15")):
        run = make_library_fixture(self.root)
        for i, number in enumerate(numbers):
            output = run / f"clip{i + 1}/identity"
            result = jnr_consensus([reading(number, 0, 0), reading(number, 10, .4)])
            rows(output / "jersey_tracks.jsonl", [{"raw_track_id": 0, **result}])
            write_json(output / "jersey_summary.json", {"backend": "uncertainty_jnr",
                "provenance": FakeJNR.provenance, "thresholds": {"min_score": .8}})
        return run

    def test_different_numbers_veto_and_signature_updates(self):
        run = self.jersey_fixture()
        result = build_person_library(PersonLibraryOptions(run, max_distance=.2, use_jersey_evidence=True))
        self.assertEqual(result["global_person_count"], 3)
        self.assertGreater(result["jersey_blocked_pairs"], 0)
        manifest = json.loads((run / "library/library_manifest.json").read_text())
        self.assertTrue(library_is_current(run, manifest))
        (run / "clip1/identity/jersey_tracks.jsonl").write_text("")
        self.assertFalse(library_is_current(run, manifest))

    def test_same_number_never_overrides_kpr_distance(self):
        run = self.jersey_fixture(("15", "15", "15"))
        result = build_person_library(PersonLibraryOptions(run, max_distance=.2, use_jersey_evidence=True))
        self.assertEqual(result["global_person_count"], 2)
        self.assertEqual(result["candidate_number_archives"], 3)

    def test_cluster_unknown_cannot_bridge_different_numbers(self):
        nodes = [{"node_id": str(i), "clip_name": str(i), "feature_indices": [i],
                  "jersey": {"status": "candidate_consensus", "number": n}}
                 for i, n in enumerate(("15", None, "32"))]
        groups, pairs, _ = group_nodes(nodes, np.zeros((3, 3)), max_distance=.2)
        self.assertEqual(len(groups), 2)
        self.assertTrue(any("different_reliable_jersey_numbers" in p["auto_block_reasons"] for p in pairs))

    def test_incompatible_and_missing_jersey_rejected(self):
        run = self.jersey_fixture()
        write_json(run / "clip2/identity/jersey_summary.json", {"backend": "uncertainty_jnr",
            "provenance": {"checkpoint_sha256": "different"}, "thresholds": {"min_score": .8}})
        with self.assertRaisesRegex(ValueError, "Incompatible JNR"):
            build_person_library(PersonLibraryOptions(run, max_distance=.2, use_jersey_evidence=True))
        (run / "clip1/identity/jersey_summary.json").unlink()
        with self.assertRaises(FileNotFoundError):
            build_person_library(PersonLibraryOptions(run, use_jersey_evidence=True))

    def test_conflicting_local_archive_is_held(self):
        run = self.jersey_fixture()
        conflict = jnr_consensus([reading("15", 0, 0), reading("32", 10, .4)])
        rows(run / "clip1/identity/jersey_tracks.jsonl", [{"raw_track_id": 0, **conflict}])
        result = build_person_library(PersonLibraryOptions(run, max_distance=.2, use_jersey_evidence=True))
        self.assertEqual(result["held_archive_count"], 1)
        people = list(read_jsonl(run / "library/people.jsonl"))
        self.assertTrue(any("jersey_number_conflict" in m["hold_reasons"] for p in people for m in p["members"]))

    def test_web_search_evidence_links_and_review_preserve_jersey_constraints(self):
        from web.backend.settings import WebSettings
        from web.backend.store import ProjectStore
        from web.backend.imports import import_results
        from web.backend.person_library import library_catalog, library_person, edit_library
        run = self.jersey_fixture(("15", "15", "15"))
        build_person_library(PersonLibraryOptions(run, max_distance=.2, use_jersey_evidence=True))
        settings = WebSettings(data_root=self.root / "data/web", source_root=self.root / "data",
                               import_root=self.root / "outputs", output_root=self.root / "outputs/web")
        settings.ensure_directories()
        store = ProjectStore(settings.data_root, settings.output_root)
        imported = import_results(settings, store, run)
        pid = imported["project_id"]
        catalog = library_catalog(settings, store, pid, q="15")
        self.assertEqual(catalog["total"], 2)
        person = library_person(settings, store, pid, catalog["items"][0]["global_person_id"])
        evidence = person["members"][0]["jersey"]["tracks"][0]["readings"][0]
        self.assertIn("/artifacts/identity_raw/", evidence["crop_url"])
        self.assertTrue(evidence["crop_url"].endswith("media/cover.png"))
        result = edit_library(settings, store, pid, "detach", node_id=person["members"][0]["node_id"])
        self.assertTrue(result["summary"]["settings"]["use_jersey_evidence"])
        self.assertEqual(library_catalog(settings, store, pid)["total"], 3)

    def test_cli_enables_number_veto_after_completion(self):
        from cli import process_batch
        run = self.jersey_fixture()
        arguments = ["process_batch", "--input-dir", str(self.root / "data"), "--output-dir", str(run),
                     "--cross-clip-distance", ".2", "--jersey-jnr"]
        import contextlib
        import io
        with patch.object(sys, "argv", arguments), patch.object(process_batch, "run_batch", return_value={"failed_clips": []}), contextlib.redirect_stdout(io.StringIO()):
            process_batch.main()
        summary = json.loads((run / "library/library_manifest.json").read_text())
        self.assertTrue(summary["jersey_constraints"])
        self.assertEqual(summary["global_person_count"], 3)

    def test_command_builder_routes_to_independent_jnr_environment(self):
        source = self.root / "src/uncertainty_jnr/model.py"
        source.parent.mkdir(parents=True); source.write_text("synthetic")
        source.with_name("augmentation.py").write_text("synthetic")
        config, weights = self.root / "cfg.yaml", self.root / "model.pt"
        config.write_text("synthetic"); weights.write_text("synthetic")
        settings = replace(ExecutionSettings(), jnr_python=Path(sys.executable), jnr_root=self.root,
                           jnr_config=config, jnr_checkpoint=weights)
        paths = CommandBuilder.paths({"output_dir": str(self.root / "output"), "source_path": "video.mp4"})
        command = CommandBuilder(settings).command("jersey", paths, {"jersey_jnr": True})
        self.assertIn("cli.jersey_jnr", command)
        self.assertNotIn("--trust-checkpoint", command)


def main():
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(JNRChecks))
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
