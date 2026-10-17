"""CPU integration checks: real lossless media, synthetic model evidence."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from pipeline.identity.gallery_memory import (
    MemoryPolicy,
    incompatible,
    plan_memory,
)
from pipeline.identity.player_evidence import (
    parse_assessment,
    summarize,
    strong_identity,
    infer_team_map,
)
from pipeline.media import cut_video
from workflows import tracklet_bootstrap as flow


OBSERVATIONS = {
    "schema": "player_observations_v3",
    "body_coverage": "full",
    "body_occlusion": "none",
    "torso_visible": True,
    "torso_unoccluded": True,
    "graphic_overlay": "none",
    "blur": "mild",
    "clothing_detail": "distinct",
}


def reading(number="23", color="white", frame=0, **changes):
    raw = {
        "role": "player",
        "single_target": True,
        "unoccluded": True,
        "clear": True,
        "jersey_color": color,
        "number": number,
        "readable": number is not None,
        "view": "front",
        **changes,
    }
    return {
        **parse_assessment(json.dumps(raw)),
        "timestamp_s": frame * 0.5,
        "frame_idx": frame,
    }


def identity(number="23", color="white"):
    return summarize(
        [reading(number, color, 0), reading(number, color, 1)],
        team_map={"white": "home", "black": "away"},
    )


class Detector:
    def __init__(self):
        self.resets = []

    def create_tracker(self, **kwargs):
        self.resets.append(kwargs["only_detr"])
        return {}

    def infer_frame(self, tracker, frame):
        return [
            {"bbox_xyxy": box, "det_score": 0.99, "track_id": i, "category_id": 0}
            for i, box in enumerate(([30, 40, 110, 280], [250, 40, 330, 280]))
        ]


class KPR:
    cfg = SimpleNamespace(
        model=SimpleNamespace(
            kpr=SimpleNamespace(test_embeddings=["bn_foreg", "parts"])
        )
    )

    def extract(self, samples, batch_size):
        rng = np.random.default_rng(42)
        return rng.normal(size=(len(samples), 2, 8)), np.ones((len(samples), 2))


class TrackletChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="basket-tracklet-check-")
        cls.base = Path(cls.temp.name)
        cls.source = cls.base / "source.mkv"
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=480x320:rate=6:duration=30",
                "-c:v",
                "ffv1",
                "-threads",
                "1",
                str(cls.source),
            ],
            check=True,
        )
        cls.fixture_run = cls.base / "results"
        cls.backend = Detector()
        cls.policy = flow.WindowPolicy(
            window_s=2.0,
            candidates_per_window=2,
            min_height=128,
            min_width=40,
            min_sharpness=5.0,
            min_track_s=0.8,
            samples_per_track=3,
        )
        cls.manifest = flow.prepare(
            cls.source,
            cls.fixture_run,
            cls.backend,
            policy=cls.policy,
            detector={"fixture": True},
        )
        queue = []
        for row in cls.manifest["samples"]:
            queue.append(
                json.dumps(
                    {
                        **OBSERVATIONS,
                        "role": "player",
                        "single_target": True,
                        "unoccluded": True,
                        "clear": True,
                        "jersey_color": "white"
                        if row["raw_track_id"] == 0
                        else "black",
                        "number": "23" if row["raw_track_id"] == 0 else "11",
                        "readable": True,
                        "view": "front" if int(row["window_id"][-2:]) % 2 else "back",
                    }
                )
            )

        class Qwen:
            provenance = {"architecture": "synthetic_cpu_fixture"}

            def assess_player(self, image):
                return queue.pop(0)

        flow.assess(cls.fixture_run, Qwen())
        cls.checkpoint = cls.base / "checkpoint"
        cls.checkpoint.write_bytes(b"synthetic; not a model")
        flow.extract_features(cls.fixture_run, KPR(), cls.checkpoint)
        cls.result = flow.register(cls.fixture_run, "synthetic-game")

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_six_windows_have_continuous_trackers_and_native_media(self):
        self.assertEqual(self.backend.resets.count(False), 6)
        self.assertEqual(self.backend.resets.count(True), 12)
        self.assertEqual(len(self.manifest["windows"]), 6)
        self.assertEqual(sum(w["frames"] for w in self.manifest["windows"]), 72)
        for window in self.manifest["windows"]:
            self.assertEqual(window["media"]["width"], 480)
            self.assertEqual(window["media"]["height"], 320)
            self.assertEqual(window["media"]["fps"], 6.0)
            self.assertEqual(window["media"]["encoding"], "ffv1_cfr")

    def test_lossless_frames_retain_decoded_source_pixels(self):
        import cv2

        output = self.base / "pixel-test.mkv"
        cut_video(self.source, output, 1.0, 3.0, lossless=True)
        source = cv2.VideoCapture(str(self.source))
        derived = cv2.VideoCapture(str(output))
        try:
            for _ in range(6):
                self.assertTrue(source.read()[0])
            for _ in range(12):
                ok, a = source.read()
                valid, b = derived.read()
                self.assertTrue(ok and valid)
                np.testing.assert_array_equal(a, b)
        finally:
            source.release()
            derived.release()

    def test_resumed_completed_tracking_does_not_infer_again(self):
        backend = Detector()
        repeated = flow.prepare(
            self.source,
            self.fixture_run,
            backend,
            policy=self.policy,
            detector={"fixture": True},
        )
        self.assertEqual(repeated, self.manifest)
        self.assertEqual(backend.resets, [])

    def test_generic_duration_and_manual_bounds(self):
        self.assertEqual(len(flow.candidate_starts(3600, self.policy)), 6)
        with self.assertRaises(ValueError):
            flow.candidate_starts(5, self.policy)
        path = self.base / "starts.json"
        path.write_text("[0, 1, 4, 6, 8, 10]")
        with self.assertRaisesRegex(ValueError, "overlap"):
            flow.manual_starts(path, 30, self.policy)
        path.write_text("[0, 4, 8, 12, 16, 20]")
        self.assertEqual(flow.manual_starts(path, 30, self.policy)[-1], [20.0])

    def test_blur_overlap_border_and_small_crops_are_rejected(self):
        import cv2

        capture = cv2.VideoCapture(str(self.source))
        _, frame = capture.read()
        capture.release()
        detections = Detector().infer_frame({}, frame)
        self.assertIsNotNone(
            flow.clear_crop(frame, detections[0], detections, self.policy)
        )
        self.assertIsNone(
            flow.clear_crop(
                np.full_like(frame, 128), detections[0], detections, self.policy
            )
        )
        duplicate = {**detections[0], "track_id": 99}
        self.assertIsNone(
            flow.clear_crop(
                frame, detections[0], [detections[0], duplicate], self.policy
            )
        )
        for box in ([0, 40, 80, 280], [30, 40, 50, 80]):
            item = {**detections[0], "bbox_xyxy": box}
            self.assertIsNone(flow.clear_crop(frame, item, [item], self.policy))

    def test_vlm_requires_clear_unoccluded_single_player(self):
        for field, value in (
            ("role", "referee"),
            ("clear", False),
            ("unoccluded", False),
            ("single_target", False),
        ):
            self.assertFalse(reading(**{field: value})["usable"])
        self.assertFalse(parse_assessment("[]")["usable"])
        self.assertTrue(reading(number=None)["usable"])

    def test_multiframe_evidence_abstains_on_weak_or_conflicting_numbers(self):
        self.assertIsNotNone(strong_identity(identity()))
        for rows in (
            [reading()],
            [reading(), reading(frame=0)],
            [reading(), reading(number="32", frame=1)],
            [reading(), reading(color="black", frame=1)],
        ):
            self.assertIsNone(
                strong_identity(
                    summarize(rows, team_map={"white": "home", "black": "away"})
                )
            )
        self.assertIsNone(strong_identity(summarize([reading(), reading(frame=1)])))
        self.assertEqual(identity("00")["jersey"]["number"], "00")

    def test_integer_readings_preserve_zero_distinctions_and_strict_quality(self):
        for value, expected in (
            (0, "0"),
            (8, "8"),
            (23, "23"),
            (99, "99"),
            ("00", "00"),
        ):
            row = reading(number=value)
            self.assertEqual(row["number"], expected)
            self.assertTrue(row["usable"])
        for value in (True, False, 8.0, 100, -1, "01", "023"):
            self.assertIsNone(reading(number=value)["number"])
        self.assertIsNone(reading(number=8, readable=False)["number"])
        self.assertFalse(reading(number=23, clear=False)["usable"])
        self.assertFalse(reading(number=23, unoccluded=False)["usable"])
        self.assertTrue(reading(number=None, readable=False)["usable"])

    def test_small_quality_probe_is_bounded_native_and_read_only(self):
        from analysis.player_quality_probe import probe, select_samples
        from pipeline.media import sha256_file

        before = {p: sha256_file(p) for p in self.fixture_run.rglob("*") if p.is_file()}
        selected = select_samples(self.manifest)
        self.assertEqual(len(selected), 12)
        self.assertEqual(len({s["track_key"] for s in selected}), 12)

        class Backend:
            provenance = {"model_sha256": "cpu-synthetic-model"}
            calls = 0

            def assess_player(self, image):
                self.calls += 1
                return json.dumps(
                    {
                        **OBSERVATIONS,
                        "role": "player",
                        "single_target": True,
                        "unoccluded": True,
                        "clear": True,
                        "quality_reason": "synthetic test fixture, not model accuracy",
                        "jersey_color": "white",
                        "number": 23,
                        "readable": True,
                        "view": "front",
                    }
                )

        backend = Backend()
        with tempfile.TemporaryDirectory(dir=self.base) as tmp:
            output = Path(tmp) / "probe"
            report = probe(self.fixture_run, output, backend=backend)
            self.assertEqual(backend.calls, 12)
            self.assertEqual(report["usable_images"], 12)
            self.assertEqual(report["provenance"]["task"], "player_observations_v3")
            self.assertFalse(report["production_modified"])
            self.assertFalse(report["accuracy_evaluated"])
            self.assertTrue((output / "index.html").is_file())
            self.assertTrue(
                all(r["reading"]["number"] == "23" for r in report["samples"])
            )
            self.assertTrue(
                all(r["native_size"] == [80, 240] for r in report["samples"])
            )
            self.assertTrue(all(r["previous"] is not None for r in report["samples"]))
            self.assertEqual(
                before,
                {p: sha256_file(p) for p in self.fixture_run.rglob("*") if p.is_file()},
            )
            with self.assertRaises(FileExistsError):
                probe(self.fixture_run, output, backend=backend)
            with self.assertRaises(ValueError):
                probe(self.fixture_run, self.fixture_run / "probe", backend=backend)

    def test_v3_number_survives_incomplete_body_but_not_torso_overlay(self):
        raw = {
            **OBSERVATIONS,
            "role": "player",
            "single_target": True,
            "number": "35",
            "readable": True,
            "jersey_color": "white",
            "view": "back",
        }
        for change in (
            {"body_coverage": "upper"},
            {"graphic_overlay": "non_torso"},
            {"clothing_detail": "limited"},
        ):
            result = parse_assessment(
                json.dumps({**raw, **change}), require_observations=True
            )
            self.assertFalse(result["usable"])
            self.assertTrue(result["evidence_eligible"])
            self.assertEqual(result["number"], "35")
        for change in (
            {"graphic_overlay": "torso"},
            {"torso_unoccluded": False},
            {"role": "referee"},
            {"single_target": False},
        ):
            result = parse_assessment(
                json.dumps({**raw, **change}), require_observations=True
            )
            self.assertFalse(result["usable"])
            self.assertFalse(result["evidence_eligible"])
            self.assertIsNone(result["number"])
        self.assertFalse(
            parse_assessment('{"clear":true}', require_observations=True)["usable"]
        )
        legacy = json.dumps(
            {"role": "player", "single_target": True, "unoccluded": True, "clear": True}
        )
        self.assertFalse(parse_assessment(legacy, require_observations=True)["usable"])
        missing = {k: v for k, v in raw.items() if k != "body_coverage"}
        self.assertFalse(
            parse_assessment(json.dumps(missing), require_observations=True)["usable"]
        )

    def test_explicit_team_aliases_and_joint_evidence(self):
        rows = [reading(color="black", frame=0), reading(color="purple", frame=1)]
        aliases = {"black": "dark", "purple": "dark", "white": "light"}
        self.assertEqual(
            strong_identity(summarize(rows, team_map=aliases)), ("dark", "23")
        )
        self.assertIsNone(
            strong_identity(summarize(rows, team_map={"black": "a", "purple": "b"}))
        )
        self.assertIsNone(strong_identity(summarize(rows, team_map={"black": "a"})))
        disjoint = [reading(number=None, frame=i) for i in (0, 1)] + [
            reading(color="unknown", frame=i) for i in (2, 3)
        ]
        self.assertIsNone(
            strong_identity(summarize(disjoint, team_map={"white": "home"}))
        )

    def test_one_template_can_use_two_independent_number_observations(self):
        raw = {
            **OBSERVATIONS,
            "role": "player",
            "single_target": True,
            "jersey_color": "white",
            "number": "35",
            "readable": True,
            "view": "back",
        }
        reads = [
            {
                **parse_assessment(
                    json.dumps({**raw, "body_coverage": coverage}),
                    require_observations=True,
                ),
                "frame_idx": i,
                "timestamp_s": i,
            }
            for i, coverage in enumerate(("full", "upper"))
        ]
        track = summarize(reads, team_map={"white": "home"})
        manifest = {
            "samples": [{"sample_id": str(i), "track_key": "T"} for i in range(2)]
        }
        assessments = {"readings": dict(zip(("0", "1"), reads)), "tracks": {"T": track}}
        self.assertEqual(
            [s["sample_id"] for s in flow.accepted_samples(manifest, assessments)],
            ["0"],
        )
        reads[1]["timestamp_s"] = 0.1
        assessments["tracks"]["T"] = summarize(reads, team_map={"white": "home"})
        self.assertEqual(flow.accepted_samples(manifest, assessments), [])

    def test_reuse_copies_tracking_but_not_model_or_library(self):
        with tempfile.TemporaryDirectory(dir=self.base) as tmp:
            target = Path(tmp) / "results"
            result = flow.reuse_tracking(self.fixture_run, target)
            self.assertEqual(result, self.manifest)
            self.assertEqual(flow.load_manifest(target), self.manifest)
            self.assertFalse((target / "library").exists())
            self.assertFalse((flow.folder(target) / "assessments.json").exists())
            self.assertFalse((flow.folder(target) / "features.npz").exists())
            self.assertEqual(flow.reuse_tracking(self.fixture_run, target), result)
            crop = flow.folder(target) / result["samples"][0]["crop_path"]
            crop.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                flow.reuse_tracking(self.fixture_run, target)
            self.assertEqual(flow.load_manifest(self.fixture_run), self.manifest)

    def test_probe_baseline_is_fixed_and_number_disagreement_is_not_accepted(self):
        from analysis.player_quality_probe import probe
        from pipeline.media import sha256_file

        class Backend:
            provenance = {"model_sha256": "synthetic"}

            def assess_player(self, image):
                return json.dumps(
                    {
                        **OBSERVATIONS,
                        "role": "player",
                        "single_target": True,
                        "jersey_color": "white",
                        "number": "35",
                        "readable": True,
                        "view": "back",
                    }
                )

            def read(self, image):
                return '{"number":"53","readable":true}'

        with tempfile.TemporaryDirectory(dir=self.base) as tmp:
            root = Path(tmp)
            baseline = root / "baseline.json"
            selections = list(reversed(self.manifest["samples"][:3]))
            baseline.write_text(
                json.dumps(
                    {
                        "status": "completed_not_accuracy_evaluated",
                        "manifest_sha256": sha256_file(
                            flow.folder(self.fixture_run) / "manifest.json"
                        ),
                        "provenance": {"model_sha256": "synthetic"},
                        "samples": [
                            {
                                "sample_id": r["sample_id"],
                                "reading": {"usable": False},
                                "raw_response": "old response",
                            }
                            for r in selections
                        ],
                    }
                )
            )
            report = probe(
                self.fixture_run,
                root / "output",
                backend=Backend(),
                baseline=baseline,
                compare_number=True,
            )
            self.assertEqual(
                [r["sample_id"] for r in report["samples"]],
                [r["sample_id"] for r in selections],
            )
            self.assertEqual(report["number_disagreements"], 3)
            self.assertEqual(report["number_evidence_images"], 3)
            self.assertTrue(report["same_model_weights"])
            self.assertTrue(
                all(r["reading"]["number"] == "35" for r in report["samples"])
            )
            self.assertTrue((root / "output.tar.gz").is_file())
            data = json.loads(baseline.read_text())
            data["manifest_sha256"] = "other"
            baseline.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                probe(
                    self.fixture_run,
                    root / "wrong",
                    backend=Backend(),
                    baseline=baseline,
                )

    def test_empty_baseline_is_rejected_before_model_calls(self):
        from analysis.player_quality_probe import probe

        with tempfile.TemporaryDirectory(dir=self.base) as tmp:
            root = Path(tmp)
            baseline = root / "baseline.json"
            baseline.write_text("{}")
            with self.assertRaises(ValueError):
                probe(
                    self.fixture_run,
                    root / "probe",
                    backend=object(),
                    baseline=baseline,
                )

    def test_actual_qwen_image_patch_reconstruction(self):
        from analysis.player_quality_probe import save_image_inputs
        from PIL import Image

        # Known RGB pixels packed in the Qwen temporal/spatial merge order.
        rgb = np.arange(8 * 8 * 3, dtype=np.float32).reshape(8, 8, 3)
        frames = np.stack([rgb.transpose(2, 0, 1)] * 2)[None]
        packed = (
            frames.reshape(1, 2, 3, 2, 2, 2, 2, 2, 2)
            .transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
            .reshape(16, 24)
        )
        ip = SimpleNamespace(
            patch_size=2,
            temporal_patch_size=2,
            merge_size=2,
            do_normalize=False,
            do_rescale=False,
            size={"shortest_edge": 64, "longest_edge": 64},
        )
        processor = SimpleNamespace(
            image_processor=ip, batch_decode=lambda *a, **k: ["prompt"]
        )
        inputs = {
            "image_grid_thw": np.array([[1, 4, 4]]),
            "pixel_values": packed,
            "input_ids": np.zeros((1, 20)),
        }
        with tempfile.TemporaryDirectory(dir=self.base) as tmp:
            result = save_image_inputs(processor, inputs, tmp)
            self.assertEqual(result["processed_width"], 8)
            np.testing.assert_array_equal(
                np.asarray(Image.open(Path(tmp) / "processed.png")),
                rgb.astype(np.uint8),
            )

    def test_feature_probe_measures_rejected_images_without_registering_them(self):
        from analysis.initialization_feature_probe import probe_features
        from pipeline.media import sha256_file

        samples = self.manifest["samples"]
        before = {p: sha256_file(p) for p in self.fixture_run.rglob("*") if p.is_file()}
        with tempfile.TemporaryDirectory(dir=self.base) as tmp:
            root = Path(tmp)
            report = root / "quality.json"
            selected = [
                samples[0],
                next(s for s in samples if s["track_key"] != samples[0]["track_key"]),
            ]
            report.write_text(
                json.dumps(
                    {
                        "status": "completed_not_accuracy_evaluated",
                        "manifest_sha256": sha256_file(
                            flow.folder(self.fixture_run) / "manifest.json"
                        ),
                        "samples": [
                            {"sample_id": r["sample_id"], "reading": {"usable": False}}
                            for r in selected
                        ],
                    }
                )
            )
            result = probe_features(
                self.fixture_run, report, root / "features", KPR(), self.checkpoint
            )
            self.assertLessEqual(result["sample_count"], 4)
            self.assertGreater(result["pair_relations"]["same_track_candidate"], 0)
            self.assertFalse(result["accuracy_evaluated"])
            self.assertTrue((root / "features/pairs.csv").is_file())
            self.assertEqual(
                before,
                {p: sha256_file(p) for p in self.fixture_run.rglob("*") if p.is_file()},
            )

    def test_team_palette_requires_two_independently_supported_kits(self):
        white = [reading(frame=i) for i in (0, 1)]
        black = [reading(color="black", frame=i) for i in (0, 1)]
        self.assertEqual(infer_team_map([white, white]), {})
        self.assertEqual(
            infer_team_map([white, white, black, black]),
            {"black": "KIT-black", "white": "KIT-white"},
        )

    def test_clear_windows_can_replace_a_blurry_time_stratum(self):
        selection = []
        for i in range(6):
            selection.append(
                [
                    {
                        "start_s": i * 20 + j * 4,
                        "end_s": i * 20 + j * 4 + 2,
                        "probe_clear_people": 0 if i == 0 else 2,
                        "selection_score": 0 if i == 0 else 3 - j,
                    }
                    for j in range(2)
                ]
            )
        windows = flow.choose_windows(selection, self.policy)
        self.assertEqual(len(windows), 6)
        self.assertTrue(all(row["start_s"] >= 20 for row in windows))
        with self.assertRaisesRegex(ValueError, "Fewer than six"):
            flow.choose_windows([[selection[0][0]]] * 6, self.policy)

    def test_anonymous_initial_tracklet_waits_for_an_anchor(self):
        node = {
            "node_id": "anon",
            "clip_name": "window",
            "evidence_kind": "bootstrap_tracklet",
            "sample_count": 3,
            "intervals": [[0, 2]],
            "cover": {"archive_quality_score": 0.9},
            **summarize([reading(number=None, frame=i) for i in (0, 1)]),
        }
        records = [
            {"node": node, "features": [[[1, 0], [1, 0]]], "visibility": [[1, 1]]}
        ]
        state = plan_memory(records, {}, MemoryPolicy())["states"][0]
        self.assertEqual(state["status"], "pending")
        self.assertIn("initial_identity_needs_team_and_number", state["reasons"])

    def test_rich_clip_evidence_reaches_library_and_filters_templates(self):
        from analysis.person_library_selfcheck import make_library_fixture, rows
        from workflows.library_inputs import load_library_inputs, PersonLibraryOptions
        from workflows.gallery_curation import curate_records
        from PIL import Image

        with tempfile.TemporaryDirectory(dir=self.base) as tmp:
            root = Path(tmp)
            run = make_library_fixture(root, angles=(0,))
            clip = run / "clip1"
            palette = {"white": "home", "black": "away"}
            readings = [
                {**reading(frame=i), "crop_path": f"jersey_media/{i}.png"}
                for i in (0, 1)
            ]
            evidence = summarize(readings, team_map=palette)
            rows(
                clip / "identity/jersey_tracks.jsonl",
                [
                    {
                        "raw_track_id": 0,
                        **evidence["jersey"],
                        "team_id": evidence["team_id"],
                        "team_evidence": evidence["team_evidence"],
                    }
                ],
            )
            (clip / "identity/jersey_summary.json").write_text(
                json.dumps(
                    {
                        "backend": "qwen_vl",
                        "provenance": {"synthetic": True},
                        "thresholds": {
                            "min_gap_s": 0.5,
                            "min_support": 2,
                            "team_map": palette,
                        },
                    }
                )
            )
            nodes, features, visible, *_ = load_library_inputs(
                PersonLibraryOptions(run, use_jersey_evidence=True)
            )
            self.assertEqual(strong_identity(nodes[0]), ("home", "23"))
            raw = clip / "identity_raw"
            samples = []
            for i in (0, 1, 2):
                path = raw / f"sample-{i}.png"
                Image.fromarray(
                    np.random.default_rng(i).integers(
                        20, 230, (160, 80, 3), dtype=np.uint8
                    )
                ).save(path)
                samples.append(
                    {
                        "sample_id": f"s{i}",
                        "crop_path": path.name,
                        "track_id": 0,
                        "frame_idx": i,
                    }
                )
            rows(raw / "kpr_samples.jsonl", samples)
            np.savez(
                raw / "kpr_samples.npz",
                embeddings=np.ones((3, 2, 2)) / np.sqrt(2),
                visibility_scores=np.ones((3, 2)),
                track_ids=np.zeros(3, dtype=int),
                frame_indices=np.arange(3),
            )
            records = [
                {
                    "node": nodes[0],
                    "features": [features[0].tolist()],
                    "visibility": [visible[0].tolist()],
                }
            ]
            bank = curate_records(run, records)[nodes[0]["node_id"]]["templates"]
            self.assertTrue(bank)
            self.assertTrue(all(row["frame_idx"] in (0, 1) for row in bank))
            for track in nodes[0]["jersey"]["tracks"]:
                for r in track["readings"]:
                    r["usable"] = False
            held = curate_records(run, records)[nodes[0]["node_id"]]
            self.assertEqual(held["templates"], [])
            self.assertIn("no_clear_assessed_kpr_images", held["hold_reasons"])

    def test_clip_qwen_assesses_full_players_and_reuses_cache(self):
        from PIL import Image
        from contracts.schema import read_jsonl
        from pipeline.identity.jersey_qwen import recognize_jerseys_qwen

        with tempfile.TemporaryDirectory(dir=self.base) as tmp:
            root = Path(tmp)
            output = root / "identity"
            output.mkdir()
            mapping = root / "mapping.jsonl"
            mapping.write_text('{"raw_track_id":0,"person_id":"P0"}\n')
            palette = root / "teams.json"
            palette.write_text('{"white":"home","black":"away"}')
            samples = []
            for frame in (0, 1):
                relative = f"sample-{frame}.png"
                Image.new("RGB", (80, 160), (70 + frame, 90, 110)).save(
                    output / relative
                )
                samples.append(
                    {
                        "raw_track_id": 0,
                        "frame_idx": frame,
                        "timestamp_s": frame * 0.5,
                        "crop_path": relative,
                    }
                )

            class Backend:
                provenance = {"backend": "qwen_vl", "synthetic": True}
                calls = 0

                def read(self, image):
                    raise AssertionError("Legacy torso-number prompt used")

                def assess_player(self, image):
                    self.calls += 1
                    return json.dumps(
                        {
                            **OBSERVATIONS,
                            "role": "player",
                            "single_target": True,
                            "unoccluded": True,
                            "clear": True,
                            "jersey_color": "white",
                            "number": "23",
                            "readable": True,
                            "view": "unknown",
                        }
                    )

            backend = Backend()
            with patch(
                "pipeline.identity.jersey_qwen.sample_torsos", return_value=samples
            ) as sampler:
                for _ in range(2):
                    recognize_jerseys_qwen(
                        root / "no-video",
                        root / "no-tracks",
                        mapping,
                        output,
                        root / "no-model",
                        backend=backend,
                        assess_player=True,
                        team_map=palette,
                        min_gap_s=0.5,
                    )
                self.assertTrue(sampler.call_args.kwargs["full_person"])
            self.assertEqual(backend.calls, 2)
            track = list(read_jsonl(output / "jersey_tracks.jsonl"))[0]
            self.assertEqual(
                strong_identity(
                    {
                        "jersey": track,
                        "team_id": track["team_id"],
                        "team_evidence": track["team_evidence"],
                    }
                ),
                ("home", "23"),
            )

    def test_strong_identity_links_far_views_but_blocks_conflicts(self):
        def node(key, color="white", number="23", start=0):
            return {
                "node_id": key,
                "clip_name": key,
                "evidence_kind": "bootstrap_tracklet",
                "sample_count": 2,
                "source_video_id": "source",
                "source_clip_start_s": start,
                "intervals": [[0, 1]],
                "cover": {"archive_quality_score": 0.9},
                **identity(number, color),
            }

        a = node("a")
        b = node("b", start=2)
        records = [
            {"node": n, "features": [[[v, 0], [v, 0]]], "visibility": [[1, 1]]}
            for n, v in ((a, 1), (b, -1))
        ]
        plan = plan_memory(records, {}, MemoryPolicy())
        self.assertEqual(len(plan["groups"]), 1)
        self.assertTrue(plan["groups"][0]["promoted"])
        self.assertEqual(
            incompatible(a, node("c", color="black", start=2)), "different_teams"
        )
        conflict = deepcopy(records)
        from pipeline.identity.player_registry import PlayerRegistry

        self.assertEqual(
            PlayerRegistry._conflict(a, node("c", color="black", start=2)),
            "different_teams",
        )
        conflict[1]["node"]["source_clip_start_s"] = 0
        states = plan_memory(conflict, {}, MemoryPolicy())["states"]
        self.assertTrue(
            all("simultaneous_strong_identity_conflict" in s["reasons"] for s in states)
        )
        clip = {**b, "evidence_kind": "track", "source_clip_start_s": 0}
        self.assertIsNone(incompatible(a, clip))

    def test_initial_library_bounded_templates_and_stable_resume(self):
        from contracts.schema import read_jsonl

        manifest = json.loads(
            (self.fixture_run / "library/library_manifest.json").read_text()
        )
        people = list(
            read_jsonl(
                self.fixture_run / "library" / manifest["index_dir"] / "people.jsonl"
            )
        )
        self.assertEqual(manifest["candidate_person_count"], 2)
        self.assertEqual(manifest["expected_clips"], 0)
        self.assertEqual(manifest["bootstrap_photo_count"], 0)
        self.assertEqual(manifest["bootstrap_tracklet_count"], 12)
        for p in people:
            self.assertLessEqual(p["gallery_template_count"], 6)
            views = [t["view"] for m in p["members"] for t in m["memory"]["templates"]]
            self.assertLessEqual(views.count("front"), 2)
            self.assertLessEqual(views.count("back"), 2)
            self.assertEqual(p["clip_count"], 0)
        self.assertEqual(
            flow.register(self.fixture_run, "synthetic-game")["revision"],
            self.result["revision"],
        )
        with self.assertRaisesRegex(ValueError, "different match"):
            flow.register(self.fixture_run, "other-game")

    def test_curation_preview_and_rollback_reuse_identical_banks(self):
        from workflows.gallery_curation import curate_records
        from pipeline.identity.player_registry import PlayerRegistry

        with PlayerRegistry(
            self.fixture_run / "library/players.sqlite3", "synthetic-game"
        ) as registry:
            before = registry.snapshot()
            evidence = curate_records(self.fixture_run, registry.memory_records())
            revision = registry.refresh_memory(
                evidence=evidence, expected_revision=before["revision"]
            )
            self.assertEqual(registry.snapshot()["people"], before["people"])
            # No-op reprocessing leaves the current revision intact.
            self.assertEqual(revision, before["revision"])
        from workflows.gallery_maintenance import maintain

        with tempfile.TemporaryDirectory(dir=self.base) as tmp:
            output = Path(tmp) / "preview"
            report = maintain(self.fixture_run, operation="preview", output_dir=output)
            self.assertFalse(report["production_modified"])
            self.assertTrue((output / "index.html").is_file())
            originals = list((output / "original_images").glob("*.png"))
            self.assertTrue(originals)
            from PIL import Image

            with Image.open(originals[0]) as image:
                self.assertEqual(image.size, (80, 240))

    def test_source_and_crop_tampering_fail_closed(self):
        with tempfile.TemporaryDirectory(dir=self.base) as tmp:
            copy = Path(tmp) / "results"
            shutil.copytree(self.fixture_run, copy)
            crop = (
                copy / "bootstrap_tracklets" / self.manifest["samples"][0]["crop_path"]
            )
            crop.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "changed"):
                flow.load_manifest(copy)

    def test_new_clips_cannot_change_the_initial_model_or_team_palette(self):
        from pipeline.identity.player_registry import PlayerRegistry

        with PlayerRegistry(
            self.fixture_run / "library/players.sqlite3", "synthetic-game"
        ) as registry:
            before = registry.snapshot()["revision"]
            original = registry.memory_records()[0]
            for field, value in (
                ("model", {"architecture": "changed"}),
                ("team_map", {"white": "other"}),
            ):
                node = deepcopy(original["node"])
                node["node_id"] = "foreign-contract"
                node["feature_indices"] = list(range(len(original["features"])))
                node["jersey"]["identity_evidence_contract"][field] = value
                with self.assertRaisesRegex(ValueError, "Qwen model or team palette"):
                    registry.ingest(
                        [node],
                        np.asarray(original["features"]),
                        np.asarray(original["visibility"]),
                        [],
                        json.loads(registry.meta("feature_contract")),
                        max_distance=0.2,
                        novelty_distance=0.5,
                        min_margin=0.05,
                    )
                self.assertEqual(registry.snapshot()["revision"], before)

    def test_bootstrap_only_frontend_serves_original_crops_and_windows(self):
        from fastapi.testclient import TestClient
        from web.backend import app as api
        from web.backend.imports import import_results
        from web.backend.settings import WebSettings
        from web.backend.store import ProjectStore

        with tempfile.TemporaryDirectory(dir=self.base) as tmp:
            webroot = Path(tmp)
            settings = replace(
                WebSettings(),
                source_root=self.base,
                import_root=self.base,
                output_root=webroot / "output",
            )
            store = ProjectStore(settings.data_root, settings.output_root)
            project = import_results(settings, store, self.fixture_run)
            self.assertEqual(project["videos"], [])
            with (
                patch.object(api, "settings", settings),
                patch.object(api, "store", store),
                TestClient(api.app) as client,
            ):
                prefix = f"/api/projects/{project['project_id']}"
                catalog = client.get(prefix + "/person-library")
                self.assertEqual(catalog.status_code, 200, catalog.text)
                person = client.get(
                    prefix
                    + "/person-library/people/"
                    + catalog.json()["items"][0]["global_person_id"]
                )
                self.assertEqual(person.status_code, 200, person.text)
                member = person.json()["members"][0]
                image = client.get(member["cover"]["crop_url"])
                self.assertEqual(image.status_code, 200)
                window = client.get(member["window_url"])
                self.assertEqual(window.status_code, 200)
                self.assertIn("attachment", window.headers["content-disposition"])
                self.assertEqual(
                    client.get(
                        prefix + "/bootstrap-tracklet-media/unknown.png"
                    ).status_code,
                    404,
                )

    def test_zero_accepted_audit_preserves_native_crops_and_original_evidence(self):
        import tarfile
        from collections import defaultdict
        from analysis.tracklet_audit import audit, raw_object
        from pipeline.media import atomic_json, sha256_file
        from PIL import Image

        with tempfile.TemporaryDirectory(dir=self.base) as tmp:
            root = Path(tmp)
            run = root / "failed-results"
            shutil.copytree(self.fixture_run, run)
            seed = run / "bootstrap_tracklets"
            data = json.loads((seed / "assessments.json").read_text())
            grouped = defaultdict(list)
            for sample in self.manifest["samples"]:
                sid = sample["sample_id"]
                row = data["readings"][sid]
                raw = raw_object(row["raw_response"])
                raw.update(clear=False, blur="severe", number=None, readable=False)
                if len(grouped) == 0:
                    raw["extra"] = "<script>alert('untrusted')</script>"
                row.update(
                    parse_assessment(json.dumps(raw)), raw_response=json.dumps(raw)
                )
                grouped[sample["track_key"]].append(row)
            data["tracks"] = {k: summarize(v) for k, v in grouped.items()}
            data["team_map"] = {}
            atomic_json(seed / "assessments.json", data)
            before = {p: sha256_file(p) for p in run.rglob("*") if p.is_file()}
            output = root / "visual-audit"
            with patch(
                "pipeline.identity.qwen_backend.QwenBackend",
                side_effect=AssertionError("GPU model called"),
            ):
                report, files = audit(run, output, context_samples=6)
            self.assertEqual(report["accepted_for_kpr"], 0)
            self.assertEqual(
                report["rejection_reasons"],
                {"insufficient_clothing_detail": len(self.manifest["samples"])},
            )
            self.assertEqual(
                report["raw_fields"]["jersey_color"].keys(), {'"white"', '"black"'}
            )
            self.assertFalse(report["accuracy_evaluated"])
            self.assertFalse(report["production_modified"])
            self.assertEqual(
                before, {p: sha256_file(p) for p in run.rglob("*") if p.is_file()}
            )
            html = (output / "index.html").read_text()
            self.assertIn("&lt;script&gt;", html)
            self.assertNotIn("<script>alert", html)
            self.assertEqual(len(report["contexts"]), 6)
            for context in report["contexts"].values():
                with Image.open(output / context["image"]) as image:
                    self.assertEqual(image.size, (480, 320))
            self.assertTrue(
                (output / "samples.csv").read_bytes().startswith(b"\xef\xbb\xbf")
            )
            samples = list((output / "media").glob("*.png"))
            self.assertEqual(len(samples), len(self.manifest["samples"]) + 6)
            source_hashes = {s["crop_sha256"] for s in self.manifest["samples"]}
            self.assertTrue(source_hashes.issubset({sha256_file(p) for p in samples}))
            with tarfile.open(files["archive"], "r:gz") as archive:
                self.assertIn("visual-audit/index.html", archive.getnames())
                self.assertFalse(any(".mkv" in p for p in archive.getnames()))
            with self.assertRaises(FileExistsError):
                audit(run, output)
            with self.assertRaises(ValueError):
                audit(run, run / "audit")


def main():
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(TrackletChecks)
    )
    if not result.wasSuccessful():
        raise SystemExit(1)
