"""CPU tensor/runtime checks; run with --source-root pointing to pinned TCEI."""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from adapters.tcei import load_reference
from pipeline.tracking.tcei import Calibrator, TceiConfig, create_tracker

REFERENCE = None
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "MOTIP"))
from models.runtime_tracker import RuntimeTracker


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        head = nn.Linear(256, 51, bias=False)
        with torch.no_grad():
            head.weight.zero_()
            head.weight[0, 0] = 8
            head.weight[1, 1] = 8
        self.embed_to_word_layers = nn.ModuleList([head, head, head])
        self.final_features = None

    def forward(self, seq_info, use_decoder_checkpoint=False):
        for index, head in enumerate(self.embed_to_word_layers):
            features = seq_info["unknown_features"].clone()
            features[..., 10] = index + 1
            logits = head(features)
        self.final_features = features.reshape(-1, 256).detach()
        return logits, None, seq_info["unknown_masks"]


class Model(nn.Module):
    num_id_vocabulary = 50

    def __init__(self):
        super().__init__()
        self.id_decoder = Decoder()

    def forward(self, *, part, frames=None, seq_info=None):
        if part == "trajectory_modeling":
            return seq_info
        if part == "id_decoder":
            return self.id_decoder(seq_info, use_decoder_checkpoint=False)
        features = torch.zeros(1, 2, 256)
        features[0, 0, 0] = features[0, 1, 1] = 1
        return {
            "pred_logits": torch.full((1, 2, 1), 9.0),
            "outputs": features,
            "pred_boxes": torch.tensor([[[0.3, 0.5, 0.2, 0.4], [0.7, 0.5, 0.2, 0.4]]]),
        }


def tracker(model, mode, wide=None):
    arguments = {
        "model": model,
        "sequence_hw": (100, 200),
        "assignment_protocol": "object-max",
        "use_sigmoid": False,
        "dtype": torch.float32,
        "only_detr": False,
    }
    if mode == "baseline":
        return RuntimeTracker(**arguments)
    return create_tracker(
        RuntimeTracker,
        calibration=TceiConfig(mode=mode),
        reference=REFERENCE,
        wide_caches=wide,
        **arguments,
    )


class TceiChecks(unittest.TestCase):
    def setUp(self):
        # Keep fixtures on CPU even when Accelerate detects an available GPU.
        device = patch(
            "models.runtime_tracker.distributed_device",
            return_value=torch.device("cpu"),
        )
        device.start()
        self.addCleanup(device.stop)

    def test_off_matches_actual_upstream_and_no_hooks_remain(self):
        model = Model()
        baseline, off = tracker(model, "baseline"), tracker(model, "off")
        for frame in range(4):
            baseline.update(frame)
            off.update(frame)
            for key, expected in baseline.get_track_results().items():
                torch.testing.assert_close(
                    off.get_track_results()[key], expected, rtol=0, atol=0
                )
        self.assertFalse(model.id_decoder._forward_hooks)
        self.assertFalse(model.id_decoder.embed_to_word_layers[-1]._forward_pre_hooks)

    def test_enabled_hooks_use_last_shared_head_and_raw_object_features(self):
        model = Model()
        current = tracker(model, "dual")
        current.update(0)
        current.update(1)
        self.assertEqual(current.calibrator.calibrated_rows, 2)
        expected = nn.functional.normalize(model.id_decoder.final_features[0:1], dim=-1)
        torch.testing.assert_close(current.calibrator.wide[0][0][0][0], expected)
        # Object features are raw DETR features and do not contain decoder offset.
        for entries in current.calibrator.local[0].values():
            self.assertTrue(all(float(entry[0][0, 10]) == 0 for entry in entries))
        self.assertFalse(model.id_decoder._forward_hooks)
        self.assertFalse(model.id_decoder.embed_to_word_layers[-1]._forward_pre_hooks)

    def test_hook_cleanup_after_calibration_failure(self):
        model = Model()
        current = tracker(model, "dual")
        current.update(0)
        with (
            patch.object(Calibrator, "__call__", side_effect=RuntimeError("injected")),
            self.assertRaisesRegex(RuntimeError, "injected"),
        ):
            current.update(1)
        self.assertFalse(model.id_decoder._forward_hooks)
        self.assertFalse(model.id_decoder.embed_to_word_layers[-1]._forward_pre_hooks)

    def test_ablation_modes_keep_unused_caches_empty(self):
        for mode in ("object", "id"):
            current = tracker(Model(), mode)
            current.update(0)
            current.update(1)
            self.assertEqual(current.calibrator.calibrated_rows, 2)
            unused = (
                current.calibrator.wide
                if mode == "object"
                else current.calibrator.local
            )
            self.assertFalse(unused[0])
            self.assertFalse(unused[1])

    def test_real_upstream_id_decoder_feature_hook(self):
        # Import the actual decoder file without importing the DETR CUDA module.
        spec = importlib.util.spec_from_file_location(
            "basket_cpu_id_decoder", ROOT / "MOTIP/models/motip/id_decoder.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model = Model()
        model.id_decoder = module.IDDecoder(
            feature_dim=256,
            id_dim=256,
            ffn_dim_ratio=2,
            num_layers=2,
            head_dim=32,
            num_id_vocabulary=50,
            rel_pe_length=60,
            use_aux_loss=False,
            use_shared_aux_head=True,
        )
        with torch.no_grad():
            model.id_decoder.embed_to_word_layers[0].weight.zero_()
        current = tracker(model, "dual")
        current.id_thresh = 0
        captured = []
        handle = model.id_decoder.embed_to_word_layers[-1].register_forward_pre_hook(
            lambda _, inputs: captured.append(inputs[0].detach().clone())
        )
        try:
            current.update(0)
            current.update(1)
        finally:
            handle.remove()
        self.assertEqual(len(captured), 2)
        expected = nn.functional.normalize(captured[-1].reshape(-1, 256)[0:1], dim=-1)
        torch.testing.assert_close(current.calibrator.wide[0][0][0][0], expected)
        self.assertGreater(current.calibrator.calibrated_rows, 0)

    def test_empty_single_and_repeated_features_stay_finite(self):
        current = Calibrator(TceiConfig(), REFERENCE)
        feature = torch.zeros(1, 4)
        self.assertEqual(
            float(
                current.cache_logits(
                    feature, {}, 3, {}, wide=False, negative=False
                ).sum()
            ),
            0,
        )
        current.seed(10, feature, 1)
        for mapping in ({0: 10}, {0: 10, 1: 20}):
            if 20 in mapping.values():
                current.seed(20, feature, 1)
            logits = current.cache_logits(
                feature, current.local[0], 3, mapping, wide=False, negative=False
            )
            self.assertTrue(torch.isfinite(logits).all())
        for _ in range(6):
            current.seed(10, feature, 0.5)
        self.assertEqual(len(current.local[0][10]), 3)

    def test_negative_mapping_and_reused_slots(self):
        current = Calibrator(TceiConfig(), REFERENCE)
        a, b = torch.tensor([[1.0, 0.0]]), torch.tensor([[0.0, 1.0]])
        current.insert(current.local[1], 10, a, 0.2, 2, (20,))
        current.insert(current.local[1], 20, b, 0.3, 2, (10,))
        result = current.cache_logits(
            a, current.local[1], 4, {0: 10, 2: 20}, wide=False, negative=True
        )
        self.assertGreater(float(result[0, 2]), float(result[0, 0]))
        self.assertEqual(float(result[0, 1]), 0)
        reused = current.cache_logits(
            a, current.local[1], 4, {0: 30, 2: 20}, wide=False, negative=True
        )
        self.assertEqual(float(reused[0, 0]), 0)
        current.prune({20})
        self.assertEqual(set(current.local[1]), {20})

    def test_wide_cache_formula_matches_author(self):
        current = Calibrator(TceiConfig(), REFERENCE)
        feature = torch.tensor([[1.0, 0.0]])
        current.insert(current.wide[0], 0, feature, 0.2, 3)
        current.insert(current.wide[0], 1, torch.tensor([[0.0, 1.0]]), 0.3, 3)
        actual = current.cache_logits(
            feature, current.wide[0], 3, {}, wide=True, negative=False
        )
        author_cache = {
            key: [[item[0], item[1]] for item in entries]
            for key, entries in current.wide[0].items()
        }
        with patch.object(torch.Tensor, "cuda", lambda tensor: tensor):
            expected = REFERENCE.compute_cache_logits(
                feature, author_cache, 1.0, 5.0, 3, {}, 0.5, is_wide=True
            )
        torch.testing.assert_close(actual, expected)

    def test_sequence_reset_and_explicit_stream_caches(self):
        first = tracker(Model(), "dual")
        first.update(0)
        first.update(1)
        second = tracker(Model(), "dual")
        self.assertFalse(second.calibrator.local[0])
        self.assertFalse(second.calibrator.wide[0])
        shared = ({}, {})
        stream = tracker(Model(), "dual", shared)
        stream.update(0)
        stream.update(1)
        next_sequence = tracker(Model(), "dual", shared)
        self.assertTrue(next_sequence.calibrator.wide[0])
        self.assertFalse(next_sequence.calibrator.local[0])

    def test_parity_gate_detects_score_change(self):
        from analysis.sportsmot.tcei import parity

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for method in ("baseline", "off"):
                path = root / method / "val/predictions/motip/data/demo.txt"
                path.parent.mkdir(parents=True)
                path.write_text("1,0,10,10,20,30,0.9,-1,-1,-1\n")
            sequences = [{"sequence": "demo", "frames": 1}]
            parity(root, sequences)
            path.write_text("1,0,10,10,20,30,0.8,-1,-1,-1\n")
            with self.assertRaisesRegex(ValueError, "Disabled TCEI"):
                parity(root, sequences)

    def test_frozen_manifest_trackeval_and_modified_result_rejection(self):
        from analysis.sportsmot.dataset import prepare as prepare_dataset
        from analysis.sportsmot.runner import CONFIG
        from analysis.sportsmot.selfcheck import SportsMotChecks
        from analysis.sportsmot.tcei import discover_baseline, evaluate, prepare
        from analysis.teamtrack.dataset import save_json, sha256

        fixture = SportsMotChecks()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        baseline, output = fixture.output, fixture.root / "tcei"
        with contextlib.redirect_stdout(io.StringIO()):
            prepare_dataset(fixture.data, baseline, 0)
        checkpoint = fixture.root / "synthetic-checkpoint"
        checkpoint.write_bytes(b"not-a-real-model: metrics fixture only")
        save_json(
            baseline / "experiment_protocol.json",
            {
                "dataset_sha256": sha256(baseline / "dataset.json"),
                "checkpoint_sha256": sha256(checkpoint),
                "config": CONFIG,
            },
        )
        (baseline / "summary.csv").write_text("scope,HOTA\noverall,100\n")
        save_json(
            baseline / "status.json",
            {"status": "completed", "summary_sha256": sha256(baseline / "summary.csv")},
        )
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(discover_baseline([fixture.root], checkpoint), baseline)
        source = Path(REFERENCE.__file__).parents[1]
        protocol = prepare(
            output,
            baseline,
            source,
            checkpoint,
            0,
            ["baseline", "off", "dual"],
            "sequence",
        )
        for variant in protocol["variants"]:
            folder = output / variant / "val"
            worker = folder / "inference_protocol.json"
            save_json(
                worker,
                dict(
                    experiment_sha256=sha256(output / "protocol.json"),
                    **{
                        k: "synthetic"
                        for k in (
                            "torch",
                            "torchvision",
                            "numpy",
                            "opencv",
                            "cuda",
                            "gpu",
                            "runtime",
                        )
                    },
                ),
            )
            for seq in protocol["dataset"]["sequences"]:
                pred = folder / "predictions/motip/data" / f"{seq['sequence']}.txt"
                pred.parent.mkdir(parents=True, exist_ok=True)
                pred.write_text(
                    "".join(
                        f"{frame},{int(variant == 'dual' and frame > 6)},2,3,10,20,0.9,-1,-1,-1\n"
                        for frame in range(1, 13)
                    )
                )
                save_json(
                    folder / "completed" / f"{seq['sequence']}.json",
                    {
                        "frames": 12,
                        "seconds": 1,
                        "calibrated_rows": 0,
                        "protocol_sha256": sha256(worker),
                        "prediction_sha256": sha256(pred),
                    },
                )
        with contextlib.redirect_stdout(io.StringIO()):
            evaluate(output)
        import csv

        with (output / "summary.csv").open() as stream:
            rows = [row for row in csv.DictReader(stream) if row["scope"] == "overall"]
        overall = {row["variant"]: row for row in rows}
        self.assertEqual(int(overall["baseline"]["IDSW"]), 0)
        self.assertEqual(int(overall["dual"]["IDSW"]), 2)
        self.assertLess(
            float(overall["dual"]["IDF1"]), float(overall["baseline"]["IDF1"])
        )
        pred.write_text(pred.read_text().replace("0.9", "0.8"))
        with (
            self.assertRaisesRegex(ValueError, "Incomplete/modified"),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            evaluate(output)
        self.assertEqual(
            json.loads((output / "status.json").read_text())["status"], "failed"
        )
        (baseline / "summary.csv").write_text("modified")
        with self.assertRaisesRegex(ValueError, "No completed full"):
            discover_baseline([fixture.root], checkpoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()
    REFERENCE = load_reference(args.source_root.expanduser().resolve())
    unittest.main(argv=[sys.argv[0]])
