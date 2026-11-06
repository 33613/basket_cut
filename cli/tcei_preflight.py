"""Check the pinned TCEI source with the existing MOTIP GPU environment.

This is an environment/weight compatibility probe, not a tracking benchmark.
Run in a fresh process: upstream MOTIP and TCEI share top-level module names.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

from adapters.tcei import source_revision

PACKAGES = (
    "torch",
    "torchvision",
    "numpy",
    "scipy",
    "Pillow",
    "PyYAML",
    "einops",
    "accelerate",
    "wandb",
    "tqdm",
    "opencv-python",
    "opencv-python-headless",
)


def cuda_ops(report: dict) -> None:
    import torch
    from torchvision.ops import nms

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run this probe after enabling the GPU")
    # Import torch first so its shared libraries are loaded before the extension.
    extension = importlib.import_module("MultiScaleDeformableAttention")
    from models.ops.functions.ms_deform_attn_func import (
        MSDeformAttnFunction,
        ms_deform_attn_core_pytorch,
    )

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(0)
    report["cuda"] = {
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "extension_path": extension.__file__,
        "fp32": True,
        "tf32": False,
    }
    x = torch.randn(64, 64, device="cuda")
    if not torch.isfinite(x @ x.T).all():
        raise RuntimeError("Nonfinite CUDA matmul")
    boxes = torch.tensor(
        [[0, 0, 10, 10], [1, 1, 11, 11]], device="cuda", dtype=torch.float32
    )
    keep = nms(boxes, torch.tensor([0.9, 0.8], device="cuda"), 0.5)
    if keep.tolist() != [0]:
        raise RuntimeError("torchvision CUDA NMS failed")
    # Two feature levels, two heads, four channels/head, three queries.
    shapes = torch.tensor([[2, 2], [1, 1]], dtype=torch.int64, device="cuda")
    starts = torch.tensor([0, 4], dtype=torch.int64, device="cuda")
    value = torch.rand(1, 5, 2, 4, device="cuda")
    locations = torch.rand(1, 3, 2, 2, 2, 2, device="cuda")
    weights = torch.rand(1, 3, 2, 2, 2, device="cuda")
    weights /= weights.sum((-1, -2), keepdim=True)
    with torch.no_grad():
        actual = MSDeformAttnFunction.apply(
            value, shapes, starts, locations, weights, 1
        )
        expected = ms_deform_attn_core_pytorch(value, shapes, locations, weights)
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
    report["cuda"]["deformable_attention_max_abs"] = float(
        (actual - expected).abs().max()
    )


def model_smoke(root: Path, checkpoint: Path, report: dict) -> None:
    import torch
    from configs.util import load_super_config
    from models.motip import build
    from utils.misc import yaml_to_dict
    from utils.nested_tensor import nested_tensor_from_tensor_list

    from analysis.teamtrack.dataset import sha256

    report["checkpoint"] = {"path": str(checkpoint), "sha256": sha256(checkpoint)}
    config = yaml_to_dict(str(root / "configs/sportsmot.yaml"))
    config = load_super_config(config, config.get("SUPER_CONFIG_PATH"))
    config["DEVICE"] = "cuda"
    # Full strict checkpoint restores every backbone tensor. Skip the otherwise
    # redundant ImageNet download only while constructing this probe's model.
    with patch("models.deformable_detr.backbone.is_main_process", return_value=False):
        model, _ = build(config)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    del state
    model = model.eval().cuda()
    with torch.no_grad():
        frames = nested_tensor_from_tensor_list(
            [torch.rand(3, 128, 224, device="cuda")]
        )
        detections = model(frames=frames, part="detr")
        for key in ("pred_logits", "pred_boxes", "outputs"):
            if not torch.isfinite(detections[key]).all():
                raise RuntimeError(f"Nonfinite DETR {key}")
        # A synthetic two-frame history with two distinct local identities.
        # This exercises the extra feature outputs needed by TCEI even when a
        # blank smoke-test image would produce no real person detections.
        seq = {
            "trajectory_features": torch.rand(1, 1, 2, 2, 256, device="cuda"),
            "trajectory_id_labels": torch.tensor([[[[0, 1], [0, 1]]]], device="cuda"),
            "trajectory_times": torch.tensor([[[[0, 0], [1, 1]]]], device="cuda"),
            "trajectory_masks": torch.zeros(
                1, 1, 2, 2, dtype=torch.bool, device="cuda"
            ),
            "unknown_features": torch.rand(1, 1, 1, 2, 256, device="cuda"),
            "unknown_times": torch.full(
                (1, 1, 1, 2), 2, dtype=torch.int64, device="cuda"
            ),
            "unknown_masks": torch.zeros(1, 1, 1, 2, dtype=torch.bool, device="cuda"),
        }
        seq = model(seq_info=seq, part="trajectory_modeling")
        decoded = model(seq_info=seq, part="id_decoder")
        if len(decoded) != 5:
            raise RuntimeError("Expected TCEI ID decoder with five outputs")
        for tensor in (decoded[0], decoded[3], decoded[4]):
            if not torch.isfinite(tensor).all():
                raise RuntimeError("Nonfinite ID decoder outputs")
        if tuple(decoded[4].shape) != (2, config["ID_DIM"]):
            raise RuntimeError(f"Unexpected ID features: {decoded[4].shape}")
    torch.cuda.synchronize()
    report["model"] = {
        "strict_load": True,
        "detr_outputs_shape": list(detections["outputs"].shape),
        "id_logits_shape": list(decoded[0].shape),
        "id_features_shape": list(decoded[4].shape),
        "synthetic_smoke_only": True,
        "calibration_tested": False,
        "tracking_evaluated": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--stage", choices=("environment", "model"), default="model")
    args = parser.parse_args()
    if args.stage == "model" and args.checkpoint is None:
        parser.error("--checkpoint is required for --stage model")
    root = args.source_root.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().absolute() if args.checkpoint else None
    report = {
        "status": "running",
        "python": platform.python_version(),
        "interpreter": sys.executable,
        "stage": args.stage,
        "source_root": str(root),
        "packages": {},
    }
    previous_cwd = Path.cwd()
    try:
        report["source_commit"] = source_revision(root)
        # Preserve access to this project's reporting utilities after chdir.
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        sys.path.insert(0, str(root))
        os.chdir(root)
        for name in PACKAGES:
            try:
                report["packages"][name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                report["packages"][name] = None
        print(f"Python: {report['python']} | {sys.executable}", flush=True)
        print(json.dumps(report["packages"], indent=2), flush=True)
        if sys.version_info < (3, 10):  # noqa: UP036 -- probe may use a different interpreter
            raise RuntimeError(
                "TCEI requires Python >=3.10 syntax; author recommends 3.12"
            )
        for module in (
            "cv2",
            "numpy",
            "scipy",
            "einops",
            "yaml",
            "accelerate",
            "wandb",
            "tqdm",
        ):
            importlib.import_module(module)
        subprocess.run([sys.executable, "-m", "pip", "check"], check=True)
        cuda_ops(report)
        print(
            "PASS: CUDA, NMS and Deformable Attention reference comparison", flush=True
        )
        importlib.import_module("TCEI.tcei")
        importlib.import_module("submit_and_evaluate")
        print("PASS: pinned author TCEI entry imports", flush=True)
        if args.stage == "model":
            model_smoke(root, checkpoint, report)
            print(
                "PASS: original SportsMOT checkpoint strict-load and FP32 model smoke",
                flush=True,
            )
        report["status"] = (
            "passed_environment"
            if args.stage == "environment"
            else "passed_environment_and_model_smoke"
        )
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        os.chdir(previous_cwd)
        path = args.report.expanduser().absolute()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(f"Report: {path} | {report['status']}", flush=True)


if __name__ == "__main__":
    main()
