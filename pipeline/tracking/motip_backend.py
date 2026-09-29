"""Thin adapter around the upstream MOTIP runtime tracker.

The adapter deliberately lives outside ``MOTIP/``.  It translates ordinary
OpenCV frames into MOTIP inputs and translates MOTIP tensors back into plain
Python dictionaries.  Nothing downstream needs to import MOTIP internals.
"""

from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import torch
from torchvision.transforms import functional as transform_functional


@dataclass(frozen=True)
class MotipRuntimeConfig:
    assignment_protocol: str = "object-max"
    miss_tolerance: int = 60
    det_thresh: float = 0.3
    newborn_thresh: float = 0.6
    id_thresh: float = 0.2
    area_thresh: int = 0
    use_sigmoid: bool = False
    only_detr: bool = False
    max_shorter: int = 800
    max_longer: int = 1440
    fp16: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MotipBackend:
    """Load one MOTIP model and create independent per-video trackers."""

    def __init__(
        self,
        *,
        motip_root: str | Path,
        config_path: str | Path,
        checkpoint_path: str | Path,
        runtime_config: MotipRuntimeConfig,
        device: str = "cuda:0",
    ) -> None:
        self.motip_root = Path(motip_root).expanduser().resolve()
        self.config_path = self._resolve_under_root(config_path)
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        self.runtime_config = runtime_config
        self.device = torch.device(device)

        if not self.motip_root.is_dir():
            raise FileNotFoundError(f"MOTIP root does not exist: {self.motip_root}")
        if not self.config_path.is_file():
            raise FileNotFoundError(f"MOTIP config does not exist: {self.config_path}")
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"MOTIP checkpoint does not exist: {self.checkpoint_path}"
            )
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")

        root_text = str(self.motip_root)
        if root_text not in sys.path:
            sys.path.insert(0, root_text)

        # MOTIP's inherited YAML paths are relative to its repository root.
        previous_cwd = Path.cwd()
        try:
            os.chdir(self.motip_root)
            from configs.util import load_super_config
            from models.misc import load_checkpoint
            from models.motip import build as build_model
            from utils.misc import yaml_to_dict

            config = yaml_to_dict(str(self.config_path))
            config = load_super_config(config, config.get("SUPER_CONFIG_PATH"))
            model, _ = build_model(config)
            load_checkpoint(model, str(self.checkpoint_path))
        finally:
            os.chdir(previous_cwd)

        self.model = model.eval().to(self.device)
        self.dtype = torch.float16 if runtime_config.fp16 else torch.float32
        if self.dtype == torch.float16:
            self.model.half()

        from models.runtime_tracker import RuntimeTracker
        from utils.nested_tensor import nested_tensor_from_tensor_list

        self._runtime_tracker_class = RuntimeTracker
        self._nested_tensor_from_tensor_list = nested_tensor_from_tensor_list

    def _resolve_under_root(self, value: str | Path) -> Path:
        path = Path(value).expanduser()
        return path.resolve() if path.is_absolute() else (self.motip_root / path).resolve()

    def create_tracker(self, *, height: int, width: int):
        return self._runtime_tracker_class(
            model=self.model,
            sequence_hw=(height, width),
            use_sigmoid=self.runtime_config.use_sigmoid,
            assignment_protocol=self.runtime_config.assignment_protocol,
            miss_tolerance=self.runtime_config.miss_tolerance,
            det_thresh=self.runtime_config.det_thresh,
            newborn_thresh=self.runtime_config.newborn_thresh,
            id_thresh=self.runtime_config.id_thresh,
            area_thresh=self.runtime_config.area_thresh,
            only_detr=self.runtime_config.only_detr,
            dtype=self.dtype,
        )

    @torch.inference_mode()
    def infer_frame(self, tracker: Any, frame_bgr: Any) -> list[dict[str, Any]]:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        tensor = transform_functional.to_tensor(frame_rgb)
        tensor = transform_functional.resize(
            tensor,
            size=self.runtime_config.max_shorter,
            max_size=self.runtime_config.max_longer,
            antialias=True,
        )
        tensor = transform_functional.normalize(
            tensor,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        tensor = tensor.to(device=self.device, dtype=self.dtype)
        nested = self._nested_tensor_from_tensor_list([tensor])

        tracker.update(nested)
        raw = tracker.get_track_results()
        boxes = raw["bbox"].detach().float().cpu().tolist()
        scores = raw["score"].detach().float().cpu().tolist()
        categories = raw["category"].detach().cpu().tolist()
        track_ids = raw["id"].detach().cpu().tolist()

        results: list[dict[str, Any]] = []
        for bbox_xywh, score, category, track_id in zip(
            boxes, scores, categories, track_ids
        ):
            x, y, width, height = (float(value) for value in bbox_xywh)
            results.append(
                {
                    "bbox_xyxy": [x, y, x + width, y + height],
                    "det_score": float(score),
                    "category_id": int(category),
                    "track_id": int(track_id),
                }
            )
        return results

