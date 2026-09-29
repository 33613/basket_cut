"""Thin adapter around the upstream KPR feature-extraction API.

The upstream repository is intentionally imported at runtime so this project
does not copy or modify KPR internals.  Keep KPR in its own Conda environment;
MOTIP and KPR do not need compatible PyTorch versions because they exchange
data through ``tracks.jsonl``.
"""

from __future__ import annotations

import copy
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator, Sequence


@contextmanager
def working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


class KPRBackend:
    """Load KPR once and expose batched feature and distance operations."""

    def __init__(
        self,
        *,
        kpr_root: Path,
        config_path: Path,
        checkpoint_path: Path,
        use_gpu: bool = True,
        verbose: bool = False,
    ) -> None:
        self.kpr_root = kpr_root.expanduser().resolve()
        self.config_path = config_path.expanduser().resolve()
        self.checkpoint_path = checkpoint_path.expanduser().resolve()
        for path, label in (
            (self.kpr_root, "KPR root"),
            (self.config_path, "KPR config"),
            (self.checkpoint_path, "KPR checkpoint"),
        ):
            if not path.exists():
                raise FileNotFoundError(f"{label} not found: {path}")

        root_text = str(self.kpr_root)
        if root_text not in sys.path:
            sys.path.insert(0, root_text)

        try:
            import torch
            from torchreid.scripts.builder import build_config
            from torchreid.tools.feature_extractor import KPRFeatureExtractor
        except ImportError as exc:
            raise RuntimeError(
                "KPR is not installed in this environment. Activate the KPR "
                "Conda environment and run `python setup.py develop` inside "
                f"{self.kpr_root}."
            ) from exc

        if use_gpu and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for KPR but is not available")

        # KPR configuration contains relative paths, so build it from the
        # upstream repository root.  No upstream file is edited.
        with working_directory(self.kpr_root):
            # Pass the checkpoint as a command-line-style override.  Setting it
            # after build_config() would load the tensors but would skip the
            # model configuration embedded in the official checkpoint.
            config_args = SimpleNamespace(
                root=None,
                save_dir=None,
                inference_enabled=False,
                sources=None,
                targets=None,
                transforms=None,
                job_id=None,
                opts=[
                    "model.load_weights",
                    str(self.checkpoint_path),
                    "model.load_config",
                    "True",
                ],
            )
            cfg = build_config(
                args=config_args,
                config_path=str(self.config_path),
            )
            cfg.use_gpu = bool(use_gpu)
            self.extractor = KPRFeatureExtractor(cfg, verbose=verbose)

        self.cfg = cfg
        self.torch = torch

    def extract(
        self,
        samples: Sequence[dict[str, Any]],
        *,
        batch_size: int,
    ):
        """Return CPU embeddings and visibility scores for image samples."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        from torchreid.data import ImageDataset
        from torchreid.utils.tools import extract_test_embeddings

        embedding_batches = []
        visibility_batches = []
        device = self.torch.device("cuda" if self.cfg.use_gpu else "cpu")
        with self.torch.inference_mode():
            for start in range(0, len(samples), batch_size):
                batch = list(samples[start : start + batch_size])
                images = []
                prompt_masks = []
                for sample in batch:
                    prepared = ImageDataset.getitem(
                        copy.deepcopy(sample),
                        self.cfg,
                        self.extractor.keypoints_to_prompt_masks,
                        self.extractor.prompt_preprocess,
                        self.extractor.keypoints_to_target_masks,
                        self.extractor.target_preprocess,
                        self.extractor.preprocess,
                        load_masks=True,
                    )
                    images.append(prepared["image"])
                    if "prompt_masks" in prepared:
                        prompt_masks.append(prepared["prompt_masks"])

                model_args = {
                    "images": self.torch.stack(images, dim=0).to(device)
                }
                if prompt_masks:
                    if len(prompt_masks) != len(images):
                        raise RuntimeError(
                            "KPR preprocessing produced prompts for only part of a batch"
                        )
                    model_args["prompt_masks"] = self.torch.stack(
                        prompt_masks, dim=0
                    ).to(device)

                model_output = self.extractor.model(**model_args)
                embeddings, visibility, _, _ = extract_test_embeddings(
                    model_output,
                    self.cfg.model.kpr.test_embeddings,
                )
                if self.cfg.test.normalize_feature:
                    embeddings = self.torch.nn.functional.normalize(
                        embeddings, p=2, dim=-1
                    )
                embedding_batches.append(embeddings.detach().cpu())
                visibility_batches.append(visibility.detach().cpu())
        return (
            self.torch.cat(embedding_batches, dim=0),
            self.torch.cat(visibility_batches, dim=0),
        )

    def distances(
        self,
        left_embeddings,
        right_embeddings,
        left_visibility,
        right_visibility,
    ):
        """Compute KPR's visibility-weighted distance, normalized to [0, 1]."""
        from torchreid.metrics.distance import (
            compute_distance_matrix_using_bp_features,
        )

        distances, _ = compute_distance_matrix_using_bp_features(
            left_embeddings,
            right_embeddings,
            left_visibility,
            right_visibility,
            use_gpu=False,
            use_logger=False,
        )
        # The official demo divides the Euclidean result (range [0, 2]) by 2.
        return distances.detach().cpu() / 2.0
