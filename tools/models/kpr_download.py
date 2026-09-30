"""List and download the project-supported KPR checkpoint without a GPU."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tools.models.kpr_checkpoint import PUBLISHED_SHA256, sha256_file


MODEL_SPECS: dict[str, dict[str, Any]] = {
    "multidataset-sports": {
        "repo_id": "trackinglaboratory/keypoint_promptable_reid",
        "filename": (
            "kpr_dancetrack_sportsmot_posetrack21_occludedduke_"
            "market_split0.pth.tar"
        ),
        "sha256": PUBLISHED_SHA256,
        "training_sources": [
            "DanceTrack",
            "SportsMOT",
            "PoseTrack21",
            "OccludedDuke",
            "Market1501",
        ],
        "recommended_for": "cross-domain sports video identity archiving",
        "supports_prompt_modes": ["none", "keypoints"],
    }
}


@dataclass(frozen=True)
class KPRDownloadOptions:
    model: str = "multidataset-sports"
    output_dir: Path = Path("/root/autodl-tmp/models/kpr")
    list_only: bool = False
    force_download: bool = False


def download_kpr_model(options: KPRDownloadOptions) -> dict[str, Any]:
    if options.model not in MODEL_SPECS:
        raise ValueError(
            f"Unknown KPR model {options.model!r}; "
            f"expected one of {sorted(MODEL_SPECS)}"
        )
    spec = dict(MODEL_SPECS[options.model])
    if options.list_only:
        return {"model": options.model, **spec}

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required: pip install -U huggingface_hub"
        ) from exc

    output_dir = options.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = Path(
        hf_hub_download(
            repo_id=spec["repo_id"],
            filename=spec["filename"],
            local_dir=output_dir,
            force_download=bool(options.force_download),
        )
    ).resolve()
    actual_sha256 = sha256_file(path)
    if actual_sha256 != spec["sha256"]:
        raise RuntimeError(
            "Downloaded checkpoint SHA-256 mismatch; refusing to use it. "
            f"expected={spec['sha256']}, actual={actual_sha256}, path={path}"
        )
    result = {
        "model": options.model,
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": actual_sha256,
        "verified": True,
        "supports_prompt_modes": spec["supports_prompt_modes"],
    }
    return result
