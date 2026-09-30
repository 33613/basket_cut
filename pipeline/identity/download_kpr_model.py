"""List and download the project-supported KPR checkpoint without a GPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pipeline.identity.inspect_kpr_checkpoint import PUBLISHED_SHA256, sha256_file


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", choices=sorted(MODEL_SPECS), default="multidataset-sports"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("/root/autodl-tmp/models/kpr")
    )
    parser.add_argument("--list", action="store_true", help="Print model metadata only")
    parser.add_argument("--force-download", action="store_true")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    spec = dict(MODEL_SPECS[args.model])
    if args.list:
        result = {"model": args.model, **spec}
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return result

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required: pip install -U huggingface_hub"
        ) from exc

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = Path(
        hf_hub_download(
            repo_id=spec["repo_id"],
            filename=spec["filename"],
            local_dir=output_dir,
            force_download=bool(args.force_download),
        )
    ).resolve()
    actual_sha256 = sha256_file(path)
    if actual_sha256 != spec["sha256"]:
        raise RuntimeError(
            "Downloaded checkpoint SHA-256 mismatch; refusing to use it. "
            f"expected={spec['sha256']}, actual={actual_sha256}, path={path}"
        )
    result = {
        "model": args.model,
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": actual_sha256,
        "verified": True,
        "supports_prompt_modes": spec["supports_prompt_modes"],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
