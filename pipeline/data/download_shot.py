"""Selectively download SHOT samples from Hugging Face.

The SHOT repository contains videos, decoded frames, pose, gaze, head-pose,
labels and tracks.  For MOT evaluation we only need an MP4 and the aligned
``track_with_gt.txt`` file, so this downloader uses exact allow-patterns and
never downloads the complete dataset unless ``--profile full`` is requested.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


DEFAULT_REPO_ID = "muyu111/basketball"
DEFAULT_REVISION = "shotdatasets"
DEFAULT_SAMPLE = "view1/Drive_Dunk/ATLvsNJ-10-view1-3"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Selectively download SHOT basketball samples"
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/root/autodl-tmp/data/basket_cut/SHOT"),
    )
    parser.add_argument(
        "--sample",
        action="append",
        help=(
            "Exact sample directory; repeat to download multiple samples. "
            f"Default: {DEFAULT_SAMPLE}"
        ),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Keep only the first N explicit --sample values",
    )
    parser.add_argument(
        "--profile",
        choices=("video", "tracking-eval", "full"),
        default="tracking-eval",
        help=(
            "video=MP4 only; tracking-eval=MP4 plus aligned reference tracks; "
            "full=every file in the selected sample"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print selected sample paths and allow-patterns without downloading",
    )
    return parser


def normalize_sample_path(value: str) -> str:
    normalized = value.strip().strip("/")
    if len(Path(normalized).parts) < 3:
        raise ValueError(
            "A SHOT sample path must look like view/tactic/sample-name, got "
            f"{value!r}"
        )
    return normalized


def patterns_for_sample(sample: str, profile: str) -> list[str]:
    sample_name = Path(sample).name
    if profile == "video":
        return [f"{sample}/{sample_name}.mp4"]
    if profile == "tracking-eval":
        return [
            f"{sample}/{sample_name}.mp4",
            f"{sample}/{sample_name}-track_with_gt.txt",
        ]
    if profile == "full":
        return [f"{sample}/**"]
    raise ValueError(f"Unknown download profile: {profile}")


def unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))


def selected_size(root: Path, samples: Iterable[str]) -> tuple[int, int]:
    files = [
        item
        for sample in samples
        for item in (root / sample).rglob("*")
        if item.is_file()
    ]
    return len(files), sum(item.stat().st_size for item in files)


def run(args: argparse.Namespace) -> None:
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("--max-samples must be at least 1")

    if args.sample:
        samples = unique(normalize_sample_path(value) for value in args.sample)
    else:
        samples = [DEFAULT_SAMPLE]
    if args.max_samples is not None:
        samples = samples[: args.max_samples]

    patterns = unique(
        pattern
        for sample in samples
        for pattern in patterns_for_sample(sample, args.profile)
    )
    plan = {
        "repo_id": args.repo_id,
        "revision": args.revision,
        "profile": args.profile,
        "sample_count": len(samples),
        "samples": samples,
        "allow_patterns": patterns,
        "output_dir": str(args.output_dir.expanduser().resolve()),
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    if args.dry_run:
        return

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required: pip install -U huggingface_hub"
        ) from exc

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        allow_patterns=patterns,
        local_dir=output_dir,
    )
    file_count, byte_count = selected_size(output_dir, samples)
    print(
        json.dumps(
            {
                "download_complete": True,
                "selected_file_count": file_count,
                "selected_bytes": byte_count,
                "selected_gib": round(byte_count / 1024**3, 3),
                "output_dir": str(output_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
