"""Selectively download SHOT samples from Hugging Face.

The SHOT repository contains videos, decoded frames, pose, gaze, head-pose,
labels and tracks.  For MOT evaluation we only need an MP4 and the aligned
``track_with_gt.txt`` file, so this downloader uses exact allow-patterns and
never downloads the complete dataset unless ``--profile full`` is requested.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from contracts.paths import runtime_root

DEFAULT_REPO_ID = "muyu111/basketball"
DEFAULT_REVISION = "shotdatasets"
DEFAULT_SAMPLE = "view1/Drive_Dunk/ATLvsNJ-10-view1-3"


@dataclass(frozen=True)
class ShotDownloadOptions:
    repo_id: str = DEFAULT_REPO_ID
    revision: str = DEFAULT_REVISION
    output_dir: Path = runtime_root() / "data/SHOT"
    samples: tuple[str, ...] = ()
    max_samples: int | None = None
    profile: str = "tracking-eval"
    dry_run: bool = False


def normalize_sample_path(value: str) -> str:
    normalized = value.strip().strip("/")
    if len(Path(normalized).parts) < 3:
        raise ValueError(
            f"A SHOT sample path must look like view/tactic/sample-name, got {value!r}"
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


def download_shot(options: ShotDownloadOptions) -> dict:
    if options.max_samples is not None and options.max_samples < 1:
        raise ValueError("--max-samples must be at least 1")

    if options.samples:
        samples = unique(normalize_sample_path(value) for value in options.samples)
    else:
        samples = [DEFAULT_SAMPLE]
    if options.max_samples is not None:
        samples = samples[: options.max_samples]

    patterns = unique(
        pattern
        for sample in samples
        for pattern in patterns_for_sample(sample, options.profile)
    )
    plan = {
        "repo_id": options.repo_id,
        "revision": options.revision,
        "profile": options.profile,
        "sample_count": len(samples),
        "samples": samples,
        "allow_patterns": patterns,
        "output_dir": str(options.output_dir.expanduser().resolve()),
    }
    if options.dry_run:
        return {**plan, "download_complete": False, "dry_run": True}

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required: pip install -U huggingface_hub"
        ) from exc

    output_dir = options.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=options.repo_id,
        repo_type="dataset",
        revision=options.revision,
        allow_patterns=patterns,
        local_dir=output_dir,
    )
    file_count, byte_count = selected_size(output_dir, samples)
    return {
        **plan,
        "download_complete": True,
        "dry_run": False,
        "selected_file_count": file_count,
        "selected_bytes": byte_count,
        "selected_gib": round(byte_count / 1024**3, 3),
        "output_dir": str(output_dir),
    }
