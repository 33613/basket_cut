"""Portable defaults; deployment-specific settings belong outside Git."""

import os
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
KPR_FILENAME = "kpr_dancetrack_sportsmot_posetrack21_occludedduke_market_split0.pth.tar"


def runtime_root() -> Path:
    return (
        Path(os.environ.get("BASKET_RUNTIME_ROOT", str(REPOSITORY_ROOT / "runtime")))
        .expanduser()
        .resolve()
    )


def model_path(module: str, filename: str) -> Path:
    return runtime_root() / "models" / module / filename
