"""Portable subprocess settings shared by batch workflows and the Web UI."""

import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from contracts.paths import KPR_FILENAME, model_path, runtime_root

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

def _path_env(name: str, default: str | Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser().resolve()


def _python_env(name: str) -> Path:
    value = os.environ.get(name, sys.executable)
    # A venv Python is often a symlink. Resolving it selects the base interpreter
    # and silently drops the environment's packages (notably cv2 / torch).
    return Path(os.path.abspath(Path(shutil.which(value) or value).expanduser()))


@dataclass(frozen=True)
class ExecutionSettings:
    repository_root: Path = REPOSITORY_ROOT
    review_python: Path = field(default_factory=lambda: _python_env("BASKET_REVIEW_PYTHON") if os.environ.get("BASKET_REVIEW_PYTHON") else _python_env("BASKET_MOTIP_PYTHON"))
    qwen_python: Path = field(default_factory=lambda: _python_env("BASKET_QWEN_PYTHON"))
    qwen_model_dir: Path = field(default_factory=lambda: _path_env("BASKET_QWEN_MODEL_DIR", model_path("qwen", "Qwen2.5-VL-3B-Instruct")))
    motip_python: Path = field(
        default_factory=lambda: _python_env("BASKET_MOTIP_PYTHON")
    )
    kpr_python: Path = field(default_factory=lambda: _python_env("BASKET_KPR_PYTHON"))
    action_python: Path = field(
        default_factory=lambda: _python_env("BASKET_ACTION_PYTHON")
    )
    motip_checkpoint: Path = field(
        default_factory=lambda: _path_env(
            "BASKET_MOTIP_CHECKPOINT",
            model_path("motip", "r50_deformable_detr_motip_sportsmot.pth"),
        )
    )
    kpr_checkpoint: Path = field(
        default_factory=lambda: _path_env(
            "BASKET_KPR_CHECKPOINT",
            model_path("kpr", KPR_FILENAME),
        )
    )
    action_checkpoint: Path = field(
        default_factory=lambda: _path_env(
            "BASKET_ACTION_CHECKPOINT",
            model_path("action", "slowfast_multisports.pth"),
        )
    )
    action_config: Path = field(
        default_factory=lambda: _path_env(
            "BASKET_ACTION_CONFIG",
            REPOSITORY_ROOT / "MMAction2/configs/detection/slowfast/"
            "slowfast_kinetics400-pretrained-r50_8xb16-4x16x1-8e_"
            "multisports-rgb.py",
        )
    )
    action_label_map: Path = field(
        default_factory=lambda: _path_env(
            "BASKET_ACTION_LABEL_MAP",
            REPOSITORY_ROOT / "MMAction2/tools/data/multisports/label_map.txt",
        )
    )
    ffmpeg: str = field(
        default_factory=lambda: os.environ.get("BASKET_FFMPEG", "ffmpeg")
    )
