"""Environment-driven Web application settings.

The Web process is intentionally independent from all three model Conda
environments.  It launches each CLI with that environment's Python executable.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _path_env(name: str, default: str | Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser().resolve()


@dataclass(frozen=True)
class WebSettings:
    repository_root: Path = REPOSITORY_ROOT
    data_root: Path = field(
        default_factory=lambda: _path_env(
            "BASKET_WEB_DATA_ROOT", "/root/autodl-tmp/data/basket_cut/web"
        )
    )
    output_root: Path = field(
        default_factory=lambda: _path_env(
            "BASKET_WEB_OUTPUT_ROOT", "/root/autodl-tmp/outputs/basket_cut/web"
        )
    )
    import_root: Path = field(
        default_factory=lambda: _path_env(
            "BASKET_WEB_IMPORT_ROOT", "/root/autodl-tmp/outputs/basket_cut"
        )
    )
    source_root: Path = field(
        default_factory=lambda: _path_env(
            "BASKET_WEB_SOURCE_ROOT", "/root/autodl-tmp/data/basket_cut"
        )
    )
    motip_python: Path = field(
        default_factory=lambda: _path_env(
            "BASKET_MOTIP_PYTHON", "/root/autodl-tmp/envs/motip/bin/python"
        )
    )
    kpr_python: Path = field(
        default_factory=lambda: _path_env(
            "BASKET_KPR_PYTHON", "/root/autodl-tmp/envs/kpr/bin/python"
        )
    )
    action_python: Path = field(
        default_factory=lambda: _path_env(
            "BASKET_ACTION_PYTHON", "/root/autodl-tmp/envs/mmaction2/bin/python"
        )
    )
    motip_checkpoint: Path = field(
        default_factory=lambda: _path_env(
            "BASKET_MOTIP_CHECKPOINT",
            "/root/autodl-tmp/models/motip/r50_deformable_detr_motip_sportsmot.pth",
        )
    )
    kpr_checkpoint: Path = field(
        default_factory=lambda: _path_env(
            "BASKET_KPR_CHECKPOINT",
            "/root/autodl-tmp/models/kpr/"
            "kpr_dancetrack_sportsmot_posetrack21_occludedduke_"
            "market_split0.pth.tar",
        )
    )
    action_checkpoint: Path = field(
        default_factory=lambda: _path_env(
            "BASKET_ACTION_CHECKPOINT",
            "/root/autodl-tmp/models/action/slowfast_multisports/"
            "slowfast_kinetics400-pretrained-r50_8xb16-4x16x1-8e_"
            "multisports-rgb_20230320-af666368.pth",
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
    max_upload_bytes: int = field(
        default_factory=lambda: int(
            os.environ.get("BASKET_MAX_UPLOAD_BYTES", str(4 * 1024**3))
        )
    )

    @property
    def frontend_root(self) -> Path:
        return self.repository_root / "web/frontend"

    def ensure_directories(self) -> None:
        self.data_root.mkdir(parents=True, exist_ok=True)
        self.output_root.mkdir(parents=True, exist_ok=True)

    def readiness(self) -> dict[str, Any]:
        paths = {
            "repository_root": self.repository_root,
            "kpr_root": self.repository_root / "KPR",
            "data_root": self.data_root,
            "output_root": self.output_root,
            "motip_python": self.motip_python,
            "kpr_python": self.kpr_python,
            "action_python": self.action_python,
            "motip_checkpoint": self.motip_checkpoint,
            "kpr_checkpoint": self.kpr_checkpoint,
            "action_checkpoint": self.action_checkpoint,
            "action_config": self.action_config,
            "action_label_map": self.action_label_map,
        }
        return {
            "ready": all(path.exists() for path in paths.values()),
            "paths": {
                name: {"path": str(path), "exists": path.exists()}
                for name, path in paths.items()
            },
        }

    def public_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values.pop("max_upload_bytes", None)
        values["frontend_root"] = str(self.frontend_root)
        return {key: str(value) for key, value in values.items()}
