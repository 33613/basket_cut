"""Environment-driven Web application settings.

The Web process is intentionally independent from all three model Conda
environments.  It launches each CLI with that environment's Python executable.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from contracts.paths import runtime_root
from contracts.execution import ExecutionSettings, _path_env

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class WebSettings(ExecutionSettings):
    repository_root: Path = REPOSITORY_ROOT
    data_root: Path = field(
        default_factory=lambda: _path_env(
            "BASKET_WEB_DATA_ROOT", runtime_root() / "data/web"
        )
    )
    output_root: Path = field(
        default_factory=lambda: _path_env(
            "BASKET_WEB_OUTPUT_ROOT", runtime_root() / "outputs/web"
        )
    )
    import_root: Path = field(
        default_factory=lambda: _path_env(
            "BASKET_WEB_IMPORT_ROOT", runtime_root() / "outputs"
        )
    )
    source_root: Path = field(
        default_factory=lambda: _path_env(
            "BASKET_WEB_SOURCE_ROOT", runtime_root() / "data"
        )
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
            "qwen_ready": self.qwen_python.is_file() and (self.qwen_model_dir / "config.json").is_file()
                          and any(self.qwen_model_dir.glob("*.safetensors")),
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
