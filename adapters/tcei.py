"""Load pinned TCEI math without importing its colliding models/utils packages."""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

TCEI_COMMIT = "145d1b8431398156f8d9f854430e306fdee39eaa"


def source_revision(root: Path) -> str:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    if commit != TCEI_COMMIT:
        raise ValueError(f"Expected TCEI {TCEI_COMMIT}; got {commit}")
    subprocess.run(["git", "diff", "--exit-code", "HEAD", "--"], cwd=root, check=True)
    return commit


def load_reference(root: Path):
    source_revision(root)
    spec = importlib.util.spec_from_file_location(
        "basket_tcei_reference", root / "TCEI/tcei.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
