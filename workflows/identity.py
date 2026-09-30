"""Identity archive application workflow."""

from __future__ import annotations

from typing import Any

from pipeline.identity.archive import (
    IdentityArchiveOptions,
    build_identity_archive,
)


def run_identity_archive(options: IdentityArchiveOptions) -> dict[str, Any]:
    return build_identity_archive(options)
