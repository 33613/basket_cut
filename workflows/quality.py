"""Quality and identity refinement orchestration (no model dependencies)."""

from pipeline.identity.resolution import ResolutionOptions, resolve_identities
from pipeline.tracking.quality import QualityOptions, check_track_quality


def run_track_quality(options: QualityOptions) -> dict:
    return check_track_quality(options)


def run_identity_resolution(options: ResolutionOptions) -> dict:
    return resolve_identities(options)
