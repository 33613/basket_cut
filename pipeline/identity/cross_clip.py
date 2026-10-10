"""Visibility-weighted KPR part distances shared by identity memory and checks."""

from __future__ import annotations


import numpy as np


def part_distances(embeddings, visibility, *, min_common_parts=2, block_size=128):
    features = np.asarray(embeddings, dtype=np.float32)
    visible = np.asarray(visibility, dtype=np.float32)
    if features.ndim != 3 or visible.shape != features.shape[:2]:
        raise ValueError("KPR features must be N x parts x dimensions with N x parts visibility")
    if min_common_parts < 1 or block_size < 1:
        raise ValueError("min_common_parts and block_size must be positive")
    if not np.isfinite(features).all() or not np.isfinite(visible).all():
        raise ValueError("KPR features/visibility contain nonfinite values")
    if (visible < 0).any() or (visible > 1.0001).any():
        raise ValueError("Visibility must be in [0, 1]")
    norms = np.linalg.norm(features, axis=-1)
    if ((visible > 0) & (np.abs(norms - 1) > 0.02)).any():
        raise ValueError("Visible KPR prototypes must be L2-normalized per part")
    count, parts, _ = features.shape
    result = np.full((count, count), np.inf, dtype=np.float32)
    squared_norm = np.square(features).sum(axis=-1)
    for start in range(0, count, block_size):
        end = min(count, start + block_size)
        numerator = np.zeros((end - start, count), dtype=np.float32)
        denominator = np.zeros_like(numerator)
        common = np.zeros_like(numerator, dtype=np.int16)
        for part in range(parts):
            squared = (squared_norm[start:end, part, None]
                       + squared_norm[None, :, part]
                       - 2 * (features[start:end, part] @ features[:, part].T))
            weights = np.sqrt(visible[start:end, part, None] * visible[None, :, part])
            numerator += np.sqrt(np.maximum(squared, 0)) * weights
            denominator += weights
            common += weights > 0
        valid = (denominator > 1e-12) & (common >= min_common_parts)
        block = result[start:end]
        block[valid] = numerator[valid] / denominator[valid] / 2
    return result
