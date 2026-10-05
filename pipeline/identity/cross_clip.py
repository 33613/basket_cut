"""CPU-only, conservative grouping of cached KPR track prototypes.

Distances follow KPR's continuous-visibility part distance (Euclidean / 2).
Groups use complete linkage, not a transitive nearest-neighbour chain. Separate
local archives in the same clip are cannot-links; local resolution owns those.
"""

from __future__ import annotations

from collections import defaultdict
import math

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


def group_nodes(nodes, track_distances, *, max_distance, assignments=None,
                blocked_node_ids=(), neighbours=5):
    """Return candidate groups and auditable evidence; no identity probabilities."""
    if max_distance is not None and (not math.isfinite(max_distance) or max_distance < 0):
        raise ValueError("max_distance must be finite and nonnegative")
    nodes = sorted(nodes, key=lambda item: item["node_id"])
    ids = {node["node_id"] for node in nodes}
    assignments = dict(assignments or {})
    blocked = set(blocked_node_ids)
    if set(assignments) - ids or blocked - ids:
        raise ValueError("Review references unknown local archive nodes")
    if any(not isinstance(label, str) or not label.strip() or len(label) > 120
           for label in assignments.values()):
        raise ValueError("Manual labels must be nonempty strings of at most 120 characters")
    assignments = {key: value.strip() for key, value in assignments.items()}
    count = len(nodes)
    distances = np.full((count, count), np.inf, dtype=np.float32)
    for a in range(count):
        for b in range(a + 1, count):
            left, right = nodes[a]["feature_indices"], nodes[b]["feature_indices"]
            if left and right:
                # Every constituent track must agree, including multi-track local archives.
                distances[a, b] = distances[b, a] = np.max(track_distances[np.ix_(left, right)])
    groups = {index: {index} for index in range(count)}
    owner = list(range(count))
    merges = []

    def labels(group):
        return {assignments[nodes[i]["node_id"]] for i in group
                if nodes[i]["node_id"] in assignments}

    def held(index):
        return bool(nodes[index].get("hold_reasons")) or nodes[index]["node_id"] in blocked

    def number(index):
        jersey = nodes[index].get("jersey") or {}
        return jersey.get("number") if jersey.get("status") == "candidate_consensus" else None

    def numbers(group):
        return {number(i) for i in group if number(i) is not None}

    def combine(a, b, *, manual=False):
        first, second = owner[a], owner[b]
        if first == second:
            return False
        left, right = groups[first], groups[second]
        combined = left | right
        clips = [nodes[i]["clip_name"] for i in combined]
        conflict = (len(clips) != len(set(clips)) or any(held(i) for i in combined)
                    or len(labels(combined)) > 1 or len(numbers(combined)) > 1)
        maximum = float(np.max(distances[np.ix_(list(left), list(right))]))
        if conflict or (not manual and (max_distance is None or maximum > max_distance)):
            if manual:
                raise ValueError("Manual group conflicts with same-clip, held/mixed archives or jersey numbers; inspect local tracks first")
            return False
        groups[first] = combined
        del groups[second]
        for index in right:
            owner[index] = first
        merges.append({"node_a": nodes[a]["node_id"], "node_b": nodes[b]["node_id"],
                       "complete_link_distance": maximum if math.isfinite(maximum) else None,
                       "reason": "manual_label" if manual else "kpr_complete_link"})
        return True

    manual_groups = defaultdict(list)
    for index, node in enumerate(nodes):
        if node["node_id"] in assignments:
            manual_groups[assignments[node["node_id"]]].append(index)
    for label in sorted(manual_groups):
        members = manual_groups[label]
        for index in members[1:]:
            combine(members[0], index, manual=True)
    if max_distance is not None:
        candidates = [(float(distances[a, b]), a, b) for a in range(count)
                      for b in range(a + 1, count) if distances[a, b] <= max_distance]
        for _, a, b in sorted(candidates):
            combine(a, b)

    # Bounded nearest candidates plus accepted merge edges, not an unbounded N² JSON dump.
    pairs = set()
    for a in range(count):
        nearest = [b for b in np.argsort(distances[a])
                   if b != a and math.isfinite(float(distances[a, b]))][:neighbours]
        pairs.update((min(a, int(b)), max(a, int(b))) for b in nearest)
    by_id = {node["node_id"]: index for index, node in enumerate(nodes)}
    pairs.update(tuple(sorted((by_id[m["node_a"]], by_id[m["node_b"]]))) for m in merges)
    evidence = []
    for a, b in sorted(pairs):
        reasons = []
        if nodes[a]["clip_name"] == nodes[b]["clip_name"]:
            reasons.append("different_local_archives_in_same_clip")
        if held(a) or held(b):
            reasons.append("archive_held_for_review")
        if number(a) is not None and number(b) is not None and number(a) != number(b):
            reasons.append("different_reliable_jersey_numbers")
        if assignments.get(nodes[a]["node_id"]) and assignments.get(nodes[b]["node_id"]):
            if assignments[nodes[a]["node_id"]] != assignments[nodes[b]["node_id"]]:
                reasons.append("different_manual_labels")
        distance = float(distances[a, b])
        if max_distance is None:
            reasons.append("automatic_merge_disabled")
        elif distance > max_distance:
            reasons.append("distance_above_threshold")
        if not reasons and owner[a] != owner[b]:
            reasons.append("complete_link_or_cluster_conflict")
        evidence.append({"node_a": nodes[a]["node_id"], "node_b": nodes[b]["node_id"],
                         "distance": distance if math.isfinite(distance) else None,
                         "same_global_person": owner[a] == owner[b], "auto_block_reasons": reasons})
    result = []
    for members in groups.values():
        group = [nodes[index] for index in sorted(members)]
        names = labels(members)
        all_manual = all(node["node_id"] in assignments for node in group)
        result.append({"members": group, "identity_label": next(iter(names)) if names else None,
                       "status": "needs_review" if any(held(index) for index in members) else
                       "manual_grouping" if all_manual else "candidate",
                       "manual_member_count": sum(node["node_id"] in assignments for node in group)})
    return sorted(result, key=lambda group: group["members"][0]["node_id"]), evidence, merges
