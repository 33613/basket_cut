"""Conservative photo/track memory planning. No model training or file deletion.

Numbers read once are weak evidence, not identity labels. Groups require
complete-link compatibility, independent support and a nearest-rival margin.
All thresholds below are engineering defaults to validate, not probabilities.
"""
from dataclasses import asdict, dataclass
import math

import numpy as np

from pipeline.identity.cross_clip import part_distances


@dataclass(frozen=True)
class MemoryPolicy:
    version: int = 2
    strong_distance: float = .12
    supported_distance: float = .2
    novelty_distance: float = .5
    min_margin: float = .05
    gallery_limit: int = 8
    min_common_parts: int = 2
    min_quality: float = .45
    independent_gap_s: float = 1.
    duplicate_iou: float = .85
    overlap_fraction: float = .45
    max_observations: int = 2000
    max_templates: int = 4096

    def validate(self):
        values = asdict(self)
        if not all(math.isfinite(v) for v in values.values()):
            raise ValueError("Nonfinite memory policy")
        if (self.version != 2 or not 0 <= self.strong_distance <= self.supported_distance
                or self.novelty_distance <= self.supported_distance or self.min_margin < 0
                or not 0 <= self.min_quality <= 1 or self.independent_gap_s <= 0
                or not .5 <= self.duplicate_iou <= 1 or not 0 < self.overlap_fraction <= 1
                or any(type(v) is not int or v < 1 for v in
                       (self.gallery_limit, self.min_common_parts, self.max_observations, self.max_templates))):
            raise ValueError("Invalid gallery memory policy")
        return self


def number_hint(node):
    jersey = node.get("jersey") or {}
    number = jersey.get("number")
    if number is None:
        candidates = set(jersey.get("candidate_numbers", []))
        number = next(iter(candidates)) if len(candidates) == 1 else None
    return number if isinstance(number, str) and number.isascii() and number.isdigit() and 1 <= len(number) <= 2 else None


def incompatible(a, b):
    # A one-frame disagreement can only block a speculative auto merge. It
    # never establishes two verified players or changes original OCR evidence.
    na, nb = number_hint(a), number_hint(b)
    ta, tb = a.get("team_id"), b.get("team_id")
    if ta is not None and tb is not None and ta != tb:
        return "different_teams"
    if na is not None and nb is not None and na != nb:
        return "different_number_candidates"
    if a["clip_name"] == b["clip_name"]:
        x, y = a.get("intervals", []), b.get("intervals", [])
        if not x or not y or any(i[0] < j[1] and j[0] < i[1] for i in x for j in y):
            return "simultaneous_observations"
    # Separate exported clips can overlap on the original video's clock.
    if (a.get("source_video_id") and a.get("source_video_id") == b.get("source_video_id")
            and a.get("evidence_kind") != "bootstrap_photo" and b.get("evidence_kind") != "bootstrap_photo"):
        sa, sb = a.get("source_clip_start_s"), b.get("source_clip_start_s")
        if sa is not None and sb is not None and any(
                sa+x[0] < sb+y[1] and sb+y[0] < sa+x[1]
                for x in a.get("intervals", []) for y in b.get("intervals", [])):
            return "overlapping_source_tracks"
    return None


def box_overlap(a, b):
    inter = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    aa, ab = (a[2]-a[0])*(a[3]-a[1]), (b[2]-b[0])*(b[3]-b[1])
    return inter/max(aa+ab-inter, 1), inter/max(min(aa, ab), 1)


def select_templates(items, distance, limit):
    """Quality first; prefer new supplied views and nonredundant appearances."""
    chosen, remaining = [], list(items)
    while remaining and len(chosen) < limit:
        def rank(i):
            nearest = min((distance[i, j] for j in chosen), default=0.)
            diversity = min(.3, nearest) if math.isfinite(nearest) else 0.
            view = items[i].get("view")
            novel_view = bool(view and view != "unknown" and all(items[j].get("view") != view for j in chosen))
            return (items[i]["quality"] * (1 + diversity + .15*novel_view), -i)
        # Item keys are local contiguous indices, preserving source order.
        best = max(remaining, key=rank)
        chosen.append(best)
        remaining.remove(best)
        remaining = [i for i in remaining if not (
            items[i].get("source_frame") == items[best].get("source_frame")
            or (distance[i, best] < .025 and items[i].get("view") == items[best].get("view")))]
    return chosen


def plan_memory(records, evidence, policy, *, incremental=False):
    """Return a pure plan; registry owns IDs, transactions, history and undo."""
    policy.validate()
    if len(records) > policy.max_observations:
        raise ValueError("Memory planning budget exceeded; review/prune or explicitly raise max_observations")
    nodes = [r["node"] for r in records]
    count = len(nodes)
    states = [{"status": "pending", "reasons": [], "quality": 0., "templates": []} for _ in nodes]
    banks, flat, visibility = [], [], []
    for i, (record, node) in enumerate(zip(records, nodes)):
        info = evidence.get(node["node_id"], {})
        bank = info.get("templates")
        if bank is None:
            bank = [{"key": f"prototype-{k}", "feature": f, "visibility": v,
                     "quality": float(node.get("cover", {}).get("archive_quality_score", .7)),
                     "source_frame": f'{node["node_id"]}:{k}', "kind": "track_prototype"}
                    for k, (f, v) in enumerate(zip(record["features"], record["visibility"]))]
        keep = [t for t in bank if t["quality"] >= policy.min_quality
                and np.count_nonzero(np.asarray(t["visibility"]) > 0) >= policy.min_common_parts]
        states[i]["quality"] = max((t["quality"] for t in keep), default=0.)
        # During normal incremental registration established groups expose
        # only their selected templates. The complete immutable archive is
        # reconsidered only by explicit/periodic full reprocessing.
        anchor = incremental and record.get("established")
        if anchor:
            keep = [t for t in keep if t["key"] in record["active_template_keys"]]
        banks.append(keep)
        reasons = list(info.get("hold_reasons", []))
        reasons += [r for r in node.get("hold_reasons", []) if r != "ambiguous_or_uncomparable_identity"]
        if not keep and not anchor:
            reasons.append("no_usable_quality_templates")
        if record.get("rejected"):
            reasons.append("manually_rejected")
        if record.get("non_player"):
            reasons.append("non_player")
        states[i]["reasons"] = sorted(set(reasons))
        if (not keep and not anchor) or record.get("rejected") or record.get("non_player"):
            states[i]["status"] = "rejected"
        idx = []
        for template in keep:
            idx.append(len(flat)); flat.append(template["feature"]); visibility.append(template["visibility"])
        states[i]["feature_indices"] = idx
    if len(flat) > policy.max_templates:
        raise ValueError("Memory template budget exceeded; prune/review observations or explicitly raise max_templates")
    distances = part_distances(np.asarray(flat), np.asarray(visibility), min_common_parts=policy.min_common_parts) if flat else np.empty((0, 0))
    pair = np.full((count, count), np.inf, dtype=np.float32)
    for i in range(count):
        for j in range(i):
            a, b = states[i]["feature_indices"], states[j]["feature_indices"]
            if a and b:
                block = distances[np.ix_(a, b)]
                # Two supporting frames when both tracks have them; a lone
                # matching outlier must not certify a whole trajectory.
                support = min(2, len(a), len(b))
                score = max(float(np.mean(np.sort(block.min(0))[:support])),
                            float(np.mean(np.sort(block.min(1))[:support])))
                pair[i, j] = pair[j, i] = score
    duplicates = {}
    for i in sorted(range(count), key=lambda k: (-states[k]["quality"], nodes[k]["node_id"])):
        if nodes[i].get("evidence_kind") != "bootstrap_photo" or states[i]["status"] == "rejected":
            continue
        box = evidence.get(nodes[i]["node_id"], {}).get("bbox_xyxy")
        if not box: continue
        for j in range(count):
            if i == j or j in duplicates or i in duplicates or nodes[i]["clip_name"] != nodes[j]["clip_name"]:
                continue
            other = evidence.get(nodes[j]["node_id"], {}).get("bbox_xyxy")
            if not other: continue
            overlap, containment = box_overlap(box, other)
            if overlap >= policy.duplicate_iou and pair[i, j] <= policy.strong_distance:
                duplicates[j] = i
                states[j].update(status="duplicate", reasons=["duplicate_detection"], duplicate_of=nodes[i]["node_id"])
    # Geometry alone cannot decide which of two overlapping bodies is the
    # intended target. Quarantine ambiguous photos rather than delete players.
    for i in range(count):
        if nodes[i].get("evidence_kind") != "bootstrap_photo" or i in duplicates:
            continue
        a = evidence.get(nodes[i]["node_id"], {}).get("bbox_xyxy")
        if not a: continue
        for j in range(i):
            if j in duplicates or nodes[i]["clip_name"] != nodes[j]["clip_name"]: continue
            b = evidence.get(nodes[j]["node_id"], {}).get("bbox_xyxy")
            if b and box_overlap(a, b)[1] >= policy.overlap_fraction:
                for k in (i, j):
                    states[k]["reasons"] = sorted(set(states[k]["reasons"] + ["overlapping_person_targets"]))
    eligible = [i for i, state in enumerate(states) if not state["reasons"] and not records[i].get("locked")]
    anchors = {}
    for i in eligible:
        if incremental and records[i].get("established"):
            anchors.setdefault(records[i]["player"], []).append(i)
    anchored = {i for group in anchors.values() for i in group}
    groups = [[i] for i in eligible if i not in anchored] + list(anchors.values())
    # Manual groups are fixed anchors; detached/rejected observations never
    # rejoin automatically. All other pending observations participate.
    locked = {}
    for i, record in enumerate(records):
        if record.get("locked") and not record.get("detached") and not record.get("non_player") and not record.get("rejected"):
            locked.setdefault(record["player"], []).append(i)
    groups += list(locked.values())
    locks = {i: r["player"] for i, r in enumerate(records) if r.get("locked")}

    def merge_distance(a, b):
        if len({locks[i] for i in a+b if i in locks}) > 1:
            return math.inf
        if any(incompatible(nodes[i], nodes[j]) for i in a for j in b):
            return math.inf
        combined = a+b
        nums = {number_hint(nodes[i]) for i in combined} - {None}
        teams = {nodes[i].get("team_id") for i in combined} - {None}
        if len(nums) > 1 or len(teams) > 1: return math.inf
        if incremental and any(i in anchored for i in a+b):
            ai = [k for i in a for k in states[i]["feature_indices"]]
            bi = [k for i in b for k in states[i]["feature_indices"]]
            if not ai or not bi: return math.inf
            block = distances[np.ix_(ai,bi)]
            support = min(2,len(ai),len(bi))
            value = max(float(np.mean(np.sort(block.min(0))[:support])),
                        float(np.mean(np.sort(block.min(1))[:support])))
        else:
            value = float(np.max(pair[np.ix_(a, b)]))
        # Weak number agreement can relax appearance matching only with
        # positive readings on both sides, never via an unreadable bridge.
        def supported_side(group):
            numbered = [i for i in group if number_hint(nodes[i])]
            return (len({nodes[i]["clip_name"] for i in numbered}) >= 2
                    or any((nodes[i].get("jersey") or {}).get("status") == "candidate_consensus" for i in numbered))
        same_team = len(teams) == 1 and all(nodes[i].get("team_id") for i in combined)
        supported = (bool(nums) and any(number_hint(nodes[i]) for i in a) and any(number_hint(nodes[i]) for i in b)
                     and (same_team or supported_side(a) or supported_side(b)))
        threshold = policy.supported_distance if supported else policy.strong_distance
        return value if value <= threshold else math.inf

    # Cache unchanged edges: a merge only changes costs touching its group.
    # This avoids recomputing the whole quadratic candidate graph each round.
    groups = dict(enumerate(groups))
    costs = {}
    for x, a in groups.items():
        for y, b in groups.items():
            if y >= x: continue
            value = merge_distance(a,b)
            if math.isfinite(value): costs[(x,y)] = value
    while costs:
        edges = sorted((value,x,y) for (x,y),value in costs.items())
        accepted = None
        for value, x, y in edges:
            if not math.isfinite(value): break
            rivals = [d for d, u, v in edges if math.isfinite(d) and (u in (x,y) or v in (x,y)) and {u,v} != {x,y}]
            # Competing same-number views of the same complete-link cluster
            # may join together; uncertain appearance-only rivals abstain.
            a, b = groups[x], groups[y]
            shared = {number_hint(nodes[i]) for i in a+b} - {None}
            number_supported = len(shared) == 1 and all(number_hint(nodes[i]) for i in a+b) and (len(a) >= 2 or len(b) >= 2)
            if rivals and min(rivals)-value < policy.min_margin and not number_supported:
                continue
            accepted = x, y; break
        if accepted is None: break
        x, y = accepted
        groups[y] = sorted(groups[y]+groups.pop(x))
        costs = {edge:value for edge,value in costs.items() if x not in edge and y not in edge}
        for z, other in groups.items():
            if z == y: continue
            value = merge_distance(groups[y],other)
            if math.isfinite(value): costs[(max(z,y),min(z,y))] = value
    planned = []
    for group in groups.values():
        photos = [i for i in group if nodes[i].get("evidence_kind") == "bootstrap_photo"]
        independent = any(nodes[i]["clip_name"] != nodes[j]["clip_name"] and
            abs(float(nodes[i].get("timestamp_s", 0))-float(nodes[j].get("timestamp_s", 0))) >= policy.independent_gap_s
            for i in photos for j in photos)
        track_support = any(nodes[i].get("evidence_kind") != "bootstrap_photo" and nodes[i].get("sample_count", 0) >= 2 for i in group)
        # Novelty distance is diagnostic evidence, not a compulsory gate
        # against a gallery with only two initial templates.
        other = [j for j in eligible if j not in group and not any(incompatible(nodes[i],nodes[j]) for i in group)]
        nearest = min((float(np.min(pair[np.ix_(group,[j])])) for j in other), default=math.inf)
        established = any(records[i].get("established") for i in group)
        numbered = [i for i in group if number_hint(nodes[i])]
        number_backed = len({nodes[i]["clip_name"] for i in numbered}) >= 2 and len({number_hint(nodes[i]) for i in numbered}) == 1
        promoted = (independent or track_support or any(i in locks for i in group)) and (
            established or any(i in locks for i in group) or number_backed or nearest > policy.strong_distance)
        planned.append({"members": group, "promoted": promoted, "nearest_other_distance": nearest if math.isfinite(nearest) else None,
                        "novel_by_distance": math.isfinite(nearest) and nearest >= policy.novelty_distance})
        for i in group:
            if not promoted:
                states[i]["reasons"] = ["ambiguous_nearby_identity" if nearest <= policy.supported_distance else "insufficient_independent_support"]
            states[i]["status"] = "accepted" if promoted else "pending"
        if promoted:
            items, indices = {}, []
            for i in group:
                if states[i]["reasons"] or records[i].get("detached"): continue
                for k, template in enumerate(banks[i]):
                    items[len(items)] = {**template, "owner": i, "bank_index": k}
                    indices.append(states[i]["feature_indices"][k])
            selected = select_templates(items, distances[np.ix_(indices, indices)], policy.gallery_limit)
            for k in selected:
                item = items[k]
                states[item["owner"]]["templates"].append({key: value for key,value in item.items() if key not in {"feature", "visibility", "owner", "bank_index"}})
    return {"states": states, "groups": planned, "duplicates": duplicates,
            "policy": asdict(policy), "accuracy_evaluated": False}
