"""Observed ball -> upper-body contact -> separation/rim -> local shooter hypothesis.

Person boxes are a contact proxy, not hand keypoints. Scores are uncalibrated.
No annotation identity, predicted ball point, or post-goal rebound selects a shooter.
"""

import math
from collections import defaultdict

from pipeline.player_link.evidence import SHOT_LABELS, rank_shooters, track_segment
from pipeline.player_link.roles import ROLE_POLICY, match_roles, role_support


def _distance(point, box):
    x, y = point
    x1, y1, x2, y2 = box
    return math.hypot(max(x1 - x, 0, x - x2), max(y1 - y, 0, y - y2))


def _upper(box):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    return (x1 - 0.12 * w, y1 - 0.20 * h, x2 + 0.12 * w, y1 + 0.60 * h)


def rank_with_ball(event, source, tracks, actions, ball_rows, options):
    baseline = rank_shooters(event, source, tracks, actions, options)
    fps = source["fps"]
    goal = event["goal_time_s"]
    start = goal - options.lookback_s
    person = {(r.frame_idx, r.track_id): r for r in tracks}
    if len(person) != len(tracks):
        raise ValueError("Duplicate person observation")
    by_frame = defaultdict(list)
    sizes = defaultdict(int)
    for r in tracks:
        by_frame[r.frame_idx].append(r)
        sizes[r.track_id] += 1
    # Validate the original-video ball clock, then discard predicted/unconfirmed points.
    observed = []
    seen = set()
    for row in ball_rows:
        frame = row["source_frame_idx"]
        time = row["source_timestamp_s"]
        if frame in seen or abs(time - frame / fps) > 1e-6 or not math.isfinite(time):
            raise ValueError("Duplicate/mismatched original ball frame clock")
        seen.add(frame)
        ball = row.get("ball_track")
        if not ball or not ball.get("observed") or not ball.get("confirmed", False):
            continue
        if not start <= time <= goal + 0.25:
            continue
        x, y = ball["center_xy"]
        if not all(math.isfinite(v) for v in (x, y)):
            raise ValueError("Invalid ball center")
        rims = row.get("rims", [])
        rim_dist = min(
            (
                _distance((x, y), d["bbox_xyxy"])
                / max(1, d["bbox_xyxy"][2] - d["bbox_xyxy"][0])
                for d in rims
            ),
            default=math.inf,
        )
        observed.append(
            dict(
                row,
                ball=ball,
                time=time,
                local_frame=frame - event["window_start_frame"],
                rim_distance=rim_dist,
            )
        )
    observed.sort(key=lambda r: r["source_frame_idx"])
    # Anchor to a temporally supported near-rim path around the external goal time.
    near = defaultdict(list)
    for p in observed:
        if goal - 1.25 <= p["time"] <= goal + 0.25 and p["rim_distance"] <= 0.75:
            near[p["ball"]["ball_track_id"]].append(p)
    anchors = []
    for tid, points in near.items():
        for i, p in enumerate(points):
            neighborhood = [
                q
                for q in points[max(0, i - 8) : i + 9]
                if abs(q["time"] - p["time"]) <= 0.12
            ]
            if len(neighborhood) >= 2:
                anchors.append((abs(p["time"] - goal), p["rim_distance"], tid, p))
    eligible = {
        tid for tid, count in sizes.items() if count >= max(3, math.ceil(0.1 * fps))
    }
    roles = match_roles(ball_rows, by_frame, event, fps, eligible, options.lookback_s)
    path_audits = []
    reasons = []
    earlier_controls = []
    ranked = []
    if not anchors:
        reasons.append("no_supported_ball_rim_anchor")
    else:
        best_anchors = {}
        for candidate_anchor in sorted(anchors, key=lambda a: (a[0], a[1])):
            best_anchors.setdefault(candidate_anchor[2], candidate_anchor)
        for _, _, ball_id, anchor in best_anchors.values():
            before = len(ranked)
            path = [
                p
                for p in observed
                if p["ball"]["ball_track_id"] == ball_id and p["time"] <= anchor["time"]
            ]
            contacts = defaultdict(list)
            for p in path:
                if p["time"] > goal:
                    continue
                for r in by_frame[p["local_frame"]]:
                    if r.det_score < 0.3 or sizes[r.track_id] < max(
                        3, math.ceil(0.1 * fps)
                    ):
                        continue
                    box = r.bbox_xyxy
                    h = max(1, box[3] - box[1])
                    center = p["ball"]["center_xy"]
                    distance = _distance(center, _upper(box)) / h
                    if distance <= 0.08:
                        contacts[r.track_id].append(
                            dict(
                                p,
                                distance=distance,
                                centrality=math.dist(
                                    center, ((box[0] + box[2]) / 2, box[1] + 0.22 * h)
                                )
                                / h,
                                relative_xy=(
                                    (center[0] - (box[0] + box[2]) / 2) / h,
                                    (center[1] - box[1]) / h,
                                ),
                            )
                        )
            for tid, points in contacts.items():
                runs = []
                for p in points:
                    if not runs or p["time"] - runs[-1][-1]["time"] > 0.10:
                        runs.append([])
                    runs[-1].append(p)
                hypotheses = []
                for run in runs:
                    if len(run) < 3 or run[-1]["time"] - run[0]["time"] < 0.05:
                        continue
                    # A flying ball merely crossing a defender's box is weak contact.
                    # Require a short interval with stable ball-to-body displacement.
                    stable_pairs = [
                        (a, b)
                        for a in run
                        for b in run
                        if 0.10 - 1e-6 <= b["time"] - a["time"] <= 0.25
                        and math.dist(a["relative_xy"], b["relative_xy"]) <= 0.08
                    ]
                    if not stable_pairs:
                        continue
                    control_start = min(a["time"] for a, b in stable_pairs)
                    control_end = max(b["time"] for a, b in stable_pairs)
                    release = run[-1]
                    following = []
                    for p in path:
                        if (
                            not release["time"] + 0.025
                            <= p["time"]
                            <= min(anchor["time"], release["time"] + 0.5)
                        ):
                            continue
                        r = person.get((p["local_frame"], tid))
                        if r is not None:
                            separation = _distance(
                                p["ball"]["center_xy"], _upper(r.bbox_xyxy)
                            ) / max(1, r.bbox_xyxy[3] - r.bbox_xyxy[1])
                            if separation >= 0.14:
                                following.append(p)
                    separated = len(following) >= 2
                    # Dunks/putbacks can remain in hand to the rim: do not demand an arc.
                    at_rim = (
                        release["rim_distance"] <= 0.75
                        and abs(anchor["time"] - release["time"]) <= 0.20
                    )
                    if not separated and not at_rim:
                        continue
                    action_points = []
                    for a in actions:
                        if a["video_id"] != event["event_id"]:
                            raise ValueError("Action belongs to a different event")
                        t = event["window_start_s"] + a["frame_idx"] / fps
                        if (
                            a["track_id"] == tid
                            and abs(t - release["time"]) <= 0.5
                            and t <= goal
                        ):
                            for item in a.get("action_candidates", []) + a.get(
                                "selected_actions", []
                            ):
                                if item["label"] in SHOT_LABELS:
                                    action_points.append(
                                        (float(item["score"]), item["label"])
                                    )
                    action_score, label = max(
                        action_points, default=(0.0, "ball_supported_shot")
                    )
                    if not math.isfinite(action_score) or not 0 <= action_score <= 1:
                        raise ValueError("Invalid action score")
                    duration = run[-1]["time"] - run[0]["time"]
                    closeness = max(0, 1 - min(p["centrality"] for p in run))
                    ball_score = (
                        0.45
                        + 0.25 * min(1, duration / 0.25)
                        + 0.20 * closeness
                        + 0.10 * min(1, len(run) / 8)
                    )
                    score = 0.8 * ball_score + 0.2 * action_score
                    hypotheses.append(
                        {
                            "track_id": tid,
                            "track_uid": f"{event['event_id']}:T{tid:04d}",
                            "score": score,
                            "peak_score": action_score,
                            "peak_frame": release["local_frame"],
                            "peak_source_time_s": release["time"],
                            "action_label": label,
                            "support_count": len(run),
                            "action_interval_source_s": [
                                run[0]["time"],
                                release["time"],
                            ],
                            "context_padded": False,
                            "evidence": [
                                {
                                    "source_time_s": p["time"],
                                    "source_frame_idx": p["source_frame_idx"],
                                    "upper_body_distance": p["distance"],
                                }
                                for p in run
                            ],
                            "ball_evidence": {
                                "ball_track_id": ball_id,
                                "ball_score": ball_score,
                                "contact_interval_source_s": [
                                    run[0]["time"],
                                    release["time"],
                                ],
                                "contact_frames": len(run),
                                "separation_frames": len(following),
                                "relative_motion_control_proxy": True,
                                "control_interval_source_s": [
                                    control_start,
                                    control_end,
                                ],
                                "mode": "separation_then_rim"
                                if separated
                                else "hand_to_rim_candidate",
                                "rim_anchor_source_s": anchor["time"],
                                "action_score": action_score,
                                "contact_proxy": "expanded_upper_body_box_not_hand_keypoints",
                            },
                        }
                    )
                for hypothesis in hypotheses:
                    hypothesis["role_evidence"] = role_support(
                        roles, tid, start, hypothesis["peak_source_time_s"]
                    )
                if hypotheses:
                    # Last supported contact/release on this anchored ball path.
                    ranked.append(
                        max(
                            hypotheses,
                            key=lambda c: (c["peak_source_time_s"], c["score"]),
                        )
                    )
            path_audits.append(
                {
                    "ball_track_id": ball_id,
                    "rim_anchor_source_s": anchor["time"],
                    "observed_path_frames": len(path),
                    "contact_track_ids": sorted(contacts),
                    "eligible_hypothesis_track_ids": [
                        c["track_id"] for c in ranked[before:]
                    ],
                }
            )
        # A later non-overlapping stable control/release on the same anchored
        # ball path makes the earlier departure a possible pass, not the shot.
        earlier_controls = [
            c
            for c in ranked
            if any(
                other["track_id"] != c["track_id"]
                and other["ball_evidence"]["ball_track_id"]
                == c["ball_evidence"]["ball_track_id"]
                and other["ball_evidence"]["control_interval_source_s"][0]
                > c["action_interval_source_s"][1] + 0.025
                for other in ranked
            )
        ]
        ranked = [c for c in ranked if c not in earlier_controls]
        best_by_track = {}
        for candidate in sorted(
            ranked,
            key=lambda c: (
                c["role_evidence"]["holder"]["supported"],
                c["score"],
                c["peak_source_time_s"],
            ),
            reverse=True,
        ):
            best_by_track.setdefault(candidate["track_id"], candidate)
        ranked = list(best_by_track.values())
        ranked.sort(
            key=lambda c: (-c["score"], -c["peak_source_time_s"], c["track_id"])
        )
        if not ranked:
            reasons.append("anchored_ball_has_no_supported_actor_contact")
    result = dict(baseline)
    unique = (
        bool(ranked)
        and ranked[0]["score"] >= 0.55
        and (
            len(ranked) == 1
            or ranked[0]["score"] - ranked[1]["score"] >= options.min_margin
        )
    )
    if ranked and not unique:
        reasons.append("ambiguous_ball_actor_evidence")
    selection_basis = "geometric_ball_and_action_margin"
    holder_supported = [
        c
        for c in ranked
        if c["role_evidence"]["holder"]["supported"] and c["score"] >= 0.55
    ]
    if len(holder_supported) == 1:
        selected = holder_supported[0]
        if unique and ranked[0]["track_id"] != selected["track_id"]:
            unique = False
            reasons.append("conflicting_stable_holder_and_ball_action_evidence")
            selection_basis = "conflicting_visual_evidence_requires_review"
        else:
            ranked = [selected] + [c for c in ranked if c is not selected]
            unique = True
            reasons = [r for r in reasons if r != "ambiguous_ball_actor_evidence"]
            selection_basis = "exclusive_stable_holder_on_supported_ball_path"
    result.pop("trajectory", None)
    result.update(
        schema_version=3,
        status="candidate"
        if unique
        else "needs_review"
        if ranked or baseline["candidates"]
        else "no_candidate",
        shooter_track_id=ranked[0]["track_id"] if unique else None,
        track_uid=ranked[0]["track_uid"] if unique else None,
        candidates=ranked[:5],
        diagnostics=reasons,
        attribution_method="ball_contact_rim_action_roles_v3",
        selection_basis=selection_basis,
        role_evidence_used=True,
        role_policy=ROLE_POLICY,
        ball_path_audits=path_audits,
        ball_evidence_used=True,
        earlier_ball_controls=earlier_controls,
        ball_path_supported=bool(anchors),
        ball_verified=False,
        made_shot_verified=False,
        search_lookback_s=options.lookback_s,
        expanded_search=False,
        action_only_reference={
            "status": baseline["status"],
            "shooter_track_id": baseline["shooter_track_id"],
            "candidates": baseline["candidates"],
        },
        release_interval_kind="ball_contact_interval_not_exact_hand_release",
        warning="Upper-body boxes approximate contact; scores are uncalibrated. Review shooter and identity continuity.",
    )
    if unique:
        result["trajectory"] = track_segment(ranked[0], tracks, fps, event, options)
        # Do not reach back before the supported contact run to collect a
        # different identity under a reused MOTIP ID (observed in video 3/T6).
        first = math.floor(
            (ranked[0]["action_interval_source_s"][0] - event["window_start_s"]) * fps
            + 1e-6
        )
        result["trajectory"]["window_start_frame"] = max(
            first, result["trajectory"]["window_start_frame"]
        )
        result["trajectory"]["source_start_frame"] = (
            event["window_start_frame"] + result["trajectory"]["window_start_frame"]
        )
    return result
