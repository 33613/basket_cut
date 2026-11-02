"""Rank shot-action evidence; results are hypotheses, not ball-verified attribution."""

from collections import defaultdict
import math

SHOT_LABELS = frozenset(("basketball_3-point_shot", "basketball_2-point_shot", "basketball_free_throw"))


def rank_shooters(event, source, tracks, actions, options):
    """Use only observed person boxes and pre-goal shot scores; ignore annotation identities."""
    fps, offset = source["fps"], event["window_start_s"]
    observed = {(r.track_id, r.frame_idx): r for r in tracks}
    if len(observed) != len(tracks):
        raise ValueError("Duplicate track observations")
    points, seen = [], set()
    for row in actions:
        if row["video_id"] != event["event_id"]:
            raise ValueError("Action evidence belongs to a different event window")
        frame, tid = int(row["frame_idx"]), int(row["track_id"])
        key = tid, frame
        if key in seen:
            raise ValueError("Duplicate track/action frame")
        seen.add(key)
        if abs(float(row["timestamp_s"])-frame/fps) > 1/fps:
            raise ValueError("Action frame clock mismatch")
        scores = {}
        for item in row.get("action_candidates", []) + row.get("selected_actions", []):
            score = float(item["score"])
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("Invalid action score")
            if item["label"] in SHOT_LABELS:
                scores[item["label"]] = max(score, scores.get(item["label"], 0))
        if not scores or key not in observed or row.get("proposal_frame_offset", 0) != 0:
            continue
        label = max(scores, key=scores.get)
        time = offset+frame/fps
        if time <= event["goal_time_s"]-options.min_flight_s:
            points.append(dict(track_id=tid, frame_idx=frame, source_time_s=time, score=scores[label],
                               label=label, temporal_padding=bool(row.get("temporal_padding", False))))

    def candidates(lookback):
        by_track = defaultdict(list)
        for p in points:
            if event["goal_time_s"]-lookback <= p["source_time_s"] and p["score"] >= options.score_threshold:
                by_track[p["track_id"]].append(p)
        ranked = []
        for tid, rows in by_track.items():
            runs = []
            for p in sorted(rows, key=lambda p: p["frame_idx"]):
                if not runs or p["source_time_s"]-runs[-1][-1]["source_time_s"] > options.evidence_gap_s:
                    runs.append([])
                runs[-1].append(p)
            for run in runs:
                if len(run) < options.min_support:
                    continue
                best = max(run, key=lambda p: (p["score"], p["frame_idx"]))
                strongest = sorted((p["score"] for p in run), reverse=True)[:3]
                ranked.append(dict(track_id=tid, track_uid=f"{event['event_id']}:T{tid:04d}",
                    score=sum(strongest)/len(strongest), peak_score=best["score"],
                    peak_frame=best["frame_idx"], peak_source_time_s=best["source_time_s"],
                    action_label=best["label"], support_count=len(run),
                    action_interval_source_s=[run[0]["source_time_s"], run[-1]["source_time_s"]],
                    context_padded=any(p["temporal_padding"] for p in run), evidence=run))
        # Compare different people, not two temporal runs of the same track.
        best_by_track = {}
        for c in sorted(ranked, key=lambda c: (c["score"], c["peak_source_time_s"]), reverse=True):
            best_by_track.setdefault(c["track_id"], c)
        return sorted(best_by_track.values(), key=lambda c: (-c["score"], -c["peak_source_time_s"], c["track_id"]))

    ranked = candidates(options.lookback_s)
    unique = ranked and (len(ranked) == 1 or ranked[0]["score"]-ranked[1]["score"] >= options.min_margin)
    near_left_edge = ranked and ranked[0]["action_interval_source_s"][0] <= event["goal_time_s"]-options.lookback_s+options.evidence_gap_s
    expanded = (not unique or near_left_edge) and options.max_lookback_s > options.lookback_s
    if expanded:
        ranked = candidates(options.max_lookback_s)
        unique = ranked and (len(ranked) == 1 or ranked[0]["score"]-ranked[1]["score"] >= options.min_margin)
    reasons = []
    # A high-scoring earlier attempt must not hide a later possible putback by another actor.
    later_actor = ranked and any(c["peak_source_time_s"] > ranked[0]["peak_source_time_s"]+options.evidence_gap_s
                                 for c in ranked[1:])
    if later_actor:
        unique = False
        reasons.append("later_supported_shot_by_another_actor")
    if not ranked:
        reasons.append("no_supported_shot_action")
    elif not unique:
        reasons.append("ambiguous_actor_scores")
    if event["partial_pre_context"]:
        reasons.append("input_starts_inside_search_context")
    if ranked and ranked[0]["context_padded"]:
        reasons.append("action_context_repeats_edge_frames")
    result = dict(schema_version=1, event_id=event["event_id"], video_id=event["video_id"],
        goal_time_s=event["goal_time_s"], window_start_frame=event["window_start_frame"],
        window_end_frame_exclusive=event["window_end_frame_exclusive"],
        status="candidate" if unique else "needs_review" if ranked else "no_candidate",
        shooter_track_id=ranked[0]["track_id"] if unique else None,
        track_uid=ranked[0]["track_uid"] if unique else None, candidates=ranked[:5],
        search_lookback_s=options.max_lookback_s if expanded else options.lookback_s,
        expanded_search=expanded, diagnostics=reasons, requires_review=True, accuracy_evaluated=False,
        attribution_method="shot_action_evidence_only", ball_verified=False,
        release_interval_kind="action_candidate_interval_not_exact_ball_release",
        player_id=None, warning="Scores are uncalibrated action evidence. No ball-flight or made-shot verification.")
    if unique:
        result["trajectory"] = track_segment(ranked[0], tracks, fps, event, options)
    return result


def track_segment(candidate, tracks, fps, event, options):
    rows = sorted((r for r in tracks if r.track_id == candidate["track_id"]), key=lambda r: r.frame_idx)
    segments = []
    for r in rows:
        split = not segments
        if segments:
            previous = segments[-1][-1]
            a, b = previous.bbox_xyxy, r.bbox_xyxy
            gap = r.frame_idx-previous.frame_idx
            motion = math.hypot((a[0]+a[2]-b[0]-b[2])/2, (a[1]+a[3]-b[1]-b[3])/2)
            scale = max(1, math.hypot(a[2]-a[0], a[3]-a[1]))
            split = gap/fps > options.tracking_gap_s or motion/scale/max(1, gap) > 1.5
        if split:
            segments.append([])
        segments[-1].append(r)
    segment = next(s for s in segments if any(r.frame_idx == candidate["peak_frame"] for r in s))
    # Bound around the action evidence, not the entire potentially mixed MOTIP identity.
    start = max(segment[0].frame_idx, math.floor((candidate["action_interval_source_s"][0]-event["window_start_s"]-options.segment_padding_s)*fps))
    end = min(segment[-1].frame_idx, math.ceil((candidate["action_interval_source_s"][1]-event["window_start_s"]+options.segment_padding_s)*fps))
    return dict(window_start_frame=start, window_end_frame_inclusive=end,
                source_start_frame=event["window_start_frame"]+start,
                source_end_frame_inclusive=event["window_start_frame"]+end,
                identity_continuity="unverified; gap/jump splitting is heuristic, not ID-switch detection")
