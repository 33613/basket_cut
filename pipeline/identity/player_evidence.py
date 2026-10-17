"""Explicit clear-player evidence; VLM answers remain unverified observations."""

from collections import Counter
import hashlib
import json
import math

COLORS = {
    "black",
    "white",
    "red",
    "blue",
    "green",
    "yellow",
    "orange",
    "purple",
    "gray",
}
VIEWS = {"front", "back", "left", "right", "unknown"}
PLAYER_PROMPT = """Inspect only the central basketball player in this original-resolution crop.
Do not infer a hidden digit or read another person's jersey or the scoreboard.
Return exactly one JSON object with these fields:
{"role":"player|referee|other|uncertain", "single_target":true_or_false,
 "unoccluded":true_or_false, "clear":true_or_false,
 "jersey_color":"black|white|red|blue|green|yellow|orange|purple|gray|unknown",
 "number":"0-99 or 00, or null", "readable":true_or_false,
 "view":"front|back|left|right|unknown"}.
single_target is false when another person makes the intended target ambiguous.
unoccluded is false when another person or object covers a significant part of the target.
clear means this crop is suitable as an identity gallery image. A readable number
is not required for clear. jersey_color is the main shirt fabric, excluding text,
skin, court, shorts and trim. If not clear, return unknown. Return unknown view
when unsure. Preserve 0 versus 00. Use JSON null for unreadable numbers."""


def assessment_provenance(base):
    return {
        **base,
        "task": "clear_player_v1",
        "assessment_prompt_sha256": hashlib.sha256(PLAYER_PROMPT.encode()).hexdigest(),
    }


def parse_assessment(raw):
    from pipeline.identity.qwen_backend import parse_reading

    try:
        text = raw.strip()
        if text.startswith("```json") and text.endswith("```"):
            text = text[7:-3].strip()
        value = json.loads(text)
    except (TypeError, ValueError):
        value = None
    if not isinstance(value, dict):
        return {
            "usable": False,
            "number": None,
            "role": "uncertain",
            "jersey_color": "unknown",
            "view": "unknown",
            "rejection_reasons": ["invalid_assessment_json"],
        }
    reasons = []
    if value.get("role") != "player":
        reasons.append("not_clear_player")
    for field in ("single_target", "unoccluded", "clear"):
        if value.get(field) is not True:
            reasons.append("assessment_" + field + "_failed")
    reading = parse_reading(
        json.dumps({"number": value.get("number"), "readable": value.get("readable")})
    )
    color = value.get("jersey_color")
    view = value.get("view")
    return {
        "usable": not reasons,
        "number": reading["number"] if not reasons else None,
        "single_target": value.get("single_target") is True,
        "unoccluded": value.get("unoccluded") is True,
        "clear": value.get("clear") is True,
        "number_rejection_reasons": reading["rejection_reasons"],
        "role": value.get("role", "uncertain"),
        "jersey_color": color if color in COLORS and not reasons else "unknown",
        "view": view if view in VIEWS else "unknown",
        "rejection_reasons": reasons,
    }


def separated(readings, gap):
    chosen, last, frames = [], -math.inf, set()
    for row in sorted(readings, key=lambda r: (r["timestamp_s"], r["frame_idx"])):
        if row["frame_idx"] not in frames and row["timestamp_s"] - last >= gap:
            chosen.append(row)
            frames.add(row["frame_idx"])
            last = row["timestamp_s"]
    return chosen


def summarize(readings, *, min_support=2, gap=0.5, team_map=None):
    if min_support < 2 or not math.isfinite(gap) or gap <= 0:
        raise ValueError("Invalid clear-player support policy")
    usable = [r for r in readings if r.get("usable") is True]
    colors = {r["jersey_color"] for r in usable if r.get("jersey_color") in COLORS}
    color = next(iter(colors)) if len(colors) == 1 else None
    color_support = (
        separated([r for r in usable if r.get("jersey_color") == color], gap)
        if color
        else []
    )
    number_set = {r["number"] for r in usable if r.get("number") is not None}
    number = next(iter(number_set)) if len(number_set) == 1 else None
    number_support = (
        separated([r for r in usable if r.get("number") == number], gap)
        if number is not None
        else []
    )
    mapped = (team_map or {}).get(color) if len(color_support) >= min_support else None
    team = {
        "status": "candidate_consensus"
        if mapped
        else "kit_only"
        if len(color_support) >= min_support
        else "insufficient_evidence",
        "team_id": mapped,
        "kit_color": color,
        "support_frames": len(color_support),
        "method": "clear_player_multiframe_v1",
        "verified": False,
    }
    jersey = {
        "number": number if len(number_support) >= min_support else None,
        "status": "candidate_consensus"
        if len(number_support) >= min_support
        else "conflict"
        if len(number_set) > 1
        else "insufficient_evidence",
        "candidate_numbers": sorted(number_set),
        "support_frames": len(number_support),
        "readings": readings,
        "verified": False,
        "method": "clear_player_multiframe_v1",
    }
    return {
        "team_id": mapped,
        "team_evidence": team,
        "jersey": jersey,
        "usable_count": len(usable),
        "clear_support": len(separated(usable, gap)),
    }


def infer_team_map(track_readings):
    # Never force arbitrary colors into two teams. Ambiguous/single-color
    # material needs an explicit map, while anonymous tracklets remain valid.
    counts = Counter()
    for readings in track_readings:
        summary = summarize(readings)
        team = summary["team_evidence"]
        if team["status"] == "kit_only":
            counts[team["kit_color"]] += 1
    if len(counts) != 2 or min(counts.values()) < 2:
        return {}
    return {color: "KIT-" + color for color in sorted(counts)}


def strong_identity(node):
    team, jersey = node.get("team_evidence") or {}, node.get("jersey") or {}
    if (
        team.get("method") == jersey.get("method") == "clear_player_multiframe_v1"
        and team.get("status") == jersey.get("status") == "candidate_consensus"
        and team.get("team_id") == node.get("team_id")
        and node.get("team_id")
        and team.get("support_frames", 0) >= 2
        and jersey.get("support_frames", 0) >= 2
        and isinstance(jersey.get("number"), str)
        and jersey["number"].isascii()
        and jersey["number"].isdigit()
        and 1 <= len(jersey["number"]) <= 2
    ):
        return node["team_id"], jersey["number"]
    return None
