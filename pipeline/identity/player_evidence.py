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
PLAYER_PROMPT = """Assess the central person in this crop from real basketball footage.
First decide whether the visible person is a player, referee, other or uncertain.
Do not assume every person near the court is a player.
Do not infer a hidden digit or read another person's jersey or the scoreboard.
Return exactly one JSON object with these fields:
{"role":"player|referee|other|uncertain", "single_target":true_or_false,
 "unoccluded":true_or_false, "clear":true_or_false,
 "quality_reason":"brief visible evidence for the clarity judgment",
 "jersey_color":"black|white|red|blue|green|yellow|orange|purple|gray|unknown",
 "number":"0-99 or 00, or null", "readable":true_or_false,
 "view":"front|back|left|right|unknown"}.
single_target is false when another person makes the intended target ambiguous.
unoccluded is false when another person or object covers a significant part of the target.
Judge clear for BODY APPEARANCE re-identification, using clothing, body outline,
shirt/shorts boundaries and visible appearance detail, not facial recognition.
clear=true requires enough real detail to compare body/clothing appearances.
An unreadable face or jersey number alone is NOT a reason for clear=false.
clear=false when severe blur, compression/pixelation or missing body detail makes
body appearance comparison unreliable. Do not label an image clear merely
because the shirt color or one digit is recognizable. Explain the visible defect
or useful appearance detail in quality_reason; do not invent detail by zooming.
Decide number readability separately. An appearance crop can be clear while its
number is unreadable; readable digits alone do not certify appearance quality.
jersey_color is the main shirt fabric, excluding text, skin, court, shorts and trim.
Return unknown for ambiguous color or view. Preserve 0 versus 00. Return a digit
STRING for a readable number and JSON null for an unreadable number."""

NATIVE_CONTEXT = "\nNative crop dimensions before processor resizing: {width}x{height} pixels. Judge visible detail; do not decide clarity from pixel dimensions alone."


def player_prompt(size):
    width, height = size
    return PLAYER_PROMPT + NATIVE_CONTEXT.format(width=width, height=height)


def assessment_provenance(base):
    return {
        **base,
        "task": "clear_player_v2",
        "number_parser": "digit_string_or_json_integer_0_99_v2",
        "assessment_prompt_sha256": hashlib.sha256(
            (PLAYER_PROMPT + NATIVE_CONTEXT).encode()
        ).hexdigest(),
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
    number = value.get("number")
    # Qwen can produce a valid JSON integer despite the requested string.
    # Preserve explicit "00"; never guess that an integer 0 meant "00".
    normalized_integer = type(number) is int and 0 <= number <= 99
    if normalized_integer:
        number = str(number)
    reading = parse_reading(
        json.dumps({"number": number, "readable": value.get("readable")})
    )
    color = value.get("jersey_color")
    view = value.get("view")
    return {
        "usable": not reasons,
        "number": reading["number"] if not reasons else None,
        "single_target": value.get("single_target") is True,
        "unoccluded": value.get("unoccluded") is True,
        "clear": value.get("clear") is True,
        "quality_reason": value.get("quality_reason")
        if isinstance(value.get("quality_reason"), str)
        else "",
        "number_normalization": "integer_to_digit_string"
        if normalized_integer
        else None,
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
