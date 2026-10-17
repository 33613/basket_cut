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
ASSESSMENT_VERSION = "player_observations_v3"
PLAYER_PROMPT = """Describe only visible facts about the central person in this basketball crop.
Do not judge whether a re-identification algorithm will work. Do not identify a
known athlete. Return exactly one JSON object, with no additional text:
{"schema":"player_observations_v3", "role":"player|referee|other|uncertain",
 "single_target":true_or_false, "body_coverage":"full|upper|fragment|unknown",
 "body_occlusion":"none|minor|major|unknown", "torso_visible":true_or_false,
 "torso_unoccluded":true_or_false,
 "graphic_overlay":"none|non_torso|torso|unknown",
 "blur":"none|mild|severe|unknown",
 "clothing_detail":"distinct|limited|indistinct|unknown",
 "quality_reason":"specific visible features or defects, in one sentence",
 "jersey_color":"black|white|red|blue|green|yellow|orange|purple|gray|unknown",
 "number":"digit string or null", "readable":true_or_false,
 "view":"front|back|left|right|unknown"}.
full means head, torso, hips, legs and feet are visible inside the crop. A large
head/torso close-up is upper or fragment, not full. Missing body parts at crop
edges count as incomplete coverage. Other people or objects covering the target
count as body_occlusion, even if the torso is visible.
Inspect TV scoreboards, captions, logos and advertisements covering the person.
graphic_overlay=non_torso when they cover only legs/feet or other non-torso body
regions; torso when they cover the shirt/number. Overlays confined to background
are none. torso_unoccluded requires a visible shirt without intervening objects.
Judge blur separately from coverage and overlays. mild means edges are softened
but clothing boundaries remain distinguishable; severe means motion smear or
compression prevents distinguishing those boundaries. distinct clothing_detail
means shirt/shorts boundaries and visible garment patterns can be described;
limited means only coarse shapes/color; indistinct means those are not visible.
An unreadable face or number does not alone imply severe blur or indistinct
clothing. Never infer missing detail. Use unknown for uncertain observations.
jersey_color is the dominant shirt FABRIC, not numbers, colored trim, shorts,
background or scoreboard. Dark shirts with pink lettering are not pink shirts.
Read the CENTRAL person's own jersey only. A clear number can be read even when
legs are missing or covered; never infer a digit hidden by blur/objects/graphics.
Use readable=false and number=null when incomplete or uncertain. Preserve 0 and
00; return a digit STRING. Number readability is separate from body coverage."""

NATIVE_CONTEXT = "\nNative crop dimensions before processor resizing: {width}x{height} pixels. Judge visible detail; do not decide clarity from pixel dimensions alone."


def player_prompt(size):
    width, height = size
    return PLAYER_PROMPT + NATIVE_CONTEXT.format(width=width, height=height)


def assessment_provenance(base):
    return {
        **base,
        "task": ASSESSMENT_VERSION,
        "number_parser": "digit_string_or_json_integer_0_99_v2",
        "assessment_prompt_sha256": hashlib.sha256(
            (PLAYER_PROMPT + NATIVE_CONTEXT).encode()
        ).hexdigest(),
    }


def parse_assessment(raw, *, require_observations=False):
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
    # Malformed enum fields must abstain, not crash on unhashable model JSON.
    for key in (
        "role",
        "body_coverage",
        "body_occlusion",
        "graphic_overlay",
        "blur",
        "clothing_detail",
        "jersey_color",
        "view",
    ):
        if not isinstance(value.get(key), str):
            value[key] = "unknown"
    reasons = []
    observed = value.get("schema") == ASSESSMENT_VERSION
    if require_observations and not observed:
        reasons.append("missing_observation_schema")
    target = value.get("role") == "player" and value.get("single_target") is True
    if value.get("role") != "player":
        reasons.append("not_clear_player")
    if value.get("single_target") is not True:
        reasons.append("assessment_single_target_failed")
    if observed:
        clear = (
            value.get("blur") in {"none", "mild"}
            and value.get("clothing_detail") == "distinct"
        )
        unoccluded = (
            value.get("body_occlusion") == "none"
            and value.get("graphic_overlay") == "none"
        )
        if value.get("body_coverage") != "full":
            reasons.append("incomplete_body_coverage")
        if not unoccluded:
            reasons.append("body_occlusion_or_graphic_overlay")
        if not clear:
            reasons.append("insufficient_clothing_detail")
        evidence_eligible = (
            target
            and value.get("torso_visible") is True
            and value.get("torso_unoccluded") is True
            and value.get("graphic_overlay") in {"none", "non_torso"}
        )
        if not evidence_eligible:
            reasons.append("torso_evidence_unavailable")
    else:
        # Read old raw responses for audits only. Live V3 calls require observations.
        clear = value.get("clear") is True
        unoccluded = value.get("unoccluded") is True
        for field in ("unoccluded", "clear"):
            if value.get(field) is not True:
                reasons.append("assessment_" + field + "_failed")
        evidence_eligible = not reasons
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
        "number": reading["number"] if evidence_eligible else None,
        "evidence_eligible": evidence_eligible,
        "appearance_eligible": not reasons,
        "observations": {
            k: value.get(k)
            for k in (
                "body_coverage",
                "body_occlusion",
                "torso_visible",
                "torso_unoccluded",
                "graphic_overlay",
                "blur",
                "clothing_detail",
            )
        }
        if observed
        else {},
        "single_target": value.get("single_target") is True,
        "unoccluded": unoccluded,
        "clear": clear,
        "quality_reason": value.get("quality_reason")
        if isinstance(value.get("quality_reason"), str)
        else "",
        "number_normalization": "integer_to_digit_string"
        if normalized_integer
        else None,
        "number_rejection_reasons": reading["rejection_reasons"],
        "role": value.get("role", "uncertain"),
        "jersey_color": color if color in COLORS and evidence_eligible else "unknown",
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
    evidence = [
        r for r in readings if r.get("evidence_eligible", r.get("usable")) is True
    ]
    palette = team_map or {}
    colors = {r["jersey_color"] for r in evidence if r.get("jersey_color") in COLORS}
    color = next(iter(colors)) if len(colors) == 1 else None
    mapped_ids = {palette[c] for c in colors if c in palette}
    # Explicit aliases may describe one kit (black/purple). Unknown aliases or
    # observations spanning distinct team IDs cannot establish a stable team.
    mapped = (
        next(iter(mapped_ids))
        if len(mapped_ids) == 1 and colors <= palette.keys()
        else None
    )
    color_support = separated(
        [
            r
            for r in evidence
            if (mapped is not None and palette.get(r.get("jersey_color")) == mapped)
            or (mapped is None and color is not None and r.get("jersey_color") == color)
        ],
        gap,
    )
    if len(color_support) < min_support:
        mapped = None
    number_set = {r["number"] for r in evidence if r.get("number") is not None}
    number = next(iter(number_set)) if len(number_set) == 1 else None
    number_support = (
        separated([r for r in evidence if r.get("number") == number], gap)
        if number is not None
        else []
    )
    # Joint evidence stops disjoint color-only/number-only frames from creating
    # a strong identity. Keep independent weak observations for later review.
    joint_support = separated(
        [
            r
            for r in evidence
            if number is not None
            and r.get("number") == number
            and mapped is not None
            and palette.get(r.get("jersey_color")) == mapped
        ],
        gap,
    )
    team = {
        "status": "candidate_consensus"
        if mapped
        else "kit_only"
        if len(color_support) >= min_support
        else "insufficient_evidence",
        "team_id": mapped,
        "kit_color": color,
        "observed_colors": sorted(colors),
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
        "joint_support_frames": len(joint_support),
        "readings": readings,
        "verified": False,
        "method": "clear_player_multiframe_v1",
    }
    return {
        "team_id": mapped,
        "team_evidence": team,
        "jersey": jersey,
        "usable_count": len(usable),
        "evidence_count": len(evidence),
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
        and jersey.get("joint_support_frames", 2) >= 2
        and isinstance(jersey.get("number"), str)
        and jersey["number"].isascii()
        and jersey["number"].isdigit()
        and 1 <= len(jersey["number"]) <= 2
    ):
        return node["team_id"], jersey["number"]
    return None
