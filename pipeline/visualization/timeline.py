"""Framework-free helpers for aligning sparse actions with video frames."""

from __future__ import annotations

from bisect import bisect_left
from typing import Any


def nearest_action(
    actions: list[dict[str, Any]], frame_idx: int, max_gap: int
) -> dict[str, Any] | None:
    if not actions:
        return None
    indices = [int(item["frame_idx"]) for item in actions]
    position = bisect_left(indices, frame_idx)
    candidates = []
    if position < len(actions):
        candidates.append(actions[position])
    if position > 0:
        candidates.append(actions[position - 1])
    result = min(
        candidates,
        key=lambda item: (abs(int(item["frame_idx"]) - frame_idx), item["frame_idx"]),
    )
    return result if abs(int(result["frame_idx"]) - frame_idx) <= max_gap else None


def action_text(
    action: dict[str, Any] | None, *, show_top_candidate: bool
) -> str | None:
    if action is None:
        return None
    selected = action.get("selected_actions", [])
    if selected:
        best = max(selected, key=lambda item: float(item["score"]))
        return f'{best["label"]} {float(best["score"]):.2f}'
    candidates = action.get("action_candidates", [])
    if show_top_candidate and candidates:
        best = candidates[0]
        return f'top? {best["label"]} {float(best["score"]):.2f}'
    return None

