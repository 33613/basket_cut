"""Action-recognition application workflow."""

from __future__ import annotations

from typing import Any

from pipeline.action.service import ActionOptions, recognize_actions


def run_action_recognition(options: ActionOptions) -> dict[str, Any]:
    return recognize_actions(options)
