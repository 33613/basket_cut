"""Sample full temporal windows at clip boundaries without inventing source time."""
import numpy as np
import math


def physical_sampling(fps, frame_interval, predict_stepsize, reference_fps=None):
    """Return interval/cadence in actual frames; None preserves legacy behavior."""
    if any(not math.isfinite(v) or v <= 0 for v in (fps, frame_interval, predict_stepsize)):
        raise ValueError("Invalid action sampling settings")
    if reference_fps is None:
        return frame_interval, predict_stepsize
    if not math.isfinite(reference_fps) or reference_fps <= 0:
        raise ValueError("Invalid action reference FPS")
    return frame_interval*fps/reference_fps, max(1, round(predict_stepsize*fps/reference_fps))


def sample_frame_indices(center_frame, frame_count, clip_len, frame_interval):
    if frame_count < 1 or clip_len < 1 or frame_interval < 1 or not 0 <= center_frame < frame_count:
        raise ValueError("Invalid action sampling window")
    raw = center_frame - (clip_len // 2 - 1) * frame_interval + np.arange(clip_len) * frame_interval
    return np.clip(raw, 0, frame_count - 1), bool((raw < 0).any() or (raw >= frame_count).any())
