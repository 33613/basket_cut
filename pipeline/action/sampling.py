"""Sample full temporal windows at clip boundaries without inventing source time."""
import numpy as np


def sample_frame_indices(center_frame, frame_count, clip_len, frame_interval):
    if frame_count < 1 or clip_len < 1 or frame_interval < 1 or not 0 <= center_frame < frame_count:
        raise ValueError("Invalid action sampling window")
    raw = center_frame - (clip_len // 2 - 1) * frame_interval + np.arange(clip_len) * frame_interval
    return np.clip(raw, 0, frame_count - 1), bool((raw < 0).any() or (raw >= frame_count).any())
