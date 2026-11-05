"""One-ball motion association; predictions are explicitly not observations."""

import math
from collections import deque
from statistics import median


class BallTracker:
    def __init__(self, max_gap_s=0.12, gate_diagonals=0.01, max_size_ratio=2.5):
        self.max_gap_s = max_gap_s
        self.gate_diagonals = gate_diagonals
        self.max_size_ratio = max_size_ratio
        self.next_id = 0
        self.reset()

    def reset(self):
        self.track_id = None
        self.last = None
        self.velocity = (0.0, 0.0)
        self.last_observed_s = None
        self.last_update_s = None
        self.sizes = deque(maxlen=5)
        self.velocities = deque(maxlen=3)

    def update(self, detections, time_s, width, height):
        if self.last_update_s is not None and time_s <= self.last_update_s:
            raise ValueError("Ball updates must have strictly increasing timestamps")
        self.last_update_s = time_s
        if self.last is not None and time_s - self.last_observed_s > self.max_gap_s:
            self.reset()
            self.last_update_s = time_s
        diagonal = math.hypot(width, height)
        candidates = []
        for detection in detections:
            x1, y1, x2, y2 = detection["bbox_xyxy"]
            size = math.hypot(x2 - x1, y2 - y1)
            if (
                self.sizes
                and max(size / median(self.sizes), median(self.sizes) / size)
                > self.max_size_ratio
            ):
                continue
            candidates.append(((0.5 * (x1 + x2), 0.5 * (y1 + y2)), detection))
        selected = None
        if self.last is None:
            if candidates:
                selected = max(candidates, key=lambda item: item[1]["score"])
        else:
            dt = time_s - self.last_observed_s
            predicted = tuple(p + v * dt for p, v in zip(self.last, self.velocity))
            gate = (
                diagonal * self.gate_diagonals
                + 0.75 * median(self.sizes)
                + math.hypot(*self.velocity) * dt * 0.15
            )
            if candidates:
                best = min(candidates, key=lambda item: math.dist(item[0], predicted))
                if math.dist(best[0], predicted) <= gate:
                    selected = best
        if selected is not None:
            center, detection = selected
            if self.last is None:
                self.track_id = self.next_id
                self.next_id += 1
            else:
                dt = time_s - self.last_observed_s
                measured = tuple((p - q) / dt for p, q in zip(center, self.last))
                self.velocities.append(measured)
                robust = tuple(
                    median(v[axis] for v in self.velocities) for axis in (0, 1)
                )
                self.velocity = (
                    tuple(0.5 * v + 0.5 * m for v, m in zip(self.velocity, robust))
                    if len(self.velocities) > 1
                    else robust
                )
            self.last, self.last_observed_s = center, time_s
            box = detection["bbox_xyxy"]
            self.sizes.append(math.hypot(box[2] - box[0], box[3] - box[1]))
            return {
                "ball_track_id": self.track_id,
                "center_xy": list(center),
                "observed": True,
                "score": detection["score"],
                "bbox_xyxy": detection["bbox_xyxy"],
            }
        if self.last is None:
            return None
        dt = time_s - self.last_observed_s
        return {
            "ball_track_id": self.track_id,
            "center_xy": [p + v * dt for p, v in zip(self.last, self.velocity)],
            "observed": False,
            "score": None,
            "bbox_xyxy": None,
        }


def associate_rows(
    rows,
    width,
    height,
    *,
    threshold=0.15,
    max_gap_s=0.12,
    min_confirm_frames=3,
    max_size_ratio=2.5,
):
    """Offline confirmation: short seeds remain candidates, never trusted tracks.

    Raw detections stay unchanged. Observed points in a confirmed segment are
    still hypotheses, not ground truth. Motion/size gates handle isolated jumps;
    the median velocity limits the effect of a single accepted noisy point.
    """
    if min_confirm_frames < 2 or max_size_ratio <= 1:
        raise ValueError("Need at least two confirmation frames and size ratio > 1")
    tracker = BallTracker(max_gap_s, max_size_ratio=max_size_ratio)
    groups = {}
    result = []
    for row in rows:
        row = dict(row)
        balls = [
            d
            for d in row["detections"]
            if d["label"] == "ball" and d["score"] >= threshold
        ]
        track = tracker.update(balls, row["source_timestamp_s"], width, height)
        row["ball_track"] = track
        result.append(row)
        if track is not None:
            groups.setdefault(track["ball_track_id"], []).append(row)
    for group in groups.values():
        run, confirmed = [], False
        for row in group:
            if row["ball_track"]["observed"]:
                run.append(row["source_frame_idx"])
                if len(run) > 1 and run[-1] != run[-2] + 1:
                    run = run[-1:]
                if len(run) >= min_confirm_frames:
                    confirmed = True
            else:
                run = []
        for row in group:
            row["association_status"] = (
                "confirmed_segment" if confirmed else "unconfirmed_segment"
            )
            if confirmed:
                row["ball_track"]["confirmed"] = True
            else:
                row["tentative_ball_track"] = row["ball_track"]
                row["ball_track"] = None
    for row in result:
        row.setdefault("association_status", "no_track")
    return result
