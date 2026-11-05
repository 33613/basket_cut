"""One-ball motion association; predictions are explicitly not observations."""

import math


class BallTracker:
    def __init__(self, max_gap_s=0.12, gate_diagonals=0.06):
        self.max_gap_s = max_gap_s
        self.gate_diagonals = gate_diagonals
        self.next_id = 0
        self.reset()

    def reset(self):
        self.track_id = None
        self.last = None
        self.velocity = (0.0, 0.0)
        self.last_observed_s = None
        self.last_update_s = None

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
            candidates.append(((0.5 * (x1 + x2), 0.5 * (y1 + y2)), detection))
        selected = None
        if self.last is None:
            if candidates:
                selected = max(candidates, key=lambda item: item[1]["score"])
        else:
            dt = time_s - self.last_observed_s
            predicted = tuple(p + v * dt for p, v in zip(self.last, self.velocity))
            gate = (
                diagonal * self.gate_diagonals + math.hypot(*self.velocity) * dt * 0.5
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
                self.velocity = tuple(
                    0.5 * v + 0.5 * m for v, m in zip(self.velocity, measured)
                )
            self.last, self.last_observed_s = center, time_s
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
