"""External goal annotations and source-frame window planning; no model imports."""

import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re

from pipeline.media import probe_video, sha256_file, verify_frame_clock


@dataclass(frozen=True)
class LinkOptions:
    lookback_s: float = 4.0
    max_lookback_s: float = 8.0
    post_roll_s: float = .5
    min_flight_s: float = .1
    score_threshold: float = .35
    min_margin: float = .10
    min_support: int = 2
    evidence_gap_s: float = .35
    segment_padding_s: float = .5
    tracking_gap_s: float = .25
    samples_per_track: int = 12
    action_step: int = 2
    action_reference_fps: float = 25.0
    min_track_observed_s: float = 0.0

    def validate(self):
        values = asdict(self)
        if any(not math.isfinite(float(v)) for v in values.values()):
            raise ValueError("Options must be finite")
        if not 0 < self.lookback_s <= self.max_lookback_s or self.post_roll_s < 0:
            raise ValueError("Require 0 < lookback <= max_lookback and post_roll >= 0")
        if not 0 <= self.min_flight_s < self.lookback_s:
            raise ValueError("Invalid minimum flight search exclusion")
        if not 0 <= self.score_threshold <= 1 or not 0 <= self.min_margin <= 1:
            raise ValueError("Score threshold and margin must be in [0,1]")
        if min(self.min_support, self.samples_per_track, self.action_step) < 1:
            raise ValueError("Counts and action_step must be positive")
        if min(self.evidence_gap_s, self.tracking_gap_s, self.action_reference_fps) <= 0 or self.segment_padding_s < 0:
            raise ValueError("Invalid temporal settings")
        if self.min_track_observed_s < 0:
            raise ValueError("Minimum observed track time must be nonnegative")
        return self


def load_events(input_dir, annotations, match_id, options, *, ffprobe="ffprobe"):
    options.validate()
    root = Path(input_dir).resolve()
    if not match_id.strip():
        raise ValueError("match_id cannot be empty")
    path = Path(annotations)
    if path.suffix.lower() == ".json":
        rows = json.loads(path.read_text())
        if not isinstance(rows, list):
            raise ValueError("Goal JSON must be a list of event objects")
    else:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("No annotated goal events")
    sources, events, ids, source_paths = {}, [], set(), {}
    for row in rows:
        vid = str(row["video_id"])
        if not re.fullmatch(r"[A-Za-z0-9_-]+", vid):
            raise ValueError(f"Unsafe video_id: {vid}")
        source = (root / row["filename"]).resolve()
        if not source.is_relative_to(root) or not source.is_file():
            raise ValueError(f"Missing/unsafe video: {source}")
        if vid in source_paths and source_paths[vid] != source:
            raise ValueError("One video_id cannot name different files")
        source_paths[vid] = source
        if vid not in sources:
            meta = probe_video(source, ffprobe=ffprobe)
            count = verify_frame_clock(source, meta["fps"], ffprobe)
            sources[vid] = dict(video_id=vid, path=str(source), sha256=sha256_file(source),
                                frame_count=count, **{k: meta[k] for k in ("fps", "width", "height", "duration_s")})
        meta = sources[vid]
        goal = float(row["goal_time_s"])
        if not math.isfinite(goal) or not 0 <= goal < meta["duration_s"]:
            raise ValueError(f"{vid}: goal_time_s={goal} outside [0,{meta['duration_s']})")
        token = hashlib.sha256(json.dumps([match_id, vid, goal]).encode()).hexdigest()[:16]
        eid = str(row.get("event_id") or f"EV-{token}")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", eid) or eid in ids:
            raise ValueError(f"Duplicate/unsafe event_id: {eid}")
        ids.add(eid)
        start = max(0, math.floor((goal-options.max_lookback_s)*meta["fps"]))
        end = min(meta["frame_count"], math.ceil((goal+options.post_roll_s)*meta["fps"]))
        end = max(start+1, end)
        events.append(dict(event_id=eid, video_id=vid, goal_time_s=goal,
            annotation_source=str(row.get("annotation_source") or "external_goal_timestamp"),
            window_start_frame=start, window_end_frame_exclusive=end,
            window_start_s=start/meta["fps"], window_end_s=end/meta["fps"],
            partial_pre_context=goal-options.max_lookback_s < 0,
            partial_post_context=goal+options.post_roll_s > meta["frame_count"]/meta["fps"],
            evaluation_labels={k: row[k] for k in ("jersey_number", "team_id", "person_gt_id",
                "evaluation_player_name", "evaluation_cohort", "action_label") if row.get(k)}))
    return dict(schema_version=1, match_id=match_id, options=asdict(options),
                sources=sources, events=events, algorithm_version="action-evidence-v1")
