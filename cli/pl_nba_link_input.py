"""Convert PL-NBA action intervals into uncertain ball/rim search windows."""

import argparse
import csv
import json
import math
from pathlib import Path
import re


def timestamp_bounds(value):
    if not isinstance(value, str) or not re.fullmatch(r"\d+:\d{2}", value):
        raise ValueError(f"Unsupported PL-NBA timestamp: {value!r}")
    seconds, suffix = map(int, value.split(":"))
    if suffix >= 60:
        raise ValueError("Unconfirmed timestamp suffix >=60; cannot form both time hypotheses")
    return seconds + suffix / 100, seconds + suffix / 60


def convert(review, output, flight_padding_s=1.5):
    if not math.isfinite(flight_padding_s) or flight_padding_s < 0:
        raise ValueError("Flight padding must be finite and nonnegative")
    with Path(review).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("Empty timing review")
    converted, seen = [], set()
    for row in rows:
        for key in ("event_id", "video_id", "filename", "raw_action_start", "raw_action_end"):
            if not row.get(key):
                raise ValueError(f"Missing {key}")
        if row['event_id'] in seen:
            raise ValueError("Duplicate event_id")
        seen.add(row['event_id'])
        start = timestamp_bounds(row['raw_action_start'])
        end = timestamp_bounds(row['raw_action_end'])
        if any(b < a for a, b in zip(start, end)):
            raise ValueError(f"Reversed action interval: {row['event_id']}")
        item = dict(row)
        item.pop('goal_time_s', None)
        item.update(
            goal_interval_start_s=max(0., min(start) - 0.25),
            goal_interval_end_s=max(end) + flight_padding_s,
            timestamp_policy="unconfirmed_union_seconds_centiseconds_and_seconds_frames60",
            flight_padding_s=flight_padding_s,
            annotation_source="PL-NBA_action_interval_not_exact_goal",
        )
        converted.append(item)
    text = json.dumps(converted, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    output = Path(output)
    if output.exists() and output.read_text() != text:
        raise ValueError("Output already exists with different inputs; use a new filename")
    output.parent.mkdir(parents=True, exist_ok=True)
    if not output.exists():
        with output.open('x') as stream:
            stream.write(text)
    return dict(events=len(converted), output=str(output),
                measured_goal_times=False, timestamp_units_confirmed=False,
                policy="Conservative union of two time hypotheses; visual rim anchors required")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review-csv', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--flight-padding-s', type=float, default=1.5)
    args = parser.parse_args()
    print(json.dumps(convert(args.review_csv, args.output, args.flight_padding_s),
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
