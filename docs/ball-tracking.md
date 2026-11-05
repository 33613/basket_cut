# Basketball observations and PL-NBA annotations

This is an independent module on `link_player`. It does not modify MOTIP,
the existing action-only run, or the `main` branch. It does not yet attribute
shots to players or detect made baskets. Existing run directories are never overwritten.

## Environment and checkpoint

Use `/root/autodl-tmp/env/ball/bin/python` with `requirements-ball.txt`.
The operator verified torch 2.6.0+cu124, torchvision 0.21.0+cu124,
RF-DETR 1.10.1 and CUDA on the RTX 4090. A checkpoint loaded as RFDETRSmall
with classes ball, holder, rim, scorer, player, referee, ball-in-basket.
The operator subsequently ran video 37, source frames 360–659, on the 4090:
300 frames processed; ball detections on 280 frames, rims on 300, v1 accepted
observations on 250 and predictions on 50. Visual review reported usable ball
tracking with a short false association around player occlusion. These are
coverage counts and qualitative feedback, not accuracy measurements.

Checkpoint source: [basketball RF-DETR](https://github.com/feifeifei85/OpenHoopVision/blob/ec969037b2c02b983a77b260c50cb9eb36fe9fc1/weights/rf-detr-basketball.pth).
Expected bytes: 129499889. SHA-256:
`69409cbefc4b92c3cd194fb7f877c6e58e52548909e05e3b5467fe0e2fe188c5`.
Keep the upstream license and provenance when distributing its weights.
The association code in `pipeline/ball/tracker.py` is a small local motion gate,
not a copy of the upstream event pipeline. It has no separate learned weights.

The previously suggested media download URL returned 404. If the file already
exists and matches the hash, use it; a loading success alone does not verify its hash.

## First GPU run

Run from the repository root. Reuse the manually reviewed video 37, goal at 10s.
Process the 5s window [6,11), then inspect the ball path during [7,10].
RF-DETR runs on **every native source frame**, preserving short releases.

```bash
/root/autodl-tmp/env/ball/bin/python -u -m cli.ball_track \
  --input /root/autodl-fs/data/goal-demo-input/37.mp4 \
  --checkpoint /root/autodl-tmp/model/ball/rf-detr-basketball.pth \
  --output-dir /root/autodl-fs/ball-demo-37-001 \
  --start-s 6 --end-s 11 --goal-time-s 10
```

Outputs:

- `ball.jsonl`: all detector classes, accepted rim boxes and selected ball track;
  original zero-based `source_frame_idx`, original seconds and window-local frame index.
- `ball_vis.mp4`: silent H.264 preview. All ball candidate boxes gray, accepted
  ball boxes green, rim boxes blue, accepted centers yellow and predictions red.
- `summary.json`: source/checkpoint hashes, package version, dimensions, options,
  detection counts and explicit unevaluated status. Counts are not accuracy.

`observed=false` means a brief motion prediction without a detected ball.
Never use those points as independent evidence of touching a hand or entering a rim.
All raw detections are retained so future offline association can revisit a mistaken
single-ball choice. Do not extrapolate across long occlusion or camera cuts.
For the initial test choose a window without a camera cut. The module checks CFR
timing before using `source_frame_idx / fps`; it refuses irregular frame clocks.

## Continuity v2 and cached replay

V1 used a gate equal to 6% of the frame diagonal and could accept background
objects after occlusion. V2 uses 1% of the diagonal plus a term based on recent
median ball-box size and motion uncertainty; candidates with size ratios beyond
2.5 are excluded. Velocity uses a rolling median of displacement estimates.
Rejected detections and predicted positions do not update observation/velocity
history. New segments need at least 3 consecutive observed source frames;
short seeds remain under `tentative_ball_track`, with `ball_track=null`.
Confirmation is performed offline and can retrospectively retain the first
observations of a supported segment. It confirms temporal support, not visual
correctness. Persistent background objects and plausible close false detections
can still survive. No parabolic-path constraint is imposed on dunks or bounces.

Rebuild the existing 5s run without any GPU inference:

```bash
/root/autodl-tmp/env/ball/bin/python -m cli.ball_track \
  --input /root/autodl-fs/data/goal-demo-input/37.mp4 \
  --reuse-detections /root/autodl-fs/ball-demo-37-001 \
  --output-dir /root/autodl-fs/ball-demo-37-002 \
  --min-confirm-frames 3
```

The CLI inherits the original window and options, and loads no RF-DETR model.
It checks source/checkpoint hashes, complete contiguous frame IDs, row count and
original timestamps; it refuses windows outside the cache and detector resolution
changes. Thresholds below the original detector floor require fresh inference.
The old run is read-only. New summary records the reused JSONL hash. No package
upgrade or model download is needed. Compare actual false associations and
reacquisition, not just the number of accepted observations (which may decrease).

## PL-NBA source audit

Source: [holhouse/PL-NBA-Dataset](https://github.com/holhouse/PL-NBA-Dataset),
revision `0b3d5013b5828004062ed27b10c9eddfca313711`, `PL-NBA-JSON.zip`.
ZIP SHA-256: `391948eef9fd8b4cecfc58a6088edf1253b0ce4bc5531edd7147d15a8f94ee48`.
This downloaded snapshot contains 6408 JSON files from 32 games, including 2817
made-shot events. These are counts from this archive, not the README's full dataset claims.
Made classes: Layup 872, 3ptShot 792, FreeThrow 468, 2ptShot 355, Dunk 255,
PutBack/Putback 75. Select `result=made` inside JSON events rather than
trusting the possession filename, which can disagree with event contents.

Events provide `event_label`, `player`, `result`, and `timestamp_video.start/end`.
These are **action intervals**, not a separate rim-crossing frame. The README does
not define the colon timestamp units. For example `03:30` is 3.5s under
seconds:frames at 60fps, but 3.3s under seconds:centiseconds. A 60fps video alone
does not establish the annotation clock. Verify on real decoded frames before conversion.
One archive timestamp has suffix 60, so a blanket 60fps parser would also need review.

```bash
# Read the official labels without guessing timestamp units or making goal times.
/root/autodl-tmp/env/control/bin/python -m cli.pl_nba_annotations \
  --annotation-zip /root/autodl-fs/data/pl-nba-annotations/PL-NBA-JSON.zip \
  --match-id 20250501_LAL_MIN \
  --video-ids 3 6 20 43 50 68 11 30 37 63 \
  --output-dir /root/autodl-fs/pl-nba-label-audit-001
```

The default keeps original timestamp strings; `goal_time_s` stays null.
`--time-format seconds-frames --timestamp-fps 60` or
`--time-format seconds-centiseconds` are explicit operator assumptions, not
assertions that these are the dataset conventions. Even with conversion,
`action_end_s` is not silently copied into `goal_time_s`.
The exporter requires match ID when selecting numeric clip IDs, as they repeat
across games. Match the original basenames/video content before using annotations.

Examples of inconsistencies for the user's candidate game:

- Clip 3 names Alex Len while manual review identifies jersey 27; retain this as
  a disagreement, not a label correction inferred from metadata alone.
- Clip 6 is labelled Layup while manual review reports a dunk.
- Clip 11's filename says attack_failed but contains a made Layup event.

Player names and action types remain evaluation labels. Do not feed them to
the visual shooter/identity prediction. Preserve the ten existing manual goal
times and verified shooter IDs as a separate evaluation source.

## Verification

`python -m analysis.ball_selfcheck`: CPU-only association and real media/JSONL/H.264
integration with injected detector outputs; timestamp format, label filtering,
match scoping and source offsets. This does not measure the GPU detector's accuracy.
