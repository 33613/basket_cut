# Second experiment: ball-conditioned shooter attribution

Run only on `link_player`. `main` and upstream MOTIP/MMAction2 remain unchanged.
Use a **new output directory**; the action-only first experiment remains read-only.

## Evidence and scope

The ball-enabled default is a 5s source window `[goal-4, goal+1)`, clipped to
the actual video. Attribution examines the last 3s before the external goal.
MOTIP raw IDs are preserved by verifying/reusing the full-clip export.
First-round action centers are rebased into the new source window; their original
wider inference context and temporal-padding flags remain recorded. This is a
controlled ball-evidence comparison, not a fresh SlowFast experiment.

The ranker:

1. Uses only observed, temporally confirmed ball points. Predicted/red points
   cannot establish contact, release or rim proximity.
2. Finds a path with at least two supported near-rim observations close to the
   external goal time. This is a visual path anchor, not made-basket verification.
3. Searches observed player boxes for stable ball/upper-body proximity, followed
   by separation on that anchored ball path. Short relative-motion stability
   helps reject a flying ball merely crossing a defender's box.
4. Allows hand-to-rim candidates without an airborne arc, for dunks/putbacks.
5. Excludes earlier non-overlapping control/departure candidates when a later
   supported control/release exists on the same path (possible pass -> shooter).
6. Combines the ball proxy score (80%) and nearby shot-action score (20%). These
   initial heuristic weights are **not calibrated probabilities**. Weak action
   scores do not automatically eliminate an otherwise ball-supported actor.
7. Keeps ambiguous and unsupported cases unresolved. The action-only reference
   remains inside `link.json` for inspection; it does not silently override a
   missing ball path.

Contact currently uses an expanded upper-body box, not hand pose. Persistent
false ball detections, occlusion, ball-track fragmentation, overlapping bodies
and inaccurate goal times can still require review. No jersey/player annotation
is used in prediction. `ball_verified` and `made_shot_verified` stay false.

Handoff starts no earlier than the supported contact run, reducing the risk of
including a different person under a reused ID such as video 3/T6. This does not
prove identity purity; player identity remains downstream and manually reviewable.
Within that local segment, time bins select sharper, larger crops with stronger
detection scores; all original-size crops remain available. This heuristic does
not certify visibility, jersey readability or identity correctness.

## AutoDL second run

After pulling `link_player`, run from `/root/autodl-tmp/basket_cut`:

```bash
nohup env OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /root/autodl-tmp/env/control/bin/python -u -m cli.link_player \
  --env-file /root/autodl-tmp/tool/manifests/basket-autodl.env \
  --input-dir /root/autodl-fs/data/goal-demo-input \
  --annotations /root/autodl-fs/data/goal-demo-input/annotations.csv \
  --output-dir /root/autodl-fs/link-player-demo-002 \
  --match-id pl-nba-goals-001 \
  --tracking-root /root/autodl-fs/test \
  --reuse-run /root/autodl-fs/link-player-demo-001 \
  --ball-python /root/autodl-tmp/env/ball/bin/python \
  --ball-checkpoint /root/autodl-tmp/model/ball/rf-detr-basketball.pth \
  --ball-cache 37=/root/autodl-fs/ball-demo-37-002 \
  > /root/autodl-tmp/tool/manifests/link-player-demo-002.log 2>&1 < /dev/null &
```

The video-37 cache must cover the requested `[6,11)` interval and match the
source/checkpoint hashes. It is reassociated with the current continuity code;
the other nine videos run the ball detector in isolated subprocesses.
No MOTIP, SlowFast, KPR or player-library inference is rerun by this command.
Checkpoints/configs for cached MOTIP are still checked against its original marker.

```bash
/root/autodl-tmp/env/control/bin/python -m cli.link_player \
  --output-dir /root/autodl-fs/link-player-demo-002 --stage status
```

Resume with the same all-stage arguments after a cleanly completed stage. Stages
with verified markers are skipped. A nonempty partial ball-inference directory
without a completion marker is not overwritten; investigate its failure first.

## Review and deliverables

- `comparison.csv`: first vs second automatic status and selected track ID,
  ball-path support and diagnostics. This is not an accuracy table.
- `review.html`: review each event, candidates and target crops.
- `events/EV-*/evidence.mp4`: person IDs/actions, ball/rim observations, and a
  white box/SHOOTER CANDIDATE label on an automatically selected actor.
- `events/EV-*/ball/`: raw detection/trajectory JSONL, ball-only video and provenance.
- `events/EV-*/link.json`: contact interval, separation/rim evidence, score components,
  discarded earlier controls, and the action-only reference.
- `events/EV-*/handoff/`: local target segment, original-size PNGs and source clock.
- `review.csv`: separate human decisions. First-run reviews are not automatically
  imported or treated as new predictions.
- `link-player-demo-002.tar` and checksum: generated after all stages validate.

Source ball indices refer to original videos; person/action indices refer to
the event window. The only mapping is `source_frame = local_frame + window_start_frame`.
Do not reuse manual review frame bounds from the wider first experiment without rebasing.

CPU checks: `python -m analysis.link_ball_selfcheck` covers weak-action actor 0,
GT isolation, predicted-point exclusion, ambiguity, dunk/contact handling,
passer exclusion, source-offset rebasing, two-pass resume, combined video export,
tar packaging and preservation of all first-run files. It uses injected detections,
not a GPU accuracy benchmark.
