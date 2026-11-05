"""Goal timestamps -> MOTIP tracks -> SlowFast shot evidence -> shooter handoff."""

import argparse
from dataclasses import asdict, fields
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

from contracts.execution import ExecutionSettings, normalize_omp_threads
from pipeline.player_link.events import LinkOptions, load_events
from workflows import link_player as flow
from pipeline.media import sha256_file


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path)
    p.add_argument("--annotations", type=Path)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--match-id", default="pl-nba-goals-001")
    p.add_argument("--env-file", type=Path)
    p.add_argument("--tracking-root", type=Path, help="Read-only reuse of verified cli.export_goal_test tracking")
    p.add_argument("--review-file", type=Path, help="Manual CSV decisions; preserves automatic link.json")
    p.add_argument('--ball-checkpoint',type=Path,help='Enable ball-contact/rim attribution in a new run')
    p.add_argument('--ball-python',type=Path)
    p.add_argument('--ball-cache',action='append',default=[],metavar='VIDEO_ID=DIRECTORY')
    p.add_argument('--reuse-run',type=Path,help='Read-only reuse of first-run action inference')
    p.add_argument("--stage", choices=("all", "prepare", "track", "action", "ball", "link", "export", "pack", "status"), default="all")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    for field in fields(LinkOptions):
        p.add_argument("--"+field.name.replace("_", "-"), type=type(field.default), default=None)
    args = p.parse_args()
    if args.env_file:
        from cli.serve_web import load_environment
        load_environment(args.env_file)
    normalize_omp_threads()
    settings = ExecutionSettings()
    root = args.output_dir.expanduser().resolve()
    if args.stage == "status":
        print(json.dumps(flow.status(root), ensure_ascii=False, indent=2))
        return
    values = flow.saved(root)["options"] if (root / "protocol.json").exists() else asdict(LinkOptions())
    if args.ball_checkpoint and not (root/'protocol.json').exists():
        values.update(lookback_s=3.,max_lookback_s=4.,post_roll_s=1.)
    values = dict(values, **{f.name:getattr(args,f.name) for f in fields(LinkOptions) if getattr(args,f.name) is not None})
    options = LinkOptions(**values).validate()
    if (root / "protocol.json").exists() and flow.saved(root)["options"] != asdict(options):
        raise ValueError("Run options changed; use a new output directory")
    if args.stage in ("prepare", "all") or args.dry_run:
        if not args.input_dir or not args.annotations:
            p.error("--input-dir and --annotations required for prepare/all/dry-run")
    if args.dry_run:
        protocol = load_events(args.input_dir, args.annotations, args.match_id, options, ffprobe=settings.ffprobe)
        print(json.dumps(dict(dry_run=True, **protocol), ensure_ascii=False, indent=2))
        return
    existing=flow.saved(root) if (root/'protocol.json').exists() else {}
    ball_config=existing.get('ball')
    if args.ball_checkpoint:
        cache={}
        for item in args.ball_cache:
            if '=' not in item:
                p.error('--ball-cache requires VIDEO_ID=DIRECTORY')
            vid,directory=item.split('=',1)
            if vid in cache:
                p.error('Duplicate ball cache video ID')
            cache[vid]=str(Path(directory).expanduser().resolve())
        interpreter=args.ball_python or Path(os.environ.get('BASKET_BALL_PYTHON',sys.executable))
        ball_config=dict(checkpoint=str(args.ball_checkpoint.resolve()),
            checkpoint_sha256=sha256_file(args.ball_checkpoint),
            python=str(interpreter.expanduser().absolute()),cache=cache)
        if existing and existing.get('ball')!=ball_config:
            p.error('Ball configuration changed; use a new output directory')
    elif args.ball_cache or args.ball_python:
        p.error('--ball-cache/--ball-python require --ball-checkpoint')
    reuse_run=args.reuse_run or (Path(existing['reuse_run']) if existing.get('reuse_run') else None)
    if reuse_run and not ball_config:
        p.error('--reuse-run is reserved for the ball-evidence comparison run')
    tracking_root=args.tracking_root
    if reuse_run and tracking_root is None:
        old=flow.saved(reuse_run)
        if not old.get('tracking_root'):
            p.error('Prior run has no full-clip tracking cache; supply --tracking-root explicitly')
        tracking_root=Path(old['tracking_root'])
    root.parent.mkdir(parents=True, exist_ok=True)
    # Controller owns the whole run. Internal GPU workers are sequential subprocesses.
    with root.with_name("."+root.name+".link.lock").open("a") as lock:
        if not args.worker:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("This link_player run is already active") from exc
        if args.stage in ("all", "prepare"):
            flow.prepare(args.input_dir, args.annotations, root, args.match_id, options, settings, tracking_root,
                         ball_config=ball_config,reuse_run=reuse_run)
        stages = (("track","action","ball","link","export","pack") if ball_config else
                  ("track", "action", "link", "export", "pack")) if args.stage == "all" else (args.stage,)
        for stage in stages:
            print(f"[link_player] {stage}", flush=True)
            cached_stage=(stage=='track' and flow.saved(root).get('tracking_root')) or (stage=='action' and flow.saved(root).get('reuse_run'))
            if stage in ("track", "action") and not args.worker and not cached_stage:
                python = settings.motip_python if stage == "track" else settings.action_python
                command = [str(python), "-u", "-m", "cli.link_player", "--output-dir", str(root),
                           "--stage", stage, "--worker"]
                if args.env_file:
                    command += ["--env-file", str(args.env_file.resolve())]
                subprocess.run(command, cwd=settings.repository_root, check=True)
            elif stage == "track":
                flow.track(root, settings)
            elif stage == "action":
                flow.action(root, settings)
            elif stage == 'ball':
                flow.ball(root,settings)
            elif stage == "link":
                flow.link(root, options)
            elif stage == "export":
                from pipeline.player_link.export import export_run
                export_run(root, options, settings, args.review_file)
            elif stage == "pack":
                from pipeline.player_link.export import pack_run
                print(json.dumps(pack_run(root), indent=2))


if __name__ == "__main__":
    main()
