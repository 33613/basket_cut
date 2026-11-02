"""Goal timestamps -> MOTIP tracks -> SlowFast shot evidence -> shooter handoff."""

import argparse
from dataclasses import asdict, fields
import fcntl
import json
from pathlib import Path
import subprocess
import sys

from contracts.execution import ExecutionSettings, normalize_omp_threads
from pipeline.player_link.events import LinkOptions, load_events
from workflows import link_player as flow


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path)
    p.add_argument("--annotations", type=Path)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--match-id", default="pl-nba-goals-001")
    p.add_argument("--env-file", type=Path)
    p.add_argument("--tracking-root", type=Path, help="Read-only reuse of verified cli.export_goal_test tracking")
    p.add_argument("--review-file", type=Path, help="Manual CSV decisions; preserves automatic link.json")
    p.add_argument("--stage", choices=("all", "prepare", "track", "action", "link", "export", "pack", "status"), default="all")
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
    root.parent.mkdir(parents=True, exist_ok=True)
    # Controller owns the whole run. Internal GPU workers are sequential subprocesses.
    with root.with_name("."+root.name+".link.lock").open("a") as lock:
        if not args.worker:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("This link_player run is already active") from exc
        if args.stage in ("all", "prepare"):
            flow.prepare(args.input_dir, args.annotations, root, args.match_id, options, settings, args.tracking_root)
        stages = ("track", "action", "link", "export", "pack") if args.stage == "all" else (args.stage,)
        for stage in stages:
            print(f"[link_player] {stage}", flush=True)
            if stage in ("track", "action") and not args.worker:
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
