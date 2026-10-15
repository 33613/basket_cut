"""Print player action/clip references, optionally export playable MP4 clips."""

import argparse
import json
from pathlib import Path

from contracts.execution import ExecutionSettings
from workflows.player_events import query_player_events, export_player_events


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--player-id", required=True)
    parser.add_argument("--event", default="")
    parser.add_argument("--clip-id", default="")
    parser.add_argument("--min-score", type=float, default=0)
    parser.add_argument("--export-dir", type=Path)
    parser.add_argument("--scope", choices=("event", "clip"), default="event")
    parser.add_argument("--media-source", choices=("clip", "original"), default="clip")
    parser.add_argument("--before-s", type=float, default=1)
    parser.add_argument("--after-s", type=float, default=1)
    parser.add_argument("--env-file", type=Path)
    args = parser.parse_args()
    if args.env_file:
        from cli.serve_web import load_environment
        load_environment(args.env_file)
    filters = {"event": args.event, "clip_id": args.clip_id, "min_score": args.min_score}
    if args.export_dir:
        settings = ExecutionSettings()
        result = export_player_events(args.run_dir, args.player_id, args.export_dir,
            scope=args.scope, media_source=args.media_source, before_s=args.before_s, after_s=args.after_s,
            ffmpeg=settings.ffmpeg, ffprobe=settings.ffprobe, **filters)
    else:
        result = query_player_events(args.run_dir, args.player_id, **filters)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
