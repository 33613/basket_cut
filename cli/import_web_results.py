"""Register completed CLI runs in the dashboard without running any models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", required=True, type=Path)
    parser.add_argument('--env-file', type=Path)
    parser.add_argument("--name")
    parser.add_argument(
        "--prepare-media", action="store_true", help="CPU-only H.264 preview conversion"
    )
    args = parser.parse_args()
    if args.env_file:
        from cli.serve_web import load_environment
        load_environment(args.env_file)
    from web.backend.imports import import_results, prepare_browser_media
    from web.backend.settings import WebSettings
    from web.backend.store import ProjectStore

    settings = WebSettings()
    settings.ensure_directories()
    store = ProjectStore(settings.data_root, settings.output_root)
    manifest = import_results(settings, store, args.result_dir, args.name)
    media = prepare_browser_media(settings, manifest) if args.prepare_media else []
    print(
        json.dumps(
            {
                "project_id": manifest["project_id"],
                "name": manifest["name"],
                "video_count": len(manifest["videos"]),
                "read_only": True,
                "prepared_media": media,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
