"""Process a frozen video subset using existing independent model environments."""

import argparse
import json
from pathlib import Path
from contracts.execution import ExecutionSettings
from workflows.batch import run_batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument('--env-file', type=Path, help='Private BASKET_* config, same file as serve_web')
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--target", choices=("identity", "full"), default="full")
    parser.add_argument('--min-free-gb', type=float, default=2, help='Stop stages before using the configured disk reserve')
    parser.add_argument("--merge-distance", type=float)
    parser.add_argument("--cross-clip-distance", type=float,
                        help="After all clips complete, build a match-scoped KPR library; does not change per-clip IDs")
    parser.add_argument("--jersey-ocr", action="store_true")
    parser.add_argument("--allow-ocr-download", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.env_file:
        from cli.serve_web import load_environment
        load_environment(args.env_file)
    if args.merge_distance is not None and (not __import__('math').isfinite(args.merge_distance) or args.merge_distance < 0):
        parser.error("merge-distance must be finite and nonnegative")
    if args.cross_clip_distance is not None and (not __import__('math').isfinite(args.cross_clip_distance) or args.cross_clip_distance < 0):
        parser.error("cross-clip-distance must be finite and nonnegative")
    if not __import__('math').isfinite(args.min_free_gb) or args.min_free_gb < 0:
        parser.error('min-free-gb must be finite and nonnegative')
    result = run_batch(ExecutionSettings(), args.input_dir, args.output_dir, limit=args.limit,
                       selection=args.selection, target=args.target, force=args.force, dry_run=args.dry_run,
                       options={"min_free_gb": args.min_free_gb, "identity_merge_distance": args.merge_distance,
                                "jersey_ocr": args.jersey_ocr, "allow_ocr_download": args.allow_ocr_download})
    if args.cross_clip_distance is not None:
        if args.dry_run:
            result["person_library_plan"] = {"enabled": True, "max_distance": args.cross_clip_distance,
                                              "scope": "match_candidate", "runs_after_all_clips_complete": True}
        elif result.get("failed_clips"):
            result["person_library_skipped"] = "Not all expected clips completed"
        else:
            from workflows.person_library import PersonLibraryOptions, build_person_library
            result["person_library"] = build_person_library(PersonLibraryOptions(
                args.output_dir, max_distance=args.cross_clip_distance, overwrite=True))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result.get("failed_clips"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
