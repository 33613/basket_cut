"""Export manually annotated goal clips for independent tracking/ReID/library tests."""

import argparse
from contextlib import ExitStack
import fcntl
import json
from pathlib import Path
import subprocess
import sys

from contracts.execution import ExecutionSettings, normalize_omp_threads
from pipeline.media import atomic_json, sha256_file
from workflows import goal_test_export as flow


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("template", "all", "prepare", "track", "crops", "features", "shooter", "pack", "status"), default="all")
    p.add_argument("--input-dir", type=Path)
    p.add_argument("--annotations", type=Path)
    p.add_argument("--output-dir", type=Path, required=True, help="Dedicated export folder, e.g. /root/autodl-fs/test")
    p.add_argument("--match-id", default="pl-nba-goals-001")
    p.add_argument("--env-file", type=Path)
    p.add_argument("--selections", type=Path, help="CSV with video_id,track_id and optional inclusive start/end_frame")
    p.add_argument("--samples-per-track", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.samples_per_track < 1 or args.batch_size < 1:
        p.error("Sample count and batch size must be positive")
    if args.env_file:
        from cli.serve_web import load_environment
        load_environment(args.env_file)
    normalize_omp_threads()
    settings = ExecutionSettings()
    root = args.output_dir.expanduser().resolve()
    if args.stage == "template":
        flow.seed_annotations(root)
        print(f"Annotation CSV: {root}")
        return
    if args.stage == "status":
        print(json.dumps(flow.status(root), ensure_ascii=False, indent=2))
        return
    # A sibling lock avoids modifying export contents and also covers prepare.
    root.parent.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        names = [] if args.worker else ["controller"]
        if args.stage != "all":
            names.append("export")
        for name in names:
            lock = stack.enter_context(root.with_name(f".{root.name}.{name}.lock").open("a"))
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("This export is already running") from exc
        run(args, root, settings, p)


def run(args, root, settings, parser):
    if args.stage in ("prepare", "all"):
        if not args.input_dir or not args.annotations:
            parser.error("prepare/all requires --input-dir and --annotations")
        flow.prepare(args.input_dir, args.annotations, root, args.match_id, ffprobe=settings.ffprobe)
    if args.stage == "prepare":
        return
    if args.stage == "all":
        # Sequential separate interpreters release GPU memory between MOTIP and KPR.
        for stage, python in (("track", settings.motip_python), ("crops", sys.executable),
                              ("features", settings.kpr_python), ("pack", sys.executable)):
            print(f"[goal-test] {stage}", flush=True)
            command = [str(python), "-u", "-m", "cli.export_goal_test", "--stage", stage,
                       "--worker",
                       "--output-dir", str(root), "--samples-per-track", str(args.samples_per_track),
                       "--batch-size", str(args.batch_size)]
            if args.env_file:
                command += ["--env-file", str(args.env_file.resolve())]
            subprocess.run(command, cwd=settings.repository_root, check=True)
        return
    flow.validate_videos(root)
    videos = flow.manifest(root)["videos"]
    if args.stage == "track":
        config = settings.repository_root / "MOTIP/configs/r50_deformable_detr_motip_sportsmot.yaml"
        protocol = dict(checkpoint_sha256=sha256_file(settings.motip_checkpoint),
                        config_sha256=sha256_file(config), tracker_options="cli.tracking defaults v1")
        for video in videos:
            vid = video["video_id"]
            folder = root / "track" / vid
            marker = folder / "tracking_complete.json"
            expected = dict(**protocol, source_sha256=video["source_sha256"])
            if marker.exists():
                old = flow.read_json(marker)
                if old["protocol"] != expected or old["outputs"] != flow.track_signature(root, vid):
                    raise ValueError("Tracking inputs or outputs changed; use a new output directory")
                if sha256_file(folder / "tracks_vis.mp4") != old["visualization_sha256"]:
                    raise ValueError("Tracking preview changed")
                print(f"[track] {vid}: verified existing result", flush=True)
                continue
            # Incomplete tracking is regenerated; no feature/crop stage can have completed yet.
            command = [str(settings.motip_python), "-u", "-m", "cli.tracking", "--input",
                       str(root / video["video_path"]), "--video-id", vid, "--checkpoint",
                       str(settings.motip_checkpoint), "--output-dir", str(folder),
                       "--motip-root", str(settings.repository_root / "MOTIP"),
                       "--config", str(config), "--overwrite"]
            subprocess.run(command, cwd=settings.repository_root, check=True)
            flow.validated_tracks(root, video)
            # The shared renderer produces mp4v. Convert this derived preview for browser playback;
            # KPR and PNG crops always use the unchanged original video.
            preview = folder / ".tracks_vis.h264.mp4"
            subprocess.run([settings.ffmpeg, "-nostdin", "-v", "error", "-y", "-i",
                str(folder / "tracks_vis.mp4"), "-an", "-c:v", "libx264", "-crf", "18",
                "-threads", "2", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(preview)], check=True)
            preview.replace(folder / "tracks_vis.mp4")
            atomic_json(marker, dict(protocol=expected, outputs=flow.track_signature(root, vid),
                                    visualization_sha256=sha256_file(folder / "tracks_vis.mp4")))
    elif args.stage == "crops":
        for video in videos:
            if not (root / "track" / video["video_id"] / "tracking_complete.json").exists():
                raise ValueError("Tracking stage must complete before crops")
            print(f"[crops] {video['video_id']}", flush=True)
            flow.sample_tracks(root, video, samples_per_track=args.samples_per_track)
        flow.review(root)
    elif args.stage == "features":
        config = settings.repository_root / "configs/kpr/multidataset_sports_test.yaml"
        backend = None
        for video in videos:
            protocol = flow.feature_protocol(root, video, settings.kpr_checkpoint, config)
            if flow.features_complete(root, video, protocol):
                print(f"[features] {video['video_id']}: verified existing result", flush=True)
                continue
            if backend is None:
                from pipeline.identity.kpr_backend import KPRBackend
                backend = KPRBackend(kpr_root=settings.repository_root / "KPR", config_path=config,
                                     checkpoint_path=settings.kpr_checkpoint, prompt_mode="none")
            print(f"[features] {video['video_id']}", flush=True)
            flow.features(root, video, backend, protocol, batch_size=args.batch_size)
    elif args.stage == "shooter":
        count = flow.export_shooters(root, args.selections or root / "shooter_selection.csv", ffmpeg=settings.ffmpeg)
        flow.review(root)
        print(f"Manually selected shooters exported: {count}/{len(videos)}. Run --stage pack again.")
    elif args.stage == "pack":
        print(json.dumps(flow.pack(root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
