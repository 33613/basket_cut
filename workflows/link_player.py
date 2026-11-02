"""Stage orchestration for goal-to-shooter attribution. Models run in isolated environments."""

from dataclasses import asdict
from collections import Counter
import json
from pathlib import Path
import subprocess

from contracts.schema import TrackRecord, read_jsonl, write_jsonl_line
from contracts.tracks import load_tracks
from pipeline.media import atomic_json, cut_video, probe_video, sha256_file, verify_frame_clock
from pipeline.player_link.events import load_events
from pipeline.player_link.evidence import rank_shooters


def read(path):
    return json.loads(Path(path).read_text())


def saved(root):
    return read(Path(root) / "protocol.json")


def folder(root, event):
    return Path(root) / "events" / event["event_id"]


def complete(directory, stage, signature, names):
    marker = directory / f".{stage}.json"
    if not marker.exists():
        return False
    old = read(marker)
    if old["signature"] != signature:
        raise ValueError(f"{directory.name}/{stage}: inputs changed; use a new run directory")
    actual = {name: sha256_file(directory / name) for name in names}
    if old["outputs"] != actual:
        raise ValueError(f"{directory.name}/{stage}: saved artifacts changed")
    return True


def mark(directory, stage, signature, names):
    atomic_json(directory / f".{stage}.json", dict(signature=signature,
                outputs={name: sha256_file(directory / name) for name in names}))


def require_stage(directory, stage):
    marker = read(directory / f".{stage}.json")
    for name, checksum in marker["outputs"].items():
        if sha256_file(directory / name) != checksum:
            raise ValueError(f"{stage} evidence changed before downstream stage")


def verify_sources(protocol):
    for source in protocol["sources"].values():
        if sha256_file(source["path"]) != source["sha256"]:
            raise ValueError("Source video changed after planning")


def prepare(input_dir, annotations, root, match_id, options, settings, tracking_root=None):
    root = Path(root).resolve()
    inputs = Path(input_dir).resolve()
    if root == inputs or root in inputs.parents or inputs in root.parents:
        raise ValueError("Input and output directories must be separate and non-nested")
    protocol = load_events(inputs, annotations, match_id, options, ffprobe=settings.ffprobe)
    protocol["tracking_root"] = str(Path(tracking_root).resolve()) if tracking_root else None
    file = root / "protocol.json"
    if file.exists():
        if saved(root) != protocol:
            raise ValueError("Annotations/options/input changed; use a new run directory")
    else:
        if root.exists() and any(root.iterdir()):
            raise ValueError("New run directory must be empty")
        root.mkdir(parents=True, exist_ok=True)
        atomic_json(file, protocol)
    for event in protocol["events"]:
        source = protocol["sources"][event["video_id"]]
        directory = folder(root, event)
        directory.mkdir(parents=True, exist_ok=True)
        signature = dict(source_sha256=source["sha256"], event=event)
        if complete(directory, "prepare", signature, ["window.mkv", "window.json"]):
            continue
        print(f"[prepare] {event['event_id']} / video {event['video_id']}", flush=True)
        encoded = cut_video(source["path"], directory / "window.mkv", event["window_start_s"],
            event["window_end_s"], lossless=True, ffmpeg=settings.ffmpeg, ffprobe=settings.ffprobe)
        count = verify_frame_clock(directory / "window.mkv", source["fps"], settings.ffprobe)
        if count != event["window_end_frame_exclusive"]-event["window_start_frame"]:
            raise ValueError("Window encoding changed frame count; cannot safely map tracking to source")
        atomic_json(directory / "window.json", dict(event=event, encoded=encoded, frame_count=count,
                    time_mapping="source_frame=window_frame+window_start_frame"))
        mark(directory, "prepare", signature, ["window.mkv", "window.json"])
    return protocol


def model_protocol(settings, kind):
    if kind == "track":
        return dict(checkpoint=sha256_file(settings.motip_checkpoint),
                    config=sha256_file(settings.repository_root / "MOTIP/configs/r50_deformable_detr_motip_sportsmot.yaml"),
                    adapter=sha256_file(settings.repository_root / "pipeline/tracking/service.py"))
    # Include base configs referenced by the action config, without importing MMAction on the controller.
    config = settings.action_config.resolve()
    import ast
    def config_files(path, visited):
        if path in visited:
            return
        visited.add(path)
        tree = ast.parse(path.read_text())
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "_base_" for t in node.targets):
                bases = ast.literal_eval(node.value)
                for base in ([bases] if isinstance(bases, str) else bases):
                    config_files((path.parent / base).resolve(), visited)
    paths = set()
    config_files(config, paths)
    return dict(checkpoint=sha256_file(settings.action_checkpoint), labels=sha256_file(settings.action_label_map),
                configs={str(p): sha256_file(p) for p in sorted(paths)},
                adapter=sha256_file(settings.repository_root / "pipeline/action/service.py"))


def run_command(command, settings):
    subprocess.run([str(x) for x in command], cwd=settings.repository_root, check=True)


def reuse_tracking(protocol, event, directory, model):
    """Read-only reuse of the previous ten-clip test export; preserve its original raw IDs."""
    root = Path(protocol["tracking_root"])
    video = protocol["sources"][event["video_id"]]
    candidates = [v for v in read(root / "manifest.json")["videos"] if v["video_id"] == event["video_id"]]
    if len(candidates) != 1 or candidates[0]["source_sha256"] != video["sha256"]:
        raise ValueError("Reuse source does not match the annotated video SHA256")
    source = root / "track" / event["video_id"]
    marker = read(source / "tracking_complete.json")
    if (marker["protocol"]["checkpoint_sha256"] != model["checkpoint"]
            or marker["protocol"]["config_sha256"] != model["config"]
            or marker["protocol"].get("tracker_options") != "cli.tracking defaults v1"):
        raise ValueError("Reuse tracking model/settings do not match")
    for name, checksum in marker["outputs"].items():
        if sha256_file(source / name) != checksum:
            raise ValueError("Reuse tracks/meta were modified")
    meta = read(source / "video_meta.json")
    if meta["processed_frames"] != video["frame_count"] or abs(meta["fps"]-video["fps"]) > 1e-4:
        raise ValueError("Reuse tracking does not cover source frame clock")
    grouped = load_tracks(source / "tracks.jsonl", meta)
    count = event["window_end_frame_exclusive"]-event["window_start_frame"]
    with (directory / "tracks.jsonl").open("w") as stream:
        for rows in grouped.values():
            for r in rows:
                if event["window_start_frame"] <= r.frame_idx < event["window_end_frame_exclusive"]:
                    frame = r.frame_idx-event["window_start_frame"]
                    value = r.to_dict()
                    value.update(video_id=event["event_id"], frame_idx=frame, timestamp_s=frame/video["fps"])
                    write_jsonl_line(stream, value)
    atomic_json(directory / "video_meta.json", dict(meta, video_id=event["event_id"],
        frame_count=count, processed_frames=count, input_path=str(directory / "window.mkv"),
        tracking_origin="reused_full_clip", source_video_id=event["video_id"]))


def track(root, settings):
    protocol = saved(root)
    verify_sources(protocol)
    model = model_protocol(settings, "track")
    for event in protocol["events"]:
        directory = folder(root, event)
        require_stage(directory, "prepare")
        signature = dict(model=model, window=sha256_file(directory / "window.mkv"),
                         reuse=protocol["tracking_root"])
        names = ["tracks.jsonl", "video_meta.json"]
        if complete(directory, "track", signature, names):
            continue
        print(f"[track] {event['event_id']}", flush=True)
        if protocol["tracking_root"]:
            reuse_tracking(protocol, event, directory, model)
        else:
            run_command([settings.motip_python, "-u", "-m", "cli.tracking", "--input", directory / "window.mkv",
                "--video-id", event["event_id"], "--checkpoint", settings.motip_checkpoint,
                "--output-dir", directory, "--motip-root", settings.repository_root / "MOTIP",
                "--config", settings.repository_root / "MOTIP/configs/r50_deformable_detr_motip_sportsmot.yaml",
                "--no-visualization", "--overwrite"], settings)
        meta = read(directory / "video_meta.json")
        if (meta["video_id"] != event["event_id"]
                or meta["processed_frames"] != read(directory / "window.json")["frame_count"]):
            raise ValueError("MOTIP did not cover the complete event window")
        load_tracks(directory / "tracks.jsonl", meta)
        mark(directory, "track", signature, names)


def action(root, settings):
    protocol = saved(root)
    options = protocol["options"]
    model = model_protocol(settings, "action")
    for event in protocol["events"]:
        directory = folder(root, event)
        require_stage(directory, "track")
        signature = dict(model=model, window=sha256_file(directory / "window.mkv"),
            tracks=sha256_file(directory / "tracks.jsonl"), options=options)
        names = ["actions.jsonl", "actions.summary.json"]
        if complete(directory, "action", signature, names):
            continue
        print(f"[action] {event['event_id']}", flush=True)
        if not list(read_jsonl(directory / "tracks.jsonl")):
            (directory / "actions.jsonl").write_text("")
            atomic_json(directory / "actions.summary.json", dict(status="no_person_tracks"))
        else:
            run_command([settings.action_python, "-u", "-m", "cli.action", "--input", directory / "window.mkv",
                "--tracks", directory / "tracks.jsonl", "--config", settings.action_config,
                "--checkpoint", settings.action_checkpoint, "--label-map", settings.action_label_map,
                "--output", directory / "actions.jsonl", "--predict-stepsize", options["action_step"],
                "--sampling-fps", options["action_reference_fps"], "--proposal-max-gap", "0",
                "--top-k", "100", "--action-threshold", "0", "--overwrite"], settings)
        mark(directory, "action", signature, names)


def link(root, options):
    protocol = saved(root)
    for event in protocol["events"]:
        directory = folder(root, event)
        require_stage(directory, "action")
        signature = dict(tracks=sha256_file(directory / "tracks.jsonl"),
            actions=sha256_file(directory / "actions.jsonl"), options=asdict(options), version="action-evidence-v1",
            ranker=sha256_file(Path(__file__).resolve().parents[1] / "pipeline/player_link/evidence.py"))
        if complete(directory, "link", signature, ["link.json"]):
            continue
        meta = read(directory / "video_meta.json")
        grouped = load_tracks(directory / "tracks.jsonl", meta)
        result = rank_shooters(event, protocol["sources"][event["video_id"]],
            [r for rs in grouped.values() for r in rs], list(read_jsonl(directory / "actions.jsonl")), options)
        atomic_json(directory / "link.json", result)
        mark(directory, "link", signature, ["link.json"])
        print(f"[link] {event['event_id']}: {result['status']} / T={result['shooter_track_id']}", flush=True)


def status(root):
    root = Path(root)
    if not (root / "protocol.json").exists():
        return dict(status="not_prepared")
    protocol = saved(root)
    results = []
    for event in protocol["events"]:
        directory = folder(root, event)
        result = read(directory / "link.json") if (directory / "link.json").exists() else {}
        handoff = read(directory / "handoff/manifest.json") if (directory / "handoff/manifest.json").exists() else {}
        results.append(dict(event_id=event["event_id"], video_id=event["video_id"], goal_time_s=event["goal_time_s"],
            stages={stage: (directory / f".{stage}.json").exists() for stage in ("prepare", "track", "action", "link", "export")},
            attribution_status=result.get("status"), shooter_track_id=result.get("shooter_track_id"),
            handoff_status=handoff.get("attribution_status"), handoff_track_id=handoff.get("shooter_track_id")))
    done = all(all(row["stages"].values()) for row in results)
    return dict(status="completed_not_accuracy_evaluated" if done else "incomplete", match_id=protocol["match_id"],
                events=results, automatic_counts=dict(Counter(r["attribution_status"] or "not_linked" for r in results)),
                handoff_counts=dict(Counter(r["handoff_status"] or "not_exported" for r in results)), accuracy_evaluated=False)
