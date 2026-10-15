"""Resumable coarse windows -> local verification -> source-mapped clip plan.

GPU access is injected; parsing, planning, caching and export run on a CPU.
Only verified, unambiguous boundaries are automatically used. Unresolved
proposals remain in the plan for review rather than silently disappearing.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import shutil

from pipeline.media import atomic_json, cut_video, probe_video, sha256_file
from pipeline.segmentation.prompts import RULES, coarse_prompt, boundary_prompt
from contracts.lineage import clip_id_for

KINDS = {"possession_change", "camera_cut", "uncertain"}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class SegmentOptions:
    window_s: float = 20
    overlap_s: float = 6
    review_radius_s: float = 4
    coarse_frames: int = 16
    review_frames: int = 32
    merge_tolerance_s: float = .75
    min_clip_s: float = .5
    min_free_gb: float = 2
    context: str = "5v5 basketball; team identities unknown"

    def validate(self):
        for name in ("window_s", "review_radius_s", "min_clip_s"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("overlap_s", "merge_tolerance_s", "min_free_gb"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.overlap_s >= self.window_s:
            raise ValueError("overlap_s must be smaller than window_s")
        for value in (self.coarse_frames, self.review_frames):
            if isinstance(value, bool) or not isinstance(value, int) or value % 2 or not 4 <= value <= 64:
                raise ValueError("Frame counts must be even integers from 4 to 64")
        if not self.context.strip() or len(self.context) > 2000:
            raise ValueError("context must contain 1–2000 characters")


def plan_windows(duration_s, options):
    options.validate()
    if not math.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("Invalid duration")
    windows, start = [], 0.0
    while start < duration_s:
        end = min(start + options.window_s, duration_s)
        windows.append({"start_s": round(start, 9), "end_s": round(end, 9)})
        if end >= duration_s:
            break
        start += options.window_s - options.overlap_s
    return windows


def parse_json(raw):
    text = raw.strip()
    if text.startswith("```json") and text.endswith("```"):
        text = text[7:-3].strip()
    result = json.loads(text, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    if not isinstance(result, dict):
        raise ValueError("Model response must be a JSON object")
    return result


def number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Expected a finite numeric timestamp")
    return float(value)


def evidence(value):
    if (value.get("kind") not in KINDS or not isinstance(value.get("evidence"), str)
            or not value["evidence"].strip() or len(value["evidence"]) > 2000
            or not isinstance(value.get("needs_review"), bool)):
        raise ValueError("Invalid boundary kind, evidence or review flag")
    return {key: value[key] for key in ("kind", "evidence", "needs_review")}


def parse_coarse(raw, duration_s):
    values = parse_json(raw).get("boundaries")
    if not isinstance(values, list) or len(values) > 12:
        raise ValueError("Expected at most 12 candidate boundaries")
    result = []
    for value in values:
        if not isinstance(value, dict):
            raise ValueError("Invalid boundary")
        bounds = value.get("time_range_s")
        if not isinstance(bounds, list) or len(bounds) != 2:
            raise ValueError("Boundary needs a time range")
        start, end = map(number, bounds)
        if not 0 <= start <= end <= duration_s:
            raise ValueError("Candidate outside local window")
        result.append({**evidence(value), "time_range_s": [start, end]})
    return result


def parse_boundary(raw, duration_s):
    value = parse_json(raw)
    result = evidence(value)
    if not isinstance(value.get("confirmed"), bool):
        raise ValueError("confirmed must be boolean")
    point = value.get("boundary_s")
    if point is not None:
        point = number(point)
        if not 0 <= point <= duration_s or not value["confirmed"]:
            raise ValueError("Invalid verified boundary")
    if value["confirmed"] and point is None and not value["needs_review"]:
        raise ValueError("A confirmed boundary needs a time or a review flag")
    return {**result, "confirmed": value["confirmed"], "boundary_s": point}


def merge_candidates(candidates, tolerance_s):
    """Fuse overlapping-window observations; retain all supporting windows."""
    groups = []
    for row in sorted(candidates, key=lambda r: r["time_range_s"]):
        if (groups and row["kind"] == groups[-1]["kind"]
                and row["time_range_s"][0] <= groups[-1]["time_range_s"][1] + tolerance_s):
            group = groups[-1]
            group["time_range_s"][1] = max(group["time_range_s"][1], row["time_range_s"][1])
            group["support"].append(row)
        else:
            groups.append({"time_range_s": list(row["time_range_s"]), "kind": row["kind"], "support": [row]})
    return groups


def make_segments(source_id, duration_s, reviews, min_clip_s):
    accepted, unresolved = [], []
    for review in sorted(reviews, key=lambda r: r.get("source_boundary_s") or 0):
        point = review.get("source_boundary_s")
        if not review["confirmed"]:
            if review["needs_review"]:
                unresolved.append(review)
            continue
        if point is None or review["needs_review"] or review["kind"] == "uncertain":
            unresolved.append(review)
            continue
        if point - (accepted[-1]["source_boundary_s"] if accepted else 0) < min_clip_s or duration_s - point < min_clip_s:
            unresolved.append({**review, "suppressed_reason": "boundary_too_close"})
            continue
        accepted.append(review)
    bounds = [0.0, *(r["source_boundary_s"] for r in accepted), duration_s]
    segments = []
    for index, (start, end) in enumerate(zip(bounds, bounds[1:])):
        pending = [r["candidate_id"] for r in unresolved
                   if r["source_range_s"][0] < end and r["source_range_s"][1] >= start]
        segments.append({"clip_id": clip_id_for(source_id, start, end),
            "source_video_id": source_id, "start_s": start, "end_s": end,
            "start_boundary_kind": "video_start" if index == 0 else accepted[index - 1]["kind"],
            "needs_review": bool(pending), "unresolved_candidate_ids": pending})
    return segments, unresolved


def reserve(output, minimum):
    if shutil.disk_usage(output).free < minimum * 1024**3:
        raise RuntimeError("Disk reserve reached; free storage before resuming")


def analyze_video(source, output, backend, *, options=None, ffmpeg="ffmpeg", ffprobe="ffprobe"):
    options = options or SegmentOptions()
    options.validate()
    source, output = Path(source).resolve(), Path(output).resolve()
    if source.is_relative_to(output):
        raise ValueError("Segmentation output cannot contain its source video")
    output.mkdir(parents=True, exist_ok=True)
    meta = probe_video(source, ffprobe=ffprobe)
    checksum = sha256_file(source)
    source_id = "SRC-" + checksum[:20]
    protocol = {"schema_version": 1, "source": {**meta, "sha256": checksum, "source_video_id": source_id},
                "options": asdict(options), "model": backend.provenance,
                "rules_sha256": digest(RULES), "service_sha256": sha256_file(Path(__file__)),
                "prompts_sha256": sha256_file(Path(__file__).with_name("prompts.py"))}
    signature = digest(protocol)
    protocol_path = output / "protocol.json"
    if protocol_path.is_file() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("Segmentation inputs/model/settings changed; use a new output directory")
    atomic_json(protocol_path, protocol)
    state_path = output / "status.json"
    state = {"status": "running", "protocol_sha256": signature, "coarse_completed": 0, "review_completed": 0}
    atomic_json(state_path, state)
    window_root = output / "windows"
    window_root.mkdir(exist_ok=True)

    def query(key, start, end, prompt, frames, parser):
        cache = output / "responses" / f"{key}.json"
        request = {"protocol_sha256": signature, "start_s": start, "end_s": end,
                   "prompt": prompt, "num_frames": frames, "max_new_tokens": 2048}
        if cache.is_file():
            saved = json.loads(cache.read_text())
            if saved.get("request") != request:
                raise ValueError("Cached Qwen request differs from this run")
            if "error" not in saved:
                return parser(saved["raw_response"], end - start)
        reserve(output, options.min_free_gb)
        window = window_root / (key + ".mp4")
        cut_video(source, window, start, end, ffmpeg=ffmpeg, ffprobe=ffprobe, metadata=meta, preview=True)
        print(f"[segmentation] {key} {start:.2f}–{end:.2f}s, {frames} frames", flush=True)
        try:
            raw = backend.read_video(window, prompt, num_frames=frames, max_new_tokens=2048)
            saved = {"request": request, "raw_response": raw}
            atomic_json(cache, saved)
            try:
                result = parser(raw, end - start)
            except (ValueError, TypeError, KeyError) as exc:
                atomic_json(cache, {**saved, "error": str(exc)})
                raise ValueError(f"Invalid Qwen response; inspect {cache}: {exc}") from exc
            return result
        finally:
            window.unlink(missing_ok=True)

    try:
        proposals = []
        windows = plan_windows(meta["duration_s"], options)
        state["coarse_total"] = len(windows)
        for index, window in enumerate(windows):
            start, end = window["start_s"], window["end_s"]
            key = f"coarse-{index:06d}"
            rows = query(key, start, end, coarse_prompt(end - start, options.context), options.coarse_frames, parse_coarse)
            proposals.extend({**row, "window_id": key,
                "time_range_s": [start + t for t in row["time_range_s"]]} for row in rows)
            state["coarse_completed"] += 1
            atomic_json(state_path, state)
        groups = merge_candidates(proposals, options.merge_tolerance_s)
        reviews = []
        state["review_total"] = len(groups)
        for index, candidate in enumerate(groups):
            low, high = candidate["time_range_s"]
            # Keep each GPU request bounded even when coarse uncertainty is broad.
            center = (low + high) / 2
            start, end = max(0, center - options.review_radius_s), min(meta["duration_s"], center + options.review_radius_s)
            local = {"time_range_s": [max(0, low - start), min(end - start, high - start)],
                     "kind": candidate["kind"], "evidence": [s["evidence"] for s in candidate["support"]]}
            key = f"review-{index:06d}"
            row = query(key, start, end, boundary_prompt(end - start, local, options.context), options.review_frames, parse_boundary)
            point = start + row["boundary_s"] if row["boundary_s"] is not None else None
            if high - low > 2 * options.review_radius_s:
                row["needs_review"] = True
            # A local review must verify the proposed region, not another nearby event.
            if point is not None and not low - options.merge_tolerance_s <= point <= high + options.merge_tolerance_s:
                row["needs_review"] = True
            reviews.append({**row, "candidate_id": key, "source_boundary_s": point,
                            "source_range_s": [low, high], "review_window": [start, end], "support": candidate["support"]})
            state["review_completed"] += 1
            atomic_json(state_path, state)
        segments, unresolved = make_segments(source_id, meta["duration_s"], reviews, options.min_clip_s)
        result = {"schema_version": 1, "status": "completed_not_accuracy_evaluated", "protocol_sha256": signature,
                  "source": protocol["source"], "windows": windows, "boundaries": reviews, "segments": segments,
                  "unresolved": unresolved, "needs_review": bool(unresolved),
                  "time_scope": meta["time_scope"], "boundary_accuracy_evaluated": False}
        atomic_json(output / "segmentation.json", result)
        state["status"] = "completed_not_accuracy_evaluated"
        atomic_json(state_path, state)
        return result
    except Exception as exc:
        atomic_json(state_path, {**state, "status": "failed", "error": str(exc)})
        raise


def export_segments(output, *, review_file=None, ffmpeg="ffmpeg", ffprobe="ffprobe", min_free_gb=2):
    output = Path(output).resolve()
    plan = json.loads((output / "segmentation.json").read_text())
    source = plan["source"]
    if sha256_file(source["path"]) != source["sha256"]:
        raise ValueError("Original video changed after analysis")
    segments = plan["segments"]
    if review_file:
        segments = json.loads(Path(review_file).read_text())["segments"]
    normalized, end = [], 0.0
    if not isinstance(segments, list) or not segments:
        raise ValueError("Clip list must not be empty")
    for row in segments:
        start, finish = number(row["start_s"]), number(row["end_s"])
        if abs(start - end) > 1e-6 or not 0 <= start < finish <= source["duration_s"] + 1e-6:
            raise ValueError("Clips must be contiguous, ordered, nonoverlapping source intervals")
        start, finish = end, min(finish, source["duration_s"])
        clip_id = clip_id_for(source["source_video_id"], start, finish)
        normalized.append({"clip_id": clip_id, "source_video_id": source["source_video_id"],
            "start_s": start, "end_s": finish, "needs_review": False if review_file else bool(row["needs_review"]),
            "start_boundary_kind": row.get("start_boundary_kind", "manual"),
            "unresolved_candidate_ids": [] if review_file else row.get("unresolved_candidate_ids", [])})
        end = finish
    if abs(end - source["duration_s"]) > 1e-6:
        raise ValueError("Clips must cover the complete source duration")
    manifest_path = output / "clips_manifest.json"
    expected = {"source": source, "segments": normalized, "encoding": "h264_cfr_crf18_v1",
                "protocol_sha256": plan["protocol_sha256"], "manually_reviewed": review_file is not None}
    fingerprint = digest(expected)
    previous = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    if previous and previous.get("export_fingerprint") != fingerprint:
        raise ValueError("Export plan changed; use a new segmentation directory to preserve registered clip evidence")
    videos = []
    for row in normalized:
        relative = "clips/" + row["clip_id"] + ".mp4"
        path = output / relative
        sidecar = path.with_suffix(".json")
        old = json.loads(sidecar.read_text()) if sidecar.is_file() else None
        if old and old.get("export_fingerprint") == fingerprint and path.is_file() and sha256_file(path) == old.get("sha256"):
            videos.append(old)
            continue
        if path.exists():
            raise ValueError(f"Existing clip is unverified or changed: {path}")
        reserve(output, min_free_gb)
        encoded = cut_video(source["path"], path, row["start_s"], row["end_s"],
                            ffmpeg=ffmpeg, ffprobe=ffprobe, metadata=source)
        value = {**row, "path": relative, "source_path": source["path"], "source_sha256": source["sha256"],
                 "sha256": sha256_file(path), "encoded": encoded, "export_fingerprint": fingerprint,
                 "time_scope": source["time_scope"]}
        atomic_json(sidecar, value)
        videos.append(value)
    manifest = {"schema_version": 1, "status": "exported_not_accuracy_evaluated", "source": source,
                "videos": videos, "export_fingerprint": fingerprint,
                "segmentation_path": str(output / "segmentation.json"), "manually_reviewed": review_file is not None}
    atomic_json(manifest_path, manifest)
    return manifest
