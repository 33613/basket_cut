"""CPU media operations with explicit, video-relative time and atomic outputs."""

from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import subprocess


def sha256_file(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def probe_video(path, *, ffprobe="ffprobe"):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    result = subprocess.run([ffprobe, "-v", "error", "-select_streams", "v:0",
        "-show_streams", "-show_format", "-of", "json", str(path)],
        capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)
    if not data.get("streams"):
        raise ValueError(f"No video stream: {path}")
    stream, container = data["streams"][0], data.get("format", {})
    rate = stream.get("avg_frame_rate", "0/0")
    if rate == "0/0":
        rate = stream.get("r_frame_rate", "0/0")
    fps = float(Fraction(rate))
    duration = float(stream.get("duration", container.get("duration", 0)))
    if not math.isfinite(duration) or duration <= 0 or not math.isfinite(fps) or fps <= 0:
        raise ValueError("Video must have a finite positive duration and frame rate")
    return {"path": str(path), "duration_s": duration, "fps": fps, "frame_rate": rate,
            "width": int(stream["width"]), "height": int(stream["height"]),
            "video_start_pts_s": float(stream.get("start_time", 0)),
            "format_start_pts_s": float(container.get("start_time", 0)),
            "time_scope": "seconds_from_first_video_presentation_timestamp"}


def cut_video(source, output, start_s, end_s, *, ffmpeg="ffmpeg", ffprobe="ffprobe",
              metadata=None, preview=False):
    """Accurate transcoding seek, CFR output for frame_idx/fps based backends.

    Source seconds are relative to the first video PTS, including for VFR input.
    Encoding is intentional: stream copying would move cuts to keyframes.
    """
    source, output = Path(source).resolve(), Path(output).resolve()
    meta = metadata or probe_video(source, ffprobe=ffprobe)
    start, end = float(start_s), float(end_s)
    if (not all(math.isfinite(v) for v in (start, end)) or start < 0
            or end <= start or end > meta["duration_s"] + 1e-6):
        raise ValueError("Cut must be a positive interval within the source video")
    if output == source:
        raise ValueError("Cannot overwrite source video")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name("." + output.stem + ".partial.mp4")
    # -ss is relative to the container start unless -seek_timestamp is enabled.
    seek = start + meta["video_start_pts_s"] - meta["format_start_pts_s"]
    command = [ffmpeg, "-nostdin", "-v", "error", "-y", "-ss", f"{seek:.9f}",
               "-i", str(source), "-t", f"{end - start:.9f}", "-map", "0:v:0"]
    if preview:
        command += ["-an", "-vf", "scale=w='min(1280,iw)':h=-2", "-r", str(min(meta["fps"], 30)),
                    "-crf", "23"]
    else:
        command += ["-map", "0:a:0?", "-c:a", "aac", "-r", meta["frame_rate"], "-crf", "18"]
    command += ["-vsync", "cfr", "-c:v", "libx264", "-preset", "fast", "-threads", "2",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temporary)]
    try:
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(f"ffmpeg cut failed: {result.stderr[-3000:]}")
        actual = probe_video(temporary, ffprobe=ffprobe)
        if abs(actual["duration_s"] - (end - start)) > max(.15, 2 / actual["fps"]):
            raise ValueError("Encoded clip duration differs from requested source interval")
        temporary.replace(output)
        return {**actual, "path": str(output), "source_start_s": start, "source_end_s": end,
                "timestamp_quantization_s": 1 / actual["fps"], "encoding": "h264_cfr"}
    finally:
        temporary.unlink(missing_ok=True)
