"""Print native video metadata and optional derived-media comparison; CPU only."""

import argparse
import json
from pathlib import Path
import subprocess
from contracts.execution import ExecutionSettings
from pipeline.media import probe_video


def inspect(path, ffprobe):
    meta = probe_video(path, ffprobe=ffprobe)
    raw = json.loads(
        subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_streams",
                "-show_format",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    )
    stream = raw["streams"][0]
    return {
        **meta,
        "codec": stream.get("codec_name"),
        "pixel_format": stream.get("pix_fmt"),
        "stream_bit_rate": stream.get("bit_rate"),
        "file_bytes": Path(path).stat().st_size,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--derived", type=Path)
    args = p.parse_args()
    ffprobe = ExecutionSettings().ffprobe
    source = inspect(args.input, ffprobe)
    result = {
        "source": source,
        "interpretation": "Metadata describes the local file, not the resolution of an unavailable upstream rendition. Stream copy preserves compressed video; enlarged person crops cannot restore missing detail.",
    }
    if args.derived:
        derived = inspect(args.derived, ffprobe)
        result.update(
            derived=derived,
            spatial_downscale=derived["width"] < source["width"]
            or derived["height"] < source["height"],
            fps_reduced=derived["fps"] < source["fps"] - 0.01,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
