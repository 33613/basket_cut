"""Recognize jersey numbers with a local Qwen2.5-VL model."""
import argparse
import json
from pathlib import Path
from contracts.execution import ExecutionSettings


def main():
    settings = ExecutionSettings()
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("video", "tracks", "identity-map", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=settings.qwen_model_dir)
    parser.add_argument("--max-samples", type=int, default=6)
    parser.add_argument("--min-support", type=int, default=2)
    parser.add_argument("--min-gap-s", type=float, default=.25)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    from pipeline.identity.jersey_qwen import recognize_jerseys_qwen
    print(json.dumps(recognize_jerseys_qwen(**vars(parser.parse_args())), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
