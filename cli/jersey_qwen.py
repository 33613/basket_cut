"""Recognize jersey numbers with a local Qwen3-VL model."""
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
    parser.add_argument("--assess-player",action="store_true")
    parser.add_argument("--team-map",type=Path)
    parser.add_argument("--observations", type=Path, help="Verified KPR sample manifest; reuse its exact crops")
    from pipeline.identity.qwen_backend import add_gpu_options, ensure_gpu_idle
    add_gpu_options(parser)
    from pipeline.identity.jersey_qwen import recognize_jerseys_qwen
    options = vars(parser.parse_args())
    shared = options.pop("allow_shared_gpu")
    free = options.pop("qwen_min_free_gib")
    if shared and options["device"] == "cuda":
        ensure_gpu_idle(allow_shared=True, min_free_gib=free)
    print(json.dumps(recognize_jerseys_qwen(**options), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
