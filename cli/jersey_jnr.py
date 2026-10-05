"""Read jersey numbers from cached KPR crops using uncertainty-jnr."""

import argparse
import json
from pathlib import Path
from contracts.execution import ExecutionSettings
from pipeline.identity.jersey_jnr import recognize_jerseys_jnr


def main():
    settings = ExecutionSettings()
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("archive-dir", "identity-map", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--root", type=Path, default=settings.jnr_root)
    parser.add_argument("--config", type=Path, default=settings.jnr_config)
    parser.add_argument("--checkpoint", type=Path, default=settings.jnr_checkpoint)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--trust-checkpoint", action="store_true", help="Allow pickle loading of explicitly trusted author weights")
    parser.add_argument("--min-score", type=float, default=.8)
    parser.add_argument("--max-uncertainty", type=float, default=.2)
    parser.add_argument("--min-margin", type=float, default=.2)
    parser.add_argument("--min-support", type=int, default=2)
    parser.add_argument("--min-gap-s", type=float, default=.25)
    parser.add_argument("--overwrite", action="store_true")
    print(json.dumps(recognize_jerseys_jnr(**vars(parser.parse_args())), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
